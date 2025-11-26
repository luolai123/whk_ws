#!/usr/bin/env python3
"""PlannerNet inference node that mirrors the training-time de-normalization.

Pipeline:
1) Subscribe to monocular RGB images and current drone state (p0, v0, a0).
2) Convert both to PyTorch tensors on the same device.
3) Run SegNet (``DistanceClassifier``) to produce the safe probability map.
4) Run PlannerNet (``SafeNavigationPolicy``) to get raw end-state outputs.
5) Apply training-consistent de-normalization to obtain ``p_T, v_T, a_T``.
6) Compute trajectory duration ``T = 2 * ||delta_p|| / (alpha * v_max)`` with
   clamping, then generate a quintic trajectory between the boundary states.
7) Select a short-horizon reference point and map the tracking error to a
   velocity command via a simple PD controller.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from autonomy_demo.safe_navigation import quintic_coefficients, sample_quintic


@dataclass
class TrajectoryConfig:
    """Runtime parameters aligned with training normalization."""

    r_max: float
    v_max: float
    a_max: float
    alpha: float
    min_duration: float
    max_duration: float
    traj_sample_hz: float

    @property
    def alpha_v_max(self) -> float:
        return self.alpha * self.v_max

    @property
    def alpha_a_max(self) -> float:
        return (self.alpha**2) * self.a_max


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class UpBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DistanceClassifier(torch.nn.Module):
    """UNet-style segmentation network (SegNet)."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = torch.nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)
        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up1 = UpBlock(64, 32)
        self.classifier = torch.nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bottleneck = self.bottleneck(self.pool(x3))
        x = self.up3(bottleneck, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.classifier(x)


class SafeNavigationPolicy(torch.nn.Module):
    """PlannerNet that predicts Δp_raw, y_v, and y_a."""

    def __init__(self, height: int, width: int, state_dim: int = 9) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.state_dim = state_dim
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(inplace=True),
        )
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        fusion_dim = 128
        self.fc1 = torch.nn.Linear(64 + state_dim, fusion_dim)
        self.end_state_head = torch.nn.Linear(fusion_dim, 9)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.size(1) == 1:
            image = image.repeat(1, 3, 1, 1)
        features = self.backbone(image)
        pooled = self.global_pool(features).view(image.size(0), -1)
        combined = torch.cat([pooled, state], dim=1)
        hidden = torch.relu(self.fc1(combined))
        return self.end_state_head(hidden)


class PlannerNetInferenceNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seg_model = DistanceClassifier().to(self.device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.planner: Optional[SafeNavigationPolicy] = None

        self.seg_input_hw: Optional[Tuple[int, int]] = None
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None

        seg_path = pathlib.Path(
            rospy.get_param(
                "~seg_model_path",
                str(pathlib.Path.home() / "autonomy_demo" / "segmentation_model.pt"),
            )
        )
        if seg_path.exists():
            checkpoint = torch.load(seg_path, map_location=self.device)
            state_dict = checkpoint
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
                norm_meta = checkpoint.get("normalization")
                if isinstance(norm_meta, dict):
                    mean = norm_meta.get("mean")
                    std = norm_meta.get("std")
                    if mean is not None and std is not None:
                        self.norm_mean = np.asarray(mean, dtype=np.float32)
                        self.norm_std = np.clip(np.asarray(std, dtype=np.float32), 1e-4, None)
                input_size = checkpoint.get("input_size")
                if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
                    try:
                        h_val = int(input_size[0])
                        w_val = int(input_size[1])
                        if h_val > 0 and w_val > 0:
                            self.seg_input_hw = (h_val, w_val)
                    except (TypeError, ValueError):
                        self.seg_input_hw = None
            self.seg_model.load_state_dict(state_dict)
            rospy.loginfo("Loaded segmentation model from %s", seg_path)
        else:
            rospy.logwarn("Segmentation model %s not found; using random init", seg_path)
        self.seg_model.eval()

        self.config = TrajectoryConfig(
            r_max=float(rospy.get_param("~r_max", 5.0)),
            v_max=float(rospy.get_param("~v_max", 6.0)),
            a_max=float(rospy.get_param("~a_max", 3.0)),
            alpha=float(rospy.get_param("~alpha", 0.6)),
            min_duration=float(rospy.get_param("~min_duration", 0.2)),
            max_duration=float(rospy.get_param("~max_duration", 6.0)),
            traj_sample_hz=float(rospy.get_param("~traj_sample_hz", 15.0)),
        )

        self.kp_pos = float(rospy.get_param("~kp_position", 0.8))
        self.kd_vel = float(rospy.get_param("~kd_velocity", 0.2))
        self.max_cmd = float(rospy.get_param("~max_cmd_speed", 4.0))
        self.ref_index = int(rospy.get_param("~ref_index", 2))

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.path_pub = rospy.Publisher("planner_net/path", Path, queue_size=1)
        self.safe_prob_pub = rospy.Publisher("planner_net/safe_probability", Image, queue_size=1)

        rospy.Subscriber("drone/rgb/camera_info", CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber("drone/odometry", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("drone/rgb/image_raw", Image, self._image_cb, queue_size=1)

        self.camera_info: Optional[CameraInfo] = None
        self.odom: Optional[Odometry] = None
        self.image_shape: Optional[Tuple[int, int]] = None
        self._last_header: Optional[Header] = None

    def _camera_info_cb(self, info: CameraInfo) -> None:
        self.camera_info = info
        self.image_shape = (info.height, info.width)
        if self.planner is None and info.height > 0 and info.width > 0:
            self.planner = SafeNavigationPolicy(info.height, info.width).to(self.device)
            planner_path = pathlib.Path(
                rospy.get_param(
                    "~planner_path",
                    str(pathlib.Path.home() / "autonomy_demo" / "navigation_policy.pt"),
                )
            )
            if planner_path.exists():
                self.planner.load_state_dict(
                    torch.load(planner_path, map_location=self.device)
                )
                rospy.loginfo("Loaded PlannerNet from %s", planner_path)
            else:
                rospy.logwarn("PlannerNet weights %s not found; using random init", planner_path)
            self.planner.eval()

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom = msg

    def _image_cb(self, msg: Image) -> None:
        if self.planner is None or self.odom is None:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.image_shape = cv_image.shape[0:2]
        self._last_header = msg.header
        safe_prob, safe_mask = self._run_segnet(cv_image)
        delta_p, v_t, a_t = self._run_planner(cv_image)
        if delta_p is None or v_t is None or a_t is None:
            return
        p0, v0, a0 = self._current_state()
        p_t = p0 + delta_p

        radius = float(np.linalg.norm(delta_p))
        duration = 2.0 * radius / max(self.config.alpha_v_max, 1e-3)
        duration = float(np.clip(duration, self.config.min_duration, self.config.max_duration))

        coeffs = quintic_coefficients(p0, v0, a0, p_t, v_t, a_t, duration)
        steps = max(2, int(math.ceil(duration * self.config.traj_sample_hz)))
        points, velocities = sample_quintic(coeffs, duration, steps)

        ref_idx = min(self.ref_index, points.shape[0] - 1)
        ref_pos = points[ref_idx]
        ref_vel = velocities[ref_idx]

        cmd = self._pd_control(p0, v0, ref_pos, ref_vel)
        self._publish_cmd(cmd)
        self._publish_path(points, msg.header)
        self._publish_safe_prob(safe_prob, msg.header)

    def _run_segnet(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        normalized = image.astype(np.float32) / 255.0
        needs_resize = False
        if self.seg_input_hw is not None and (
            normalized.shape[0] != self.seg_input_hw[0]
            or normalized.shape[1] != self.seg_input_hw[1]
        ):
            normalized = cv2.resize(
                normalized,
                (self.seg_input_hw[1], self.seg_input_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            needs_resize = True
        if self.norm_mean is not None and self.norm_std is not None:
            normalized = (normalized - self.norm_mean) / self.norm_std
        tensor = (
            torch.from_numpy(normalized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        with torch.no_grad():
            logits = self.seg_model(tensor)
            probs = self.softmax(logits)
        safe_prob = probs[:, 0:1].squeeze(0).squeeze(0).cpu().numpy()
        if needs_resize:
            safe_prob = cv2.resize(
                safe_prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        safe_mask = safe_prob >= 0.5
        return safe_prob, safe_mask

    def _state_features(self) -> torch.Tensor:
        p0, v0, a0 = self._current_state()
        pos_norm = p0 / max(self.config.r_max, 1e-3)
        vel_norm = v0 / max(self.config.v_max, 1e-3)
        acc_norm = a0 / max(self.config.a_max, 1e-3)
        return torch.from_numpy(np.concatenate([pos_norm, vel_norm, acc_norm])).unsqueeze(0)

    def _run_planner(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.planner is None:
            return None, None, None
        normalized = image.astype(np.float32) / 255.0
        if self.norm_mean is not None and self.norm_std is not None:
            normalized = (normalized - self.norm_mean) / self.norm_std
        image_tensor = (
            torch.from_numpy(normalized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        state_tensor = self._state_features().to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            end_state_raw = self.planner(image_tensor, state_tensor)[0]
        delta_p_raw = end_state_raw[0:3]
        y_v = end_state_raw[3:6]
        y_a = end_state_raw[6:9]

        delta_p = torch.tanh(delta_p_raw) * self.config.r_max
        v_t = torch.tanh(y_v) * self.config.alpha_v_max
        a_t = torch.tanh(y_a) * self.config.alpha_a_max

        return (
            delta_p.cpu().numpy(),
            v_t.cpu().numpy(),
            a_t.cpu().numpy(),
        )

    def _current_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.odom is not None
        p0 = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        v0 = np.array(
            [
                self.odom.twist.twist.linear.x,
                self.odom.twist.twist.linear.y,
                self.odom.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        # Acceleration is not provided in odometry; assume zero for inference.
        a0 = np.zeros(3, dtype=np.float32)
        return p0, v0, a0

    def _pd_control(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
    ) -> np.ndarray:
        pos_error = ref_pos - p0
        vel_error = ref_vel - v0
        cmd = self.kp_pos * pos_error + self.kd_vel * vel_error
        norm = float(np.linalg.norm(cmd))
        if norm > self.max_cmd > 0.0:
            cmd = cmd / norm * self.max_cmd
        return cmd

    def _publish_cmd(self, cmd_vec: np.ndarray) -> None:
        cmd = Twist()
        cmd.linear.x = float(cmd_vec[0])
        cmd.linear.y = float(cmd_vec[1])
        cmd.linear.z = float(cmd_vec[2])
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def _publish_path(self, points: np.ndarray, header: Header) -> None:
        if not self.path_pub.get_num_connections():
            return
        path = Path()
        path.header = header
        path.poses = []
        for pt in points:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _publish_safe_prob(self, safe_prob: np.ndarray, header: Header) -> None:
        if not self.safe_prob_pub.get_num_connections():
            return
        prob_uint8 = np.clip(safe_prob * 255.0, 0, 255).astype(np.uint8)
        img_msg = self.bridge.cv2_to_imgmsg(prob_uint8, encoding="mono8")
        img_msg.header = header
        self.safe_prob_pub.publish(img_msg)

    def spin(self) -> None:
        rospy.loginfo("PlannerNet inference node started")
        rospy.spin()


def main() -> None:
    rospy.init_node("planner_net_inference")
    PlannerNetInferenceNode().spin()


if __name__ == "__main__":
    main()
