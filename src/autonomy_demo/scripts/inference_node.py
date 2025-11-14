#!/usr/bin/env python3
"""Inference node that loads trained models and emits safe navigation primitives."""

import math
import pathlib
import time
from typing import List, Optional, Tuple

import numpy as np
import rospy
import torch
import torch.nn.functional as F
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    PointStamped,
    Pose,
    PoseArray,
    PoseStamped,
    Vector3Stamped,
)
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, Header
from tf_conversions import transformations

from autonomy_demo.safe_navigation import (
    clamp_normalized,
    compute_direction_from_pixel,
    find_largest_safe_region,
    quintic_coefficients,
    sample_quintic,
)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bottleneck = self.bottleneck(self.pool(x3))
        x = self.up3(bottleneck, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.classifier(x)


class SafeNavigationPolicy(torch.nn.Module):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(inplace=True),
        )
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(64 + 1, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, mask: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        features = self.backbone(mask)
        pooled = self.global_pool(features).view(mask.size(0), -1)
        combined = torch.cat([pooled, speed.unsqueeze(1)], dim=1)
        hidden = torch.relu(self.fc1(combined))
        return torch.tanh(self.fc2(hidden))


class InferenceNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistanceClassifier()
        self.model_input_hw: Optional[Tuple[int, int]] = None
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        model_path = rospy.get_param(
            "~model_path", str(pathlib.Path.home() / "autonomy_demo" / "model.pt")
        )
        model_path = pathlib.Path(model_path)
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
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
                        height = int(input_size[0])
                        width = int(input_size[1])
                        if height > 0 and width > 0:
                            self.model_input_hw = (height, width)
                    except (TypeError, ValueError):
                        self.model_input_hw = None
            self.model.load_state_dict(state_dict)
            rospy.loginfo("Loaded classifier weights from %s", model_path)
        else:
            rospy.logwarn("Classifier weights %s not found - using random initialization", model_path)
        self.model.to(self.device)
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=1)

        self.policy: Optional[SafeNavigationPolicy] = None
        self.policy_state: Optional[dict] = None
        policy_path = rospy.get_param(
            "~policy_path", str(pathlib.Path.home() / "autonomy_demo" / "navigation_policy.pt")
        )
        policy_path = pathlib.Path(policy_path)
        if policy_path.exists():
            try:
                self.policy_state = torch.load(policy_path, map_location=self.device)
                rospy.loginfo("Loaded navigation policy from %s", policy_path)
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logwarn("Failed to load navigation policy %s: %s", policy_path, exc)
                self.policy_state = None
        else:
            rospy.logwarn("Navigation policy weights %s not found", policy_path)

        self.min_safe_fraction = float(rospy.get_param("~min_safe_fraction", 0.05))
        self.default_speed = float(rospy.get_param("~default_speed", 3.0))
        self.safe_threshold = float(rospy.get_param("~safe_probability_threshold", 0.55))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.3))
        self.min_clearance_fraction = float(rospy.get_param("~min_clearance_fraction", 0.08))
        self.clearance_softening = float(rospy.get_param("~clearance_softening", 0.18))
        self.max_heading_rate = math.radians(rospy.get_param("~max_heading_rate_deg", 110.0))
        self.max_jerk = float(rospy.get_param("~max_jerk_mps3", 35.0))
        pitch_deg = float(rospy.get_param("~camera_pitch_deg", 10.0))
        pitch_rad = -math.radians(pitch_deg)
        self._camera_mount_quat = transformations.quaternion_from_euler(0.0, pitch_rad, 0.0)
        self._camera_mount_matrix = (
            transformations.quaternion_matrix(self._camera_mount_quat)[0:3, 0:3]
        ).astype(np.float32)
        self._camera_to_body = self._camera_mount_matrix.T

        self.camera_info: Optional[CameraInfo] = None
        self.odom: Optional[Odometry] = None
        self.tan_half_h: Optional[float] = None
        self.tan_half_v: Optional[float] = None
        self.image_shape: Optional[Tuple[int, int]] = None
        self.goal_world: Optional[np.ndarray] = None
        self.goal_frame: Optional[str] = None
        self.goal_direction_blend = float(rospy.get_param("~goal_direction_blend", 0.4))
        self.goal_bias_distance = float(rospy.get_param("~goal_bias_distance", 10.0))
        self.plan_hold_time = rospy.Duration.from_sec(
            float(rospy.get_param("~plan_hold_time", self.primitive_dt * 0.6))
        )
        self.plan_similarity_epsilon = float(rospy.get_param("~plan_similarity_epsilon", 0.35))
        self._last_plan_signature: Optional[Tuple[float, float, float, float]] = None
        self._last_plan_stamp = rospy.Time(0.0)
        self._smoothed_command: Optional[np.ndarray] = None

        self.label_pub = rospy.Publisher("drone/rgb/distance_class", Image, queue_size=1)
        self.safe_center_pub = rospy.Publisher("drone/safe_center", PointStamped, queue_size=1)
        self.primitive_pub = rospy.Publisher("drone/movement_primitive", Vector3Stamped, queue_size=1)
        self.command_pub = rospy.Publisher("drone/movement_command", Vector3Stamped, queue_size=1)
        self.offset_pub = rospy.Publisher("drone/movement_offsets", Float32MultiArray, queue_size=1)
        self.fallback_pub = rospy.Publisher("drone/fallback_primitives", PoseArray, queue_size=1)
        self.trajectory_pub = rospy.Publisher("drone/safe_trajectory", Path, queue_size=1)

        rospy.Subscriber("drone/rgb/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber("drone/odometry", Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber("drone/rgb/image_raw", Image, self.image_callback, queue_size=1)
        rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=1)

        primitive_steps = int(rospy.get_param("~primitive_steps", 4))
        self.primitive_steps = max(3, min(5, primitive_steps))
        self.primitive_dt = float(rospy.get_param("~primitive_dt", 0.25))
        self.primitive_duration = self.primitive_steps * self.primitive_dt
        self.distance_scale_min = float(rospy.get_param("~distance_scale_min", 0.7))
        self.distance_scale_max = float(rospy.get_param("~distance_scale_max", 1.4))
        self.duration_scale_min = float(rospy.get_param("~duration_scale_min", 0.7))
        self.duration_scale_max = float(rospy.get_param("~duration_scale_max", 1.3))

    def camera_info_callback(self, info: CameraInfo) -> None:
        self.camera_info = info
        self.image_shape = (info.height, info.width)
        fx = info.K[0]
        if fx <= 0.0:
            fx = info.width / (2.0 * math.tan(math.radians(120.0) / 2.0))
        fov_h = 2.0 * math.atan(info.width / (2.0 * fx))
        self.tan_half_h = math.tan(fov_h / 2.0)
        aspect = info.height / float(info.width)
        self.tan_half_v = self.tan_half_h * aspect
        self._ensure_policy()

    def odom_callback(self, msg: Odometry) -> None:
        self.odom = msg

    def goal_callback(self, msg: PoseStamped) -> None:
        self.goal_world = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32
        )
        self.goal_frame = msg.header.frame_id or "map"
        if self.odom is not None and abs(self.goal_world[2]) < 1e-3:
            self.goal_world[2] = float(self.odom.pose.pose.position.z)

    def _goal_vector(self, origin: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if self.goal_world is None:
            return None, None
        goal_vec = self.goal_world.astype(np.float32) - origin.astype(np.float32)
        goal_distance = float(np.linalg.norm(goal_vec))
        if goal_distance < max(self.goal_tolerance * 0.5, 1e-3):
            return None, goal_distance
        return goal_vec / goal_distance, goal_distance

    def _ensure_policy(self) -> None:
        if self.policy is not None:
            return
        if self.image_shape is None:
            return
        height, width = self.image_shape
        self.policy = SafeNavigationPolicy(height, width).to(self.device)
        if self.policy_state:
            self.policy.load_state_dict(self.policy_state)
        else:
            rospy.logwarn_once("Navigation policy is running with neutral outputs (no weights loaded)")
        self.policy.eval()

    def image_callback(self, msg: Image) -> None:
        if self.camera_info is None or self.odom is None:
            return
        if self.tan_half_h is None or self.tan_half_v is None:
            return
        start_time = time.perf_counter()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        height, width, _ = cv_image.shape
        self.image_shape = (height, width)
        self._ensure_policy()

        normalized = cv_image.astype(np.float32) / 255.0
        needs_resize = False
        if self.model_input_hw is not None and (
            height != self.model_input_hw[0] or width != self.model_input_hw[1]
        ):
            normalized = cv2.resize(
                normalized,
                (self.model_input_hw[1], self.model_input_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            needs_resize = True
        if self.norm_mean is not None and self.norm_std is not None:
            normalized = (normalized - self.norm_mean) / self.norm_std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = self.softmax(logits)

        safe_prob = probs[:, 0, :, :].squeeze(0).cpu().numpy()
        obstacle_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy()
        if needs_resize:
            safe_prob = cv2.resize(safe_prob, (width, height), interpolation=cv2.INTER_LINEAR)
            obstacle_prob = cv2.resize(
                obstacle_prob, (width, height), interpolation=cv2.INTER_LINEAR
            )
        safe_prob = cv2.GaussianBlur(safe_prob, (5, 5), 0)
        obstacle_prob = cv2.GaussianBlur(obstacle_prob, (5, 5), 0)
        safe_mask = safe_prob >= self.safe_threshold
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(safe_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        safe_mask = cleaned.astype(bool)
        distance_field = cv2.distanceTransform(
            safe_mask.astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float32)

        color_map = np.zeros((height, width, 3), dtype=np.uint8)
        color_map[:, :, 0] = 255  # default to red for obstacle/unknown
        color_map[safe_mask, 0] = 0
        color_map[safe_mask, 1] = 255
        label_msg = self.bridge.cv2_to_imgmsg(color_map, encoding="rgb8")
        label_msg.header = msg.header
        label_msg.header.frame_id = rospy.get_param("~output_frame", msg.header.frame_id)
        self.label_pub.publish(label_msg)

        region = find_largest_safe_region(safe_mask, self.min_safe_fraction)
        if region is None:
            self._publish_fallback(msg.header.stamp)
            self._log_timing(start_time)
            return

        center_row, center_col = region.centroid
        total_pixels = height * width
        safe_fraction = region.area / float(total_pixels)

        safe_msg = PointStamped()
        safe_msg.header = msg.header
        safe_msg.point.x = (center_col + 0.5) / width
        safe_msg.point.y = (center_row + 0.5) / height
        safe_msg.point.z = safe_fraction
        self.safe_center_pub.publish(safe_msg)

        fov_deg = math.degrees(2.0 * math.atan(self.tan_half_h))
        base_direction = compute_direction_from_pixel(center_col, center_row, width, height, fov_deg)
        base_direction = clamp_normalized(base_direction.dot(self._camera_to_body))
        origin = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        rotation = transformations.quaternion_matrix(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )[0:3, 0:3]
        base_world_direction = clamp_normalized(rotation.dot(base_direction))
        goal_direction, goal_distance = self._goal_vector(origin)
        adjusted_direction = base_world_direction
        if goal_direction is not None:
            safe_ratio = min(1.0, max(0.0, safe_fraction / max(self.min_safe_fraction, 1e-3)))
            blend = np.clip(
                self.goal_direction_blend + 0.35 * safe_ratio,
                0.0,
                0.95,
            )
            adjusted_direction = clamp_normalized(
                (1.0 - blend) * base_world_direction + blend * goal_direction
            )
        else:
            goal_distance = None

        velocity = self.odom.twist.twist.linear
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if speed < 1e-3:
            speed = self.default_speed
        base_vector_world = base_world_direction * speed

        distance_scale, duration_scale = self._policy_scales(safe_mask, speed)
        plan = self._plan_quintic_path(
            origin,
            rotation,
            adjusted_direction,
            speed,
            distance_scale,
            duration_scale,
            goal_distance,
        )

        if plan is None:
            self._publish_fallback(msg.header.stamp)
            self._log_timing(start_time)
            return

        path_points = plan["points"]
        velocities = plan["velocities"]
        command_vector_world = plan["command_vector"]
        travel_distance = plan["distance"]
        travel_duration = plan["duration"]
        signature = (
            round(float(command_vector_world[0]), 3),
            round(float(command_vector_world[1]), 3),
            round(float(command_vector_world[2]), 3),
            round(float(travel_distance), 2),
        )
        now_stamp = msg.header.stamp
        if not self._should_publish_plan(signature, now_stamp):
            self._log_timing(start_time)
            return
        if self._smoothed_command is None:
            self._smoothed_command = command_vector_world.copy()
        else:
            self._smoothed_command = (
                0.35 * command_vector_world + 0.65 * self._smoothed_command
            )
        command_vector_world = self._smoothed_command.copy()

        primitive_msg = Vector3Stamped()
        primitive_msg.header = msg.header
        primitive_msg.vector.x = base_vector_world[0]
        primitive_msg.vector.y = base_vector_world[1]
        primitive_msg.vector.z = base_vector_world[2]
        self.primitive_pub.publish(primitive_msg)

        command_msg = Vector3Stamped()
        command_msg.header = msg.header
        command_msg.vector.x = command_vector_world[0]
        command_msg.vector.y = command_vector_world[1]
        command_msg.vector.z = command_vector_world[2]
        self.command_pub.publish(command_msg)

        offsets_msg = Float32MultiArray()
        offsets_msg.data = [travel_distance, distance_scale, duration_scale, travel_duration]
        self.offset_pub.publish(offsets_msg)

        self._publish_trajectory(msg.header, path_points, velocities)

        self._publish_fallback(msg.header.stamp, include_default=False)
        self._log_timing(start_time)

    def _generate_fallback_vectors(self, speed: float) -> List[np.ndarray]:
        patterns = [
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            np.array([-0.8, 0.5, 0.0], dtype=np.float32),
            np.array([-0.8, -0.5, 0.0], dtype=np.float32),
            np.array([-0.6, 0.0, 0.6], dtype=np.float32),
        ]
        scaled = []
        for vec in patterns:
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                continue
            scaled.append(vec / norm * speed)
        return scaled

    def _publish_fallback(self, stamp: rospy.Time, include_default: bool = True) -> None:
        if self.odom is None:
            return
        self._last_plan_signature = None
        self._smoothed_command = None
        speed = self.default_speed
        vectors = self._generate_fallback_vectors(speed)
        rotation = transformations.quaternion_matrix(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )[0:3, 0:3]
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.odom.header.frame_id or "map"
        for vec in vectors:
            world_vec = rotation.dot(vec)
            pose = Pose()
            pose.position.x = world_vec[0]
            pose.position.y = world_vec[1]
            pose.position.z = world_vec[2]
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        self.fallback_pub.publish(pose_array)
        if include_default:
            if vectors:
                direction = rotation.dot(
                    vectors[0] / max(float(np.linalg.norm(vectors[0])), 1e-6)
                )
                self._publish_vector_trajectory(stamp, direction)
            else:
                self._publish_empty_trajectory(stamp)
        if include_default:
            primitive_msg = Vector3Stamped()
            primitive_msg.header.stamp = stamp
            primitive_msg.header.frame_id = self.odom.header.frame_id or "map"
            primitive_msg.vector.x = 0.0
            primitive_msg.vector.y = 0.0
            primitive_msg.vector.z = 0.0
            self.command_pub.publish(primitive_msg)
            offsets_msg = Float32MultiArray()
            offsets_msg.data = [0.0, 0.0, 0.0]
            self.offset_pub.publish(offsets_msg)

    def _log_timing(self, start_time: float) -> None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if elapsed_ms > 2.0:
            rospy.logwarn_throttle(1.0, "Navigation inference exceeded 2 ms: %.3f ms", elapsed_ms)

    def _should_publish_plan(
        self, signature: Tuple[float, float, float, float], stamp: rospy.Time
    ) -> bool:
        if self._last_plan_signature is None:
            self._last_plan_signature = signature
            self._last_plan_stamp = stamp
            return True
        age = (stamp - self._last_plan_stamp).to_sec()
        if age > self.plan_hold_time.to_sec():
            self._last_plan_signature = signature
            self._last_plan_stamp = stamp
            return True
        delta = sum(abs(a - b) for a, b in zip(signature, self._last_plan_signature))
        if delta >= self.plan_similarity_epsilon:
            self._last_plan_signature = signature
            self._last_plan_stamp = stamp
            return True
        return False


    def _policy_scales(self, safe_mask: np.ndarray, speed: float) -> Tuple[float, float]:
        if self.policy is None:
            return 1.0, 1.0
        mask_tensor = (
            torch.from_numpy(safe_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        speed_norm = torch.tensor([(speed - 3.0) / 4.0], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            scales = self.policy(mask_tensor, speed_norm).squeeze(0).cpu().numpy()
        distance_scale = float(
            np.clip(1.0 + 0.25 * scales[0], self.distance_scale_min, self.distance_scale_max)
        )
        duration_scale = float(
            np.clip(1.0 + 0.2 * scales[1], self.duration_scale_min, self.duration_scale_max)
        )
        return distance_scale, duration_scale

    def _plan_quintic_path(
        self,
        origin: np.ndarray,
        rotation: np.ndarray,
        world_direction: np.ndarray,
        speed: float,
        distance_scale: float,
        duration_scale: float,
        goal_distance: Optional[float],
    ) -> Optional[dict]:
        if self.odom is None:
            return None
        base_speed = speed if speed > 1e-3 else self.default_speed
        base_distance = base_speed * self.primitive_duration
        target_distance = max(0.5, base_distance * distance_scale)
        if goal_distance is not None:
            target_distance = min(target_distance, max(goal_distance - self.goal_tolerance, 0.3))
        duration = max(0.2, self.primitive_duration * duration_scale)
        origin = origin.astype(np.float32)
        world_direction = clamp_normalized(world_direction)
        target_point = origin + world_direction * target_distance
        twist = self.odom.twist.twist.linear
        start_vel_body = np.array([twist.x, twist.y, twist.z], dtype=np.float32)
        start_vel_world = rotation.dot(start_vel_body)
        end_speed = max(self.default_speed * 0.5, min(base_speed, self.default_speed * 1.2))
        end_vel = world_direction * end_speed
        zero_acc = np.zeros(3, dtype=np.float32)
        coeffs = quintic_coefficients(
            origin,
            start_vel_world,
            zero_acc,
            target_point,
            end_vel,
            zero_acc,
            duration,
        )
        points, velocities = sample_quintic(coeffs, duration, self.primitive_steps)
        if points.size == 0:
            return None
        command_vector = velocities[1] if velocities.shape[0] >= 2 else velocities[0]
        return {
            'points': points,
            'velocities': velocities,
            'command_vector': command_vector,
            'distance': float(target_distance),
            'duration': float(duration),
        }

    def _publish_trajectory(self, header, points: np.ndarray, velocities: Optional[np.ndarray] = None) -> None:
        if self.odom is None:
            return
        path_msg = Path()
        path_msg.header.stamp = header.stamp
        path_msg.header.frame_id = self.odom.header.frame_id or "map"

        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            self._publish_empty_trajectory(header.stamp)
            return

        for idx, point in enumerate(points):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            if velocities is not None and idx < velocities.shape[0]:
                direction = velocities[idx]
                if np.linalg.norm(direction) < 1e-5 and idx + 1 < velocities.shape[0]:
                    direction = velocities[idx + 1]
            elif idx < points.shape[0] - 1:
                direction = points[idx + 1] - point
            elif idx > 0:
                direction = point - points[idx - 1]
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            direction = clamp_normalized(direction)
            yaw = math.atan2(direction[1], direction[0])
            pitch = math.atan2(direction[2], math.sqrt(direction[0] ** 2 + direction[1] ** 2))
            quat = transformations.quaternion_from_euler(0.0, pitch, yaw)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            path_msg.poses.append(pose)

        self.trajectory_pub.publish(path_msg)

    def _publish_vector_trajectory(self, stamp: rospy.Time, unit_vector: np.ndarray) -> None:
        if self.odom is None:
            return
        norm = float(np.linalg.norm(unit_vector))
        if norm < 1e-6:
            self._publish_empty_trajectory(stamp)
            return
        unit_vector = unit_vector / norm
        steps = max(2, self.primitive_steps)
        step_length = self.default_speed / steps

        origin = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        points = [origin]
        current = origin.copy()
        for _ in range(steps):
            current = current + unit_vector * step_length
            points.append(current.copy())

        header = Header()
        header.stamp = stamp
        header.frame_id = self.odom.header.frame_id or "map"
        self._publish_trajectory(header, np.asarray(points, dtype=np.float32))

    def _publish_empty_trajectory(self, stamp: rospy.Time) -> None:
        if self.odom is None:
            return
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.odom.header.frame_id or "map"
        start_pose = PoseStamped()
        start_pose.header = path_msg.header
        start_pose.pose = self.odom.pose.pose
        path_msg.poses.append(start_pose)
        self.trajectory_pub.publish(path_msg)


def main() -> None:
    rospy.init_node("distance_inference")
    InferenceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
