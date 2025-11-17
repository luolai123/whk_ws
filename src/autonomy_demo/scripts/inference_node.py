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
    PrimitiveConfig,
    apply_goal_offset,
    clamp_normalized,
    compute_direction_from_pixel,
    find_largest_safe_region,
    orientation_rate_score,
    jerk_score,
    normalize_navigation_inputs,
    primitive_quintic_trajectory,
    primitive_state_dim,
    primitive_state_vector,
    project_direction_to_pixel,
    sample_motion_primitives,
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
    def __init__(self, height: int, width: int, state_dim: int = 8) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.state_dim = state_dim
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(inplace=True),
        )
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(64 + state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 4)

    def forward(self, mask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(mask)
        pooled = self.global_pool(features).view(mask.size(0), -1)
        combined = torch.cat([pooled, state], dim=1)
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
        self._body_to_camera = self._camera_mount_matrix
        self._camera_to_body = self._camera_mount_matrix.T
        rospy.loginfo(
            "camera_to_body rotation (pitch %.1f deg about Y):\n%s",
            pitch_deg,
            self._camera_to_body,
        )

        self.camera_info: Optional[CameraInfo] = None
        self.odom: Optional[Odometry] = None
        self.tan_half_h: Optional[float] = None
        self.tan_half_v: Optional[float] = None
        self.image_shape: Optional[Tuple[int, int]] = None
        self.goal_world: Optional[np.ndarray] = None
        self.goal_frame: Optional[str] = None
        self.goal_direction_blend = float(rospy.get_param("~goal_direction_blend", 0.4))
        self.goal_bias_distance = float(rospy.get_param("~goal_bias_distance", 10.0))

        primitive_steps = int(rospy.get_param("~primitive_steps", 4))
        self.primitive_steps = max(3, min(5, primitive_steps))
        self.primitive_dt = float(rospy.get_param("~primitive_dt", 0.25))
        self.primitive_duration = self.primitive_steps * self.primitive_dt
        self.primitive_candidate_count = int(rospy.get_param("~primitive_candidate_count", 24))
        primitive_seed = int(rospy.get_param("~primitive_seed", 0))
        self._rng = np.random.default_rng(primitive_seed or None)
        forward_mean = float(rospy.get_param("~v_forward_mean", 2.0))
        forward_sigma = float(rospy.get_param("~v_forward_sigma", 0.45))
        v_std_unit = float(rospy.get_param("~v_std_unit", 0.22))
        a_std_unit = float(rospy.get_param("~a_std_unit", 0.35))
        radio_range = float(rospy.get_param("~radio_range", 5.0))
        vel_max_train = float(rospy.get_param("~vel_max_train", 6.0))
        acc_max_train = float(rospy.get_param("~acc_max_train", 3.0))
        goal_length_scale = float(rospy.get_param("~goal_length_scale", 1.0))
        offset_gain = float(rospy.get_param("~offset_gain", 0.25))
        yaw_range_deg = float(
            rospy.get_param("~yaw_range_deg", rospy.get_param("~yaw_std_deg", 360.0))
        )
        pitch_std_deg = float(rospy.get_param("~pitch_std_deg", 30.0))
        roll_std_deg = float(rospy.get_param("~roll_std_deg", 30.0))
        horizon_fov = float(rospy.get_param("~horizon_camera_fov", 90.0))
        vertical_fov = float(rospy.get_param("~vertical_camera_fov", 60.0))
        self.primitive_config = PrimitiveConfig(
            radio_range=radio_range,
            vel_max_train=vel_max_train,
            acc_max_train=acc_max_train,
            forward_log_mean=math.log(max(0.2, forward_mean)),
            forward_log_sigma=max(0.05, forward_sigma),
            v_std_unit=max(0.05, v_std_unit),
            a_std_unit=max(0.05, a_std_unit),
            yaw_range_deg=yaw_range_deg,
            pitch_std_deg=pitch_std_deg,
            roll_std_deg=roll_std_deg,
            horizon_camera_fov=horizon_fov,
            vertical_camera_fov=vertical_fov,
            goal_length_scale=max(0.2, goal_length_scale),
            offset_gain=max(0.05, offset_gain),
        )
        self.goal_feature_dim = 4
        self.policy_state_dim = primitive_state_dim(self.primitive_config) + self.goal_feature_dim
        samples_per_step = int(rospy.get_param("~path_samples_per_step", 3))
        self.path_samples_per_step = max(1, samples_per_step)

        publish_default = max(0.02, self.primitive_dt / max(self.primitive_steps, 1))
        self.plan_publish_period = rospy.Duration.from_sec(
            float(rospy.get_param("~plan_publish_period", publish_default))
        )
        hold_default = max(self.plan_publish_period.to_sec(), self.primitive_dt * 0.6)
        self.plan_hold_time = rospy.Duration.from_sec(
            float(rospy.get_param("~plan_hold_time", hold_default))
        )
        self.fixed_altitude = float(rospy.get_param("~fixed_altitude", 2.0))
        self.fixed_altitude = max(0.1, self.fixed_altitude)
        self.enforce_fixed_altitude = bool(rospy.get_param("~enforce_fixed_altitude", True))
        self.offset_filter_alpha = float(rospy.get_param("~offset_filter_alpha", 0.6))
        self.offset_filter_alpha = max(0.0, min(1.0, self.offset_filter_alpha))
        self.plan_similarity_epsilon = float(rospy.get_param("~plan_similarity_epsilon", 0.35))
        self.safety_gate = float(rospy.get_param("~primitive_safety_gate", 0.6))
        self.clearance_gate = float(rospy.get_param("~primitive_clearance_gate", 0.08))
        self._last_plan_signature: Optional[Tuple[float, float, float, float]] = None
        self._last_plan_stamp = rospy.Time(0.0)
        self._smoothed_command: Optional[np.ndarray] = None
        self._last_committed_offset: Optional[np.ndarray] = None

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

    def _current_speed(self) -> float:
        if self.odom is None:
            return self.default_speed
        lin = self.odom.twist.twist.linear
        speed = math.sqrt(lin.x ** 2 + lin.y ** 2 + lin.z ** 2)
        if speed < 1e-3:
            return self.default_speed
        return float(speed)

    def goal_callback(self, msg: PoseStamped) -> None:
        self.goal_world = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32
        )
        self.goal_frame = msg.header.frame_id or "map"
        if self.odom is not None and abs(self.goal_world[2]) < 1e-3:
            self.goal_world[2] = float(self.odom.pose.pose.position.z)
        if self.enforce_fixed_altitude:
            self.goal_world[2] = self.fixed_altitude

    def _goal_vector(
        self, origin: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray], Optional[float]]:
        if self.goal_world is None:
            return None, None, None, None
        goal_vec = self.goal_world.astype(np.float32) - origin.astype(np.float32)
        goal_distance = float(np.linalg.norm(goal_vec))
        if goal_distance < max(self.goal_tolerance * 0.5, 1e-3):
            return None, goal_distance, None, None

        speed = self._current_speed()
        step_distance = speed * 0.8
        step_distance = max(0.2, min(step_distance, self.primitive_config.radio_range))
        local_distance = goal_distance if goal_distance < 0.5 else min(goal_distance, step_distance)
        direction = goal_vec / goal_distance
        local_goal = origin + direction * local_distance
        return direction, goal_distance, local_goal, local_distance

    def _ensure_policy(self) -> None:
        if self.policy is not None:
            return
        if self.image_shape is None:
            return
        height, width = self.image_shape
        self.policy = SafeNavigationPolicy(height, width, self.policy_state_dim).to(self.device)
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

        cluster_mask = np.zeros_like(safe_mask)
        min_r, max_r, min_c, max_c = region.bounds
        cluster_mask[min_r : max_r + 1, min_c : max_c + 1] = region.mask

        center_row, center_col = region.centroid
        total_pixels = height * width
        safe_fraction = region.area / float(total_pixels)

        safe_msg = PointStamped()
        safe_msg.header = msg.header
        safe_msg.point.x = (center_col + 0.5) / width
        safe_msg.point.y = (center_row + 0.5) / height
        safe_msg.point.z = safe_fraction
        self.safe_center_pub.publish(safe_msg)

        clearance_map = cv2.distanceTransform(safe_mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        max_clearance = float(np.max(clearance_map))
        if max_clearance > 1e-6:
            clearance_map /= max_clearance

        fov_deg = math.degrees(2.0 * math.atan(self.tan_half_h))
        base_direction_camera = compute_direction_from_pixel(
            center_col, center_row, width, height, fov_deg
        )
        base_direction_body = clamp_normalized(self._camera_to_body.dot(base_direction_camera))
        origin = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        if self.enforce_fixed_altitude:
            origin[2] = self.fixed_altitude
        rotation = transformations.quaternion_matrix(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )[0:3, 0:3]
        goal_direction, goal_distance, local_goal, local_distance = self._goal_vector(origin)

        plan = self._plan_motion_primitive(
            origin,
            rotation,
            base_direction_camera,
            cluster_mask,
            clearance_map,
            center_row,
            center_col,
            goal_direction,
            goal_distance,
            local_goal,
            local_distance,
        )

        if plan is None:
            self._publish_fallback(msg.header.stamp)
            self._log_timing(start_time)
            return

        path_points = plan["points_world"]
        velocities = plan["velocities_world"]
        command_vector_world = plan["command_vector"]
        travel_distance = plan["distance"]
        travel_duration = plan["duration"]
        distance_scale = plan["distance_scale"]
        duration_scale = plan["duration_scale"]
        base_velocity_body = plan["sample"].start_vel_body.copy()
        if self.enforce_fixed_altitude:
            base_velocity_body[2] = 0.0
        base_vector_world = rotation.dot(base_velocity_body)
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

        plan_sample = plan["sample"]
        rospy.loginfo_throttle(
            1.0,
            "Goal dir (cam): %s | yaw/pitch/roll offsets (deg): %.1f/%.1f/%.1f | goal dir (body): %s | offset norm: %.3f",
            np.array2string(plan_sample.base_direction_camera, precision=3),
            math.degrees(plan_sample.yaw_offset),
            math.degrees(plan_sample.pitch_offset),
            math.degrees(plan_sample.roll_offset),
            np.array2string(plan_sample.goal_direction_body, precision=3),
            plan["offset_norm"],
        )

        primitive_msg = Vector3Stamped()
        primitive_msg.header = msg.header
        primitive_msg.vector.x = base_vector_world[0]
        primitive_msg.vector.y = base_vector_world[1]
        primitive_msg.vector.z = base_vector_world[2]
        self.primitive_pub.publish(primitive_msg)

        self._last_committed_offset = plan.get("offset_vector", None)
        if self._last_committed_offset is not None:
            self._last_committed_offset = self._last_committed_offset.copy()

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
        self._last_committed_offset = None
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
        publish_period = self.plan_publish_period.to_sec()
        hold_time = self.plan_hold_time.to_sec()

        if age >= publish_period:
            self._last_plan_signature = signature
            self._last_plan_stamp = stamp
            return True

        delta = sum(abs(a - b) for a, b in zip(signature, self._last_plan_signature))
        if delta >= self.plan_similarity_epsilon:
            self._last_plan_signature = signature
            self._last_plan_stamp = stamp
            return True

        if age >= hold_time:
            self._last_plan_signature = signature
            self._last_plan_stamp = stamp
            return True

        return False


    def _policy_offsets(
        self, safe_mask: np.ndarray, state_vec: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        if self.policy is None:
            return np.zeros(3, dtype=np.float32), 1.0
        mask_tensor = (
            torch.from_numpy(safe_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        state_tensor = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.policy(mask_tensor, state_tensor).squeeze(0).cpu().numpy()
        offset = np.clip(outputs[0:3], -1.0, 1.0)
        duration_scale = float(
            np.clip(1.0 + 0.2 * outputs[3], self.duration_scale_min, self.duration_scale_max)
        )
        offset_vec = offset * self.primitive_config.radio_range
        offset_vec = offset_vec.astype(np.float32)
        if self.enforce_fixed_altitude:
            offset_vec[2] = 0.0
        if self._last_committed_offset is not None and self.offset_filter_alpha < 1.0:
            offset_vec = (
                self.offset_filter_alpha * offset_vec
                + (1.0 - self.offset_filter_alpha) * self._last_committed_offset
            )
        return offset_vec, duration_scale

    def _plan_motion_primitive(
        self,
        origin: np.ndarray,
        rotation: np.ndarray,
        base_direction_camera: np.ndarray,
        cluster_mask: np.ndarray,
        clearance_map: np.ndarray,
        center_row: float,
        center_col: float,
        goal_direction: Optional[np.ndarray],
        goal_distance: Optional[float],
        local_goal: Optional[np.ndarray],
        local_distance: Optional[float],
    ) -> Optional[dict]:
        if cluster_mask.size == 0:
            return None
        height, width = cluster_mask.shape
        if width == 0 or height == 0:
            return None
        fov_deg = math.degrees(2.0 * math.atan(self.tan_half_h)) if self.tan_half_h else 120.0
        primitives = sample_motion_primitives(
            base_direction_camera,
            self._camera_to_body,
            self._rng,
            self.primitive_config,
            self.primitive_candidate_count,
        )
        best_plan: Optional[dict] = None
        best_score = -float("inf")
        origin = origin.astype(np.float32)
        if self.enforce_fixed_altitude:
            origin = origin.copy()
            origin[2] = self.fixed_altitude
        sample_count = max(
            self.primitive_steps,
            self.primitive_steps * self.path_samples_per_step,
        )
        target_distance = local_distance if local_distance is not None else goal_distance
        for sample in primitives:
            state_vec = primitive_state_vector(sample, self.primitive_config)
            target_hint = np.zeros(3, dtype=np.float32)
            target_bias = 1.0
            if local_goal is not None:
                local_hint = rotation.T.dot(local_goal - origin)
                target_hint = clamp_normalized(local_hint)
                target_bias = min(
                    1.0, np.linalg.norm(local_hint) / max(self.primitive_config.radio_range, 1e-3)
                )
            elif goal_direction is not None:
                target_hint = clamp_normalized(rotation.T.dot(goal_direction))
                target_bias = 0.5
            goal_features = np.concatenate(
                [target_hint, np.array([target_bias], dtype=np.float32)]
            ).astype(np.float32)
            state_vec = np.concatenate([state_vec, goal_features]).astype(np.float32)
            offset_vec, duration_scale = self._policy_offsets(cluster_mask, state_vec)
            goal_body = apply_goal_offset(sample, offset_vec, self.primitive_config)
            if self.enforce_fixed_altitude:
                goal_body = goal_body.copy()
                goal_body[2] = 0.0
            points_body, velocities_body, duration = primitive_quintic_trajectory(
                sample,
                goal_body,
                duration_scale,
                sample_count,
            )
            if points_body.size == 0:
                continue
            if self.enforce_fixed_altitude:
                points_body[:, 2] = 0.0
                velocities_body[:, 2] = 0.0
            evaluation = self._evaluate_candidate(
                points_body,
                cluster_mask,
                clearance_map,
                center_row,
                center_col,
                width,
                height,
                fov_deg,
            )
            if evaluation is None:
                continue
            if (
                evaluation["safety"] < self.safety_gate
                or evaluation["clearance"] < self.clearance_gate
            ):
                continue
            world_points = origin + rotation.dot(points_body.T).T
            world_velocities = rotation.dot(velocities_body.T).T
            if self.enforce_fixed_altitude:
                world_points[:, 2] = self.fixed_altitude
                world_velocities[:, 2] = 0.0
            travel_distance = float(np.linalg.norm(world_points[-1] - origin))
            jerk_metric = jerk_score(
                world_points, duration / max(points_body.shape[0] - 1, 1)
            )
            orientation_metric = orientation_rate_score(world_velocities)
            goal_alignment = 0.0
            if goal_direction is not None:
                final_dir = clamp_normalized(world_points[-1] - origin)
                goal_alignment = float(np.dot(final_dir, goal_direction))
            goal_alignment_score = (goal_alignment + 1.0) * 0.5
            diag = math.sqrt(width ** 2 + height ** 2)
            goal_score = math.exp(-evaluation["goal_error"] / max(diag, 1e-3))
            _, _, normalized_goal = normalize_navigation_inputs(
                origin,
                world_velocities[-1],
                (world_points[-1] - origin),
                self.primitive_config,
            )
            normalized_local_goal = None
            if local_goal is not None:
                _, _, normalized_local_goal = normalize_navigation_inputs(
                    origin, world_velocities[-1], local_goal - origin, self.primitive_config
                )
            normalized_goal_error = 0.0
            if normalized_local_goal is not None:
                normalized_goal_error = float(
                    np.linalg.norm(normalized_goal - normalized_local_goal)
                )
            smoothness_score = evaluation["smoothness"]
            jerk_score_val = max(0.0, min(1.0, jerk_metric))
            orientation_score_val = max(0.0, min(1.0, orientation_metric))
            command_vector = (
                world_velocities[1] if world_velocities.shape[0] >= 2 else world_velocities[0]
            )
            continuity_score = 1.0
            if self._smoothed_command is not None:
                prev_dir = clamp_normalized(self._smoothed_command)
                new_dir = clamp_normalized(command_vector)
                heading_delta = math.acos(
                    max(-1.0, min(1.0, float(np.dot(prev_dir, new_dir))))
                )
                continuity_score = math.exp(-heading_delta)
            score = (
                0.45 * evaluation["safety"]
                + 0.2 * evaluation["clearance"]
                + 0.18 * goal_score
                + 0.1 * goal_alignment_score
                + 0.07 * smoothness_score
                + 0.03 * jerk_score_val
                + 0.02 * orientation_score_val
                + 0.05 * continuity_score
            )
            if normalized_local_goal is not None:
                score -= min(0.35, normalized_goal_error * 0.25)
            if target_distance is not None:
                overreach = max(
                    0.0,
                    travel_distance - max(target_distance - self.goal_tolerance, 0.5),
                )
                if overreach > 0.0:
                    score -= min(0.35, overreach / max(target_distance, 1.0))
            if score <= best_score:
                continue
            best_plan = {
                "points_world": world_points,
                "velocities_world": world_velocities,
                "command_vector": command_vector,
                "distance": travel_distance,
                "duration": duration,
                "distance_scale": float(
                    np.clip(
                        evaluation["distance_scale"],
                        self.distance_scale_min,
                        self.distance_scale_max,
                    )
                ),
                "duration_scale": duration_scale,
                "offset_norm": float(np.linalg.norm(offset_vec)),
                "offset_vector": offset_vec.copy(),
                "sample": sample,
            }
            best_score = score
        return best_plan

    def _evaluate_candidate(
        self,
        points_body: np.ndarray,
        cluster_mask: np.ndarray,
        clearance_map: np.ndarray,
        center_row: float,
        center_col: float,
        width: int,
        height: int,
        fov_deg: float,
    ) -> Optional[dict]:
        if points_body.shape[0] < 2:
            return None
        safety_hits: List[float] = []
        clearance_values: List[float] = []
        smoothness_sum = 0.0
        last_col = center_col
        last_row = center_row
        prev_dir: Optional[np.ndarray] = None
        for idx in range(1, points_body.shape[0]):
            direction_body = clamp_normalized(points_body[idx])
            direction_camera = self._body_to_camera.dot(direction_body)
            col, row = project_direction_to_pixel(
                direction_camera, width, height, fov_deg
            )
            last_col, last_row = col, row
            col_i = int(round(col))
            row_i = int(round(row))
            if 0 <= row_i < height and 0 <= col_i < width:
                safe = bool(cluster_mask[row_i, col_i])
                safety_hits.append(1.0 if safe else 0.0)
                clearance_values.append(float(clearance_map[row_i, col_i]))
            else:
                safety_hits.append(0.0)
                clearance_values.append(0.0)
            if prev_dir is not None:
                smoothness_sum += max(
                    -1.0, min(1.0, float(np.dot(prev_dir, direction_body)))
                )
            prev_dir = direction_body
        if not safety_hits:
            return None
        safety_ratio = sum(safety_hits) / len(safety_hits)
        clearance_min = max(0.0, min(clearance_values))
        smoothness_score = max(
            0.0,
            min(1.0, (smoothness_sum / max(len(safety_hits) - 1, 1) + 1.0) * 0.5),
        )
        goal_error = math.sqrt((last_col - center_col) ** 2 + (last_row - center_row) ** 2)
        distance_scale = float(
            np.linalg.norm(points_body[-1]) / max(self.primitive_config.radio_range, 1e-3)
        )
        return {
            "safety": safety_ratio,
            "clearance": clearance_min,
            "goal_error": goal_error,
            "smoothness": smoothness_score,
            "distance_scale": distance_scale,
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
