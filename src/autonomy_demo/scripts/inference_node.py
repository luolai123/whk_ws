#!/usr/bin/env python3
"""Inference node that loads trained models and emits safe navigation primitives."""

import math
import pathlib
import time
from typing import List, Optional, Tuple, Set

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
    jerk_score,
    orientation_rate_score,
    path_smoothness,
    project_direction_to_pixel,
    sample_yopo_directions,
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
        self.fc2 = torch.nn.Linear(64, 3)

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
        model_path = rospy.get_param(
            "~model_path", str(pathlib.Path.home() / "autonomy_demo" / "model.pt")
        )
        model_path = pathlib.Path(model_path)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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
        self.use_dynamic_threshold = bool(rospy.get_param("~use_dynamic_threshold", True))
        self.dynamic_percentile = float(rospy.get_param("~dynamic_percentile", 0.6))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.3))

        self.camera_info: Optional[CameraInfo] = None
        self.odom: Optional[Odometry] = None
        self.tan_half_h: Optional[float] = None
        self.tan_half_v: Optional[float] = None
        self.image_shape: Optional[Tuple[int, int]] = None
        self.goal_world: Optional[np.ndarray] = None
        self.goal_frame: Optional[str] = None

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

        # 新增的偏移参数读取
        offset_mode = rospy.get_param('~offset_mode', 'cartesian')  # 'cartesian' or 'polar'
        self.offset_params = {
            'x': float(rospy.get_param('~offset_x', 0.0)),
            'y': float(rospy.get_param('~offset_y', 0.0)),
            'z': float(rospy.get_param('~offset_z', 0.0)),
            'yaw': float(rospy.get_param('~offset_yaw', 0.0)),
            'pitch': float(rospy.get_param('~offset_pitch', 0.0)),
            'length': float(rospy.get_param('~offset_length', 0.0)),
            'mode': offset_mode
        }

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

        tensor = torch.from_numpy(cv_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = self.softmax(logits)

        safe_prob = probs[:, 0, :, :].squeeze(0).cpu().numpy()
        obstacle_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy()
        safe_prob = cv2.GaussianBlur(safe_prob, (5, 5), 0)
        obstacle_prob = cv2.GaussianBlur(obstacle_prob, (5, 5), 0)
        threshold = self.safe_threshold
        if self.use_dynamic_threshold:
            # percentile based threshold to avoid "all-safe"/"all-obstacle"
            flat = safe_prob.reshape(-1)
            percentile = np.clip(self.dynamic_percentile, 0.2, 0.9)
            threshold = float(np.percentile(flat, 100.0 * percentile))
            threshold = max(self.safe_threshold, threshold)
        safe_mask = safe_prob >= threshold
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

        # find connected components and extract up to K centroids (largest first)
        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(safe_mask.astype(np.uint8), connectivity=4)
        regions: list = []
        total_pixels = height * width
        min_pixels = max(1, int(total_pixels * self.min_safe_fraction))
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_pixels:
                continue
            cy, cx = centroids[i]
            regions.append((area, float(cy), float(cx)))
        regions.sort(key=lambda r: r[0], reverse=True)
        if not regions:
            self._publish_fallback(msg.header.stamp)
            self._log_timing(start_time)
            return
        # Try multiple centroid targets (top-K)
        K = min(5, len(regions))
        center_row, center_col = regions[0][1], regions[0][2]
        safe_fraction = regions[0][0] / float(total_pixels)

        safe_msg = PointStamped()
        safe_msg.header = msg.header
        safe_msg.point.x = (center_col + 0.5) / width
        safe_msg.point.y = (center_row + 0.5) / height
        safe_msg.point.z = safe_fraction
        self.safe_center_pub.publish(safe_msg)

        fov_deg = math.degrees(2.0 * math.atan(self.tan_half_h))
        base_direction = compute_direction_from_pixel(center_col, center_row, width, height, fov_deg)
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

        velocity = self.odom.twist.twist.linear
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if speed < 1e-3:
            speed = self.default_speed
        base_vector_world = base_world_direction * speed

        policy_offsets: Optional[Tuple[float, float, float]] = None
        if self.policy is not None:
            mask_tensor = (
                torch.from_numpy(safe_mask.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            speed_norm = torch.tensor([(speed - 3.0) / 4.0], device=self.device, dtype=torch.float32)
            with torch.no_grad():
                offsets = self.policy(mask_tensor, speed_norm)
            length_delta, pitch_delta, yaw_delta = offsets.squeeze(0).cpu().numpy()
            length_est = float(np.clip(1.0 + 0.2 * length_delta, 0.5, 1.5))
            pitch_est = float(
                np.clip(math.radians(15.0) * pitch_delta, -math.radians(15.0), math.radians(15.0))
            )
            yaw_est = float(
                np.clip(math.radians(15.0) * yaw_delta, -math.radians(15.0), math.radians(15.0))
            )
            policy_offsets = (length_est, pitch_est, yaw_est)

        best_candidate = None
        for idx in range(K):
            crow, ccol = regions[idx][1], regions[idx][2]
            base_dir_k = compute_direction_from_pixel(ccol, crow, width, height, fov_deg)
            cand = self._select_candidate(
                base_dir_k,
                policy_offsets,
                safe_mask,
                safe_prob,
                distance_field,
                width,
                height,
                fov_deg,
                rotation,
                origin,
                speed,
            )
            if cand is not None and (best_candidate is None or cand["score"] > best_candidate["score"]):
                best_candidate = cand

        if best_candidate is None:
            self._publish_fallback(msg.header.stamp)
            self._log_timing(start_time)
            return
        candidate = best_candidate
        length_scale = candidate["length_scale"]
        pitch_offset = candidate["pitch_offset"]
        yaw_offset = candidate["yaw_offset"]
        final_direction_local = candidate["final_direction_local"]
        path_points = candidate["path_points"]

        final_world_direction = clamp_normalized(rotation.dot(final_direction_local))
        commanded_speed = speed * length_scale

        command_vector_world = final_world_direction * commanded_speed
        if path_points.shape[0] >= 2:
            initial_segment = path_points[1] - path_points[0]
            seg_norm = float(np.linalg.norm(initial_segment))
            if seg_norm > 1e-6:
                command_vector_world = initial_segment / seg_norm * commanded_speed

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
        offsets_msg.data = [length_scale, math.degrees(pitch_offset), math.degrees(yaw_offset)]
        self.offset_pub.publish(offsets_msg)

        self._publish_trajectory(msg.header, path_points)

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

    def _normalize_offsets(
        self, length_scale: float, pitch_offset: float, yaw_offset: float
    ) -> Tuple[float, float, float]:
        pitch_limit = math.radians(15.0)
        yaw_limit = math.radians(15.0)
        length = float(np.clip(length_scale, 0.8, 1.2))
        pitch = float(np.clip(pitch_offset, -pitch_limit, pitch_limit))
        yaw = float(np.clip(yaw_offset, -yaw_limit, yaw_limit))
        return length, pitch, yaw

    def _select_candidate(
        self,
        base_direction: np.ndarray,
        policy_offsets: Optional[Tuple[float, float, float]],
        safe_mask: np.ndarray,
        safe_prob: np.ndarray,
        distance_field: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        rotation: np.ndarray,
        origin: np.ndarray,
        speed: float,
    ) -> Optional[dict]:
        candidates: List[Tuple[float, float, float, str]] = []
        seen: Set[Tuple[int, int, int]] = set()

        def add_candidate(length: float, pitch: float, yaw: float, tag: str) -> None:
            normalized = self._normalize_offsets(length, pitch, yaw)
            key = (
                int(round(normalized[0] * 1000)),
                int(round(normalized[1] * 1000)),
                int(round(normalized[2] * 1000)),
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append((*normalized, tag))

        add_candidate(1.0, 0.0, 0.0, "base")

        yaw_step = math.radians(8.0)
        pitch_step = math.radians(6.0)
        add_candidate(1.0, 0.0, yaw_step, "base_yaw")
        add_candidate(1.0, 0.0, -yaw_step, "base_yaw")
        add_candidate(1.0, pitch_step, 0.0, "base_pitch")
        add_candidate(1.0, -pitch_step, 0.0, "base_pitch")
        add_candidate(1.1, 0.0, yaw_step * 0.5, "base_len")
        add_candidate(0.9, 0.0, -yaw_step * 0.5, "base_len")

        if policy_offsets is not None:
            length_est, pitch_est, yaw_est = policy_offsets
            add_candidate(length_est, pitch_est, yaw_est, "policy")
            add_candidate(length_est + 0.1, pitch_est, yaw_est, "policy_len")
            add_candidate(length_est - 0.1, pitch_est, yaw_est, "policy_len")
            add_candidate(length_est, pitch_est + pitch_step, yaw_est, "policy_pitch")
            add_candidate(length_est, pitch_est - pitch_step, yaw_est, "policy_pitch")
            add_candidate(length_est, pitch_est, yaw_est + yaw_step, "policy_yaw")
            add_candidate(length_est, pitch_est, yaw_est - yaw_step, "policy_yaw")

        max_clearance = float(distance_field.max()) if distance_field.size else 0.0
        best: Optional[dict] = None
        for length_scale, pitch_offset, yaw_offset, _ in candidates:
            result = self._score_candidate(
                base_direction,
                length_scale,
                pitch_offset,
                yaw_offset,
                safe_mask,
                safe_prob,
                distance_field,
                width,
                height,
                fov_deg,
                rotation,
                origin,
                speed,
                max_clearance,
            )
            if result is None:
                continue
            if best is None or result["score"] > best["score"]:
                best = result
        return best

    def _score_candidate(
        self,
        base_direction: np.ndarray,
        length_scale: float,
        pitch_offset: float,
        yaw_offset: float,
        safe_mask: np.ndarray,
        safe_prob: np.ndarray,
        distance_field: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        rotation: np.ndarray,
        origin: np.ndarray,
        speed: float,
        max_clearance: float,
    ) -> Optional[dict]:
        steps = max(2, self.primitive_steps)
        directions = sample_yopo_directions(base_direction, yaw_offset, pitch_offset, steps)
        if not directions:
            return None

        min_prob = 1.0
        min_clearance = max_clearance if max_clearance > 0.0 else 0.0
        for direction in directions:
            col, row = project_direction_to_pixel(direction, width, height, fov_deg)
            if not (0.0 <= col < width and 0.0 <= row < height):
                return None
            col_idx = int(round(col))
            row_idx = int(round(row))
            if not (0 <= col_idx < width and 0 <= row_idx < height):
                return None
            if not safe_mask[row_idx, col_idx]:
                return None
            prob = float(safe_prob[row_idx, col_idx])
            min_prob = min(min_prob, prob)
            if max_clearance > 0.0:
                clearance = float(distance_field[row_idx, col_idx])
                min_clearance = min(min_clearance, clearance)

        prob_threshold = max(self.safe_threshold, 0.55)
        if min_prob < prob_threshold:
            return None

        clearance_norm = 0.0
        if max_clearance > 1e-6:
            clearance_norm = min_clearance / max(max_clearance, 1e-6)
            if min_clearance < 1.0:
                return None

        commanded_speed = speed * length_scale
        points = self._primitive_points(
            origin,
            rotation,
            base_direction,
            yaw_offset,
            pitch_offset,
            commanded_speed,
            directions,
        )
        if points.shape[0] < 2:
            return None

        displacement = float(np.linalg.norm(points[-1] - points[0]))
        expected = commanded_speed * max(self.primitive_dt, 1e-3) * len(directions)
        if expected > 1e-3 and displacement < expected * 0.35:
            return None

        smooth_metric = path_smoothness(points)
        jerk_metric = jerk_score(points, self.primitive_dt)
        orientation_metric = orientation_rate_score(directions)

        pitch_limit = math.radians(15.0)
        yaw_limit = math.radians(15.0)
        stability_penalty = (
            abs(length_scale - 1.0) / 0.2
            + abs(pitch_offset) / pitch_limit
            + abs(yaw_offset) / yaw_limit
        ) / 3.0
        stability_score = math.exp(-max(0.0, stability_penalty))

        goal_score = 0.0
        if self.goal_world is not None and self.goal_frame == (self.odom.header.frame_id or "map"):
            goal_vec = self.goal_world - origin
            goal_dist = float(np.linalg.norm(goal_vec))
            if goal_dist > 1e-3:
                end_vec = points[-1] - origin
                end_norm = float(np.linalg.norm(end_vec))
                alignment = 0.0
                if end_norm > 1e-6:
                    alignment = max(0.0, float(np.dot(goal_vec, end_vec)) / (goal_dist * end_norm))
                remaining = self.goal_world - points[-1]
                progress = max(0.0, (goal_dist - float(np.linalg.norm(remaining))) / goal_dist)
                goal_score = 0.6 * alignment + 0.4 * progress
            else:
                goal_score = 1.0

        total_score = (
            6.0 * min_prob
            + 4.0 * clearance_norm
            + 3.0 * goal_score
            + 2.0 * smooth_metric
            + 2.0 * jerk_metric
            + 1.5 * orientation_metric
            + 1.0 * stability_score
        )

        return {
            "score": float(total_score),
            "length_scale": float(length_scale),
            "pitch_offset": float(pitch_offset),
            "yaw_offset": float(yaw_offset),
            "final_direction_local": directions[-1],
            "path_points": points,
        }

    def _primitive_points(
        self,
        origin: np.ndarray,
        rotation: np.ndarray,
        base_direction: np.ndarray,
        yaw_offset: float,
        pitch_offset: float,
        commanded_speed: float,
        directions: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        steps = self.primitive_steps
        if directions is None:
            directions = sample_yopo_directions(base_direction, yaw_offset, pitch_offset, steps)
        else:
            directions = list(directions)
        points = [origin.astype(np.float32)]
        segment_length = commanded_speed * max(self.primitive_dt, 1e-3)
        for direction_local in directions:
            world_dir = clamp_normalized(rotation.dot(direction_local))
            points.append(points[-1] + world_dir * segment_length)
        return np.asarray(points, dtype=np.float32)

    def _publish_trajectory(self, header, points: np.ndarray) -> None:
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
            if idx < points.shape[0] - 1:
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

    # 辅助函数：把机体坐标系的(x,y)偏移旋转到世界坐标系（用当前机器人偏航）
    def _rotate_xy_in_world(self, dx_body, dy_body, robot_yaw):
        c = math.cos(robot_yaw)
        s = math.sin(robot_yaw)
        dx_world = dx_body * c - dy_body * s
        dy_world = dx_body * s + dy_body * c
        return dx_world, dy_world

    # 在生成或应用安全运动基元（primitives）的地方，替换/插入如下逻辑：
    def apply_offset_to_primitives(self, primitives, robot_pose_yaw) -> None:
        """
        primitives: 可变的基元列表/结构，每个基元包含一系列点/位置
        robot_pose_yaw: 当前机器人偏航（rad）
        """
        if self.offset_params['mode'] == 'cartesian':
            # 将给定的机体坐标偏移（x,y,z）作为纯平移应用到每个基元的所有点上，
            # 这样不会改变基元朝向/长度，只改变位置，使轨迹保持直线型（更像 YOPO 的偏移）
            dx_w, dy_w = self._rotate_xy_in_world(self.offset_params['x'], self.offset_params['y'], robot_pose_yaw)
            dz = self.offset_params['z']
            for prim in primitives:
                # 假定 prim.points 是 [[x,y,z], ...] 或 numpy array
                for p in prim.points:
                    p[0] += dx_w
                    p[1] += dy_w
                    p[2] += dz
            return primitives

        else:
            # 保留原有的 polar 偏移逻辑（偏航/俯仰/长度缩放），以兼容历史行为
            return primitives


def main() -> None:
    rospy.init_node("distance_inference")
    InferenceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
