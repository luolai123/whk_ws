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
    cubic_hermite_path,
    find_largest_safe_region,
    is_pixel_safe,
    project_direction_to_pixel,
    rotate_direction,
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

        self.trajectory_steps = int(rospy.get_param("~trajectory_steps", 12))

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
        safe_mask = safe_prob >= self.safe_threshold
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(safe_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        safe_mask = cleaned.astype(bool)

        color_map = np.zeros((height, width, 3), dtype=np.uint8)
        color_map[:, :, 1] = np.clip(safe_prob * 255.0, 0, 255).astype(np.uint8)
        color_map[:, :, 0] = np.clip(obstacle_prob * 255.0, 0, 255).astype(np.uint8)
        color_map[:, :, 2] = np.clip((1.0 - safe_prob) * 80.0, 0, 255).astype(np.uint8)
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

        yaw_offset = 0.0
        pitch_offset = 0.0
        length_scale = 1.0
        final_direction_local = base_direction

        if self.policy is not None:
            mask_tensor = torch.from_numpy(safe_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            speed_norm = torch.tensor([(speed - 3.0) / 4.0], device=self.device, dtype=torch.float32)
            with torch.no_grad():
                offsets = self.policy(mask_tensor, speed_norm)
            length_delta, pitch_delta, yaw_delta = offsets.squeeze(0).cpu().numpy()
            length_scale = float(np.clip(1.0 + 0.2 * length_delta, 0.5, 1.5))
            pitch_offset = float(np.clip(math.radians(15.0) * pitch_delta, -math.radians(15.0), math.radians(15.0)))
            yaw_offset = float(np.clip(math.radians(15.0) * yaw_delta, -math.radians(15.0), math.radians(15.0)))
            final_direction_local = rotate_direction(base_direction, yaw_offset, pitch_offset)
            col_pred, row_pred = project_direction_to_pixel(final_direction_local, width, height, fov_deg)
            if not is_pixel_safe(safe_mask, col_pred, row_pred):
                for scale in (0.5, 0.25, 0.0):
                    test_yaw = yaw_offset * scale
                    test_pitch = pitch_offset * scale
                    test_length = 1.0 + (length_scale - 1.0) * scale
                    direction_candidate = rotate_direction(base_direction, test_yaw, test_pitch)
                    col_candidate, row_candidate = project_direction_to_pixel(
                        direction_candidate, width, height, fov_deg
                    )
                    if is_pixel_safe(safe_mask, col_candidate, row_candidate):
                        yaw_offset = test_yaw
                        pitch_offset = test_pitch
                        length_scale = float(np.clip(test_length, 0.5, 1.5))
                        final_direction_local = direction_candidate
                        break
                else:
                    yaw_offset = 0.0
                    pitch_offset = 0.0
                    length_scale = 1.0
                    final_direction_local = base_direction
        else:
            final_direction_local = base_direction

        yaw_offset, pitch_offset, length_scale = self._evaluate_offsets(
            base_direction,
            yaw_offset,
            pitch_offset,
            length_scale,
            safe_mask,
            width,
            height,
            fov_deg,
        )
        final_direction_local = rotate_direction(base_direction, yaw_offset, pitch_offset)

        final_world_direction = clamp_normalized(rotation.dot(final_direction_local))
        commanded_speed = speed * length_scale
        path_points = self._plan_trajectory_points(
            origin,
            rotation,
            base_direction,
            final_direction_local,
            final_world_direction,
            yaw_offset,
            pitch_offset,
            commanded_speed,
            safe_mask,
            width,
            height,
            fov_deg,
        )

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

    def _evaluate_offsets(
        self,
        base_direction: np.ndarray,
        yaw_offset: float,
        pitch_offset: float,
        length_scale: float,
        safe_mask: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
    ) -> Tuple[float, float, float]:
        target_scale = float(length_scale)
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            yaw_candidate = yaw_offset * scale
            pitch_candidate = pitch_offset * scale
            length_candidate = 1.0 + (target_scale - 1.0) * scale
            length_candidate = float(np.clip(length_candidate, 0.5, 1.5))
            if self._trajectory_is_safe(
                base_direction,
                yaw_candidate,
                pitch_candidate,
                length_candidate,
                safe_mask,
                width,
                height,
                fov_deg,
            ):
                return yaw_candidate, pitch_candidate, length_candidate
        return 0.0, 0.0, 1.0

    def _trajectory_is_safe(
        self,
        base_direction: np.ndarray,
        yaw_offset: float,
        pitch_offset: float,
        length_scale: float,
        safe_mask: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
    ) -> bool:
        directions = sample_yopo_directions(
            base_direction,
            yaw_offset,
            pitch_offset,
            max(2, self.trajectory_steps),
        )
        for direction in directions:
            col, row = project_direction_to_pixel(direction, width, height, fov_deg)
            if not is_pixel_safe(safe_mask, col, row):
                return False
        return True

    def _primitive_points(
        self,
        origin: np.ndarray,
        rotation: np.ndarray,
        base_direction: np.ndarray,
        yaw_offset: float,
        pitch_offset: float,
        commanded_speed: float,
    ) -> np.ndarray:
        steps = max(2, self.trajectory_steps)
        directions = sample_yopo_directions(base_direction, yaw_offset, pitch_offset, steps)
        points = [origin.astype(np.float32)]
        segment_length = commanded_speed / max(1, steps)
        for direction_local in directions:
            world_dir = clamp_normalized(rotation.dot(direction_local))
            points.append(points[-1] + world_dir * segment_length)
        return np.asarray(points, dtype=np.float32)

    def _path_is_safe_local(
        self,
        points: np.ndarray,
        rotation: np.ndarray,
        safe_mask: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
    ) -> bool:
        if safe_mask.size == 0:
            return False
        local_basis = rotation.T
        for idx in range(points.shape[0] - 1):
            segment = points[idx + 1] - points[idx]
            norm = float(np.linalg.norm(segment))
            if norm < 1e-6:
                continue
            direction_world = segment / norm
            direction_local = local_basis.dot(direction_world)
            col, row = project_direction_to_pixel(direction_local, width, height, fov_deg)
            if not is_pixel_safe(safe_mask, col, row):
                return False
        return True

    def _plan_trajectory_points(
        self,
        origin: np.ndarray,
        rotation: np.ndarray,
        base_direction: np.ndarray,
        final_direction_local: np.ndarray,
        final_direction_world: np.ndarray,
        yaw_offset: float,
        pitch_offset: float,
        commanded_speed: float,
        safe_mask: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
    ) -> np.ndarray:
        primitive = self._primitive_points(
            origin,
            rotation,
            base_direction,
            yaw_offset,
            pitch_offset,
            commanded_speed,
        )

        if self.goal_world is None or self.odom is None:
            return primitive

        odom_frame = self.odom.header.frame_id or "map"
        if self.goal_frame and self.goal_frame not in ("", odom_frame):
            rospy.logwarn_throttle(
                5.0,
                "Goal frame %s does not match odometry frame %s; following primitive path",
                self.goal_frame,
                odom_frame,
            )
            return primitive

        goal_vector = self.goal_world - origin
        distance_to_goal = float(np.linalg.norm(goal_vector))
        if distance_to_goal <= self.goal_tolerance:
            self.goal_world = None
            return primitive
        if distance_to_goal < 1e-3:
            return primitive

        steps = max(2, self.trajectory_steps)
        tangent_start = final_direction_world * min(distance_to_goal, commanded_speed)
        tangent_end = clamp_normalized(goal_vector) * min(distance_to_goal, commanded_speed * 0.5)
        hermite_points = cubic_hermite_path(origin, self.goal_world, tangent_start, tangent_end, steps)
        if self._path_is_safe_local(hermite_points, rotation, safe_mask, width, height, fov_deg):
            return hermite_points.astype(np.float32)
        return primitive

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
        steps = max(2, self.trajectory_steps)
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
