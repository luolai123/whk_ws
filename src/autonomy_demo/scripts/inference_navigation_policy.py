#!/usr/bin/env python3
"""独立的安全基元偏航推理节点 - 根据安全区域作为下一个目标点"""

import math
import pathlib
import time
from typing import List, Optional, Set, Tuple

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
    calculate_yaw,
    compute_direction_from_pixel,
    denormalize_weight,
    find_largest_safe_region,
    jerk_score,
    orientation_rate_score,
    path_smoothness,
    Poly5Solver,
    project_direction_to_pixel,
    smooth_trajectory,
    smoothness_penalty,
    sample_yopo_directions,
    rotate_direction,
)

# 从原始推理脚本导入模型定义
import sys
import pathlib
# 添加当前目录到路径以便导入
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from inference_node import SafeNavigationPolicy


class NavigationPolicyInferenceNode:
    """安全导航策略推理节点 - 根据安全区域生成目标点和偏航指令"""
    
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy: Optional[SafeNavigationPolicy] = None
        self.policy_state: Optional[dict] = None
        policy_path = rospy.get_param(
            "~policy_path", str(pathlib.Path.home() / "autonomy_demo" / "navigation_policy.pt")
        )
        policy_path = pathlib.Path(policy_path)
        if policy_path.exists():
            try:
                self.policy_state = torch.load(policy_path, map_location=self.device)
                rospy.loginfo("已加载导航策略从 %s", policy_path)
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logwarn("加载导航策略失败 %s: %s", policy_path, exc)
                self.policy_state = None
        else:
            rospy.logwarn("导航策略权重 %s 未找到", policy_path)

        self.min_safe_fraction = float(rospy.get_param("~min_safe_fraction", 0.05))
        self.default_speed = float(rospy.get_param("~default_speed", 3.0))
        self.safe_threshold = float(rospy.get_param("~safe_probability_threshold", 0.55))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.3))
        # Safety & scoring parameters
        self.min_clearance_px = float(rospy.get_param("~min_clearance_px", 1.0))
        self.prob_threshold = float(rospy.get_param("~min_safe_prob", 0.55))
        self.safety_veto = bool(rospy.get_param("~safety_veto", True))
        self.weight = {
            "prob": float(rospy.get_param("~w_prob", 6.0)),
            "clearance": float(rospy.get_param("~w_clearance", 4.0)),
            "goal": float(rospy.get_param("~w_goal", 3.0)),
            "smooth": float(rospy.get_param("~w_smooth", 2.0)),
            "jerk": float(rospy.get_param("~w_jerk", 2.0)),
            "jerk_peak": float(rospy.get_param("~w_jerk_peak", 1.0)),
            "orient": float(rospy.get_param("~w_orient", 1.5)),
            "stability": float(rospy.get_param("~w_stability", 1.0)),
        }

        self.smoothness_gain = float(rospy.get_param("~smoothness_gain", 0.02))
        self.smoothness_vel_ref = float(
            rospy.get_param("~smoothness_vel_ref", self.default_speed)
        )
        self.max_yaw_rate = float(
            rospy.get_param("~max_yaw_rate", math.radians(60.0))
        )

        self.last_poly_coeffs: Optional[np.ndarray] = None
        self.last_poly_duration: float = 0.0
        self.last_target_yaw: Optional[float] = None

        self.camera_info: Optional[CameraInfo] = None
        self.odom: Optional[Odometry] = None
        self.tan_half_h: Optional[float] = None
        self.tan_half_v: Optional[float] = None
        self.image_shape: Optional[Tuple[int, int]] = None
        self.goal_world: Optional[np.ndarray] = None
        self.goal_frame: Optional[str] = None

        self.safe_center_pub = rospy.Publisher("drone/safe_center", PointStamped, queue_size=1)
        self.primitive_pub = rospy.Publisher("drone/movement_primitive", Vector3Stamped, queue_size=1)
        self.command_pub = rospy.Publisher("drone/movement_command", Vector3Stamped, queue_size=1)
        self.offset_pub = rospy.Publisher("drone/movement_offsets", Float32MultiArray, queue_size=1)
        self.fallback_pub = rospy.Publisher("drone/fallback_primitives", PoseArray, queue_size=1)
        self.trajectory_pub = rospy.Publisher("drone/safe_trajectory", Path, queue_size=1)

        rospy.Subscriber("drone/rgb/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber("drone/odometry", Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber("drone/safe_mask", Image, self.safe_mask_callback, queue_size=1)
        rospy.Subscriber("drone/safe_probability", Image, self.safe_probability_callback, queue_size=1)
        rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=1)

        primitive_steps = int(rospy.get_param("~primitive_steps", 4))
        self.primitive_steps = max(3, min(5, primitive_steps))
        self.primitive_dt = float(rospy.get_param("~primitive_dt", 0.25))
        self.primitive_smooth_window = max(
            1, int(rospy.get_param("~primitive_smooth_window", 3))
        )

        self.current_safe_mask: Optional[np.ndarray] = None
        self.current_safe_prob: Optional[np.ndarray] = None
        self.mask_header: Optional[Header] = None

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

    def safe_mask_callback(self, msg: Image) -> None:
        """接收安全掩码"""
        self.current_safe_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8") > 127
        self.mask_header = msg.header
        self._process_navigation()

    def safe_probability_callback(self, msg: Image) -> None:
        """接收安全概率图"""
        prob_uint8 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        self.current_safe_prob = prob_uint8.astype(np.float32) / 255.0

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
            rospy.logwarn_once("导航策略使用中性输出运行（未加载权重）")
        self.policy.eval()

    def _process_navigation(self) -> None:
        """处理导航逻辑"""
        if self.camera_info is None or self.odom is None:
            return
        if self.tan_half_h is None or self.tan_half_v is None:
            return
        if self.current_safe_mask is None or self.mask_header is None:
            return
        if self.current_safe_prob is None:
            self.current_safe_prob = self.current_safe_mask.astype(np.float32)
        
        start_time = time.perf_counter()
        safe_mask = self.current_safe_mask
        safe_prob = self.current_safe_prob
        height, width = safe_mask.shape
        self.image_shape = (height, width)
        self._ensure_policy()

        distance_field = cv2.distanceTransform(
            safe_mask.astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float32)

        # 查找连通区域
        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
            safe_mask.astype(np.uint8), connectivity=4
        )
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
            self._publish_fallback(self.mask_header.stamp)
            self._log_timing(start_time)
            return

        K = min(5, len(regions))
        center_row, center_col = regions[0][1], regions[0][2]
        safe_fraction = regions[0][0] / float(total_pixels)

        safe_msg = PointStamped()
        safe_msg.header = self.mask_header
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
            self._publish_fallback(self.mask_header.stamp)
            self._log_timing(start_time)
            return
        
        candidate = best_candidate
        length_scale = candidate["length_scale"]
        pitch_offset = candidate["pitch_offset"]
        yaw_offset = candidate["yaw_offset"]
        final_direction_local = candidate["final_direction_local"]
        path_points = candidate["path_points"]

        final_world_direction = candidate.get("final_direction_world")
        if final_world_direction is None:
            final_world_direction = clamp_normalized(rotation.dot(final_direction_local))
        commanded_speed = speed * length_scale

        command_vector_world = final_world_direction * commanded_speed
        if path_points.shape[0] >= 2:
            initial_segment = path_points[1] - path_points[0]
            seg_norm = float(np.linalg.norm(initial_segment))
            if seg_norm > 1e-6:
                command_vector_world = initial_segment / seg_norm * commanded_speed

        primitive_msg = Vector3Stamped()
        primitive_msg.header = self.mask_header
        primitive_msg.vector.x = base_vector_world[0]
        primitive_msg.vector.y = base_vector_world[1]
        primitive_msg.vector.z = base_vector_world[2]
        self.primitive_pub.publish(primitive_msg)

        command_msg = Vector3Stamped()
        command_msg.header = self.mask_header
        command_msg.vector.x = command_vector_world[0]
        command_msg.vector.y = command_vector_world[1]
        command_msg.vector.z = command_vector_world[2]
        self.command_pub.publish(command_msg)

        offsets_msg = Float32MultiArray()
        offsets_msg.data = [length_scale, math.degrees(pitch_offset), math.degrees(yaw_offset)]
        self.offset_pub.publish(offsets_msg)

        self._publish_trajectory(self.mask_header, path_points)
        self.last_poly_coeffs = candidate.get("poly_coeffs")
        self.last_poly_duration = float(candidate.get("poly_duration", 0.0))
        self.last_target_yaw = float(math.atan2(final_world_direction[1], final_world_direction[0]))
        self._publish_fallback(self.mask_header.stamp, include_default=False)
        self._log_timing(start_time)

    def _current_velocity(self) -> np.ndarray:
        if self.odom is None:
            return np.zeros(3, dtype=np.float32)
        return np.array(
            [
                self.odom.twist.twist.linear.x,
                self.odom.twist.twist.linear.y,
                self.odom.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )

    def _current_acceleration(self) -> np.ndarray:
        # Acceleration feedback is not available; fall back to the last solved
        # polynomial end-state if present to encourage continuity.
        if self.last_poly_coeffs is not None and self.last_poly_duration > 0.0:
            solver = Poly5Solver(self.last_poly_duration)
            _, _, acc, _ = solver.evaluate(self.last_poly_coeffs, self.last_poly_duration)
            return acc
        return np.zeros(3, dtype=np.float32)

    def _limit_yaw_offset(self, yaw_offset: float, base_direction: np.ndarray) -> float:
        dt = self.primitive_dt * max(1, self.primitive_steps)
        max_change = self.max_yaw_rate * dt
        yaw_base = math.atan2(base_direction[1], base_direction[0])
        previous_yaw = self.last_target_yaw if self.last_target_yaw is not None else yaw_base
        limited_yaw = calculate_yaw(
            previous_yaw,
            self._current_velocity(),
            rotate_direction(base_direction, yaw_offset, 0.0, 0.0),
            self.max_yaw_rate,
            dt,
        )
        limited_offset = float(np.clip(limited_yaw - yaw_base, -max_change, max_change))
        return limited_offset

    def _build_poly_trajectory(
        self,
        origin: np.ndarray,
        rotation: np.ndarray,
        final_direction_local: np.ndarray,
        commanded_speed: float,
        steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        duration = self.primitive_dt * max(1, steps)
        final_dir_world = clamp_normalized(rotation.dot(final_direction_local))
        start_vel = self._current_velocity()
        start_acc = self._current_acceleration()
        if self.last_poly_coeffs is not None and self.last_poly_duration > 0.0:
            previous_solver = Poly5Solver(self.last_poly_duration)
            _, prev_vel, prev_acc, _ = previous_solver.evaluate(
                self.last_poly_coeffs, self.last_poly_duration
            )
            inherit_gain = 0.6
            start_vel = inherit_gain * prev_vel + (1.0 - inherit_gain) * start_vel
            start_acc = inherit_gain * prev_acc + (1.0 - inherit_gain) * start_acc
        end_vel = final_dir_world * commanded_speed
        end_acc = np.zeros(3, dtype=np.float32)
        end_pos = origin + end_vel * duration * 0.5  # trapezoidal distance estimate

        solver = Poly5Solver(duration)
        coeffs = solver.solve(origin, start_vel, start_acc, end_pos, end_vel, end_acc)
        points, velocities = solver.sample(coeffs, steps)
        smoothed_points = smooth_trajectory(points, self.primitive_smooth_window)
        if smoothed_points.shape[0] > 1:
            dt = duration / float(max(smoothed_points.shape[0] - 1, 1))
            smoothed_velocities = np.gradient(smoothed_points, dt, axis=0)
        else:
            smoothed_velocities = velocities
        return (
            smoothed_points,
            smoothed_velocities,
            coeffs,
            duration,
            solver.jerk_hessian(),
            final_dir_world,
        )

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
            rospy.logwarn_throttle(1.0, "导航推理超过 2 ms: %.3f ms", elapsed_ms)

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
        yaw_offset = self._limit_yaw_offset(yaw_offset, base_direction)
        steps = max(2, self.primitive_steps)
        directions = sample_yopo_directions(
            base_direction, yaw_offset, pitch_offset, 0.0, steps
        )
        if directions:
            dir_array = smooth_trajectory(np.stack(directions, axis=0), self.primitive_smooth_window)
            directions = [clamp_normalized(vec) for vec in dir_array]
        if not directions:
            return None

        # Evaluate along YOPO directions on image grid
        min_prob = 1.0
        min_clearance = max_clearance if max_clearance > 0.0 else 0.0
        collision_count = 0
        for direction in directions:
            col, row = project_direction_to_pixel(direction, width, height, fov_deg)
            if not (0.0 <= col < width and 0.0 <= row < height):
                return None
            col_idx = int(round(col))
            row_idx = int(round(row))
            if not (0 <= col_idx < width and 0 <= row_idx < height):
                return None
            inside_safe = bool(safe_mask[row_idx, col_idx])
            if not inside_safe:
                collision_count += 1
            prob = float(safe_prob[row_idx, col_idx])
            min_prob = min(min_prob, prob)
            if max_clearance > 0.0:
                clearance = float(distance_field[row_idx, col_idx])
                min_clearance = min(min_clearance, clearance)

        collision_rate = collision_count / float(max(1, len(directions)))

        # One-vote veto on safety (probability, min clearance, collision)
        prob_threshold = max(self.safe_threshold, self.prob_threshold)
        if self.safety_veto:
            if (min_prob < prob_threshold) or (min_clearance < self.min_clearance_px) or (collision_rate > 0.0):
                return None
        else:
            # if not veto, still discourage unsafe
            if min_clearance < self.min_clearance_px:
                min_clearance = 0.0

        # Build primitive points and kinematics
        commanded_speed = speed * length_scale
        (
            points,
            velocities,
            coeffs,
            duration,
            jerk_hess,
            final_dir_world,
        ) = self._build_poly_trajectory(
            origin,
            rotation,
            directions[-1],
            commanded_speed,
            steps,
        )
        if points.shape[0] < 2:
            return None

        displacement = float(np.linalg.norm(points[-1] - points[0]))
        expected = commanded_speed * max(duration, 1e-3)
        if expected > 1e-3 and displacement < expected * 0.35:
            return None

        # Smoothness & dynamics
        smooth_metric = path_smoothness(points)
        jerk_dt = duration / max(points.shape[0] - 1, 1)
        jerk_metric = jerk_score(points, jerk_dt)
        smooth_penalty = smoothness_penalty(self.last_poly_coeffs, coeffs, jerk_hess)
        smooth_gain = math.exp(-self.smoothness_gain * smooth_penalty)
        smooth_weight = denormalize_weight(
            self.weight["smooth"], commanded_speed, self.smoothness_vel_ref
        )
        # jerk_peak: max |jerk| over segments (normalized)
        if points.shape[0] >= 4:
            v = np.diff(points, axis=0) / max(self.primitive_dt, 1e-3)
            a = np.diff(v, axis=0) / max(self.primitive_dt, 1e-3)
            j = np.diff(a, axis=0) / max(self.primitive_dt, 1e-3)
            jerk_peak = float(np.max(np.linalg.norm(j, axis=1))) if j.size else 0.0
            jerk_peak_score = math.exp(-jerk_peak)
        else:
            jerk_peak = 0.0
            jerk_peak_score = 1.0
        orientation_metric = orientation_rate_score(directions)

        pitch_limit = math.radians(15.0)
        yaw_limit = math.radians(15.0)
        stability_penalty = (
            abs(length_scale - 1.0) / 0.2
            + abs(pitch_offset) / pitch_limit
            + abs(yaw_offset) / yaw_limit
        ) / 3.0
        stability_score = math.exp(-max(0.0, stability_penalty))

        # Goal attainment score
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

        # Normalize clearance to [0,1]
        clearance_norm = 0.0
        if max_clearance > 1e-6:
            clearance_norm = min_clearance / max(max_clearance, 1e-6)

        # Weighted score (after safety veto)
        smooth_combo = 0.5 * smooth_metric + 0.5 * smooth_gain
        total_score = (
            self.weight["prob"] * min_prob
            + self.weight["clearance"] * clearance_norm
            + self.weight["goal"] * goal_score
            + smooth_weight * smooth_combo
            + self.weight["jerk"] * jerk_metric
            + self.weight["jerk_peak"] * jerk_peak_score
            + self.weight["orient"] * orientation_metric
            + self.weight["stability"] * stability_score
        )

        return {
            "score": float(total_score),
            "length_scale": float(length_scale),
            "pitch_offset": float(pitch_offset),
            "yaw_offset": float(yaw_offset),
            "final_direction_local": directions[-1],
            "final_direction_world": final_dir_world,
            "path_points": points,
            "poly_coeffs": coeffs,
            "poly_duration": float(duration),
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
            directions = sample_yopo_directions(
                base_direction, yaw_offset, pitch_offset, 0.0, steps
            )
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


def main() -> None:
    rospy.init_node("navigation_policy_inference")
    NavigationPolicyInferenceNode()
    rospy.loginfo("安全导航策略推理节点已启动")
    rospy.spin()


if __name__ == "__main__":
    main()

