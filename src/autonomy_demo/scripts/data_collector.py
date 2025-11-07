#!/usr/bin/env python3
"""Automated RGB data collector with distance-based labeling."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf_conversions import transformations
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Obstacle:
    shape: str
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray

    @property
    def radius(self) -> float:
        if self.shape == "sphere":
            return float(self.size[0]) / 2.0
        return float(max(self.size[0], self.size[1])) / 2.0


class DataCollector:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.pose: Optional[PoseStamped] = None
        self.obstacles: List[Obstacle] = []

        self.max_range = rospy.get_param("~max_range", 12.0)
        self.fov_deg = rospy.get_param("~fov_deg", 120.0)
        self.near_threshold = rospy.get_param("~near_threshold", 4.0)
        offset_raw = rospy.get_param("~camera_offset", [0.15, 0.0, 0.05])
        self.camera_offset = self._parse_offset(offset_raw)
        self.output_dir = Path(rospy.get_param("~output_dir", str(Path.home() / "autonomy_demo" / "dataset")))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_count = 0

        rospy.Subscriber("drone/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("world/obstacles", MarkerArray, self.obstacle_callback)
        rospy.Subscriber("drone/rgb/image_raw", Image, self.image_callback, queue_size=5)

        rospy.loginfo("Data collector will store samples in %s", self.output_dir)

    def pose_callback(self, msg: PoseStamped) -> None:
        self.pose = msg

    def obstacle_callback(self, msg: MarkerArray) -> None:
        parsed: List[Obstacle] = []
        for marker in msg.markers:
            center = np.array(
                [
                    float(marker.pose.position.x),
                    float(marker.pose.position.y),
                    float(marker.pose.position.z),
                ],
                dtype=np.float32,
            )
            size = np.array(
                [
                    max(float(marker.scale.x), 1e-3),
                    max(float(marker.scale.y), 1e-3),
                    max(float(marker.scale.z), 1e-3),
                ],
                dtype=np.float32,
            )
            quat = [
                float(marker.pose.orientation.x),
                float(marker.pose.orientation.y),
                float(marker.pose.orientation.z),
                float(marker.pose.orientation.w),
            ]
            if marker.type == Marker.SPHERE:
                rotation = np.identity(3, dtype=np.float32)
                shape = "sphere"
            else:
                rotation = self._quaternion_to_matrix(quat)
                shape = "box"
            parsed.append(
                Obstacle(shape=shape, center=center, size=size, rotation=rotation)
            )
        self.obstacles = parsed

    def compute_distances(self, width: int) -> np.ndarray:
        if self.pose is None:
            return np.full(width, self.max_range, dtype=np.float32)

        pose = self.pose.pose
        base_position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32
        )
        rotation = transformations.quaternion_matrix(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        )
        basis = rotation[0:3, 0:3]
        forward = self._normalize(basis[:, 0])
        right = self._normalize(basis[:, 1])
        camera_position = base_position + basis.dot(self.camera_offset)

        fov = math.radians(self.fov_deg)
        tan_half_h = math.tan(fov / 2.0)
        distances = np.full(width, self.max_range, dtype=np.float32)

        for u in range(width):
            u_ndc = 2.0 * ((u + 0.5) / float(width)) - 1.0
            direction = self._normalize(forward + u_ndc * tan_half_h * right)
            hit = self._cast_ray(camera_position, direction)
            if hit is None:
                continue
            distance, _ = hit
            if distance <= 0.0:
                continue
            distances[u] = min(distance, self.max_range)
        return distances

    def image_callback(self, msg: Image) -> None:
        if self.pose is None or not self.obstacles:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        height, width, _ = cv_image.shape
        distances = self.compute_distances(width)
        labels = (distances < self.near_threshold).astype(np.uint8)
        label_map = np.repeat(labels[np.newaxis, :], height, axis=0)

        output_path = self.output_dir / f"sample_{self.sample_count:06d}.npz"
        np.savez_compressed(
            output_path,
            image=cv_image,
            label=label_map,
            distances=distances,
            header=self._header_to_dict(msg.header),
        )
        self.sample_count += 1
        rospy.loginfo_throttle(5.0, "Captured %d samples", self.sample_count)

    @staticmethod
    def _header_to_dict(header: Header) -> dict:
        return {
            "stamp": header.stamp.to_sec(),
            "frame_id": header.frame_id,
            "seq": header.seq,
        }

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return vec
        return vec / norm

    @staticmethod
    def _quaternion_to_matrix(quat: List[float]) -> np.ndarray:
        if abs(sum(q * q for q in quat)) <= 1e-12:
            return np.identity(3, dtype=np.float32)
        matrix = transformations.quaternion_matrix(quat)
        return matrix[0:3, 0:3].astype(np.float32)

    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray) -> Optional[tuple]:
        hit_distance: Optional[float] = None
        hit_normal: Optional[np.ndarray] = None
        for obstacle in self.obstacles:
            if obstacle.shape == "sphere":
                result = self._intersect_sphere(origin, direction, obstacle)
            else:
                result = self._intersect_box(origin, direction, obstacle)
            if result is None:
                continue
            distance, normal = result
            if distance <= 0.0:
                continue
            if hit_distance is None or distance < hit_distance:
                hit_distance = distance
                hit_normal = normal
        if hit_distance is None or hit_normal is None:
            return None
        return hit_distance, hit_normal

    @staticmethod
    def _intersect_sphere(
        origin: np.ndarray, direction: np.ndarray, obstacle: Obstacle
    ) -> Optional[tuple]:
        center = obstacle.center
        radius = obstacle.radius
        oc = origin - center
        b = 2.0 * float(np.dot(direction, oc))
        c = float(np.dot(oc, oc)) - radius * radius
        discriminant = b * b - 4.0 * c
        if discriminant < 0.0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        t_hit = None
        for t in (t1, t2):
            if t_hit is None or (0.0 < t < t_hit):
                if t > 0.0:
                    t_hit = t
        if t_hit is None:
            return None
        point = origin + direction * t_hit
        normal = DataCollector._normalize(point - center)
        return t_hit, normal

    @staticmethod
    def _intersect_box(
        origin: np.ndarray, direction: np.ndarray, obstacle: Obstacle
    ) -> Optional[tuple]:
        half_extents = obstacle.size / 2.0
        rotation = obstacle.rotation
        center = obstacle.center
        local_origin = rotation.T.dot(origin - center)
        local_direction = rotation.T.dot(direction)

        tmin = -float("inf")
        tmax = float("inf")
        for i in range(3):
            if abs(local_direction[i]) < 1e-6:
                if abs(local_origin[i]) > half_extents[i]:
                    return None
                continue
            inv_dir = 1.0 / local_direction[i]
            t1 = (-half_extents[i] - local_origin[i]) * inv_dir
            t2 = (half_extents[i] - local_origin[i]) * inv_dir
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmax < tmin:
                return None

        if tmax < 0.0:
            return None
        t_hit = tmin if tmin > 0.0 else tmax
        if t_hit < 0.0:
            return None

        local_point = local_origin + local_direction * t_hit
        abs_diff = [abs(abs(local_point[i]) - half_extents[i]) for i in range(3)]
        axis = int(np.argmin(abs_diff))
        if abs_diff[axis] > 1e-4:
            axis = int(np.argmax(np.abs(local_point / half_extents)))
        normal_local = np.zeros(3, dtype=np.float32)
        normal_local[axis] = math.copysign(1.0, local_point[axis])
        normal_world = rotation.dot(normal_local)
        normal_world = DataCollector._normalize(normal_world)
        return t_hit, normal_world

    @staticmethod
    def _parse_offset(value) -> np.ndarray:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
            except (TypeError, ValueError):
                pass
        return np.array([0.15, 0.0, 0.05], dtype=np.float32)


def main() -> None:
    rospy.init_node("data_collector")
    collector = DataCollector()
    rospy.spin()


if __name__ == "__main__":
    main()
