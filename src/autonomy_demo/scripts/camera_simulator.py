#!/usr/bin/env python3
"""RGB camera simulator that renders a synthetic forward-facing scene."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf_conversions import transformations


@dataclass
class Obstacle:
    """Simple container describing a rendered obstacle."""

    shape: str
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray

    @property
    def radius(self) -> float:
        if self.shape == "sphere":
            return float(self.size[0]) / 2.0
        return float(max(self.size[0], self.size[1])) / 2.0


class CameraSimulator:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.image_width = self._get_int_param("~width", 128)
        self.image_height = self._get_int_param("~height", 72)
        self.fov_deg = self._get_float_param("~fov_deg", 120.0)
        self.max_range = self._get_float_param("~max_range", 12.0)
        self.publish_rate = self._get_float_param("~rate", 10.0)
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.camera_frame = rospy.get_param("~camera_frame", "rgb_optical")
        offset_raw = rospy.get_param("~camera_offset", [0.15, 0.0, 0.05])
        offset_parsed = self._maybe_parse_literal(offset_raw, "~camera_offset")
        if isinstance(offset_parsed, (list, tuple)) and len(offset_parsed) >= 3:
            try:
                self.camera_offset = [float(offset_parsed[0]), float(offset_parsed[1]), float(offset_parsed[2])]
            except (TypeError, ValueError):
                rospy.logwarn("Camera offset parameter malformed, using default offset")
                self.camera_offset = [0.15, 0.0, 0.05]
        else:
            self.camera_offset = [0.15, 0.0, 0.05]

        self.bridge = CvBridge()
        self.latest_pose: Optional[PoseStamped] = None
        self.obstacles: List[Obstacle] = []

        self.image_pub = rospy.Publisher("drone/rgb/image_raw", Image, queue_size=1)
        self.info_pub = rospy.Publisher("drone/rgb/camera_info", CameraInfo, queue_size=1)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.publish_static_transform()

        rospy.Subscriber("drone/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("world/obstacles", MarkerArray, self.obstacle_callback)

        rospy.loginfo("Camera simulator ready")

    def pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    def obstacle_callback(self, msg: MarkerArray) -> None:
        obstacles: List[Obstacle] = []
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
            obstacles.append(Obstacle(shape=shape, center=center, size=size, rotation=rotation))
        self.obstacles = obstacles

    def render_image(self) -> Optional[np.ndarray]:
        if self.latest_pose is None:
            return None

        pose = self.latest_pose.pose
        base_position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32
        )
        rotation = transformations.quaternion_matrix(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        )
        basis = rotation[0:3, 0:3]
        forward = self._normalize(basis[:, 0])
        right = self._normalize(basis[:, 1])
        up = self._normalize(basis[:, 2])

        camera_offset = np.array(self.camera_offset, dtype=np.float32)
        camera_position = base_position + basis.dot(camera_offset)

        width = self.image_width
        height = self.image_height
        image = np.zeros((height, width, 3), dtype=np.uint8)
        if width <= 0 or height <= 0:
            return image

        fov_h = math.radians(self.fov_deg)
        aspect = height / float(width)
        fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * aspect)
        tan_half_h = math.tan(fov_h / 2.0)
        tan_half_v = math.tan(fov_v / 2.0)

        light_dir = self._normalize(np.array([-0.2, -0.4, -1.0], dtype=np.float32))
        ground_color = np.array([88, 120, 80], dtype=np.float32)
        sky_color_top = np.array([120, 170, 220], dtype=np.float32)
        sky_color_horizon = np.array([180, 200, 220], dtype=np.float32)
        obstacle_color = np.array([160, 160, 160], dtype=np.float32)

        for v in range(height):
            v_ndc = 1.0 - 2.0 * ((v + 0.5) / float(height))
            y_component = v_ndc * tan_half_v
            for u in range(width):
                u_ndc = 2.0 * ((u + 0.5) / float(width)) - 1.0
                x_component = u_ndc * tan_half_h
                direction = self._normalize(
                    forward + x_component * right + y_component * up
                )
                hit = self._cast_ray(camera_position, direction)
                if hit is not None:
                    distance, normal = hit
                    lambert = max(0.0, float(np.dot(normal, -light_dir)))
                    shade = 0.25 + 0.75 * lambert
                    attenuation = 1.0 / (1.0 + 0.08 * max(distance, 0.0))
                    base = obstacle_color * shade * attenuation
                    color = np.clip(base, 0.0, 255.0)
                    image[v, u, :] = color.astype(np.uint8)
                    continue

                # Check ground plane intersection when looking downward
                if direction[2] < -1e-4:
                    t_ground = (camera_position[2]) / -direction[2]
                    if 0.0 < t_ground <= self.max_range * 3.0:
                        point = camera_position + direction * t_ground
                        tiling = (math.sin(point[0] * 0.6) + math.sin(point[1] * 0.6)) * 0.5
                        base = ground_color * (0.7 + 0.2 * tiling)
                        image[v, u, :] = np.clip(base, 0.0, 255.0).astype(np.uint8)
                        continue

                # Otherwise render sky gradient
                t = max(0.0, min(1.0, (direction[2] + 1.0) * 0.5))
                sky = sky_color_horizon * (1.0 - t) + sky_color_top * t
                image[v, u, :] = np.clip(sky, 0.0, 255.0).astype(np.uint8)

        return image

    def publish(self) -> None:
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            image = self.render_image()
            if image is not None:
                msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = self.camera_frame
                info = CameraInfo()
                info.header = msg.header
                info.height = self.image_height
                info.width = self.image_width
                fx = self.image_width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
                fy = fx
                cx = self.image_width / 2.0
                cy = self.image_height / 2.0
                info.distortion_model = "plumb_bob"
                info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
                info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
                info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
                self.image_pub.publish(msg)
                self.info_pub.publish(info)
            rate.sleep()

    def publish_static_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.base_frame
        transform.child_frame_id = self.camera_frame
        transform.transform.translation.x = float(self.camera_offset[0]) if len(self.camera_offset) > 0 else 0.15
        transform.transform.translation.y = float(self.camera_offset[1]) if len(self.camera_offset) > 1 else 0.0
        transform.transform.translation.z = float(self.camera_offset[2]) if len(self.camera_offset) > 2 else 0.05
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(transform)

    @staticmethod
    def _maybe_parse_literal(value, name: str = ""):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                if name:
                    rospy.logwarn("Failed to parse parameter %s as literal, using raw string", name)
                else:
                    rospy.logwarn("Failed to parse parameter value '%s' as literal", value)
        return value

    @classmethod
    def _get_float_param(cls, name: str, default: float) -> float:
        raw = rospy.get_param(name, default)
        parsed = cls._maybe_parse_literal(raw, name)
        try:
            return float(parsed)
        except (TypeError, ValueError):
            rospy.logwarn("Parameter %s could not be parsed as float, using default", name)
            return float(default)

    @classmethod
    def _get_int_param(cls, name: str, default: int) -> int:
        raw = rospy.get_param(name, default)
        parsed = cls._maybe_parse_literal(raw, name)
        try:
            return int(parsed)
        except (TypeError, ValueError):
            rospy.logwarn("Parameter %s could not be parsed as int, using default", name)
            return int(default)

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
            result: Optional[tuple]
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
        normal = CameraSimulator._normalize(point - center)
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
        normal_local = np.zeros(3, dtype=np.float32)
        abs_diff = [abs(abs(local_point[i]) - half_extents[i]) for i in range(3)]
        axis = int(np.argmin(abs_diff))
        if abs_diff[axis] > 1e-4:
            axis = int(np.argmax(np.abs(local_point / half_extents)))
        normal_local[axis] = math.copysign(1.0, local_point[axis])

        normal_world = rotation.dot(normal_local)
        normal_world = CameraSimulator._normalize(normal_world)
        return t_hit, normal_world


def main() -> None:
    rospy.init_node("camera_simulator")
    camera = CameraSimulator()
    camera.publish()


if __name__ == "__main__":
    main()
