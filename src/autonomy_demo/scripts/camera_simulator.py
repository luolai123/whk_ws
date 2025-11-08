#!/usr/bin/env python3
"""RGB camera simulator that renders a synthetic forward-facing scene."""

from __future__ import annotations

import ast
import math
from typing import Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import MarkerArray
import tf2_ros
from tf_conversions import transformations

try:
    from autonomy_demo.obstacle_field import ObstacleField
except ImportError:
    import os
    import sys

    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "src"),
    ]

    try:
        import rospkg  # type: ignore
    except ImportError:
        rospkg = None  # type: ignore
    if rospkg is not None:
        try:
            pkg_path = rospkg.RosPack().get_path("autonomy_demo")
        except rospkg.ResourceNotFound:
            pkg_path = None
        if pkg_path:
            candidate_paths.append(os.path.join(pkg_path, "src"))

    for path in candidate_paths:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and abs_path not in sys.path:
            sys.path.append(abs_path)
    from autonomy_demo.obstacle_field import ObstacleField


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
        self.max_obstacle_candidates = max(
            0, self._get_int_param("~max_obstacle_candidates", 512)
        )
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

        self.hardware_accel = rospy.get_param("~hardware_accel", False)
        self.hardware_device = rospy.get_param("~hardware_device", "cuda")
        self._torch = None
        self._device = None
        self._use_torch = False
        if self.hardware_accel:
            try:
                import torch

                requested_device = self.hardware_device
                try:
                    device = torch.device(requested_device)
                except (TypeError, ValueError):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device.type == "cuda" and not torch.cuda.is_available():
                    rospy.logwarn(
                        "hardware_accel requested but CUDA not available; falling back to CPU"
                    )
                else:
                    self._torch = torch
                    self._device = device
                    self._use_torch = True
                    rospy.loginfo(
                        "Camera simulator using torch hardware acceleration on %s", device
                    )
            except ImportError:
                rospy.logwarn(
                    "Torch is not available; camera simulator will use CPU ray casting"
                )

        self.bridge = CvBridge()
        self.latest_pose: Optional[PoseStamped] = None
        self.obstacle_field = ObstacleField()
        self.obstacle_field.max_candidates = self.max_obstacle_candidates
        self._local_rays = self._precompute_rays()
        self._pixel_count = self._local_rays.shape[0]
        if self._use_torch:
            self._torch_rays = self._torch.from_numpy(self._local_rays).to(
                device=self._device, dtype=self._torch.float32
            )
        else:
            self._torch_rays = None
        self._camera_offset_vec = np.array(self.camera_offset, dtype=np.float32)
        self._light_dir = self._normalize(np.array([-0.2, -0.4, -1.0], dtype=np.float32))
        self._ground_color = np.array([88, 120, 80], dtype=np.float32)
        self._sky_color_top = np.array([120, 170, 220], dtype=np.float32)
        self._sky_color_horizon = np.array([180, 200, 220], dtype=np.float32)
        self._obstacle_color = np.array([160, 160, 160], dtype=np.float32)

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
        self.obstacle_field.update_from_markers(
            msg.markers,
            use_torch=self._use_torch,
            torch_module=self._torch,
            device=self._device,
        )

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
        basis = rotation[0:3, 0:3].astype(np.float32)
        camera_position = base_position + basis.dot(self._camera_offset_vec)

        width = int(self.image_width)
        height = int(self.image_height)
        if width <= 0 or height <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        directions_world = self._local_rays.dot(basis.T)

        if self._use_torch and self.obstacle_field.supports_torch:
            torch = self._torch
            basis_t = torch.from_numpy(basis).to(device=self._device, dtype=torch.float32)
            origin_t = torch.from_numpy(camera_position).to(
                device=self._device, dtype=torch.float32
            )
            directions_t = self._torch_rays @ basis_t.t()
            ray_result = self.obstacle_field.cast_rays_torch(
                torch, self._device, origin_t, directions_t, float(self.max_range)
            )
            distances = ray_result.distances.detach().cpu().numpy()
            normals = ray_result.normals.detach().cpu().numpy()
            hit_mask = ray_result.hit_mask.detach().cpu().numpy()
        else:
            ray_result = self.obstacle_field.cast_rays_cpu(
                camera_position, directions_world, float(self.max_range)
            )
            distances = ray_result.distances
            normals = ray_result.normals
            hit_mask = ray_result.hit_mask

        directions = directions_world
        sky_mix = np.clip((directions[:, 2] + 1.0) * 0.5, 0.0, 1.0)
        pixels = (
            self._sky_color_horizon * (1.0 - sky_mix)[:, None]
            + self._sky_color_top * sky_mix[:, None]
        )
        pixels = np.clip(pixels, 0.0, 255.0).astype(np.uint8)

        if np.any(hit_mask):
            lambert = np.dot(normals[hit_mask], -self._light_dir)
            lambert = np.clip(lambert, 0.0, 1.0)
            shade = 0.25 + 0.75 * lambert
            attenuation = 1.0 / (1.0 + 0.08 * np.maximum(distances[hit_mask], 0.0))
            base = self._obstacle_color * shade[:, None] * attenuation[:, None]
            pixels[hit_mask] = np.clip(base, 0.0, 255.0).astype(np.uint8)

        remaining_idx = np.where(~hit_mask)[0]
        if remaining_idx.size:
            subset = directions[remaining_idx]
            dir_z = subset[:, 2]
            t_ground = np.full(subset.shape[0], np.inf, dtype=np.float32)
            ground_dirs = dir_z < -1e-4
            t_ground[ground_dirs] = camera_position[2] / -dir_z[ground_dirs]
            ground_valid = (
                ground_dirs
                & (t_ground > 0.0)
                & (t_ground <= float(self.max_range) * 3.0)
            )
            if np.any(ground_valid):
                points = camera_position + subset[ground_valid] * t_ground[ground_valid][:, None]
                tiling = (np.sin(points[:, 0] * 0.6) + np.sin(points[:, 1] * 0.6)) * 0.5
                base = self._ground_color * (0.7 + 0.2 * tiling)[:, None]
                pixels[remaining_idx[ground_valid]] = np.clip(base, 0.0, 255.0).astype(
                    np.uint8
                )

        return pixels.reshape(height, width, 3)

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

    def _precompute_rays(self) -> np.ndarray:
        width = max(1, int(self.image_width))
        height = max(1, int(self.image_height))
        fov_h = math.radians(self.fov_deg)
        aspect = height / float(width)
        tan_half_h = math.tan(fov_h / 2.0)
        tan_half_v = tan_half_h * aspect

        u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
        v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
        u = (u * 2.0) - 1.0
        v = 1.0 - (v * 2.0)

        x_components = u * tan_half_h
        y_components = v[:, np.newaxis] * tan_half_v

        ones = np.ones((height, width), dtype=np.float32)
        x_grid = np.broadcast_to(x_components, (height, width))
        y_grid = np.broadcast_to(y_components, (height, width))

        local_dirs = np.stack((ones, x_grid, y_grid), axis=-1)
        norms = np.linalg.norm(local_dirs, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        local_dirs /= norms
        return local_dirs.reshape(-1, 3)

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


def main() -> None:
    rospy.init_node("camera_simulator")
    camera = CameraSimulator()
    camera.publish()


if __name__ == "__main__":
    main()
