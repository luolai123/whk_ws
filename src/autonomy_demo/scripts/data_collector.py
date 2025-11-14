#!/usr/bin/env python3
"""Automated RGB data collector with distance-based labeling."""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf_conversions import transformations
from visualization_msgs.msg import MarkerArray

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


class DataCollector:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.pose: Optional[PoseStamped] = None
        self.obstacle_field = ObstacleField()
        max_candidates_raw = rospy.get_param("~max_obstacle_candidates", 512)
        try:
            max_candidates = int(max_candidates_raw)
        except (TypeError, ValueError):
            rospy.logwarn(
                "Data collector received invalid max_obstacle_candidates; defaulting to 512"
            )
            max_candidates = 512
        self.obstacle_field.max_candidates = max(0, max_candidates)

        self.max_range = rospy.get_param("~max_range", 12.0)
        self.fov_deg = rospy.get_param("~fov_deg", 120.0)
        self.near_threshold = rospy.get_param("~near_threshold", 4.0)
        offset_raw = rospy.get_param("~camera_offset", [0.15, 0.0, 0.05])
        self.camera_offset = self._parse_offset(offset_raw)
        pitch_deg = float(rospy.get_param("~camera_pitch_deg", 10.0))
        pitch_rad = -math.radians(pitch_deg)
        self._mount_quat = transformations.quaternion_from_euler(0.0, pitch_rad, 0.0)
        self._mount_matrix = (
            transformations.quaternion_matrix(self._mount_quat)[0:3, 0:3]
        ).astype(np.float32)
        self._ray_cache: dict[Tuple[int, int], np.ndarray] = {}
        self.output_dir = Path(
            rospy.get_param("~output_dir", str(Path.home() / "autonomy_demo" / "dataset"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_count = 0

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
                        "hardware_accel requested for data collector but CUDA is unavailable; using CPU"
                    )
                else:
                    self._torch = torch
                    self._device = device
                    self._use_torch = True
                    rospy.loginfo("Data collector using torch hardware acceleration on %s", device)
            except ImportError:
                rospy.logwarn(
                    "Torch is not available; data collector will run ray casting on the CPU"
                )

        rospy.Subscriber("drone/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("world/obstacles", MarkerArray, self.obstacle_callback)
        rospy.Subscriber("drone/rgb/image_raw", Image, self.image_callback, queue_size=5)

        rospy.loginfo("Data collector will store samples in %s", self.output_dir)

    def pose_callback(self, msg: PoseStamped) -> None:
        self.pose = msg

    def obstacle_callback(self, msg: MarkerArray) -> None:
        self.obstacle_field.update_from_markers(
            msg.markers,
            use_torch=self._use_torch,
            torch_module=self._torch,
            device=self._device,
        )

    def compute_distances(self, width: int, height: int) -> np.ndarray:
        if self.pose is None:
            return np.full((height, width), self.max_range, dtype=np.float32)

        pose = self.pose.pose
        base_position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32
        )
        rotation = transformations.quaternion_matrix(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        )
        basis = rotation[0:3, 0:3].astype(np.float32)
        camera_position = base_position + basis.dot(self.camera_offset)

        local_dirs = self._ray_directions(width, height)
        body_dirs = local_dirs.dot(self._mount_matrix.T)
        directions = body_dirs.dot(basis.T)

        if self._use_torch and self.obstacle_field.supports_torch:
            torch = self._torch
            origin_t = torch.from_numpy(camera_position).to(
                device=self._device, dtype=torch.float32
            )
            directions_t = torch.from_numpy(directions).to(
                device=self._device, dtype=torch.float32
            )
            ray_result = self.obstacle_field.cast_rays_torch(
                torch, self._device, origin_t, directions_t, float(self.max_range)
            )
            distances = ray_result.distances.detach().cpu().numpy()
        else:
            ray_result = self.obstacle_field.cast_rays_cpu(
                camera_position, directions, float(self.max_range)
            )
            distances = ray_result.distances

        distances = np.where(
            np.isfinite(distances), distances.astype(np.float32), float(self.max_range)
        )
        distances = np.clip(distances, 0.0, float(self.max_range))
        return distances.reshape(height, width)

    def _ray_directions(self, width: int, height: int) -> np.ndarray:
        key = (int(width), int(height))
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached

        fov = math.radians(self.fov_deg)
        tan_half_h = math.tan(fov / 2.0)
        aspect = height / float(max(width, 1))
        tan_half_v = tan_half_h * aspect

        u = (np.arange(width, dtype=np.float32) + 0.5) / float(max(width, 1))
        v = (np.arange(height, dtype=np.float32) + 0.5) / float(max(height, 1))
        u_ndc = (u * 2.0) - 1.0
        v_ndc = 1.0 - (v * 2.0)

        x_components = u_ndc * tan_half_h
        y_components = v_ndc[:, np.newaxis] * tan_half_v

        ones = np.ones((height, width), dtype=np.float32)
        x_grid = np.broadcast_to(x_components, (height, width))
        y_grid = np.broadcast_to(y_components, (height, width))
        local_dirs = np.stack((ones, x_grid, y_grid), axis=-1)
        norms = np.linalg.norm(local_dirs, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        local_dirs = local_dirs / norms
        flattened = local_dirs.reshape(-1, 3)
        self._ray_cache[key] = flattened
        return flattened

    def image_callback(self, msg: Image) -> None:
        if self.pose is None:
            return
        if (
            self.obstacle_field.sphere_centers.size == 0
            and self.obstacle_field.box_centers.size == 0
        ):
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        height, width, _ = cv_image.shape
        distances = self.compute_distances(width, height)
        labels = (distances < self.near_threshold).astype(np.uint8)
        label_map = labels


        output_path = self.output_dir / f"sample_{self.sample_count:06d}.npz"
        pose = self.pose.pose
        pose_position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32
        )
        pose_orientation = np.array(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            dtype=np.float32,
        )
        snapshot = self.obstacle_field.snapshot()
        np.savez_compressed(
            output_path,
            image=cv_image,
            label=label_map,
            distances=distances,
            header=self._header_to_dict(msg.header),
            pose_position=pose_position,
            pose_orientation=pose_orientation,
            camera_offset=self.camera_offset.astype(np.float32),
            sphere_centers=snapshot["sphere_centers"],
            sphere_radii=snapshot["sphere_radii"],
            box_centers=snapshot["box_centers"],
            box_half_extents=snapshot["box_half_extents"],
            box_rotations=snapshot["box_rotations"],
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
