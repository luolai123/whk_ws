#!/usr/bin/env python3
"""Offline dataset generator that mirrors the simulation pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import yaml
from tf_conversions import transformations

from autonomy_demo.obstacle_field import ObstacleField

try:  # rospkg is optional when running completely offline
    import rospkg  # type: ignore
except Exception:  # pragma: no cover - best effort import
    rospkg = None  # type: ignore


@dataclass
class ObstacleSpec:
    shape: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    yaw: float = 0.0


class DatasetGenerator:
    """Procedurally builds environments and captures RGB/label pairs."""

    def __init__(self, config: dict, overwrite: bool = False, output_override: str | None = None):
        self.config = config
        dataset_cfg = config.get("dataset", {})
        world_cfg = config.get("world", {})
        camera_cfg = config.get("camera", {})

        self.seed = int(dataset_cfg.get("seed", 12345))
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

        self.env_count = int(dataset_cfg.get("env_count", 5))
        self.samples_per_env = int(dataset_cfg.get("samples_per_env", 200))
        self.safe_distance = float(dataset_cfg.get("safe_distance", 1.0))
        self.max_attempts = int(dataset_cfg.get("max_attempts", 500))
        self.near_threshold = float(dataset_cfg.get("near_threshold", 4.0))

        # 姿态采样遵循“均值 0、标准差 30°”的正态分布，保持与在线采样一致
        roll_std_deg = float(
            dataset_cfg.get(
                "roll_std_deg", dataset_cfg.get("roll_range_deg", 30.0)
            )
        )
        pitch_std_deg = float(
            dataset_cfg.get(
                "pitch_std_deg", dataset_cfg.get("pitch_range_deg", 30.0)
            )
        )
        self.roll_std = math.radians(roll_std_deg)
        self.pitch_std = math.radians(pitch_std_deg)

        self.z_range = dataset_cfg.get("z_range", [1.0, 4.0])
        if not isinstance(self.z_range, Sequence) or len(self.z_range) != 2:
            self.z_range = [1.0, 4.0]

        self.world_size = tuple(float(v) for v in world_cfg.get("size", [120.0, 120.0, 12.0]))
        self.height_range = tuple(float(v) for v in world_cfg.get("height_range", [1.2, 7.5]))
        self.size_range = tuple(float(v) for v in world_cfg.get("size_range", [1.0, 4.5]))
        self.obstacle_count = int(world_cfg.get("obstacle_count", 70))
        self.obstacle_density = float(world_cfg.get("obstacle_density", 0.0))
        self.sphere_ratio = max(0.0, min(1.0, float(world_cfg.get("sphere_ratio", 0.3))))
        self.gate_ratio = max(0.0, min(1.0 - self.sphere_ratio, float(world_cfg.get("gate_ratio", 0.2))))
        self.gate_opening_range = tuple(float(v) for v in world_cfg.get("gate_opening_range", [3.0, 6.5]))
        self.gate_height_range = tuple(float(v) for v in world_cfg.get("gate_height_range", [4.0, 8.5]))
        self.gate_post_thickness_range = tuple(
            float(v) for v in world_cfg.get("gate_post_thickness_range", [0.4, 0.9])
        )
        self.gate_depth_range = tuple(float(v) for v in world_cfg.get("gate_depth_range", [0.7, 1.4]))
        self.gate_top_thickness_range = tuple(
            float(v) for v in world_cfg.get("gate_top_thickness_range", [0.3, 0.6])
        )

        self.image_width = int(camera_cfg.get("width", 128))
        self.image_height = int(camera_cfg.get("height", 72))
        self.fov_deg = float(camera_cfg.get("fov_deg", 120.0))
        self.max_range = float(camera_cfg.get("max_range", 12.0))
        offset = camera_cfg.get("camera_offset", [0.15, 0.0, 0.05])
        if isinstance(offset, Sequence) and len(offset) >= 3:
            self.camera_offset = np.array([float(offset[0]), float(offset[1]), float(offset[2])], dtype=np.float32)
        else:
            self.camera_offset = np.array([0.15, 0.0, 0.05], dtype=np.float32)

        override = output_override
        if isinstance(override, str) and override.strip() in {"", "__from_config__"}:
            override = None
        output_dir = override or dataset_cfg.get("output_dir")
        if output_dir is None:
            output_dir = str(Path.home() / "autonomy_demo" / "dataset_auto")
        self.output_dir = Path(output_dir).expanduser()
        if self.output_dir.exists() and overwrite:
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._local_rays = self._precompute_rays(self.image_width, self.image_height, self.fov_deg)
        self._light_dir = self._normalize(np.array([-0.2, -0.4, -1.0], dtype=np.float32))
        self._ground_color = np.array([88, 120, 80], dtype=np.float32)
        self._sky_color_top = np.array([120, 170, 220], dtype=np.float32)
        self._sky_color_horizon = np.array([180, 200, 220], dtype=np.float32)
        self._obstacle_color = np.array([160, 160, 160], dtype=np.float32)

        manifest_path = self.output_dir / "manifest.json"
        manifest = {
            "seed": self.seed,
            "env_count": self.env_count,
            "samples_per_env": self.samples_per_env,
            "safe_distance": self.safe_distance,
            "near_threshold": self.near_threshold,
            "world": world_cfg,
            "camera": camera_cfg,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def run(self) -> None:
        total_samples = self.env_count * self.samples_per_env
        progress = 0
        for env_idx in range(self.env_count):
            field = self._build_environment(env_idx)
            env_dir = self.output_dir / f"env_{env_idx:03d}"
            env_dir.mkdir(parents=True, exist_ok=True)
            snapshot = field.snapshot()
            np.savez_compressed(env_dir / "world_snapshot.npz", **snapshot)
            for sample_idx in range(self.samples_per_env):
                pose = self._sample_pose(field)
                image, depth_map, label_map = self._render_sample(field, pose)
                sample_path = env_dir / f"sample_{sample_idx:05d}.npz"
                np.savez_compressed(
                    sample_path,
                    image=image,
                    label=label_map,
                    depth=depth_map,
                    distances=depth_map,
                    pose_position=pose[0].astype(np.float32),
                    pose_orientation=pose[1].astype(np.float32),
                    camera_offset=self.camera_offset.astype(np.float32),
                    near_threshold=np.float32(self.near_threshold),
                    env_index=np.int32(env_idx),
                    sample_index=np.int32(sample_idx),
                )
                progress += 1
                if progress % 10 == 0 or progress == total_samples:
                    self._print_progress(progress, total_samples)
        self._print_progress(total_samples, total_samples)
        print("\nDataset generation completed. Saved to", self.output_dir)

    def _build_environment(self, env_idx: int) -> ObstacleField:
        obstacle_total = self._compute_obstacle_total()
        half_world_x = self.world_size[0] / 2.0
        half_world_y = self.world_size[1] / 2.0
        obstacles: List[ObstacleSpec] = []
        attempts = 0
        while len(obstacles) < obstacle_total and attempts < obstacle_total * 50:
            attempts += 1
            x = self.rng.uniform(-half_world_x, half_world_x)
            y = self.rng.uniform(-half_world_y, half_world_y)
            choice = self.rng.random()
            if choice < self.sphere_ratio:
                obs = self._create_sphere(x, y)
                if not self._obstacles_within_world([obs], half_world_x, half_world_y):
                    continue
                obstacles.append(obs)
            elif choice < self.sphere_ratio + self.gate_ratio:
                gate = self._create_gate(x, y)
                if gate:
                    if not self._obstacles_within_world(gate, half_world_x, half_world_y):
                        continue
                    obstacles.extend(gate)
                    continue
                box = self._create_box(x, y)
                if not self._obstacles_within_world([box], half_world_x, half_world_y):
                    continue
                obstacles.append(box)
            else:
                box = self._create_box(x, y)
                if not self._obstacles_within_world([box], half_world_x, half_world_y):
                    continue
                obstacles.append(box)

        field = ObstacleField()
        markers = [self._obstacle_to_marker(idx, obs) for idx, obs in enumerate(obstacles)]
        field.update_from_markers(markers)
        return field

    def _sample_pose(self, field: ObstacleField) -> Tuple[np.ndarray, np.ndarray]:
        half_x = self.world_size[0] / 2.0
        half_y = self.world_size[1] / 2.0
        z_min = float(self.z_range[0])
        z_max = float(self.z_range[1])
        for _ in range(self.max_attempts):
            x = self.rng.uniform(-half_x * 0.9, half_x * 0.9)
            y = self.rng.uniform(-half_y * 0.9, half_y * 0.9)
            z = self.rng.uniform(z_min, z_max)
            point = np.array([x, y, z], dtype=np.float32)
            distance = field.distance_to_point(point)
            if distance < self.safe_distance:
                continue
            roll = self._sample_gaussian_angle(self.roll_std)
            pitch = self._sample_gaussian_angle(self.pitch_std)
            yaw = self.rng.uniform(0.0, 2.0 * math.pi)
            quat = transformations.quaternion_from_euler(roll, pitch, yaw)
            return point, np.array(quat, dtype=np.float32)
        raise RuntimeError("Failed to sample a safe pose within the maximum attempts")

    def _render_sample(
        self, field: ObstacleField, pose: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        position, quat = pose
        basis = transformations.quaternion_matrix(quat)[0:3, 0:3].astype(np.float32)
        camera_position = position + basis.dot(self.camera_offset)
        directions_world = self._local_rays.dot(basis.T)
        ray_result = field.cast_rays_cpu(camera_position, directions_world, float(self.max_range))
        distances = np.where(
            np.isfinite(ray_result.distances), ray_result.distances, float(self.max_range)
        ).astype(np.float32)
        normals = ray_result.normals
        hit_mask = ray_result.hit_mask

        pixels = self._shade_pixels(directions_world, distances, normals, hit_mask, camera_position)
        height = self.image_height
        width = self.image_width
        image = pixels.reshape(height, width, 3).astype(np.uint8)
        depth_map = distances.reshape(height, width)
        label_map = (depth_map < self.near_threshold).astype(np.uint8)
        return image, depth_map, label_map

    def _shade_pixels(
        self,
        directions: np.ndarray,
        distances: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        camera_position: np.ndarray,
    ) -> np.ndarray:
        sky_mix = np.clip((directions[:, 2] + 1.0) * 0.5, 0.0, 1.0)
        pixels = (
            self._sky_color_horizon * (1.0 - sky_mix)[:, None]
            + self._sky_color_top * sky_mix[:, None]
        )

        if np.any(hit_mask):
            lambert = np.dot(normals[hit_mask], -self._light_dir)
            lambert = np.clip(lambert, 0.0, 1.0)
            shade = 0.25 + 0.75 * lambert
            attenuation = 1.0 / (1.0 + 0.08 * np.maximum(distances[hit_mask], 0.0))
            base = self._obstacle_color * shade[:, None] * attenuation[:, None]
            pixels[hit_mask] = np.clip(base, 0.0, 255.0)

        remaining_idx = np.where(~hit_mask)[0]
        if remaining_idx.size:
            subset = directions[remaining_idx]
            dir_z = subset[:, 2]
            t_ground = np.full(subset.shape[0], np.inf, dtype=np.float32)
            ground_dirs = dir_z < -1e-4
            t_ground[ground_dirs] = camera_position[2] / -dir_z[ground_dirs]
            ground_valid = ground_dirs & (t_ground > 0.0) & (t_ground <= float(self.max_range) * 3.0)
            if np.any(ground_valid):
                points = camera_position + subset[ground_valid] * t_ground[ground_valid][:, None]
                tiling = (np.sin(points[:, 0] * 0.6) + np.sin(points[:, 1] * 0.6)) * 0.5
                base = self._ground_color * (0.7 + 0.2 * tiling)[:, None]
                pixels[remaining_idx[ground_valid]] = np.clip(base, 0.0, 255.0)

        return np.clip(pixels, 0.0, 255.0).astype(np.uint8)

    def _create_box(self, x: float, y: float) -> ObstacleSpec:
        height = self.rng.uniform(self.height_range[0], self.height_range[1])
        max_height = self.world_size[2] * 0.95
        height = min(height, max_height)
        sx = self.rng.uniform(self.size_range[0], self.size_range[1])
        sy = self.rng.uniform(self.size_range[0], self.size_range[1])
        yaw = self.rng.uniform(-math.pi, math.pi)
        return ObstacleSpec(
            shape="box",
            position=(x, y, height / 2.0),
            size=(sx, sy, height),
            yaw=yaw,
        )

    def _create_sphere(self, x: float, y: float) -> ObstacleSpec:
        diameter = self.rng.uniform(self.size_range[0], self.size_range[1])
        radius = min(diameter / 2.0, self.world_size[2] / 2.0)
        return ObstacleSpec(
            shape="sphere",
            position=(x, y, radius),
            size=(radius * 2.0, radius * 2.0, radius * 2.0),
            yaw=0.0,
        )

    def _create_gate(self, x: float, y: float) -> List[ObstacleSpec]:
        max_height = self.world_size[2] * 0.95
        gate_height = self.rng.uniform(self.gate_height_range[0], self.gate_height_range[1])
        gate_height = min(gate_height, max_height)
        opening = self.rng.uniform(self.gate_opening_range[0], self.gate_opening_range[1])
        post_thickness = self.rng.uniform(
            self.gate_post_thickness_range[0], self.gate_post_thickness_range[1]
        )
        depth = self.rng.uniform(self.gate_depth_range[0], self.gate_depth_range[1])
        top_thickness = self.rng.uniform(
            self.gate_top_thickness_range[0], self.gate_top_thickness_range[1]
        )
        top_thickness = min(top_thickness, max(gate_height * 0.3, self.gate_top_thickness_range[0]))
        beam_height = gate_height - top_thickness / 2.0
        yaw = self.rng.uniform(-math.pi, math.pi)
        half_opening = opening / 2.0
        left_pos = (x - half_opening, y)
        right_pos = (x + half_opening, y)

        posts = [
            ObstacleSpec(
                shape="box",
                position=(left_pos[0], left_pos[1], gate_height / 2.0),
                size=(post_thickness, depth, gate_height),
                yaw=yaw,
            ),
            ObstacleSpec(
                shape="box",
                position=(right_pos[0], right_pos[1], gate_height / 2.0),
                size=(post_thickness, depth, gate_height),
                yaw=yaw,
            ),
            ObstacleSpec(
                shape="box",
                position=(x, y, gate_height - top_thickness / 2.0),
                size=(opening + post_thickness, depth, top_thickness),
                yaw=yaw,
            ),
        ]
        return posts

    def _obstacles_within_world(
        self, obstacles: Sequence[ObstacleSpec], half_world_x: float, half_world_y: float
    ) -> bool:
        for obstacle in obstacles:
            if obstacle.shape == "sphere":
                radius = obstacle.size[0] / 2.0
                if (
                    obstacle.position[0] - radius < -half_world_x
                    or obstacle.position[0] + radius > half_world_x
                    or obstacle.position[1] - radius < -half_world_y
                    or obstacle.position[1] + radius > half_world_y
                ):
                    return False
            else:
                min_x, max_x, min_y, max_y = self._box_bounds(obstacle)
                if (
                    min_x < -half_world_x
                    or max_x > half_world_x
                    or min_y < -half_world_y
                    or max_y > half_world_y
                ):
                    return False
        return True

    def _box_bounds(self, obstacle: ObstacleSpec) -> Tuple[float, float, float, float]:
        corners = self._box_corners(obstacle)
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        return (min(xs), max(xs), min(ys), max(ys))

    def _box_corners(self, obstacle: ObstacleSpec) -> List[Tuple[float, float]]:
        half_x = obstacle.size[0] / 2.0
        half_y = obstacle.size[1] / 2.0
        cos_yaw = math.cos(obstacle.yaw)
        sin_yaw = math.sin(obstacle.yaw)
        local_corners = [
            (-half_x, -half_y),
            (-half_x, half_y),
            (half_x, -half_y),
            (half_x, half_y),
        ]
        world_corners: List[Tuple[float, float]] = []
        for lx, ly in local_corners:
            rx = lx * cos_yaw - ly * sin_yaw
            ry = lx * sin_yaw + ly * cos_yaw
            world_corners.append((obstacle.position[0] + rx, obstacle.position[1] + ry))
        return world_corners

    def _obstacle_to_marker(self, idx: int, obstacle: ObstacleSpec):
        class _Pose:
            def __init__(self):
                self.position = type("Point", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
                self.orientation = type("Quaternion", (), {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})()

        class _Marker:
            CUBE = 1
            SPHERE = 2

            def __init__(self):
                self.id = 0
                self.type = self.CUBE
                self.pose = _Pose()
                self.scale = type("Vector3", (), {"x": 1.0, "y": 1.0, "z": 1.0})()
                self.SPHERE = _Marker.SPHERE

        marker = _Marker()
        marker.id = idx
        marker.type = marker.SPHERE if obstacle.shape == "sphere" else marker.CUBE
        marker.pose.position.x = obstacle.position[0]
        marker.pose.position.y = obstacle.position[1]
        marker.pose.position.z = obstacle.position[2]
        marker.scale.x = obstacle.size[0]
        marker.scale.y = obstacle.size[1]
        marker.scale.z = obstacle.size[2]
        if obstacle.shape == "box":
            qx, qy, qz, qw = self._yaw_to_quaternion(obstacle.yaw)
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw
        return marker

    def _compute_obstacle_total(self) -> int:
        if self.obstacle_density > 0.0:
            area = self.world_size[0] * self.world_size[1]
            total = int(area * self.obstacle_density)
            return max(total, 1)
        return max(self.obstacle_count, 1)

    def _yaw_to_quaternion(self, yaw: float) -> Tuple[float, float, float, float]:
        half = yaw * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def _precompute_rays(self, width: int, height: int, fov_deg: float) -> np.ndarray:
        width = max(1, int(width))
        height = max(1, int(height))
        fov_h = math.radians(fov_deg)
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

    def _sample_gaussian_angle(self, std: float, limit: float = math.radians(90.0)) -> float:
        """Sample an angle with mean 0 and standard deviation ``std``.

        The result is clipped to ``limit`` (default ±90°) to keep poses
        physically plausible while still covering >99% of the target normal
        distribution described in the requirements.
        """

        std = max(0.0, float(std))
        limit = abs(float(limit))
        if std == 0.0 or limit == 0.0:
            return 0.0
        sample = self.np_rng.normal(0.0, std)
        return float(np.clip(sample, -limit, limit))

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return vec
        return vec / norm

    def _print_progress(self, current: int, total: int) -> None:
        width = 40
        progress = min(max(current / float(total), 0.0), 1.0)
        filled = int(width * progress)
        if filled >= width:
            bar = "=" * width
        else:
            bar = "=" * filled + ">" + " " * (width - filled - 1)
        print(f"\r[{bar}] {progress * 100:5.1f}%", end="")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _get_package_root(pkg: str = "autonomy_demo") -> Path | None:
    if rospkg is None:
        return None
    try:
        return Path(rospkg.RosPack().get_path(pkg))
    except Exception:
        return None


def _resolve_package_uri(uri: str) -> Path | None:
    payload = uri[len("package://") :]
    if "/" not in payload:
        return None
    pkg_name, rel = payload.split("/", 1)
    root = _get_package_root(pkg_name)
    if root is None:
        return None
    return root / rel


def resolve_config_path(raw_path: str) -> Path:
    cleaned = (raw_path or "").strip()
    if not cleaned:
        raise FileNotFoundError("Configuration path is empty")

    if cleaned.startswith("package://"):
        candidate = _resolve_package_uri(cleaned)
        if candidate and candidate.exists():
            return candidate
        raise FileNotFoundError(f"Configuration file not found: {cleaned}")

    expanded = Path(cleaned).expanduser()
    if expanded.exists():
        return expanded

    candidates: List[Path] = []

    if not expanded.is_absolute():
        candidates.append(Path.cwd() / expanded)

    stripped = cleaned.lstrip("/")
    if stripped and stripped != cleaned:
        candidates.append(Path.cwd() / stripped)

    pkg_root = _get_package_root()
    if pkg_root is not None:
        if not expanded.is_absolute():
            candidates.append(pkg_root / expanded)
        if stripped:
            candidates.append(pkg_root / stripped)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Configuration file not found: {cleaned}")


def _get_cli_args() -> List[str]:
    try:
        import rospy  # type: ignore

        return list(rospy.myargv()[1:])
    except Exception:
        return sys.argv[1:]


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "on"}:
        return True
    if value_str in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset offline")
    parser.add_argument("config", type=str, help="Path to the dataset YAML configuration")
    parser.add_argument(
        "--overwrite",
        type=_parse_bool,
        default=False,
        help="Remove any existing output directory first (true/false)",
    )
    parser.add_argument("--output", type=str, default=None, help="Override the output directory")
    args = parser.parse_args(_get_cli_args())

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    generator = DatasetGenerator(config, overwrite=args.overwrite, output_override=args.output)
    generator.run()


if __name__ == "__main__":
    main()
