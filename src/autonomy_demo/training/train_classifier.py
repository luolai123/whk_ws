#!/usr/bin/env python3
"""Training pipelines for the distance classifier and safe navigation policy."""

import argparse
import math
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from autonomy_demo.safe_navigation import (
    PrimitiveConfig,
    apply_goal_offset,
    clamp_normalized,
    compute_primitive_duration,
    compute_direction_from_pixel,
    find_largest_safe_region,
    jerk_score,
    normalize_navigation_inputs,
    orientation_rate_score,
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


def _infer_image(sample: np.lib.npyio.NpzFile) -> np.ndarray:
    if "image" in sample:
        return sample["image"].astype(np.float32)
    # fall back to any HxWxC array (C>=3)
    for key in sample.files:
        arr = sample[key]
        if arr.ndim == 3 and max(arr.shape) >= 8:
            return arr.astype(np.float32)
    raise KeyError("No RGB image found in sample")


def _infer_distance_map(sample: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    for key in ("distances", "distance", "depth", "depth_map", "distance_map"):
        if key in sample:
            return sample[key].astype(np.float32)
    # scan for anonymous float maps (arr_X)
    for key in sample.files:
        arr = sample[key]
        if arr.ndim == 2 and arr.dtype.kind == "f" and arr.size > 16:
            return arr.astype(np.float32)
    return None


def _infer_label_map(
    sample: np.lib.npyio.NpzFile, near_threshold: float = 4.0
) -> np.ndarray:
    for key in ("label", "labels", "label_map", "safe_mask", "mask"):
        if key in sample:
            arr = sample[key]
            if arr.ndim == 2:
                return arr.astype(np.int64)

    distances = _infer_distance_map(sample)
    if distances is not None:
        threshold = near_threshold
        if "near_threshold" in sample:
            try:
                threshold = float(sample["near_threshold"])  # type: ignore[index]
            except (TypeError, ValueError):
                threshold = near_threshold
        if threshold <= 0.0:
            threshold = near_threshold
        return (distances < threshold).astype(np.int64)

    # attempt to derive from anonymous integer masks
    for key in sample.files:
        arr = sample[key]
        if arr.ndim == 2 and arr.dtype.kind in ("b", "i", "u") and arr.size > 16:
            return arr.astype(np.int64)

    raise KeyError("No label, fallback mask, or distance map found in sample")


def _find_channel_axis(array: np.ndarray) -> int:
    if array.ndim < 3:
        return array.ndim - 1
    for axis, size in enumerate(array.shape):
        if size <= 4:
            return axis
    return array.ndim - 1


def _normalize_image_array(
    image: np.ndarray, target_hw: Optional[Tuple[int, int]]
) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3:
        channel_axis = _find_channel_axis(arr)
        if channel_axis != 2:
            arr = np.moveaxis(arr, channel_axis, 2)
    else:
        raise ValueError("Unsupported image array dimensions")

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[..., :3]

    if target_hw is not None and (arr.shape[0], arr.shape[1]) != target_hw:
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32)


def _normalize_label_array(
    label: np.ndarray, target_hw: Optional[Tuple[int, int]]
) -> np.ndarray:
    arr = np.asarray(label)
    if arr.ndim == 3:
        channel_axis = _find_channel_axis(arr)
        arr = np.take(arr, indices=0, axis=channel_axis)
    elif arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError("Unsupported label dimensions")
    if target_hw is not None and arr.shape != target_hw:
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.int64)


def _normalize_distance_array(
    distances: np.ndarray, target_hw: Optional[Tuple[int, int]]
) -> np.ndarray:
    arr = np.asarray(distances)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError("Unsupported distance map dimensions")
    if target_hw is not None and arr.shape != target_hw:
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32)


def apply_offsets_torch(
    base_direction: torch.Tensor,
    yaw_offset: torch.Tensor,
    pitch_offset: torch.Tensor,
    roll_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rotate ``base_direction`` by yaw/pitch/roll offsets using Torch ops.

    The helper mirrors :func:`autonomy_demo.safe_navigation.rotate_direction`
    but stays fully in Torch so the accelerated navigation trainer can backprop
    through the projected directions without bouncing to NumPy.
    """

    if base_direction.dim() == 1:
        base = base_direction.unsqueeze(0)
        squeeze = True
    elif base_direction.dim() == 2 and base_direction.size(-1) == 3:
        base = base_direction
        squeeze = False
    else:
        raise ValueError("base_direction must have shape (3,) or (N, 3)")

    def _prepare_offset(offset: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(offset):
            offset = torch.as_tensor(offset, dtype=base.dtype, device=base.device)
        else:
            offset = offset.to(dtype=base.dtype, device=base.device)
        if offset.dim() == 0:
            offset = offset.view(1, 1)
        elif offset.dim() == 1:
            offset = offset.view(-1, 1)
        elif offset.dim() != 2 or offset.size(-1) != 1:
            raise ValueError(f"{name} must broadcast to (N, 1)")
        if offset.size(0) == 1 and base.size(0) > 1:
            offset = offset.expand(base.size(0), 1)
        elif offset.size(0) != base.size(0):
            raise ValueError(f"{name} batch dimension must match base_direction")
        return offset

    yaw = _prepare_offset(yaw_offset, "yaw_offset")
    pitch = _prepare_offset(pitch_offset, "pitch_offset")
    if roll_offset is None:
        roll = torch.zeros(1, 1, dtype=base.dtype, device=base.device)
    else:
        roll = _prepare_offset(roll_offset, "roll_offset")

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    x, y, z = base[:, 0:1], base[:, 1:2], base[:, 2:3]

    # Roll around the x-axis.
    y_roll = y * cr - z * sr
    z_roll = y * sr + z * cr

    # Pitch around the y-axis.
    x_pitch = x * cp + z_roll * sp
    z_pitch = -x * sp + z_roll * cp

    # Yaw around the z-axis.
    x_yaw = x_pitch * cy - y_roll * sy
    y_yaw = x_pitch * sy + y_roll * cy

    rotated = torch.cat([x_yaw, y_yaw, z_pitch], dim=1)
    norm = torch.linalg.norm(rotated, dim=1, keepdim=True).clamp_min(1e-6)
    rotated = rotated / norm

    if squeeze:
        return rotated.squeeze(0)
    return rotated


class ObstacleDataset(Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        indices: Optional[List[int]] = None,
        augment: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.files: List[pathlib.Path] = sorted(data_dir.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {data_dir}")
        if indices is None:
            self.indices = list(range(len(self.files)))
        else:
            self.indices = list(indices)
        self.augment = augment
        if target_hw is not None:
            self.target_hw = target_hw
        else:
            if not self.indices:
                raise ValueError(
                    "Cannot determine target size from an empty subset; provide target_hw"
                )
            ref_idx = self.indices[0]
            with np.load(self.files[ref_idx]) as sample:
                first_image = _normalize_image_array(_infer_image(sample), None)
            self.target_hw = (first_image.shape[0], first_image.shape[1])
        self.channel_mean: Optional[np.ndarray] = (
            np.asarray(mean, dtype=np.float32) if mean is not None else None
        )
        self.channel_std: Optional[np.ndarray] = (
            np.asarray(std, dtype=np.float32) if std is not None else None
        )
        if self.channel_std is not None:
            self.channel_std = np.clip(self.channel_std, 1e-4, None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[self.indices[idx]]
        with np.load(file_path) as sample:
            image = _normalize_image_array(_infer_image(sample), self.target_hw) / 255.0
            label = _normalize_label_array(_infer_label_map(sample), self.target_hw)

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            if random.random() < 0.25:
                brightness = 0.1 * (random.random() - 0.5)
                image = np.clip(image + brightness, 0.0, 1.0).astype(np.float32)
            if random.random() < 0.2:
                noise = np.random.normal(0.0, 0.02, size=image.shape).astype(np.float32)
                image = np.clip(image + noise, 0.0, 1.0).astype(np.float32)

        if self.channel_mean is not None and self.channel_std is not None:
            image = (image - self.channel_mean) / self.channel_std

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        label_tensor = torch.from_numpy(label)
        return image_tensor, label_tensor

    def estimate_class_weights(self, sample_limit: int = 256) -> torch.Tensor:
        safe_pixels = 0
        obstacle_pixels = 0
        sample_indices = self.indices
        if len(sample_indices) > sample_limit:
            sample_indices = random.sample(sample_indices, sample_limit)
        for idx in sample_indices:
            with np.load(self.files[idx]) as sample:
                label = _normalize_label_array(
                    _infer_label_map(sample), self.target_hw
                )
            safe_pixels += int(np.count_nonzero(label == 0))
            obstacle_pixels += int(np.count_nonzero(label == 1))

        total = safe_pixels + obstacle_pixels
        if total == 0:
            return torch.ones(2, dtype=torch.float32)

        freq_safe = safe_pixels / total
        freq_obstacle = obstacle_pixels / total
        weights = torch.tensor(
            [1.0 / max(freq_safe, 1e-6), 1.0 / max(freq_obstacle, 1e-6)], dtype=torch.float32
        )
        weights = weights / weights.sum() * 2.0
        return weights

    def estimate_mean_std(
        self, sample_indices: Optional[List[int]] = None, sample_limit: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = sample_indices if sample_indices is not None else self.indices
        if not indices:
            raise ValueError("Cannot estimate statistics without any samples")
        if len(indices) > sample_limit:
            indices = random.sample(indices, sample_limit)
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        for idx in indices:
            with np.load(self.files[idx]) as sample:
                image = _normalize_image_array(_infer_image(sample), self.target_hw) / 255.0
            reshaped = image.reshape(-1, 3)
            channel_sum += reshaped.sum(axis=0)
            channel_sq_sum += np.square(reshaped).sum(axis=0)
            pixel_count += reshaped.shape[0]
        mean = channel_sum / max(1, pixel_count)
        variance = channel_sq_sum / max(1, pixel_count) - np.square(mean)
        std = np.sqrt(np.clip(variance, 1e-8, None))
        return mean.astype(np.float32), std.astype(np.float32)


class NavigationDataset(Dataset):
    def __init__(self, data_dir: pathlib.Path) -> None:
        self.files: List[pathlib.Path] = sorted(data_dir.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {data_dir}")
        with np.load(self.files[0]) as first:
            label = _normalize_label_array(_infer_label_map(first), None)
        self.height, self.width = label.shape
        self.target_hw: Tuple[int, int] = (self.height, self.width)
        self._snapshot_cache: Dict[pathlib.Path, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        with np.load(self.files[idx]) as sample:
            label = _normalize_label_array(_infer_label_map(sample), self.target_hw)
            safe_mask = (label == 0).astype(np.float32)
            distances = _infer_distance_map(sample)
            if distances is None:
                raise KeyError("Navigation sample is missing a distance/depth map")
            distances = _normalize_distance_array(distances, self.target_hw)

            def _get(name: str, fallback: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if name in sample:
                    return sample[name].astype(np.float32)
                return fallback

            metadata: Dict[str, Any] = {
                "pose_position": _get("pose_position", None),
                "pose_orientation": _get("pose_orientation", None),
                "camera_offset": _get(
                    "camera_offset", np.array([0.15, 0.0, 0.05], dtype=np.float32)
                ),
                "sphere_centers": _get("sphere_centers", np.empty((0, 3), dtype=np.float32)),
                "sphere_radii": _get("sphere_radii", np.empty((0,), dtype=np.float32)),
                "box_centers": _get("box_centers", np.empty((0, 3), dtype=np.float32)),
                "box_half_extents": _get("box_half_extents", np.empty((0, 3), dtype=np.float32)),
                "box_rotations": _get("box_rotations", np.empty((0, 3, 3), dtype=np.float32)),
            }
            env_dir = self.files[idx].parent
            snapshot_path = env_dir / "world_snapshot.npz"
            if snapshot_path.exists():
                cached = self._snapshot_cache.get(snapshot_path)
                if cached is None:
                    with np.load(snapshot_path) as snapshot:
                        cached = {key: snapshot[key].astype(np.float32) for key in snapshot.files}
                    self._snapshot_cache[snapshot_path] = cached
                for key in ("sphere_centers", "sphere_radii", "box_centers", "box_half_extents", "box_rotations"):
                    if key in cached and cached[key].size:
                        metadata[key] = cached[key]
        return safe_mask, distances, metadata


def _sample_goal_pixel(region: "SafeRegion", rng: random.Random) -> Tuple[float, float]:
    min_r, max_r, min_c, max_c = region.bounds
    mask = region.mask
    if mask.size == 0:
        return region.clearance_center
    coords = np.argwhere(mask)
    if coords.size == 0:
        return region.clearance_center
    anchor_row, anchor_col = region.clearance_center
    local_centroid = np.array([
        anchor_row - min_r,
        anchor_col - min_c,
    ])
    dists = np.linalg.norm(coords - local_centroid, axis=1)
    if dists.size == 0:
        choice = coords[rng.randrange(len(coords))]
    else:
        threshold = np.quantile(dists, 0.75)
        candidates = coords[dists >= threshold]
        if candidates.size == 0:
            candidates = coords
        choice = candidates[rng.randrange(len(candidates))]
    goal_row = float(choice[0] + min_r)
    goal_col = float(choice[1] + min_c)
    return goal_row, goal_col


def _quintic_coefficients_torch(
    start_pos: torch.Tensor,
    start_vel: torch.Tensor,
    start_acc: torch.Tensor,
    end_pos: torch.Tensor,
    end_vel: torch.Tensor,
    end_acc: torch.Tensor,
    duration: torch.Tensor,
) -> torch.Tensor:
    """Torch-friendly quintic solver for端到端末端状态学习."""

    duration = torch.clamp(duration, min=torch.tensor(1e-3, device=duration.device))
    a0 = start_pos
    a1 = start_vel
    a2 = start_acc * 0.5

    t1 = duration
    t2 = t1 * t1
    t3 = t2 * t1
    t4 = t3 * t1
    t5 = t4 * t1

    rhs0 = end_pos - (a0 + a1 * t1 + a2 * t2)
    rhs1 = end_vel - (a1 + 2.0 * a2 * t1)
    rhs2 = end_acc - (2.0 * a2)

    mat = torch.stack(
        [
            torch.stack([t3, t4, t5]),
            torch.stack([3.0 * t2, 4.0 * t3, 5.0 * t4]),
            torch.stack([6.0 * t1, 12.0 * t2, 20.0 * t3]),
        ]
    )
    rhs = torch.stack([rhs0, rhs1, rhs2])
    high_coeffs = torch.linalg.solve(mat, rhs)

    coeffs = torch.zeros((6, 3), device=duration.device, dtype=duration.dtype)
    coeffs[0] = a0
    coeffs[1] = a1
    coeffs[2] = a2
    coeffs[3] = high_coeffs[0]
    coeffs[4] = high_coeffs[1]
    coeffs[5] = high_coeffs[2]
    return coeffs


def _sample_quintic_torch(
    coeffs: torch.Tensor, duration: torch.Tensor, steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample quintic points/velocities with gradients preserved."""

    steps = max(int(steps), 1)
    duration = torch.clamp(duration, min=torch.tensor(1e-3, device=duration.device))
    times = (
        torch.arange(steps + 1, device=coeffs.device, dtype=coeffs.dtype)
        / max(steps, 1)
        * duration
    )
    powers = torch.stack([times**p for p in range(6)])  # (6, steps+1)
    points = torch.einsum("pc,pt->tc", coeffs, powers)
    vel_powers = torch.stack([p * (times ** (p - 1)) if p > 0 else torch.zeros_like(times) for p in range(6)])
    velocities = torch.einsum("pc,pt->tc", coeffs, vel_powers)
    return points, velocities


def _compute_esdf_distance(points: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    """Approximate ESDF using privileged obstacle snapshots."""

    device = points.device
    distances = torch.full((points.shape[0],), float("inf"), device=device)

    sphere_centers = metadata.get("sphere_centers")
    sphere_radii = metadata.get("sphere_radii")
    if sphere_centers is not None and sphere_centers.numel() > 0:
        diff = points[:, None, :] - sphere_centers[None, :, :]
        sphere_dist = torch.linalg.norm(diff, dim=2) - sphere_radii[None, :]
        distances = torch.minimum(distances, torch.min(sphere_dist, dim=1).values)

    box_centers = metadata.get("box_centers")
    box_half_extents = metadata.get("box_half_extents")
    box_rotations = metadata.get("box_rotations")
    if (
        box_centers is not None
        and box_half_extents is not None
        and box_rotations is not None
        and box_centers.numel() > 0
    ):
        rel = points[:, None, :] - box_centers[None, :, :]
        local = torch.einsum("bij,pbj->pbi", box_rotations.transpose(1, 2), rel)
        excess = torch.abs(local) - box_half_extents[None, :, :]
        outside = torch.clamp(excess, min=0.0)
        outside_norm = torch.linalg.norm(outside, dim=2)
        inside = torch.clamp(torch.max(excess, dim=2).values, max=0.0)
        box_dist = outside_norm + inside
        distances = torch.minimum(distances, torch.min(box_dist, dim=1).values)

    return distances


def _metadata_to_torch(metadata: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move特权场景几何到 GPU/CPU，供 ESDF & 末端状态 head 反传."""

    result: Dict[str, Any] = {}
    for key in (
        "sphere_centers",
        "sphere_radii",
        "box_centers",
        "box_half_extents",
        "box_rotations",
    ):
        arr = metadata.get(key)
        if arr is None:
            continue
        tensor = torch.as_tensor(arr, device=device, dtype=torch.float32)
        result[key] = tensor
    return result


def _compute_duration_torch(goal_body: torch.Tensor, config: PrimitiveConfig, duration_scale: torch.Tensor) -> torch.Tensor:
    """Differentiable版本的路径时长 T = 2R / (v_max * α)。"""

    radius = torch.linalg.norm(goal_body)
    effective_speed = torch.clamp(
        torch.tensor(config.vel_max_train, device=goal_body.device, dtype=goal_body.dtype),
        min=1e-3,
    )
    duration = (2.0 * radius) / effective_speed
    duration = duration * duration_scale
    return torch.clamp(duration, min=0.2)


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
        fusion_dim = 128
        self.fc1 = torch.nn.Linear(64 + state_dim, fusion_dim)
        self.action_head = torch.nn.Linear(fusion_dim, 4)
        # 末端状态 head：Δp(3) + v_T(3) + a_T(3)
        self.end_state_head = torch.nn.Linear(fusion_dim, 9)

    def forward(
        self, mask: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(mask)
        pooled = self.global_pool(features).view(mask.size(0), -1)
        combined = torch.cat([pooled, state], dim=1)
        hidden = F.relu(self.fc1(combined))
        action = torch.tanh(self.action_head(hidden))
        end_state = torch.tanh(self.end_state_head(hidden))
        return action, end_state


class SegmentationLoss(torch.nn.Module):
    def __init__(
        self, class_weights: Optional[torch.Tensor] = None, dice_weight: float = 1.0
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.view(-1))
        else:
            self.class_weights = None  # type: ignore[assignment]
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.class_weights)
        probs = torch.softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=probs.shape[1]).permute(0, 3, 1, 2)
        labels_one_hot = labels_one_hot.to(dtype=probs.dtype)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * labels_one_hot, dim=dims)
        cardinality = torch.sum(probs + labels_one_hot, dim=dims)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return ce + self.dice_weight * dice.mean()


def add_noise(mask: np.ndarray, noise_rate: float) -> np.ndarray:
    if noise_rate <= 0.0:
        return mask
    noise = np.random.rand(*mask.shape) < noise_rate
    noisy = mask.copy()
    noisy[noise] = 1.0 - noisy[noise]
    return noisy


def evaluate_classifier(
    model: DistanceClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    if len(loader.dataset) == 0:
        return None

    model.eval()
    safe_tp = 0
    safe_fp = 0
    safe_fn = 0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            safe_pred = preds == 0
            safe_true = labels == 0
            safe_tp += int(torch.logical_and(safe_pred, safe_true).sum().item())
            safe_fp += int(torch.logical_and(safe_pred, ~safe_true).sum().item())
            safe_fn += int(torch.logical_and(~safe_pred, safe_true).sum().item())
            total_correct += int((preds == labels).sum().item())
            total_pixels += labels.numel()

    union = safe_tp + safe_fp + safe_fn
    iou = safe_tp / union if union > 0 else 0.0
    precision = safe_tp / (safe_tp + safe_fp) if (safe_tp + safe_fp) > 0 else 0.0
    recall = safe_tp / (safe_tp + safe_fn) if (safe_tp + safe_fn) > 0 else 0.0
    accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def train(
    model: DistanceClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[List[float], List[float]]:
    class_weights: Optional[torch.Tensor] = None
    if isinstance(train_loader.dataset, ObstacleDataset):
        class_weights = train_loader.dataset.estimate_class_weights()
    elif isinstance(train_loader.dataset, Dataset) and hasattr(train_loader.dataset, "dataset"):
        base = getattr(train_loader.dataset, "dataset")
        if isinstance(base, ObstacleDataset):
            class_weights = base.estimate_class_weights()

    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = SegmentationLoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val += loss.item()
        avg_val_loss = running_val / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        metrics = evaluate_classifier(model, val_loader, device)
        if metrics is None:
            print(
                f"Epoch {epoch + 1}/{epochs} - train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}"
            )
        else:
            print(
                "Epoch {}/{} - train loss: {:.4f}, val loss: {:.4f}, IoU: {:.3f}, Acc: {:.3f}, Prec: {:.3f}, Rec: {:.3f}".format(
                    epoch + 1,
                    epochs,
                    avg_train_loss,
                    avg_val_loss,
                    metrics["iou"],
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                )
            )
    return train_losses, val_losses


def train_navigation_policy(
    dataset: NavigationDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_rate: float,
    policy_output: pathlib.Path,
    primitive_config: PrimitiveConfig,
    samples_per_second: int,
    camera_pitch_deg: float,
    seed: int,
) -> None:
    goal_feature_dim = 4
    state_dim = primitive_state_dim(primitive_config) + goal_feature_dim
    policy = SafeNavigationPolicy(dataset.height, dataset.width, state_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    diag = math.sqrt(dataset.width ** 2 + dataset.height ** 2)
    camera_pitch = -math.radians(camera_pitch_deg)
    camera_to_body = np.array(
        [
            [math.cos(camera_pitch), 0.0, math.sin(camera_pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(camera_pitch), 0.0, math.cos(camera_pitch)],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(seed or None)
    indices = list(range(len(dataset)))

    for epoch in range(epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_count = 0
        policy.train()
        metrics_accumulator = {
            "safety": 0.0,
            "clearance": 0.0,
            "goal": 0.0,
            "goal_alignment": 0.0,
            "smoothness": 0.0,
            "jerk": 0.0,
            "orientation": 0.0,
        }

        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_samples = 0

            for idx in batch:
                safe_mask, distances, _metadata = dataset[idx]
                noisy_mask = add_noise(safe_mask, noise_rate)
                region = find_largest_safe_region(noisy_mask.astype(bool), 0.05)
                if region is None:
                    continue

                center_row, center_col = region.centroid
                goal_row, goal_col = _sample_goal_pixel(region, random)
                mask_tensor = torch.from_numpy(noisy_mask).to(device=device, dtype=torch.float32)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

                goal_direction_camera = compute_direction_from_pixel(
                    goal_col, goal_row, dataset.width, dataset.height, 120.0
                ).astype(np.float32)
                base_direction_camera = compute_direction_from_pixel(
                    center_col, center_row, dataset.width, dataset.height, 120.0
                ).astype(np.float32)
                sample = sample_motion_primitives(
                    base_direction_camera,
                    camera_to_body,
                    rng,
                    primitive_config,
                    1,
                )[0]
                state_vec = primitive_state_vector(sample, primitive_config)
                goal_direction_body = clamp_normalized(
                    camera_to_body.dot(goal_direction_camera)
                )
                goal_bias = math.sqrt(
                    (goal_col - center_col) ** 2 + (goal_row - center_row) ** 2
                ) / max(diag, 1e-3)
                goal_features = np.concatenate(
                    [goal_direction_body, np.array([goal_bias], dtype=np.float32)]
                )
                enriched_state = np.concatenate([state_vec, goal_features]).astype(np.float32)
                state_tensor = torch.from_numpy(enriched_state).unsqueeze(0).to(device=device)

                action_out, end_state = policy(mask_tensor, state_tensor)
                offset_raw = action_out[0, 0:3]
                duration_delta = action_out[0, 3]
                duration_scale = torch.clamp(1.0 + 0.2 * duration_delta, 0.7, 1.3)

                base_goal_body = (
                    torch.from_numpy(sample.goal_direction_body * sample.goal_length)
                    .to(device=device, dtype=torch.float32)
                )
                goal_body = base_goal_body + offset_raw * primitive_config.radio_range

                delta_p = end_state[0, 0:3] * (primitive_config.radio_range * 0.5)
                end_vel = end_state[0, 3:6] * primitive_config.vel_max_train
                end_acc = end_state[0, 6:9] * primitive_config.acc_max_train
                end_pos = goal_body + delta_p

                duration_t = _compute_duration_torch(goal_body, primitive_config, duration_scale)
                sample_count = max(2, int(math.ceil(float(duration_t.detach().cpu()) * samples_per_second)))

                start_vel = torch.from_numpy(sample.start_vel_body).to(device=device, dtype=torch.float32)
                start_acc = torch.from_numpy(sample.start_acc_body).to(device=device, dtype=torch.float32)

                coeffs = _quintic_coefficients_torch(
                    torch.zeros(3, device=device, dtype=torch.float32),
                    start_vel,
                    start_acc,
                    end_pos,
                    end_vel,
                    end_acc,
                    duration_t,
                )
                points_t, velocities_t = _sample_quintic_torch(
                    coeffs, duration_t, sample_count
                )
                if points_t.numel() == 0:
                    continue

                normalized_goal_vec = goal_body / torch.clamp(
                    torch.linalg.norm(goal_body), min=torch.tensor(1e-6, device=device)
                )
                normalized_endpoint = points_t[-1] / torch.clamp(
                    torch.linalg.norm(points_t[-1]), min=torch.tensor(1e-6, device=device)
                )
                normalized_goal_error = torch.norm(normalized_endpoint - normalized_goal_vec)

                dt = duration_t / torch.tensor(
                    max(sample_count, 1), device=device, dtype=torch.float32
                )
                vel = torch.diff(points_t, dim=0) / torch.clamp(dt, min=1e-3)
                acc = torch.diff(vel, dim=0) / torch.clamp(dt, min=1e-3)
                jerk = torch.diff(acc, dim=0) / torch.clamp(dt, min=1e-3)
                Js = torch.sum(jerk ** 2) * dt

                metadata_torch = _metadata_to_torch(_metadata, device)
                esdf = _compute_esdf_distance(points_t, metadata_torch)
                clearance_penalty = F.relu(0.2 - esdf)
                Jc = torch.mean(clearance_penalty**2)

                Jg = torch.mean((points_t[-1] - goal_body) ** 2)

                loss = 0.6 * Jc + 0.25 * Js + 0.15 * Jg
                batch_loss += loss
                valid_samples += 1

                metrics_accumulator["safety"] += float(torch.mean((esdf > 0.0).float()).item())
                metrics_accumulator["clearance"] += float(torch.mean(esdf).item())
                metrics_accumulator["goal"] += float(Jg.detach().cpu())
                metrics_accumulator["goal_alignment"] += float(
                    normalized_goal_error.detach().cpu()
                )
                metrics_accumulator["smoothness"] += float(Js.detach().cpu())
                metrics_accumulator["jerk"] += float(torch.max(torch.linalg.norm(jerk, dim=1)).item())
                metrics_accumulator["orientation"] += float(
                    torch.linalg.norm(end_vel).detach().cpu()
                )

                epoch_count += 1

            if valid_samples == 0:
                continue
            batch_loss = batch_loss / valid_samples
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * valid_samples

        if epoch_count:
            avg_loss = epoch_loss / epoch_count
            averaged_metrics = {
                key: value / epoch_count for key, value in metrics_accumulator.items()
            }
            print(
                "Policy epoch {}/{} - avg loss: {:.4f}, clearance_hit: {:.3f}, clearance_mean: {:.3f}, Jg: {:.4f}, goal_norm_err: {:.4f}, Js: {:.4f}, jerk_peak: {:.4f}, |v_T|: {:.3f}".format(
                    epoch + 1,
                    epochs,
                    avg_loss,
                    averaged_metrics["safety"],
                    averaged_metrics["clearance"],
                    averaged_metrics["goal"],
                    averaged_metrics["goal_alignment"],
                    averaged_metrics["smoothness"],
                    averaged_metrics["jerk"],
                    averaged_metrics["orientation"],
                )
            )
        else:
            print(f"Policy epoch {epoch + 1}/{epochs} - avg loss: nan")

    policy_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), policy_output)
    print(f"Saved navigation policy to {policy_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the distance classifier and navigation policy")
    parser.add_argument(
        "dataset",
        type=pathlib.Path,
        help="Directory tree containing *.npz samples (searched recursively)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path.home() / "autonomy_demo" / "model.pt")
    parser.add_argument("--no_policy", action="store_true", help="Skip navigation policy training")
    parser.add_argument("--policy_epochs", type=int, default=40)
    parser.add_argument("--policy_batch", type=int, default=8)
    parser.add_argument("--policy_lr", type=float, default=5e-4)
    parser.add_argument("--policy_noise", type=float, default=0.03)
    parser.add_argument(
        "--policy_output",
        type=pathlib.Path,
        default=pathlib.Path.home() / "autonomy_demo" / "navigation_policy.pt",
    )
    parser.add_argument("--camera_pitch_deg", type=float, default=10.0)
    parser.add_argument("--path_samples_per_step", type=int, default=12)
    parser.add_argument("--radio_range", type=float, default=5.0)
    parser.add_argument("--vel_max_train", type=float, default=6.0)
    parser.add_argument("--acc_max_train", type=float, default=3.0)
    parser.add_argument("--v_forward_mean", type=float, default=2.0)
    parser.add_argument("--v_forward_sigma", type=float, default=0.45)
    parser.add_argument("--v_std_unit", type=float, default=0.22)
    parser.add_argument("--a_std_unit", type=float, default=0.35)
    parser.add_argument("--goal_length_scale", type=float, default=1.0)
    parser.add_argument("--offset_gain", type=float, default=0.25)
    parser.add_argument("--yaw_range_deg", type=float, default=360.0)
    parser.add_argument("--pitch_std_deg", type=float, default=30.0)
    parser.add_argument("--roll_std_deg", type=float, default=30.0)
    parser.add_argument("--policy_seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dataset = ObstacleDataset(args.dataset)
    total_len = len(base_dataset)
    min_samples = 10_000
    if total_len < min_samples:
        raise ValueError(
            f"Dataset only contains {total_len} samples; collect at least {min_samples} before training"
        )
    all_indices = list(range(total_len))
    random.shuffle(all_indices)

    if total_len < 2:
        train_indices = all_indices
        val_indices: List[int] = []
    else:
        val_len = max(1, int(total_len * args.val_split))
        train_len = total_len - val_len
        if train_len <= 0:
            train_len = 1
            val_len = total_len - train_len
        val_indices = all_indices[:val_len]
        train_indices = all_indices[val_len:]

    channel_mean, channel_std = base_dataset.estimate_mean_std(train_indices)
    target_hw = base_dataset.target_hw

    train_dataset = ObstacleDataset(
        args.dataset,
        indices=train_indices,
        augment=True,
        mean=channel_mean,
        std=channel_std,
        target_hw=target_hw,
    )
    val_dataset = ObstacleDataset(
        args.dataset,
        indices=val_indices,
        augment=False,
        mean=channel_mean,
        std=channel_std,
        target_hw=target_hw,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch)

    model = DistanceClassifier().to(device)
    train(model, train_loader, val_loader, device, args.epochs, args.lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "normalization": {
            "mean": channel_mean.tolist(),
            "std": channel_std.tolist(),
        },
        "input_size": [int(target_hw[0]), int(target_hw[1])],
    }
    torch.save(checkpoint, args.output)
    print(f"Saved trained model to {args.output}")

    if not args.no_policy:
        nav_dataset = NavigationDataset(args.dataset)
        primitive_config = PrimitiveConfig(
            radio_range=args.radio_range,
            vel_max_train=args.vel_max_train,
            acc_max_train=args.acc_max_train,
            forward_log_mean=math.log(max(0.2, args.v_forward_mean)),
            forward_log_sigma=max(0.05, args.v_forward_sigma),
            v_std_unit=max(0.05, args.v_std_unit),
            a_std_unit=max(0.05, args.a_std_unit),
            yaw_range_deg=args.yaw_range_deg,
            pitch_std_deg=args.pitch_std_deg,
            roll_std_deg=args.roll_std_deg,
            goal_length_scale=max(0.2, args.goal_length_scale),
            offset_gain=max(0.05, args.offset_gain),
        )
        samples_per_second = max(1, args.path_samples_per_step)
        train_navigation_policy(
            nav_dataset,
            device,
            args.policy_epochs,
            args.policy_batch,
            args.policy_lr,
            args.policy_noise,
            args.policy_output,
            primitive_config,
            samples_per_second,
            args.camera_pitch_deg,
            args.policy_seed,
        )


if __name__ == "__main__":
    main()
