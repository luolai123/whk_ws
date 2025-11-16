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
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset

from autonomy_demo.safe_navigation import (
    PrimitiveConfig,
    apply_goal_offset,
    clamp_normalized,
    compute_direction_from_pixel,
    find_largest_safe_region,
    jerk_score,
    orientation_rate_score,
    primitive_quintic_trajectory,
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
        self.files: List[pathlib.Path] = sorted(data_dir.glob("*.npz"))
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
        self.files: List[pathlib.Path] = sorted(data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {data_dir}")
        with np.load(self.files[0]) as first:
            label = _normalize_label_array(_infer_label_map(first), None)
        self.height, self.width = label.shape
        self.target_hw: Tuple[int, int] = (self.height, self.width)

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
            }
        return safe_mask, distances, metadata


def _sample_goal_pixel(region: "SafeRegion", rng: random.Random) -> Tuple[float, float]:
    min_r, max_r, min_c, max_c = region.bounds
    mask = region.mask
    if mask.size == 0:
        return region.centroid
    coords = np.argwhere(mask)
    if coords.size == 0:
        return region.centroid
    local_centroid = np.array([
        region.centroid[0] - min_r,
        region.centroid[1] - min_c,
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


class SafeNavigationPolicy(torch.nn.Module):
    def __init__(self, height: int, width: int, state_dim: int) -> None:
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
        hidden = F.relu(self.fc1(combined))
        output = torch.tanh(self.fc2(hidden))
        return output


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
    primitive_dt: float,
    primitive_steps: int,
    samples_per_step: int,
    camera_pitch_deg: float,
    seed: int,
) -> None:
    policy = SafeNavigationPolicy(dataset.height, dataset.width, state_dim=8).to(device)
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
    body_to_camera = camera_to_body.T
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

                clearance_map = cv2.distanceTransform(
                    (noisy_mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5
                ).astype(np.float32)
                max_clearance = float(np.max(clearance_map))
                if max_clearance > 1e-6:
                    clearance_norm = clearance_map / max_clearance
                else:
                    clearance_norm = clearance_map

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
                state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device=device)

                outputs = policy(mask_tensor, state_tensor)
                # 使用带噪策略输出，允许对不可微奖励应用 REINFORCE（log_prob）梯度
                action_std = torch.full_like(outputs[0], 0.1)
                action_dist = Normal(outputs[0], action_std)
                sampled_action = action_dist.rsample()
                log_prob = action_dist.log_prob(sampled_action).sum()

                offset_raw = sampled_action[0:3]
                duration_delta = sampled_action[3]
                duration_scale = torch.clamp(
                    1.0 + 0.2 * duration_delta,
                    0.7,
                    1.3,
                )
                offset_vec = offset_raw.detach().cpu().numpy() * primitive_config.radio_range
                goal_body = apply_goal_offset(sample, offset_vec, primitive_config)
                sample_count = max(primitive_steps * samples_per_step, primitive_steps)
                points_np, velocities_np, duration_val = primitive_quintic_trajectory(
                    sample,
                    goal_body,
                    float(duration_scale.detach().cpu()),
                    sample_count,
                )
                if points_np.size == 0:
                    continue

                safety_hits: List[float] = []
                clearance_values: List[float] = []
                smoothness_sum = 0.0
                prev_dir: Optional[np.ndarray] = None
                last_col = center_col
                last_row = center_row
                for point in points_np[1:]:
                    direction_body = clamp_normalized(point)
                    direction_camera = body_to_camera.dot(direction_body)
                    col_pix, row_pix = project_direction_to_pixel(
                        direction_camera,
                        dataset.width,
                        dataset.height,
                        120.0,
                    )
                    last_col, last_row = col_pix, row_pix
                    col_i = int(round(col_pix))
                    row_i = int(round(row_pix))
                    if 0 <= row_i < dataset.height and 0 <= col_i < dataset.width:
                        safety_hits.append(float(noisy_mask[row_i, col_i]))
                        clearance_values.append(float(clearance_norm[row_i, col_i]))
                    else:
                        safety_hits.append(0.0)
                        clearance_values.append(0.0)
                    if prev_dir is not None:
                        smoothness_sum += max(
                            -1.0, min(1.0, float(np.dot(prev_dir, direction_body)))
                        )
                    prev_dir = direction_body

                if not safety_hits:
                    continue

                safety_ratio = sum(safety_hits) / len(safety_hits)
                clearance_min = max(0.0, min(clearance_values))
                if safety_ratio < 0.45 or clearance_min < 0.05:
                    continue

                smoothness_score = max(
                    0.0,
                    min(1.0, (smoothness_sum / max(len(safety_hits) - 1, 1) + 1.0) * 0.5),
                )
                goal_error = math.sqrt((last_col - goal_col) ** 2 + (last_row - goal_row) ** 2)
                goal_direction_camera = compute_direction_from_pixel(
                    goal_col,
                    goal_row,
                    dataset.width,
                    dataset.height,
                    120.0,
                )
                final_direction = clamp_normalized(points_np[-1])
                final_cam = body_to_camera.dot(final_direction)
                goal_alignment = float(
                    max(-1.0, min(1.0, np.dot(clamp_normalized(final_cam), goal_direction_camera)))
                )
                jerk_metric_val = jerk_score(points_np, primitive_dt / max(samples_per_step, 1))
                orientation_metric_val = orientation_rate_score(velocities_np)

                safety_score = torch.tensor(safety_ratio, device=device, dtype=torch.float32)
                clearance_score = torch.tensor(
                    min(1.0, clearance_min), device=device, dtype=torch.float32
                )
                goal_score = torch.exp(
                    -torch.tensor(goal_error / diag, device=device, dtype=torch.float32)
                )
                goal_alignment_score = torch.tensor(
                    (goal_alignment + 1.0) * 0.5, device=device, dtype=torch.float32
                )
                smoothness_score_t = torch.tensor(
                    smoothness_score, device=device, dtype=torch.float32
                )
                jerk_score_t = torch.tensor(
                    max(0.0, min(1.0, jerk_metric_val)), device=device, dtype=torch.float32
                )
                orientation_score_t = torch.tensor(
                    max(0.0, min(1.0, orientation_metric_val)),
                    device=device,
                    dtype=torch.float32,
                )

                reward = (
                    0.45 * safety_score
                    + 0.2 * clearance_score
                    + 0.15 * goal_score
                    + 0.08 * goal_alignment_score
                    + 0.07 * smoothness_score_t
                    + 0.03 * jerk_score_t
                    + 0.02 * orientation_score_t
                )
                reward_tensor = reward.detach()
                loss = -reward_tensor * log_prob
                batch_loss += loss
                valid_samples += 1

                metrics_accumulator["safety"] += safety_ratio
                metrics_accumulator["clearance"] += clearance_min
                metrics_accumulator["goal"] += float(goal_score.detach().cpu())
                metrics_accumulator["goal_alignment"] += (goal_alignment + 1.0) * 0.5
                metrics_accumulator["smoothness"] += smoothness_score
                metrics_accumulator["jerk"] += jerk_metric_val
                metrics_accumulator["orientation"] += orientation_metric_val

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
                "Policy epoch {}/{} - avg loss: {:.4f}, safety: {:.3f}, clearance: {:.3f}, goal: {:.3f}, align: {:.3f}, smoothness: {:.3f}, jerk: {:.3f}, orientation: {:.3f}".format(
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
    parser.add_argument("dataset", type=pathlib.Path, help="Directory containing *.npz samples")
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
    parser.add_argument("--primitive_steps", type=int, default=4)
    parser.add_argument("--primitive_dt", type=float, default=0.25)
    parser.add_argument("--path_samples_per_step", type=int, default=3)
    parser.add_argument("--radio_range", type=float, default=5.0)
    parser.add_argument("--vel_max_train", type=float, default=6.0)
    parser.add_argument("--acc_max_train", type=float, default=3.0)
    parser.add_argument("--v_forward_mean", type=float, default=2.0)
    parser.add_argument("--v_forward_sigma", type=float, default=0.45)
    parser.add_argument("--v_std_unit", type=float, default=0.22)
    parser.add_argument("--a_std_unit", type=float, default=0.35)
    parser.add_argument("--goal_length_scale", type=float, default=1.0)
    parser.add_argument("--offset_gain", type=float, default=0.25)
    parser.add_argument("--yaw_std_deg", type=float, default=20.0)
    parser.add_argument("--pitch_std_deg", type=float, default=10.0)
    parser.add_argument("--policy_seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dataset = ObstacleDataset(args.dataset)
    total_len = len(base_dataset)
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
            yaw_std_deg=args.yaw_std_deg,
            pitch_std_deg=args.pitch_std_deg,
            goal_length_scale=max(0.2, args.goal_length_scale),
            offset_gain=max(0.05, args.offset_gain),
        )
        primitive_steps = max(3, min(5, args.primitive_steps))
        samples_per_step = max(1, args.path_samples_per_step)
        train_navigation_policy(
            nav_dataset,
            device,
            args.policy_epochs,
            args.policy_batch,
            args.policy_lr,
            args.policy_noise,
            args.policy_output,
            primitive_config,
            args.primitive_dt,
            primitive_steps,
            samples_per_step,
            args.camera_pitch_deg,
            args.policy_seed,
        )


if __name__ == "__main__":
    main()
