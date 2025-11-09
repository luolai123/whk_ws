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


class ObstacleDataset(Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        indices: Optional[List[int]] = None,
        augment: bool = False,
    ) -> None:
        self.files: List[pathlib.Path] = sorted(data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {data_dir}")
        if indices is None:
            self.indices = list(range(len(self.files)))
        else:
            self.indices = list(indices)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[self.indices[idx]]
        sample = np.load(file_path)
        image = sample["image"].astype(np.float32) / 255.0
        label = sample["label"].astype(np.int64)

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
            sample = np.load(self.files[idx])
            label = sample["label"].astype(np.int64)
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


class NavigationDataset(Dataset):
    def __init__(self, data_dir: pathlib.Path) -> None:
        self.files: List[pathlib.Path] = sorted(data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {data_dir}")
        first = np.load(self.files[0])
        label = first["label"]
        self.height, self.width = label.shape

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        sample = np.load(self.files[idx])
        label = sample["label"].astype(np.uint8)
        safe_mask = (label == 0).astype(np.float32)
        distances = sample["distances"].astype(np.float32)

        def _get(name: str, fallback: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if name in sample:
                return sample[name].astype(np.float32)
            return fallback

        metadata: Dict[str, Any] = {
            "pose_position": _get("pose_position", None),
            "pose_orientation": _get("pose_orientation", None),
            "camera_offset": _get("camera_offset", np.array([0.15, 0.0, 0.05], dtype=np.float32)),
            "sphere_centers": _get("sphere_centers", np.empty((0, 3), dtype=np.float32)),
            "sphere_radii": _get("sphere_radii", np.empty((0,), dtype=np.float32)),
            "box_centers": _get("box_centers", np.empty((0, 3), dtype=np.float32)),
            "box_half_extents": _get("box_half_extents", np.empty((0, 3), dtype=np.float32)),
        }
        return safe_mask, distances, metadata


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


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.identity(3, dtype=np.float32)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def apply_offsets_torch(
    base_direction: torch.Tensor, yaw_offset: torch.Tensor, pitch_offset: torch.Tensor
) -> torch.Tensor:
    cp = torch.cos(pitch_offset)
    sp = torch.sin(pitch_offset)
    cy = torch.cos(yaw_offset)
    sy = torch.sin(yaw_offset)

    dir_pitch = torch.stack(
        [
            base_direction[0] * cp + base_direction[2] * sp,
            base_direction[1],
            -base_direction[0] * sp + base_direction[2] * cp,
        ]
    )
    rotated = torch.stack(
        [
            dir_pitch[0] * cy - dir_pitch[1] * sy,
            dir_pitch[0] * sy + dir_pitch[1] * cy,
            dir_pitch[2],
        ]
    )
    norm = torch.linalg.norm(rotated)
    if norm < 1e-6:
        return base_direction
    return rotated / norm


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


def compute_ray_safety(
    direction_world: torch.Tensor,
    origin_world: torch.Tensor,
    sphere_centers: torch.Tensor,
    sphere_radii: torch.Tensor,
    max_range: float,
) -> torch.Tensor:
    if sphere_centers.numel() == 0:
        return torch.tensor(max_range, device=direction_world.device, dtype=torch.float32)
    offsets = sphere_centers - origin_world
    projections = torch.matmul(offsets, direction_world)
    closest = projections
    perp_sq = torch.sum(offsets * offsets, dim=1) - closest * closest
    radii_sq = sphere_radii * sphere_radii
    discriminant = radii_sq - perp_sq
    valid = discriminant >= 0.0
    discriminant = torch.clamp(discriminant, min=0.0)
    sqrt_disc = torch.sqrt(discriminant)
    distances = closest - sqrt_disc
    distances = torch.where(valid & (distances > 0.0), distances, torch.full_like(distances, float("inf")))
    min_distance = torch.min(distances)
    return torch.clamp(min_distance, min=0.0, max=max_range)


def train_navigation_policy(
    dataset: NavigationDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_rate: float,
    policy_output: pathlib.Path,
) -> None:
    policy = SafeNavigationPolicy(dataset.height, dataset.width).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    max_range = 12.0
    tan_half_h = math.tan(math.radians(120.0) / 2.0)
    tan_half_v = tan_half_h * (dataset.height / float(dataset.width))
    tan_half_h_t = torch.tensor(tan_half_h, device=device)
    tan_half_v_t = torch.tensor(tan_half_v, device=device)
    pitch_limit = torch.tensor(math.radians(15.0), device=device)
    yaw_limit = torch.tensor(math.radians(15.0), device=device)
    diag = math.sqrt(dataset.width ** 2 + dataset.height ** 2)
    diag_t = torch.tensor(diag, device=device)
    primitive_dt = 0.25

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
            "stability": 0.0,
            "smoothness": 0.0,
            "speed": 0.0,
            "jerk": 0.0,
            "orientation": 0.0,
        }

        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_samples = 0

            for idx in batch:
                safe_mask, distances, metadata = dataset[idx]
                noisy_mask = add_noise(safe_mask, noise_rate)
                region = find_largest_safe_region(noisy_mask.astype(bool), 0.05)
                if region is None:
                    continue

                center_row, center_col = region.centroid
                mask_tensor = torch.from_numpy(noisy_mask).to(device=device, dtype=torch.float32)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

                speed = torch.rand(1, device=device) * 4.0 + 3.0
                normalized_speed = (speed - 3.0) / 4.0

                outputs = policy(mask_tensor, normalized_speed)
                length_delta, pitch_delta, yaw_delta = outputs[0]
                length_scale = torch.clamp(1.0 + 0.2 * length_delta, 0.5, 1.5)
                pitch_offset = pitch_limit * pitch_delta
                yaw_offset = yaw_limit * yaw_delta

                base_direction_np = compute_direction_from_pixel(
                    center_col, center_row, dataset.width, dataset.height, 120.0
                )
                base_direction = torch.from_numpy(base_direction_np).to(device=device)
                rotated_dir = apply_offsets_torch(base_direction, yaw_offset, pitch_offset)

                horizontal = torch.atan2(rotated_dir[1], rotated_dir[0])
                vertical = torch.atan2(
                    rotated_dir[2], torch.sqrt(rotated_dir[0] * rotated_dir[0] + rotated_dir[1] * rotated_dir[1])
                )
                u = torch.tan(horizontal) / tan_half_h_t
                v = torch.tan(vertical) / tan_half_v_t
                u = torch.clamp(u, -1.0, 1.0)
                v = torch.clamp(v, -1.0, 1.0)
                col = ((u + 1.0) * 0.5) * dataset.width - 0.5
                row = (1.0 - (v + 1.0) * 0.5) * dataset.height - 0.5

                distances_t = torch.from_numpy(distances).to(device=device)
                clearance_map = cv2.distanceTransform(
                    (noisy_mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5
                ).astype(np.float32)
                max_clearance = float(np.max(clearance_map))
                if max_clearance > 1e-6:
                    clearance_norm = clearance_map / max_clearance
                else:
                    clearance_norm = clearance_map
                clearance_t = torch.from_numpy(clearance_norm).to(device=device)

                row0 = torch.clamp(torch.floor(row), 0, dataset.height - 1)
                row1 = torch.clamp(row0 + 1, 0, dataset.height - 1)
                col0 = torch.clamp(torch.floor(col), 0, dataset.width - 1)
                col1 = torch.clamp(col0 + 1, 0, dataset.width - 1)
                row_frac = row - row0
                col_frac = col - col0

                def bilinear(sample: torch.Tensor) -> torch.Tensor:
                    s00 = sample[row0.long(), col0.long()]
                    s01 = sample[row0.long(), col1.long()]
                    s10 = sample[row1.long(), col0.long()]
                    s11 = sample[row1.long(), col1.long()]
                    return (
                        s00 * (1.0 - row_frac) * (1.0 - col_frac)
                        + s01 * (1.0 - row_frac) * col_frac
                        + s10 * row_frac * (1.0 - col_frac)
                        + s11 * row_frac * col_frac
                    )

                interp_dist = bilinear(distances_t)
                clearance_center = bilinear(clearance_t)

                sphere_centers = metadata.get("sphere_centers")
                sphere_radii = metadata.get("sphere_radii")
                box_centers = metadata.get("box_centers")
                box_half_extents = metadata.get("box_half_extents")
                if sphere_centers is None:
                    sphere_centers = np.empty((0, 3), dtype=np.float32)
                if sphere_radii is None:
                    sphere_radii = np.empty((0,), dtype=np.float32)
                if box_centers is None:
                    box_centers = np.empty((0, 3), dtype=np.float32)
                if box_half_extents is None:
                    box_half_extents = np.empty((0, 3), dtype=np.float32)
                box_radii = np.linalg.norm(box_half_extents, axis=1) if box_half_extents.size else np.empty((0,), dtype=np.float32)

                all_centers = np.concatenate([sphere_centers, box_centers], axis=0)
                all_radii = np.concatenate([sphere_radii, box_radii], axis=0)
                centers_t = torch.from_numpy(all_centers).to(device=device)
                radii_t = torch.from_numpy(all_radii).to(device=device)

                pose_position = metadata.get("pose_position")
                pose_orientation = metadata.get("pose_orientation")
                camera_offset = metadata.get("camera_offset")
                if pose_position is None or pose_orientation is None or camera_offset is None:
                    continue
                rotation = quaternion_to_matrix(pose_orientation)
                camera_position = pose_position + rotation.dot(camera_offset)
                origin_t = torch.from_numpy(camera_position).to(device=device)
                rotation_t = torch.from_numpy(rotation).to(device=device)
                world_direction = rotation_t.matmul(rotated_dir)
                world_direction = world_direction / torch.linalg.norm(world_direction)

                safety_ray = compute_ray_safety(world_direction, origin_t, centers_t, radii_t, max_range)
                safety_score = torch.clamp(safety_ray / max_range, 0.0, 1.0)

                center_col_t = torch.tensor(center_col, device=device, dtype=torch.float32)
                center_row_t = torch.tensor(center_row, device=device, dtype=torch.float32)
                goal_distance = torch.sqrt(
                    (col - center_col_t) ** 2 + (row - center_row_t) ** 2
                )
                goal_score = torch.exp(-(goal_distance / diag_t))

                stability_penalty = (
                    torch.abs(length_scale - 1.0) / 0.2
                    + torch.abs(pitch_offset) / pitch_limit
                    + torch.abs(yaw_offset) / yaw_limit
                ) / 3.0
                stability_score = torch.exp(-stability_penalty)

                smoothness_penalty = torch.sqrt(
                    (yaw_offset / yaw_limit) ** 2 + (pitch_offset / pitch_limit) ** 2
                )
                smoothness_score = torch.exp(-smoothness_penalty)

                commanded_speed = speed * length_scale
                speed_penalty = torch.abs(commanded_speed - speed) / torch.clamp(speed, min=1.0)
                speed_score = torch.exp(-speed_penalty)

                dirs = sample_yopo_directions(
                    base_direction_np,
                    float(yaw_offset.detach().cpu().numpy()),
                    float(pitch_offset.detach().cpu().numpy()),
                    6,
                )
                points = [np.zeros(3, dtype=np.float32)]
                for direction in dirs:
                    points.append(points[-1] + direction)

                clearance_values: List[float] = []
                for direction in dirs:
                    col_dir, row_dir = project_direction_to_pixel(
                        direction, dataset.width, dataset.height, 120.0
                    )
                    col_i = int(round(col_dir))
                    row_i = int(round(row_dir))
                    if 0 <= col_i < dataset.width and 0 <= row_i < dataset.height:
                        clearance_values.append(float(clearance_norm[row_i, col_i]))
                    else:
                        clearance_values.append(0.0)
                if clearance_values:
                    min_clearance_val = max(0.0, min(clearance_values))
                else:
                    min_clearance_val = float(clearance_center.detach().cpu().item())
                clearance_score = torch.tensor(
                    min(1.0, max(0.0, min_clearance_val)),
                    device=device,
                    dtype=torch.float32,
                )

                jerk_metric_val = jerk_score(points, primitive_dt)
                jerk_score_t = torch.tensor(
                    float(max(0.0, min(1.0, jerk_metric_val))),
                    device=device,
                    dtype=torch.float32,
                )
                orientation_metric_val = orientation_rate_score(dirs)
                orientation_score_t = torch.tensor(
                    float(max(0.0, min(1.0, orientation_metric_val))),
                    device=device,
                    dtype=torch.float32,
                )

                reward = (
                    0.45 * safety_score
                    + 0.20 * clearance_score
                    + 0.18 * goal_score
                    + 0.06 * smoothness_score
                    + 0.04 * jerk_score_t
                    + 0.04 * orientation_score_t
                    + 0.02 * stability_score
                    + 0.01 * speed_score
                )
                loss = -reward
                batch_loss += loss
                valid_samples += 1

                metrics_accumulator["safety"] += float(safety_score.detach().cpu())
                metrics_accumulator["clearance"] += float(clearance_score.detach().cpu())
                metrics_accumulator["goal"] += float(goal_score.detach().cpu())
                metrics_accumulator["stability"] += float(stability_score.detach().cpu())
                metrics_accumulator["speed"] += float(speed_score.detach().cpu())
                metrics_accumulator["jerk"] += float(jerk_metric_val)
                metrics_accumulator["orientation"] += float(orientation_metric_val)

                trajectory_smooth = path_smoothness(points)
                metrics_accumulator["smoothness"] += trajectory_smooth
                epoch_count += 1

            if valid_samples == 0:
                continue
            batch_loss = batch_loss / valid_samples
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * valid_samples

        if epoch_count:
            avg_loss = epoch_loss / epoch_count
        else:
            avg_loss = float("nan")
        if epoch_count:
            averaged_metrics = {
                key: value / epoch_count for key, value in metrics_accumulator.items()
            }
            print(
                "Policy epoch {}/{} - avg loss: {:.4f}, safety: {:.3f}, clearance: {:.3f}, goal: {:.3f}, smoothness: {:.3f}, jerk: {:.3f}, orientation: {:.3f}, stability: {:.3f}, speed: {:.3f}".format(
                    epoch + 1,
                    epochs,
                    avg_loss,
                    averaged_metrics["safety"],
                    averaged_metrics["clearance"],
                    averaged_metrics["goal"],
                    averaged_metrics["smoothness"],
                    averaged_metrics["jerk"],
                    averaged_metrics["orientation"],
                    averaged_metrics["stability"],
                    averaged_metrics["speed"],
                )
            )
        else:
            print(f"Policy epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.4f}")

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = ObstacleDataset(args.dataset)
    total_len = len(full_dataset)
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

    train_dataset = ObstacleDataset(args.dataset, indices=train_indices, augment=True)
    val_dataset = ObstacleDataset(args.dataset, indices=val_indices, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch)

    model = DistanceClassifier().to(device)
    train(model, train_loader, val_loader, device, args.epochs, args.lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved trained model to {args.output}")

    if not args.no_policy:
        nav_dataset = NavigationDataset(args.dataset)
        train_navigation_policy(
            nav_dataset,
            device,
            args.policy_epochs,
            args.policy_batch,
            args.policy_lr,
            args.policy_noise,
            args.policy_output,
        )


if __name__ == "__main__":
    main()
