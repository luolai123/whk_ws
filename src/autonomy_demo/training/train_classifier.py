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
    quintic_coefficients,
    sample_quintic,
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

    @staticmethod
    def _load_label(sample: np.lib.npyio.NpzFile) -> np.ndarray:
        if "label" in sample:
            return sample["label"].astype(np.int64)
        for key in ("labels", "label_map", "safe_mask"):
            if key in sample:
                return sample[key].astype(np.int64)

        distances: Optional[np.ndarray] = None
        if "distances" in sample:
            distances = sample["distances"].astype(np.float32)
        elif "depth" in sample:
            distances = sample["depth"].astype(np.float32)

        if distances is not None:
            threshold = 0.0
            if "near_threshold" in sample:
                threshold = float(sample["near_threshold"])  # type: ignore[index]
            if threshold <= 0.0:
                threshold = 4.0
            mask = (distances < threshold).astype(np.int64)
            return mask

        raise KeyError("No label, fallback mask, or distance map found in sample")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[self.indices[idx]]
        with np.load(file_path) as sample:
            image = sample["image"].astype(np.float32) / 255.0
            label = self._load_label(sample)

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
            with np.load(self.files[idx]) as sample:
                label = self._load_label(sample)
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
        self.fc2 = torch.nn.Linear(64, 2)

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
                distance_delta, duration_delta = outputs[0]
                distance_scale = torch.clamp(1.0 + 0.25 * distance_delta, 0.6, 1.4)
                duration_scale = torch.clamp(1.0 + 0.2 * duration_delta, 0.7, 1.3)

                base_direction_np = compute_direction_from_pixel(
                    center_col, center_row, dataset.width, dataset.height, 120.0
                ).astype(np.float32)

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

                base_distance = torch.clamp(interp_dist, 1.0, max_range)
                target_distance = base_distance * distance_scale
                primitive_steps = 4
                primitive_duration = primitive_dt * primitive_steps
                duration_value = torch.clamp(primitive_duration * duration_scale, min=primitive_dt)

                speed_value = float(speed.item())
                target_distance_val = float(target_distance.detach().cpu())
                duration_val = float(duration_value.detach().cpu())
                start_pos = np.zeros(3, dtype=np.float32)
                start_vel = base_direction_np * speed_value
                end_pos = base_direction_np * target_distance_val
                end_vel = base_direction_np * speed_value
                coeffs = quintic_coefficients(
                    start_pos,
                    start_vel,
                    np.zeros(3, dtype=np.float32),
                    end_pos,
                    end_vel,
                    np.zeros(3, dtype=np.float32),
                    duration_val,
                )
                points_np, velocities_np = sample_quintic(coeffs, duration_val, primitive_steps)
                path_points = [pt for pt in points_np]
                directions = []
                samples: List[Tuple[int, int]] = []
                last_col = center_col
                last_row = center_row
                for point in points_np[1:]:
                    norm = np.linalg.norm(point)
                    if norm < 1e-4:
                        continue
                    direction = point / norm
                    directions.append(direction)
                    col_pix, row_pix = project_direction_to_pixel(
                        direction, dataset.width, dataset.height, 120.0
                    )
                    last_col, last_row = col_pix, row_pix
                    col_i = int(round(col_pix))
                    row_i = int(round(row_pix))
                    if 0 <= row_i < dataset.height and 0 <= col_i < dataset.width:
                        samples.append((row_i, col_i))
                    else:
                        samples.append((-1, -1))

                if not samples:
                    continue

                safety_hits = []
                clearance_values = []
                for row_i, col_i in samples:
                    if row_i >= 0:
                        safety_hits.append(float(noisy_mask[row_i, col_i]))
                        clearance_values.append(float(clearance_norm[row_i, col_i]))
                    else:
                        safety_hits.append(0.0)
                        clearance_values.append(0.0)

                safety_score = torch.tensor(
                    float(min(safety_hits)), device=device, dtype=torch.float32
                )
                if clearance_values:
                    min_clearance_val = max(0.0, min(clearance_values))
                else:
                    min_clearance_val = float(clearance_center.detach().cpu())
                clearance_score = torch.tensor(
                    min(1.0, min_clearance_val), device=device, dtype=torch.float32
                )

                goal_error = math.sqrt(
                    (last_col - center_col) ** 2 + (last_row - center_row) ** 2
                )
                goal_score = torch.exp(-torch.tensor(goal_error / diag, device=device, dtype=torch.float32))

                stability_penalty = (
                    torch.abs(distance_scale - 1.0) / 0.3 + torch.abs(duration_scale - 1.0) / 0.3
                ) * 0.5
                stability_score = torch.exp(-stability_penalty)

                avg_speed = target_distance_val / max(duration_val, 1e-3)
                speed_penalty = abs(avg_speed - speed_value) / max(speed_value, 1.0)
                speed_score = torch.exp(-torch.tensor(speed_penalty, device=device, dtype=torch.float32))

                smoothness_metric = path_smoothness(path_points)
                smoothness_score = torch.tensor(
                    float(max(0.0, min(1.0, smoothness_metric))), device=device, dtype=torch.float32
                )

                jerk_metric_val = jerk_score(path_points, primitive_dt)
                jerk_score_t = torch.tensor(
                    float(max(0.0, min(1.0, jerk_metric_val))),
                    device=device,
                    dtype=torch.float32,
                )

                orientation_metric_val = orientation_rate_score(directions)
                orientation_score_t = torch.tensor(
                    float(max(0.0, min(1.0, orientation_metric_val))),
                    device=device,
                    dtype=torch.float32,
                )

                reward = (
                    0.5 * safety_score
                    + 0.18 * clearance_score
                    + 0.18 * goal_score
                    + 0.04 * smoothness_score
                    + 0.04 * jerk_score_t
                    + 0.02 * orientation_score_t
                    + 0.02 * stability_score
                    + 0.02 * speed_score
                )
                loss = -reward
                batch_loss += loss
                valid_samples += 1

                metrics_accumulator['safety'] += float(safety_score.detach().cpu())
                metrics_accumulator['clearance'] += float(clearance_score.detach().cpu())
                metrics_accumulator['goal'] += float(goal_score.detach().cpu())
                metrics_accumulator['stability'] += float(stability_score.detach().cpu())
                metrics_accumulator['speed'] += float(speed_score.detach().cpu())
                metrics_accumulator['jerk'] += float(jerk_metric_val)
                metrics_accumulator['orientation'] += float(orientation_metric_val)
                metrics_accumulator['smoothness'] += float(smoothness_metric)

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
