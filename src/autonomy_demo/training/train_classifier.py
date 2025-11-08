#!/usr/bin/env python3
"""Training pipelines for the distance classifier and safe navigation policy."""

import argparse
import math
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from autonomy_demo.safe_navigation import (
    compute_direction_from_pixel,
    find_largest_safe_region
)


class DistanceClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ObstacleDataset(Dataset):
    def __init__(self, data_dir: pathlib.Path) -> None:
        self.files: List[pathlib.Path] = sorted(data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = np.load(self.files[idx])
        image = sample["image"].astype(np.float32) / 255.0
        label = sample["label"].astype(np.int64)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        label_tensor = torch.from_numpy(label)
        return image_tensor, label_tensor


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


def train(
    model: DistanceClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[List[float], List[float]]:
    criterion = torch.nn.CrossEntropyLoss()
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

        print(f"Epoch {epoch + 1}/{epochs} - train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}")
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

    indices = list(range(len(dataset)))

    for epoch in range(epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_count = 0
        policy.train()

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
                col0 = torch.clamp(torch.floor(col), 0, dataset.width - 1)
                col1 = torch.clamp(col0 + 1, 0, dataset.width - 1)
                frac = col - col0
                dist0 = distances_t[col0.long()]
                dist1 = distances_t[col1.long()]
                interp_dist = dist0 * (1.0 - frac) + dist1 * frac

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
                accuracy_col = torch.exp(-((col - center_col_t) / dataset.width) ** 2)
                accuracy_row = torch.exp(-((row - center_row_t) / dataset.height) ** 2)
                accuracy_score = 0.5 * (accuracy_col + accuracy_row)

                stability_penalty = (
                    torch.abs(length_scale - 1.0) / 0.2
                    + torch.abs(pitch_offset) / pitch_limit
                    + torch.abs(yaw_offset) / yaw_limit
                ) / 3.0
                stability_score = torch.exp(-stability_penalty)

                reward = 0.5 * safety_score + 0.3 * accuracy_score + 0.2 * stability_score
                loss = -reward
                batch_loss += loss
                valid_samples += 1

            if valid_samples == 0:
                continue
            batch_loss = batch_loss / valid_samples
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * valid_samples
            epoch_count += valid_samples

        if epoch_count:
            avg_loss = epoch_loss / epoch_count
        else:
            avg_loss = float("nan")
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
    dataset = ObstacleDataset(args.dataset)
    if len(dataset) < 2:
        train_set = Subset(dataset, list(range(len(dataset))))
        val_set = Subset(dataset, [])
    else:
        val_len = max(1, int(len(dataset) * args.val_split))
        train_len = len(dataset) - val_len
        if train_len <= 0:
            train_len = 1
            val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

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
