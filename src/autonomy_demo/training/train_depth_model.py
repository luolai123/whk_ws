#!/usr/bin/env python3
"""DepthAnything-style RGB->Depth training using simulator distances as supervision.
- Input: RGB image (H,W,3) in [0,255]
- Target: per-pixel distance map (H,W) in meters from dataset 'distances'
- Loss: L1 + SSIM + edge-aware gradient loss
- Output: depth model weights .pt
"""

import argparse
import pathlib
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
			x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
		x = torch.cat([skip, x], dim=1)
		return self.conv(x)


class DepthUNet(torch.nn.Module):
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
		self.pred = torch.nn.Conv2d(32, 1, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x1 = self.enc1(x)
		x2 = self.enc2(self.pool(x1))
		x3 = self.enc3(self.pool(x2))
		b = self.bottleneck(self.pool(x3))
		x = self.up3(b, x3)
		x = self.up2(x, x2)
		x = self.up1(x, x1)
		depth = self.pred(x)
		return depth


class DepthDataset(Dataset):
	def __init__(self, data_dir: pathlib.Path, indices: Optional[List[int]] = None, augment: bool = True) -> None:
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
		sample = np.load(self.files[self.indices[idx]])
		image = sample["image"].astype(np.float32) / 255.0
		depth = sample["distances"].astype(np.float32)
		if self.augment:
			if random.random() < 0.5:
				image = np.flip(image, axis=1).copy()
				depth = np.flip(depth, axis=1).copy()
			if random.random() < 0.25:
				image = np.clip(image + (random.random() - 0.5) * 0.1, 0.0, 1.0)
			if random.random() < 0.2:
				noise = np.random.normal(0.0, 0.02, size=image.shape).astype(np.float32)
				image = np.clip(image + noise, 0.0, 1.0)
		img_t = torch.from_numpy(image).permute(2, 0, 1)
		depth_t = torch.from_numpy(depth).unsqueeze(0)
		return img_t, depth_t


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	# Simple SSIM on patches via avg pooling
	C1 = 0.01 ** 2
	C2 = 0.03 ** 2
	pool = torch.nn.AvgPool2d(3, 1)
	mu_x = pool(pred)
	mu_y = pool(target)
	sigma_x = pool(pred * pred) - mu_x * mu_x
	sigma_y = pool(target * target) - mu_y * mu_y
	sigma_xy = pool(pred * target) - mu_x * mu_y
	ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))
	return torch.clamp((1.0 - ssim) * 0.5, 0.0, 1.0).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	def grad(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		dx = img[:, :, :, 1:] - img[:, :, :, :-1]
		dy = img[:, :, 1:, :] - img[:, :, :-1, :]
		return dx, dy
	dx_p, dy_p = grad(pred)
	dx_t, dy_t = grad(target)
	return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


def train_depth(
	model: DepthUNet,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: torch.device,
	epochs: int,
	lr: float,
	max_range: float,
) -> None:
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
	model.train()
	for epoch in range(epochs):
		model.train()
		running = 0.0
		for rgb, depth in train_loader:
			rgb = rgb.to(device)
			depth = depth.to(device)
			optimizer.zero_grad()
			pred = model(rgb)
			pred = torch.relu(pred)  # non-negative distances
			l1 = F.l1_loss(pred, depth)
			ssim = ssim_loss(pred / max_range, depth / max_range)
			g = gradient_loss(pred, depth)
			loss = l1 + 0.5 * ssim + 0.1 * g
			loss.backward()
			optimizer.step()
			running += float(loss.item())
		avg_train = running / max(1, len(train_loader))
		# validation
		model.eval()
		val_running = 0.0
		with torch.no_grad():
			for rgb, depth in val_loader:
				rgb = rgb.to(device)
				depth = depth.to(device)
				pred = torch.relu(model(rgb))
				l1 = F.l1_loss(pred, depth)
				ssim = ssim_loss(pred / max_range, depth / max_range)
				g = gradient_loss(pred, depth)
				val_running += float((l1 + 0.5 * ssim + 0.1 * g).item())
		avg_val = val_running / max(1, len(val_loader))
		scheduler.step(avg_val)
		print(f"Epoch {epoch+1}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train RGB->Depth model with simulator supervision")
	parser.add_argument("dataset", type=pathlib.Path)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--val_split", type=float, default=0.2)
	parser.add_argument("--max_range", type=float, default=12.0)
	parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path.home() / "autonomy_demo" / "depth_model.pt")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	full = DepthDataset(args.dataset, augment=True)
	n = len(full)
	idx = list(range(n))
	random.shuffle(idx)
	if n < 2:
		train_idx = idx
		val_idx: List[int] = []
	else:
		val_len = max(1, int(n * args.val_split))
		train_idx = idx[val_len:]
		val_idx = idx[:val_len]
	train_ds = DepthDataset(args.dataset, indices=train_idx, augment=True)
	val_ds = DepthDataset(args.dataset, indices=val_idx, augment=False)
	train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=args.batch)
	print(f"Depth train={len(train_ds)} val={len(val_ds)}")
	model = DepthUNet().to(device)
	train_depth(model, train_loader, val_loader, device, args.epochs, args.lr, float(args.max_range))
	args.output.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), args.output)
	print(f"Saved depth model to {args.output}")


if __name__ == "__main__":
	main()
