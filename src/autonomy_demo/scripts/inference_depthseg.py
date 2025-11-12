#!/usr/bin/env python3
"""RGB-only depth inference -> free-space segmentation.
- Loads depth model trained by train_depth_model.py
- Converts depth to free-space: plane-aware ground masking + hysteresis threshold + morphology
- Publishes /drone/rgb/distance_class (green=safe, red=obstacle), /drone/safe_mask, /drone/depth
"""

import math
import pathlib
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


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
			x = torch.nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
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
		return self.pred(x)


class DepthSegNode:
	def __init__(self) -> None:
		self.bridge = CvBridge()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = DepthUNet().to(self.device)
		model_path = pathlib.Path(rospy.get_param("~depth_model", str(pathlib.Path.home() / "autonomy_demo" / "depth_model.pt"))).expanduser()
		if model_path.exists():
			self.model.load_state_dict(torch.load(model_path, map_location=self.device))
			rospy.loginfo("Loaded depth model from %s", model_path)
		else:
			rospy.logwarn("Depth model %s not found - using random init", model_path)
		self.model.eval()
		self.max_range = float(rospy.get_param("~max_range", 12.0))
		self.hyst_low = float(rospy.get_param("~hysteresis_low", 0.35))
		self.hyst_high = float(rospy.get_param("~hysteresis_high", 0.45))
		self.morph = int(rospy.get_param("~morph", 3))
		self.floor_ratio = float(rospy.get_param("~floor_ratio", 0.18))
		self.min_region_fraction = float(rospy.get_param("~min_region_fraction", 0.002))
		self.invert_mask = bool(rospy.get_param("~invert_mask", False))

		self.cam_info: Optional[CameraInfo] = None
		self.depth_pub = rospy.Publisher("drone/depth", Image, queue_size=1)
		self.mask_pub = rospy.Publisher("drone/safe_mask", Image, queue_size=1)
		self.label_pub = rospy.Publisher("drone/rgb/distance_class", Image, queue_size=1)
		rospy.Subscriber("drone/rgb/camera_info", CameraInfo, self.cam_cb, queue_size=1)
		rospy.Subscriber("drone/rgb/image_raw", Image, self.img_cb, queue_size=1)

	def cam_cb(self, info: CameraInfo) -> None:
		self.cam_info = info

	def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
		H, W = mask.shape
		min_pixels = max(1, int(H * W * max(0.0, min(0.1, self.min_region_fraction))))
		num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
		clean = np.zeros_like(mask, dtype=bool)
		for i in range(1, num_labels):
			if int(stats[i, cv2.CC_STAT_AREA]) >= min_pixels:
				clean[labels_cc == i] = True
		return clean

	def _depth_to_mask(self, depth: np.ndarray) -> np.ndarray:
		# 1) clip/normalize depth
		depth = np.clip(depth, 0.0, self.max_range)
		H, W = depth.shape
		# 2) 粗地面估计：取下方 floor_ratio 的行的中位数作为地面估计
		h0 = int(max(0, H * (1.0 - self.floor_ratio)))
		floor_band = depth[h0:, :]
		floor_med = float(np.median(floor_band)) if floor_band.size else self.max_range * 0.5
		# 3) 归一化并双阈值（近=障碍，远=安全）。这里我们要得到"安全"，所以阈值反向
		norm = depth / max(self.max_range, 1e-6)
		# 近处不可达，远处更安全。用floor基线做相对阈值
		low = min(1.0, max(0.0, self.hyst_low * (floor_med / self.max_range)))
		high = min(1.0, max(0.0, self.hyst_high * (floor_med / self.max_range)))
		seed_safe = norm >= high
		seed_obst = norm <= low
		mask = seed_safe.copy()
		# 4) 区域生长：把非确定区域根据邻接扩展到靠近的种子
		undecided = ~(seed_safe | seed_obst)
		kernel = np.ones((3, 3), np.uint8)
		for _ in range(2):
			grow = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
			mask = np.where(undecided & grow, True, mask)
		# 5) 形态学清理 + 小连通域移除
		if self.morph > 0:
			k = np.ones((self.morph, self.morph), np.uint8)
			mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
			mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
			mask = mask.astype(bool)
		mask = self._remove_small_components(mask)
		return mask

	def img_cb(self, msg: Image) -> None:
		if self.cam_info is None:
			return
		start = time.perf_counter()
		rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
		rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
		with torch.no_grad():
			pred = torch.relu(self.model(rgb_t))
		depth = pred.squeeze(0).squeeze(0).cpu().numpy()
		mask = self._depth_to_mask(depth)
		if self.invert_mask:
			mask = ~mask
		# publish depth
		depth_norm = (np.clip(depth / self.max_range, 0.0, 1.0) * 255).astype(np.uint8)
		depth_msg = self.bridge.cv2_to_imgmsg(depth_norm, encoding="mono8")
		depth_msg.header = msg.header
		self.depth_pub.publish(depth_msg)
		# publish mask + color overlay (green=safe, red=obstacle)
		mask_msg = self.bridge.cv2_to_imgmsg((mask.astype(np.uint8) * 255), encoding="mono8")
		mask_msg.header = msg.header
		self.mask_pub.publish(mask_msg)
		color = np.zeros_like(rgb)
		color[:, :, 0] = 255  # red obstacle by default
		color[mask, 0] = 0
		color[mask, 1] = 255  # green safe
		label_msg = self.bridge.cv2_to_imgmsg(color, encoding="rgb8")
		label_msg.header = msg.header
		self.label_pub.publish(label_msg)
		elapsed = (time.perf_counter() - start) * 1000.0
		if elapsed > 3.0:
			rospy.logwarn_throttle(1.0, "Depth inference %.2f ms", elapsed)


def main() -> None:
	rospy.init_node("depth_segmentation")
	DepthSegNode()
	rospy.loginfo("Depth-based segmentation node started")
	rospy.spin()


if __name__ == "__main__":
	main()
