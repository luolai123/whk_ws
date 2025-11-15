#!/usr/bin/env python3
"""独立的图像分割推理节点 - 用于区分安全区域和障碍物"""

import pathlib
import time
from typing import Optional, Tuple

import numpy as np
import rospy
import torch
import torch.nn.functional as F
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

# 从原始推理脚本导入模型定义
import sys
import pathlib
# 添加当前目录到路径以便导入
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from inference_node import ConvBlock, DistanceClassifier, UpBlock


class SegmentationInferenceNode:
	"""
	图像分割推理节点 - 仅进行安全区域和障碍物的二分类
	
	推理时：
	- 输入：RGB图像（3通道）
	- 输出：每个像素的二分类结果（安全/障碍物）
	- 不依赖任何环境障碍物信息，完全基于RGB图像进行推理
	"""
	
	def __init__(self) -> None:
		self.bridge = CvBridge()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = DistanceClassifier()
		model_path = rospy.get_param(
			"~model_path", str(pathlib.Path.home() / "autonomy_demo" / "segmentation_model.pt")
		)
		# 展开用户主目录路径
		model_path = pathlib.Path(model_path).expanduser()
		if model_path.exists():
			try:
				self.model.load_state_dict(torch.load(model_path, map_location=self.device))
				rospy.loginfo("已加载分割模型权重从 %s", model_path)
			except Exception as e:
				rospy.logerr("加载分割模型失败 %s: %s - 使用随机初始化", model_path, e)
		else:
			rospy.logwarn("分割模型权重 %s 未找到 - 使用随机初始化", model_path)
			rospy.logwarn("请确保模型文件存在，否则推理结果将不正确！")
		self.model.to(self.device)
		self.model.eval()
		self.softmax = torch.nn.Softmax(dim=1)

		self.safe_threshold = float(rospy.get_param("~safe_probability_threshold", 0.55))
		self.use_dynamic_threshold = bool(rospy.get_param("~use_dynamic_threshold", True))
		self.dynamic_percentile = float(rospy.get_param("~dynamic_percentile", 0.6))
		# speckle 抑制参数
		self.median_ksize = int(rospy.get_param("~median_ksize", 3))
		self.min_region_fraction = float(rospy.get_param("~min_region_fraction", 0.002))
		self.invert_mask = bool(rospy.get_param("~invert_mask", False))

		self.camera_info: Optional[CameraInfo] = None
		self.image_shape: Optional[Tuple[int, int]] = None

		self.label_pub = rospy.Publisher("drone/rgb/distance_class", Image, queue_size=1)
		self.safe_mask_pub = rospy.Publisher("drone/safe_mask", Image, queue_size=1)
		self.probability_pub = rospy.Publisher("drone/safe_probability", Image, queue_size=1)

		rospy.Subscriber("drone/rgb/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)
		rospy.Subscriber("drone/rgb/image_raw", Image, self.image_callback, queue_size=1)

	def camera_info_callback(self, info: CameraInfo) -> None:
		self.camera_info = info
		self.image_shape = (info.height, info.width)

	def _filter_small_components(self, mask: np.ndarray) -> np.ndarray:
		"""去除小连通域"""
		h, w = mask.shape
		min_pixels = max(1, int(h * w * max(0.0, min(0.1, self.min_region_fraction))))
		num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
		clean = np.zeros_like(mask, dtype=bool)
		for i in range(1, num_labels):
			if int(stats[i, cv2.CC_STAT_AREA]) >= min_pixels:
				clean[labels_cc == i] = True
		return clean

	def image_callback(self, msg: Image) -> None:
		if self.camera_info is None:
			return
		start_time = time.perf_counter()
		cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
		height, width, _ = cv_image.shape
		self.image_shape = (height, width)

		# 只使用RGB图像（3通道），不依赖任何环境障碍物信息
		# 确保图像格式正确：RGB，值范围[0, 1]
		tensor = torch.from_numpy(cv_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
		if tensor.shape[1] != 3:
			rospy.logerr_once("输入图像不是RGB格式（3通道），当前通道数: %d", tensor.shape[1])
			return
		tensor = tensor.to(self.device)
		
		with torch.no_grad():
			logits = self.model(tensor)
			probs = self.softmax(logits)

		safe_prob = probs[:, 0, :, :].squeeze(0).cpu().numpy()
		obstacle_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy()
		
		# 检查模型输出是否合理（调试用）
		safe_prob_min = float(np.min(safe_prob))
		safe_prob_max = float(np.max(safe_prob))
		safe_prob_mean = float(np.mean(safe_prob))
		rospy.logdebug_throttle(10.0, "模型输出统计: safe_prob [min=%.3f, max=%.3f, mean=%.3f]", 
							safe_prob_min, safe_prob_max, safe_prob_mean)
		
		# 高斯 + 中值滤波平滑概率，降低大空地的斑点
		safe_prob = cv2.GaussianBlur(safe_prob, (5, 5), 0)
		if self.median_ksize >= 3 and self.median_ksize % 2 == 1:
			safe_prob = cv2.medianBlur((safe_prob * 255).astype(np.uint8), self.median_ksize).astype(np.float32) / 255.0
			obstacle_prob = cv2.medianBlur((obstacle_prob * 255).astype(np.uint8), self.median_ksize).astype(np.float32) / 255.0
		else:
			obstacle_prob = cv2.GaussianBlur(obstacle_prob, (5, 5), 0)
		
		# 动态阈值（与原始代码保持一致）
		threshold = self.safe_threshold
		if self.use_dynamic_threshold:
			# percentile based threshold to avoid "all-safe"/"all-obstacle"
			flat = safe_prob.reshape(-1)
			percentile = np.clip(self.dynamic_percentile, 0.2, 0.9)
			threshold = float(np.percentile(flat, 100.0 * percentile))
			threshold = max(self.safe_threshold, threshold)
		
		safe_mask = safe_prob >= threshold
		
		# 形态学操作清理 + 小连通域剔除
		kernel = np.ones((3, 3), dtype=np.uint8)
		cleaned = cv2.morphologyEx(safe_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
		cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
		safe_mask = cleaned.astype(bool)
		safe_mask = self._filter_small_components(safe_mask)
		
		# 允许运行时反转（若训练标签或可视化语义相反）
		if self.invert_mask:
			safe_mask = ~safe_mask

		# 发布彩色分类结果（绿色=安全，红色=障碍）
		color_map = np.zeros((height, width, 3), dtype=np.uint8)
		color_map[:, :, 0] = 255  # 默认红色（障碍）
		color_map[safe_mask, 0] = 0
		color_map[safe_mask, 1] = 255  # 绿色（安全）
		label_msg = self.bridge.cv2_to_imgmsg(color_map, encoding="rgb8")
		label_msg.header = msg.header
		label_msg.header.frame_id = rospy.get_param("~output_frame", msg.header.frame_id)
		self.label_pub.publish(label_msg)
		
		# 调试信息：输出统计信息
		safe_pixels = np.sum(safe_mask)
		total_pixels = height * width
		safe_ratio = safe_pixels / total_pixels if total_pixels > 0 else 0.0
		rospy.logdebug_throttle(5.0, "分割结果: 安全像素比例=%.2f%%, 阈值=%.3f, 安全像素数=%d/%d", 
							safe_ratio * 100.0, threshold, safe_pixels, total_pixels)

		# 发布安全掩码（单通道）
		safe_mask_msg = self.bridge.cv2_to_imgmsg(
			(safe_mask.astype(np.uint8) * 255), encoding="mono8"
		)
		safe_mask_msg.header = msg.header
		safe_mask_msg.header.frame_id = rospy.get_param("~output_frame", msg.header.frame_id)
		self.safe_mask_pub.publish(safe_mask_msg)

		# 发布安全概率图（单通道浮点）
		safe_prob_uint8 = (np.clip(safe_prob, 0.0, 1.0) * 255).astype(np.uint8)
		prob_msg = self.bridge.cv2_to_imgmsg(safe_prob_uint8, encoding="mono8")
		prob_msg.header = msg.header
		prob_msg.header.frame_id = rospy.get_param("~output_frame", msg.header.frame_id)
		self.probability_pub.publish(prob_msg)

		elapsed_ms = (time.perf_counter() - start_time) * 1000.0
		if elapsed_ms > 2.0:
			rospy.logwarn_throttle(1.0, "分割推理超过 2 ms: %.3f ms", elapsed_ms)


def main() -> None:
	rospy.init_node("segmentation_inference")
	SegmentationInferenceNode()
	rospy.loginfo("图像分割推理节点已启动")
	rospy.spin()


if __name__ == "__main__":
	main()

