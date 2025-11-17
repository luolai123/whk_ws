#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全基元偏航训练（提速版）
主要优化：
1) 将 find_largest_safe_region 与 distanceTransform 从训练循环中移除，改为样本级预缓存；
2) 缓存与几何/障碍物/位姿相关的张量到指定 device，避免重复构造；
3) 启用 AMP（自动混合精度）与可选 torch.compile；
4) 清理无效计算与重复的 numpy<->torch 转换。
"""

import argparse
import math
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from autonomy_demo.safe_navigation import (
    compute_direction_from_pixel,
    # find_largest_safe_region,  # 已不再在训练内使用
    jerk_score,
    orientation_rate_score,
    path_smoothness,
    project_direction_to_pixel,
    sample_yopo_directions,
)

# 从原始训练脚本导入必要的类和函数（保持一致性）
import sys
import pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).parent))
from train_classifier import (
    NavigationDataset,
    SafeNavigationPolicy,
    add_noise,
    apply_offsets_torch,
    compute_ray_safety,
    quaternion_to_matrix,
)


# -----------------------------
# 工具：样本级预处理与缓存
# -----------------------------

class _SampleCache:
    """
    针对每个样本缓存以下信息（多数驻留在 device 上）：
    - mask_np: 原始安全掩码 (float32, HxW)
    - clr_np / clr_t: 归一化 clearance 距离图 (numpy / torch)
    - center_rc: (row, col) 以 clr_np 最大值为中心的“最大安全点”
    - base_dir_np / base_dir_t: 基于中心像素反投影到相机坐标系的单位方向
    - centers_t / radii_t: 圆+盒的障碍物合并后参数
    - origin_t / rotation_t: 相机世界坐标与旋转
    - center_row_t / center_col_t: torch 标量
    """
    __slots__ = (
        "mask_np", "clr_np", "clr_t",
        "center_rc", "base_dir_np", "base_dir_t",
        "centers_t", "radii_t",
        "origin_t", "rotation_t",
        "center_row_t", "center_col_t",
    )

    def __init__(self):
        self.mask_np = None
        self.clr_np = None
        self.clr_t  = None
        self.center_rc = (0, 0)
        self.base_dir_np = None
        self.base_dir_t  = None
        self.centers_t = None
        self.radii_t   = None
        self.origin_t  = None
        self.rotation_t= None
        self.center_row_t = None
        self.center_col_t = None


def _build_dataset_cache(
    dataset: NavigationDataset,
    device: torch.device,
    fov_deg: float = 120.0,
) -> List[_SampleCache]:
    """
    对数据集做一次性预处理与缓存。
    注意：为了速度，我们对“安全区中心”的估计采用预先计算的 clearance map 的最大值位置，
          而不是每次在训练循环用连通域寻找最大安全区。
    """
    H, W = dataset.height, dataset.width
    caches: List[_SampleCache] = []

    for idx in range(len(dataset)):
        cache = _SampleCache()
        safe_mask, _distances_unused, metadata = dataset[idx]

        # ---- 基本校验 ----
        if not isinstance(safe_mask, np.ndarray):
            raise TypeError(f"dataset[{idx}] safe_mask 必须是 numpy.ndarray")

        mask = safe_mask.astype(np.float32)  # HxW
        cache.mask_np = mask

        # ---- clearance map：一次性计算并归一化 ----
        # 使用 cv2.distanceTransform 的 L2 距离；只计算一次，训练时直接复用
        clr = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        m = float(np.max(clr))
        if m > 1e-6:
            clr /= m
        cache.clr_np = clr
        cache.clr_t  = torch.from_numpy(clr).to(device=device, dtype=torch.float32)

        # ---- 以 clr 最大值作为“安全中心像素” ----
        flat_idx = int(np.argmax(clr))
        r, c = divmod(flat_idx, W)
        cache.center_rc = (r, c)
        cache.center_row_t = torch.tensor(r, device=device, dtype=torch.float32)
        cache.center_col_t = torch.tensor(c, device=device, dtype=torch.float32)

        # ---- 几何：基于中心像素的相机射线（相机坐标系）----
        base_dir_np = compute_direction_from_pixel(c, r, W, H, fov_deg)  # np.float32(3,)
        cache.base_dir_np = base_dir_np
        cache.base_dir_t  = torch.from_numpy(base_dir_np).to(device=device, dtype=torch.float32)

        # ---- 障碍物合并（球 + 盒转等半径）----
        sphere_centers = metadata.get("sphere_centers")
        sphere_radii   = metadata.get("sphere_radii")
        box_centers    = metadata.get("box_centers")
        box_half_ext   = metadata.get("box_half_extents")

        if sphere_centers is None:
            sphere_centers = np.empty((0, 3), dtype=np.float32)
        if sphere_radii is None:
            sphere_radii = np.empty((0,), dtype=np.float32)
        if box_centers is None:
            box_centers = np.empty((0, 3), dtype=np.float32)
        if box_half_ext is None:
            box_half_ext = np.empty((0, 3), dtype=np.float32)

        box_radii = np.linalg.norm(box_half_ext, axis=1) if box_half_ext.size else np.empty((0,), dtype=np.float32)
        all_centers = np.concatenate([sphere_centers, box_centers], axis=0)
        all_radii   = np.concatenate([sphere_radii, box_radii], axis=0)

        cache.centers_t = torch.from_numpy(all_centers).to(device=device, dtype=torch.float32) if all_centers.size else torch.empty((0,3), device=device, dtype=torch.float32)
        cache.radii_t   = torch.from_numpy(all_radii).to(device=device, dtype=torch.float32)   if all_radii.size   else torch.empty((0,),  device=device, dtype=torch.float32)

        # ---- 相机位姿 ----
        pose_position  = metadata.get("pose_position")
        pose_orientation = metadata.get("pose_orientation")
        camera_offset  = metadata.get("camera_offset")
        if pose_position is None or pose_orientation is None or camera_offset is None:
            # 缓存为空，训练时会跳过该样本
            cache.origin_t   = None
            cache.rotation_t = None
        else:
            rot = quaternion_to_matrix(pose_orientation)         # 3x3 (np)
            cam_pos = pose_position + rot.dot(camera_offset)     # 3   (np)
            cache.origin_t   = torch.from_numpy(cam_pos).to(device=device, dtype=torch.float32)
            cache.rotation_t = torch.from_numpy(rot).to(device=device, dtype=torch.float32)

        caches.append(cache)

    return caches


# -----------------------------
# 训练主循环（提速版）
# -----------------------------

def train_navigation_policy(
    dataset: NavigationDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_rate: float,
    policy_output: pathlib.Path,
    use_amp: bool = True,
    try_compile: bool = True,
) -> None:
    """训练安全导航策略（安全基元偏航，提速版）"""

    # 模型与优化器
    policy = SafeNavigationPolicy(dataset.height, dataset.width).to(device)
    if try_compile:
        try:
            policy = torch.compile(policy)  # 需要 PyTorch 2.x
        except Exception:
            pass

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # 常量（放 device）
    fov_deg = 120.0
    max_range = 12.0
    tan_half_h = math.tan(math.radians(fov_deg) / 2.0)
    tan_half_v = tan_half_h * (dataset.height / float(dataset.width))
    tan_half_h_t = torch.tensor(tan_half_h, device=device, dtype=torch.float32)
    tan_half_v_t = torch.tensor(tan_half_v, device=device, dtype=torch.float32)
    pitch_limit  = torch.tensor(math.radians(15.0), device=device, dtype=torch.float32)
    yaw_limit    = torch.tensor(math.radians(15.0), device=device, dtype=torch.float32)
    diag_t       = torch.tensor(math.sqrt(dataset.width ** 2 + dataset.height ** 2), device=device, dtype=torch.float32)
    primitive_dt = 0.25

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # 一次性构建样本缓存：移除训练时热点开销
    caches = _build_dataset_cache(dataset, device, fov_deg=fov_deg)

    indices = list(range(len(dataset)))

    for epoch in range(epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_count = 0
        policy.train()
        metrics_acc = {
            "safety": 0.0, "clearance": 0.0, "goal": 0.0,
            "stability": 0.0, "smoothness": 0.0, "speed": 0.0,
            "jerk": 0.0, "orientation": 0.0,
        }

        for start in range(0, len(indices), batch_size):
            batch = indices[start: start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss_accum = 0.0
            valid_samples = 0

            for idx in batch:
                cache = caches[idx]

                # 缺少位姿/相机信息的样本跳过
                if cache.origin_t is None or cache.rotation_t is None:
                    continue

                # 训练输入：对 mask 做轻微噪声增强（保持原逻辑，但不重新算 clearance）
                noisy_mask = add_noise(cache.mask_np, noise_rate) if noise_rate > 0 else cache.mask_np
                mask_tensor = torch.from_numpy(noisy_mask).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                # 随速（与原脚本一致）
                speed = torch.rand(1, device=device, dtype=torch.float32) * 4.0 + 3.0
                normalized_speed = (speed - 3.0) / 4.0

                # 前向（AMP）
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    outputs = policy(mask_tensor, normalized_speed)
                # 假设输出形状与原脚本一致：取第一个样本的三个分量
                length_delta, pitch_delta, yaw_delta = outputs[0]
                length_scale = torch.clamp(1.0 + 0.2 * length_delta, 0.5, 1.5)
                pitch_offset = pitch_limit * pitch_delta
                yaw_offset   = yaw_limit   * yaw_delta

                # 以预缓存的中心像素-相机方向为基，应用偏航/俯仰微偏移
                rotated_dir = apply_offsets_torch(cache.base_dir_t, yaw_offset, pitch_offset)  # (3,)
                # 投影到像素坐标（保持与原始流程相同）
                horizontal = torch.atan2(rotated_dir[1], rotated_dir[0])
                vertical   = torch.atan2(rotated_dir[2], torch.sqrt(rotated_dir[0]*rotated_dir[0] + rotated_dir[1]*rotated_dir[1]))
                u = torch.tan(horizontal) / tan_half_h_t
                v = torch.tan(vertical)   / tan_half_v_t
                u = torch.clamp(u, -1.0, 1.0)
                v = torch.clamp(v, -1.0, 1.0)
                col = ((u + 1.0) * 0.5) * dataset.width  - 0.5
                row = (1.0 - (v + 1.0) * 0.5) * dataset.height - 0.5

                # ---- clearance_center：对预缓存的 clr_t 做一次双线性插值 ----
                # 构造四邻域索引（Clamp 到边界）
                H, W = dataset.height, dataset.width
                row0 = torch.clamp(torch.floor(row), 0, H - 1)
                row1 = torch.clamp(row0 + 1,         0, H - 1)
                col0 = torch.clamp(torch.floor(col), 0, W - 1)
                col1 = torch.clamp(col0 + 1,         0, W - 1)
                rf = row - row0
                cf = col - col0

                def _bilinear(sample_2d: torch.Tensor) -> torch.Tensor:
                    s00 = sample_2d[row0.long(), col0.long()]
                    s01 = sample_2d[row0.long(), col1.long()]
                    s10 = sample_2d[row1.long(), col0.long()]
                    s11 = sample_2d[row1.long(), col1.long()]
                    return (
                        s00 * (1.0 - rf) * (1.0 - cf)
                        + s01 * (1.0 - rf) * cf
                        + s10 * rf * (1.0 - cf)
                        + s11 * rf * cf
                    )

                clearance_center = _bilinear(cache.clr_t)

                # ---- 世界坐标系下的安全射线评估 ----
                world_direction = cache.rotation_t.matmul(rotated_dir)
                world_direction = world_direction / torch.linalg.norm(world_direction)
                safety_ray = compute_ray_safety(world_direction, cache.origin_t, cache.centers_t, cache.radii_t, max_range)
                safety_score = torch.clamp(safety_ray / max_range, 0.0, 1.0)

                # ---- 目标性（偏离中心像素的距离）----
                goal_distance = torch.sqrt((col - cache.center_col_t) ** 2 + (row - cache.center_row_t) ** 2)
                goal_score    = torch.exp(-(goal_distance / diag_t))

                # ---- 稳定性/平滑/速度一致性（保持原逻辑）----
                stability_penalty = (
                    torch.abs(length_scale - 1.0) / 0.2
                    + torch.abs(pitch_offset) / pitch_limit
                    + torch.abs(yaw_offset)   / yaw_limit
                ) / 3.0
                stability_score  = torch.exp(-stability_penalty)

                smoothness_penalty = torch.sqrt((yaw_offset / yaw_limit) ** 2 + (pitch_offset / pitch_limit) ** 2)
                smoothness_score   = torch.exp(-smoothness_penalty)

                commanded_speed = speed * length_scale
                speed_penalty   = torch.abs(commanded_speed - speed) / torch.clamp(speed, min=1.0)
                speed_score     = torch.exp(-speed_penalty)

                # ---- YOPO 方向采样，用于 jerk / orientation / clearance 多点评估 ----
                dirs = sample_yopo_directions(
                    cache.base_dir_np,
                    float(yaw_offset.detach().cpu().numpy()),
                    float(pitch_offset.detach().cpu().numpy()),
                    0.0,
                    6,
                )
                points = [np.zeros(3, dtype=np.float32)]
                for direction in dirs:
                    points.append(points[-1] + direction)

                # 从预缓存的 clr_np 读取沿路线的最小 clearance 值（无需重新 distanceTransform）
                clr_vals: List[float] = []
                for direction in dirs:
                    col_dir, row_dir = project_direction_to_pixel(direction, dataset.width, dataset.height, fov_deg)
                    ci, ri = int(round(col_dir)), int(round(row_dir))
                    if 0 <= ci < dataset.width and 0 <= ri < dataset.height:
                        clr_vals.append(float(cache.clr_np[ri, ci]))
                    else:
                        clr_vals.append(0.0)
                if clr_vals:
                    min_clearance_val = max(0.0, min(clr_vals))
                else:
                    min_clearance_val = float(clearance_center.detach().cpu().item())
                clearance_score = torch.tensor(min(1.0, max(0.0, min_clearance_val)), device=device, dtype=torch.float32)

                # jerk / orientation（与原逻辑一致）
                jerk_metric_val = jerk_score(points, primitive_dt)
                jerk_score_t    = torch.tensor(float(max(0.0, min(1.0, jerk_metric_val))), device=device, dtype=torch.float32)
                orientation_metric_val = orientation_rate_score(dirs)
                orientation_score_t    = torch.tensor(float(max(0.0, min(1.0, orientation_metric_val))), device=device, dtype=torch.float32)

                # ---- 奖励 & 损失（与原脚本权重一致）----
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

                # 反传（AMP）
                scaler.scale(loss).backward()
                batch_loss_accum += float(loss.detach().cpu().item())
                valid_samples += 1

                # 统计指标（与原脚本一致）
                metrics_acc["safety"]      += float(safety_score.detach().cpu())
                metrics_acc["clearance"]   += float(clearance_score.detach().cpu())
                metrics_acc["goal"]        += float(goal_score.detach().cpu())
                metrics_acc["stability"]   += float(stability_score.detach().cpu())
                metrics_acc["speed"]       += float(speed_score.detach().cpu())
                metrics_acc["jerk"]        += float(jerk_metric_val)
                metrics_acc["orientation"] += float(orientation_metric_val)
                metrics_acc["smoothness"]  += path_smoothness(points)
                epoch_count += 1

            if valid_samples == 0:
                # 该 batch 没有有效样本
                scaler.step(optimizer)  # no-op，但保持步进一致
                scaler.update()
                continue

            # 优化器步进
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += batch_loss_accum

        # ---- 轮次日志 ----
        if epoch_count:
            avg_loss = epoch_loss / epoch_count
            averaged_metrics = {k: v / epoch_count for k, v in metrics_acc.items()}
            print(
                "Policy epoch {}/{} - avg loss: {:.4f}, safety: {:.3f}, clearance: {:.3f}, goal: {:.3f}, "
                "smoothness: {:.3f}, jerk: {:.3f}, orientation: {:.3f}, stability: {:.3f}, speed: {:.3f}".format(
                    epoch + 1, epochs, avg_loss,
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
            print(f"Policy epoch {epoch + 1}/{epochs} - no valid samples")

    policy_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), policy_output)
    print(f"已保存导航策略到 {policy_output}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练安全导航策略（安全基元偏航，提速版）")
    parser.add_argument("dataset", type=pathlib.Path, help="包含 *.npz 样本的目录")
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--noise", type=float, default=0.03, help="添加到安全掩码的噪声率")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.home() / "autonomy_demo" / "navigation_policy.pt",
        help="策略模型输出路径",
    )
    parser.add_argument("--no-amp", action="store_true", help="禁用 AMP（混合精度）")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile 加速")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    nav_dataset = NavigationDataset(args.dataset)
    print(f"数据集大小: {len(nav_dataset)}")
    print(f"图像尺寸: {nav_dataset.height} x {nav_dataset.width}")

    train_navigation_policy(
        nav_dataset,
        device,
        args.epochs,
        args.batch,
        args.lr,
        args.noise,
        args.output,
        use_amp=not args.no_amp,
        try_compile=not args.no_compile,
    )


if __name__ == "__main__":
    main()
