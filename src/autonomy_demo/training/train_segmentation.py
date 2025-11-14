#!/usr/bin/env python3
"""独立的图像分割训练脚本 - 用于区分安全区域和障碍物的二分类训练"""

import argparse
import pathlib
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 从原始训练脚本导入模型定义
import sys
import pathlib
# 添加当前目录到路径以便导入
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_classifier import (
    ConvBlock,
    DistanceClassifier,
    ObstacleDataset,
    SegmentationLoss,
    TeacherDistanceClassifier,
    evaluate_classifier,
)


def train_segmentation(
    model: DistanceClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_distill: bool = False,
    modality_dropout_p: float = 0.3,
    distill_weight: float = 0.5,
    teacher_weight: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """
    训练图像分割模型
    
    训练时：
    - 输入：RGB图像（3通道）
    - 标签：从环境障碍物信息生成的二分类标签（0=安全，1=障碍物）
    - 模型只使用RGB图像进行推理，不依赖障碍物信息
    
    推理时：
    - 只使用RGB图像，不依赖任何环境信息
    """
    class_weights: Optional[torch.Tensor] = None
    if isinstance(train_loader.dataset, ObstacleDataset):
        class_weights = train_loader.dataset.estimate_class_weights()
    elif isinstance(train_loader.dataset, Dataset) and hasattr(train_loader.dataset, "dataset"):
        base = getattr(train_loader.dataset, "dataset")
        if isinstance(base, ObstacleDataset):
            class_weights = base.estimate_class_weights()

    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"类别权重: 安全={class_weights[0]:.3f}, 障碍物={class_weights[1]:.3f}")

    criterion = SegmentationLoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    kldiv = torch.nn.KLDivLoss(reduction="batchmean")

    teacher: Optional[TeacherDistanceClassifier] = None
    if use_distill:
        teacher = TeacherDistanceClassifier().to(device)
        with torch.no_grad():
            student_state = model.state_dict()
            teacher_state = teacher.state_dict()
            for name, t_weight in teacher_state.items():
                if name in student_state and student_state[name].shape == t_weight.shape:
                    teacher_state[name] = student_state[name]
            teacher.load_state_dict(teacher_state)
        teacher.train()
        teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        model.train()
        if teacher is not None:
            teacher.train()
        running_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                images, labels, obstacle = batch
            else:
                images, labels = batch  # type: ignore[misc]
                obstacle = torch.zeros((images.shape[0], 1, images.shape[2], images.shape[3]), dtype=images.dtype)
            
            # 确保只使用RGB图像（3通道），不依赖障碍物信息
            # 注意：obstacle信息仅用于教师模型（如果启用蒸馏），学生模型只使用RGB
            images = images.to(device)
            labels = labels.to(device)
            obstacle = obstacle.to(device)
            optimizer.zero_grad()
            if teacher is not None:
                teacher_optimizer.zero_grad()

            # 学生模型：只使用RGB图像（3通道）
            outputs_student = model(images)
            loss_student = criterion(outputs_student, labels)
            loss = loss_student

            if teacher is not None:
                if modality_dropout_p > 0.0 and teacher.training:
                    drop_mask = (torch.rand(obstacle.shape[0], device=device) < modality_dropout_p).float().view(-1, 1, 1, 1)
                    obstacle_in = obstacle * (1.0 - drop_mask)
                else:
                    obstacle_in = obstacle
                outputs_teacher = teacher(images, obstacle_in)
                loss_teacher = criterion(outputs_teacher, labels)
                logprob_student = torch.log_softmax(outputs_student, dim=1)
                prob_teacher = torch.softmax(outputs_teacher.detach(), dim=1)
                loss_distill = kldiv(logprob_student, prob_teacher)
                loss = loss_student + distill_weight * loss_distill

            loss.backward()
            optimizer.step()
            if teacher is not None:
                loss_teacher.backward()
                teacher_optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, labels, _obstacle = batch
                else:
                    images, labels = batch  # type: ignore[misc]
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val += loss.item()
        
        avg_val_loss = running_val / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练图像分割模型（安全区域 vs 障碍物）")
    parser.add_argument("dataset", type=pathlib.Path, help="包含 *.npz 样本的目录")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--val_split", type=float, default=0.2, help="验证集比例")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.home() / "autonomy_demo" / "segmentation_model.pt",
        help="模型输出路径",
    )
    parser.add_argument("--distill", action="store_true", help="启用教师-学生蒸馏（使用障碍物通道）")
    parser.add_argument("--distill_weight", type=float, default=0.5, help="KL蒸馏损失权重")
    parser.add_argument("--teacher_weight", type=float, default=0.0, help="教师分支的监督损失权重")
    parser.add_argument("--modality_dropout", type=float, default=0.3, help="教师输入中丢弃障碍物通道的概率")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
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

    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    
    # 验证数据集格式
    sample = train_dataset[0]
    print(f"输入图像形状: {sample[0].shape} (应为 [3, H, W])")
    print(f"标签形状: {sample[1].shape} (应为 [H, W])")
    print(f"标签值范围: {sample[1].min().item()} - {sample[1].max().item()} (应为 0-1)")
    safe_pixels = (sample[1] == 0).sum().item()
    obstacle_pixels = (sample[1] == 1).sum().item()
    print(f"样本标签统计: 安全={safe_pixels}, 障碍物={obstacle_pixels}")

    model = DistanceClassifier().to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    train_segmentation(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        use_distill=args.distill,
        modality_dropout_p=max(0.0, min(1.0, args.modality_dropout)),
        distill_weight=max(0.0, args.distill_weight),
        teacher_weight=max(0.0, args.teacher_weight),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"已保存训练好的分割模型到 {args.output}")


if __name__ == "__main__":
    main()

