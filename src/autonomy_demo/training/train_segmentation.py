#!/usr/bin/env python3
"""Standalone segmentation trainer for the safety classifier."""

import argparse
import pathlib
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Reuse the core implementations from the primary training script.
from autonomy_demo.training.train_classifier import (
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
    """Train the RGB-only segmentation network."""

    class_weights: Optional[torch.Tensor] = None
    if isinstance(train_loader.dataset, ObstacleDataset):
        class_weights = train_loader.dataset.estimate_class_weights()
    elif isinstance(train_loader.dataset, Dataset) and hasattr(train_loader.dataset, "dataset"):
        base_dataset = getattr(train_loader.dataset, "dataset")
        if isinstance(base_dataset, ObstacleDataset):
            class_weights = base_dataset.estimate_class_weights()

    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(
            f"Class weights -> safe: {class_weights[0].item():.3f}, obstacle: {class_weights[1].item():.3f}"
        )

    criterion = SegmentationLoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    kldiv = torch.nn.KLDivLoss(reduction="batchmean")

    teacher: Optional[TeacherDistanceClassifier] = None
    if use_distill:
        teacher = TeacherDistanceClassifier().to(device)
        with torch.no_grad():
            teacher_state = teacher.state_dict()
            student_state = model.state_dict()
            for name, weight in teacher_state.items():
                if name in student_state and student_state[name].shape == weight.shape:
                    teacher_state[name] = student_state[name]
            teacher.load_state_dict(teacher_state)
        teacher.train()
        teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    else:
        teacher_optimizer = None

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
                obstacle = torch.zeros(
                    (images.shape[0], 1, images.shape[2], images.shape[3]), dtype=images.dtype
                )

            images = images.to(device)
            labels = labels.to(device)
            obstacle = obstacle.to(device)
            optimizer.zero_grad()
            if teacher is not None and teacher_optimizer is not None:
                teacher_optimizer.zero_grad()

            outputs_student = model(images)
            loss_student = criterion(outputs_student, labels)
            loss = loss_student

            if teacher is not None and teacher_optimizer is not None:
                if modality_dropout_p > 0.0 and teacher.training:
                    drop_mask = (
                        torch.rand(obstacle.shape[0], device=device) < modality_dropout_p
                    ).float().view(-1, 1, 1, 1)
                    obstacle_in = obstacle * (1.0 - drop_mask)
                else:
                    obstacle_in = obstacle
                outputs_teacher = teacher(images, obstacle_in)
                loss_teacher = criterion(outputs_teacher, labels)
                logprob_student = torch.log_softmax(outputs_student, dim=1)
                prob_teacher = torch.softmax(outputs_teacher.detach(), dim=1)
                loss_distill = kldiv(logprob_student, prob_teacher)
                loss = loss_student + distill_weight * loss_distill + teacher_weight * loss_teacher

            loss.backward()
            optimizer.step()
            if teacher is not None and teacher_optimizer is not None:
                loss_teacher.backward()
                teacher_optimizer.step()
            running_loss += float(loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch  # type: ignore[misc]
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                running_val += float(val_loss.item())

        avg_val_loss = running_val / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        metrics = evaluate_classifier(model, val_loader, device)
        if metrics is None:
            print(
                f"Epoch {epoch + 1}/{epochs} - train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}"
            )
        else:
            print(
                (
                    "Epoch {}/{} - train loss: {:.4f}, val loss: {:.4f}, IoU: {:.3f}, Acc: {:.3f}, "
                    "Prec: {:.3f}, Rec: {:.3f}"
                ).format(
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
    parser = argparse.ArgumentParser(
        description="Train a binary segmentation network that separates safe regions from obstacles"
    )
    parser.add_argument("dataset", type=pathlib.Path, help="Directory with *.npz samples")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.home() / "autonomy_demo" / "segmentation_model.pt",
        help="Path to store the trained weights",
    )
    parser.add_argument("--distill", action="store_true", help="Enable teacher-student distillation")
    parser.add_argument(
        "--distill_weight", type=float, default=0.5, help="Weight applied to KL divergence"
    )
    parser.add_argument("--teacher_weight", type=float, default=0.0, help="Teacher supervision weight")
    parser.add_argument(
        "--modality_dropout",
        type=float,
        default=0.3,
        help="Probability of dropping the obstacle channel for the teacher",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    full_dataset = ObstacleDataset(args.dataset)
    total_len = len(full_dataset)
    all_indices = list(range(total_len))
    random.shuffle(all_indices)

    if total_len < 2:
        train_indices = all_indices
        val_indices: List[int] = []
    else:
        val_len = max(1, int(total_len * args.val_split))
        train_len = max(1, total_len - val_len)
        if train_len + val_len > total_len:
            val_len = total_len - train_len
        val_indices = all_indices[:val_len]
        train_indices = all_indices[val_len:]

    train_dataset = ObstacleDataset(args.dataset, indices=train_indices, augment=True)
    val_dataset = ObstacleDataset(args.dataset, indices=val_indices, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch)

    print(f"Train samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample image shape: {tuple(sample[0].shape)}, label shape: {tuple(sample[1].shape)}")
        print(
            "Label stats -> safe: {}, obstacle: {}".format(
                int((sample[1] == 0).sum().item()), int((sample[1] == 1).sum().item())
            )
        )

    model = DistanceClassifier().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    print(f"Saved segmentation model to {args.output}")


if __name__ == "__main__":
    main()
