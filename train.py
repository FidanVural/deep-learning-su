"""
train.py

Training loops for HW1b: Transfer Learning and Knowledge Distillation.

Features:
  - Standard training loop (baseline, transfer learning)
  - Knowledge distillation training loop
  - Learning rate scheduling with warmup
  - Early stopping
  - Label smoothing support
"""

from typing import Dict, List, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LambdaLR,
)

from parameters import TrainParams
from utils.label_smoothing import LabelSmoothingCrossEntropy
from utils.distillation import DistillationLoss, CustomDistillationLoss


# Optimizer Builder
def _get_optimizer(
    model: nn.Module,
    params: TrainParams,
) -> torch.optim.Optimizer:
    """
    Builds optimizer from TrainParams.

    Args:
        model: The neural network.
        params: Training configuration.

    Returns:
        Configured optimizer.
    """

    if params.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
        )
    elif params.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=params.lr,
            momentum=params.momentum,
            weight_decay=params.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {params.optimizer}")


# LR Scheduler Builder
def _get_scheduler(
    optimizer: torch.optim.Optimizer,
    params: TrainParams,
) -> Optional[Union[torch.optim.lr_scheduler.LRScheduler, ReduceLROnPlateau]]:
    """
    Builds learning rate scheduler from TrainParams.

    Args:
        optimizer: The optimizer to wrap.
        params: Training configuration.

    Returns:
        Scheduler instance or None.
    """

    if params.scheduler == "step":
        return StepLR(optimizer, step_size=params.step_size, gamma=0.1)
    elif params.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=params.epochs)
    elif params.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
    elif params.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {params.scheduler}")


# Warmup Scheduler
def _get_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
) -> LambdaLR:
    """
    Creates a warmup learning rate scheduler.

    Args:
        optimizer: The optimizer.
        warmup_epochs: Number of warmup epochs.

    Returns:
        LambdaLR scheduler for warmup.
    """

    def warmup_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    return LambdaLR(optimizer, lr_lambda=warmup_lambda)


# Loss Function Builder
def _get_criterion(params: TrainParams) -> nn.Module:
    """
    Builds loss function from TrainParams.

    Args:
        params: Training configuration.

    Returns:
        Loss function (CrossEntropyLoss or LabelSmoothingCrossEntropy).
    """

    if params.label_smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=params.label_smoothing)
    else:
        return nn.CrossEntropyLoss()


# Single Training Epoch
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int = 50,
    epoch: int = 0,
) -> float:
    """
    Runs one training epoch.

    Args:
        model: The neural network.
        loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Compute device.
        log_interval: Print loss every N batches.
        epoch: Current epoch number.

    Returns:
        Average training loss over the epoch.
    """

    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx:>4}/{num_batches}]  "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / num_batches


# Single Training Epoch with Distillation
def train_one_epoch_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    distillation_criterion: nn.Module,
    device: torch.device,
    log_interval: int = 50,
    epoch: int = 0,
) -> float:
    """
    Runs one training epoch with knowledge distillation.

    Args:
        student_model: Student network to train.
        teacher_model: Teacher network (frozen).
        loader: Training DataLoader.
        optimizer: Optimizer for student.
        distillation_criterion: Distillation loss function.
        device: Compute device.
        log_interval: Print loss every N batches.
        epoch: Current epoch number.

    Returns:
        Average training loss over the epoch.
    """

    student_model.train()
    teacher_model.eval()
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Student forward pass
        student_output = student_model(data)

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_output = teacher_model(data)

        # Distillation loss
        loss = distillation_criterion(student_output, teacher_output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx:>4}/{num_batches}]  "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / num_batches


# Evaluation
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the given DataLoader.

    Args:
        model: The neural network.
        loader: DataLoader to evaluate on.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (average_loss, accuracy).
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        total_loss += criterion(output, target).item()

        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


# Full Training Loop
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: TrainParams,
    teacher_model: Optional[nn.Module] = None,
) -> Dict[str, List[float]]:
    """
    Full training loop with early stopping and LR scheduling.

    Supports both standard training and knowledge distillation.

    Args:
        model: The neural network to train (student if distillation).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        params: TrainParams with all training settings.
        teacher_model: Optional teacher model for knowledge distillation.

    Returns:
        History dict with training metrics.
    """

    device = torch.device(params.device)
    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = _get_optimizer(model, params)
    scheduler = _get_scheduler(optimizer, params)
    warmup_scheduler = _get_warmup_scheduler(optimizer, params.warmup_epochs)

    # Setup loss function
    if params.use_distillation and teacher_model is not None:
        # Knowledge distillation
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

        if params.use_custom_distillation:
            criterion = CustomDistillationLoss(
                temperature=params.temperature,
                alpha=params.alpha,
            )
        else:
            criterion = DistillationLoss(
                temperature=params.temperature,
                alpha=params.alpha,
            )

        eval_criterion = nn.CrossEntropyLoss()
    else:
        # Standard training
        criterion = _get_criterion(params)
        eval_criterion = criterion

    # Training history
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {params.epochs}")
    print(f"Optimizer: {params.optimizer}")
    print(f"Learning Rate: {params.lr}")
    print(f"Scheduler: {params.scheduler}")
    print(f"Label Smoothing: {params.label_smoothing}")
    print(f"Distillation: {params.use_distillation}")
    if params.use_distillation:
        print(f"  Temperature: {params.temperature}")
        print(f"  Alpha: {params.alpha}")
    print(f"{'='*80}\n")

    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch [{epoch}/{params.epochs}]")
        print(f"{'-'*80}")

        # Training
        if params.use_distillation and teacher_model is not None:
            train_loss = train_one_epoch_distillation(
                student_model=model,
                teacher_model=teacher_model,
                loader=train_loader,
                optimizer=optimizer,
                distillation_criterion=criterion,
                device=device,
                log_interval=params.log_interval,
                epoch=epoch,
            )
        else:
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                log_interval=params.log_interval,
                epoch=epoch,
            )

        # Validation
        if epoch % params.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, eval_criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Update schedulers
        if epoch <= params.warmup_epochs:
            warmup_scheduler.step()
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Print progress
        print(f"{'-'*80}")
        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%) | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model / Early stopping
        if val_loss < best_val_loss and epoch % params.eval_interval == 0:
            best_val_loss = val_loss
            patience_counter = 0

            if params.save_best_only:
                os.makedirs(os.path.dirname(params.save_path), exist_ok=True)
                torch.save(model.state_dict(), params.save_path)
                print(f"  --> Best model saved (val_loss = {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter > 0:
                print(f"  --> No improvement ({patience_counter}/{params.patience})")

            if params.patience > 0 and patience_counter >= params.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")

    return history
