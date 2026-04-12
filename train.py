"""
train.py

Training loops for HW2: Data Augmentation and Adversarial Robustness.

Supports:
  - Standard fine-tuning (with or without AugMix)
  - Knowledge distillation (with AugMix teacher)
  - Learning rate scheduling with warmup
  - Early stopping
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
from utils.distillation import DistillationLoss, CustomDistillationLoss
from utils.label_smoothing import LabelSmoothingCrossEntropy


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
            model.parameters(), lr=params.lr, weight_decay=params.weight_decay,
        )
    elif params.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=params.lr, momentum=params.momentum,
            weight_decay=params.weight_decay, nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {params.optimizer}")


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
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
    elif params.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {params.scheduler}")


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


def _get_criterion(params: TrainParams) -> nn.Module:
    """
    Builds loss function from TrainParams.

    Args:
        params: Training configuration.

    Returns:
        Loss function.
    """
    if params.label_smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=params.label_smoothing)
    return nn.CrossEntropyLoss()


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

        student_output = student_model(data)
        with torch.no_grad():
            teacher_output = teacher_model(data)

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
        model: Neural network to train (student if distillation).
        train_loader: Training DataLoader (may use AugMix).
        val_loader: Validation DataLoader.
        params: TrainParams with all training settings.
        teacher_model: Optional teacher model for knowledge distillation.

    Returns:
        History dict with training metrics.
    """
    device = torch.device(params.device)
    model = model.to(device)

    optimizer = _get_optimizer(model, params)
    scheduler = _get_scheduler(optimizer, params)
    warmup_scheduler = _get_warmup_scheduler(optimizer, params.warmup_epochs)

    if params.use_distillation and teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

        if params.use_custom_distillation:
            criterion = CustomDistillationLoss(
                temperature=params.temperature, alpha=params.alpha,
            )
        else:
            criterion = DistillationLoss(
                temperature=params.temperature, alpha=params.alpha,
            )
        eval_criterion = nn.CrossEntropyLoss()
    else:
        criterion = _get_criterion(params)
        eval_criterion = criterion

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
    print(f"Distillation: {params.use_distillation}")
    if params.use_distillation:
        print(f"  Temperature: {params.temperature}")
        print(f"  Alpha: {params.alpha}")
    print(f"{'='*80}\n")

    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch [{epoch}/{params.epochs}]")
        print(f"{'-'*80}")

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

        if epoch % params.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, eval_criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch <= params.warmup_epochs:
            warmup_scheduler.step()
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"{'-'*80}")
        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%) | "
            f"LR: {current_lr:.6f}"
        )

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
    print(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")

    return history
