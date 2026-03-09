"""
train.py

Training loop for MNIST MLP classification.

Features:
  - Per-epoch train / validation evaluation
  - Early stopping based on validation loss
  - LR scheduling (StepLR, CosineAnnealingLR, ReduceLROnPlateau)
  - Optional L1 regularization (L2 handled by optimizer weight_decay)
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from parameters import TrainParams


# Helpers
def _get_optimizer(model: nn.Module, params: TrainParams) -> torch.optim.Optimizer:
    """
    Builds AdamW optimizer with L2 weight decay from TrainParams.

    Args:
        model:  The neural network.
        params: Training configuration.

    Returns:
        Configured AdamW optimizer.
    """

    return torch.optim.AdamW(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )


def _get_scheduler(
    optimizer: torch.optim.Optimizer,
    params: TrainParams,
) -> Optional[Union[torch.optim.lr_scheduler.LRScheduler, ReduceLROnPlateau]]:
    """
    Builds an LR scheduler based on TrainParams.scheduler.

    Supported options: 'step', 'cosine', 'plateau', 'none'.

    Note: 'plateau' returns a ReduceLROnPlateau which requires
    scheduler.step(val_loss) instead of scheduler.step().

    Args:
        optimizer: The optimizer to wrap.
        params:    Training configuration.

    Returns:
        Scheduler instance, or None if scheduler == 'none'.
    """

    if params.scheduler == "step":
        return StepLR(optimizer, step_size=params.step_size, gamma=0.5)
    elif params.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=params.epochs)
    elif params.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    return None


def _l1_loss(model: nn.Module, l1_lambda: float) -> torch.Tensor:
    """
    Computes L1 regularization penalty over all model parameters.

    Args:
        model:      The neural network.
        l1_lambda:  Regularization coefficient.

    Returns:
        Scalar L1 penalty tensor.
    """

    return l1_lambda * sum(p.abs().sum() for p in model.parameters())


# Single Epoch
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    l1_lambda: float = 0.0,
    log_interval: int = 100,
) -> float:
    """
    Runs one full training epoch.

    Args:
        model:        The neural network (in train mode after this call).
        loader:       Training DataLoader.
        optimizer:    Optimizer.
        criterion:    Loss function.
        device:       Compute device.
        l1_lambda:    L1 regularization coefficient (0 = disabled).
        log_interval: Print loss every N batches.

    Returns:
        Average training loss over the epoch.
    """

    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if l1_lambda > 0.0:
            loss = loss + _l1_loss(model, l1_lambda)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f"    Batch [{batch_idx:>4}/{len(loader)}]  loss: {loss.item():.4f}")

    return total_loss / len(loader)


# Evaluation
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the given DataLoader (no gradient computation).

    Args:
        model:     The neural network.
        loader:    DataLoader to evaluate on.
        criterion: Loss function.
        device:    Compute device.

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
) -> Dict[str, List[float]]:
    """
    Full training loop with early stopping and LR scheduling.

    Saves the best model (lowest val loss) to params.save_path.

    Args:
        model:        The neural network to train.
        train_loader: DataLoader for training data.
        val_loader:   DataLoader for validation data.
        params:       TrainParams dataclass with all training settings.

    Returns:
        History dict with keys 'train_loss', 'val_loss', 'val_acc'.
    """

    device = torch.device(params.device)
    model = model.to(device)

    optimizer = _get_optimizer(model, params)
    scheduler = _get_scheduler(optimizer, params)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch [{epoch}/{params.epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            l1_lambda=params.l1_lambda,
            log_interval=params.log_interval,
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc:  {val_acc:.4f}"
        )

        # --- Save best model / Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), params.save_path)
            print(f"  --> Best model saved  (val_loss = {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  --> No improvement ({patience_counter}/{params.patience})")
            if params.patience > 0 and patience_counter >= params.patience:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    return history
