"""
test.py

Evaluation and testing functions for HW1b.

Features:
  - Model evaluation on test set
  - Confusion matrix generation
  - Per-class accuracy analysis
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the test set.

    Args:
        model: Trained neural network.
        test_loader: DataLoader for the test split.
        device: Compute device.

    Returns:
        Tuple of (average_loss, accuracy).
    """

    model.eval()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        total_loss += criterion(output, target).item()

        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def test_with_details(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """
    Evaluates model with detailed metrics including per-class accuracy.

    Args:
        model: Trained neural network.
        test_loader: DataLoader for the test split.
        device: Compute device.
        num_classes: Number of classes.

    Returns:
        Dictionary with detailed metrics.
    """

    model.eval()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class statistics
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    # Confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # All predictions and targets
    all_preds = []
    all_targets = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        total_loss += criterion(output, target).item()

        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        # Update confusion matrix
        for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
            confusion_matrix[t, p] += 1
            class_correct[t] += (t == p)
            class_total[t] += 1

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    # Per-class accuracy
    class_accuracy = class_correct / (class_total + 1e-8)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "class_accuracy": class_accuracy,
        "confusion_matrix": confusion_matrix,
        "predictions": np.array(all_preds),
        "targets": np.array(all_targets),
    }


def print_test_results(
    results: Dict[str, Any],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Prints test results in a formatted way.

    Args:
        results: Dictionary from test_with_details().
        class_names: List of class names (optional).
    """

    num_classes = len(results["class_accuracy"])

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    print(f"\n{'='*80}")
    print(f"Test Results")
    print(f"{'='*80}")
    print(f"Test Loss    : {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"{'='*80}")

    print(f"\nPer-Class Accuracy:")
    print(f"{'-'*80}")
    for i, (name, acc) in enumerate(zip(class_names, results["class_accuracy"])):
        print(f"  {name:>15}: {acc:.4f} ({acc*100:>6.2f}%)")
    print(f"{'-'*80}\n")


def load_and_test(
    model: nn.Module,
    checkpoint_path: str,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Loads a checkpoint and evaluates on test set with detailed metrics.

    Args:
        model: Model architecture (uninitialized weights).
        checkpoint_path: Path to model checkpoint.
        test_loader: DataLoader for test split.
        device: Compute device.
        num_classes: Number of classes.
        class_names: List of class names (optional).

    Returns:
        Dictionary with test metrics.
    """

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    print(f"\nLoaded checkpoint from: {checkpoint_path}")

    # Evaluate
    results = test_with_details(model, test_loader, device, num_classes)

    # Print results
    print_test_results(results, class_names)

    return results
