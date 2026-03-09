"""
test.py

Evaluates a trained MLP model on the MNIST test set.

Loads the saved checkpoint from disk, runs inference over the test
DataLoader, and prints loss and accuracy.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mlp import MLP, MLPConfig
from parameters import ModelParams, TrainParams


# Core Evaluation
@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the test set (no gradient computation).

    Args:
        model:       Trained neural network.
        test_loader: DataLoader for the test split.
        device:      Compute device.

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


# Load & Test Entry Point
def load_and_test(
    model_params: ModelParams,
    train_params: TrainParams,
    test_loader: DataLoader,
) -> None:
    """
    Loads the best saved checkpoint and evaluates on the test set.

    Args:
        model_params: ModelParams dataclass (defines architecture).
        train_params: TrainParams dataclass (provides save_path and device).
        test_loader:  DataLoader for the test split.
    """

    config = MLPConfig(
        input_size=model_params.input_size,
        hidden_sizes=model_params.hidden_sizes,
        num_classes=model_params.num_classes,
        activation=model_params.activation,
        dropout=model_params.dropout,
        use_bn=model_params.use_bn,
    )

    model = MLP(config)
    device = torch.device(train_params.device)

    model.load_state_dict(
        torch.load(train_params.save_path, map_location=device)
    )

    loss, acc = test(model, test_loader, device)

    print(f"\n{'='*40}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"{'='*40}\n")
