"""
utils/visualize.py

Visualization utilities for MNIST MLP classification.

Provides:
  - plot_history:          training/validation loss and accuracy curves
  - plot_tsne:             t-SNE projection of learned feature representations
  - plot_confusion_matrix: per-class prediction heatmap
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


MNIST_CLASSES = [str(i) for i in range(10)]


# ======================
# Training Curves
# ======================

def plot_history(
    history: Dict[str, List[float]],
    save_path: str = "training_curves.png",
) -> None:
    """
    Plots train/val loss and validation accuracy curves and saves to disk.

    Args:
        history:   Dict with 'train_loss', 'val_loss', 'val_acc' lists.
        save_path: File path for the saved figure.
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Training curves saved to '{save_path}'")
    plt.show()


# ======================
# t-SNE
# ======================

@torch.no_grad()
def plot_tsne(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str = "tsne.png",
    n_samples: int = 2000,
) -> None:
    """
    Extracts features from the last hidden layer and plots t-SNE projection.

    Samples up to n_samples points for speed. Each point is coloured
    by its true digit class (0-9).

    Args:
        model:     Trained MLP model with .flatten and .hidden_layers attributes.
        loader:    DataLoader to extract features from.
        device:    Compute device.
        save_path: File path for the saved figure.
        n_samples: Maximum number of samples to project.
    """

    model.eval()
    model = model.to(device)

    all_features: List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []

    for data, target in loader:
        data = data.to(device)

        # Pass through flatten + all hidden layers (skip final classifier)
        x = model.flatten(data)
        for layer in model.hidden_layers:
            x = layer(x)

        all_features.append(x.cpu().numpy())
        all_labels.append(target.numpy())

        if sum(len(f) for f in all_features) >= n_samples:
            break

    features = np.concatenate(all_features)[:n_samples]
    labels   = np.concatenate(all_labels)[:n_samples]

    print("  Running t-SNE (this may take a moment)...")
    embeddings = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="tab10",
        s=6,
        alpha=0.7,
    )
    plt.colorbar(scatter, ax=ax, ticks=range(10), label="Digit class")
    ax.set_title("t-SNE of learned representations")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  t-SNE plot saved to '{save_path}'")
    plt.show()


# ======================
# Confusion Matrix
# ======================

@torch.no_grad()
def plot_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str = "confusion_matrix.png",
    num_classes: int = 10,
) -> None:
    """
    Computes and plots the confusion matrix as a heatmap.

    Rows = true labels, columns = predicted labels.

    Args:
        model:       Trained MLP model.
        loader:      DataLoader to evaluate on.
        device:      Compute device.
        save_path:   File path for the saved figure.
        num_classes: Number of output classes.
    """

    model.eval()
    model = model.to(device)

    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        preds = model(data).argmax(dim=1)
        for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
            matrix[t][p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(MNIST_CLASSES)
    ax.set_yticklabels(MNIST_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() * 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Confusion matrix saved to '{save_path}'")
    plt.show()
