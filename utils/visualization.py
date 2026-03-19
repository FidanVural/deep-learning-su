"""
utils/visualization.py

Visualization utilities for HW1b experiments.

Provides functions for:
  - Training/validation loss and accuracy curves
  - Confusion matrix heatmaps
  - t-SNE feature embedding visualization
  - Model comparison bar charts

Uses matplotlib and scikit-learn (TSNE).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots training and validation loss/accuracy curves.

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'val_acc', 'lr'.
        title: Plot title.
        save_path: If provided, saves the figure to this path instead of showing.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#F44336")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#4CAF50")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, history["lr"], label="Learning Rate", color="#FF9800")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> None:
    """
    Plots a confusion matrix as a heatmap.

    Args:
        confusion_matrix: (num_classes, num_classes) array.
        class_names: List of class names for axis labels.
        title: Plot title.
        save_path: If provided, saves the figure.
        normalize: If True, normalizes rows to show percentages.
    """

    num_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        cm_display = confusion_matrix.astype(float) / (row_sums + 1e-8) * 100
        fmt = ".1f"
        cbar_label = "Percentage (%)"
    else:
        cm_display = confusion_matrix.astype(float)
        fmt = ".0f"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_display.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, f"{cm_display[i, j]:{fmt}}",
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts feature embeddings from the model's penultimate layer.

    Requires the model to have a `get_feature_maps` method.

    Args:
        model: Trained neural network with get_feature_maps() method.
        loader: DataLoader to extract features from.
        device: Compute device.

    Returns:
        Tuple of (features array of shape (N, D), labels array of shape (N,)).
    """

    model.eval()
    model = model.to(device)

    all_features = []
    all_labels = []

    for data, target in loader:
        data = data.to(device)
        features = model.get_feature_maps(data)
        all_features.append(features.cpu().numpy())
        all_labels.append(target.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "t-SNE Visualization",
    save_path: Optional[str] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    max_samples: int = 5000,
    seed: int = 42,
) -> None:
    """
    Generates a t-SNE 2D visualization of feature embeddings.

    Args:
        features: Feature array of shape (N, D).
        labels: Label array of shape (N,).
        class_names: List of class names for the legend.
        title: Plot title.
        save_path: If provided, saves the figure.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of t-SNE iterations.
        max_samples: Maximum samples to use (t-SNE is O(N^2)).
        seed: Random seed for reproducibility.
    """

    num_classes = len(np.unique(labels))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Subsample if too many points
    if len(features) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    print(f"Running t-SNE on {len(features)} samples...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.get_cmap("tab10", num_classes)

    for class_idx in range(num_classes):
        mask = labels == class_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[cmap(class_idx)],
            label=class_names[class_idx],
            alpha=0.6,
            s=10,
        )

    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_model_comparison(
    model_names: List[str],
    accuracies: List[float],
    flops: Optional[List[float]] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing model accuracy (and optionally FLOPs).

    Args:
        model_names: List of model names.
        accuracies: Corresponding accuracy values (0-1).
        flops: Optional list of FLOPs (in millions) for each model.
        title: Plot title.
        save_path: If provided, saves the figure.
    """

    n = len(model_names)

    if flops is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    bars = ax1.bar(model_names, [a * 100 for a in accuracies], color=colors)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Test Accuracy")
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc * 100:.1f}%",
            ha="center", fontsize=10,
        )
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    if flops is not None:
        bars2 = ax2.bar(model_names, flops, color=colors)
        ax2.set_ylabel("FLOPs (M)")
        ax2.set_title("Computational Cost")
        for bar, f in zip(bars2, flops):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(flops) * 0.02,
                f"{f:.1f}M",
                ha="center", fontsize=10,
            )
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def load_and_plot_history(
    history_path: str,
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> None:
    """
    Loads a history.json file and plots training curves.

    Args:
        history_path: Path to history.json.
        title: Plot title.
        save_path: If provided, saves the figure.
    """

    with open(history_path, "r") as f:
        history = json.load(f)

    plot_training_history(history, title=title, save_path=save_path)
