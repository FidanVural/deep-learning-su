"""
utils/visualization.py

Visualization utilities for HW2 experiments.

Provides functions for:
  - Training/validation curves
  - Corruption robustness bar charts
  - Adversarial accuracy comparison
  - t-SNE visualization of clean vs adversarial embeddings
  - Grad-CAM comparison (clean vs adversarial)
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


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


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
        save_path: If provided, saves the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#F44336")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#4CAF50")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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
        print(f"Training history saved to: {save_path}")

    plt.close(fig)


def plot_corruption_results(
    results: Dict[str, Dict[str, float]],
    title: str = "CIFAR-10-C Corruption Robustness",
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing model accuracy across all CIFAR-10-C corruptions.

    Args:
        results: Dict mapping model name to {corruption_name: accuracy} dict.
        title: Plot title.
        save_path: If provided, saves the figure.
    """
    model_names = list(results.keys())
    corruptions = [k for k in list(results.values())[0].keys() if k != "mean"]

    x = np.arange(len(corruptions))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        accs = [results[model_name].get(c, 0.0) * 100 for c in corruptions]
        bars = ax.bar(x + i * width, accs, width, label=model_name, color=color)

    ax.set_xlabel("Corruption Type")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(corruptions, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    mean_text = " | ".join(
        f"{name}: {results[name].get('mean', 0.0)*100:.1f}%"
        for name in model_names
    )
    ax.text(0.5, -0.22, f"Mean: {mean_text}", transform=ax.transAxes, ha="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Corruption results saved to: {save_path}")

    plt.close(fig)


def plot_adversarial_comparison(
    results: Dict[str, Dict[str, float]],
    title: str = "Adversarial Robustness Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart comparing clean vs adversarial accuracy for multiple models.

    Args:
        results: Dict mapping model name -> {metric_name: value}.
                 Expected metrics: 'clean_acc', 'adv_acc_linf', 'adv_acc_l2'.
        title: Plot title.
        save_path: If provided, saves the figure.
    """
    model_names = list(results.keys())
    metrics = ["clean_acc", "adv_acc_linf", "adv_acc_l2"]
    metric_labels = ["Clean", r"PGD-20 $L_\infty$", r"PGD-20 $L_2$"]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4CAF50", "#F44336", "#FF9800"]

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[m].get(metric, 0.0) * 100 for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=8,
            )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Adversarial comparison saved to: {save_path}")

    plt.close(fig)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extracts feature embeddings from the model's penultimate layer.

    Args:
        model: Trained model with get_feature_maps() method.
        data: Input tensor of shape (N, C, H, W).
        device: Compute device.
        batch_size: Batch size for extraction.

    Returns:
        Feature array of shape (N, D).
    """
    model.eval()
    model = model.to(device)

    all_features = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size].to(device)
        features = model.get_feature_maps(batch)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def plot_tsne_adversarial(
    clean_features: np.ndarray,
    adv_features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "t-SNE: Clean vs Adversarial",
    save_path: Optional[str] = None,
    max_samples: int = 3000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
) -> None:
    """
    Generates t-SNE visualization comparing clean and adversarial embeddings.

    Clean samples are plotted as filled circles, adversarial samples as X markers.

    Args:
        clean_features: Clean sample features (N, D).
        adv_features: Adversarial sample features (N, D).
        labels: True class labels (N,).
        class_names: List of class names for legend.
        title: Plot title.
        save_path: If provided, saves the figure.
        max_samples: Maximum number of samples to plot.
        perplexity: t-SNE perplexity.
        n_iter: t-SNE iterations.
        seed: Random seed.
    """
    num_classes = len(np.unique(labels))
    if class_names is None:
        class_names = CIFAR10_CLASSES[:num_classes]

    if len(clean_features) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(clean_features), max_samples, replace=False)
        clean_features = clean_features[idx]
        adv_features = adv_features[idx]
        labels = labels[idx]

    combined = np.concatenate([clean_features, adv_features], axis=0)
    n = len(clean_features)

    print(f"Running t-SNE on {len(combined)} samples (clean + adversarial)...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    embeddings = tsne.fit_transform(combined)

    clean_emb = embeddings[:n]
    adv_emb = embeddings[n:]

    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = plt.cm.get_cmap("tab10", num_classes)

    for cls in range(num_classes):
        mask = labels == cls
        ax.scatter(
            clean_emb[mask, 0], clean_emb[mask, 1],
            c=[cmap(cls)], marker="o", alpha=0.5, s=15,
            label=f"{class_names[cls]} (clean)",
        )
        ax.scatter(
            adv_emb[mask, 0], adv_emb[mask, 1],
            c=[cmap(cls)], marker="x", alpha=0.5, s=15,
            label=f"{class_names[cls]} (adv)",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(
        markerscale=2, fontsize=7, loc="center left",
        bbox_to_anchor=(1.0, 0.5), ncol=1,
    )
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE adversarial plot saved to: {save_path}")

    plt.close(fig)


def plot_gradcam_comparison(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    clean_cams: List[np.ndarray],
    adv_cams: List[np.ndarray],
    labels: torch.Tensor,
    clean_preds: torch.Tensor,
    adv_preds: torch.Tensor,
    class_names: Optional[List[str]] = None,
    title: str = "Grad-CAM: Clean vs Adversarial",
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side Grad-CAM comparison between clean and adversarial samples.

    Args:
        clean_images: Clean images (N, C, H, W), normalized.
        adv_images: Adversarial images (N, C, H, W), normalized.
        clean_cams: List of Grad-CAM heatmaps for clean images.
        adv_cams: List of Grad-CAM heatmaps for adversarial images.
        labels: True labels (N,).
        clean_preds: Predictions on clean images (N,).
        adv_preds: Predictions on adversarial images (N,).
        class_names: List of class names.
        title: Plot title.
        save_path: If provided, saves the figure.
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    num_samples = len(clean_images)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_samples):
        clean_img = (clean_images[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        adv_img = (adv_images[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        true_lbl = labels[i].item()
        clean_pred = clean_preds[i].item()
        adv_pred = adv_preds[i].item()

        axes[i, 0].imshow(clean_img)
        axes[i, 0].imshow(clean_cams[i], cmap="jet", alpha=0.4)
        axes[i, 0].set_title(f"Clean\nTrue: {class_names[true_lbl]}\nPred: {class_names[clean_pred]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(clean_cams[i], cmap="jet")
        axes[i, 1].set_title("Clean CAM", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(adv_img)
        axes[i, 2].imshow(adv_cams[i], cmap="jet", alpha=0.4)
        axes[i, 2].set_title(f"Adversarial\nTrue: {class_names[true_lbl]}\nPred: {class_names[adv_pred]}", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(adv_cams[i], cmap="jet")
        axes[i, 3].set_title("Adv CAM", fontsize=9)
        axes[i, 3].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM comparison saved to: {save_path}")

    plt.close(fig)
