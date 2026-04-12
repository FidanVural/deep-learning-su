"""
utils/__init__.py

Utility modules for HW2: Data Augmentation and Adversarial Robustness.
"""

from .augmix import AugMixTransform, get_augmix_dataloaders
from .cifar10c import load_cifar10c, CIFAR10C_CORRUPTIONS
from .pgd_attack import PGDAttack
from .gradcam import GradCAM, visualize_gradcam
from .visualization import (
    plot_training_history,
    plot_corruption_results,
    plot_adversarial_comparison,
    plot_tsne_adversarial,
    plot_gradcam_comparison,
)

__all__ = [
    "AugMixTransform",
    "get_augmix_dataloaders",
    "load_cifar10c",
    "CIFAR10C_CORRUPTIONS",
    "PGDAttack",
    "GradCAM",
    "visualize_gradcam",
    "plot_training_history",
    "plot_corruption_results",
    "plot_adversarial_comparison",
    "plot_tsne_adversarial",
    "plot_gradcam_comparison",
]
