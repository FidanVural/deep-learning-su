"""
utils/__init__.py

Utility functions for HW1b: Transfer Learning and Knowledge Distillation.
"""

from .data_cifar import DataConfig, get_dataloaders
from .label_smoothing import LabelSmoothingCrossEntropy
from .distillation import DistillationLoss, distillation_loss
from .flops_counter import count_flops, compare_model_complexity
from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    extract_features,
    plot_tsne,
    plot_model_comparison,
    load_and_plot_history,
)

__all__ = [
    "DataConfig",
    "get_dataloaders",
    "LabelSmoothingCrossEntropy",
    "DistillationLoss",
    "distillation_loss",
    "count_flops",
    "compare_model_complexity",
    "plot_training_history",
    "plot_confusion_matrix",
    "extract_features",
    "plot_tsne",
    "plot_model_comparison",
    "load_and_plot_history",
]
