"""
utils/label_smoothing.py

Label Smoothing Cross Entropy Loss implementation.

Label smoothing is a regularization technique that prevents the model from
becoming too confident about its predictions. It replaces hard targets (one-hot)
with soft targets by distributing a small amount of probability mass to all classes.

Reference:
  - Müller et al., "When Does Label Smoothing Help?", NeurIPS 2019
  - Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.

    For a given smoothing parameter ε (epsilon), the soft target is:
        y_smooth = (1 - ε) * y_hard + ε / K

    where y_hard is the one-hot encoded label and K is the number of classes.

    Args:
        smoothing: Smoothing parameter (0.0 = no smoothing, typical: 0.1).
        reduction: Specifies the reduction to apply to the output:
                   'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()

        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes label smoothing cross entropy loss.

        Args:
            input: Predicted logits of shape (N, C) where C is number of classes.
            target: Ground truth class indices of shape (N,).

        Returns:
            Loss value (scalar if reduction='mean' or 'sum', tensor otherwise).
        """

        if self.smoothing == 0.0:
            # No smoothing, use standard cross entropy
            return F.cross_entropy(input, target, reduction=self.reduction)

        num_classes = input.size(-1)
        log_probs = F.log_softmax(input, dim=-1)

        # Create soft targets
        # (1 - ε) for true class, ε / K for all classes
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Compute KL divergence
        loss = torch.sum(-true_dist * log_probs, dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def label_smoothing_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    smoothing: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Functional interface for label smoothing cross entropy loss.

    Args:
        input: Predicted logits of shape (N, C).
        target: Ground truth class indices of shape (N,).
        smoothing: Smoothing parameter (0.0 = no smoothing).
        reduction: 'none' | 'mean' | 'sum'.

    Returns:
        Loss value.
    """

    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing, reduction=reduction)
    return criterion(input, target)
