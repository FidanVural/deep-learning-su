"""
utils/distillation.py

Knowledge Distillation loss implementation.

Knowledge Distillation (KD) transfers knowledge from a large teacher model to
a smaller student model by training the student to mimic the teacher's soft
predictions (probability distributions) instead of just hard labels.

The distillation loss combines:
  1. Soft target loss: KL divergence between student and teacher soft predictions
  2. Hard target loss: Standard cross-entropy with ground truth labels

Reference:
  - Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
  - Yuan et al., "Revisiting Knowledge Distillation via Label Smoothing", CVPR 2020
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining soft and hard targets.

    Loss = α * T² * KL(softmax(student/T), softmax(teacher/T)) + (1-α) * CE(student, target)

    where:
      - T is the temperature (higher T = softer probability distributions)
      - α is the weight balancing soft and hard targets
      - KL is Kullback-Leibler divergence
      - CE is cross-entropy loss

    Args:
        temperature: Temperature for softening probability distributions (typical: 3-5).
        alpha: Weight for distillation loss (typical: 0.5-0.9).
               alpha=1.0 means only soft targets, alpha=0.0 means only hard targets.
        reduction: Reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes knowledge distillation loss.

        Args:
            student_logits: Student model logits of shape (N, C).
            teacher_logits: Teacher model logits of shape (N, C).
            targets: Ground truth class indices of shape (N,).

        Returns:
            Loss value.
        """

        # Soft targets with temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence loss (soft targets)
        # Note: KL divergence with log_softmax and softmax is more numerically stable
        distillation_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean" if self.reduction == "mean" else self.reduction,
        )

        # Scale by T^2 (as per Hinton's paper)
        distillation_loss = distillation_loss * (self.temperature ** 2)

        # Hard target loss
        student_loss = F.cross_entropy(
            student_logits,
            targets,
            reduction=self.reduction,
        )

        # Combined loss
        loss = self.alpha * distillation_loss + (1.0 - self.alpha) * student_loss

        return loss


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Functional interface for knowledge distillation loss.

    Args:
        student_logits: Student model logits of shape (N, C).
        teacher_logits: Teacher model logits of shape (N, C).
        targets: Ground truth class indices of shape (N,).
        temperature: Temperature for softening distributions.
        alpha: Weight for distillation loss.
        reduction: 'none' | 'mean' | 'sum'.

    Returns:
        Loss value.
    """

    criterion = DistillationLoss(
        temperature=temperature,
        alpha=alpha,
        reduction=reduction,
    )
    return criterion(student_logits, teacher_logits, targets)


class CustomDistillationLoss(nn.Module):
    """
    Custom Knowledge Distillation Loss with modified soft target generation.

    This variant assigns probability only to the true class from the teacher's
    prediction, while other classes receive equal probability. This creates a
    difficulty-aware training signal.

    Args:
        temperature: Temperature for softening (typically not used for teacher here).
        alpha: Weight for distillation loss.
        reduction: Reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes custom distillation loss.

        Args:
            student_logits: Student model logits of shape (N, C).
            teacher_logits: Teacher model logits of shape (N, C).
            targets: Ground truth class indices of shape (N,).

        Returns:
            Loss value.
        """

        num_classes = student_logits.size(-1)
        batch_size = student_logits.size(0)

        # Get teacher's soft predictions
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Create custom soft targets:
        # True class gets teacher's probability, others get equal distribution
        with torch.no_grad():
            # Extract teacher's confidence for true class
            true_class_probs = teacher_probs.gather(1, targets.unsqueeze(1))  # (N, 1)

            # Create soft targets
            soft_targets = torch.zeros_like(teacher_probs)
            remaining_prob = 1.0 - true_class_probs
            soft_targets.fill_(1.0 / (num_classes - 1))
            soft_targets *= remaining_prob  # Scale by remaining probability
            soft_targets.scatter_(1, targets.unsqueeze(1), true_class_probs)

        # Student soft predictions
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        # KL divergence
        distillation_loss = F.kl_div(
            student_log_probs,
            soft_targets,
            reduction="batchmean" if self.reduction == "mean" else self.reduction,
        )

        # Hard target loss
        student_loss = F.cross_entropy(
            student_logits,
            targets,
            reduction=self.reduction,
        )

        # Combined loss
        loss = self.alpha * distillation_loss + (1.0 - self.alpha) * student_loss

        return loss
