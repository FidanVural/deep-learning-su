"""
utils/pgd_attack.py

Projected Gradient Descent (PGD) adversarial attack implementation.

PGD is an iterative first-order attack that projects the perturbation
onto an Lp ball at each step. It is considered one of the strongest
first-order adversaries.

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to
    Adversarial Attacks", ICLR 2018.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PGDAttack:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    Supports both L-infinity and L2 norm constraints.

    Args:
        model: Target model to attack.
        eps: Maximum perturbation magnitude.
        step_size: Per-step perturbation size.
        steps: Number of PGD iterations.
        norm: Norm type ('linf' or 'l2').
        random_start: Whether to initialize perturbation randomly.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        step_size: float,
        steps: int = 20,
        norm: str = "linf",
        random_start: bool = True,
    ) -> None:
        self.model = model
        self.eps = eps
        self.step_size = step_size
        self.steps = steps
        self.norm = norm
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()

    def _project_linf(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Projects perturbation onto L-inf ball of radius eps."""
        return torch.clamp(perturbation, -self.eps, self.eps)

    def _project_l2(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Projects perturbation onto L2 ball of radius eps."""
        batch_size = perturbation.shape[0]
        flat = perturbation.view(batch_size, -1)
        norms = flat.norm(p=2, dim=1, keepdim=True)
        factor = torch.min(torch.ones_like(norms), self.eps / (norms + 1e-12))
        flat = flat * factor
        return flat.view_as(perturbation)

    def _random_init_linf(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Random initialization within L-inf ball."""
        return torch.empty(shape, device=device).uniform_(-self.eps, self.eps)

    def _random_init_l2(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Random initialization within L2 ball."""
        noise = torch.randn(shape, device=device)
        flat = noise.view(shape[0], -1)
        norms = flat.norm(p=2, dim=1, keepdim=True)
        flat = flat / (norms + 1e-12)
        r = torch.rand(shape[0], 1, device=device) ** (1.0 / np.prod(shape[1:]))
        flat = flat * r * self.eps
        return flat.view(shape)

    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generates adversarial examples using PGD.

        Args:
            images: Clean input images of shape (B, C, H, W).
            labels: True class labels of shape (B,).

        Returns:
            Adversarial images of shape (B, C, H, W).
        """
        self.model.eval()
        device = images.device

        if self.random_start:
            if self.norm == "linf":
                delta = self._random_init_linf(images.shape, device)
            else:
                delta = self._random_init_l2(images.shape, device)
        else:
            delta = torch.zeros_like(images)

        delta = delta.detach()
        delta.requires_grad_(True)

        for _ in range(self.steps):
            adv_images = torch.clamp(images + delta, 0.0, 1.0)
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            grad = delta.grad.detach()

            with torch.no_grad():
                if self.norm == "linf":
                    delta_update = self.step_size * grad.sign()
                    delta = delta + delta_update
                    delta = self._project_linf(delta)
                else:
                    grad_flat = grad.view(grad.shape[0], -1)
                    grad_norms = grad_flat.norm(p=2, dim=1, keepdim=True)
                    grad_normalized = grad_flat / (grad_norms + 1e-12)
                    delta_flat = delta.view(delta.shape[0], -1)
                    delta_flat = delta_flat + self.step_size * grad_normalized
                    delta = delta_flat.view_as(delta)
                    delta = self._project_l2(delta)

                delta = torch.clamp(images + delta, 0.0, 1.0) - images

            delta = delta.detach()
            delta.requires_grad_(True)

        adv_images = torch.clamp(images + delta.detach(), 0.0, 1.0)
        return adv_images

    def evaluate(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluates model accuracy under PGD attack on the entire dataset.

        Args:
            loader: DataLoader with (images, labels).
            device: Compute device.

        Returns:
            Dictionary with 'clean_acc', 'adv_acc', and 'attack_success_rate'.
        """
        self.model.eval()

        clean_correct = 0
        adv_correct = 0
        total = 0

        for data, target in loader:
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                clean_output = self.model(data)
                clean_pred = clean_output.argmax(dim=1)
                clean_correct += (clean_pred == target).sum().item()

            adv_data = self.attack(data, target)

            with torch.no_grad():
                adv_output = self.model(adv_data)
                adv_pred = adv_output.argmax(dim=1)
                adv_correct += (adv_pred == target).sum().item()

            total += target.size(0)

        clean_acc = clean_correct / total
        adv_acc = adv_correct / total
        attack_success = 1.0 - adv_acc

        return {
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "attack_success_rate": attack_success,
        }


def generate_adversarial_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 4.0 / 255.0,
    step_size: float = 1.0 / 255.0,
    steps: int = 20,
    norm: str = "linf",
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates adversarial samples from a model and returns both clean and adversarial data.

    Args:
        model: Model used to generate adversarial examples.
        loader: DataLoader with clean data.
        device: Compute device.
        eps: Perturbation budget.
        step_size: Step size per PGD iteration.
        steps: Number of PGD steps.
        norm: Norm constraint ('linf' or 'l2').
        max_batches: If set, limits the number of batches processed.

    Returns:
        Tuple of (clean_images, adv_images, labels, predictions_on_adv).
    """
    attacker = PGDAttack(model, eps=eps, step_size=step_size, steps=steps, norm=norm)

    all_clean = []
    all_adv = []
    all_labels = []
    all_preds = []

    model.eval()

    for batch_idx, (data, target) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        data, target = data.to(device), target.to(device)
        adv_data = attacker.attack(data, target)

        with torch.no_grad():
            adv_preds = model(adv_data).argmax(dim=1)

        all_clean.append(data.cpu())
        all_adv.append(adv_data.cpu())
        all_labels.append(target.cpu())
        all_preds.append(adv_preds.cpu())

    return (
        torch.cat(all_clean),
        torch.cat(all_adv),
        torch.cat(all_labels),
        torch.cat(all_preds),
    )
