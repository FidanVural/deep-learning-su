"""
utils/gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Grad-CAM produces a coarse localization map highlighting important regions
in the image for the predicted class, using gradients flowing into the
final convolutional layer.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

from typing import List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM visualization for convolutional neural networks.

    Hooks into a target convolutional layer and computes the gradient-weighted
    activation map for a given class.

    Args:
        model: Trained neural network.
        target_layer: The convolutional layer to visualize (e.g., model.layer4[-1].conv2).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers forward and backward hooks on the target layer."""

        def forward_hook(module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for the input image.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Heatmap of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def _get_target_layer(model: nn.Module) -> nn.Module:
    """
    Attempts to automatically find the last convolutional layer.

    Supports TransferResNet (model.model.layer4), ResNetCIFAR (model.layer4),
    and SimpleCNN (model.conv_blocks[-1]).

    Args:
        model: Neural network model.

    Returns:
        Target convolutional layer module.
    """
    if hasattr(model, "model") and hasattr(model.model, "layer4"):
        return model.model.layer4[-1]
    elif hasattr(model, "layer4"):
        return model.layer4[-1]
    elif hasattr(model, "conv_blocks"):
        return model.conv_blocks[-1]
    elif hasattr(model, "features"):
        for layer in reversed(list(model.features.modules())):
            if isinstance(layer, nn.Conv2d):
                return layer
        return model.features[-3]
    else:
        raise ValueError("Cannot auto-detect target layer for Grad-CAM.")


def visualize_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    title: str = "Grad-CAM",
    save_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Generates and plots Grad-CAM heatmaps for a set of images.

    Displays the original image, Grad-CAM overlay, and prediction info.

    Args:
        model: Trained model.
        images: Batch of images (B, C, H, W), expected to be normalized.
        labels: True labels (B,).
        predictions: Model predictions (B,).
        class_names: List of class names.
        title: Plot title.
        save_path: If provided, saves the figure.
        device: Compute device.
    """
    target_layer = _get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    num_samples = len(images)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    model = model.to(device)

    for i in range(num_samples):
        img = images[i].unsqueeze(0).to(device)
        true_label = labels[i].item()
        pred_label = predictions[i].item()

        cam = grad_cam.generate(img, target_class=pred_label)

        img_denorm = images[i].cpu() * std + mean
        img_denorm = img_denorm.clamp(0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"True: {class_names[true_label]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam, cmap="jet", alpha=0.8)
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_np)
        axes[i, 2].imshow(cam, cmap="jet", alpha=0.4)
        axes[i, 2].set_title(f"Pred: {class_names[pred_label]}")
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM plot saved to: {save_path}")

    plt.close(fig)
