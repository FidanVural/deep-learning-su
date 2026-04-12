"""
models/transfer_models.py

Transfer Learning wrapper for pre-trained ResNet-18.

Provides two approaches for transfer learning on CIFAR-10:
  1. Resize approach: Resize CIFAR images to ImageNet size (224x224),
     freeze early layers, fine-tune later layers
  2. Modify approach: Modify first conv layer for 32x32 inputs,
     fine-tune all or most layers

Uses PyTorch's torchvision pre-trained models.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class TransferModelConfig:
    """Configuration for transfer learning models."""

    num_classes: int = 10
    pretrained: bool = True
    freeze_until: Optional[str] = None  # None, "layer1", "layer2", "layer3", "layer4"
    modify_first_conv: bool = False  # True to adapt conv1 for 32x32


class TransferResNet(nn.Module):
    """
    ResNet-18 with transfer learning for CIFAR-10.

    Two modes:
      1. Resize mode (modify_first_conv=False):
         - Use with resized images (224x224)
         - Freeze early layers, train later layers
      2. Modify mode (modify_first_conv=True):
         - Use with original CIFAR size (32x32)
         - Modify first conv layer
         - Fine-tune most/all layers

    Args:
        config: TransferModelConfig with transfer learning settings.
    """

    def __init__(self, config: TransferModelConfig) -> None:
        super().__init__()

        self.config = config

        # Load pre-trained ResNet-18
        if config.pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18(weights=None)

        # Modify first conv layer if needed (for 32x32 inputs)
        if config.modify_first_conv:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.model.maxpool = nn.Identity()

        # Replace final FC layer for CIFAR num_classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, config.num_classes)

        # Freeze layers if specified
        if config.freeze_until is not None:
            self._freeze_layers(config.freeze_until)

    def _freeze_layers(self, freeze_until: str) -> None:
        """
        Freezes layers up to and including the specified layer.

        Args:
            freeze_until: Layer name to freeze until ("layer1", "layer2", "layer3", "layer4").
        """

        layers_to_freeze = ["conv1", "bn1"]

        if freeze_until in ["layer1", "layer2", "layer3", "layer4"]:
            layers_to_freeze.append("layer1")
        if freeze_until in ["layer2", "layer3", "layer4"]:
            layers_to_freeze.append("layer2")
        if freeze_until in ["layer3", "layer4"]:
            layers_to_freeze.append("layer3")
        if freeze_until == "layer4":
            layers_to_freeze.append("layer4")

        for name, param in self.model.named_parameters():
            for layer_name in layers_to_freeze:
                if name.startswith(layer_name):
                    param.requires_grad = False
                    break

        num_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        num_total = sum(1 for _ in self.model.parameters())
        print(f"Frozen {num_frozen}/{num_total} parameters (up to {freeze_until})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet."""
        return self.model(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature maps before the classifier (for t-SNE visualization).

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Feature tensor of shape (B, 512).
        """

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def unfreeze_all(self) -> None:
        """Unfreezes all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
