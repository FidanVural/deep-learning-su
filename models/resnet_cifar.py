"""
models/resnet_cifar.py

ResNet-18 architecture adapted for CIFAR-10/100 from scratch.

This is a modified version of ResNet-18 designed for small 32x32 images,
not using ImageNet pretrained weights. Suitable for training from scratch
and as a teacher model for knowledge distillation.

Key differences from standard ResNet-18:
  - First conv layer: 7x7 -> 3x3 (no stride, no MaxPool)
  - Smaller feature maps throughout
  - Optimized for 32x32 input size

"""

from dataclasses import dataclass
from typing import List, Type

import torch
import torch.nn as nn


@dataclass
class ResNetCIFARConfig:
    """
    Configuration for ResNet CIFAR architecture.

    num_blocks controls the depth of the network. Each element sets the number
    of residual blocks in the corresponding layer (layer1..layer4). Since each
    BasicBlock contains 2 conv layers, total depth = 2*sum(num_blocks) + 2
    (the +2 accounts for conv1 and fc). For example [2,2,2,2] gives
    2*(2+2+2+2)+2 = 18 layers, hence ResNet-18.
    """

    num_classes: int = 10
    block_type: str = "basic"  # "basic" for ResNet-18/34
    num_blocks: List[int] = None  # [2,2,2,2]=ResNet-18, [3,4,6,3]=ResNet-34

    def __post_init__(self):
        if self.num_blocks is None:
            self.num_blocks = [2, 2, 2, 2]  # ResNet-18


class BasicBlock(nn.Module):
    """
    Basic ResNet block for ResNet-18/34.

    Structure:
        Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10/100 (32x32 images).

    This version is designed to be trained from scratch on CIFAR datasets.
    It uses a smaller first conv layer (3x3 instead of 7x7) and no MaxPooling
    to preserve spatial resolution for small images.

    Args:
        config: ResNetCIFARConfig with architecture hyperparameters.
    """

    def __init__(self, config: ResNetCIFARConfig) -> None:
        super().__init__()

        self.config = config
        self.in_planes = 64

        # First conv layer (adapted for CIFAR)
        # No stride, no MaxPool to preserve resolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, config.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, config.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, config.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, config.num_blocks[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, config.num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        Creates a residual layer with multiple blocks.

        Args:
            block: Block class (BasicBlock).
            planes: Number of output channels.
            num_blocks: Number of blocks in this layer.
            stride: Stride for the first block (for downsampling).

        Returns:
            Sequential module containing the blocks.
        """

        downsample = None

        # Add downsampling if needed
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        # First block (with potential downsampling)
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initializes model weights using He initialization."""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Logits tensor of shape (B, num_classes).
        """

        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature maps before the classifier.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Feature tensor of shape (B, 512).
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def resnet18_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """
    Constructs ResNet-18 for CIFAR-10/100.

    Args:
        num_classes: Number of output classes.

    Returns:
        ResNetCIFAR model.
    """

    config = ResNetCIFARConfig(
        num_classes=num_classes,
        num_blocks=[2, 2, 2, 2],
    )
    return ResNetCIFAR(config)
