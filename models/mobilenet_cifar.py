"""
models/mobilenet_cifar.py

MobileNetV2 architecture adapted for CIFAR-10/100.

MobileNetV2 uses inverted residual blocks with depthwise separable convolutions,
making it highly efficient for mobile and embedded devices.

Key features:
  - Inverted residuals with linear bottlenecks
  - Depthwise separable convolutions
  - Lightweight and efficient (fewer FLOPs than ResNet)

Reference:
  - Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class MobileNetCIFARConfig:
    """Configuration for MobileNetV2 CIFAR architecture."""

    num_classes: int = 10
    width_mult: float = 1.0  # Width multiplier (0.5, 0.75, 1.0, etc.)
    input_size: int = 32  # Input image size


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (MobileNetV2 building block).

    Structure:
        1x1 Conv (expansion) -> DWConv -> 1x1 Conv (projection)
        with residual connection if input == output

    Args:
        inp: Number of input channels.
        oup: Number of output channels.
        stride: Stride for depthwise convolution.
        expand_ratio: Channel expansion ratio.
    """

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []

        # Expansion phase (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Projection phase (linear bottleneck)
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""

        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetCIFAR(nn.Module):
    """
    MobileNetV2 adapted for CIFAR-10/100 (32x32 images).

    This is a lightweight and efficient architecture using inverted residuals
    and depthwise separable convolutions.

    Args:
        config: MobileNetCIFARConfig with architecture hyperparameters.
    """

    def __init__(self, config: MobileNetCIFARConfig) -> None:
        super().__init__()

        self.config = config

        # Building first layer
        input_channel = int(32 * config.width_mult)
        last_channel = int(1280 * config.width_mult) if config.width_mult > 1.0 else 1280

        # Inverted residual settings: [expand_ratio, output_channels, num_blocks, stride]
        inverted_residual_setting = [
            [1, 16, 1, 1],   # 32x32 -> 32x32
            [6, 24, 2, 1],   # 32x32 -> 32x32 (stride 2 -> 1 for CIFAR)
            [6, 32, 3, 2],   # 32x32 -> 16x16
            [6, 64, 4, 2],   # 16x16 -> 8x8
            [6, 96, 3, 1],   # 8x8 -> 8x8
            [6, 160, 3, 2],  # 8x8 -> 4x4
            [6, 320, 1, 1],  # 4x4 -> 4x4
        ]

        # First conv layer (adapted for CIFAR)
        self.features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * config.width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        # Last conv layer
        self.features.append(
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False)
        )
        self.features.append(nn.BatchNorm2d(last_channel))
        self.features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*self.features)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, config.num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initializes model weights."""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Logits tensor of shape (B, num_classes).
        """

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature maps before the classifier.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Feature tensor of shape (B, last_channel).
        """

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def mobilenetv2_cifar(num_classes: int = 10, width_mult: float = 1.0) -> MobileNetCIFAR:
    """
    Constructs MobileNetV2 for CIFAR-10/100.

    Args:
        num_classes: Number of output classes.
        width_mult: Width multiplier for model size.

    Returns:
        MobileNetCIFAR model.
    """

    config = MobileNetCIFARConfig(
        num_classes=num_classes,
        width_mult=width_mult,
        input_size=32,
    )
    return MobileNetCIFAR(config)
