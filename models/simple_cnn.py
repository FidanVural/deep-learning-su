"""
models/simple_cnn.py

Simple CNN architecture for CIFAR-10 classification (student model).

A lightweight convolutional network designed to be small and efficient,
suitable for knowledge distillation experiments.

Architecture:
    Conv Block 1 -> Conv Block 2 -> Conv Block 3 -> FC -> Output

Each Conv Block:
    Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


@dataclass
class SimpleCNNConfig:
    """Configuration for SimpleCNN architecture."""

    num_classes: int = 10
    channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    hidden_size: int = 256
    dropout: float = 0.3
    input_channels: int = 3


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for CIFAR-10.

    A compact CNN architecture with 3 convolutional blocks followed by
    fully connected layers. Designed as a student model for knowledge distillation.

    Args:
        config: SimpleCNNConfig with architecture hyperparameters.
    """

    def __init__(self, config: SimpleCNNConfig) -> None:
        super().__init__()

        self.config = config

        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()

        in_channels = config.input_channels
        for out_channels in config.channels:
            block = self._make_conv_block(in_channels, out_channels)
            self.conv_blocks.append(block)
            in_channels = out_channels

        # Calculate flattened size after conv layers
        # CIFAR-10 images: 32x32
        # After each block with MaxPool(2): 32 -> 16 -> 8 -> 4
        self.feature_size = config.channels[-1] * 4 * 4

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        # Output layer
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        # Initialize weights
        self._initialize_weights()

    @staticmethod
    def _make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Creates a convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            Sequential block with Conv -> BN -> ReLU -> MaxPool.
        """

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Logits tensor of shape (B, num_classes).
        """

        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Fully connected layers
        x = self.fc(x)

        # Classification
        x = self.classifier(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature maps before the classifier.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Feature tensor of shape (B, hidden_size).
        """

        for block in self.conv_blocks:
            x = block(x)

        x = self.fc(x)

        return x
