"""
models/mlp.py

Multi-Layer Perceptron (MLP) model for MNIST classification.

Architecture per hidden layer:
    Linear -> BatchNorm1d -> Activation (ReLU / GELU) -> Dropout

Uses nn.ModuleList to hold the hidden layer blocks and nn.Sequential to compose each block.
"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten


# Model Configuration
@dataclass
class MLPConfig:
    """Hyperparameters that define the MLP architecture."""

    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10
    activation: str = "relu"   # "relu" or "gelu"
    dropout: float = 0.3       # 0.0 means no dropout
    use_bn: bool = True        # whether to include BatchNorm1d layers


# MLP Model
class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron for image classification.

    Each hidden block follows the order recommended in the literature:
        Linear -> BatchNorm1d -> Activation -> Dropout

    Args:
        config: MLPConfig dataclass with architecture hyperparameters.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        self.config = config

        self.flatten: Flatten = nn.Flatten()

        self.hidden_layers: nn.ModuleList = nn.ModuleList()

        in_features = config.input_size

        for hidden_size in config.hidden_sizes:
            block = self._build_block(in_features, hidden_size, config)
            self.hidden_layers.append(block)
            in_features = hidden_size

        self.classifier: nn.Linear = nn.Linear(in_features, config.num_classes)

    @staticmethod
    def _build_block(
        in_features: int,
        out_features: int,
        config: MLPConfig,
    ) -> nn.Sequential:
        """
        Builds a single hidden layer block.

        Order: Linear -> [BatchNorm1d] -> Activation -> [Dropout]

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            config: MLPConfig controlling optional components.

        Returns:
            nn.Sequential block.
        """

        layers: List[nn.Module] = []

        layers.append(nn.Linear(in_features, out_features))

        if config.use_bn:
            layers.append(nn.BatchNorm1d(out_features))

        if config.activation == "relu":
            layers.append(nn.ReLU())
        elif config.activation == "gelu":
            layers.append(nn.GELU())
        else:
            raise ValueError(f"Unknown activation: '{config.activation}'. Choose 'relu' or 'gelu'.")

        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, 28, 28) or (B, 784).

        Returns:
            Logits tensor of shape (B, num_classes).
        """

        x = self.flatten(x)

        for layer in self.hidden_layers:
            x = layer(x)

        return self.classifier(x)
