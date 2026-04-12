"""
models/__init__.py

Model architectures for HW2.
"""

from .simple_cnn import SimpleCNN, SimpleCNNConfig
from .resnet_cifar import ResNetCIFAR, ResNetCIFARConfig, resnet18_cifar

__all__ = [
    "SimpleCNN",
    "SimpleCNNConfig",
    "ResNetCIFAR",
    "ResNetCIFARConfig",
    "resnet18_cifar",
]
