"""
models/__init__.py

Model architectures for HW1b: Transfer Learning and Knowledge Distillation.
"""

from .simple_cnn import SimpleCNN, SimpleCNNConfig
from .resnet_cifar import ResNetCIFAR, ResNetCIFARConfig
from .mobilenet_cifar import MobileNetCIFAR, MobileNetCIFARConfig
from .transfer_models import TransferResNet, TransferModelConfig

__all__ = [
    "SimpleCNN",
    "SimpleCNNConfig",
    "ResNetCIFAR",
    "ResNetCIFARConfig",
    "MobileNetCIFAR",
    "MobileNetCIFARConfig",
    "TransferResNet",
    "TransferModelConfig",
]
