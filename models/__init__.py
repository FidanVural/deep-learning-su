"""
models/__init__.py

Model architectures for HW2.
"""

from .simple_cnn import SimpleCNN, SimpleCNNConfig
from .resnet_cifar import ResNetCIFAR, ResNetCIFARConfig, resnet18_cifar
from .mobilenet_cifar import MobileNetCIFAR, MobileNetCIFARConfig, mobilenetv2_cifar
from .transfer_models import TransferResNet, TransferModelConfig

__all__ = [
    "SimpleCNN",
    "SimpleCNNConfig",
    "ResNetCIFAR",
    "ResNetCIFARConfig",
    "resnet18_cifar",
    "MobileNetCIFAR",
    "MobileNetCIFARConfig",
    "mobilenetv2_cifar",
    "TransferResNet",
    "TransferModelConfig",
]
