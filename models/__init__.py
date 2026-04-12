"""
models/__init__.py

Model architectures for HW2. Re-exports models from the parent hw1b codebase.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.simple_cnn import SimpleCNN, SimpleCNNConfig
from models.resnet_cifar import ResNetCIFAR, ResNetCIFARConfig, resnet18_cifar
from models.mobilenet_cifar import MobileNetCIFAR, MobileNetCIFARConfig, mobilenetv2_cifar
from models.transfer_models import TransferResNet, TransferModelConfig

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
