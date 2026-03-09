"""
utils/data.py

MNIST dataset loader with train/validation split.

Uses torchvision for data loading and transforms.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Data Configuration
@dataclass
class DataConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    val_split: float = 0.1
    num_workers: int = 2
    normalize: bool = True
    seed: int = 42


# Transforms
def get_transforms(normalize: bool = True):
    """
    Returns MNIST transforms.

    Normalization values are MNIST standard.
    """

    transform_list = [transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            )
        )

    return transforms.Compose(transform_list)


# Dataset Split
def split_dataset(dataset, val_split: float, seed: int):
    """
    Splits dataset into train and validation sets.
    """

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed) # important for reproducability

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_dataset, val_dataset


# Main Loader Function
def get_dataloaders(
    config: DataConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns:

    train_loader
    val_loader
    test_loader
    """

    transform = get_transforms(config.normalize)

    # Train dataset
    full_train_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Split train/val
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        config.val_split,
        config.seed
    )

    # Test dataset
    test_dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader