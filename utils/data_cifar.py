"""
utils/data_cifar.py

CIFAR-10/100 dataset loader with train/validation split and optional resizing.

Features:
  - Support for both CIFAR-10 and CIFAR-100
  - Optional image resizing (e.g., 32x32 -> 224x224 for transfer learning)
  - Data augmentation (RandomCrop, RandomHorizontalFlip)
  - Normalization with CIFAR statistics
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms


# Data Configuration
@dataclass
class DataConfig:
    """Configuration for CIFAR dataset loading."""

    data_dir: str = "./data"
    dataset: str = "cifar10"  # cifar10 or cifar100
    batch_size: int = 128
    val_split: float = 0.1
    num_workers: int = 4
    resize_to: Optional[int] = None  # None for 32x32, or 224 for ImageNet size
    normalize: bool = True
    augment: bool = True
    seed: int = 42


# CIFAR Statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_transforms(
    dataset: str = "cifar10",
    resize_to: Optional[int] = None,
    normalize: bool = True,
    augment: bool = True,
    train: bool = True,
) -> transforms.Compose:
    """
    Returns CIFAR transforms for training or testing.

    Args:
        dataset: "cifar10" or "cifar100".
        resize_to: Target image size (None for original 32x32).
        normalize: Whether to apply normalization.
        augment: Whether to apply data augmentation (train only).
        train: Whether this is for training or testing.

    Returns:
        Composed transforms.
    """

    transform_list = []

    # Resize if needed (for transfer learning)
    if resize_to is not None:
        transform_list.append(transforms.Resize(resize_to))

    # Data augmentation (train only)
    if train and augment:
        if resize_to is None or resize_to == 32:
            # Standard CIFAR augmentation
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            # Augmentation for resized images
            transform_list.extend([
                transforms.RandomCrop(resize_to, padding=resize_to // 8),
                transforms.RandomHorizontalFlip(),
            ])

    # To tensor
    transform_list.append(transforms.ToTensor())

    # Normalization
    if normalize:
        if dataset == "cifar10":
            mean, std = CIFAR10_MEAN, CIFAR10_STD
        else:
            mean, std = CIFAR100_MEAN, CIFAR100_STD

        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


class TransformSubset(Dataset):
    """
    Wraps a Subset with a different transform, overriding the parent dataset's transform.

    This is used to apply test-time transforms (no augmentation) to the validation
    split while the training split retains augmentation transforms.

    Args:
        subset: A torch Subset from random_split.
        transform: The transform to apply instead of the parent's.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        img, target = self.subset.dataset.data[self.subset.indices[index]], \
                       self.subset.dataset.targets[self.subset.indices[index]]
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.subset)


def split_dataset(
    dataset: datasets.VisionDataset,
    val_split: float,
    seed: int
) -> Tuple[Subset, Subset]:
    """
    Splits dataset into train and validation sets.

    Args:
        dataset: The full training dataset.
        val_split: Fraction of data to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_dataset, val_dataset


def get_dataloaders(
    config: DataConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates CIFAR-10/100 data loaders for train, validation, and test sets.

    Args:
        config: DataConfig with dataset configuration.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """

    # Select dataset class
    if config.dataset == "cifar10":
        dataset_class = datasets.CIFAR10
    elif config.dataset == "cifar100":
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Training transforms
    train_transform = get_transforms(
        dataset=config.dataset,
        resize_to=config.resize_to,
        normalize=config.normalize,
        augment=config.augment,
        train=True,
    )

    # Test transforms (no augmentation)
    test_transform = get_transforms(
        dataset=config.dataset,
        resize_to=config.resize_to,
        normalize=config.normalize,
        augment=False,
        train=False,
    )

    # Load datasets
    full_train_dataset = dataset_class(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = dataset_class(
        root=config.data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Split train into train/val
    train_subset, val_subset = split_dataset(
        full_train_dataset,
        config.val_split,
        config.seed
    )

    # Re-wrap val subset with test transforms (no augmentation)
    train_dataset = train_subset
    val_dataset = TransformSubset(val_subset, test_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, val_loader, test_loader
