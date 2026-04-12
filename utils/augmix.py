"""
utils/augmix.py

AugMix data augmentation implementation for improving model robustness.

AugMix combines multiple augmentation chains with a mixing strategy
that interpolates between augmented images and the original image
using Dirichlet and Beta distributions.

Reference:
    Hendrycks et al., "AugMix: A Simple Data Processing Method to Improve
    Robustness and Uncertainty", ICLR 2020.
"""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


# CIFAR-10 statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ── Individual augmentation operations ──────────────────────────────────────

def autocontrast(pil_img: Image.Image, _level: int) -> Image.Image:
    """Applies autocontrast to the image."""
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img: Image.Image, _level: int) -> Image.Image:
    """Applies histogram equalization."""
    return ImageOps.equalize(pil_img)


def posterize(pil_img: Image.Image, level: int) -> Image.Image:
    """Reduces the number of bits per channel."""
    level = int((level / 10.0) * 4)
    return ImageOps.posterize(pil_img, max(1, 4 - level))


def rotate(pil_img: Image.Image, level: int) -> Image.Image:
    """Rotates the image by a random angle proportional to level."""
    degrees = (level / 10.0) * 30.0
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def solarize(pil_img: Image.Image, level: int) -> Image.Image:
    """Inverts all pixels above a threshold."""
    threshold = int((level / 10.0) * 256)
    return ImageOps.solarize(pil_img, 256 - threshold)


def shear_x(pil_img: Image.Image, level: int) -> Image.Image:
    """Applies horizontal shear."""
    factor = (level / 10.0) * 0.3
    if np.random.uniform() > 0.5:
        factor = -factor
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def shear_y(pil_img: Image.Image, level: int) -> Image.Image:
    """Applies vertical shear."""
    factor = (level / 10.0) * 0.3
    if np.random.uniform() > 0.5:
        factor = -factor
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def translate_x(pil_img: Image.Image, level: int) -> Image.Image:
    """Translates horizontally."""
    pixels = int((level / 10.0) * pil_img.size[0] / 3)
    if np.random.uniform() > 0.5:
        pixels = -pixels
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def translate_y(pil_img: Image.Image, level: int) -> Image.Image:
    """Translates vertically."""
    pixels = int((level / 10.0) * pil_img.size[1] / 3)
    if np.random.uniform() > 0.5:
        pixels = -pixels
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def color(pil_img: Image.Image, level: int) -> Image.Image:
    """Adjusts color balance."""
    factor = 1.0 + (level / 10.0) * 0.9 * (1 if np.random.uniform() > 0.5 else -1)
    return ImageEnhance.Color(pil_img).enhance(max(0.0, factor))


def contrast(pil_img: Image.Image, level: int) -> Image.Image:
    """Adjusts contrast."""
    factor = 1.0 + (level / 10.0) * 0.9 * (1 if np.random.uniform() > 0.5 else -1)
    return ImageEnhance.Contrast(pil_img).enhance(max(0.0, factor))


def brightness(pil_img: Image.Image, level: int) -> Image.Image:
    """Adjusts brightness."""
    factor = 1.0 + (level / 10.0) * 0.9 * (1 if np.random.uniform() > 0.5 else -1)
    return ImageEnhance.Brightness(pil_img).enhance(max(0.0, factor))


def sharpness(pil_img: Image.Image, level: int) -> Image.Image:
    """Adjusts sharpness."""
    factor = 1.0 + (level / 10.0) * 0.9 * (1 if np.random.uniform() > 0.5 else -1)
    return ImageEnhance.Sharpness(pil_img).enhance(max(0.0, factor))


AUGMENTATIONS = [
    autocontrast, equalize, posterize, rotate, solarize,
    shear_x, shear_y, translate_x, translate_y,
    color, contrast, brightness, sharpness,
]


# ── AugMix Transform ───────────────────────────────────────────────────────

class AugMixTransform:
    """
    AugMix data augmentation that produces a single augmented image.

    Mixes multiple augmentation chains with the original image using
    Dirichlet and Beta-distributed mixing weights.

    Args:
        severity: Severity of individual augmentation operations (1-10).
        width: Number of parallel augmentation chains.
        depth: Depth of each chain (-1 for stochastic depth [1,3]).
        alpha: Dirichlet distribution concentration parameter.
        preprocess: Preprocessing transform (ToTensor + Normalize) applied after mixing.
    """

    def __init__(
        self,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        preprocess: Optional[transforms.Compose] = None,
    ) -> None:
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.preprocess = preprocess or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Applies AugMix to a PIL image.

        Args:
            img: Input PIL image.

        Returns:
            Augmented tensor.
        """
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = torch.zeros_like(self.preprocess(img))

        for i in range(self.width):
            img_aug = img.copy()
            chain_depth = self.depth if self.depth > 0 else np.random.randint(1, 4)

            for _ in range(chain_depth):
                op = np.random.choice(AUGMENTATIONS)
                img_aug = op(img_aug, self.severity)

            mix += ws[i] * self.preprocess(img_aug)

        return (1.0 - m) * self.preprocess(img) + m * mix


class AugMixDataset(Dataset):
    """
    Wraps a CIFAR dataset to apply AugMix on top of basic augmentations.

    Args:
        dataset: Base CIFAR dataset (should have PIL-returning transforms or raw data).
        augmix_transform: AugMixTransform instance.
        basic_transform: Basic augmentation (RandomCrop, RandomFlip) applied before AugMix.
    """

    def __init__(
        self,
        dataset: datasets.VisionDataset,
        augmix_transform: AugMixTransform,
        basic_transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.dataset = dataset
        self.augmix_transform = augmix_transform
        self.basic_transform = basic_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.dataset.data[index], self.dataset.targets[index]
        img = Image.fromarray(img)

        if self.basic_transform is not None:
            img = self.basic_transform(img)

        img_tensor = self.augmix_transform(img)
        return img_tensor, target

    def __len__(self) -> int:
        return len(self.dataset)


class TransformSubset(Dataset):
    """
    Wraps a Subset with a specified transform.

    Args:
        subset: A torch Subset from random_split.
        transform: Transform to apply.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.subset.dataset.data[self.subset.indices[index]]
        target = self.subset.dataset.targets[self.subset.indices[index]]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.subset)


class AugMixSubset(Dataset):
    """
    Wraps a Subset with AugMix augmentation.

    Args:
        subset: A torch Subset.
        augmix_transform: AugMixTransform instance.
        basic_transform: Optional basic augmentation before AugMix.
    """

    def __init__(
        self,
        subset: Subset,
        augmix_transform: AugMixTransform,
        basic_transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.subset = subset
        self.augmix_transform = augmix_transform
        self.basic_transform = basic_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.subset.dataset.data[self.subset.indices[index]]
        target = self.subset.dataset.targets[self.subset.indices[index]]
        img = Image.fromarray(img)

        if self.basic_transform is not None:
            img = self.basic_transform(img)

        img_tensor = self.augmix_transform(img)
        return img_tensor, target

    def __len__(self) -> int:
        return len(self.subset)


def get_augmix_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates CIFAR-10 data loaders with AugMix for training.

    Args:
        data_dir: Root directory for CIFAR-10 data.
        batch_size: Batch size.
        val_split: Fraction of training data for validation.
        num_workers: Number of DataLoader workers.
        severity: AugMix severity.
        width: AugMix chain width.
        depth: AugMix chain depth.
        alpha: Dirichlet concentration parameter.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    basic_augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    augmix = AugMixTransform(
        severity=severity,
        width=width,
        depth=depth,
        alpha=alpha,
        preprocess=preprocess,
    )

    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    train_dataset = AugMixSubset(train_subset, augmix, basic_augment)
    val_dataset = TransformSubset(val_subset, test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
