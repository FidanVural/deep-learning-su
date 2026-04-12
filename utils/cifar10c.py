"""
utils/cifar10c.py

CIFAR-10-C corrupted dataset loader.

CIFAR-10-C contains 15 types of algorithmically generated corruptions,
each at 5 severity levels. It is used to benchmark model robustness
against common image corruptions.

Reference:
    Hendrycks and Dietterich, "Benchmarking Neural Network Robustness
    to Common Corruptions and Perturbations", ICLR 2019.
"""

from typing import Dict, List, Optional, Tuple

import os
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def load_cifar10c(
    data_dir: str,
    corruption: str,
    severity: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
) -> DataLoader:
    """
    Loads a specific corruption type and severity from CIFAR-10-C.

    Each corruption .npy file has shape (50000, 32, 32, 3) with uint8 values,
    stacked as [severity1(10000), severity2(10000), ..., severity5(10000)].
    Labels file has shape (50000,).

    Args:
        data_dir: Directory containing CIFAR-10-C .npy files.
        corruption: Name of the corruption (e.g., "gaussian_noise").
        severity: Severity level (1-5).
        batch_size: Batch size.
        num_workers: DataLoader workers.
        normalize: Whether to apply CIFAR-10 normalization.

    Returns:
        DataLoader for the corrupted subset.

    Raises:
        FileNotFoundError: If the corruption file does not exist.
    """
    images_path = os.path.join(data_dir, f"{corruption}.npy")
    labels_path = os.path.join(data_dir, "labels.npy")

    if not os.path.exists(images_path):
        raise FileNotFoundError(
            f"CIFAR-10-C file not found: {images_path}. "
            f"Download from: https://zenodo.org/record/2535967"
        )

    images = np.load(images_path)
    labels = np.load(labels_path)

    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000
    images = images[start_idx:end_idx]
    labels = labels[start_idx:end_idx]

    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    transform = transforms.Compose(transform_list)

    tensors = []
    for img in images:
        from PIL import Image
        pil_img = Image.fromarray(img)
        tensors.append(transform(pil_img))

    data_tensor = torch.stack(tensors)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data_tensor, label_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_all_cifar10c(
    data_dir: str,
    severity: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
) -> Dict[str, DataLoader]:
    """
    Loads all 15 corruption types from CIFAR-10-C at a given severity.

    Args:
        data_dir: Directory containing CIFAR-10-C .npy files.
        severity: Severity level (1-5).
        batch_size: Batch size.
        num_workers: DataLoader workers.
        normalize: Whether to apply CIFAR-10 normalization.

    Returns:
        Dictionary mapping corruption name to DataLoader.
    """
    loaders = {}
    for corruption in CIFAR10C_CORRUPTIONS:
        try:
            loaders[corruption] = load_cifar10c(
                data_dir, corruption, severity, batch_size, num_workers, normalize,
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    return loaders


def evaluate_corruptions(
    model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    severity: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Evaluates model accuracy on all CIFAR-10-C corruptions.

    Args:
        model: Trained model.
        data_dir: CIFAR-10-C data directory.
        device: Compute device.
        severity: Corruption severity level.
        batch_size: Batch size.
        num_workers: DataLoader workers.

    Returns:
        Dictionary mapping corruption name to accuracy.
    """
    model.eval()
    results: Dict[str, float] = {}
    loaders = load_all_cifar10c(data_dir, severity, batch_size, num_workers)

    for corruption, loader in loaders.items():
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = correct / total if total > 0 else 0.0
        results[corruption] = accuracy
        print(f"  {corruption:>25s}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    mean_accuracy = np.mean(list(results.values())) if results else 0.0
    results["mean"] = float(mean_accuracy)
    print(f"  {'Mean Corruption Acc':>25s}: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")

    return results
