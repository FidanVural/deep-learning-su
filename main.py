"""
main.py

Entry point for HW1b: Transfer Learning and Knowledge Distillation on CIFAR-10.

Supports three main tasks:
  1. Baseline: Train a model from scratch
  2. Transfer: Transfer learning from ImageNet pre-trained models
  3. Distillation: Knowledge distillation with teacher-student framework
"""

import os
import sys
import torch

from parameters import get_params, DataParams, ModelParams, TrainParams
from utils.data_cifar import DataConfig, get_dataloaders
from utils.flops_counter import count_flops

from models.simple_cnn import SimpleCNN, SimpleCNNConfig
from models.resnet_cifar import ResNetCIFAR, ResNetCIFARConfig, resnet18_cifar
from models.mobilenet_cifar import MobileNetCIFAR, MobileNetCIFARConfig, mobilenetv2_cifar
from models.transfer_models import TransferResNet, TransferModelConfig

from train import train
from test import load_and_test
from utils.visualization import plot_training_history, plot_confusion_matrix


# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def build_data_config(data_params: DataParams) -> DataConfig:
    """
    Converts DataParams to DataConfig.

    Args:
        data_params: DataParams from argparse.

    Returns:
        DataConfig for data loader.
    """

    return DataConfig(
        data_dir=data_params.data_dir,
        dataset=data_params.dataset,
        batch_size=data_params.batch_size,
        val_split=data_params.val_split,
        num_workers=data_params.num_workers,
        resize_to=data_params.resize_to,
        normalize=data_params.normalize,
        augment=data_params.augment,
        seed=data_params.seed,
    )


def build_model(model_params: ModelParams) -> torch.nn.Module:
    """
    Builds the model based on ModelParams.

    Args:
        model_params: ModelParams from argparse.

    Returns:
        PyTorch model instance.
    """

    model_type = model_params.model_type

    if model_type == "simple_cnn":
        config = SimpleCNNConfig(
            num_classes=model_params.num_classes,
            channels=model_params.cnn_channels,
            hidden_size=model_params.cnn_hidden_size,
            dropout=model_params.dropout,
        )
        return SimpleCNN(config)

    elif model_type == "resnet18":
        return resnet18_cifar(num_classes=model_params.num_classes)

    elif model_type == "mobilenetv2":
        return mobilenetv2_cifar(num_classes=model_params.num_classes, width_mult=1.0)

    elif model_type == "transfer_resnet":
        config = TransferModelConfig(
            num_classes=model_params.num_classes,
            pretrained=model_params.pretrained,
            freeze_until=model_params.freeze_until,
            modify_first_conv=model_params.modify_first_conv,
        )
        return TransferResNet(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_teacher_model(
    teacher_path: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Loads a pre-trained teacher model for knowledge distillation.

    Args:
        teacher_path: Path to teacher checkpoint.
        num_classes: Number of output classes.
        device: Device to load the model on.

    Returns:
        Loaded teacher model.
    """

    # Assume teacher is ResNet-18 CIFAR
    teacher_model = resnet18_cifar(num_classes=num_classes)
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher_model.eval()

    print(f"\nLoaded teacher model from: {teacher_path}")

    return teacher_model


def main() -> None:
    """
    Main entry point for HW1b.

    Parses arguments, builds data loaders and model, then
    runs training and/or testing based on --mode.
    """

    # Parse arguments
    data_params, model_params, train_params = get_params()

    # CUDA guard
    if train_params.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        train_params.device = "cpu"

    device = torch.device(train_params.device)

    # Setup output directory
    out_dir = train_params.save_path
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "best_model.pth")
    train_params.save_path = checkpoint_path

    # Reproducibility
    torch.manual_seed(data_params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(data_params.seed)

    print(f"\n{'='*80}")
    print(f"HW1b: Transfer Learning and Knowledge Distillation")
    print(f"{'='*80}")
    print(f"Task: {train_params.task}")
    print(f"Model: {model_params.model_type}")
    print(f"Dataset: {data_params.dataset}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(f"{'='*80}\n")

    # --- Data ---
    data_config = build_data_config(data_params)
    train_loader, val_loader, test_loader = get_dataloaders(data_config)

    print(f"Dataset splits:")
    print(f"  Train: {len(train_loader.dataset):>6} samples")
    print(f"  Val:   {len(val_loader.dataset):>6} samples")
    print(f"  Test:  {len(test_loader.dataset):>6} samples")

    # --- Model ---
    model = build_model(model_params)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_params.model_type}")
    print(f"Total trainable parameters: {num_params:,}")

    # Count FLOPs
    if data_params.resize_to is not None:
        input_size = (3, data_params.resize_to, data_params.resize_to)
    else:
        input_size = (3, 32, 32)

    try:
        flops, params = count_flops(model, input_size=input_size, device="cpu", verbose=False)
        print(f"FLOPs: {flops:.2f} M")
        print(f"Parameters: {params:.2f} M")
    except Exception as e:
        print(f"Could not count FLOPs: {e}")

    # --- Teacher Model (for distillation) ---
    teacher_model = None
    if train_params.use_distillation and train_params.teacher_model_path is not None:
        if not os.path.exists(train_params.teacher_model_path):
            print(f"\nError: Teacher model not found at {train_params.teacher_model_path}")
            sys.exit(1)

        teacher_model = load_teacher_model(
            train_params.teacher_model_path,
            model_params.num_classes,
            device,
        )

    # --- Training ---
    if train_params.mode in ("train", "both"):
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}\n")

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            params=train_params,
            teacher_model=teacher_model,
        )

        # Save history
        import json
        history_path = os.path.join(out_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")

        # Plot training curves
        plot_path = os.path.join(out_dir, "training_curves.png")
        plot_training_history(
            history,
            title=f"{model_params.model_type} - {train_params.task}",
            save_path=plot_path,
        )

    # --- Testing ---
    if train_params.mode in ("test", "both"):
        print(f"\n{'='*80}")
        print(f"Testing")
        print(f"{'='*80}\n")

        if os.path.exists(checkpoint_path):
            results = load_and_test(
                model=model,
                checkpoint_path=checkpoint_path,
                test_loader=test_loader,
                device=device,
                num_classes=model_params.num_classes,
                class_names=CIFAR10_CLASSES if data_params.dataset == "cifar10" else None,
            )

            # Save test results
            import json
            results_path = os.path.join(out_dir, "test_results.json")
            json_results = {
                "loss": float(results["loss"]),
                "accuracy": float(results["accuracy"]),
                "class_accuracy": results["class_accuracy"].tolist(),
            }
            with open(results_path, "w") as f:
                json.dump(json_results, f, indent=2)
            print(f"\nTest results saved to: {results_path}")

            # Plot confusion matrix
            cm_path = os.path.join(out_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                results["confusion_matrix"],
                class_names=CIFAR10_CLASSES if data_params.dataset == "cifar10" else None,
                title=f"{model_params.model_type} - Confusion Matrix",
                save_path=cm_path,
            )
        else:
            print(f"No checkpoint found at: {checkpoint_path}")
            print("Please train the model first.")

    print(f"\n{'='*80}")
    print(f"Done!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
