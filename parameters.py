"""
parameters.py

Argument parsing and configuration dataclasses for HW1b:
Transfer Learning and Knowledge Distillation on CIFAR-10.

Uses argparse for CLI arguments and dataclasses for typed parameter groups.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Parameter Dataclasses
@dataclass
class DataParams:
    """Configuration for dataset loading and preprocessing."""

    data_dir: str = "./data"
    dataset: str = "cifar10"  # cifar10 or cifar100
    batch_size: int = 128
    val_split: float = 0.1
    num_workers: int = 4
    resize_to: Optional[int] = None  # None or 224 for transfer learning
    normalize: bool = True
    augment: bool = True
    seed: int = 42


@dataclass
class ModelParams:
    """Configuration for model architecture."""

    model_type: str = "simple_cnn"  # simple_cnn, resnet18, mobilenetv2, transfer_resnet
    num_classes: int = 10
    pretrained: bool = False

    # Transfer learning specific
    freeze_until: Optional[str] = None  # None, "layer1", "layer2", "layer3", "layer4"
    modify_first_conv: bool = False  # True to adapt conv1 for 32x32 inputs

    # SimpleCNN specific
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_hidden_size: int = 256
    dropout: float = 0.3


@dataclass
class TrainParams:
    """Configuration for the training loop."""

    mode: str = "train"  # train, test, both
    task: str = "baseline"  # baseline, transfer, distillation

    # Training hyperparameters
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"  # adamw, sgd

    # Learning rate scheduling
    scheduler: str = "cosine"  # step, cosine, plateau, none
    step_size: int = 30
    warmup_epochs: int = 5

    # Regularization
    label_smoothing: float = 0.0  # 0.0 = no smoothing, typical: 0.1

    # Knowledge Distillation specific
    use_distillation: bool = False
    use_custom_distillation: bool = False
    teacher_model_path: Optional[str] = None
    temperature: float = 4.0
    alpha: float = 0.5  # weight for distillation loss vs hard target loss

    # Early stopping
    patience: int = 15

    # System
    device: str = "cuda"
    save_path: str = "hw1b/outputs/default"
    log_interval: int = 50
    save_best_only: bool = True

    # Evaluation
    eval_interval: int = 1  # evaluate every N epochs


# Argument Parser
def get_params() -> Tuple[DataParams, ModelParams, TrainParams]:
    """
    Parses CLI arguments and returns typed parameter dataclasses.

    Returns:
        Tuple of (DataParams, ModelParams, TrainParams).
    """

    parser = argparse.ArgumentParser(
        description="HW1b: Transfer Learning and Knowledge Distillation on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Mode and Task ---
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="train",
        help="Run mode: train, test, or both"
    )
    parser.add_argument(
        "--task",
        choices=["baseline", "transfer", "distillation"],
        default="baseline",
        help="Task type: baseline training, transfer learning, or knowledge distillation"
    )

    # --- Data ---
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resize_to", type=int, default=None, help="Resize images to this size (for transfer learning)")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Model ---
    parser.add_argument(
        "--model_type",
        choices=["simple_cnn", "resnet18", "mobilenetv2", "transfer_resnet"],
        default="simple_cnn",
        help="Model architecture"
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights (for transfer learning)")
    parser.add_argument(
        "--freeze_until",
        choices=["layer1", "layer2", "layer3", "layer4"],
        default=None,
        help="Freeze layers up to and including this layer"
    )
    parser.add_argument(
        "--modify_first_conv",
        action="store_true",
        help="Modify first conv layer for 32x32 inputs (transfer learning)"
    )
    parser.add_argument(
        "--cnn_channels",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Channel sizes for SimpleCNN"
    )
    parser.add_argument("--cnn_hidden_size", type=int, default=256, help="Hidden size for SimpleCNN FC layer")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="Optimizer")
    parser.add_argument(
        "--scheduler",
        choices=["step", "cosine", "plateau", "none"],
        default="cosine",
        help="LR scheduler"
    )
    parser.add_argument("--step_size", type=int, default=30, help="StepLR step size")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for LR")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor (0.0-1.0)")

    # --- Knowledge Distillation ---
    parser.add_argument("--use_distillation", action="store_true", help="Enable knowledge distillation")
    parser.add_argument("--use_custom_distillation", action="store_true", help="Use custom distillation (true-class probability from teacher, equal for others)")
    parser.add_argument("--teacher_model_path", type=str, default=None, help="Path to teacher model checkpoint")
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Distillation loss weight (0-1)")

    # --- Early Stopping ---
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # --- System ---
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu or cuda")
    parser.add_argument("--save_path", type=str, default="hw1b/outputs/default", help="Output directory")
    parser.add_argument("--log_interval", type=int, default=50, help="Batch log frequency")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation epoch interval")

    args = parser.parse_args()

    # Build DataParams
    data_params = DataParams(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        resize_to=args.resize_to,
        normalize=True,
        augment=not args.no_augment,
        seed=args.seed,
    )

    # Build ModelParams
    model_params = ModelParams(
        model_type=args.model_type,
        num_classes=100 if args.dataset == "cifar100" else 10,
        pretrained=args.pretrained,
        freeze_until=args.freeze_until,
        modify_first_conv=args.modify_first_conv,
        cnn_channels=args.cnn_channels,
        cnn_hidden_size=args.cnn_hidden_size,
        dropout=args.dropout,
    )

    # Build TrainParams
    train_params = TrainParams(
        mode=args.mode,
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        step_size=args.step_size,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        use_distillation=args.use_distillation,
        use_custom_distillation=args.use_custom_distillation,
        teacher_model_path=args.teacher_model_path,
        temperature=args.temperature,
        alpha=args.alpha,
        patience=args.patience,
        device=args.device,
        save_path=args.save_path,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )

    return data_params, model_params, train_params
