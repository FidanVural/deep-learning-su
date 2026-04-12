"""
parameters.py

Argument parsing and configuration dataclasses for HW2:
Data Augmentation, Adversarial Robustness, and Transferability on CIFAR-10.

Uses argparse for CLI arguments and dataclasses for typed parameter groups.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataParams:
    """Configuration for dataset loading and preprocessing."""

    data_dir: str = "./data"
    cifar10c_dir: str = "./data/CIFAR-10-C"
    dataset: str = "cifar10"
    batch_size: int = 128
    val_split: float = 0.1
    num_workers: int = 4
    resize_to: Optional[int] = None
    normalize: bool = True
    augment: bool = True
    use_augmix: bool = False
    seed: int = 42


@dataclass
class AugMixParams:
    """Configuration for AugMix data augmentation."""

    severity: int = 3
    width: int = 3
    depth: int = -1
    alpha: float = 1.0


@dataclass
class ModelParams:
    """Configuration for model architecture."""

    model_type: str = "transfer_resnet"
    num_classes: int = 10
    pretrained: bool = True
    freeze_until: Optional[str] = None
    modify_first_conv: bool = False
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_hidden_size: int = 256
    dropout: float = 0.3


@dataclass
class TrainParams:
    """Configuration for the training loop."""

    mode: str = "train"
    task: str = "finetune"

    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"

    scheduler: str = "cosine"
    step_size: int = 30
    warmup_epochs: int = 5

    label_smoothing: float = 0.0

    use_distillation: bool = False
    use_custom_distillation: bool = False
    teacher_model_path: Optional[str] = None
    temperature: float = 4.0
    alpha: float = 0.5

    patience: int = 15

    device: str = "cuda"
    save_path: str = "hw2/outputs/default"
    log_interval: int = 50
    save_best_only: bool = True
    eval_interval: int = 1


@dataclass
class AttackParams:
    """Configuration for adversarial attacks."""

    attack_type: str = "pgd"
    pgd_steps: int = 20
    eps_linf: float = 4.0 / 255.0
    eps_l2: float = 0.25
    step_size_linf: float = 1.0 / 255.0
    step_size_l2: float = 0.05
    random_start: bool = True


@dataclass
class VisParams:
    """Configuration for visualization tasks."""

    num_gradcam_samples: int = 2
    tsne_max_samples: int = 5000
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    output_dir: str = "hw2/outputs/plots"


def get_params() -> Tuple[DataParams, AugMixParams, ModelParams, TrainParams, AttackParams, VisParams]:
    """
    Parses CLI arguments and returns typed parameter dataclasses.

    Returns:
        Tuple of (DataParams, AugMixParams, ModelParams, TrainParams, AttackParams, VisParams).
    """

    parser = argparse.ArgumentParser(
        description="HW2: Data Augmentation and Adversarial Robustness on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Mode and Task ---
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both", "attack", "visualize", "all"],
        default="all",
        help="Run mode",
    )
    parser.add_argument(
        "--task",
        choices=[
            "finetune",
            "finetune_augmix",
            "test_corruption",
            "test_adversarial",
            "distillation_augmix",
            "transferability",
        ],
        default="finetune",
        help="Task type",
    )

    # --- Data ---
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cifar10c_dir", type=str, default="./data/CIFAR-10-C")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize_to", type=int, default=None)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--use_augmix", action="store_true", help="Enable AugMix augmentation")
    parser.add_argument("--seed", type=int, default=42)

    # --- AugMix ---
    parser.add_argument("--augmix_severity", type=int, default=3, help="AugMix severity (1-10)")
    parser.add_argument("--augmix_width", type=int, default=3, help="AugMix chain width")
    parser.add_argument("--augmix_depth", type=int, default=-1, help="AugMix chain depth (-1 for random)")
    parser.add_argument("--augmix_alpha", type=float, default=1.0, help="Dirichlet alpha for AugMix mixing")

    # --- Model ---
    parser.add_argument(
        "--model_type",
        choices=["simple_cnn", "resnet18", "mobilenetv2", "transfer_resnet"],
        default="transfer_resnet",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_until", choices=["layer1", "layer2", "layer3", "layer4"], default=None)
    parser.add_argument("--modify_first_conv", action="store_true")
    parser.add_argument("--cnn_channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--cnn_hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["step", "cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--step_size", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # --- Knowledge Distillation ---
    parser.add_argument("--use_distillation", action="store_true")
    parser.add_argument("--use_custom_distillation", action="store_true")
    parser.add_argument("--teacher_model_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)

    # --- Early Stopping ---
    parser.add_argument("--patience", type=int, default=15)

    # --- Adversarial Attack ---
    parser.add_argument("--attack_type", choices=["pgd"], default="pgd")
    parser.add_argument("--pgd_steps", type=int, default=20, help="Number of PGD steps")
    parser.add_argument("--eps_linf", type=float, default=4.0 / 255.0, help="L-inf epsilon")
    parser.add_argument("--eps_l2", type=float, default=0.25, help="L2 epsilon")
    parser.add_argument("--step_size_linf", type=float, default=1.0 / 255.0, help="PGD step size for L-inf")
    parser.add_argument("--step_size_l2", type=float, default=0.05, help="PGD step size for L2")
    parser.add_argument("--no_random_start", action="store_true")

    # --- Visualization ---
    parser.add_argument("--num_gradcam_samples", type=int, default=2)
    parser.add_argument("--tsne_max_samples", type=int, default=5000)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_n_iter", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="hw2/outputs/plots")

    # --- System ---
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="hw2/outputs/default")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a pre-trained model checkpoint")

    args = parser.parse_args()

    data_params = DataParams(
        data_dir=args.data_dir,
        cifar10c_dir=args.cifar10c_dir,
        dataset="cifar10",
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        resize_to=args.resize_to,
        normalize=True,
        augment=not args.no_augment,
        use_augmix=args.use_augmix,
        seed=args.seed,
    )

    augmix_params = AugMixParams(
        severity=args.augmix_severity,
        width=args.augmix_width,
        depth=args.augmix_depth,
        alpha=args.augmix_alpha,
    )

    model_params = ModelParams(
        model_type=args.model_type,
        num_classes=10,
        pretrained=args.pretrained,
        freeze_until=args.freeze_until,
        modify_first_conv=args.modify_first_conv,
        cnn_channels=args.cnn_channels,
        cnn_hidden_size=args.cnn_hidden_size,
        dropout=args.dropout,
    )

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

    attack_params = AttackParams(
        attack_type=args.attack_type,
        pgd_steps=args.pgd_steps,
        eps_linf=args.eps_linf,
        eps_l2=args.eps_l2,
        step_size_linf=args.step_size_linf,
        step_size_l2=args.step_size_l2,
        random_start=not args.no_random_start,
    )

    vis_params = VisParams(
        num_gradcam_samples=args.num_gradcam_samples,
        tsne_max_samples=args.tsne_max_samples,
        tsne_perplexity=args.tsne_perplexity,
        tsne_n_iter=args.tsne_n_iter,
        output_dir=args.output_dir,
    )

    return data_params, augmix_params, model_params, train_params, attack_params, vis_params
