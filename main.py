"""
main.py

Entry point for HW2: Data Augmentation and Adversarial Robustness on CIFAR-10.

Orchestrates five main tasks:
  1. Evaluate fine-tuned model robustness on CIFAR-10-C
  2. Fine-tune with AugMix, compare clean & corrupted accuracy
  3. PGD adversarial attacks (L-inf, L2) + Grad-CAM + t-SNE
  4. Knowledge distillation with AugMix teacher
  5. Adversarial transferability from teacher to student
"""

import os
import json

import torch

from parameters import (
    get_params,
    DataParams,
    AugMixParams,
    ModelParams,
    TrainParams,
    AttackParams,
    VisParams,
)
from utils.augmix import get_augmix_dataloaders
from utils.visualization import (
    plot_training_history,
    plot_corruption_results,
    plot_adversarial_comparison,
    CIFAR10_CLASSES,
)
from utils.data_cifar import DataConfig, get_dataloaders
from utils.flops_counter import count_flops

from models.simple_cnn import SimpleCNN, SimpleCNNConfig
from models.resnet_cifar import ResNetCIFAR, ResNetCIFARConfig, resnet18_cifar

from train import train
from test import (
    test_clean,
    test_corruptions,
    test_adversarial,
    test_gradcam_adversarial,
    test_tsne_adversarial,
    test_transferability,
)


def build_data_config(data_params: DataParams) -> DataConfig:
    """
    Converts DataParams to DataConfig for standard data loading.

    Args:
        data_params: DataParams from argparse.

    Returns:
        DataConfig for CIFAR data loaders.
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

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Loads model weights from a checkpoint file.

    Args:
        model: Model architecture (uninitialized weights).
        checkpoint_path: Path to checkpoint file.
        device: Compute device.

    Returns:
        Model with loaded weights.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def save_results(results: dict, path: str) -> None:
    """
    Saves results dictionary to a JSON file.

    Args:
        results: Dictionary to save.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {path}")


def run_finetune(
    model_params: ModelParams,
    data_params: DataParams,
    train_params: TrainParams,
    use_augmix: bool = False,
    augmix_params: AugMixParams = None,
) -> None:
    """
    Fine-tunes a model with standard or AugMix augmentation.

    Args:
        model_params: Model configuration.
        data_params: Data configuration.
        train_params: Training configuration.
        use_augmix: Whether to use AugMix augmentation.
        augmix_params: AugMix configuration (required if use_augmix=True).
    """
    device = torch.device(train_params.device)
    suffix = "augmix" if use_augmix else "standard"

    out_dir = os.path.join(train_params.save_path, suffix)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "best_model.pth")
    train_params_copy = TrainParams(**{
        k: v for k, v in train_params.__dict__.items()
    })
    train_params_copy.save_path = checkpoint_path

    print(f"\n{'='*80}")
    print(f"Fine-tuning with {suffix.upper()} augmentation")
    print(f"{'='*80}\n")

    if use_augmix and augmix_params is not None:
        train_loader, val_loader, test_loader = get_augmix_dataloaders(
            data_dir=data_params.data_dir,
            batch_size=data_params.batch_size,
            val_split=data_params.val_split,
            num_workers=data_params.num_workers,
            severity=augmix_params.severity,
            width=augmix_params.width,
            depth=augmix_params.depth,
            alpha=augmix_params.alpha,
            seed=data_params.seed,
        )
    else:
        data_config = build_data_config(data_params)
        train_loader, val_loader, test_loader = get_dataloaders(data_config)

    model = build_model(model_params)
    model = model.to(device)

    flops_m, params_m = count_flops(model, input_size=(3, 32, 32), verbose=False)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_params.model_type} | Params: {params_m:.2f}M | FLOPs: {flops_m:.2f}M")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        params=train_params_copy,
    )

    history_path = os.path.join(out_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_path = os.path.join(out_dir, "training_curves.png")
    plot_training_history(
        history,
        title=f"{model_params.model_type} - {suffix}",
        save_path=plot_path,
    )


def run_corruption_test(
    model_params: ModelParams,
    data_params: DataParams,
    train_params: TrainParams,
    vis_params: VisParams,
    checkpoint_standard: str,
    checkpoint_augmix: str,
) -> None:
    """
    Tests both standard and AugMix models on CIFAR-10-C corruptions.

    Args:
        model_params: Model configuration.
        data_params: Data configuration.
        train_params: Training configuration.
        vis_params: Visualization configuration.
        checkpoint_standard: Path to standard fine-tuned model.
        checkpoint_augmix: Path to AugMix fine-tuned model.
    """
    device = torch.device(train_params.device)

    print(f"\n{'='*80}")
    print("CIFAR-10-C Corruption Robustness Test")
    print(f"{'='*80}")

    data_config = build_data_config(data_params)
    _, _, test_loader = get_dataloaders(data_config)

    all_corruption_results = {}

    for name, ckpt_path in [("Standard", checkpoint_standard), ("AugMix", checkpoint_augmix)]:
        if not os.path.exists(ckpt_path):
            print(f"\n  Checkpoint not found for {name}: {ckpt_path}")
            continue

        print(f"\n--- {name} Model ---")
        model = build_model(model_params)
        model = load_model_checkpoint(model, ckpt_path, device)

        _, clean_acc = test_clean(model, test_loader, device)

        corruption_results = test_corruptions(
            model, data_params.cifar10c_dir, device,
            severity=5, batch_size=data_params.batch_size,
            num_workers=data_params.num_workers,
        )
        corruption_results["clean"] = clean_acc
        all_corruption_results[name] = corruption_results

    if all_corruption_results:
        os.makedirs(vis_params.output_dir, exist_ok=True)
        save_results(all_corruption_results, os.path.join(vis_params.output_dir, "corruption_results.json"))

        plot_corruption_results(
            all_corruption_results,
            title="CIFAR-10-C Robustness: Standard vs AugMix",
            save_path=os.path.join(vis_params.output_dir, "corruption_comparison.png"),
        )


def run_adversarial_test(
    model_params: ModelParams,
    data_params: DataParams,
    train_params: TrainParams,
    attack_params: AttackParams,
    vis_params: VisParams,
    checkpoint_standard: str,
    checkpoint_augmix: str,
) -> None:
    """
    Tests adversarial robustness for both models, generates Grad-CAM and t-SNE plots.

    Args:
        model_params: Model configuration.
        data_params: Data configuration.
        train_params: Training configuration.
        attack_params: Attack configuration.
        vis_params: Visualization configuration.
        checkpoint_standard: Path to standard fine-tuned model.
        checkpoint_augmix: Path to AugMix fine-tuned model.
    """
    device = torch.device(train_params.device)

    print(f"\n{'='*80}")
    print("Adversarial Robustness Test (PGD)")
    print(f"{'='*80}")

    data_config = build_data_config(data_params)
    _, _, test_loader = get_dataloaders(data_config)

    all_adv_results = {}

    for name, ckpt_path in [("Standard", checkpoint_standard), ("AugMix", checkpoint_augmix)]:
        if not os.path.exists(ckpt_path):
            print(f"\n  Checkpoint not found for {name}: {ckpt_path}")
            continue

        print(f"\n--- {name} Model ---")
        model = build_model(model_params)
        model = load_model_checkpoint(model, ckpt_path, device)

        adv_results = test_adversarial(model, test_loader, device, attack_params)
        all_adv_results[name] = adv_results

        test_gradcam_adversarial(
            model, test_loader, device, attack_params, vis_params,
            model_name=name.lower(),
        )

        test_tsne_adversarial(
            model, test_loader, device, attack_params, vis_params,
            model_name=name.lower(),
        )

    if all_adv_results:
        os.makedirs(vis_params.output_dir, exist_ok=True)
        save_results(all_adv_results, os.path.join(vis_params.output_dir, "adversarial_results.json"))

        plot_adversarial_comparison(
            all_adv_results,
            title="Adversarial Robustness: Standard vs AugMix",
            save_path=os.path.join(vis_params.output_dir, "adversarial_comparison.png"),
        )


def run_distillation_augmix(
    model_params: ModelParams,
    data_params: DataParams,
    train_params: TrainParams,
    augmix_params: AugMixParams,
    teacher_checkpoint: str,
) -> None:
    """
    Knowledge distillation using an AugMix-trained teacher.

    Args:
        model_params: Model configuration.
        data_params: Data configuration.
        train_params: Training configuration (with distillation enabled).
        augmix_params: AugMix configuration.
        teacher_checkpoint: Path to AugMix teacher checkpoint.
    """
    device = torch.device(train_params.device)

    print(f"\n{'='*80}")
    print("Knowledge Distillation with AugMix Teacher")
    print(f"{'='*80}")

    if not os.path.exists(teacher_checkpoint):
        print(f"Error: Teacher checkpoint not found: {teacher_checkpoint}")
        return

    teacher_model = build_model(model_params)
    teacher_model = load_model_checkpoint(teacher_model, teacher_checkpoint, device)

    student_config = SimpleCNNConfig(
        num_classes=model_params.num_classes,
        channels=[32, 64, 128],
        hidden_size=256,
        dropout=0.3,
    )
    student_model = SimpleCNN(student_config)
    student_model = student_model.to(device)

    data_config = build_data_config(data_params)
    train_loader, val_loader, test_loader = get_dataloaders(data_config)

    out_dir = os.path.join(train_params.save_path, "distillation_augmix_teacher")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "best_model.pth")

    distill_params = TrainParams(**{k: v for k, v in train_params.__dict__.items()})
    distill_params.save_path = checkpoint_path
    distill_params.use_distillation = True

    print(f"\nTeacher: {model_params.model_type} (AugMix-trained)")
    print(f"Student: SimpleCNN")

    history = train(
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        params=distill_params,
        teacher_model=teacher_model,
    )

    history_path = os.path.join(out_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_path = os.path.join(out_dir, "training_curves.png")
    plot_training_history(
        history,
        title="KD with AugMix Teacher",
        save_path=plot_path,
    )


def run_transferability_test(
    model_params: ModelParams,
    data_params: DataParams,
    train_params: TrainParams,
    attack_params: AttackParams,
    vis_params: VisParams,
    teacher_checkpoint: str,
    student_checkpoint: str,
) -> None:
    """
    Tests adversarial transferability from teacher to student.

    Args:
        model_params: Model configuration.
        data_params: Data configuration.
        train_params: Training configuration.
        attack_params: Attack configuration.
        vis_params: Visualization configuration.
        teacher_checkpoint: Path to teacher model checkpoint.
        student_checkpoint: Path to student model checkpoint.
    """
    device = torch.device(train_params.device)

    print(f"\n{'='*80}")
    print("Adversarial Transferability Test")
    print(f"{'='*80}")

    if not os.path.exists(teacher_checkpoint):
        print(f"Error: Teacher checkpoint not found: {teacher_checkpoint}")
        return
    if not os.path.exists(student_checkpoint):
        print(f"Error: Student checkpoint not found: {student_checkpoint}")
        return

    teacher_model = build_model(model_params)
    teacher_model = load_model_checkpoint(teacher_model, teacher_checkpoint, device)

    student_config = SimpleCNNConfig(num_classes=model_params.num_classes)
    student_model = SimpleCNN(student_config)
    student_model = load_model_checkpoint(student_model, student_checkpoint, device)

    data_config = build_data_config(data_params)
    _, _, test_loader = get_dataloaders(data_config)

    transfer_results = test_transferability(
        teacher_model, student_model, test_loader, device, attack_params,
    )

    os.makedirs(vis_params.output_dir, exist_ok=True)
    save_results(transfer_results, os.path.join(vis_params.output_dir, "transferability_results.json"))


def main() -> None:
    """
    Main entry point for HW2.

    Parses arguments and dispatches to appropriate task handler(s).
    """
    data_params, augmix_params, model_params, train_params, attack_params, vis_params = get_params()

    if train_params.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        train_params.device = "cpu"

    device = torch.device(train_params.device)
    torch.manual_seed(data_params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(data_params.seed)

    print(f"\n{'='*80}")
    print("HW2: Data Augmentation and Adversarial Robustness")
    print(f"{'='*80}")
    print(f"Task:   {train_params.task}")
    print(f"Mode:   {train_params.mode}")
    print(f"Model:  {model_params.model_type}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    base_save = train_params.save_path
    ckpt_standard = os.path.join(base_save, "standard", "best_model.pth")
    ckpt_augmix = os.path.join(base_save, "augmix", "best_model.pth")
    ckpt_student = os.path.join(base_save, "distillation_augmix_teacher", "best_model.pth")

    task = train_params.task

    if task == "finetune":
        run_finetune(model_params, data_params, train_params, use_augmix=False)

    elif task == "finetune_augmix":
        run_finetune(model_params, data_params, train_params, use_augmix=True, augmix_params=augmix_params)

    elif task == "test_corruption":
        run_corruption_test(
            model_params, data_params, train_params, vis_params,
            ckpt_standard, ckpt_augmix,
        )

    elif task == "test_adversarial":
        run_adversarial_test(
            model_params, data_params, train_params, attack_params, vis_params,
            ckpt_standard, ckpt_augmix,
        )

    elif task == "distillation_augmix":
        run_distillation_augmix(
            model_params, data_params, train_params, augmix_params,
            teacher_checkpoint=ckpt_augmix,
        )

    elif task == "transferability":
        run_transferability_test(
            model_params, data_params, train_params, attack_params, vis_params,
            teacher_checkpoint=ckpt_augmix,
            student_checkpoint=ckpt_student,
        )

    else:
        print(f"Unknown task: {task}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
