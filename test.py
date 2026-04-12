"""
test.py

Evaluation and testing functions for HW2.

Features:
  - Clean accuracy evaluation
  - CIFAR-10-C corruption robustness evaluation
  - PGD adversarial robustness evaluation
  - Grad-CAM adversarial visualization
  - t-SNE adversarial embedding visualization
  - Adversarial transferability evaluation
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.cifar10c import evaluate_corruptions
from utils.pgd_attack import PGDAttack, generate_adversarial_samples
from utils.gradcam import GradCAM, _get_target_layer, visualize_gradcam
from utils.visualization import (
    extract_features,
    plot_tsne_adversarial,
    plot_gradcam_comparison,
    CIFAR10_CLASSES,
)
from parameters import AttackParams, VisParams


@torch.no_grad()
def test_clean(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates model accuracy on clean test set.

    Args:
        model: Trained model.
        test_loader: DataLoader for clean test data.
        device: Compute device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    print(f"  Clean Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return avg_loss, accuracy


def test_corruptions(
    model: nn.Module,
    cifar10c_dir: str,
    device: torch.device,
    severity: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Evaluates model on all CIFAR-10-C corruptions.

    Args:
        model: Trained model.
        cifar10c_dir: Directory with CIFAR-10-C .npy files.
        device: Compute device.
        severity: Corruption severity level (1-5).
        batch_size: Batch size.
        num_workers: DataLoader workers.

    Returns:
        Dictionary mapping corruption name to accuracy.
    """
    model = model.to(device)
    model.eval()

    print(f"\nEvaluating CIFAR-10-C corruptions (severity={severity}):")
    print(f"{'-'*60}")

    results = evaluate_corruptions(
        model, cifar10c_dir, device, severity, batch_size, num_workers,
    )
    return results


def test_adversarial(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_params: AttackParams,
) -> Dict[str, float]:
    """
    Evaluates adversarial robustness under PGD attacks.

    Tests both L-inf and L2 norms.

    Args:
        model: Trained model.
        test_loader: DataLoader for clean test data.
        device: Compute device.
        attack_params: AttackParams configuration.

    Returns:
        Dictionary with 'clean_acc', 'adv_acc_linf', 'adv_acc_l2'.
    """
    model = model.to(device)
    model.eval()

    results: Dict[str, float] = {}

    # L-inf attack
    print(f"\nPGD-{attack_params.pgd_steps} L-inf (eps={attack_params.eps_linf:.4f}):")
    attacker_linf = PGDAttack(
        model=model,
        eps=attack_params.eps_linf,
        step_size=attack_params.step_size_linf,
        steps=attack_params.pgd_steps,
        norm="linf",
        random_start=attack_params.random_start,
    )
    linf_results = attacker_linf.evaluate(test_loader, device)
    results["clean_acc"] = linf_results["clean_acc"]
    results["adv_acc_linf"] = linf_results["adv_acc"]
    print(f"  Clean: {linf_results['clean_acc']:.4f} | Adv: {linf_results['adv_acc']:.4f}")

    # L2 attack
    print(f"\nPGD-{attack_params.pgd_steps} L2 (eps={attack_params.eps_l2:.4f}):")
    attacker_l2 = PGDAttack(
        model=model,
        eps=attack_params.eps_l2,
        step_size=attack_params.step_size_l2,
        steps=attack_params.pgd_steps,
        norm="l2",
        random_start=attack_params.random_start,
    )
    l2_results = attacker_l2.evaluate(test_loader, device)
    results["adv_acc_l2"] = l2_results["adv_acc"]
    print(f"  Clean: {l2_results['clean_acc']:.4f} | Adv: {l2_results['adv_acc']:.4f}")

    return results


def test_gradcam_adversarial(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_params: AttackParams,
    vis_params: VisParams,
    model_name: str = "model",
) -> None:
    """
    Generates Grad-CAM visualizations for clean vs adversarial samples.

    Finds samples where the model is correct on clean but wrong on adversarial,
    then shows Grad-CAM heatmaps for both.

    Args:
        model: Trained model.
        test_loader: DataLoader for clean test data.
        device: Compute device.
        attack_params: Attack configuration.
        vis_params: Visualization configuration.
        model_name: Name for file saving.
    """
    model = model.to(device)
    model.eval()

    attacker = PGDAttack(
        model=model,
        eps=attack_params.eps_linf,
        step_size=attack_params.step_size_linf,
        steps=attack_params.pgd_steps,
        norm="linf",
        random_start=attack_params.random_start,
    )

    target_layer = _get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    clean_imgs_collect = []
    adv_imgs_collect = []
    labels_collect = []
    clean_preds_collect = []
    adv_preds_collect = []
    clean_cams_collect = []
    adv_cams_collect = []

    num_needed = vis_params.num_gradcam_samples

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)

        adv_data = attacker.attack(data, target)

        with torch.no_grad():
            adv_output = model(adv_data)
            adv_pred = adv_output.argmax(dim=1)

        misclassified = (clean_pred == target) & (adv_pred != target)
        indices = misclassified.nonzero(as_tuple=True)[0]

        for idx in indices:
            if len(clean_imgs_collect) >= num_needed:
                break

            i = idx.item()
            clean_cam = grad_cam.generate(data[i].unsqueeze(0), target_class=clean_pred[i].item())
            adv_cam = grad_cam.generate(adv_data[i].unsqueeze(0), target_class=adv_pred[i].item())

            clean_imgs_collect.append(data[i].cpu())
            adv_imgs_collect.append(adv_data[i].cpu())
            labels_collect.append(target[i].cpu())
            clean_preds_collect.append(clean_pred[i].cpu())
            adv_preds_collect.append(adv_pred[i].cpu())
            clean_cams_collect.append(clean_cam)
            adv_cams_collect.append(adv_cam)

        if len(clean_imgs_collect) >= num_needed:
            break

    if clean_imgs_collect:
        os.makedirs(vis_params.output_dir, exist_ok=True)
        save_path = os.path.join(vis_params.output_dir, f"gradcam_{model_name}.png")

        plot_gradcam_comparison(
            clean_images=torch.stack(clean_imgs_collect),
            adv_images=torch.stack(adv_imgs_collect),
            clean_cams=clean_cams_collect,
            adv_cams=adv_cams_collect,
            labels=torch.stack(labels_collect),
            clean_preds=torch.stack(clean_preds_collect),
            adv_preds=torch.stack(adv_preds_collect),
            class_names=CIFAR10_CLASSES,
            title=f"Grad-CAM: Clean vs Adversarial ({model_name})",
            save_path=save_path,
        )
    else:
        print("  Warning: No misclassified adversarial samples found for Grad-CAM.")


def test_tsne_adversarial(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_params: AttackParams,
    vis_params: VisParams,
    model_name: str = "model",
) -> None:
    """
    Generates t-SNE visualization of clean vs adversarial sample embeddings.

    Args:
        model: Trained model.
        test_loader: DataLoader for clean test data.
        device: Compute device.
        attack_params: Attack configuration.
        vis_params: Visualization configuration.
        model_name: Name for file saving.
    """
    model = model.to(device)
    model.eval()

    print(f"\nGenerating adversarial samples for t-SNE ({model_name})...")

    clean_images, adv_images, labels, _ = generate_adversarial_samples(
        model=model,
        loader=test_loader,
        device=device,
        eps=attack_params.eps_linf,
        step_size=attack_params.step_size_linf,
        steps=attack_params.pgd_steps,
        norm="linf",
        max_batches=10,
    )

    print("Extracting features...")
    clean_features = extract_features(model, clean_images, device)
    adv_features = extract_features(model, adv_images, device)

    os.makedirs(vis_params.output_dir, exist_ok=True)
    save_path = os.path.join(vis_params.output_dir, f"tsne_adversarial_{model_name}.png")

    plot_tsne_adversarial(
        clean_features=clean_features,
        adv_features=adv_features,
        labels=labels.numpy(),
        class_names=CIFAR10_CLASSES,
        title=f"t-SNE: Clean vs Adversarial ({model_name})",
        save_path=save_path,
        max_samples=vis_params.tsne_max_samples,
        perplexity=vis_params.tsne_perplexity,
        n_iter=vis_params.tsne_n_iter,
    )


def test_transferability(
    teacher_model: nn.Module,
    student_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_params: AttackParams,
) -> Dict[str, float]:
    """
    Tests adversarial transferability from teacher to student.

    Generates adversarial samples using PGD on the teacher model,
    then evaluates them on the student model.

    Args:
        teacher_model: Model used to generate adversarial examples.
        student_model: Model to evaluate adversarial examples on.
        test_loader: DataLoader for clean test data.
        device: Compute device.
        attack_params: Attack configuration.

    Returns:
        Dictionary with transfer attack results.
    """
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    teacher_model.eval()
    student_model.eval()

    print(f"\nAdversarial Transferability Test:")
    print(f"  Generating PGD-{attack_params.pgd_steps} L-inf (eps={attack_params.eps_linf:.4f}) on teacher...")

    clean_images, adv_images, labels, teacher_adv_preds = generate_adversarial_samples(
        model=teacher_model,
        loader=test_loader,
        device=device,
        eps=attack_params.eps_linf,
        step_size=attack_params.step_size_linf,
        steps=attack_params.pgd_steps,
        norm="linf",
    )

    teacher_adv_correct = (teacher_adv_preds == labels).sum().item()
    teacher_adv_acc = teacher_adv_correct / len(labels)

    student_correct_clean = 0
    student_correct_adv = 0
    total = len(labels)

    with torch.no_grad():
        batch_size = 256
        for i in range(0, total, batch_size):
            clean_batch = clean_images[i : i + batch_size].to(device)
            adv_batch = adv_images[i : i + batch_size].to(device)
            label_batch = labels[i : i + batch_size].to(device)

            clean_preds = student_model(clean_batch).argmax(dim=1)
            adv_preds = student_model(adv_batch).argmax(dim=1)

            student_correct_clean += (clean_preds == label_batch).sum().item()
            student_correct_adv += (adv_preds == label_batch).sum().item()

    student_clean_acc = student_correct_clean / total
    student_adv_acc = student_correct_adv / total

    results = {
        "teacher_adv_acc": teacher_adv_acc,
        "student_clean_acc": student_clean_acc,
        "student_adv_acc_transfer": student_adv_acc,
        "transfer_attack_success": 1.0 - student_adv_acc,
    }

    print(f"\n  Teacher Adv Accuracy:      {teacher_adv_acc:.4f} ({teacher_adv_acc*100:.2f}%)")
    print(f"  Student Clean Accuracy:    {student_clean_acc:.4f} ({student_clean_acc*100:.2f}%)")
    print(f"  Student Transfer Adv Acc:  {student_adv_acc:.4f} ({student_adv_acc*100:.2f}%)")
    print(f"  Transfer Attack Success:   {results['transfer_attack_success']:.4f}")

    return results
