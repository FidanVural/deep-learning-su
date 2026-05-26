"""
test.py

Evaluation and testing for HW4 Part 1.
Loads trained model checkpoints and evaluates on the test set.

Supports:
  - Part (b): LSTM/GRU return prediction evaluation
  - Part (c): Rolling average forecasting evaluation
  - Part (d): Turning point detection evaluation (BiLSTM)
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

from dataset import get_dataloaders, prepare_datasets, TICKERS
from models import StockLSTM, StockGRU, StockBiLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="HW4 Part 1: Evaluation")
    parser.add_argument("--task", type=str, default="all",
                        choices=["return", "rolling_avg", "turning_point", "all"])
    parser.add_argument("--model", type=str, default="both",
                        choices=["lstm", "gru", "both"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="hw4/checkpoints")
    parser.add_argument("--report_dir", type=str, default="hw4/reports")
    parser.add_argument("--data_dir", type=str, default="hw4/data")
    return parser.parse_args()


@torch.no_grad()
def evaluate_regression(model, loader, device):
    """Evaluate a regression model on a dataloader. Returns loss, predictions, targets."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item() * X.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return avg_loss, preds, targets


def plot_predictions(preds, targets, model_name, task_name, report_dir, n_show=200):
    """Plot predicted vs actual returns for each horizon."""
    os.makedirs(report_dir, exist_ok=True)
    D = preds.shape[1]
    fig, axes = plt.subplots(D, 1, figsize=(12, 3 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]
        ax.plot(targets[:n_show, d], label="Actual", alpha=0.8)
        ax.plot(preds[:n_show, d], label="Predicted", alpha=0.8)
        ax.set_ylabel(f"d={d+1} Return")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample")
    fig.suptitle(f"{model_name} - {task_name}: Predicted vs Actual Returns", y=1.01)
    path = os.path.join(report_dir, f"{task_name}_{model_name}_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Prediction plot saved: {path}")


def test_return_prediction(args, mode="return", task_label="return"):
    """Evaluate Part (b) or (c) models on the test set."""
    print(f"\n{'='*60}")
    print(f"Testing: {task_label}")
    print(f"{'='*60}")

    _, _, test_loader = get_dataloaders(
        tickers=TICKERS, batch_size=args.batch_size,
        mode=mode, rolling_window=3, save_dir=args.data_dir,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}

    models_to_test = []
    if args.model in ("lstm", "both"):
        models_to_test.append(("StockLSTM", StockLSTM(
            input_size=4, hidden_size=args.hidden_size,
            num_layers=args.num_layers, dropout=args.dropout, output_size=5,
        )))
    if args.model in ("gru", "both"):
        models_to_test.append(("StockGRU", StockGRU(
            input_size=4, hidden_size=args.hidden_size,
            num_layers=args.num_layers, dropout=args.dropout, output_size=5,
        )))

    for model_name, model in models_to_test:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{task_label}_{model_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
        model = model.to(device)

        test_loss, preds, targets = evaluate_regression(model, test_loader, device)
        per_horizon_mse = ((preds - targets) ** 2).mean(dim=0).tolist()

        print(f"  {model_name} Test MSE: {test_loss:.6f}")
        for d, mse in enumerate(per_horizon_mse):
            print(f"    d={d+1}: {mse:.6f}")

        plot_predictions(preds.numpy(), targets.numpy(), model_name, task_label, args.report_dir)

        results[model_name] = {
            "test_mse": test_loss,
            "per_horizon_mse": {f"d={d+1}": mse for d, mse in enumerate(per_horizon_mse)},
        }

    return results


def test_turning_point(args):
    """Evaluate Part (d) BiLSTM on the test set."""
    print(f"\n{'='*60}")
    print("Testing: Turning Point Detection (BiLSTM)")
    print(f"{'='*60}")

    GAMMA = 0.10
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, _, test_ds = prepare_datasets(
        tickers=TICKERS, mode="return", use_max_price=True, save_dir=args.data_dir,
    )
    test_X = test_ds.X
    test_labels = (test_ds.y.max(dim=1).values > GAMMA).float().unsqueeze(1)

    print(f"  Test samples: {len(test_X)}, Buy ratio: {test_labels.mean():.3f}")

    # Load model
    ckpt_path = os.path.join(args.checkpoint_dir, "turning_point_bilstm.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return {}

    model = StockBiLSTM(
        input_size=4, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout,
    )
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        scores = torch.sigmoid(model(test_X.to(device))).cpu().squeeze()
    pred_labels = (scores > 0.5).int().numpy()
    true_labels = test_labels.squeeze().int().numpy()

    # Metrics
    tp = ((pred_labels == 1) & (true_labels == 1)).sum()
    fp = ((pred_labels == 1) & (true_labels == 0)).sum()
    fn = ((pred_labels == 0) & (true_labels == 1)).sum()
    tn = ((pred_labels == 0) & (true_labels == 0)).sum()
    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Confusion matrix
    os.makedirs(args.report_dir, exist_ok=True)
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Pass", "Buy"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Turning Point Detection - Confusion Matrix\n(γ=1.1, BiLSTM)")
    path = os.path.join(args.report_dir, "turning_point_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {path}")

    # Precision-Recall curve
    prec_arr, rec_arr, thresholds = precision_recall_curve(true_labels, scores.numpy())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_arr, prec_arr, "b-", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Turning Point Detection - Precision-Recall Curve")
    ax.axhline(y=true_labels.mean(), color="r", linestyle="--",
               label=f"Random baseline ({true_labels.mean():.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.report_dir, "turning_point_pr_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PR curve saved: {path}")

    results = {
        "test_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold_gamma": 1.1,
    }
    return results


def main():
    args = parse_args()
    all_results = {}

    if args.task in ("return", "all"):
        all_results["part_b_return"] = test_return_prediction(
            args, mode="return", task_label="return"
        )

    if args.task in ("rolling_avg", "all"):
        all_results["part_c_rolling_avg"] = test_return_prediction(
            args, mode="rolling_avg", task_label="rolling_avg"
        )

    if args.task in ("turning_point", "all"):
        all_results["part_d_turning_point"] = test_turning_point(args)

    # Comparison
    if "part_b_return" in all_results and "part_c_rolling_avg" in all_results:
        print(f"\n{'='*60}")
        print("COMPARISON: Part (b) vs Part (c)")
        print(f"{'='*60}")
        for model_name in ["StockLSTM", "StockGRU"]:
            b = all_results["part_b_return"].get(model_name, {})
            c = all_results["part_c_rolling_avg"].get(model_name, {})
            if b and c:
                print(f"  {model_name}:")
                print(f"    Return MSE:      {b['test_mse']:.6f}")
                print(f"    Rolling Avg MSE: {c['test_mse']:.6f}")
                print(f"    Improvement:     {(b['test_mse'] - c['test_mse']) / b['test_mse'] * 100:.1f}%")

    # Save test results
    os.makedirs(args.report_dir, exist_ok=True)
    results_path = os.path.join(args.report_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTest results saved to: {results_path}")


if __name__ == "__main__":
    main()
