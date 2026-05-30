"""
train.py

Training pipeline for HW4 Part 1.
Supports:
  - Part (b): LSTM/GRU return prediction
  - Part (c): Rolling average forecasting
  - Part (d): Turning point detection with Bi-LSTM
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_dataloaders, prepare_datasets, TICKERS
from models import StockLSTM, StockGRU, StockBiLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="HW4 Part 1: Training")
    parser.add_argument("--task", type=str, default="all",
                        choices=["return", "rolling_avg", "turning_point", "all"],
                        help="Which task to train")
    parser.add_argument("--model", type=str, default="both",
                        choices=["lstm", "gru", "both"],
                        help="Which model to train (for return/rolling_avg tasks)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="hw4/checkpoints")
    parser.add_argument("--report_dir", type=str, default="hw4/reports")
    parser.add_argument("--data_dir", type=str, default="hw4/data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item() * X.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, args, model_name, task_name):
    """Full training loop with early stopping."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    save_path = os.path.join(args.save_dir, f"{task_name}_{model_name}.pt")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {model_name} for task: {task_name}")
    print(f"Device: {device}, Epochs: {args.epochs}, LR: {args.lr}")
    print(f"{'='*60}")

    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.1f}s. Best Val MSE: {best_val_loss:.6f}")
    print(f"  Checkpoint saved: {save_path}")

    # Save training curves
    os.makedirs(args.report_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train MSE")
    ax.plot(history["val_loss"], label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name} - {task_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.report_dir, f"{task_name}_{model_name}_loss.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return best_val_loss, len(history["train_loss"])


def train_return_prediction(args, mode="return", task_label="return"):
    """Train Part (b) or Part (c) models."""
    train_loader, val_loader, _ = get_dataloaders(
        tickers=TICKERS, batch_size=args.batch_size,
        mode=mode, rolling_window=3, save_dir=args.data_dir,
    )

    models_to_train = []
    if args.model in ("lstm", "both"):
        models_to_train.append(("StockLSTM", StockLSTM(
            input_size=4, hidden_size=args.hidden_size,
            num_layers=args.num_layers, dropout=args.dropout, output_size=5,
        )))
    if args.model in ("gru", "both"):
        models_to_train.append(("StockGRU", StockGRU(
            input_size=4, hidden_size=args.hidden_size,
            num_layers=args.num_layers, dropout=args.dropout, output_size=5,
        )))

    results = {}
    for model_name, model in models_to_train:
        best_val, epochs = train_model(
            model, train_loader, val_loader, args, model_name, task_label
        )
        results[model_name] = {"best_val_mse": best_val, "epochs_trained": epochs}

    return results


def train_turning_point(args):
    """Train Part (d) - BiLSTM for turning point detection."""
    print(f"\n{'='*60}")
    print("Training: Turning Point Detection (BiLSTM)")
    print(f"{'='*60}")

    GAMMA = 0.10
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, _ = prepare_datasets(
        tickers=TICKERS, mode="return", use_max_price=True, save_dir=args.data_dir,
    )

    def to_binary_labels(dataset):
        labels = (dataset.y.max(dim=1).values > GAMMA).float().unsqueeze(1)
        return dataset.X, labels

    train_X, train_labels = to_binary_labels(train_ds)
    val_X, val_labels = to_binary_labels(val_ds)

    buy_ratio = train_labels.mean().item()
    print(f"  Train buy ratio: {buy_ratio:.3f}")
    print(f"  Val buy ratio: {val_labels.mean():.3f}")

    train_loader = DataLoader(
        TensorDataset(train_X, train_labels),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_labels),
        batch_size=args.batch_size, shuffle=False
    )

    model = StockBiLSTM(
        input_size=4, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)

    pos_weight = torch.tensor([(1 - buy_ratio) / (buy_ratio + 1e-8)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    save_path = os.path.join(args.save_dir, "turning_point_bilstm.pt")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item() * X.size(0)
                predicted = (torch.sigmoid(pred) > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best Val BCE: {best_val_loss:.4f}")
    print(f"  Checkpoint saved: {save_path}")

    # Save training curves
    os.makedirs(args.report_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train BCE")
    ax1.plot(history["val_loss"], label="Val BCE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Turning Point Detection - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Turning Point Detection - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(args.report_dir, "turning_point_bilstm_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"best_val_bce": best_val_loss, "epochs_trained": len(history["train_loss"]), "threshold_gamma": GAMMA}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    all_results = {}

    if args.task in ("return", "all"):
        print("\n" + "#" * 60)
        print("# Part (b): d-Day Return Prediction")
        print("#" * 60)
        all_results["part_b_return"] = train_return_prediction(
            args, mode="return", task_label="return"
        )

    if args.task in ("rolling_avg", "all"):
        print("\n" + "#" * 60)
        print("# Part (c): Rolling Average Return Prediction")
        print("#" * 60)
        all_results["part_c_rolling_avg"] = train_return_prediction(
            args, mode="rolling_avg", task_label="rolling_avg"
        )

    if args.task in ("turning_point", "all"):
        print("\n" + "#" * 60)
        print("# Part (d): Turning Point Detection")
        print("#" * 60)
        all_results["part_d_turning_point"] = train_turning_point(args)

    # Save training results
    results_path = os.path.join(args.report_dir, "train_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTraining results saved to: {results_path}")
    print("Run test.py to evaluate on the test set.")


if __name__ == "__main__":
    main()
