"""
main.py

Entry point for MNIST classification with MLP.

Parses CLI arguments, builds data loaders and the MLP model,
then runs training and/or evaluation depending on the selected mode.
"""

import os
import torch

from parameters import get_params, DataParams, ModelParams, TrainParams
from utils.data import DataConfig, get_dataloaders
from utils.visualize import plot_history, plot_tsne, plot_confusion_matrix
from models.mlp import MLP, MLPConfig
from train import train
from test import load_and_test


def _build_data_config(data_params: DataParams) -> DataConfig:
    """
    Converts DataParams to DataConfig expected by the data loader.

    Args:
        data_params: DataParams dataclass from argparse.

    Returns:
        DataConfig dataclass.
    """

    return DataConfig(
        data_dir=data_params.data_dir,
        batch_size=data_params.batch_size,
        val_split=data_params.val_split,
        num_workers=data_params.num_workers,
        seed=data_params.seed,
    )


def _build_model(model_params: ModelParams) -> MLP:
    """
    Instantiates an MLP from ModelParams.

    Args:
        model_params: ModelParams dataclass from argparse.

    Returns:
        MLP model.
    """

    config = MLPConfig(
        input_size=model_params.input_size,
        hidden_sizes=model_params.hidden_sizes,
        num_classes=model_params.num_classes,
        activation=model_params.activation,
        dropout=model_params.dropout,
        use_bn=model_params.use_bn,
    )
    return MLP(config)


# Main
def main() -> None:
    """
    Main entry point.

    Parses arguments, builds data loaders and model, then
    runs training and/or testing depending on --mode.
    """

    data_params, model_params, train_params = get_params()

    # CUDA guard
    if train_params.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        train_params.device = "cpu"

    # save_path is an output directory; model file goes inside it
    out_dir = train_params.save_path
    os.makedirs(out_dir, exist_ok=True)
    train_params.save_path = os.path.join(out_dir, "best_model.pth")

    # Reproducibility
    torch.manual_seed(data_params.seed)

    # --- Data ---
    data_config = _build_data_config(data_params)
    train_loader, val_loader, test_loader = get_dataloaders(data_config)

    print(f"\nDataset splits:")
    print(f"  Train : {len(train_loader.dataset):>6} samples")
    print(f"  Val   : {len(val_loader.dataset):>6} samples")
    print(f"  Test  : {len(test_loader.dataset):>6} samples")

    # --- Model ---
    model = _build_model(model_params)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {num_params:,}\n")

    # --- Train ---
    if train_params.mode in ("train", "both"):
        history = train(model, train_loader, val_loader, train_params)
        run_name = os.path.basename(out_dir)
        plot_history(history, save_path=f"{out_dir}/{run_name}_training_curves.png")

    # --- Test + Visualize ---
    if train_params.mode in ("test", "both"):
        device = torch.device(train_params.device)
        load_and_test(model_params, train_params, test_loader)
        model.load_state_dict(torch.load(train_params.save_path, map_location=device))
        plot_confusion_matrix(model, test_loader, device, save_path=f"{out_dir}/{run_name}_confusion_matrix.png")
        plot_tsne(model, test_loader, device, save_path=f"{out_dir}/{run_name}_tsne.png")


if __name__ == "__main__":
    main()
