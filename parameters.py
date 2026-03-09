"""
parameters.py

Argument parsing and configuration dataclasses for MNIST MLP classification.

Uses argparse for CLI arguments and dataclasses for typed parameter groups.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


# Parameter Dataclasses
@dataclass
class DataParams:
    """Configuration for dataset loading and preprocessing."""

    data_dir: str = "./data"
    batch_size: int = 128
    val_split: float = 0.1
    num_workers: int = 2
    seed: int = 42


@dataclass
class ModelParams:
    """Configuration for MLP model architecture."""

    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10
    activation: str = "relu"
    dropout: float = 0.3
    use_bn: bool = True


@dataclass
class TrainParams:
    """Configuration for the training loop."""

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    l1_lambda: float = 0.0
    patience: int = 5
    scheduler: str = "step"
    step_size: int = 5
    device: str = "cpu"
    save_path: str = "outputs/main"
    log_interval: int = 100
    mode: str = "both"


# Argument Parser
def get_params() -> Tuple[DataParams, ModelParams, TrainParams]:
    """
    Parses CLI arguments and returns typed parameter dataclasses.

    Returns:
        Tuple of (DataParams, ModelParams, TrainParams).
    """

    parser = argparse.ArgumentParser(
        description="HW1a: MNIST Classification with MLP"
    )

    # --- Mode ---
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Run mode: train, test, or both (default: both)"
    )
    # --- Data ---
    parser.add_argument(
        "--data_dir",    
        type=str,   
        default="./data",  
        help="Dataset directory"
    )
    parser.add_argument(
        "--batch_size",  
        type=int,   
        default=128,       
        help="Batch size"
    )
    parser.add_argument(
        "--val_split",   
        type=float, 
        default=0.1,       
        help="Fraction for validation"
    )
    parser.add_argument(
        "--num_workers", 
        type=int,   
        default=2,         
        help="DataLoader workers"
    )
    parser.add_argument(
        "--seed",        
        type=int,   
        default=42,        
        help="Random seed"
    )
    # --- Model ---
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Width of each hidden layer (e.g. --hidden_sizes 512 256 128)"
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu"],
        default="relu",
        help="Activation function (default: relu)"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.3,  
        help="Dropout probability (0 = disabled)"
    )
    parser.add_argument(
        "--no_bn",   
        action="store_true",      
        help="Disable BatchNorm layers"
    )
    # --- Training ---
    parser.add_argument(
        "--epochs",       
        type=int,   
        default=20,               
        help="Max training epochs"
    )
    parser.add_argument(
        "--lr",           
        type=float, 
        default=1e-3,             
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-4,             
        help="L2 weight decay (AdamW)"
    )
    parser.add_argument(
        "--l1_lambda",    
        type=float, 
        default=0.0,              
        help="L1 regularization coefficient"
    )
    parser.add_argument(
        "--patience",     
        type=int,   
        default=5,                
        help="Early stopping patience"
    )
    parser.add_argument(
        "--scheduler",
        choices=["step", "cosine", "plateau", "none"],
        default="step",
        help="LR scheduler: step, cosine, plateau, or none (default: step)"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=5,
        help="Epoch interval for StepLR (default: 5)"
    )
    parser.add_argument(
        "--device",       
        type=str,   
        default="cpu",            
        help="Device: cpu or cuda"
    )
    parser.add_argument(
        "--save_path",    
        type=str,   
        default="outputs/main",
        help="Output directory (best_model.pth and plots saved inside)"
    )
    parser.add_argument(
        "--log_interval", 
        type=int,   
        default=100,              
        help="Batch log frequency"
    )

    args = parser.parse_args()

    data_params = DataParams(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model_params = ModelParams(
        input_size=784,
        hidden_sizes=args.hidden_sizes,
        num_classes=10,
        activation=args.activation,
        dropout=args.dropout,
        use_bn=not args.no_bn,
    )

    train_params = TrainParams(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        patience=args.patience,
        scheduler=args.scheduler,
        step_size=args.step_size,
        device=args.device,
        save_path=args.save_path,
        log_interval=args.log_interval,
        mode=args.mode,
    )

    return data_params, model_params, train_params
