"""
main.py

Entry point for HW4: Sequence Modeling.

Orchestrates training and evaluation for:
  Part 1 (a-d): Financial Forecasting with LSTM/GRU
  Part 2: Transformer-based Communication Protocol (bonus)
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="HW4: Sequence Modeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "test", "all"],
                        help="Run mode: train only, test only, or both")
    parser.add_argument("--task", type=str, default="all",
                        choices=["return", "rolling_avg", "turning_point", "part2", "all"],
                        help="Which task to run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build common arguments for subprocess calls
    common_args = [
        "--batch_size", str(args.batch_size),
        "--device", args.device,
    ]

    train_args = common_args + [
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
    ]

    task_arg = args.task if args.task != "part2" else "all"

    # --- Part 1: Financial Forecasting ---
    if args.task != "part2":
        if args.mode in ("train", "all"):
            print("\n" + "=" * 70)
            print("  TRAINING - Part 1: Financial Forecasting")
            print("=" * 70)
            from train import main as train_main
            sys.argv = ["train.py", "--task", task_arg] + train_args
            train_main()

        if args.mode in ("test", "all"):
            print("\n" + "=" * 70)
            print("  TESTING - Part 1: Financial Forecasting")
            print("=" * 70)
            from test import main as test_main
            sys.argv = ["test.py", "--task", task_arg] + common_args
            test_main()

    # --- Part 2: Communication System ---
    if args.task in ("part2", "all"):
        if args.mode in ("train", "all"):
            print("\n" + "=" * 70)
            print("  Part 2: Transformer Communication System")
            print("=" * 70)
            from part2_comm import train_comm_system
            train_comm_system(
                epochs=200,
                batch_size=256,
                lr=args.lr,
                device=args.device,
            )

    print("\n" + "=" * 70)
    print("  All tasks completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
