"""
utils/flops_counter.py

FLOP (Floating Point Operations) counting utility for model complexity analysis.

Uses ptflops library to count FLOPs and parameters for PyTorch models.

Install: pip install ptflops
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: ptflops not available. Install with: pip install ptflops")


def count_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (3, 32, 32),
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Counts FLOPs and parameters for a given model.

    Args:
        model: PyTorch model to analyze.
        input_size: Input tensor size (C, H, W) without batch dimension.
        device: Device to run the model on.
        verbose: Whether to print detailed results.

    Returns:
        Tuple of (FLOPs in millions, Parameters in millions).
    """

    if not PTFLOPS_AVAILABLE:
        print("ptflops is not installed. Returning dummy values.")
        return 0.0, 0.0

    model = model.to(device)
    model.eval()

    # Count FLOPs using ptflops
    macs, params = get_model_complexity_info(
        model,
        input_size,
        as_strings=False,
        print_per_layer_stat=verbose,
        verbose=verbose,
    )

    # Convert to millions
    # Note: MACs (Multiply-Accumulate Operations) ≈ 2 * FLOPs for most operations
    flops_millions = macs * 2 / 1e6
    params_millions = params / 1e6

    if verbose:
        print(f"\n{'='*60}")
        print(f"Model Complexity Analysis")
        print(f"{'='*60}")
        print(f"Input size: {input_size}")
        print(f"FLOPs: {flops_millions:.2f} M")
        print(f"Parameters: {params_millions:.2f} M")
        print(f"{'='*60}\n")

    return flops_millions, params_millions


def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_model_complexity(
    models: Dict[str, nn.Module],
    input_size: Tuple[int, ...] = (3, 32, 32),
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Compares complexity (FLOPs and parameters) for multiple models.

    Args:
        models: Dictionary mapping model names to model instances.
        input_size: Input tensor size (C, H, W).
        device: Device to run the models on.

    Returns:
        Dictionary mapping model names to their complexity metrics.
    """

    results = {}

    print(f"\n{'='*80}")
    print(f"Model Complexity Comparison")
    print(f"{'='*80}\n")

    for name, model in models.items():
        print(f"Analyzing: {name}")
        print(f"{'-'*80}")

        flops, params = count_flops(
            model,
            input_size=input_size,
            device=device,
            verbose=False,
        )

        results[name] = {
            "flops_M": flops,
            "params_M": params,
        }

        print(f"  FLOPs: {flops:>10.2f} M")
        print(f"  Params: {params:>10.2f} M")
        print()

    # Print comparison table
    print(f"{'='*80}")
    print(f"{'Model':<30} {'FLOPs (M)':>15} {'Params (M)':>15} {'Speedup':>15}")
    print(f"{'='*80}")

    # Find baseline (first model or largest)
    baseline_name = list(models.keys())[0]
    baseline_flops = results[baseline_name]["flops_M"]

    for name, metrics in results.items():
        speedup = baseline_flops / metrics["flops_M"] if metrics["flops_M"] > 0 else 0.0
        print(
            f"{name:<30} "
            f"{metrics['flops_M']:>15.2f} "
            f"{metrics['params_M']:>15.2f} "
            f"{speedup:>15.2f}x"
        )

    print(f"{'='*80}\n")

    return results


def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (3, 32, 32),
    batch_size: int = 1,
    device: str = "cpu",
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    Profiles model inference time.

    Args:
        model: PyTorch model.
        input_size: Input tensor size (C, H, W).
        batch_size: Batch size for profiling.
        device: Device to run on.
        num_iterations: Number of iterations for profiling.

    Returns:
        Dictionary with profiling metrics.
    """

    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_size).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Profile
    if device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
    else:
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000

    avg_time_ms = elapsed_time_ms / num_iterations
    throughput = 1000.0 / avg_time_ms  # images/second

    return {
        "avg_time_ms": avg_time_ms,
        "throughput_fps": throughput,
        "total_time_ms": elapsed_time_ms,
    }
