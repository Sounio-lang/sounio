#!/usr/bin/env python3
"""
Sounio Compiler: Bayesian Optimization for Attention Weights
=============================================================

This script calibrates the attention-based register allocation weights
using Bayesian Optimization. It finds optimal theta* = (w1, w2, w3, w4, w5, w6)
that minimize a composite objective of runtime and epistemic quality.

Usage:
    python tune_attention_weights.py --benchmark-dir ./benchmarks --iterations 50

Requirements:
    pip install botorch gpytorch torch numpy

Author: Demetrios Chiuratto Agourakis
Date: 2025-12-28
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import numpy as np

# Optional: Use botorch if available, otherwise fallback to simple random search
try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False
    print("Warning: botorch not installed. Using random search fallback.")
    import torch


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AttentionConfig:
    """Attention weight configuration."""
    w_use_density: float = 1.0
    w_crosses_call: float = 0.5
    w_next_use_distance: float = 0.3
    w_confidence: float = 0.5
    w_uncertainty: float = 0.3
    w_provenance: float = 0.2

    def to_tensor(self) -> 'torch.Tensor':
        return torch.tensor([
            self.w_use_density,
            self.w_crosses_call,
            self.w_next_use_distance,
            self.w_confidence,
            self.w_uncertainty,
            self.w_provenance,
        ], dtype=torch.float64)

    @classmethod
    def from_tensor(cls, t: 'torch.Tensor') -> 'AttentionConfig':
        return cls(
            w_use_density=t[0].item(),
            w_crosses_call=t[1].item(),
            w_next_use_distance=t[2].item(),
            w_confidence=t[3].item(),
            w_uncertainty=t[4].item(),
            w_provenance=t[5].item(),
        )

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'AttentionConfig':
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""
    total_time_ms: float
    total_spills: int
    high_confidence_spills: int
    low_confidence_spills: int
    epistemic_quality: float
    provenance_preservation: float
    compile_success: bool
    error: Optional[str] = None


# =============================================================================
# Search Space Definition
# =============================================================================

# Parameter bounds: [lower, upper] for each weight
PARAM_BOUNDS = torch.tensor([
    [0.1, 2.0],   # w_use_density
    [0.0, 1.0],   # w_crosses_call
    [0.0, 1.0],   # w_next_use_distance
    [0.1, 1.0],   # w_confidence
    [0.1, 1.0],   # w_uncertainty
    [0.0, 0.5],   # w_provenance
], dtype=torch.float64).T  # Transpose for botorch format

PARAM_NAMES = [
    'w_use_density',
    'w_crosses_call',
    'w_next_use_distance',
    'w_confidence',
    'w_uncertainty',
    'w_provenance',
]


# =============================================================================
# Objective Function
# =============================================================================

def run_benchmark(
    config: AttentionConfig,
    benchmark_dir: Path,
    souc_path: str = 'souc',
    config_path: str = '/tmp/attention_config.json',
) -> BenchmarkResult:
    """
    Compile and run benchmarks with given attention config.
    Returns metrics for optimization.
    """
    # Write config
    config.to_json(config_path)

    try:
        # Compile with attention policy
        compile_result = subprocess.run(
            [souc_path, 'build',
             '--alloc-policy=attention',
             f'--attention-config={config_path}',
             '--emit-metrics=/tmp/alloc_metrics.json',
             str(benchmark_dir)],
            capture_output=True,
            timeout=60,
        )

        if compile_result.returncode != 0:
            return BenchmarkResult(
                total_time_ms=float('inf'),
                total_spills=1000,
                high_confidence_spills=100,
                low_confidence_spills=0,
                epistemic_quality=0.0,
                provenance_preservation=0.0,
                compile_success=False,
                error=compile_result.stderr.decode()[:500],
            )

        # Run benchmark
        start = time.perf_counter()
        run_result = subprocess.run(
            [str(benchmark_dir / 'run_all')],
            capture_output=True,
            timeout=120,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse metrics
        try:
            with open('/tmp/alloc_metrics.json') as f:
                metrics = json.load(f)
        except:
            metrics = {}

        # Extract values with defaults
        total_spills = metrics.get('total_spills', 0)
        high_conf = metrics.get('high_confidence_spills', 0)
        low_conf = metrics.get('low_confidence_spills', 0)

        # Compute epistemic quality
        eq = (low_conf + 1) / (high_conf + 1) if total_spills > 0 else 1.0

        # Provenance preservation
        prov_total = metrics.get('provenance_values_total', 1)
        prov_preserved = metrics.get('provenance_values_preserved', 1)
        pp = prov_preserved / prov_total if prov_total > 0 else 1.0

        return BenchmarkResult(
            total_time_ms=elapsed_ms,
            total_spills=total_spills,
            high_confidence_spills=high_conf,
            low_confidence_spills=low_conf,
            epistemic_quality=eq,
            provenance_preservation=pp,
            compile_success=True,
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            total_time_ms=float('inf'),
            total_spills=1000,
            high_confidence_spills=100,
            low_confidence_spills=0,
            epistemic_quality=0.0,
            provenance_preservation=0.0,
            compile_success=False,
            error="Timeout",
        )
    except Exception as e:
        return BenchmarkResult(
            total_time_ms=float('inf'),
            total_spills=1000,
            high_confidence_spills=100,
            low_confidence_spills=0,
            epistemic_quality=0.0,
            provenance_preservation=0.0,
            compile_success=False,
            error=str(e),
        )


def objective_function(
    params: 'torch.Tensor',
    benchmark_dir: Path,
    weights: dict = None,
) -> float:
    """
    Compute objective to minimize.

    Objective = w_time * normalized_time
              + w_spills * high_confidence_spills
              - w_quality * epistemic_quality
              - w_prov * provenance_preservation

    Lower is better.
    """
    if weights is None:
        weights = {
            'time': 1.0,
            'high_conf_spills': 10.0,
            'epistemic_quality': 5.0,
            'provenance': 3.0,
        }

    config = AttentionConfig.from_tensor(params)
    result = run_benchmark(config, benchmark_dir)

    if not result.compile_success:
        return 1e6  # Large penalty for failures

    # Normalize time (assume baseline ~100ms)
    norm_time = result.total_time_ms / 100.0

    objective = (
        weights['time'] * norm_time
        + weights['high_conf_spills'] * result.high_confidence_spills
        - weights['epistemic_quality'] * result.epistemic_quality
        - weights['provenance'] * result.provenance_preservation
    )

    return objective


# =============================================================================
# Simulated Objective (for testing without compiler)
# =============================================================================

def simulated_objective(params: 'torch.Tensor') -> float:
    """
    Simulated objective for testing the optimization loop.
    Models expected behavior: high w_confidence and low w_uncertainty = good.
    """
    config = AttentionConfig.from_tensor(params)

    # Simulated metrics based on intuition
    base_quality = (
        0.5 * config.w_confidence
        - 0.3 * config.w_uncertainty
        + 0.2 * config.w_provenance
        + 0.1 * config.w_use_density
    )

    # Add noise
    noise = np.random.normal(0, 0.05)

    # Simulate some spills (fewer with better config)
    simulated_high_conf_spills = max(0, 10 - int(base_quality * 15) + np.random.randint(-2, 3))
    simulated_eq = min(1.0, max(0.1, base_quality + noise))

    # Objective (minimize)
    objective = (
        simulated_high_conf_spills * 10
        - simulated_eq * 5
        - config.w_provenance * 3
    )

    return objective


# =============================================================================
# Optimization Loop
# =============================================================================

def bayesian_optimization(
    objective_fn,
    n_initial: int = 5,
    n_iterations: int = 50,
    verbose: bool = True,
) -> Tuple[AttentionConfig, List[dict]]:
    """
    Run Bayesian Optimization to find optimal weights.
    """
    bounds = PARAM_BOUNDS
    dim = bounds.shape[1]

    # Initial random samples
    X = torch.rand(n_initial, dim, dtype=torch.float64)
    X = bounds[0] + (bounds[1] - bounds[0]) * X

    Y = torch.tensor([[objective_fn(x)] for x in X], dtype=torch.float64)

    history = []

    for i in range(n_initial):
        config = AttentionConfig.from_tensor(X[i])
        history.append({
            'iteration': i,
            'config': asdict(config),
            'objective': Y[i].item(),
            'best_so_far': Y[:i+1].min().item(),
        })

    if verbose:
        print(f"Initial best: {Y.min().item():.4f}")

    for i in range(n_iterations):
        if HAS_BOTORCH:
            # Fit GP
            gp = SingleTaskGP(X, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Acquisition function
            best_f = Y.min()
            ei = ExpectedImprovement(gp, best_f=best_f, maximize=False)

            # Optimize acquisition
            candidate, acq_value = optimize_acqf(
                ei,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )

            new_x = candidate.squeeze(0)
        else:
            # Random search fallback
            new_x = torch.rand(dim, dtype=torch.float64)
            new_x = bounds[0] + (bounds[1] - bounds[0]) * new_x

        new_y = objective_fn(new_x)

        X = torch.cat([X, new_x.unsqueeze(0)])
        Y = torch.cat([Y, torch.tensor([[new_y]], dtype=torch.float64)])

        config = AttentionConfig.from_tensor(new_x)
        history.append({
            'iteration': n_initial + i,
            'config': asdict(config),
            'objective': new_y,
            'best_so_far': Y.min().item(),
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"Iteration {n_initial + i + 1}: best = {Y.min().item():.4f}")

    # Return best config
    best_idx = Y.argmin()
    best_config = AttentionConfig.from_tensor(X[best_idx])

    return best_config, history


def random_search(
    objective_fn,
    n_samples: int = 100,
    verbose: bool = True,
) -> Tuple[AttentionConfig, List[dict]]:
    """
    Simple random search as baseline/fallback.
    """
    bounds = PARAM_BOUNDS
    dim = bounds.shape[1]

    best_config = None
    best_obj = float('inf')
    history = []

    for i in range(n_samples):
        x = torch.rand(dim, dtype=torch.float64)
        x = bounds[0] + (bounds[1] - bounds[0]) * x

        obj = objective_fn(x)
        config = AttentionConfig.from_tensor(x)

        if obj < best_obj:
            best_obj = obj
            best_config = config

        history.append({
            'iteration': i,
            'config': asdict(config),
            'objective': obj,
            'best_so_far': best_obj,
        })

        if verbose and (i + 1) % 20 == 0:
            print(f"Sample {i + 1}: best = {best_obj:.4f}")

    return best_config, history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for Sounio Attention Weights'
    )
    parser.add_argument(
        '--benchmark-dir',
        type=Path,
        default=Path('./benchmarks'),
        help='Directory containing benchmark programs',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Number of optimization iterations',
    )
    parser.add_argument(
        '--initial-samples',
        type=int,
        default=5,
        help='Number of initial random samples',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='optimal_attention_config.json',
        help='Output file for optimal configuration',
    )
    parser.add_argument(
        '--history-output',
        type=str,
        default='optimization_history.json',
        help='Output file for optimization history',
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Use simulated objective (for testing without compiler)',
    )
    parser.add_argument(
        '--method',
        choices=['bo', 'random'],
        default='bo',
        help='Optimization method: bo (Bayesian) or random',
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SOUNIO ATTENTION WEIGHT OPTIMIZATION")
    print("=" * 70)
    print(f"Method: {'Bayesian Optimization' if args.method == 'bo' else 'Random Search'}")
    print(f"Iterations: {args.iterations}")
    print(f"Simulated: {args.simulate}")
    print("=" * 70)

    # Select objective function
    if args.simulate:
        objective_fn = simulated_objective
    else:
        objective_fn = lambda params: objective_function(params, args.benchmark_dir)

    # Run optimization
    if args.method == 'bo' and HAS_BOTORCH:
        best_config, history = bayesian_optimization(
            objective_fn,
            n_initial=args.initial_samples,
            n_iterations=args.iterations,
            verbose=True,
        )
    else:
        total_samples = args.initial_samples + args.iterations
        best_config, history = random_search(
            objective_fn,
            n_samples=total_samples,
            verbose=True,
        )

    # Save results
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)
    for name, value in asdict(best_config).items():
        print(f"  {name}: {value:.4f}")

    best_config.to_json(args.output)
    print(f"\nSaved to: {args.output}")

    with open(args.history_output, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {args.history_output}")

    # Generate Rust code
    print("\n" + "=" * 70)
    print("RUST CODE (copy to AttentionConfig::calibrated())")
    print("=" * 70)
    print(f"""
impl AttentionConfig {{
    /// BO-calibrated configuration (optimized {time.strftime('%Y-%m-%d')})
    pub fn calibrated() -> Self {{
        Self {{
            w_use_density: {best_config.w_use_density:.4f},
            w_crosses_call: {best_config.w_crosses_call:.4f},
            w_next_use_distance: {best_config.w_next_use_distance:.4f},
            w_confidence: {best_config.w_confidence:.4f},
            w_uncertainty: {best_config.w_uncertainty:.4f},
            w_provenance: {best_config.w_provenance:.4f},
            high_confidence_threshold: 0.7,
            low_confidence_threshold: 0.3,
        }}
    }}
}}
""")


if __name__ == '__main__':
    main()
