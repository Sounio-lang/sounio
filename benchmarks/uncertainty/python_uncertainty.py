#!/usr/bin/env python3
"""
Uncertainty Propagation Benchmarks - Python (uncertainties library)

Compares Python's `uncertainties` package against Sounio's native Knowledge<T>.
GUM-compliant first-order Taylor series propagation.

Reference: JCGM 100:2008 (GUM)
"""

import time
import numpy as np
from uncertainties import ufloat, umath
from uncertainties.unumpy import uarray
import statistics

# =============================================================================
# Benchmark 1: GUM Propagation Chain
# =============================================================================

def bench_gum_chain(n_ops: int) -> float:
    """Chain of arithmetic operations with uncertainty propagation."""
    # Initial values with uncertainty
    x = ufloat(10.0, 0.1)  # 10.0 ± 0.1
    y = ufloat(5.0, 0.05)  # 5.0 ± 0.05

    start = time.perf_counter()

    result = x
    for i in range(n_ops):
        if i % 4 == 0:
            result = result + y
        elif i % 4 == 1:
            result = result - y * 0.1
        elif i % 4 == 2:
            result = result * ufloat(1.01, 0.001)
        else:
            result = result / ufloat(1.01, 0.001)

    elapsed = time.perf_counter() - start

    # Verify result has uncertainty
    assert result.std_dev > 0, "Uncertainty should propagate"

    return elapsed


# =============================================================================
# Benchmark 2: Monte Carlo Portfolio
# =============================================================================

def bench_monte_carlo(n_samples: int) -> float:
    """Monte Carlo simulation with uncertain parameters."""

    # Portfolio parameters with uncertainty
    returns = [ufloat(0.08, 0.02), ufloat(0.12, 0.03), ufloat(0.06, 0.01)]
    weights = [0.4, 0.35, 0.25]

    start = time.perf_counter()

    # Simulate portfolio value over time
    initial_value = 1000000.0
    results = []

    for _ in range(n_samples):
        value = ufloat(initial_value, 0)
        for ret, weight in zip(returns, weights):
            value = value * (1 + ret * weight)
        results.append(value.nominal_value)

    # Compute statistics
    mean_val = statistics.mean(results)
    std_val = statistics.stdev(results) if len(results) > 1 else 0

    elapsed = time.perf_counter() - start

    return elapsed


# =============================================================================
# Benchmark 3: Scientific Formula (Creatinine Clearance)
# =============================================================================

def bench_cockcroft_gault(n_patients: int) -> float:
    """
    Cockcroft-Gault equation for creatinine clearance.

    CrCl = ((140 - age) × weight × [0.85 if female]) / (72 × SCr)

    All inputs have measurement uncertainty.
    """

    start = time.perf_counter()

    for i in range(n_patients):
        # Patient parameters with typical measurement uncertainties
        age = ufloat(65.0 + (i % 30), 0.5)  # years ± 0.5
        weight = ufloat(70.0 + (i % 40), 1.0)  # kg ± 1.0
        scr = ufloat(1.2 + (i % 10) * 0.1, 0.05)  # mg/dL ± 0.05
        is_female = i % 2 == 0

        # Calculate CrCl with uncertainty propagation
        gender_factor = 0.85 if is_female else 1.0
        numerator = (140 - age) * weight * gender_factor
        denominator = 72 * scr
        crcl = numerator / denominator

        # Access results to prevent optimization
        _ = crcl.nominal_value
        _ = crcl.std_dev

    elapsed = time.perf_counter() - start

    return elapsed


# =============================================================================
# Benchmark 4: Matrix Operations with Uncertainty
# =============================================================================

def bench_matrix_uncertainty(n_size: int) -> float:
    """
    Matrix operations where each element has uncertainty.
    Tests uncertainty propagation through linear algebra.
    """

    # Create matrices with uncertain elements
    A_nominal = np.random.randn(n_size, n_size) + np.eye(n_size) * 5
    A_uncert = np.abs(A_nominal) * 0.01  # 1% relative uncertainty

    b_nominal = np.random.randn(n_size)
    b_uncert = np.abs(b_nominal) * 0.01

    # Convert to uncertain arrays
    A = uarray(A_nominal, A_uncert)
    b = uarray(b_nominal, b_uncert)

    start = time.perf_counter()

    # Matrix-vector multiplication
    result = np.dot(A, b)

    # Access all uncertainties
    for r in result:
        _ = r.nominal_value
        _ = r.std_dev

    elapsed = time.perf_counter() - start

    return elapsed


# =============================================================================
# Benchmark 5: Transcendental Functions
# =============================================================================

def bench_transcendental(n_ops: int) -> float:
    """
    Uncertainty propagation through transcendental functions.
    Tests: sin, cos, exp, log, sqrt
    """

    x = ufloat(1.0, 0.01)

    start = time.perf_counter()

    for _ in range(n_ops):
        y = umath.sin(x)
        y = umath.cos(y)
        y = umath.exp(y * 0.1)
        y = umath.log(y + 1)
        y = umath.sqrt(y + 1)
        x = y

    elapsed = time.perf_counter() - start

    return elapsed


# =============================================================================
# Main
# =============================================================================

def run_benchmark(name: str, func, *args, warmup: int = 3, runs: int = 10):
    """Run benchmark with warmup and multiple iterations."""

    # Warmup
    for _ in range(warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(runs):
        t = func(*args)
        times.append(t)

    median = statistics.median(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0

    print(f"{name}:")
    print(f"  Median: {median*1000:.2f} ms")
    print(f"  Mean:   {mean*1000:.2f} ms ± {std*1000:.2f} ms")
    print(f"  Min:    {min(times)*1000:.2f} ms")
    print(f"  Max:    {max(times)*1000:.2f} ms")
    print()

    return median


if __name__ == "__main__":
    print("=" * 60)
    print("Uncertainty Propagation Benchmarks - Python (uncertainties)")
    print("=" * 60)
    print()

    # Run benchmarks
    results = {}

    results["GUM Chain (10K)"] = run_benchmark(
        "GUM Chain (10K ops)", bench_gum_chain, 10_000
    )

    results["GUM Chain (100K)"] = run_benchmark(
        "GUM Chain (100K ops)", bench_gum_chain, 100_000
    )

    results["Monte Carlo (10K)"] = run_benchmark(
        "Monte Carlo (10K samples)", bench_monte_carlo, 10_000
    )

    results["Monte Carlo (100K)"] = run_benchmark(
        "Monte Carlo (100K samples)", bench_monte_carlo, 100_000
    )

    results["Cockcroft-Gault (10K)"] = run_benchmark(
        "Cockcroft-Gault (10K patients)", bench_cockcroft_gault, 10_000
    )

    results["Matrix (50x50)"] = run_benchmark(
        "Matrix Uncertainty (50x50)", bench_matrix_uncertainty, 50
    )

    results["Transcendental (10K)"] = run_benchmark(
        "Transcendental (10K ops)", bench_transcendental, 10_000
    )

    # Summary
    print("=" * 60)
    print("Summary (median times in ms):")
    print("=" * 60)
    for name, time_s in results.items():
        print(f"  {name}: {time_s*1000:.2f} ms")
