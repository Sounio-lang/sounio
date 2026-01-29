#!/usr/bin/env julia
"""
Uncertainty Propagation Benchmarks - Julia (Measurements.jl)

Compares Julia's `Measurements.jl` package against Sounio's native Knowledge<T>.
GUM-compliant first-order Taylor series propagation.

Reference: JCGM 100:2008 (GUM)
"""

using Measurements
using BenchmarkTools
using Statistics
using LinearAlgebra
using Printf

# =============================================================================
# Benchmark 1: GUM Propagation Chain
# =============================================================================

function bench_gum_chain(n_ops::Int)
    """Chain of arithmetic operations with uncertainty propagation."""
    # Initial values with uncertainty
    x = 10.0 ± 0.1  # 10.0 ± 0.1
    y = 5.0 ± 0.05  # 5.0 ± 0.05

    result = x
    for i in 1:n_ops
        if i % 4 == 1
            result = result + y
        elseif i % 4 == 2
            result = result - y * 0.1
        elseif i % 4 == 3
            result = result * (1.01 ± 0.001)
        else
            result = result / (1.01 ± 0.001)
        end
    end

    # Verify result has uncertainty
    @assert Measurements.uncertainty(result) > 0 "Uncertainty should propagate"

    return result
end


# =============================================================================
# Benchmark 2: Monte Carlo Portfolio
# =============================================================================

function bench_monte_carlo(n_samples::Int)
    """Monte Carlo simulation with uncertain parameters."""

    # Portfolio parameters with uncertainty
    returns = [0.08 ± 0.02, 0.12 ± 0.03, 0.06 ± 0.01]
    weights = [0.4, 0.35, 0.25]

    # Simulate portfolio value over time
    initial_value = 1_000_000.0
    results = Vector{Float64}(undef, n_samples)

    for i in 1:n_samples
        value = initial_value ± 0
        for (ret, weight) in zip(returns, weights)
            value = value * (1 + ret * weight)
        end
        results[i] = Measurements.value(value)
    end

    # Compute statistics
    mean_val = mean(results)
    std_val = std(results)

    return mean_val, std_val
end


# =============================================================================
# Benchmark 3: Scientific Formula (Creatinine Clearance)
# =============================================================================

function bench_cockcroft_gault(n_patients::Int)
    """
    Cockcroft-Gault equation for creatinine clearance.

    CrCl = ((140 - age) × weight × [0.85 if female]) / (72 × SCr)

    All inputs have measurement uncertainty.
    """

    total = 0.0 ± 0.0

    for i in 1:n_patients
        # Patient parameters with typical measurement uncertainties
        age = (65.0 + (i % 30)) ± 0.5  # years ± 0.5
        weight = (70.0 + (i % 40)) ± 1.0  # kg ± 1.0
        scr = (1.2 + (i % 10) * 0.1) ± 0.05  # mg/dL ± 0.05
        is_female = i % 2 == 0

        # Calculate CrCl with uncertainty propagation
        gender_factor = is_female ? 0.85 : 1.0
        numerator = (140 - age) * weight * gender_factor
        denominator = 72 * scr
        crcl = numerator / denominator

        total = total + crcl
    end

    return total
end


# =============================================================================
# Benchmark 4: Matrix Operations with Uncertainty
# =============================================================================

function bench_matrix_uncertainty(n_size::Int)
    """
    Matrix operations where each element has uncertainty.
    Tests uncertainty propagation through linear algebra.
    """

    # Create matrices with uncertain elements
    A_nominal = randn(n_size, n_size) + I * 5
    A_uncert = abs.(A_nominal) .* 0.01  # 1% relative uncertainty
    A = A_nominal .± A_uncert

    b_nominal = randn(n_size)
    b_uncert = abs.(b_nominal) .* 0.01
    b = b_nominal .± b_uncert

    # Matrix-vector multiplication
    result = A * b

    # Access all uncertainties
    sum_val = sum(Measurements.value.(result))
    sum_unc = sum(Measurements.uncertainty.(result))

    return sum_val, sum_unc
end


# =============================================================================
# Benchmark 5: Transcendental Functions
# =============================================================================

function bench_transcendental(n_ops::Int)
    """
    Uncertainty propagation through transcendental functions.
    Tests: sin, cos, exp, log, sqrt
    """

    x = 1.0 ± 0.01

    for _ in 1:n_ops
        y = sin(x)
        y = cos(y)
        y = exp(y * 0.1)
        y = log(y + 1)
        y = sqrt(y + 1)
        x = y
    end

    return x
end


# =============================================================================
# Main
# =============================================================================

function run_benchmark(name::String, func, args...; warmup::Int=3, runs::Int=10)
    """Run benchmark with warmup and multiple iterations."""

    # Warmup
    for _ in 1:warmup
        func(args...)
    end

    # Timed runs
    times = Vector{Float64}(undef, runs)
    for i in 1:runs
        t = @elapsed func(args...)
        times[i] = t
    end

    median_t = median(times)
    mean_t = mean(times)
    std_t = std(times)

    @printf("%s:\n", name)
    @printf("  Median: %.2f ms\n", median_t * 1000)
    @printf("  Mean:   %.2f ms ± %.2f ms\n", mean_t * 1000, std_t * 1000)
    @printf("  Min:    %.2f ms\n", minimum(times) * 1000)
    @printf("  Max:    %.2f ms\n", maximum(times) * 1000)
    println()

    return median_t
end


function main()
    println("=" ^ 60)
    println("Uncertainty Propagation Benchmarks - Julia (Measurements.jl)")
    println("=" ^ 60)
    println()

    # Run benchmarks
    results = Dict{String, Float64}()

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
    println("=" ^ 60)
    println("Summary (median times in ms):")
    println("=" ^ 60)
    for (name, time_s) in sort(collect(results), by=x->x[1])
        @printf("  %s: %.2f ms\n", name, time_s * 1000)
    end
end

main()
