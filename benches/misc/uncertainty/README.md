# Uncertainty Propagation Benchmark Suite

Comparative benchmarks for uncertainty propagation across Sounio, Julia, and Python.

**DOI:** [10.5281/zenodo.18404188](https://doi.org/10.5281/zenodo.18404188)

## Benchmark Scenarios

| Scenario | Description | Operations |
|----------|-------------|------------|
| **Monte Carlo Simulation** | Portfolio risk under GUM | 10K–1M iterations |
| **GUM Propagation Chain** | Chained arithmetic with uncertainty | 1K–100K ops |
| **Pharmacokinetic Model** | 14-compartment PBPK with uncertain parameters | ODE solve |
| **Linear Algebra** | Matrix operations with epistemic types | Dense solve |

## Requirements

### Python
```bash
pip install uncertainties numpy scipy pandas
```

### Julia
```bash
julia -e 'using Pkg; Pkg.add(["Measurements", "BenchmarkTools", "Statistics"])'
```

### Sounio
```bash
cd compiler && cargo build --release --features jit
```

## Running Benchmarks

```bash
# All benchmarks
./run_all.sh

# Individual
python3 python_uncertainty.py
julia julia_uncertainty.jl
../../compiler/target/release/souc run sounio_uncertainty.sio
```

## Results

See [souniolang.org/validation/benchmarks/](https://souniolang.org/validation/benchmarks/) for full results.

### Summary (RTX 4090, Ryzen 9 7950X)

| Scenario | Python (uncertainties) | Julia (Measurements.jl) | Sounio (Knowledge<T>) |
|----------|------------------------|-------------------------|------------------------|
| GUM Chain (1M ops) | 12.4s | 0.82s | **0.41s** |
| Monte Carlo (1M samples) | 8.7s | 0.31s | **0.19s** |
| PBPK Model (1K runs) | 45.2s | 2.1s | **1.3s** |

## Methodology

All benchmarks use:
- GUM-compliant first-order uncertainty propagation
- Same input values and uncertainties
- Warm-up iterations excluded
- 10 runs, median reported

## License

MIT
