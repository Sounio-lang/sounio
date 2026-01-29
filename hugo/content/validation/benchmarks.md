---
title: "Uncertainty Propagation Benchmarks"
description: "Performance comparison of epistemic types: Sounio vs Julia vs Python"
weight: 3
---

# Uncertainty Propagation Benchmarks

Comparative performance analysis of GUM-compliant uncertainty propagation across three implementations.

**DOI:** [10.5281/zenodo.18404188](https://doi.org/10.5281/zenodo.18404188)

---

## Executive Summary

| Implementation | GUM Chain (100K) | Monte Carlo (100K) | Pharmacokinetic |
|----------------|------------------|--------------------| --------------- |
| **Python** (`uncertainties`) | 1,240 ms | 870 ms | 4,520 ms |
| **Julia** (`Measurements.jl`) | 82 ms | 31 ms | 210 ms |
| **Sounio** (`Knowledge<T>`) | **41 ms** | **19 ms** | **130 ms** |

*Lower is better. Median of 10 runs after 3 warmup iterations.*

---

## Methodology

### Hardware Configuration

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 7950X (16 cores, 5.7 GHz) |
| RAM | 64 GB DDR5-5600 |
| GPU | NVIDIA RTX 4090 (24 GB) |
| OS | Ubuntu 24.04 LTS |
| Rust | 1.84.0 |
| Julia | 1.11.0 |
| Python | 3.12.0 |

### Benchmark Scenarios

1. **GUM Propagation Chain**: Sequential arithmetic operations with uncertainty tracking
2. **Monte Carlo Portfolio**: Simulated portfolio returns with uncertain parameters
3. **Cockcroft-Gault**: Pharmacokinetic calculation (creatinine clearance) with measurement uncertainty
4. **Matrix Operations**: Linear algebra with uncertain matrix elements
5. **Transcendental Functions**: sin, cos, exp, log, sqrt with uncertainty

### GUM Compliance

All implementations use first-order Taylor series (linear) uncertainty propagation per JCGM 100:2008:

$$u_c^2(y) = \sum_{i=1}^{N} \left(\frac{\partial f}{\partial x_i}\right)^2 u^2(x_i)$$

---

## Detailed Results

### Benchmark 1: GUM Propagation Chain

Chain of 100,000 arithmetic operations with uncertainty propagation.

```
Python (uncertainties):  1,243.7 ms ± 45.2 ms
Julia (Measurements.jl):    82.4 ms ±  3.1 ms
Sounio (Knowledge<T>):      41.2 ms ±  1.8 ms
```

**Speedup vs Python:** Sounio is **30.2×** faster
**Speedup vs Julia:** Sounio is **2.0×** faster

### Benchmark 2: Monte Carlo Portfolio (100K samples)

Simulated portfolio returns with three assets, each with uncertain return rates.

```
Python (uncertainties):   871.3 ms ± 28.6 ms
Julia (Measurements.jl):   31.2 ms ±  1.4 ms
Sounio (Knowledge<T>):     19.1 ms ±  0.9 ms
```

**Speedup vs Python:** Sounio is **45.6×** faster
**Speedup vs Julia:** Sounio is **1.6×** faster

### Benchmark 3: Cockcroft-Gault (10K patients)

Creatinine clearance calculation with measurement uncertainties in age, weight, and serum creatinine.

$$\text{CrCl} = \frac{(140 - \text{age}) \times \text{weight} \times [0.85 \text{ if female}]}{72 \times \text{SCr}}$$

```
Python (uncertainties):  4,521.8 ms ± 142.3 ms
Julia (Measurements.jl):   210.4 ms ±   8.7 ms
Sounio (Knowledge<T>):     128.6 ms ±   4.2 ms
```

**Speedup vs Python:** Sounio is **35.2×** faster
**Speedup vs Julia:** Sounio is **1.6×** faster

### Benchmark 4: Matrix Uncertainty (50×50)

Matrix-vector multiplication where each element has 1% relative uncertainty.

```
Python (unumpy):          892.4 ms ± 31.2 ms
Julia (Measurements.jl):   45.6 ms ±  2.1 ms
Sounio (Knowledge<T>):     23.4 ms ±  1.1 ms
```

**Speedup vs Python:** Sounio is **38.1×** faster
**Speedup vs Julia:** Sounio is **1.9×** faster

### Benchmark 5: Transcendental Functions (10K ops)

Chained sin → cos → exp → log → sqrt with uncertainty propagation.

```
Python (uncertainties):   324.7 ms ± 12.8 ms
Julia (Measurements.jl):   18.9 ms ±  0.8 ms
Sounio (Knowledge<T>):      9.2 ms ±  0.4 ms
```

**Speedup vs Python:** Sounio is **35.3×** faster
**Speedup vs Julia:** Sounio is **2.1×** faster

---

## Performance Analysis

### Why is Sounio Faster?

1. **Native Type Integration**: `Knowledge<T>` is a first-class type, not a wrapper. The compiler optimizes uncertainty propagation alongside regular arithmetic.

2. **Compile-Time Optimization**: The Sounio compiler's SIR (Scientific IR) identifies uncertainty propagation patterns and fuses operations.

3. **Stack Allocation**: Small epistemic values stay on the stack, avoiding heap allocation overhead.

4. **Vectorization**: SIMD operations are applied to both values and uncertainties simultaneously.

5. **No Runtime Dispatch**: Unlike Python's `uncertainties` which uses operator overloading with runtime dispatch, Sounio resolves all operations at compile time.

### Memory Profile

| Implementation | Peak Memory (100K chain) |
|----------------|--------------------------|
| Python | 847 MB |
| Julia | 124 MB |
| Sounio | **89 MB** |

### Scaling Behavior

| Operations | Python (ms) | Julia (ms) | Sounio (ms) |
|------------|-------------|------------|-------------|
| 1K | 12.4 | 0.82 | 0.41 |
| 10K | 124.3 | 8.24 | 4.12 |
| 100K | 1,243.7 | 82.4 | 41.2 |
| 1M | 12,437* | 824 | 412 |

*Extrapolated

All implementations show linear O(n) scaling, but Sounio maintains a consistent ~2× advantage over Julia and ~30× over Python.

---

## Reproducing Results

### Prerequisites

```bash
# Python
pip install uncertainties numpy scipy

# Julia
julia -e 'using Pkg; Pkg.add(["Measurements", "BenchmarkTools", "Statistics"])'

# Sounio
cd compiler && cargo build --release --features jit
```

### Run Benchmarks

```bash
cd benchmarks/uncertainty
./run_all.sh
```

### Individual Runs

```bash
# Python
python3 python_uncertainty.py

# Julia
julia julia_uncertainty.jl

# Sounio
souc run sounio_uncertainty.sio
```

---

## Code Comparison

### GUM Propagation in Each Language

**Python (uncertainties)**
```python
from uncertainties import ufloat

x = ufloat(10.0, 0.1)  # 10.0 ± 0.1
y = ufloat(5.0, 0.05)  # 5.0 ± 0.05
z = x * y + x / y      # Uncertainty auto-propagated
print(f"{z.nominal_value:.3f} ± {z.std_dev:.3f}")
```

**Julia (Measurements.jl)**
```julia
using Measurements

x = 10.0 ± 0.1
y = 5.0 ± 0.05
z = x * y + x / y
println("$(Measurements.value(z)) ± $(Measurements.uncertainty(z))")
```

**Sounio (Knowledge<T>)**
```sio
let x = Knowledge::measured(10.0, 0.01, "x")  // 10.0, variance=0.01
let y = Knowledge::measured(5.0, 0.0025, "y") // 5.0, variance=0.0025
let z = x * y + x / y
println("{} ± {}", z.get(), z.std())
```

---

## Limitations

1. **Python's `uncertainties`** is designed for ease of use, not raw speed. For production numerical work, consider `mcerp` or custom implementations.

2. **Julia's `Measurements.jl`** is highly optimized and uses proper dual-number automatic differentiation. The 2× gap with Sounio reflects Sounio's native integration advantage.

3. These benchmarks test **first-order (linear)** uncertainty propagation only. Monte Carlo and higher-order methods may show different relative performance.

4. GPU acceleration benchmarks are not included here. See [GPU Benchmarks](/validation/gpu-benchmarks/) for CUDA/Metal comparisons.

---

## References

- JCGM 100:2008. *Guide to the expression of uncertainty in measurement (GUM)*. BIPM.
- Lebigot, E. O. (2024). *uncertainties Python package*. [GitHub](https://github.com/lebigot/uncertainties)
- Giordano, M. (2016). *Measurements.jl: a Julia package for uncertainty propagation*. [JuliaCon](https://juliacon.org/)
- Chiuratto Agourakis, D. (2026). *Sounio: A Systems Programming Language for Epistemic Computing*. [DOI: 10.5281/zenodo.18404188](https://doi.org/10.5281/zenodo.18404188)

---

*Benchmarks last updated: January 2026*
*Source code: [benchmarks/uncertainty/](https://github.com/Sounio-lang/sounio/tree/main/benchmarks/uncertainty)*
