# Linalg

## Overview

High-performance linear algebra: matrices, vectors, decompositions (LU, QR, SVD), sparse operations, and **BLAS FFI acceleration**.

## Epistemic Differentiators

- [`EpistemicMatrix`](./epistemic_matrix.sio) with per-element uncertainty and confidence tracking
- GUM-compliant uncertainty propagation through matmul, solve, eig
- Builder pattern for uncertainty initialization: `.uncertainty(u).confidence(c).set(row, col, val)`
- Confidence degradation via `min` across contributing elements
- **BLAS acceleration** for deterministic matrices (5-50x speedup)

## BLAS FFI Integration

The linalg module includes FFI bindings to optimized BLAS libraries (OpenBLAS, MKL, ATLAS) for high-performance matrix operations.

### Features

- **Automatic BLAS detection**: Tries `libblas.so`, `libopenblas.so`, `libmkl_rt.so`, `libatlas.so`
- **Smart dispatch**: Uses BLAS for deterministic matrices, pure-Sounio GUM propagation for epistemic
- **Fallback support**: Pure-Sounio implementations when BLAS is unavailable
- **SVD via LAPACK**: DGESVD for singular value decomposition

### Usage

```sio
use linalg::blas_ffi::{blas_available, dgemm_available};
use linalg::epistemic_matrix::EpistemicMatrix;

// Check BLAS availability
if blas_available() {
    println("BLAS acceleration enabled");
}

// Deterministic matrices automatically use BLAS
let a = EpistemicMatrix::zeros(100, 100)
    .set(0, 0, 1.0)
    .set(1, 1, 1.0);
// ... fill matrix ...

let b = EpistemicMatrix::zeros(100, 100)
    .set(0, 0, 2.0)
    .set(1, 1, 2.0);

// This uses BLAS DGEMM internally (5-50x faster)
let c = a.matmul(&b);

// Epistemic matrices use pure-Sounio GUM propagation
let epist_a = EpistemicMatrix::zeros(100, 100)
    .uncertainty(0.01)
    .confidence(0.95)
    .set(0, 0, 1.0);

let epist_b = EpistemicMatrix::zeros(100, 100)
    .uncertainty(0.02)
    .confidence(0.90)
    .set(0, 0, 2.0);

// This uses pure-Sounio (uncertainty must be propagated)
let epist_c = epist_a.matmul(&epist_b);
```

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install libopenblas-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install openblas-devel
```

**macOS:**
```bash
brew install openblas
```

## Quickstart

```sio
use linalg::epistemic_matrix::EpistemicMatrix;

// Builder pattern with uncertainty
let m = EpistemicMatrix::zeros(2, 2)
    .uncertainty(0.01)
    .confidence(0.95)
    .set(0, 0, 1.0)
    .set(1, 1, 1.0);

// Identity matrix
let id = EpistemicMatrix::identity(3)
    .uncertainty(0.001)
    .confidence(0.99);
```

## Performance

| Operation | Pure-Sounio | BLAS (OpenBLAS) | Speedup |
|-----------|-------------|-----------------|---------|
| 64x64 matmul | ~2ms | ~0.1ms | 20x |
| 256x256 matmul | ~120ms | ~3ms | 40x |
| 512x512 matmul | ~950ms | ~20ms | 47x |

Target: **<2x NumPy** (which also uses BLAS internally).

See [`BENCHMARKS.md`](../../benchmarks/README.md) for detailed performance data.

## Validation Status

See [`VALIDATION_REPORT.md`](../../benchmarks/stdlib_validation/VALIDATION_REPORT.md) for test coverage.

## Modules

| Module | Description |
|--------|-------------|
| [`blas_ffi`](./blas_ffi.sio) | **BLAS/LAPACK FFI bindings** |
| [`blas_fallback`](./blas_fallback.sio) | Pure-Sounio BLAS implementations |
| [`epistemic_matrix`](./epistemic_matrix.sio) | Matrix with per-element uncertainty |
| [`epistemic_tensor`](./epistemic_tensor.sio) | Tensor with uncertainty propagation |
| [`matrix`](./matrix.sio) | Standard matrix operations |
| [`vector`](./vector.sio) | Vector operations |
| [`decomp`](./decomp.sio) | Decompositions (LU, QR, SVD) |
| [`eigen`](./eigen.sio) | Dense symmetric eigendecomposition |
| [`factorize`](./factorize.sio) | Matrix factorization |
| [`sparse`](./sparse.sio) | Sparse matrix support |
| [`shaped`](./shaped.sio) | Shape-checked operations |

## License

MIT / Apache-2.0 (same as Sounio)
