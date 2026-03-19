<!-- docs:meta
topic_id: repo.docs.stdlib.linalg.blas-ffi
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.stdlib.linalg.blas-ffi
-->

# BLAS FFI Performance

## Overview

Sounio's linalg module includes FFI bindings to optimized BLAS libraries (OpenBLAS, MKL, ATLAS) for high-performance linear algebra operations.

## Performance Results

| Operation | Pure-Sounio | BLAS (OpenBLAS) | Speedup |
|-----------|-------------|-----------------|---------|
| 64x64 DGEMM | ~2ms | ~0.1ms | 20x |
| 256x256 DGEMM | ~120ms | ~3ms | 40x |
| 512x512 DGEMM | ~950ms | ~20ms | 47x |
| 1024x1024 DGEMM | ~7.6s | ~150ms | 50x |

**Target:** <2x NumPy (which also uses BLAS internally)

## Features

- **Automatic library detection**: Tries `libblas.so`, `libopenblas.so`, `libmkl_rt.so`, `libatlas.so`
- **Smart dispatch**: Uses BLAS for deterministic matrices, pure-Sounio GUM propagation for epistemic
- **Fallback support**: Pure-Sounio implementations when BLAS is unavailable
- **SVD via LAPACK**: DGESVD for singular value decomposition

## Usage

### Check BLAS Availability

```sio
use linalg::blas_ffi::{blas_available, dgemm_available, dgesvd_available};

if blas_available() {
    println("BLAS acceleration enabled");
}
if dgemm_available() {
    println("DGEMM (matrix multiply) available");
}
if dgesvd_available() {
    println("DGESVD (SVD) available");
}
```

### Deterministic Matrices (BLAS Path)

```sio
use linalg::epistemic_matrix::EpistemicMatrix;

// Create deterministic matrices (no uncertainty)
let a = EpistemicMatrix::zeros(256, 256);
let b = EpistemicMatrix::zeros(256, 256);
// ... fill matrices with values ...

// This uses BLAS DGEMM internally (40x faster)
let c = a.matmul(&b);
```

### Epistemic Matrices (Pure-Sounio GUM)

```sio
use linalg::epistemic_matrix::EpistemicMatrix;

// Create epistemic matrices (with uncertainty)
let a = EpistemicMatrix::zeros(256, 256)
    .uncertainty(0.01)
    .confidence(0.95);
// ... fill with set() ...

let b = EpistemicMatrix::zeros(256, 256)
    .uncertainty(0.02)
    .confidence(0.90);
// ... fill with set() ...

// This uses pure-Sounio (uncertainty must be propagated via GUM)
let c = a.matmul(&b);

// Result has propagated uncertainty
println("C[0,0] = " + str(c.get_val(0,0)) + " ± " + str(c.get_unc(0,0)));
```

### Direct BLAS Calls

```sio
use linalg::blas_ffi::{blas_dgemm_rowmajor, blas_svd_rowmajor};

// Direct DGEMM call
let m = 256;
let n = 256;
let k = 256;
let alpha = 1.0;
let beta = 0.0;

let rc = blas_dgemm_rowmajor(
    m, n, k,
    alpha,
    &a,  // matrix A (m x k)
    &b,  // matrix B (k x n)
    beta,
    &mut c  // matrix C (m x n), output
);

// Direct SVD call
let mut s: [f64; 256] = [0.0; 256];  // singular values
let mut u: [f64; 65536] = [0.0; 65536];  // left singular vectors
let mut vt: [f64; 65536] = [0.0; 65536];  // right singular vectors

let info = blas_svd_rowmajor(m, n, &mut a, &mut s, &mut u, &mut vt);
```

## Installation

### Ubuntu/Debian

```bash
sudo apt-get install libopenblas-dev
```

### Fedora/RHEL

```bash
sudo dnf install openblas-devel
```

### macOS

```bash
brew install openblas
```

### Intel MKL (Optional, for Intel CPUs)

```bash
# Download from Intel oneAPI Base Toolkit
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    EpistemicMatrix                       │
│  ┌─────────────────────────────────────────────────────┐│
│  │ matmul()                                            ││
│  │  ├─ is_deterministic()?                             ││
│  │  │   ├─ YES + BLAS available → matmul_blas()        ││
│  │  │   │                              └─> DGEMM FFI   ││
│  │  │   └─ NO or no BLAS → matmul_gum()                ││
│  │  │                       └─> Pure-Sounio GUM        ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    blas_ffi.sio                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │ blas_dgemm_rowmajor()  → libblas.so!dgemm_          ││
│  │ blas_svd_rowmajor()    → libblas.so!dgesvd_         ││
│  │ blas_available()       → dlopen() check             ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              System BLAS Library                         │
│  libblas.so → libopenblas.so.0 → OpenBLAS runtime       │
│  or libmkl_rt.so → Intel MKL runtime                    │
│  or libatlas.so → ATLAS runtime                         │
└─────────────────────────────────────────────────────────┘
```

## Comparison vs NumPy

NumPy also uses BLAS internally via `numpy.dot()` and `@` operator. Our target is to be within 2x of NumPy performance:

| Matrix Size | Sounio BLAS | NumPy | Ratio |
|-------------|-------------|-------|-------|
| 256x256 | 3ms | 2ms | 1.5x |
| 512x512 | 20ms | 15ms | 1.3x |
| 1024x1024 | 150ms | 120ms | 1.25x |

The small overhead comes from FFI call overhead and Sounio's runtime checks.

## Files

| File | Description |
|------|-------------|
| [`blas_ffi.sio`](../../../stdlib/linalg/blas_ffi.sio) | FFI bindings to BLAS/LAPACK |
| [`blas_fallback.sio`](../../../stdlib/linalg/blas_fallback.sio) | Pure-Sounio fallback implementations |
| [`epistemic_matrix.sio`](../../../stdlib/linalg/epistemic_matrix.sio) | EpistemicMatrix with BLAS dispatch |
| [`blas_ffi_test.sio`](../../../tests/stdlib/linalg/blas_ffi_test.sio) | Integration tests |
| [`blas_benchmark.sio`](../../../tests/stdlib/linalg/blas_benchmark.sio) | Performance benchmarks |

## License

MIT / Apache-2.0 (same as Sounio)
