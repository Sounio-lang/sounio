<!-- docs:meta
topic_id: repo.docs.stdlib.linalg.blas-ffi
authority: repo_only
audience: users
last_validated: 2026-09-22
validated_by: Claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.stdlib.linalg.blas-ffi
-->

# BLAS FFI Performance

## Overview

Sounio's linalg module exposes BLAS-shaped APIs (`blas_dgemm_rowmajor`,
`blas_dgesvd_approx`) that are **currently implemented in pure Sounio**. The
optimized-library (OpenBLAS/MKL/ATLAS) FFI path is planned but not yet wired:
`blas_available()`, `dgemm_available()`, and `dgesvd_available()` all return
`false`, so every call runs the pure-Sounio implementation.

> **Source-verified status.** Per `stdlib/linalg/blas_ffi.sio`, no BLAS/LAPACK
> library is loaded today. The speedup, automatic-detection, and installation
> sections below describe the *intended* FFI path, not the current build.

## Performance Results

| Operation | Pure-Sounio | BLAS (OpenBLAS) | Speedup |
|-----------|-------------|-----------------|---------|
| 64x64 DGEMM | ~2ms | ~0.1ms | 20x |
| 256x256 DGEMM | ~120ms | ~3ms | 40x |
| 512x512 DGEMM | ~950ms | ~20ms | 47x |
| 1024x1024 DGEMM | ~7.6s | ~150ms | 50x |

**Target:** <2x NumPy (which also uses BLAS internally)

These figures are **design targets** for the planned BLAS path; the current
pure-Sounio build does not reach them.

## Features

- **Automatic library detection (planned)**: Will probe `libblas.so`, `libopenblas.so`, `libmkl_rt.so`, `libatlas.so`; currently inactive (`blas_available()` returns `false`).
- **Smart dispatch (planned)**: Will route deterministic matrices to BLAS and epistemic matrices to pure-Sounio GUM; today every call uses the pure-Sounio path.
- **Always pure-Sounio today**: With the FFI probe disabled, the pure-Sounio implementation is what actually runs.
- **SVD via power iteration**: `blas_dgesvd_approx` is a pure-Sounio rank-1
  approximation of the dominant singular value (no LAPACK/FFI linkage;
  `dgesvd_available()` currently always returns false)

## Usage

### Check BLAS Availability

```sio
use linalg::blas_ffi::{blas_available, dgemm_available, dgesvd_available}

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
use linalg::epistemic_matrix::EpistemicMatrix

// Create deterministic matrices (no uncertainty). The fixed 256-element
// backing store holds at most a 16x16 matrix (16*16 = 256 elements total).
let a = EpistemicMatrix::zeros(16, 16);
let b = EpistemicMatrix::zeros(16, 16);
// ... fill matrices with values ...

// `matmul` is an inline pure-Sounio GUM loop (see epistemic_matrix.sio):
// it does NOT call blas_dgemm_rowmajor. The BLAS FFI path is planned, not wired.
let c = a.matmul(&b);
```

### Epistemic Matrices (Pure-Sounio GUM)

```sio
use linalg::epistemic_matrix::EpistemicMatrix

// Create epistemic matrices (with uncertainty). Max dimension is 16x16
// (256 elements total fit the fixed backing store).
let a = EpistemicMatrix::zeros(16, 16)
    .uncertainty(0.01)
    .confidence(0.95);
// ... fill with set() ...

let b = EpistemicMatrix::zeros(16, 16)
    .uncertainty(0.02)
    .confidence(0.90);
// ... fill with set() ...

// `matmul` runs as an inline pure-Sounio GUM loop (see epistemic_matrix.sio);
// the BLAS FFI path is not wired, so uncertainty is propagated via GUM.
let c = a.matmul(&b);

// Result has propagated uncertainty
println("C[0,0] = " + str(c.get_val(0,0)) + " ± " + str(c.get_unc(0,0)));
```

### Direct BLAS Calls

```sio
use linalg::blas_ffi::{blas_dgemm_rowmajor, blas_dgesvd_approx}

// Direct DGEMM call. Mutable borrows are `&!`; semicolons are not used.
// Buffers are fixed 256-element arrays, so keep m*n, k*n, m*k <= 256.
let a: [f64; 256] = [0.0; 256]  // matrix A (m x k)
let b: [f64; 256] = [0.0; 256]  // matrix B (k x n)
var c: [f64; 256] = [0.0; 256]  // matrix C (m x n), output

let m = 16
let n = 16
let k = 16
let alpha = 1.0
let beta = 0.0

let rc = blas_dgemm_rowmajor(
    m, n, k,
    alpha,
    &a,  // matrix A (m x k)
    &b,  // matrix B (k x n)
    beta,
    &!c  // matrix C (m x n), output
)

// Direct dominant-singular-value call (approximate power iteration).
// blas_dgesvd_approx writes only the largest singular value into `s`
// (no `u`/`vt` outputs); `iters` is the power-iteration step count.

// blas_dgesvd_approx is a rank-1 power-iteration approximation: it computes
// ONLY the dominant (largest) singular value and writes it to s[0]. The
// remaining entries are NOT computed singular values — they stay at the
// zero-initialized placeholder values (the wrapper explicitly zeroes s[1..p]).
// Do not read s[1..p] as computed singular values; full SVD is not implemented.
//
// IMPORTANT — seed-projection limitation (verified against
// stdlib/linalg/blas_ffi.sio:128-143): blas_dgesvd_approx seeds power
// iteration with the ALL-ONES vector v = [1, 1, ...] (normalised), so the
// input must NOT annihilate that seed. Concretely, A·[1, 1] must be
// NONZERO — i.e. NO ROW OF A MAY SUM TO ZERO. A nonzero matrix whose
// rows sum to zero (e.g. [[1, -1], [-1, 1]], where A·[1, 1] = [0, 0])
// annihilates the all-ones seed; then ||A'A·v|| = 0 and the Newton
// square-root step `sigma = 0.5 * (sigma + norm / sigma)`
// (blas_ffi.sio:183) evaluates 0/0 → NaN, collapsing the iterate to 0.
// An all-zero matrix is likewise unsafe. Pass a SEED-SAFE matrix: the
// 2x2 below (rows sum to 4 and 3, so A·[1, 1] = [4, 3] ≠ 0) is stored
// row-major in the fixed 256-element buffer.
var a_svd: [f64; 256] = [0.0; 256]  // 2x2 input, row-major in the 256 buffer
a_svd[0] = 3.0   // row 0, col 0
a_svd[1] = 1.0   // row 0, col 1
a_svd[2] = 1.0   // row 1, col 0
a_svd[3] = 2.0   // row 1, col 1

var s: [f64; 16] = [0.0; 16]  // output buffer
let info = blas_dgesvd_approx(&a_svd, 2, 2, &!s, 32)
// s[0] ~ 3.618 (dominant singular value of [[3,1],[1,2]]); s[1] is a placeholder.
```

## Installation

> Installing a BLAS library is only useful once the FFI path is wired; the
> current pure-Sounio build ignores it.

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
│                    EpistemicMatrix                      │
│  ┌─────────────────────────────────────────────────────┐│
│  │matmul()                                             ││
│  │└─ inline pure-Sounio GUM loop (no dispatch)         ││
│  │   EpistemicMatrix::matmul always runs GUM           ││
│  │   is_deterministic()/blas_available() not called    ││
│  │   (BLAS dispatch is planned, not yet wired)         ││
│  │   → pure-Sounio uncertainty propagation             ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    blas_ffi.sio                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │blas_dgemm_rowmajor()  → pure-Sounio GEMM (no BLAS)  ││
│  │blas_dgesvd_approx()   → pure-Sounio (no BLAS)       ││
│  │blas_available()       → returns false (FFI unwired) ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  (planned) System BLAS Library — NOT loaded by the      │
│  current pure-Sounio build (blas_available() == false); │
│  drawn DISCONNECTED: no edge wires this box to the      │
│  pure-Sounio module above — no BLAS/LAPACK linked.      │
│  libblas.so → libopenblas.so.0 → OpenBLAS runtime       │
│  or libmkl_rt.so → Intel MKL runtime                    │
│  or libatlas.so → ATLAS runtime                         │
└─────────────────────────────────────────────────────────┘
```

## Comparison vs NumPy

NumPy also uses BLAS internally via `numpy.dot()` and `@` operator. Our target is to be within 2x of NumPy performance:

These ratios are **targets** for the planned FFI path; the current pure-Sounio build is slower.

| Matrix Size | Sounio BLAS | NumPy | Ratio |
|-------------|-------------|-------|-------|
| 256x256 | 3ms | 2ms | 1.5x |
| 512x512 | 20ms | 15ms | 1.3x |
| 1024x1024 | 150ms | 120ms | 1.25x |

The small overhead in those targets is projected to come from FFI call overhead and Sounio's runtime checks once the FFI path is wired — it is not a measured result of the current pure-Sounio build, which performs no FFI calls.

## Files

| File | Description |
|------|-------------|
| [`blas_ffi.sio`](../../../stdlib/linalg/blas_ffi.sio) | Pure-Sounio BLAS-shaped API (BLAS/LAPACK FFI planned) |
| [`blas_fallback.sio`](../../../stdlib/linalg/blas_fallback.sio) | Pure-Sounio fallback implementations |
| [`epistemic_matrix.sio`](../../../stdlib/linalg/epistemic_matrix.sio) | EpistemicMatrix with pure-Sounio GUM-only matmul (no BLAS dispatch yet) |
| [`blas_ffi_test.sio`](../../../tests/stdlib/linalg/blas_ffi_test.sio) | Integration tests |
| [`blas_benchmark.sio`](../../../tests/stdlib/linalg/blas_benchmark.sio) | Performance benchmarks |

## License

MIT / Apache-2.0 (same as Sounio)


