# Matrix Decompositions API Reference

Matrix decompositions are fundamental algorithms for solving linear systems, computing eigenvalues, and performing numerical analysis. This document describes the decomposition algorithms available in Sounio's `linalg` module.

## Overview

| Decomposition | Form | Use Cases | Complexity |
|---------------|------|-----------|------------|
| LU | A = LU | Linear systems, determinant | O(n^3) |
| LU with Pivoting | PA = LU | Numerical stability | O(n^3) |
| Cholesky | A = LL^T | Symmetric positive definite | O(n^3/3) |
| QR | A = QR | Least squares, eigenvalues | O(n^3) |
| SVD | A = USV^T | Rank, pseudoinverse | O(n^3) |
| Thomas | Tridiagonal | Banded systems | O(n) |

---

## LU Decomposition

Factorizes a matrix into lower triangular (L) and upper triangular (U) factors.

### Theory

For a square matrix A, LU decomposition finds L and U such that:
```
A = L * U
```

where:
- L is lower triangular with ones on the diagonal
- U is upper triangular

### Implementation

```sio
/// LU decomposition result
struct LUDecomp {
    /// Lower triangular factor (unit diagonal)
    L: &[f64],
    /// Upper triangular factor
    U: &[f64],
    /// Matrix dimension
    n: i64,
    /// Whether decomposition succeeded
    success: bool,
}

/// Compute LU decomposition without pivoting
fn lu_decompose(A: &[f64], n: i64) -> LUDecomp
```

### Algorithm (Doolittle)

```sio
/// LU decomposition using Doolittle algorithm
fn lu_doolittle(A: &[f64], n: i64, L: &![f64], U: &![f64]) -> bool {
    var i: i64 = 0
    while i < n {
        // Upper triangular
        var j: i64 = i
        while j < n {
            var sum: f64 = 0.0
            var k: i64 = 0
            while k < i {
                sum = sum + L[(i * n + k) as usize] * U[(k * n + j) as usize]
                k = k + 1
            }
            U[(i * n + j) as usize] = A[(i * n + j) as usize] - sum
            j = j + 1
        }

        // Lower triangular
        j = i
        while j < n {
            if i == j {
                L[(i * n + i) as usize] = 1.0
            } else {
                var sum: f64 = 0.0
                var k: i64 = 0
                while k < i {
                    sum = sum + L[(j * n + k) as usize] * U[(k * n + i) as usize]
                    k = k + 1
                }
                if abs(U[(i * n + i) as usize]) < 1e-15 {
                    return false  // Singular
                }
                L[(j * n + i) as usize] = (A[(j * n + i) as usize] - sum) / U[(i * n + i) as usize]
            }
            j = j + 1
        }
        i = i + 1
    }
    return true
}
```

### Solving Systems with LU

Once A = LU is computed, solve Ax = b:

1. Solve Ly = b (forward substitution)
2. Solve Ux = y (backward substitution)

```sio
/// Forward substitution: solve Ly = b
fn forward_substitute(L: &[f64], b: &[f64], n: i64, y: &![f64]) {
    var i: i64 = 0
    while i < n {
        var sum: f64 = 0.0
        var j: i64 = 0
        while j < i {
            sum = sum + L[(i * n + j) as usize] * y[j as usize]
            j = j + 1
        }
        y[i as usize] = (b[i as usize] - sum) / L[(i * n + i) as usize]
        i = i + 1
    }
}

/// Backward substitution: solve Ux = y
fn backward_substitute(U: &[f64], y: &[f64], n: i64, x: &![f64]) {
    var i: i64 = n - 1
    while i >= 0 {
        var sum: f64 = 0.0
        var j: i64 = i + 1
        while j < n {
            sum = sum + U[(i * n + j) as usize] * x[j as usize]
            j = j + 1
        }
        x[i as usize] = (y[i as usize] - sum) / U[(i * n + i) as usize]
        i = i - 1
    }
}
```

---

## LU Decomposition with Partial Pivoting

For numerical stability, use pivoting to avoid small pivot elements.

### Theory

Find permutation P, lower triangular L, and upper triangular U such that:
```
P * A = L * U
```

### Implementation

```sio
/// LU with pivoting result
struct LUPDecomp {
    /// Combined L\U storage (L below diagonal, U on and above)
    LU: &[f64],
    /// Pivot indices
    pivot: &[i64],
    /// Number of row swaps (for determinant sign)
    swaps: i64,
    /// Matrix dimension
    n: i64,
    /// Success flag
    success: bool,
}
```

### Algorithm

```sio
/// LU decomposition with partial pivoting
fn lu_pivot(A: &[f64], n: i64, LU: &![f64], pivot: &![i64]) -> (bool, i64) {
    // Copy A to LU
    var i: i64 = 0
    while i < n * n {
        LU[i as usize] = A[i as usize]
        i = i + 1
    }

    // Initialize pivot
    i = 0
    while i < n {
        pivot[i as usize] = i
        i = i + 1
    }

    var swaps: i64 = 0

    // Main elimination
    i = 0
    while i < n {
        // Find pivot
        var max_val: f64 = 0.0
        var max_idx: i64 = i
        var k: i64 = i
        while k < n {
            let val = abs(LU[(k * n + i) as usize])
            if val > max_val {
                max_val = val
                max_idx = k
            }
            k = k + 1
        }

        if max_val < 1e-15 {
            return (false, 0)  // Singular
        }

        // Swap rows if necessary
        if max_idx != i {
            var j: i64 = 0
            while j < n {
                let tmp = LU[(i * n + j) as usize]
                LU[(i * n + j) as usize] = LU[(max_idx * n + j) as usize]
                LU[(max_idx * n + j) as usize] = tmp
                j = j + 1
            }
            let tmp_piv = pivot[i as usize]
            pivot[i as usize] = pivot[max_idx as usize]
            pivot[max_idx as usize] = tmp_piv
            swaps = swaps + 1
        }

        // Elimination
        k = i + 1
        while k < n {
            let factor = LU[(k * n + i) as usize] / LU[(i * n + i) as usize]
            LU[(k * n + i) as usize] = factor  // Store L factor

            var j: i64 = i + 1
            while j < n {
                LU[(k * n + j) as usize] = LU[(k * n + j) as usize] - factor * LU[(i * n + j) as usize]
                j = j + 1
            }
            k = k + 1
        }

        i = i + 1
    }

    return (true, swaps)
}
```

### Determinant from LU

```sio
/// Compute determinant from LU decomposition
fn det_from_lu(LU: &[f64], n: i64, swaps: i64) -> f64 {
    var det: f64 = 1.0
    var i: i64 = 0
    while i < n {
        det = det * LU[(i * n + i) as usize]
        i = i + 1
    }
    // Odd number of swaps negates determinant
    if swaps % 2 == 1 {
        det = -det
    }
    return det
}
```

---

## Cholesky Decomposition

For symmetric positive definite matrices, Cholesky is more efficient and stable.

### Theory

For symmetric positive definite A, find lower triangular L such that:
```
A = L * L^T
```

### Algorithm

```sio
/// Cholesky decomposition
/// Returns false if matrix is not positive definite
fn cholesky(A: &[f64], n: i64, L: &![f64]) -> bool {
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j <= i {
            var sum: f64 = 0.0

            if j == i {
                // Diagonal element
                var k: i64 = 0
                while k < j {
                    sum = sum + L[(j * n + k) as usize] * L[(j * n + k) as usize]
                    k = k + 1
                }
                let val = A[(i * n + i) as usize] - sum
                if val <= 0.0 {
                    return false  // Not positive definite
                }
                L[(i * n + j) as usize] = sqrt(val)
            } else {
                // Off-diagonal element
                var k: i64 = 0
                while k < j {
                    sum = sum + L[(i * n + k) as usize] * L[(j * n + k) as usize]
                    k = k + 1
                }
                L[(i * n + j) as usize] = (A[(i * n + j) as usize] - sum) / L[(j * n + j) as usize]
            }
            j = j + 1
        }
        i = i + 1
    }
    return true
}
```

### Solving with Cholesky

```sio
/// Solve Ax = b where A = LL^T
fn cholesky_solve(L: &[f64], b: &[f64], n: i64, x: &![f64]) {
    // Forward: Ly = b
    var y: [f64; 128] = [0.0; 128]
    forward_substitute(L, b, n, &!y[0..n as usize])

    // Backward: L^T x = y
    // (Need L^T for backward substitution)
    var i: i64 = n - 1
    while i >= 0 {
        var sum: f64 = 0.0
        var j: i64 = i + 1
        while j < n {
            sum = sum + L[(j * n + i) as usize] * x[j as usize]
            j = j + 1
        }
        x[i as usize] = (y[i as usize] - sum) / L[(i * n + i) as usize]
        i = i - 1
    }
}
```

---

## QR Decomposition

Factorizes into orthogonal Q and upper triangular R.

### Theory

For any m x n matrix A (m >= n):
```
A = Q * R
```

where:
- Q is m x m orthogonal (Q^T Q = I)
- R is m x n upper triangular

### Applications

- Least squares: min ||Ax - b||
- Eigenvalue computation (QR iteration)
- Orthogonalization

### Gram-Schmidt Algorithm

```sio
/// QR decomposition using modified Gram-Schmidt
fn qr_decompose(A: &[f64], m: i64, n: i64, Q: &![f64], R: &![f64]) {
    // Copy A to Q
    var i: i64 = 0
    while i < m * n {
        Q[i as usize] = A[i as usize]
        i = i + 1
    }

    // Modified Gram-Schmidt
    var k: i64 = 0
    while k < n {
        // Compute R[k,k] = ||Q[:,k]||
        var norm: f64 = 0.0
        i = 0
        while i < m {
            norm = norm + Q[(i * n + k) as usize] * Q[(i * n + k) as usize]
            i = i + 1
        }
        R[(k * n + k) as usize] = sqrt(norm)

        // Normalize Q[:,k]
        let r_kk = R[(k * n + k) as usize]
        if r_kk > 1e-15 {
            i = 0
            while i < m {
                Q[(i * n + k) as usize] = Q[(i * n + k) as usize] / r_kk
                i = i + 1
            }
        }

        // Orthogonalize remaining columns
        var j: i64 = k + 1
        while j < n {
            // R[k,j] = Q[:,k]^T * Q[:,j]
            var dot: f64 = 0.0
            i = 0
            while i < m {
                dot = dot + Q[(i * n + k) as usize] * Q[(i * n + j) as usize]
                i = i + 1
            }
            R[(k * n + j) as usize] = dot

            // Q[:,j] = Q[:,j] - R[k,j] * Q[:,k]
            i = 0
            while i < m {
                Q[(i * n + j) as usize] = Q[(i * n + j) as usize] - dot * Q[(i * n + k) as usize]
                i = i + 1
            }
            j = j + 1
        }

        k = k + 1
    }
}
```

### Least Squares with QR

Solve min ||Ax - b||:

```sio
/// Least squares solve using QR
fn qr_least_squares(Q: &[f64], R: &[f64], b: &[f64], m: i64, n: i64, x: &![f64]) {
    // Compute Q^T * b
    var c: [f64; 128] = [0.0; 128]
    var j: i64 = 0
    while j < n {
        var i: i64 = 0
        while i < m {
            c[j as usize] = c[j as usize] + Q[(i * n + j) as usize] * b[i as usize]
            i = i + 1
        }
        j = j + 1
    }

    // Solve Rx = c (back substitution)
    backward_substitute(R, &c[0..n as usize], n, x)
}
```

---

## Thomas Algorithm (Tridiagonal)

Specialized O(n) solver for tridiagonal systems.

### Theory

For tridiagonal matrix with:
- Lower diagonal: a[1..n-1]
- Main diagonal: b[0..n-1]
- Upper diagonal: c[0..n-2]

Solve Ax = d efficiently.

### Implementation

See [Mat14Tridiag](matrices.md#mat14tridiag) for the 14-element implementation.

### General Algorithm

```sio
/// Thomas algorithm for general tridiagonal system
fn thomas_solve(
    lower: &[f64],    // a[1..n-1], lower[0] unused
    diag: &[f64],     // b[0..n-1]
    upper: &[f64],    // c[0..n-2], upper[n-1] unused
    rhs: &[f64],      // d[0..n-1]
    n: i64,
    x: &![f64]
) {
    // Forward elimination
    var c_prime: [f64; 128] = [0.0; 128]
    var d_prime: [f64; 128] = [0.0; 128]

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    var i: i64 = 1
    while i < n {
        let w = diag[i as usize] - lower[i as usize] * c_prime[(i - 1) as usize]
        if i < n - 1 {
            c_prime[i as usize] = upper[i as usize] / w
        }
        d_prime[i as usize] = (rhs[i as usize] - lower[i as usize] * d_prime[(i - 1) as usize]) / w
        i = i + 1
    }

    // Back substitution
    x[(n - 1) as usize] = d_prime[(n - 1) as usize]
    i = n - 2
    while i >= 0 {
        x[i as usize] = d_prime[i as usize] - c_prime[i as usize] * x[(i + 1) as usize]
        i = i - 1
    }
}
```

### Complexity

| Operation | General LU | Thomas |
|-----------|------------|--------|
| Setup | O(n^3) | O(n) |
| Solve | O(n^2) | O(n) |
| Memory | O(n^2) | O(n) |

---

## SVD (Singular Value Decomposition)

The most general decomposition, works for any matrix.

### Theory

For any m x n matrix A:
```
A = U * S * V^T
```

where:
- U is m x m orthogonal
- S is m x n diagonal (singular values)
- V is n x n orthogonal

### Properties

- Singular values are non-negative: s_1 >= s_2 >= ... >= 0
- Rank = number of non-zero singular values
- Condition number = s_1 / s_n

### Applications

- Pseudoinverse: A^+ = V * S^+ * U^T
- Low-rank approximation
- Principal Component Analysis (PCA)
- Least squares (more stable than QR)

### Note

Full SVD implementation is complex. For small matrices, consider using iterative methods or specialized libraries. The standard library provides SVD through numerical backends for production use.

---

## Algorithm Selection Guide

| Problem | Recommended Method |
|---------|-------------------|
| General linear system | LU with pivoting |
| Symmetric positive definite | Cholesky |
| Least squares | QR |
| Tridiagonal | Thomas |
| Rank-deficient | SVD |
| Eigenvalues | QR iteration |
| Condition estimation | SVD |

---

## Numerical Stability Notes

1. **Pivoting**: Always use partial pivoting for LU in production
2. **Condition number**: Check before solving; ill-conditioned matrices may give inaccurate results
3. **Positive definiteness**: Cholesky will fail if matrix is not positive definite
4. **Diagonal dominance**: Thomas algorithm is stable for diagonally dominant systems
5. **Orthogonality loss**: Modified Gram-Schmidt is more stable than classical

---

## See Also

- [Vector Types](vectors.md)
- [Matrix Types](matrices.md)
- [ODE Solvers](../ode/solvers.md) (uses LU for implicit methods)
