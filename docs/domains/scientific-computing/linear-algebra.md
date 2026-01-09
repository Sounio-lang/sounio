# Linear Algebra

Sounio provides a comprehensive linear algebra library with both dense and sparse matrix operations, designed for scientific computing applications.

## Module Overview

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `linalg::vector` | Fixed-size vectors | `Vec2`, `Vec3`, `Vec4`, `Vec14` |
| `linalg::matrix` | Fixed-size matrices | `Mat2`, `Mat3`, `Mat4` |
| `linalg::sparse::csr` | Compressed Sparse Row | `CSRMatrix` |
| `linalg::sparse::csc` | Compressed Sparse Column | `CSCMatrix` |
| `linalg::sparse::coo` | Coordinate format | `COOMatrix` |
| `linalg::sparse::solvers` | Iterative solvers | CG, BiCGSTAB, GMRES |
| `linalg::sparse::precond` | Preconditioners | Jacobi, SSOR, ILU(0) |

## Dense Vectors

### Fixed-Size Vectors

Sounio provides fixed-size vector types optimized for common dimensions:

```sio
use linalg::vector::*

// 2D vector
struct Vec2 {
    x: f64,
    y: f64
}

// 3D vector
struct Vec3 {
    x: f64,
    y: f64,
    z: f64
}

// 4D vector (homogeneous coordinates, quaternions)
struct Vec4 {
    x: f64,
    y: f64,
    z: f64,
    w: f64
}

// 14-element vector (PBPK compartments)
struct Vec14 {
    v: [f64; 14]
}
```

### Vector Operations

```sio
use linalg::vector::*

fn main() -> i32 {
    // Construction
    let a = vec3_new(1.0, 2.0, 3.0)
    let b = vec3_new(4.0, 5.0, 6.0)

    // Arithmetic
    let sum = vec3_add(a, b)           // (5, 7, 9)
    let diff = vec3_sub(a, b)          // (-3, -3, -3)
    let scaled = vec3_scale(a, 2.0)    // (2, 4, 6)
    let neg = vec3_neg(a)              // (-1, -2, -3)

    // Products
    let dot = vec3_dot(a, b)           // 1*4 + 2*5 + 3*6 = 32
    let cross = vec3_cross(a, b)       // (-3, 6, -3)
    let hadamard = vec3_hadamard(a, b) // (4, 10, 18)

    // Norms
    let norm = vec3_norm(a)            // sqrt(1 + 4 + 9) = sqrt(14)
    let norm_sq = vec3_norm_squared(a) // 14
    let unit = vec3_normalize(a)       // a / ||a||

    // Distance
    let dist = vec3_distance(a, b)

    return 0
}
```

### Vec14 for PBPK Models

The `Vec14` type is optimized for 14-compartment PBPK models:

```sio
use linalg::vector::*

fn main() -> i32 {
    // Initialize state vector
    var state = vec14_zeros()

    // Set compartment values (indexed 0-13)
    state = vec14_set(state, 0, 100.0)   // Gut
    state = vec14_set(state, 1, 50.0)    // Liver

    // Vector operations
    let norm = vec14_norm(state)
    let total = vec14_sum(state)

    // Element access
    let gut_value = vec14_get(state, 0)

    return 0
}
```

## Dense Matrices

### Fixed-Size Matrices

```sio
use linalg::matrix::*

// 2x2 matrix (row-major storage)
struct Mat2 {
    m00: f64, m01: f64,
    m10: f64, m11: f64
}

// 3x3 matrix
struct Mat3 {
    m00: f64, m01: f64, m02: f64,
    m10: f64, m11: f64, m12: f64,
    m20: f64, m21: f64, m22: f64
}

// 4x4 matrix
struct Mat4 {
    m: [f64; 16]  // Row-major
}
```

### Matrix Operations

```sio
use linalg::matrix::*

fn main() -> i32 {
    // Construction
    let I = mat3_identity()
    let A = mat3_from_rows(
        vec3_new(1.0, 2.0, 3.0),
        vec3_new(4.0, 5.0, 6.0),
        vec3_new(7.0, 8.0, 10.0)
    )

    // Arithmetic
    let sum = mat3_add(A, I)
    let diff = mat3_sub(A, I)
    let scaled = mat3_scale(A, 2.0)
    let product = mat3_mul(A, I)

    // Matrix-vector multiplication
    let v = vec3_new(1.0, 1.0, 1.0)
    let Av = mat3_mul_vec(A, v)

    // Properties
    let det = mat3_determinant(A)
    let tr = mat3_trace(A)
    let frob = mat3_frobenius_norm(A)

    // Inverse and transpose
    let At = mat3_transpose(A)
    let Ainv = mat3_inverse(A)  // Returns zero matrix if singular

    return 0
}
```

### Linear Solve (Dense)

For small systems, direct solve using precomputed inverse:

```sio
use linalg::matrix::*

fn main() -> i32 {
    // Solve Ax = b
    let A = mat3_from_rows(
        vec3_new(4.0, 1.0, 0.0),
        vec3_new(1.0, 4.0, 1.0),
        vec3_new(0.0, 1.0, 4.0)
    )
    let b = vec3_new(1.0, 2.0, 1.0)

    // Check if solvable
    let det = mat3_determinant(A)
    if abs_f64(det) < 1e-10 {
        println("Matrix is singular")
        return 1
    }

    // Solve
    let Ainv = mat3_inverse(A)
    let x = mat3_mul_vec(Ainv, b)

    // Verify: compute residual
    let Ax = mat3_mul_vec(A, x)
    let r = vec3_sub(Ax, b)
    println("Residual norm: ", vec3_norm(r))

    return 0
}
```

### Specialized Matrices for PBPK

```sio
use linalg::matrix::*

// Diagonal matrix (14x14) - efficient storage
struct Mat14Diag {
    diag: [f64; 14]
}

// Tridiagonal matrix (14x14) - for Thomas algorithm
struct Mat14Tridiag {
    sub: [f64; 13],   // Sub-diagonal
    diag: [f64; 14],  // Main diagonal
    sup: [f64; 13]    // Super-diagonal
}

// Thomas algorithm for tridiagonal solve
fn tridiag_solve(A: &Mat14Tridiag, b: &Vec14) -> Vec14 {
    // O(n) algorithm for tridiagonal systems
    // ...
}
```

## Sparse Matrices

For large-scale problems (thousands to millions of unknowns), use sparse matrix formats.

### Format Selection Guide

| Operation | Recommended Format | Reason |
|-----------|-------------------|--------|
| Build matrix | COO | O(1) insertion |
| Matrix-vector multiply | CSR | Efficient row access |
| Transpose multiply | CSC | Efficient column access |
| Row slicing | CSR | Row pointers |
| Column slicing | CSC | Column pointers |
| Forward triangular solve | CSC | Column-by-column |
| Backward triangular solve | CSR | Row-by-row |

### COO Format (Construction)

Coordinate format stores triplets (row, col, value). Best for building matrices:

```sio
use linalg::sparse::*

fn main() -> i32 {
    // Create empty COO matrix
    let coo = coo_new(1000, 1000)  // 1000x1000

    // Add entries (row, col, value)
    coo_add(&!coo, 0, 0, 4.0)
    coo_add(&!coo, 0, 1, -1.0)
    coo_add(&!coo, 1, 0, -1.0)
    coo_add(&!coo, 1, 1, 4.0)

    // Add diagonal in bulk
    var i: i64 = 0
    while i < 1000 {
        coo_add(&!coo, i, i, 4.0)
        if i > 0 {
            coo_add(&!coo, i, i-1, -1.0)
        }
        if i < 999 {
            coo_add(&!coo, i, i+1, -1.0)
        }
        i = i + 1
    }

    // Convert to CSR for computation
    let A = coo_to_csr(&coo)

    return 0
}
```

### Convenience Functions

```sio
use linalg::sparse::*

// Tridiagonal matrix from vectors
let sub: [f64; 9] = [-1.0; 9]   // Sub-diagonal
let diag: [f64; 10] = [4.0; 10] // Main diagonal
let sup: [f64; 9] = [-1.0; 9]   // Super-diagonal

let coo = coo_tridiag(&sub, &diag, &sup)
let A = coo_to_csr(&coo)

// 2D Laplacian (5-point stencil)
let coo = coo_laplacian_2d(100)  // 100x100 grid = 10000 unknowns
let A = coo_to_csr(&coo)
```

### CSR Format (Computation)

Compressed Sparse Row format for efficient matrix-vector products:

```sio
use linalg::sparse::*

// CSR structure
struct CSRMatrix {
    nrows: i64,
    ncols: i64,
    nnz: i64,           // Number of non-zeros
    row_ptr: &[i64],    // Row pointers (length nrows+1)
    col_idx: &[i64],    // Column indices (length nnz)
    values: &[f64]      // Values (length nnz)
}
```

**Matrix-vector multiplication:**

```sio
use linalg::sparse::*

fn main() -> i32 {
    // Build and convert
    let coo = coo_laplacian_2d(10)
    let A = coo_to_csr(&coo)

    // Create vectors
    var x: [f64; 100] = [1.0; 100]
    var y: [f64; 100] = [0.0; 100]

    // y = A * x
    csr_matvec(&A, &x, &!y)

    // y = alpha * A * x + beta * y (BLAS-like)
    csr_matvec_axpby(&A, &x, &!y, 2.0, 1.0)  // y = 2*A*x + y

    // y = A^T * x (transpose multiply)
    csr_matvec_transpose(&A, &x, &!y)

    return 0
}
```

**Matrix properties:**

```sio
use linalg::sparse::*

fn main() -> i32 {
    let A = coo_to_csr(&coo_laplacian_2d(10))

    // Norms
    let frob = csr_frobenius_norm(&A)  // sqrt(sum(A_ij^2))
    let inf = csr_infinity_norm(&A)    // max_i(sum_j |A_ij|)

    // Extract diagonal
    var diag: [f64; 100] = [0.0; 100]
    csr_get_diagonal(&A, &!diag)

    // Check symmetry
    let sym_struct = csr_is_symmetric_structure(&A)  // Sparsity pattern
    let sym_values = csr_is_symmetric(&A, 1e-10)     // Values too

    // Row access
    let row_view = csr_row(&A, 5)  // View of row 5
    // row_view.indices, row_view.values, row_view.nnz

    return 0
}
```

### CSC Format (Column Operations)

Compressed Sparse Column format for efficient column access:

```sio
use linalg::sparse::*

// CSC structure (analogous to CSR)
struct CSCMatrix {
    nrows: i64,
    ncols: i64,
    nnz: i64,
    col_ptr: &[i64],    // Column pointers (length ncols+1)
    row_idx: &[i64],    // Row indices (length nnz)
    values: &[f64]      // Values (length nnz)
}
```

**Triangular solves (forward/backward substitution):**

```sio
use linalg::sparse::*

// Forward solve: L * x = b (L lower triangular)
fn solve_lower(L: &CSCMatrix, b: &[f64], x: &![f64]) {
    csc_lower_triangular_solve(L, b, x)
}

// Backward solve: U * x = b (U upper triangular)
fn solve_upper(U: &CSRMatrix, b: &[f64], x: &![f64]) {
    // Use CSR for backward solve
    // ...
}
```

## Iterative Solvers

For large sparse systems, iterative methods are essential.

### Solver Configuration

```sio
use linalg::sparse::*

struct SolverConfig {
    max_iter: i64,      // Maximum iterations
    tol: f64,           // Convergence tolerance
    verbose: bool       // Print progress
}

fn solver_config_default() -> SolverConfig {
    return SolverConfig {
        max_iter: 1000,
        tol: 1e-10,
        verbose: false
    }
}

struct SolverResult {
    converged: bool,
    iterations: i64,
    residual_norm: f64,
    solution: &[f64]
}
```

### CG: Conjugate Gradient

For symmetric positive definite (SPD) matrices:

```sio
use linalg::sparse::*

fn main() -> i32 {
    // Build SPD system (Laplacian is SPD)
    let coo = coo_laplacian_2d(10)
    let A = coo_to_csr(&coo)

    // Right-hand side and initial guess
    var b: [f64; 100] = [1.0; 100]
    var x0: [f64; 100] = [0.0; 100]

    // Solve with CG
    let config = solver_config_default()
    let result = cg_solve(&A, &b, &x0, &config)

    if result.converged {
        println("CG converged in ", result.iterations, " iterations")
        println("Final residual: ", result.residual_norm)
    } else {
        println("CG failed to converge")
    }

    return 0
}
```

**CG algorithm:**
1. Initialize: r = b - A*x, p = r, rho = r'*r
2. Iterate until convergence:
   - q = A*p
   - alpha = rho / (p'*q)
   - x = x + alpha*p
   - r = r - alpha*q
   - Check convergence: ||r|| < tol
   - rho_new = r'*r
   - beta = rho_new / rho
   - p = r + beta*p
   - rho = rho_new

### BiCGSTAB: Biconjugate Gradient Stabilized

For general non-symmetric matrices:

```sio
use linalg::sparse::*

fn main() -> i32 {
    // Non-symmetric matrix (e.g., convection-diffusion)
    let A = build_convection_diffusion_matrix(100)

    var b: [f64; 100] = [1.0; 100]
    var x0: [f64; 100] = [0.0; 100]

    let config = solver_config_default()
    let result = bicgstab_solve(&A, &b, &x0, &config)

    if result.converged {
        println("BiCGSTAB converged in ", result.iterations, " iterations")
    }

    return 0
}
```

**BiCGSTAB properties:**
- Works for non-symmetric matrices
- More stable than standard BiCG
- May stagnate on some problems
- Memory efficient: O(n) storage

### GMRES: Generalized Minimum Residual

For general matrices with guaranteed convergence (if not breakdown):

```sio
use linalg::sparse::*

fn main() -> i32 {
    let A = build_general_matrix(100)

    var b: [f64; 100] = [1.0; 100]
    var x0: [f64; 100] = [0.0; 100]

    var config = solver_config_default()
    // GMRES-specific: restart parameter
    let restart = 30  // Restart every 30 iterations

    let result = gmres_solve(&A, &b, &x0, &config)

    return 0
}
```

**GMRES properties:**
- Minimizes residual over Krylov subspace
- Guaranteed convergence (in exact arithmetic)
- Memory grows with iterations: O(n * k) for k iterations
- Use restarting (GMRES(m)) for memory control

### Solver Selection Guide

| Matrix Property | Recommended Solver |
|-----------------|-------------------|
| Symmetric positive definite | CG |
| Symmetric indefinite | MINRES |
| General non-symmetric | BiCGSTAB or GMRES |
| Guaranteed convergence needed | GMRES |
| Memory constrained | BiCGSTAB |

## Preconditioners

Preconditioners dramatically improve convergence of iterative solvers.

### Jacobi Preconditioner

Diagonal scaling (simplest preconditioner):

```sio
use linalg::sparse::*

fn main() -> i32 {
    let A = coo_to_csr(&coo_laplacian_2d(10))

    // Build Jacobi preconditioner: M = diag(A)
    let M = jacobi_precond(&A)

    // Solve with preconditioned CG
    var b: [f64; 100] = [1.0; 100]
    var x0: [f64; 100] = [0.0; 100]
    let config = solver_config_default()

    let result = pcg_solve_jacobi(&A, &b, &x0, &M, &config)

    println("PCG (Jacobi) converged in ", result.iterations, " iterations")

    return 0
}
```

**Jacobi application:**
```sio
// Apply preconditioner: z = M^{-1} * r
// For Jacobi: z_i = r_i / A_ii
fn jacobi_apply(M: &JacobiPrecond, r: &[f64], z: &![f64]) {
    var i: i64 = 0
    while i < M.n {
        z[i as usize] = r[i as usize] / M.diag[i as usize]
        i = i + 1
    }
}
```

### SSOR Preconditioner

Symmetric Successive Over-Relaxation:

```sio
use linalg::sparse::*

fn main() -> i32 {
    let A = coo_to_csr(&coo_laplacian_2d(10))

    // Build SSOR preconditioner with omega = 1.5
    let M = ssor_precond(&A, 1.5)

    // Apply: z = (D/omega + L)^{-1} * D * (D/omega + L)^{-T} * r
    var r: [f64; 100] = [1.0; 100]
    var z: [f64; 100] = [0.0; 100]
    ssor_apply(&M, &r, &!z)

    return 0
}
```

### ILU(0) Preconditioner

Incomplete LU with no fill-in:

```sio
use linalg::sparse::*

fn main() -> i32 {
    let A = coo_to_csr(&coo_laplacian_2d(10))

    // Build ILU(0) factorization
    // Maintains same sparsity pattern as A
    let M = ilu0_precond(&A)

    // Apply: z = U^{-1} * L^{-1} * r
    var r: [f64; 100] = [1.0; 100]
    var z: [f64; 100] = [0.0; 100]
    ilu0_apply(&M, &r, &!z)

    return 0
}
```

**ILU(0) properties:**
- More powerful than Jacobi/SSOR
- Same memory as original matrix
- Good for structured problems (FEM, FDM)
- May fail for indefinite matrices

### Preconditioner Selection Guide

| Problem Type | Recommended Preconditioner | Speedup |
|--------------|---------------------------|---------|
| Well-conditioned | Jacobi | 2-5x |
| Moderately conditioned | SSOR | 5-20x |
| Ill-conditioned, structured | ILU(0) | 10-100x |
| Very ill-conditioned | ILU(k) or AMG | 50-500x |

## Complete Example: 2D Poisson Solver

```sio
use linalg::sparse::*

fn solve_poisson_2d(n: i64) -> bool {
    println("Solving 2D Poisson on ", n, "x", n, " grid")
    let n_unknowns = n * n

    // Build 5-point stencil Laplacian
    let coo = coo_laplacian_2d(n)
    let A = coo_to_csr(&coo)

    // Right-hand side: f(x,y) = 1
    var b: [f64; 10000] = [0.0; 10000]
    var i: i64 = 0
    while i < n_unknowns {
        b[i as usize] = 1.0
        i = i + 1
    }

    // Initial guess: zero
    var x0: [f64; 10000] = [0.0; 10000]

    // Build Jacobi preconditioner
    let M = jacobi_precond(&A)

    // Solve with preconditioned CG
    let config = solver_config_default()
    let result = pcg_solve_jacobi(&A, &b, &x0, &M, &config)

    if result.converged {
        println("  Converged in ", result.iterations, " iterations")
        println("  Final residual: ", result.residual_norm)
        return true
    } else {
        println("  Failed to converge")
        return false
    }
}

fn main() -> i32 {
    // Test on increasing grid sizes
    if !solve_poisson_2d(10) { return 1 }   // 100 unknowns
    if !solve_poisson_2d(50) { return 1 }   // 2500 unknowns
    if !solve_poisson_2d(100) { return 1 }  // 10000 unknowns

    println("All tests passed")
    return 0
}
```

## Performance Tips

### 1. Build with COO, Compute with CSR/CSC

```sio
// O(1) insertion into COO
let coo = coo_new(n, n)
// ... add entries ...

// One-time conversion
let A = coo_to_csr(&coo)  // O(nnz)

// Efficient SpMV
// ... many csr_matvec calls ...
```

### 2. Reuse Preconditioners

```sio
// Build once
let M = ilu0_precond(&A)

// Solve multiple systems with same A
let result1 = pcg_solve_ilu0(&A, &b1, &x0, &M, &config)
let result2 = pcg_solve_ilu0(&A, &b2, &x0, &M, &config)
```

### 3. Choose Right Format for Operation

```sio
// For y = A*x, use CSR
let y = csr_matvec(&A_csr, &x, &!y)

// For y = A^T*x, use CSC (or CSR with transpose flag)
let y = csc_matvec(&A_csc, &x, &!y)  // This IS the transpose
```

### 4. Monitor Convergence

```sio
var config = solver_config_default()
config.verbose = true  // Print iteration info

let result = cg_solve(&A, &b, &x0, &config)

// Diagnose slow convergence
if result.iterations > 100 {
    println("Consider stronger preconditioner")
}
```
