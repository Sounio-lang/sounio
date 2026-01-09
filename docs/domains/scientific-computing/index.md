# Scientific Computing in Sounio

Sounio provides a comprehensive scientific computing stack designed for researchers and engineers who need production-grade numerical methods with uncertainty quantification built in.

## Overview

The scientific computing modules in Sounio span four major areas:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **ODE Solvers** | Numerical integration | RK4, Tsit5, BDF, Rosenbrock |
| **Linear Algebra** | Matrix/vector operations | Dense, sparse, iterative solvers |
| **Autodiff** | Automatic differentiation | Forward and reverse mode |
| **GPU Kernels** | Parallel computing | CUDA-style kernels, shared memory |

## Design Philosophy

### Uncertainty as First-Class

Scientific computation inherently involves uncertainty: measurement error, numerical precision limits, model approximations. Sounio's epistemic types integrate naturally with numerical methods:

```sio
use epistemic::Knowledge
use ode::*

// Uncertain initial condition propagates through integration
let y0 = Knowledge::new(
    value: 100.0,
    uncertainty: 0.5,
    confidence: 0.95,
    source: "measurement"
)

let config = default_config()
let result = solve_with_uncertainty(y0, 0.0, 10.0, &config)

// Result carries propagated uncertainty
println("Final: ", result.value, " +/- ", result.uncertainty)
```

### Performance Without Sacrifice

Sounio achieves high performance through:

1. **Zero-cost abstractions** - High-level APIs compile to efficient machine code
2. **Cache-aware layouts** - Data structures optimized for memory hierarchy
3. **GPU acceleration** - Native kernel syntax for parallel computation
4. **SIMD vectorization** - Automatic vectorization where applicable

### Dimensional Safety

Units of measure are checked at compile time:

```sio
use units::*

let mass: kg = 70.0
let dose: mg = 500.0
let volume: L = 0.5
let concentration: mg/L = dose / volume  // Compile-time verified

// This would be a compile error:
// let wrong: mg = mass + dose  // kg + mg type mismatch
```

## Quick Start Examples

### Solving an ODE

```sio
use ode::*

// Exponential decay: dy/dt = -k*y
fn decay_rhs(t: f64, y: &[f64], dydt: &![f64]) {
    let k = 0.1
    dydt[0] = -k * y[0]
}

fn main() -> i32 {
    let y0: [f64; 1] = [100.0]
    let config = default_config()

    // Solve from t=0 to t=10
    let result = solve_exp_decay(100.0, 0.0, 10.0, config)

    println("Final value: ", result.u_final)
    println("Steps taken: ", result.nsteps)

    return 0
}
```

### Computing a Gradient

```sio
use autodiff::dual::*

// Compute d/dx(x^2 + sin(x)) at x=1.0
fn main() -> i32 {
    let x = dual_var(1.0)           // Create variable with dx/dx = 1
    let x_sq = dual_mul(x, x)       // x^2
    let sin_x = dual_sin(x)         // sin(x)
    let result = dual_add(x_sq, sin_x)

    println("f(1) = ", result.val)   // Value: 1 + sin(1)
    println("f'(1) = ", result.dot)  // Derivative: 2*1 + cos(1)

    return 0
}
```

### Sparse Linear System

```sio
use linalg::sparse::*

fn main() -> i32 {
    // Build 2D Laplacian on 10x10 grid (100 unknowns)
    let coo = coo_laplacian_2d(10)
    let A = coo_to_csr(&coo)

    // Right-hand side
    var b: [f64; 100] = [1.0; 100]
    var x0: [f64; 100] = [0.0; 100]

    // Solve with preconditioned conjugate gradient
    let M = jacobi_precond(&A)
    let config = solver_config_default()
    let result = pcg_solve_jacobi(&A, &b, &x0, &M, &config)

    if result.converged {
        println("Solved in ", result.iterations, " iterations")
    }

    return 0
}
```

### GPU Kernel

```sio
use gpu::*

// Vector addition on GPU
kernel fn vector_add(
    a: &[f64],
    b: &[f64],
    c: &![f64],
    n: i32
) {
    let tid = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    if tid < n {
        c[tid] = a[tid] + b[tid]
    }
}
```

## Module Reference

### ODE Solvers (`ode::*`)

Numerical integrators for initial value problems dy/dt = f(t, y).

| Solver | Type | Order | Best For |
|--------|------|-------|----------|
| RK4 | Explicit, fixed-step | 4 | Real-time, embedded systems |
| Tsit5 | Explicit, adaptive | 5(4) | General non-stiff problems |
| DOPRI5 | Explicit, adaptive | 5(4) | Robust non-stiff integration |
| BDF | Implicit, multistep | 1-5 | Very stiff systems |
| Rosenbrock | Implicit, single-step | 3-4 | Moderately stiff systems |

See [ODE Solvers](./ode-solvers.md) for detailed documentation.

### Linear Algebra (`linalg::*`)

Dense and sparse matrix operations with iterative solvers.

**Dense Operations:**
- Fixed-size vectors: `Vec2`, `Vec3`, `Vec4`, `Vec14`
- Fixed-size matrices: `Mat2`, `Mat3`, `Mat4`
- Specialized matrices: `Mat14Diag`, `Mat14Tridiag` (PBPK applications)

**Sparse Formats:**
- COO (Coordinate/Triplet) - Easy construction
- CSR (Compressed Sparse Row) - Efficient SpMV
- CSC (Compressed Sparse Column) - Efficient column ops

**Iterative Solvers:**
- CG (Conjugate Gradient) - Symmetric positive definite
- BiCGSTAB - General non-symmetric
- GMRES - Guaranteed convergence

See [Linear Algebra](./linear-algebra.md) for detailed documentation.

### Automatic Differentiation (`autodiff::*`)

Compute derivatives automatically through program execution.

| Mode | Module | Best For |
|------|--------|----------|
| Forward | `autodiff::dual` | Few inputs, many outputs |
| Reverse | `autodiff::tape` | Many inputs, few outputs (ML) |

See [Automatic Differentiation](./autodiff.md) for detailed documentation.

### GPU Computing (`gpu::*`)

Native GPU kernel syntax for parallel computation.

**Submodules:**
- `gpu::fft` - Fast Fourier Transform with batch processing
- `gpu::smooth` - Separable 3D Gaussian convolution
- `gpu::stats` - Statistical computations (correlation, mean, variance)

See [GPU Kernels](./gpu-kernels.md) for detailed documentation.

## Performance Considerations

### Memory Layout

Sounio uses row-major storage for matrices by default, matching C/C++ conventions. For column-major operations (BLAS compatibility), explicit transpose or CSC format is recommended.

### Numerical Precision

All scientific computing modules use `f64` (64-bit IEEE floating point) by default. For applications requiring `f32`, specialized functions are available with `_f32` suffix.

### Error Handling

Numerical methods can fail for various reasons:
- Convergence failure (iterative solvers, Newton methods)
- Singular matrices
- Invalid inputs (negative values for sqrt, etc.)

Sounio handles these through result types:

```sio
let result = cg_solve(&A, &b, &x0, &config)

if result.converged {
    // Use result.solution
} else {
    println("Failed after ", result.iterations, " iterations")
    println("Residual: ", result.residual_norm)
}
```

## Integration with Epistemic Types

Scientific computations can propagate uncertainty through `Knowledge<T>`:

```sio
use epistemic::Knowledge
use linalg::sparse::*

// Uncertain matrix elements
let A_uncertain = Knowledge::new(
    value: A,
    uncertainty: compute_matrix_uncertainty(&A),
    confidence: 0.95,
    source: "finite_element_assembly"
)

// Solution carries uncertainty from both A and b
let x_uncertain = solve_with_uncertainty(&A_uncertain, &b_uncertain, &config)
```

## Further Reading

- [ODE Solvers](./ode-solvers.md) - Complete guide to numerical integration
- [Linear Algebra](./linear-algebra.md) - Dense and sparse operations
- [Automatic Differentiation](./autodiff.md) - Forward and reverse mode AD
- [GPU Kernels](./gpu-kernels.md) - Parallel computing with GPU acceleration
- [Optimization](./optimization.md) - Nonlinear optimization algorithms
