# Optimization Algorithms API Reference

The `optimization` module provides algorithms for finding minima and maxima of functions. These algorithms integrate with Sounio's automatic differentiation for gradient-based methods.

## Overview

| Algorithm | Type | Gradients | Use Case |
|-----------|------|-----------|----------|
| Gradient Descent | First-order | Required | Simple, convex problems |
| Momentum | First-order | Required | Accelerated convergence |
| Adam | Adaptive | Required | Deep learning, noisy gradients |
| BFGS | Quasi-Newton | Required | Smooth, well-conditioned |
| L-BFGS | Limited-memory | Required | Large-scale optimization |
| Nelder-Mead | Derivative-free | Not used | Non-differentiable, noisy |
| Powell | Derivative-free | Not used | Line search based |
| Trust Region | Second-order | Optional | Robust convergence |

---

## Common Types

### Optimization Result

```sio
pub struct OptResult {
    /// Solution vector
    pub x: &[f64],
    /// Function value at solution
    pub fx: f64,
    /// Gradient at solution (if computed)
    pub grad: &[f64],
    /// Whether optimization succeeded
    pub success: bool,
    /// Number of iterations
    pub iterations: i64,
    /// Number of function evaluations
    pub nfeval: i64,
    /// Number of gradient evaluations
    pub ngeval: i64,
    /// Termination message
    pub message: &str,
}
```

### Configuration

```sio
pub struct OptConfig {
    /// Maximum iterations
    pub max_iter: i64,
    /// Gradient norm tolerance
    pub gtol: f64,
    /// Function value tolerance
    pub ftol: f64,
    /// Step size tolerance
    pub xtol: f64,
    /// Learning rate (for gradient methods)
    pub lr: f64,
    /// Print progress
    pub verbose: bool,
}

pub fn default_config() -> OptConfig {
    return OptConfig {
        max_iter: 1000,
        gtol: 1e-6,
        ftol: 1e-9,
        xtol: 1e-9,
        lr: 0.01,
        verbose: false,
    }
}
```

---

## Gradient Descent

The simplest first-order optimization method.

### Algorithm

```
x_{k+1} = x_k - lr * grad f(x_k)
```

### Implementation

```sio
/// Gradient descent optimization
pub fn gradient_descent(
    f: fn(&[f64]) -> f64,
    grad_f: fn(&[f64]) -> &[f64],
    x0: &[f64],
    config: &OptConfig
) -> OptResult {
    let n = x0.len() as i64
    var x: [f64; 128] = [0.0; 128]
    var i: i64 = 0
    while i < n {
        x[i as usize] = x0[i as usize]
        i = i + 1
    }

    var iter: i64 = 0
    var nfeval: i64 = 0
    var ngeval: i64 = 0

    while iter < config.max_iter {
        // Compute gradient
        let g = grad_f(&x[0..n as usize])
        ngeval = ngeval + 1

        // Check convergence
        var grad_norm: f64 = 0.0
        i = 0
        while i < n {
            grad_norm = grad_norm + g[i as usize] * g[i as usize]
            i = i + 1
        }
        grad_norm = sqrt(grad_norm)

        if grad_norm < config.gtol {
            let fx = f(&x[0..n as usize])
            nfeval = nfeval + 1
            return OptResult {
                x: &x[0..n as usize],
                fx: fx,
                grad: g,
                success: true,
                iterations: iter,
                nfeval: nfeval,
                ngeval: ngeval,
                message: "Converged: gradient norm below tolerance",
            }
        }

        // Update
        i = 0
        while i < n {
            x[i as usize] = x[i as usize] - config.lr * g[i as usize]
            i = i + 1
        }

        iter = iter + 1
    }

    let fx = f(&x[0..n as usize])
    return OptResult {
        x: &x[0..n as usize],
        fx: fx,
        grad: grad_f(&x[0..n as usize]),
        success: false,
        iterations: iter,
        nfeval: nfeval + 1,
        ngeval: ngeval + 1,
        message: "Max iterations reached",
    }
}
```

### Example

```sio
use autodiff::*
use optimization::*

// Rosenbrock function
fn rosenbrock(x: &[f64]) -> f64 {
    let a = 1.0 - x[0]
    let b = x[1] - x[0] * x[0]
    return a * a + 100.0 * b * b
}

// Gradient using autodiff
fn rosenbrock_grad(x: &[f64]) -> &[f64] {
    let g = grad_rosenbrock(x[0], x[1])
    var result: [f64; 2] = [g.dx, g.dy]
    return &result
}

fn main() {
    let x0 = [-1.0, 1.0]
    var config = default_config()
    config.lr = 0.001
    config.max_iter = 10000

    let result = gradient_descent(rosenbrock, rosenbrock_grad, &x0, &config)

    if result.success {
        println("Minimum found at: (", result.x[0], ", ", result.x[1], ")")
        println("f(x) = ", result.fx)
        println("Iterations: ", result.iterations)
    }
}
```

---

## Gradient Descent with Momentum

Accelerates convergence by accumulating a velocity term.

### Algorithm

```
v_{k+1} = beta * v_k + grad f(x_k)
x_{k+1} = x_k - lr * v_{k+1}
```

### Implementation

```sio
pub struct MomentumConfig {
    /// Base configuration
    pub base: OptConfig,
    /// Momentum coefficient (0.9 typical)
    pub beta: f64,
}

pub fn momentum_config_default() -> MomentumConfig {
    return MomentumConfig {
        base: default_config(),
        beta: 0.9,
    }
}

pub fn gradient_descent_momentum(
    f: fn(&[f64]) -> f64,
    grad_f: fn(&[f64]) -> &[f64],
    x0: &[f64],
    config: &MomentumConfig
) -> OptResult {
    let n = x0.len() as i64
    var x: [f64; 128] = [0.0; 128]
    var v: [f64; 128] = [0.0; 128]  // Velocity

    var i: i64 = 0
    while i < n {
        x[i as usize] = x0[i as usize]
        v[i as usize] = 0.0
        i = i + 1
    }

    var iter: i64 = 0

    while iter < config.base.max_iter {
        let g = grad_f(&x[0..n as usize])

        // Update velocity and position
        i = 0
        while i < n {
            v[i as usize] = config.beta * v[i as usize] + g[i as usize]
            x[i as usize] = x[i as usize] - config.base.lr * v[i as usize]
            i = i + 1
        }

        // Check convergence (gradient norm)
        var grad_norm: f64 = 0.0
        i = 0
        while i < n {
            grad_norm = grad_norm + g[i as usize] * g[i as usize]
            i = i + 1
        }
        if sqrt(grad_norm) < config.base.gtol {
            break
        }

        iter = iter + 1
    }

    let fx = f(&x[0..n as usize])
    return OptResult {
        x: &x[0..n as usize],
        fx: fx,
        grad: grad_f(&x[0..n as usize]),
        success: iter < config.base.max_iter,
        iterations: iter,
        nfeval: 1,
        ngeval: iter + 1,
        message: if iter < config.base.max_iter { "Converged" } else { "Max iterations" },
    }
}
```

---

## Adam Optimizer

Adaptive Moment Estimation - combines momentum with per-parameter learning rates.

### Algorithm

```
m_k = beta1 * m_{k-1} + (1 - beta1) * g_k           // First moment
v_k = beta2 * v_{k-1} + (1 - beta2) * g_k^2         // Second moment
m_hat = m_k / (1 - beta1^k)                          // Bias correction
v_hat = v_k / (1 - beta2^k)
x_{k+1} = x_k - lr * m_hat / (sqrt(v_hat) + epsilon)
```

### Configuration

```sio
pub struct AdamConfig {
    pub base: OptConfig,
    /// First moment decay (default 0.9)
    pub beta1: f64,
    /// Second moment decay (default 0.999)
    pub beta2: f64,
    /// Numerical stability (default 1e-8)
    pub epsilon: f64,
}

pub fn adam_config_default() -> AdamConfig {
    return AdamConfig {
        base: default_config(),
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    }
}
```

### Implementation

```sio
pub fn adam(
    f: fn(&[f64]) -> f64,
    grad_f: fn(&[f64]) -> &[f64],
    x0: &[f64],
    config: &AdamConfig
) -> OptResult {
    let n = x0.len() as i64
    var x: [f64; 128] = [0.0; 128]
    var m: [f64; 128] = [0.0; 128]  // First moment
    var v: [f64; 128] = [0.0; 128]  // Second moment

    var i: i64 = 0
    while i < n {
        x[i as usize] = x0[i as usize]
        i = i + 1
    }

    var iter: i64 = 0
    var beta1_t: f64 = 1.0
    var beta2_t: f64 = 1.0

    while iter < config.base.max_iter {
        let g = grad_f(&x[0..n as usize])

        beta1_t = beta1_t * config.beta1
        beta2_t = beta2_t * config.beta2

        i = 0
        while i < n {
            // Update biased moments
            m[i as usize] = config.beta1 * m[i as usize] + (1.0 - config.beta1) * g[i as usize]
            v[i as usize] = config.beta2 * v[i as usize] + (1.0 - config.beta2) * g[i as usize] * g[i as usize]

            // Bias correction
            let m_hat = m[i as usize] / (1.0 - beta1_t)
            let v_hat = v[i as usize] / (1.0 - beta2_t)

            // Update
            x[i as usize] = x[i as usize] - config.base.lr * m_hat / (sqrt(v_hat) + config.epsilon)
            i = i + 1
        }

        // Check convergence
        var grad_norm: f64 = 0.0
        i = 0
        while i < n {
            grad_norm = grad_norm + g[i as usize] * g[i as usize]
            i = i + 1
        }
        if sqrt(grad_norm) < config.base.gtol {
            break
        }

        iter = iter + 1
    }

    let fx = f(&x[0..n as usize])
    return OptResult {
        x: &x[0..n as usize],
        fx: fx,
        grad: grad_f(&x[0..n as usize]),
        success: iter < config.base.max_iter,
        iterations: iter,
        nfeval: 1,
        ngeval: iter + 1,
        message: if iter < config.base.max_iter { "Converged" } else { "Max iterations" },
    }
}
```

---

## Line Search

Line search finds the optimal step size along a descent direction.

### Backtracking Line Search

```sio
/// Backtracking line search with Armijo condition
fn backtracking_line_search(
    f: fn(&[f64]) -> f64,
    x: &[f64],
    d: &[f64],         // Search direction
    grad: &[f64],
    alpha0: f64,       // Initial step size
    c: f64,            // Armijo constant (0.0001 typical)
    rho: f64           // Backtracking factor (0.5 typical)
) -> f64 {
    let n = x.len() as i64
    let fx = f(x)

    // Compute directional derivative: grad^T * d
    var dg: f64 = 0.0
    var i: i64 = 0
    while i < n {
        dg = dg + grad[i as usize] * d[i as usize]
        i = i + 1
    }

    var alpha = alpha0
    var x_new: [f64; 128] = [0.0; 128]

    while alpha > 1e-10 {
        // Compute x + alpha * d
        i = 0
        while i < n {
            x_new[i as usize] = x[i as usize] + alpha * d[i as usize]
            i = i + 1
        }

        let fx_new = f(&x_new[0..n as usize])

        // Armijo condition: f(x + alpha*d) <= f(x) + c*alpha*grad^T*d
        if fx_new <= fx + c * alpha * dg {
            return alpha
        }

        alpha = rho * alpha
    }

    return alpha
}
```

---

## BFGS (Quasi-Newton)

BFGS approximates the inverse Hessian for superlinear convergence.

### Algorithm

```
1. s_k = x_{k+1} - x_k
2. y_k = grad f(x_{k+1}) - grad f(x_k)
3. Update H_k (inverse Hessian approximation)
4. d_k = -H_k * grad f(x_k)
5. Line search for step size
6. x_{k+1} = x_k + alpha * d_k
```

### Configuration

```sio
pub struct BFGSConfig {
    pub base: OptConfig,
    /// Line search parameters
    pub c1: f64,        // Armijo constant
    pub c2: f64,        // Wolfe curvature constant
}

pub fn bfgs_config_default() -> BFGSConfig {
    return BFGSConfig {
        base: default_config(),
        c1: 1e-4,
        c2: 0.9,
    }
}
```

---

## Nelder-Mead (Derivative-Free)

Simplex-based method that doesn't require gradients.

### Algorithm

1. Maintain a simplex of n+1 points
2. Order vertices by function value
3. Reflect worst point through centroid
4. Accept, expand, or contract based on function values
5. Shrink if no improvement

### Configuration

```sio
pub struct NelderMeadConfig {
    pub base: OptConfig,
    /// Reflection coefficient (default 1.0)
    pub alpha: f64,
    /// Expansion coefficient (default 2.0)
    pub gamma: f64,
    /// Contraction coefficient (default 0.5)
    pub rho: f64,
    /// Shrink coefficient (default 0.5)
    pub sigma: f64,
}

pub fn nelder_mead_config_default() -> NelderMeadConfig {
    return NelderMeadConfig {
        base: default_config(),
        alpha: 1.0,
        gamma: 2.0,
        rho: 0.5,
        sigma: 0.5,
    }
}
```

### When to Use

- Non-differentiable objectives
- Noisy function evaluations
- Simple problems with few variables (<10)

---

## Constrained Optimization

### Box Constraints

```sio
pub struct BoxConstraints {
    pub lower: &[f64],
    pub upper: &[f64],
}

/// Project point onto box
fn project_box(x: &![f64], bounds: &BoxConstraints) {
    let n = x.len() as i64
    var i: i64 = 0
    while i < n {
        if x[i as usize] < bounds.lower[i as usize] {
            x[i as usize] = bounds.lower[i as usize]
        }
        if x[i as usize] > bounds.upper[i as usize] {
            x[i as usize] = bounds.upper[i as usize]
        }
        i = i + 1
    }
}
```

### Projected Gradient Descent

```sio
pub fn projected_gradient_descent(
    f: fn(&[f64]) -> f64,
    grad_f: fn(&[f64]) -> &[f64],
    x0: &[f64],
    bounds: &BoxConstraints,
    config: &OptConfig
) -> OptResult
```

---

## Algorithm Selection Guide

| Problem Type | Recommended Algorithm |
|--------------|----------------------|
| Convex, smooth | BFGS or L-BFGS |
| Large-scale convex | L-BFGS |
| Deep learning | Adam |
| Noisy gradients | Adam or SGD with momentum |
| Non-differentiable | Nelder-Mead |
| Simple, small | Gradient descent |
| Box constraints | Projected gradient |

---

## Convergence Criteria

### Gradient Norm

```sio
||grad f(x)|| < gtol
```

### Function Value Change

```sio
|f(x_{k+1}) - f(x_k)| < ftol
```

### Step Size

```sio
||x_{k+1} - x_k|| < xtol
```

---

## Performance Notes

1. **Gradient computation**: Dominates cost; use autodiff for efficiency
2. **Line search**: Essential for convergence in non-convex problems
3. **Preconditioning**: Scale variables for better conditioning
4. **Warm starting**: Use previous solution as initial guess

---

## Complete Example

```sio
use autodiff::*
use optimization::*

fn main() {
    // Minimize Rosenbrock starting from (-1, 1)
    let x0 = [-1.0, 1.0]

    // Try different optimizers
    println("Gradient Descent:")
    var config_gd = default_config()
    config_gd.lr = 0.001
    config_gd.max_iter = 50000
    let result_gd = gradient_descent(rosenbrock, rosenbrock_grad, &x0, &config_gd)
    println("  Iterations: ", result_gd.iterations)
    println("  f(x) = ", result_gd.fx)

    println("Adam:")
    var config_adam = adam_config_default()
    config_adam.base.lr = 0.01
    config_adam.base.max_iter = 10000
    let result_adam = adam(rosenbrock, rosenbrock_grad, &x0, &config_adam)
    println("  Iterations: ", result_adam.iterations)
    println("  f(x) = ", result_adam.fx)
}
```

---

## See Also

- [Automatic Differentiation](../autodiff/index.md)
- [Linear Algebra](../linalg/matrices.md) (for BFGS Hessian updates)
- [ODE Solvers](../ode/solvers.md) (for optimal control)
