# Optimization

Sounio provides optimization algorithms for finding minima of scalar and vector-valued functions, with applications ranging from curve fitting to machine learning.

## Algorithm Overview

| Algorithm | Type | Best For | Requires |
|-----------|------|----------|----------|
| **Gradient Descent** | First-order | Smooth, convex | Gradient |
| **BFGS** | Quasi-Newton | General smooth | Gradient |
| **L-BFGS** | Limited-memory QN | Large-scale | Gradient |
| **Levenberg-Marquardt** | Gauss-Newton | Nonlinear least squares | Jacobian |
| **Nelder-Mead** | Derivative-free | Non-smooth, noisy | Function only |
| **Differential Evolution** | Evolutionary | Global, discontinuous | Function only |
| **SLSQP** | Constrained | Constrained NLP | Gradient |

## Gradient-Based Methods

### Gradient Descent

The simplest optimization algorithm: move in the direction of steepest descent.

```sio
use optimization::*

struct GDConfig {
    learning_rate: f64,   // Step size
    max_iter: i64,        // Maximum iterations
    tol: f64,             // Convergence tolerance
    momentum: f64         // Momentum coefficient (0 = none)
}

fn gd_config_default() -> GDConfig {
    return GDConfig {
        learning_rate: 0.01,
        max_iter: 10000,
        tol: 1e-8,
        momentum: 0.0
    }
}

struct OptResult {
    x: &[f64],            // Solution
    f_val: f64,           // Function value at solution
    converged: bool,      // Convergence flag
    iterations: i64,      // Iterations taken
    grad_norm: f64        // Final gradient norm
}
```

**Basic usage:**

```sio
use optimization::*
use autodiff::tape::*

// Minimize Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
fn rosenbrock(x: &[f64]) -> f64 {
    let a = 1.0 - x[0]
    let b = x[1] - x[0] * x[0]
    return a * a + 100.0 * b * b
}

fn rosenbrock_grad(x: &[f64], g: &![f64]) {
    // df/dx = -2(1-x) - 400x(y-x^2)
    // df/dy = 200(y-x^2)
    let a = 1.0 - x[0]
    let b = x[1] - x[0] * x[0]
    g[0] = -2.0 * a - 400.0 * x[0] * b
    g[1] = 200.0 * b
}

fn main() -> i32 {
    var x0: [f64; 2] = [-1.0, 1.0]
    let config = gd_config_default()

    let result = gradient_descent(rosenbrock, rosenbrock_grad, &x0, &config)

    if result.converged {
        println("Converged to (", result.x[0], ", ", result.x[1], ")")
        println("f(x*) = ", result.f_val)
        println("Iterations: ", result.iterations)
    }

    return 0
}
```

**With momentum (accelerated gradient descent):**

```sio
var config = gd_config_default()
config.momentum = 0.9  // Heavy ball method
config.learning_rate = 0.001

let result = gradient_descent(f, grad_f, &x0, &config)
```

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

Quasi-Newton method that approximates the inverse Hessian using gradient information.

```sio
use optimization::*

struct BFGSConfig {
    max_iter: i64,        // Maximum iterations
    tol: f64,             // Gradient norm tolerance
    line_search_c1: f64,  // Armijo condition parameter
    line_search_c2: f64,  // Curvature condition parameter
    max_ls_iter: i64      // Max line search iterations
}

fn bfgs_config_default() -> BFGSConfig {
    return BFGSConfig {
        max_iter: 1000,
        tol: 1e-8,
        line_search_c1: 1e-4,
        line_search_c2: 0.9,
        max_ls_iter: 20
    }
}
```

**Usage:**

```sio
use optimization::*

fn main() -> i32 {
    var x0: [f64; 2] = [-1.0, 1.0]
    let config = bfgs_config_default()

    let result = bfgs_minimize(rosenbrock, rosenbrock_grad, &x0, &config)

    if result.converged {
        println("BFGS converged in ", result.iterations, " iterations")
        println("Solution: (", result.x[0], ", ", result.x[1], ")")
    }

    return 0
}
```

**BFGS algorithm:**
1. Initialize H_0 = I (identity matrix)
2. Compute search direction: p_k = -H_k * grad(f)
3. Line search: find alpha satisfying Wolfe conditions
4. Update: x_{k+1} = x_k + alpha * p_k
5. Compute s_k = x_{k+1} - x_k, y_k = grad_{k+1} - grad_k
6. Update H: H_{k+1} = (I - rho*s*y^T) H_k (I - rho*y*s^T) + rho*s*s^T
7. Check convergence: ||grad|| < tol

### L-BFGS (Limited-Memory BFGS)

For large-scale problems where storing the full Hessian approximation is impractical.

```sio
use optimization::*

struct LBFGSConfig {
    m: i64,               // Number of corrections to store (typical: 5-20)
    max_iter: i64,
    tol: f64,
    line_search_c1: f64,
    line_search_c2: f64
}

fn lbfgs_config_default() -> LBFGSConfig {
    return LBFGSConfig {
        m: 10,
        max_iter: 1000,
        tol: 1e-8,
        line_search_c1: 1e-4,
        line_search_c2: 0.9
    }
}
```

**Usage:**

```sio
use optimization::*

// Large-scale problem: 10000 variables
fn large_quadratic(x: &[f64]) -> f64 {
    var sum: f64 = 0.0
    var i: i64 = 0
    while i < 10000 {
        sum = sum + (x[i as usize] - i as f64) * (x[i as usize] - i as f64)
        i = i + 1
    }
    return sum
}

fn large_quadratic_grad(x: &[f64], g: &![f64]) {
    var i: i64 = 0
    while i < 10000 {
        g[i as usize] = 2.0 * (x[i as usize] - i as f64)
        i = i + 1
    }
}

fn main() -> i32 {
    var x0: [f64; 10000] = [0.0; 10000]
    let config = lbfgs_config_default()

    let result = lbfgs_minimize(large_quadratic, large_quadratic_grad, &x0, &config)

    println("L-BFGS converged in ", result.iterations, " iterations")

    return 0
}
```

## Nonlinear Least Squares

### Levenberg-Marquardt

For minimizing sum of squared residuals: min ||r(x)||^2 where r: R^n -> R^m.

```sio
use optimization::*

struct LMConfig {
    max_iter: i64,
    tol_f: f64,           // Function tolerance
    tol_x: f64,           // Parameter tolerance
    tol_g: f64,           // Gradient tolerance
    lambda_init: f64,     // Initial damping parameter
    lambda_up: f64,       // Factor to increase lambda
    lambda_down: f64      // Factor to decrease lambda
}

fn lm_config_default() -> LMConfig {
    return LMConfig {
        max_iter: 1000,
        tol_f: 1e-10,
        tol_x: 1e-10,
        tol_g: 1e-10,
        lambda_init: 0.01,
        lambda_up: 10.0,
        lambda_down: 0.1
    }
}

struct LMResult {
    x: &[f64],            // Solution
    residual_norm: f64,   // ||r(x*)||
    converged: bool,
    iterations: i64,
    jacobian_evals: i64
}
```

**Curve fitting example:**

```sio
use optimization::*

// Fit exponential decay: y = A * exp(-k * t)
// Data: t_i, y_i (measured)
// Parameters: x = [A, k]
// Residuals: r_i = y_i - A * exp(-k * t_i)

fn exp_decay_residuals(x: &[f64], t: &[f64], y: &[f64], r: &![f64]) {
    let A = x[0]
    let k = x[1]

    var i: i64 = 0
    while i < t.len() as i64 {
        let model = A * exp_f64(-k * t[i as usize])
        r[i as usize] = y[i as usize] - model
        i = i + 1
    }
}

fn exp_decay_jacobian(x: &[f64], t: &[f64], J: &![f64]) {
    let A = x[0]
    let k = x[1]

    // J[i, 0] = dr_i/dA = -exp(-k*t_i)
    // J[i, 1] = dr_i/dk = A*t_i*exp(-k*t_i)
    var i: i64 = 0
    while i < t.len() as i64 {
        let exp_kt = exp_f64(-k * t[i as usize])
        J[(i * 2 + 0) as usize] = -exp_kt
        J[(i * 2 + 1) as usize] = A * t[i as usize] * exp_kt
        i = i + 1
    }
}

fn main() -> i32 {
    // Measured data
    let t: [f64; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    let y: [f64; 10] = [100.0, 82.0, 67.0, 55.0, 45.0, 37.0, 30.0, 25.0, 20.0, 16.0]

    // Initial guess
    var x0: [f64; 2] = [100.0, 0.1]

    let config = lm_config_default()
    let result = levenberg_marquardt(
        |x, r| exp_decay_residuals(x, &t, &y, r),
        |x, J| exp_decay_jacobian(x, &t, J),
        &x0, 10, 2, &config
    )

    if result.converged {
        println("Fitted parameters:")
        println("  A = ", result.x[0])
        println("  k = ", result.x[1])
        println("Residual norm: ", result.residual_norm)
    }

    return 0
}
```

**LM algorithm:**
1. Compute residual r(x) and Jacobian J(x)
2. Solve normal equations: (J^T J + lambda I) delta = -J^T r
3. If ||r(x + delta)|| < ||r(x)||:
   - Accept: x = x + delta, decrease lambda
4. Else:
   - Reject: increase lambda, try again
5. Check convergence

## Derivative-Free Methods

### Nelder-Mead (Simplex)

Direct search method using a simplex of n+1 points in n dimensions.

```sio
use optimization::*

struct NMConfig {
    max_iter: i64,
    tol_f: f64,           // Function value tolerance
    tol_x: f64,           // Simplex size tolerance
    alpha: f64,           // Reflection coefficient (1.0)
    gamma: f64,           // Expansion coefficient (2.0)
    rho: f64,             // Contraction coefficient (0.5)
    sigma: f64            // Shrink coefficient (0.5)
}

fn nm_config_default() -> NMConfig {
    return NMConfig {
        max_iter: 10000,
        tol_f: 1e-8,
        tol_x: 1e-8,
        alpha: 1.0,
        gamma: 2.0,
        rho: 0.5,
        sigma: 0.5
    }
}
```

**Usage:**

```sio
use optimization::*

// Non-smooth function (no gradient)
fn abs_sum(x: &[f64]) -> f64 {
    var sum: f64 = 0.0
    var i: i64 = 0
    while i < x.len() as i64 {
        sum = sum + abs_f64(x[i as usize])
        i = i + 1
    }
    return sum
}

fn main() -> i32 {
    var x0: [f64; 3] = [1.0, -2.0, 3.0]
    let config = nm_config_default()

    let result = nelder_mead(abs_sum, &x0, &config)

    println("Nelder-Mead result: (", result.x[0], ", ", result.x[1], ", ", result.x[2], ")")
    println("f(x*) = ", result.f_val)

    return 0
}
```

**Nelder-Mead algorithm:**
1. Initialize simplex with n+1 vertices
2. Order vertices: f(x_1) <= f(x_2) <= ... <= f(x_{n+1})
3. Compute centroid of best n vertices
4. Reflect worst vertex through centroid
5. If better than best: try expansion
6. If worse than second-worst: try contraction
7. If contraction fails: shrink simplex
8. Repeat until convergence

### Differential Evolution

Population-based global optimization for non-convex problems.

```sio
use optimization::*

struct DEConfig {
    pop_size: i64,        // Population size (typically 10*n)
    max_iter: i64,        // Maximum generations
    tol: f64,             // Convergence tolerance
    F: f64,               // Mutation factor (0.5-1.0)
    CR: f64,              // Crossover probability (0.7-1.0)
    bounds_lo: &[f64],    // Lower bounds
    bounds_hi: &[f64]     // Upper bounds
}

fn de_config_default(n: i64, lo: &[f64], hi: &[f64]) -> DEConfig {
    return DEConfig {
        pop_size: 10 * n,
        max_iter: 1000,
        tol: 1e-8,
        F: 0.8,
        CR: 0.9,
        bounds_lo: lo,
        bounds_hi: hi
    }
}
```

**Global optimization example:**

```sio
use optimization::*

// Rastrigin function: many local minima, global minimum at origin
fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len()
    var sum: f64 = 10.0 * n as f64
    var i: i64 = 0
    while i < n as i64 {
        let xi = x[i as usize]
        sum = sum + xi * xi - 10.0 * cos_f64(2.0 * 3.14159 * xi)
        i = i + 1
    }
    return sum
}

fn main() -> i32 {
    let n: i64 = 5
    var lo: [f64; 5] = [-5.12; 5]
    var hi: [f64; 5] = [5.12; 5]

    let config = de_config_default(n, &lo, &hi)

    let result = differential_evolution(rastrigin, n, &config)

    println("DE found minimum at:")
    var i: i64 = 0
    while i < n {
        println("  x[", i, "] = ", result.x[i as usize])
        i = i + 1
    }
    println("f(x*) = ", result.f_val)  // Should be near 0

    return 0
}
```

**DE algorithm:**
1. Initialize random population within bounds
2. For each generation:
   - For each individual x_i:
     - Select 3 distinct individuals a, b, c
     - Mutant: v = a + F * (b - c)
     - Trial: crossover between v and x_i
     - If f(trial) < f(x_i): replace x_i
3. Check convergence (population spread)

## Constrained Optimization

### SLSQP (Sequential Least Squares Programming)

For problems with equality and inequality constraints.

```sio
use optimization::*

struct SLSQPConfig {
    max_iter: i64,
    tol: f64,
    ftol: f64,            // Function tolerance
    eps: f64              // Finite difference step
}

fn slsqp_config_default() -> SLSQPConfig {
    return SLSQPConfig {
        max_iter: 100,
        tol: 1e-8,
        ftol: 1e-8,
        eps: 1e-8
    }
}

// Constraint function type
// Returns c(x) where c(x) >= 0 for inequality constraints
// Returns h(x) where h(x) = 0 for equality constraints
```

**Constrained optimization example:**

```sio
use optimization::*

// Minimize: f(x,y) = (x-1)^2 + (y-2)^2
// Subject to: g(x,y) = x + y - 1 >= 0  (inequality)
//             h(x,y) = x - y = 0        (equality)

fn objective(x: &[f64]) -> f64 {
    let a = x[0] - 1.0
    let b = x[1] - 2.0
    return a * a + b * b
}

fn objective_grad(x: &[f64], g: &![f64]) {
    g[0] = 2.0 * (x[0] - 1.0)
    g[1] = 2.0 * (x[1] - 2.0)
}

fn ineq_constraint(x: &[f64]) -> f64 {
    return x[0] + x[1] - 1.0  // Must be >= 0
}

fn eq_constraint(x: &[f64]) -> f64 {
    return x[0] - x[1]  // Must be = 0
}

fn main() -> i32 {
    var x0: [f64; 2] = [0.0, 0.0]
    let config = slsqp_config_default()

    let result = slsqp_minimize(
        objective, objective_grad,
        &[ineq_constraint],  // Inequality constraints
        &[eq_constraint],    // Equality constraints
        &x0, &config
    )

    if result.converged {
        println("Constrained minimum at (", result.x[0], ", ", result.x[1], ")")
        println("f(x*) = ", result.f_val)

        // Verify constraints
        println("g(x*) = ", ineq_constraint(result.x), " >= 0")
        println("h(x*) = ", eq_constraint(result.x), " = 0")
    }

    return 0
}
```

### Bound Constraints

For simple box constraints (lower <= x <= upper):

```sio
use optimization::*

fn main() -> i32 {
    var x0: [f64; 2] = [0.5, 0.5]
    var lower: [f64; 2] = [0.0, 0.0]
    var upper: [f64; 2] = [1.0, 1.0]

    let config = lbfgs_config_default()

    let result = lbfgsb_minimize(
        objective, objective_grad,
        &x0, &lower, &upper, &config
    )

    return 0
}
```

## Applications

### Parameter Estimation

```sio
use optimization::*
use ode::*

// Estimate PK parameters from concentration data
fn fit_pk_model(
    times: &[f64],
    concentrations: &[f64],
    initial_params: &[f64]
) -> &[f64] {
    // Residual function
    fn residuals(params: &[f64], t: &[f64], c_obs: &[f64], r: &![f64]) {
        // Simulate PK model with params
        var i: i64 = 0
        while i < t.len() as i64 {
            let c_pred = simulate_pk(params, t[i as usize])
            r[i as usize] = c_obs[i as usize] - c_pred
            i = i + 1
        }
    }

    let config = lm_config_default()
    let result = levenberg_marquardt(
        |p, r| residuals(p, times, concentrations, r),
        |p, J| compute_pk_jacobian(p, times, J),
        initial_params, times.len() as i64, initial_params.len() as i64,
        &config
    )

    return result.x
}
```

### Maximum Likelihood Estimation

```sio
use optimization::*

// Fit normal distribution: estimate mu, sigma from data
fn negative_log_likelihood(params: &[f64], data: &[f64]) -> f64 {
    let mu = params[0]
    let sigma = params[1]

    if sigma <= 0.0 { return 1e10 }  // Invalid

    var nll: f64 = 0.0
    let n = data.len() as i64

    var i: i64 = 0
    while i < n {
        let x = data[i as usize]
        let z = (x - mu) / sigma
        nll = nll + 0.5 * z * z + ln_f64(sigma) + 0.5 * ln_f64(2.0 * 3.14159)
        i = i + 1
    }

    return nll
}

fn main() -> i32 {
    // Sample data
    let data: [f64; 100] = [/* ... */]

    // Initial guess: sample mean and std
    var x0: [f64; 2] = [compute_mean(&data), compute_std(&data)]

    let config = bfgs_config_default()
    let result = bfgs_minimize(
        |p| negative_log_likelihood(p, &data),
        |p, g| nll_gradient(p, &data, g),
        &x0, &config
    )

    println("MLE estimates: mu = ", result.x[0], ", sigma = ", result.x[1])

    return 0
}
```

### Neural Network Training

```sio
use optimization::*
use autodiff::tape::*

fn train_network(
    X: &[[f64; 10]],  // Input features
    y: &[f64],        // Labels
    epochs: i64
) -> &[f64] {
    // Initialize weights
    var weights = random_init(110)  // 10*10 + 10 parameters

    var config = gd_config_default()
    config.learning_rate = 0.01
    config.momentum = 0.9

    var epoch: i64 = 0
    while epoch < epochs {
        // Compute loss and gradient using autodiff
        var tape = tape_new()

        // Build computation graph
        let loss_node = compute_loss(&!tape, &weights, X, y)

        // Backward pass
        tape_backward(&!tape, loss_node)

        // Update weights
        var i: i64 = 0
        while i < 110 {
            let grad = tape_grad(&tape, i)
            weights[i as usize] = weights[i as usize] - config.learning_rate * grad
            i = i + 1
        }

        epoch = epoch + 1
    }

    return &weights
}
```

## Algorithm Selection Guide

| Problem Type | Algorithm | Notes |
|--------------|-----------|-------|
| Smooth, convex | BFGS | Fast quadratic convergence |
| Large-scale smooth | L-BFGS | Memory-efficient |
| Nonlinear least squares | Levenberg-Marquardt | Exploits structure |
| Non-smooth | Nelder-Mead | No derivatives needed |
| Global, multi-modal | Differential Evolution | Escapes local minima |
| Constrained | SLSQP | Handles eq/ineq constraints |
| Simple bounds | L-BFGS-B | Projected gradient |

## Convergence Tips

### 1. Scale Variables

```sio
// Bad: variables on different scales
var x: [f64; 2] = [1e-6, 1e6]

// Good: normalize to similar magnitudes
var x_scaled: [f64; 2] = [x[0] * 1e6, x[1] * 1e-6]
// Optimize, then scale back
```

### 2. Provide Good Initial Guess

```sio
// Use domain knowledge
let x0 = physics_based_estimate(data)

// Or: run quick global search first
let x0 = differential_evolution_coarse(f, bounds)
let result = bfgs_minimize(f, grad_f, &x0, &config)
```

### 3. Check Gradient Accuracy

```sio
fn verify_gradient(f: fn(&[f64]) -> f64, grad_f: fn(&[f64], &![f64]), x: &[f64]) -> bool {
    var g_analytic: [f64; 10] = [0.0; 10]
    grad_f(x, &!g_analytic)

    var g_numerical: [f64; 10] = [0.0; 10]
    finite_difference_gradient(f, x, &!g_numerical, 1e-7)

    var max_err: f64 = 0.0
    var i: i64 = 0
    while i < 10 {
        let err = abs_f64(g_analytic[i as usize] - g_numerical[i as usize])
        if err > max_err { max_err = err }
        i = i + 1
    }

    return max_err < 1e-5
}
```

### 4. Monitor Convergence

```sio
var config = bfgs_config_default()

let result = bfgs_minimize(f, grad_f, &x0, &config)

if !result.converged {
    println("WARNING: Did not converge")
    println("  Iterations: ", result.iterations)
    println("  Gradient norm: ", result.grad_norm)

    // Try different algorithm or settings
}
```
