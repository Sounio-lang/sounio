# ODE Solvers API Reference

The `ode` module provides numerical solvers for ordinary differential equations (ODEs). Sounio includes both explicit and implicit methods, covering non-stiff and stiff systems commonly encountered in scientific computing, pharmacokinetics, and systems biology.

## Overview

| Solver | Order | Type | Stiffness | Use Case |
|--------|-------|------|-----------|----------|
| RK4 | 4 | Explicit, fixed-step | Non-stiff | Simple problems, predictable timing |
| Tsit5 | 5(4) | Explicit, adaptive | Non-stiff | General purpose, automatic step control |
| BDF | 1-5 | Implicit, multistep | Stiff | Chemical kinetics, PBPK |
| Rosenbrock | 3-4 | Implicit, single-step | Moderately stiff | Fast Jacobian changes |

---

## RK4 Solver

Classic 4th-order Runge-Kutta method with fixed step size. Simple, robust, and deterministic.

### Type Definitions

```sio
struct ScalarState {
    u: f64,
    t: f64
}

struct RK4Solution {
    t_final: f64,
    u_final: f64,
    nsteps: i64,
    nfeval: i64
}
```

### Functions

#### `rk4_step_scalar`

Take a single RK4 step for a scalar ODE.

```sio
fn rk4_step_scalar(s: ScalarState, dt: f64) -> ScalarStepResult
```

**Parameters:**
- `s`: Current state (value and time)
- `dt`: Step size

**Returns:** `ScalarStepResult` containing the new state after one step.

**Algorithm:**
```
k1 = f(u, t)
k2 = f(u + 0.5*dt*k1, t + 0.5*dt)
k3 = f(u + 0.5*dt*k2, t + 0.5*dt)
k4 = f(u + dt*k3, t + dt)
u_new = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

#### `solve_rk4_scalar`

Solve a scalar ODE from t0 to t_end with fixed steps.

```sio
fn solve_rk4_scalar(u0: f64, t0: f64, t_end: f64, n_steps: i64) -> RK4Solution
```

**Parameters:**
- `u0`: Initial value
- `t0`: Initial time
- `t_end`: Final time
- `n_steps`: Number of steps to take

**Example:**
```sio
// Solve exponential decay du/dt = -0.1*u
let sol = solve_rk4_scalar(100.0, 0.0, 10.0, 100)
println(sol.u_final)  // ~36.79
```

### Multi-Compartment PK Model

RK4 also supports 3-compartment pharmacokinetic models.

```sio
struct PKState3 {
    gut: f64,
    central: f64,
    periph: f64,
    t: f64
}

struct PKParams3 {
    ka: f64,   // Absorption rate
    ke: f64,   // Elimination rate
    k12: f64,  // Central -> peripheral
    k21: f64   // Peripheral -> central
}

fn solve_rk4_pk3(s0: PKState3, p: PKParams3, t_end: f64, n_steps: i64) -> RK4SolutionPK3
```

**Example:**
```sio
let s0 = PKState3 { gut: 500.0, central: 0.0, periph: 0.0, t: 0.0 }
let p = default_pk3_params()
let sol = solve_rk4_pk3(s0, p, 24.0, 1000)
println(sol.central)  // Central compartment concentration at t=24h
```

---

## Tsit5 Solver

Tsitouras 5(4) adaptive Runge-Kutta method. 5th-order accurate with embedded 4th-order error estimation. This is the recommended solver for non-stiff problems.

### Configuration

```sio
struct ODEConfig {
    rtol: f64,        // Relative tolerance (default: 1e-4)
    atol: f64,        // Absolute tolerance (default: 1e-7)
    dt_init: f64,     // Initial step size (0 = auto)
    dt_min: f64,      // Minimum step size (default: 1e-12)
    dt_max: f64,      // Maximum step size (default: 1e6)
    max_steps: i64,   // Maximum steps (default: 1e6)
    safety: f64,      // Safety factor for step control (default: 0.9)
    max_growth: f64,  // Maximum step growth factor (default: 10.0)
    min_shrink: f64   // Minimum step shrink factor (default: 0.2)
}
```

#### `default_config`

Returns default configuration suitable for most problems.

```sio
fn default_config() -> ODEConfig
```

#### `high_accuracy_config`

Returns configuration for high-precision applications.

```sio
fn high_accuracy_config() -> ODEConfig
```

### Solution Structure

```sio
struct ODESolution {
    success: bool,    // Whether integration succeeded
    nsteps: i64,      // Number of accepted steps
    nfeval: i64,      // Number of function evaluations
    nreject: i64,     // Number of rejected steps
    t_final: f64,     // Final time reached
    u_final: f64      // Solution at final time
}
```

### Functions

#### `solve_exp_decay`

Solve the exponential decay ODE with adaptive stepping.

```sio
fn solve_exp_decay(u0: f64, t0: f64, t_end: f64, config: ODEConfig) -> ODESolution
```

**Example:**
```sio
let config = default_config()
let sol = solve_exp_decay(100.0, 0.0, 10.0, config)

if sol.success {
    println("Solution: ", sol.u_final)
    println("Steps: ", sol.nsteps)
    println("Rejected: ", sol.nreject)
}
```

### Butcher Tableau

The Tsit5 method uses a 7-stage FSAL (First Same As Last) scheme:

| Property | Value |
|----------|-------|
| Order | 5 (with 4th-order embedding) |
| Stages | 7 |
| Function evals per step | 6 (FSAL) |
| Stability | Explicit, conditional |

**Reference:** Ch. Tsitouras (2011), "Runge-Kutta pairs of order 5(4)", Computers & Mathematics with Applications 62(2): 770-775

---

## BDF Solver

Backward Differentiation Formula methods for stiff ODEs. BDF methods are implicit multistep methods that are L-stable (all orders) or A-stable (orders 1-2).

### When to Use BDF

Use BDF when your system exhibits stiffness:
- Eigenvalue spread > 1000 (lambda_max / lambda_min)
- Explicit methods require impractically small steps
- Chemical kinetics with fast equilibration
- Circuit simulation with parasitic capacitances
- PBPK with fast blood circulation

### Configuration

```sio
pub struct BDFConfig {
    pub max_order: i64,       // Maximum BDF order (1-5, default 5)
    pub rtol: f64,            // Relative tolerance (default: 1e-6)
    pub atol: f64,            // Absolute tolerance (default: 1e-9)
    pub h0: f64,              // Initial step size (0 = auto)
    pub h_min: f64,           // Minimum step size (default: 1e-14)
    pub h_max: f64,           // Maximum step size (default: 1e10)
    pub max_newton_iter: i64, // Newton iterations per step (default: 10)
    pub newton_tol: f64,      // Newton tolerance (default: 1e-10)
    pub max_steps: i64,       // Maximum total steps (default: 100000)
    pub use_fd_jacobian: bool // Use finite difference Jacobian (default: true)
}
```

#### `bdf_config_default`

Returns default BDF configuration.

```sio
pub fn bdf_config_default() -> BDFConfig
```

### Result Structure

```sio
pub struct BDFResult {
    pub y_final: &[f64],    // Solution at final time
    pub t_final: f64,       // Final time reached
    pub success: bool,      // Whether solver succeeded
    pub nsteps: i64,        // Number of steps taken
    pub nfeval: i64,        // Number of function evaluations
    pub njeval: i64,        // Number of Jacobian evaluations
    pub nreject: i64,       // Number of rejected steps
    pub final_order: i64,   // Final BDF order used
    pub message: &str       // Status message
}
```

### Functions

#### `bdf_solve`

Solve an ODE system using BDF method.

```sio
pub fn bdf_solve(
    rhs: fn(f64, &[f64], &![f64]),
    y0: &[f64],
    t0: f64,
    t_end: f64,
    config: &BDFConfig
) -> BDFResult
```

**Parameters:**
- `rhs`: Function f(t, y, dydt) computing the right-hand side
- `y0`: Initial condition vector
- `t0`: Initial time
- `t_end`: Final time
- `config`: Solver configuration

**Example:**
```sio
use ode::bdf::*

// Robertson chemical kinetics (classic stiff test problem)
fn robertson_rhs(t: f64, y: &[f64], dydt: &![f64]) {
    dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2]
    dydt[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]
    dydt[2] = 3e7 * y[1] * y[1]
}

let config = bdf_config_default()
let y0 = [1.0, 0.0, 0.0]
let result = bdf_solve(robertson_rhs, &y0, 0.0, 1e5, &config)

if result.success {
    println("Final: ", result.y_final[0], result.y_final[1], result.y_final[2])
}
```

### BDF Coefficients

BDF-k approximates the derivative using k previous solution values:

| Order | Formula | Gamma | Stability |
|-------|---------|-------|-----------|
| 1 | y_n - y_{n-1} = h * f_n | 1.0 | A-stable |
| 2 | (3/2)y_n - 2y_{n-1} + (1/2)y_{n-2} = h * f_n | 2/3 | A-stable |
| 3 | (11/6)y_n - 3y_{n-1} + (3/2)y_{n-2} - (1/3)y_{n-3} = h * f_n | 6/11 | A(alpha)-stable |
| 4 | Higher-order coefficients | 12/25 | A(alpha)-stable |
| 5 | Higher-order coefficients | 60/137 | A(alpha)-stable |

**Note:** BDF methods of order > 2 are not A-stable but are A(alpha)-stable with alpha decreasing as order increases.

---

## Rosenbrock Solver

Rosenbrock (linearly implicit) methods avoid Newton iteration by linearizing the implicit equations. Each step requires only ONE matrix factorization.

### When to Use Rosenbrock

Prefer Rosenbrock over BDF when:
- Problem is moderately stiff (eigenvalue ratio < 10^6)
- Jacobian evaluation is expensive
- You want embedded error estimation
- Step size changes frequently

Prefer BDF when:
- Problem is very stiff (eigenvalue ratio > 10^6)
- Long integration with nearly constant Jacobian

### Configuration

```sio
pub struct RosenbrockConfig {
    pub rtol: f64,          // Relative tolerance (default: 1e-6)
    pub atol: f64,          // Absolute tolerance (default: 1e-9)
    pub h0: f64,            // Initial step size (0 = auto)
    pub h_min: f64,         // Minimum step size (default: 1e-14)
    pub h_max: f64,         // Maximum step size (default: 1e10)
    pub max_steps: i64,     // Maximum steps (default: 100000)
    pub dense_jacobian: bool // Use dense Jacobian (default: true)
}
```

#### `rosenbrock_config_default`

Returns default Rosenbrock configuration.

```sio
pub fn rosenbrock_config_default() -> RosenbrockConfig
```

### Result Structure

```sio
pub struct RosenbrockResult {
    pub y_final: &[f64],    // Solution at final time
    pub t_final: f64,       // Final time reached
    pub success: bool,      // Success flag
    pub nsteps: i64,        // Number of steps
    pub nfeval: i64,        // Function evaluations
    pub njeval: i64,        // Jacobian evaluations
    pub nreject: i64,       // Rejected steps
    pub message: &str       // Status message
}
```

### Methods Implemented

#### ROS3 (3rd Order, L-stable)

```sio
pub fn ros3_solve(
    rhs: fn(f64, &[f64], &![f64]),
    y0: &[f64],
    t0: f64,
    t_end: f64,
    config: &RosenbrockConfig
) -> RosenbrockResult
```

**Properties:**
- 3 stages
- L-stable (damps all stiff modes)
- Embedded error estimation

**Example:**
```sio
use ode::rosenbrock::*

fn exp_decay_rhs(t: f64, y: &[f64], dydt: &![f64]) {
    let k = 0.1
    dydt[0] = -k * y[0]
}

let y0: [f64; 1] = [100.0]
let config = rosenbrock_config_default()
let result = ros3_solve(exp_decay_rhs, &y0, 0.0, 10.0, &config)
```

### Algorithm

At each step, solve s linear systems:

```
(I - gamma*h*J) k_i = h*f(t + alpha_i*h, y + sum(a_ij*k_j)) + h*J*sum(c_ij*k_j)
y_{n+1} = y_n + sum(b_i * k_i)
```

where J = df/dy is evaluated once per step.

---

## Solver Selection Guide

### Decision Tree

```
Is the problem stiff?
|
+-- No --> Use Tsit5 (adaptive, robust)
|
+-- Yes --> How stiff?
    |
    +-- Moderately stiff (ratio < 10^6) --> Rosenbrock
    |
    +-- Very stiff (ratio > 10^6) --> BDF
```

### Problem Characteristics

| Problem Type | Recommended Solver |
|--------------|-------------------|
| Oscillators (harmonic, Van der Pol) | Tsit5 |
| Chemical kinetics | BDF |
| PBPK models | BDF or Rosenbrock |
| Neural ODEs | Tsit5 |
| Circuit simulation | BDF |
| Population dynamics | Tsit5 |

---

## Error Control

All adaptive solvers use the error estimate:

```
err = |y_new - y_embedded| / (atol + rtol * max(|y|, |y_new|))
```

A step is accepted if `err <= 1.0`.

### Step Size Control

New step size is computed as:

```
h_new = h * safety * (1/err)^(1/(order+1))
```

with limits:
- `h_new >= min_shrink * h`
- `h_new <= max_growth * h`
- `h_min <= h_new <= h_max`

---

## Performance Notes

1. **RK4**: 4 function evaluations per step, predictable cost
2. **Tsit5**: 6-7 function evaluations per step (FSAL), adaptive
3. **BDF**: 1 Jacobian + O(n) Newton iterations per step
4. **Rosenbrock**: 1 Jacobian + 1 factorization + O(s) solves per step

For systems with n variables:
- Explicit methods: O(n) per step
- Implicit methods: O(n^2) to O(n^3) per step (due to linear algebra)

---

## See Also

- [Event Handling](events.md)
- [Linear Algebra](../linalg/matrices.md)
- [Automatic Differentiation](../autodiff/index.md) (for analytical Jacobians)
