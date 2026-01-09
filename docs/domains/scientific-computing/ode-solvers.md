# ODE Solvers

Sounio provides a comprehensive suite of ordinary differential equation (ODE) solvers for initial value problems:

```
dy/dt = f(t, y),    y(t0) = y0
```

## Solver Selection Guide

| Problem Type | Recommended Solver | Why |
|--------------|-------------------|-----|
| Non-stiff, smooth | Tsit5 | Fast, accurate, adaptive |
| Non-stiff, general | DOPRI5 | Robust, widely used |
| Real-time/embedded | RK4 | Fixed step, predictable timing |
| Moderately stiff | Rosenbrock | L-stable, single matrix factorization |
| Very stiff | BDF | Variable order, L-stable |

### What is Stiffness?

A system is **stiff** when it contains dynamics on vastly different timescales. The eigenvalue ratio (largest/smallest eigenvalue magnitude) indicates stiffness:

- **Non-stiff**: Ratio < 100
- **Mildly stiff**: Ratio 100 - 10^4
- **Moderately stiff**: Ratio 10^4 - 10^6
- **Very stiff**: Ratio > 10^6

Stiff problems require **implicit** methods (BDF, Rosenbrock) because explicit methods need impractically small timesteps.

## Explicit Methods (Non-Stiff)

### RK4: Classic Runge-Kutta

The 4th-order Runge-Kutta method is the workhorse of numerical integration. Fixed-step, predictable, easy to understand.

**Reference:** Kutta (1901)

```sio
use ode::rk4::*

// State structure for scalar ODE
struct ScalarState {
    u: f64,
    t: f64
}

// Solution structure
struct RK4Solution {
    t_final: f64,
    u_final: f64,
    nsteps: i64,
    nfeval: i64
}

// Solve exponential decay: du/dt = -0.1*u
fn main() -> i32 {
    let sol = solve_rk4_scalar(100.0, 0.0, 10.0, 100)

    println("Final value: ", sol.u_final)
    println("Steps: ", sol.nsteps)
    println("Function evaluations: ", sol.nfeval)

    return 0
}
```

**Multi-dimensional systems:**

```sio
use ode::rk4::*

// 3-compartment pharmacokinetic model
struct PKState3 {
    gut: f64,
    central: f64,
    periph: f64,
    t: f64
}

struct PKParams3 {
    ka: f64,   // Absorption rate
    ke: f64,   // Elimination rate
    k12: f64,  // Central -> peripheral rate
    k21: f64   // Peripheral -> central rate
}

fn default_pk3_params() -> PKParams3 {
    return PKParams3 {
        ka: 1.5,
        ke: 0.1,
        k12: 0.3,
        k21: 0.15
    }
}

fn main() -> i32 {
    // 500 mg oral dose
    let s0 = PKState3 { gut: 500.0, central: 0.0, periph: 0.0, t: 0.0 }
    let p = default_pk3_params()

    // Solve for 24 hours with 1000 steps
    let sol = solve_rk4_pk3(s0, p, 24.0, 1000)

    println("Gut amount (mg): ", sol.gut)
    println("Central amount (mg): ", sol.central)
    println("Peripheral amount (mg): ", sol.periph)

    return 0
}
```

**When to use RK4:**
- Real-time systems requiring predictable timing
- Embedded systems with memory constraints
- Educational/debugging purposes
- When you know the appropriate step size

**When NOT to use RK4:**
- Stiff systems (use BDF or Rosenbrock)
- When optimal step size is unknown (use Tsit5)
- Long integrations where efficiency matters

### Tsit5: Tsitouras 5(4)

Adaptive 5th-order method with embedded 4th-order error estimation. The recommended solver for most non-stiff problems.

**Reference:** Tsitouras (2011), "Runge-Kutta pairs of order 5(4)", Computers & Mathematics with Applications

```sio
use ode::tsit5::*

// Solver configuration
struct ODEConfig {
    rtol: f64,         // Relative tolerance
    atol: f64,         // Absolute tolerance
    dt_init: f64,      // Initial step size (0 = auto)
    dt_min: f64,       // Minimum step size
    dt_max: f64,       // Maximum step size
    max_steps: i64,    // Maximum number of steps
    safety: f64,       // Safety factor for step control
    max_growth: f64,   // Maximum step growth factor
    min_shrink: f64    // Minimum step shrink factor
}

fn default_config() -> ODEConfig {
    return ODEConfig {
        rtol: 0.0001,
        atol: 0.0000001,
        dt_init: 0.1,
        dt_min: 0.000000000001,
        dt_max: 1000000.0,
        max_steps: 1000000,
        safety: 0.9,
        max_growth: 10.0,
        min_shrink: 0.2
    }
}

fn high_accuracy_config() -> ODEConfig {
    return ODEConfig {
        rtol: 0.0000000001,
        atol: 0.000000000001,
        dt_init: 0.0,          // Auto-detect
        dt_min: 0.000000000000001,
        dt_max: 1.0,
        max_steps: 10000000,
        safety: 0.9,
        max_growth: 5.0,
        min_shrink: 0.1
    }
}
```

**Solving with Tsit5:**

```sio
use ode::tsit5::*

fn main() -> i32 {
    let config = default_config()
    let sol = solve_exp_decay(100.0, 0.0, 10.0, config)

    if sol.success {
        println("Final value: ", sol.u_final)
        println("Steps taken: ", sol.nsteps)
        println("Rejected steps: ", sol.nreject)
        println("Function evaluations: ", sol.nfeval)
    } else {
        println("Integration failed")
    }

    return 0
}
```

**Adaptive step size control:**

Tsit5 adjusts the step size based on local error estimation:

1. Compute 5th-order solution and 4th-order solution
2. Estimate error as difference between solutions
3. Compute error norm: `err_norm = |error| / (atol + rtol * max(|y|, |y_new|))`
4. If `err_norm <= 1`: Accept step, potentially grow step size
5. If `err_norm > 1`: Reject step, shrink step size

New step size: `h_new = h * safety * (1/err_norm)^(1/5)`

### DOPRI5: Dormand-Prince 5(4)

Another adaptive 5th-order method, widely used in scientific computing (MATLAB's `ode45`).

**Reference:** Dormand & Prince (1980)

```sio
use ode::*

fn main() -> i32 {
    let config = default_config()
    let sol = solve_dopri5_exp(100.0, 0.0, 10.0, config)

    println("DOPRI5 result: ", sol.u_final)
    println("Steps: ", sol.nsteps)
    println("Rejected: ", sol.nreject)

    return 0
}
```

**Tsit5 vs DOPRI5:**
- Tsit5 typically requires fewer function evaluations
- DOPRI5 has better established dense output formulas
- Both achieve similar accuracy for most problems

## Implicit Methods (Stiff)

### BDF: Backward Differentiation Formulas

Variable-order (1-5) implicit multistep method. The standard choice for very stiff systems.

**Reference:** Gear (1971), Numerical Initial Value Problems in Ordinary Differential Equations

```sio
use ode::bdf::*

// BDF configuration
struct BDFConfig {
    max_order: i64,        // Maximum BDF order (1-5, default 5)
    rtol: f64,             // Relative tolerance
    atol: f64,             // Absolute tolerance
    h0: f64,               // Initial step size (0 = auto)
    h_min: f64,            // Minimum step size
    h_max: f64,            // Maximum step size
    max_newton_iter: i64,  // Max Newton iterations per step
    newton_tol: f64,       // Newton convergence tolerance
    max_steps: i64,        // Maximum total steps
    use_fd_jacobian: bool  // Use finite difference Jacobian
}

fn bdf_config_default() -> BDFConfig {
    return BDFConfig {
        max_order: 5,
        rtol: 1e-6,
        atol: 1e-9,
        h0: 0.0,
        h_min: 1e-14,
        h_max: 1e10,
        max_newton_iter: 10,
        newton_tol: 1e-10,
        max_steps: 100000,
        use_fd_jacobian: true
    }
}
```

**Solving stiff systems:**

```sio
use ode::bdf::*

// Robertson chemical kinetics - classic stiff test problem
// Eigenvalue ratio ~ 10^8
fn robertson_rhs(t: f64, y: &[f64], dydt: &![f64]) {
    dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2]
    dydt[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]
    dydt[2] = 3e7 * y[1] * y[1]
}

fn main() -> i32 {
    let y0: [f64; 3] = [1.0, 0.0, 0.0]
    var config = bdf_config_default()
    config.max_steps = 50000

    let result = bdf_solve(robertson_rhs, &y0, 0.0, 1e5, &config)

    if result.success {
        println("Final: y = [", result.y_final[0], ", ",
                result.y_final[1], ", ", result.y_final[2], "]")
        println("Steps: ", result.nsteps)
        println("Jacobian evaluations: ", result.njeval)
    }

    return 0
}
```

**BDF order selection:**

BDF methods of different orders have different stability properties:

| Order | Stability | Accuracy |
|-------|-----------|----------|
| 1 | L-stable | O(h) |
| 2 | A-stable | O(h^2) |
| 3 | A(alpha)-stable | O(h^3) |
| 4 | A(alpha)-stable | O(h^4) |
| 5 | A(alpha)-stable | O(h^5) |

The solver automatically adjusts order based on solution behavior:
- Start with BDF1 (most stable)
- Increase order as solution becomes smooth
- Decrease order when entering stiff transients

### Rosenbrock: Linearly Implicit Methods

Single-step implicit methods that avoid Newton iteration by linearizing. Faster than BDF for moderately stiff problems.

**Reference:** Rang & Angermann (2005), Hairer & Wanner (1996)

```sio
use ode::rosenbrock::*

// Rosenbrock configuration
struct RosenbrockConfig {
    rtol: f64,           // Relative tolerance
    atol: f64,           // Absolute tolerance
    h0: f64,             // Initial step size (0 = auto)
    h_min: f64,          // Minimum step size
    h_max: f64,          // Maximum step size
    max_steps: i64,      // Maximum steps
    dense_jacobian: bool // Use dense Jacobian
}

fn rosenbrock_config_default() -> RosenbrockConfig {
    return RosenbrockConfig {
        rtol: 1e-6,
        atol: 1e-9,
        h0: 0.0,
        h_min: 1e-14,
        h_max: 1e10,
        max_steps: 100000,
        dense_jacobian: true
    }
}
```

**ROS3 solver (3rd order, L-stable):**

```sio
use ode::rosenbrock::*

fn exp_decay_rhs(t: f64, y: &[f64], dydt: &![f64]) {
    let k = 0.1
    dydt[0] = -k * y[0]
}

fn main() -> i32 {
    let y0: [f64; 1] = [100.0]
    let config = rosenbrock_config_default()

    let result = ros3_solve(exp_decay_rhs, &y0, 0.0, 10.0, &config)

    if result.success {
        println("ROS3 result: ", result.y_final[0])
        println("Steps: ", result.nsteps)
        println("Jacobian evaluations: ", result.njeval)
    }

    return 0
}
```

**Rosenbrock vs BDF:**

| Aspect | Rosenbrock | BDF |
|--------|------------|-----|
| Newton iterations | None (one matrix factorization per step) | Multiple per step |
| Order | Fixed (3 or 4) | Variable (1-5) |
| Startup | No history needed | Needs history buildup |
| Very stiff | May need small steps | Handles better |
| Step size changes | Efficient | History interpolation needed |

**When to use Rosenbrock:**
- Moderately stiff problems (eigenvalue ratio < 10^6)
- When Jacobian evaluation is expensive
- Rapid step size changes needed
- Single-step methods preferred

**When to use BDF:**
- Very stiff problems (eigenvalue ratio > 10^6)
- Long integrations with nearly constant Jacobian
- Variable-order adaptation important

## ODE Definition Patterns

### Function Signature

Sounio ODE right-hand sides use mutable reference syntax `&!` for the output:

```sio
// Correct: Using &! for mutable output
fn my_ode(t: f64, y: &[f64], dydt: &![f64]) {
    dydt[0] = -0.1 * y[0]
    dydt[1] = 0.1 * y[0] - 0.05 * y[1]
}
```

### Parameterized ODEs

For ODEs with parameters, use structs:

```sio
struct ModelParams {
    k1: f64,
    k2: f64,
    k3: f64
}

fn parameterized_ode(t: f64, y: &[f64], dydt: &![f64], p: &ModelParams) {
    dydt[0] = -p.k1 * y[0]
    dydt[1] = p.k1 * y[0] - p.k2 * y[1]
    dydt[2] = p.k2 * y[1] - p.k3 * y[2]
}
```

### Time-Dependent Forcing

```sio
fn forced_oscillator(t: f64, y: &[f64], dydt: &![f64]) {
    let omega = 2.0 * 3.14159
    let forcing = sin_f64(omega * t)

    dydt[0] = y[1]
    dydt[1] = -y[0] - 0.1 * y[1] + forcing
}
```

## Error Control and Tolerances

### Understanding Tolerances

- **`rtol`** (relative tolerance): Controls relative error, typically 1e-4 to 1e-8
- **`atol`** (absolute tolerance): Controls absolute error, important near zero

The local error is scaled by:
```
scale = atol + rtol * max(|y|, |y_new|)
```

**Guidelines:**
- For values ~1: `rtol` dominates
- For values ~1e-10: `atol` dominates
- Set `atol` based on smallest meaningful value in your problem

### Convergence Monitoring

```sio
let result = solve_exp_decay(100.0, 0.0, 10.0, config)

// Check convergence
if !result.success {
    println("WARNING: Integration may be inaccurate")
    println("Reached t = ", result.t_final, " (target: 10.0)")
}

// Monitor efficiency
let efficiency = result.nsteps as f64 / result.nfeval as f64
println("Steps per function evaluation: ", efficiency)

// Check rejected steps
let reject_rate = result.nreject as f64 / result.nsteps as f64
if reject_rate > 0.3 {
    println("WARNING: High rejection rate (", reject_rate, ")")
    println("Consider reducing tolerances or using different solver")
}
```

## PBPK Applications

The ODE module includes specialized solvers for Physiologically-Based Pharmacokinetic (PBPK) models:

```sio
use ode::pbpk14::*

// 14-compartment PBPK model
// Compartments: gut, liver, kidney, heart, brain, muscle,
//               skin, adipose, bone, spleen, lung, arterial,
//               venous, rest

fn main() -> i32 {
    // Initialize with IV bolus to venous compartment
    var state = pbpk14_init()
    state.venous = 100.0  // 100 mg IV dose

    let params = pbpk14_default_params()
    let result = solve_pbpk14(state, params, 24.0, 10000)

    println("Plasma concentration at 24h: ", result.arterial)

    return 0
}
```

## Best Practices

### 1. Start with Tsit5

For most problems, start with Tsit5 and default tolerances:

```sio
let config = default_config()
let result = solve_exp_decay(y0, t0, t_end, config)
```

### 2. Check for Stiffness

If Tsit5 requires many steps or fails:

```sio
if result.nsteps > 10000 || !result.success {
    // Try BDF or Rosenbrock
    let config = bdf_config_default()
    let result = bdf_solve(rhs, &y0, t0, t_end, &config)
}
```

### 3. Validate with Tighter Tolerances

```sio
// Reference solution with tight tolerances
var tight_config = high_accuracy_config()
let reference = solve_exp_decay(y0, t0, t_end, tight_config)

// Compare with default tolerances
var default_config = default_config()
let standard = solve_exp_decay(y0, t0, t_end, default_config)

let rel_err = abs_f64(standard.u_final - reference.u_final) / abs_f64(reference.u_final)
println("Estimated relative error: ", rel_err)
```

### 4. Use Appropriate Step Limits

```sio
// For expensive RHS functions
var config = default_config()
config.max_steps = 10000  // Limit iterations

// For long integrations
config.dt_max = (t_end - t0) / 100.0  // Ensure reasonable resolution
```

## Performance Comparison

Typical performance on standard test problems (relative to RK4 baseline):

| Solver | Exponential Decay | Lotka-Volterra | Robertson |
|--------|------------------|----------------|-----------|
| RK4 | 1.0x | 1.0x | Fails |
| Tsit5 | 0.3x | 0.4x | Slow |
| DOPRI5 | 0.35x | 0.45x | Slow |
| BDF | 2.0x | 1.5x | 1.0x |
| ROS3 | 1.5x | 1.2x | 1.5x |

Lower is better (fewer function evaluations for same accuracy).
