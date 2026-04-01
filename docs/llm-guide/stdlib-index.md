# Sounio Stdlib Index

What's available, what's stable, how to import it. Use this before writing a function
from scratch — it probably already exists.

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✓ | Production — verified working in tests |
| ~ | Beta — works but may have edge cases |
| ✗ | Stub — scaffolding only, do not use |

---

## Core

### `test::helpers` ✓
Test assertion utilities. Use in every test file.

```sio
use test::helpers::{check_near, check_near_relative, check_eq_i64, TestResult}

check_near(a, b, 1e-9)           // |a - b| < tol
check_near_relative(a, b, 1e-6)  // |a - b| / max(|a|,|b|) < rtol
check_eq_i64(a, b)               // integer equality
check_near_zero(x, 1e-12)        // |x| < tol
```

### `core::result` ~
Monomorphic result types (no generics).

```sio
use core::result::{IntResult, FloatResult}
// IntResult { value: i64, ok: bool }
// FloatResult { value: f64, ok: bool }
```

### `collections::vec` ✓
Dynamic integer/float vectors.

```sio
use collections::vec::{IntVec, FloatVec}
let v = IntVec::new()
v.push(42)
let n = v.len()
let x = v.get(0)
```

---

## Epistemic / Uncertainty

### `epistemic::gum` ✓
GUM-compliant uncertainty propagation (JCGM 100:2008).

```sio
use epistemic::gum::{GUMUncertainty, type_a_uncertainty, type_b_uncertainty,
                     gum_combine, gum_expanded}

let u_a = type_a_uncertainty(0.05, 10)       // std_dev, n_measurements
let u_b = type_b_uncertainty(0.02)           // rectangular distribution
let u_c = gum_combine([u_a, u_b; 2], 2)     // combined standard uncertainty
let u_exp = gum_expanded(u_c, 0.95)          // expanded at 95% coverage
```

**Key types:**
```sio
struct GUMUncertainty {
    std_uncertainty: f64,
    degrees_of_freedom: f64,
    sensitivity: f64,
}
```

### `epistemic::knowledge` ~
`Knowledge<T>` — value + uncertainty propagation.

```sio
let k: Knowledge<f64> = measure(500.0, uncertainty: 2.5)
// Arithmetic on Knowledge<T> propagates uncertainty automatically
let k2 = k * 2.0    // uncertainty doubles
```

---

## Probability / Statistics

### `prob::normal` ✓
Normal distribution: PDF, CDF, confidence intervals.

```sio
use prob::normal::{Normal, normal_new, normal_pdf, normal_cdf, normal_ci_95}

let dist = normal_new(0.0, 1.0)       // mu, sigma
let p = normal_pdf(dist, 1.0)         // 0.242
let cdf = normal_cdf(dist, 1.64)      // ~0.95
let ci = normal_ci_95(dist)           // NormalCI { lo, hi }
```

### `prob::beta` ✓
Beta distribution: PDF, CDF, mean, variance.

```sio
use prob::beta::{Beta, beta_new, beta_pdf, beta_mean}

let b = beta_new(2.0, 5.0)
let m = beta_mean(b)          // 0.286
```

### `bayes::prior` ✓
Prior distributions for Bayesian inference.

```sio
use bayes::prior::{prior_normal, prior_uniform, prior_beta, log_prob}

let p = prior_normal(0.0, 1.0)
let lp = log_prob(p, 0.5)     // PriorEval { log_prob: f64, valid: bool }
```

### `stats::descriptive` ✓
Descriptive statistics over fixed-size arrays.

```sio
use stats::descriptive::{mean, variance, std_dev, median_sorted}

// All functions take (data: [f64; 100], n: i64) — pass actual count as n
let m = mean(data, n)
let s = std_dev(data, n)
```

**Note:** Array size is fixed at 100. For other sizes, copy the pattern.

---

## Numerical Methods

### `ode::rk4` ✓
Runge-Kutta 4th order ODE solver.

```sio
use ode::rk4::{rk4_step, rk4_integrate, RK4State, RK4Solution}

// Define your RHS as a function:
fn my_ode(u: f64, t: f64) -> f64 { 0.0 - u }   // du/dt = -u (decay)

let sol = rk4_integrate(1.0, 0.0, 5.0, 100)  // u0, t0, t_end, n_steps
// RK4Solution { values: [f64; 1024], times: [f64; 1024], n: i64 }
```

For multi-compartment ODE, use the struct-based API (`PKState3`, `PKParams3`).

### `integrate::quad` ~
Numerical quadrature (Simpson, Gauss).

```sio
use integrate::quad::{quad_simpson, quad_gauss5}
```

### `roots::bisect` ~
Root finding via bisection.

```sio
use roots::bisect::{bisect}
```

### `optimize::gradient` ~
Gradient-based optimization (SGD, simple line search).

---

## Linear Algebra / Math

### `linalg::mat` ✓
Matrix operations over flat `[f64; N]` arrays (row-major).

```sio
use linalg::mat::{mat_mul, mat_add, mat_transpose, mat_inv_2x2}

// mat_mul(A, B, m, k, n) — A is [m×k], B is [k×n], result is [m×n]
// All stored as flat [f64; N]
```

### `complex::lib` ✓
Complex numbers.

```sio
use complex::lib::{Complex, complex_new, complex_add, complex_mul,
                   complex_abs, complex_conj, complex_exp}

let z = complex_new(1.0, 2.0)
let w = complex_mul(z, complex_i())
```

### `math::sedenion64` ✓
16-dimensional sedenion arithmetic (Cayley-Dickson).

```sio
use math::sedenion64::{Sedenion64, sed64, sed64_mul, sed64_norm_sq,
                       sed64_basis, sed64_to_array, sed64_from_array}

let a = sed64_basis(1)               // e1
let b = sed64_basis(5)               // e5
let c = sed64_mul(a, b)              // sedenion product
let n = sed64_norm_sq(c)             // ||c||²
```

### `signal::fft` ✓
FFT (up to 256 points) + epistemic spectrum.

```sio
use signal::fft::{fft, ifft, fft_magnitude, EpistemicSpectrum}
```

---

## Scientific Computing

### `pbpk` ~
Physiologically-based pharmacokinetic models.

```sio
use pbpk::{PBPK14State, pbpk14_rhs, rk4_step_pbpk14}
```

### `epistemic::gum` + `ode::rk4` together
This is the dissertation pattern — GUM uncertainty through ODE:

```sio
use epistemic::gum::{type_a_uncertainty, gum_combine}
use ode::rk4::{rk4_integrate}

// 1. Set uncertain initial condition
let u0_unc = type_a_uncertainty(0.05, 5)     // 5% std uncertainty

// 2. Solve ODE
let sol = rk4_integrate(1.0, 0.0, 10.0, 200)

// 3. Report with GUM uncertainty at final time
```

---

## Neural Networks

### `nn::dense` ✓
Single-neuron dense layer with autograd tape.

```sio
use nn::dense::{DenseLayer, Tape, dense_new, dense_forward,
                dense_grad_w, dense_update, mse_loss, backward}

let layer = dense_new(0.5, 0.0)      // weight, bias
let tape = dense_forward(layer, x)   // forward pass
let grad_w = dense_grad_w(tape, target)
let updated = dense_update(layer, tape, target, 0.01)   // lr = 0.01
```

### `nn::dense2` ✓
Two-layer dense network.

### `snn::fractal_nn` ✓
Fractal Sedenion Neural Network (experimental).

```sio
use snn::fractal_nn::{FractalSedenionNN, fsnn_forward, fsnn_g2_score}
```

---

## Algebra

### `algebra::clifford` ~
Clifford algebra over f64.

### `math::g2_variety` ✓
G₂ variety and zero-divisor pairs for sedenion algebra.

```sio
use math::g2_variety::{G2Pair, g2_pair_check, g2_proximity,
                       g2_count_basis_variety_pairs}
```

---

## Importing Patterns

```sio
// Single function
use module_name::{function_name}

// Multiple
use prob::normal::{Normal, normal_new, normal_pdf}

// Nested module
use stdlib::epistemic::gum::{GUMUncertainty}

// The compiler resolves stdlib:: paths automatically when SOUNIO_STDLIB_PATH is set
```

**Module path = directory path** under `stdlib/`. If the module is at
`stdlib/prob/normal.sio`, import as `use prob::normal::{...}`.

---

## What to use for common tasks

| Task | Module |
|------|--------|
| Float comparison in tests | `test::helpers::check_near` |
| Normal distribution | `prob::normal` |
| Bayesian prior | `bayes::prior` |
| ODE solving | `ode::rk4` |
| GUM uncertainty | `epistemic::gum` |
| Matrix math | `linalg::mat` |
| Complex numbers | `complex::lib` |
| Sedenion algebra | `math::sedenion64` |
| Neural net layer | `nn::dense` |
| Descriptive stats | `stats::descriptive` |
| FFT | `signal::fft` |
