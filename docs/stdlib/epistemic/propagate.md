# Uncertainty Propagation

This module provides functions for propagating uncertainty through arbitrary computations. When computing `f(X)` where `X` has variance, the result has variance that must be correctly computed.

## The Delta Method

The delta method (also called error propagation or linearization) approximates variance through first-order Taylor expansion:

```
Var(f(X)) = (df/dx)^2 * Var(X)
```

For multivariate functions:

```
Var(f(X,Y)) = (df/dx)^2 * Var(X) + (df/dy)^2 * Var(Y) + 2*(df/dx)*(df/dy)*Cov(X,Y)
```

## Univariate Functions

### `exp`

Propagate variance through exponential.

```sio
pub fn exp(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(e^X) = e^(2X) * Var(X)`

**Derivation:**
- Derivative: `d(e^x)/dx = e^x`
- Delta method: `Var(e^X) = (e^X)^2 * Var(X) = e^(2X) * Var(X)`

**Example:**
```sio
use epistemic::propagate

let x = Knowledge::measured(2.0, 0.1, "sensor")
let y = propagate::exp(x)
// y.value = e^2 = 7.389
// y.variance = e^4 * 0.1 = 5.459
```

**Confidence Decay:** 0.95

### `ln`

Propagate variance through natural logarithm.

```sio
pub fn ln(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(ln(X)) = Var(X) / X^2`

**Derivation:**
- Derivative: `d(ln(x))/dx = 1/x`
- Delta method: `Var(ln(X)) = (1/X)^2 * Var(X) = Var(X) / X^2`

**Example:**
```sio
let x = Knowledge::measured(10.0, 1.0, "sensor")
let y = propagate::ln(x)
// y.value = ln(10) = 2.303
// y.variance = 1.0 / 100 = 0.01
```

**Confidence Decay:** 0.95

### `sqrt`

Propagate variance through square root.

```sio
pub fn sqrt(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(sqrt(X)) = Var(X) / (4X)`

**Derivation:**
- Derivative: `d(sqrt(x))/dx = 1/(2*sqrt(x))`
- Delta method: `Var(sqrt(X)) = 1/(4X) * Var(X)`

**Example:**
```sio
let x = Knowledge::measured(100.0, 4.0, "sensor")
let y = propagate::sqrt(x)
// y.value = 10.0
// y.variance = 4.0 / 400 = 0.01
```

**Confidence Decay:** 0.98

### `square`

Propagate variance through squaring.

```sio
pub fn square(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(X^2) = 4X^2 * Var(X)`

**Derivation:**
- Derivative: `d(x^2)/dx = 2x`
- Delta method: `Var(X^2) = (2X)^2 * Var(X) = 4X^2 * Var(X)`

**Confidence Decay:** 0.98

### `pow`

Propagate variance through power function.

```sio
pub fn pow(x: Knowledge<f64>, n: f64) -> Knowledge<f64>
```

**Variance Formula:** `Var(X^n) = n^2 * X^(2n-2) * Var(X)`

**Derivation:**
- Derivative: `d(x^n)/dx = n * x^(n-1)`
- Delta method: `Var(X^n) = (n * X^(n-1))^2 * Var(X)`

**Confidence Decay:** 0.95

### `sin`

Propagate variance through sine.

```sio
pub fn sin(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(sin(X)) = cos^2(X) * Var(X)`

**Confidence Decay:** 0.98

### `cos`

Propagate variance through cosine.

```sio
pub fn cos(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(cos(X)) = sin^2(X) * Var(X)`

**Confidence Decay:** 0.98

### `tan`

Propagate variance through tangent.

```sio
pub fn tan(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(tan(X)) = sec^4(X) * Var(X)`

**Confidence Decay:** 0.95

### `inverse`

Propagate variance through inverse (1/x).

```sio
pub fn inverse(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(1/X) = Var(X) / X^4`

**Derivation:**
- Derivative: `d(1/x)/dx = -1/x^2`
- Delta method: `Var(1/X) = (1/X^2)^2 * Var(X) = Var(X) / X^4`

**Confidence Decay:** 0.97

### `sigmoid`

Propagate variance through logistic sigmoid.

```sio
pub fn sigmoid(x: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(sigma(X)) = sigma(X)^2 * (1 - sigma(X))^2 * Var(X)`

Where `sigma(x) = 1 / (1 + e^(-x))`

**Confidence Decay:** 0.95

## Bivariate Functions

### `sum`

Sum with variance propagation (independent variables).

```sio
pub fn sum(x: Knowledge<f64>, y: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(X + Y) = Var(X) + Var(Y)`

### `diff`

Difference with variance propagation.

```sio
pub fn diff(x: Knowledge<f64>, y: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(X - Y) = Var(X) + Var(Y)`

**Note:** Variances ADD for subtraction, they do not subtract!

### `product`

Product with variance propagation.

```sio
pub fn product(x: Knowledge<f64>, y: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(XY) = Y^2 * Var(X) + X^2 * Var(Y)`

### `quotient`

Quotient with variance propagation.

```sio
pub fn quotient(x: Knowledge<f64>, y: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(X/Y) = (1/Y^2) * Var(X) + (X^2/Y^4) * Var(Y)`

### `linear_combo`

Linear combination: `a*X + b*Y`.

```sio
pub fn linear_combo(
    a: f64, x: Knowledge<f64>,
    b: f64, y: Knowledge<f64>,
) -> Knowledge<f64>
```

**Variance Formula:** `Var(aX + bY) = a^2 * Var(X) + b^2 * Var(Y)`

**Example:**
```sio
let x = Knowledge::measured(10.0, 1.0, "sensor_A")
let y = Knowledge::measured(20.0, 4.0, "sensor_B")
let z = propagate::linear_combo(2.0, x, 3.0, y)
// z.value = 2*10 + 3*20 = 80
// z.variance = 4*1 + 9*4 = 40
```

## Correlated Propagation

When variables are correlated, the covariance term must be included.

### `sum_correlated`

Sum with known correlation.

```sio
pub fn sum_correlated(
    x: Knowledge<f64>,
    y: Knowledge<f64>,
    correlation: f64,
) -> Knowledge<f64>
```

**Variance Formula:**
```
Var(X + Y) = Var(X) + Var(Y) + 2 * rho * sigma_X * sigma_Y
```

**Parameters:**
- `correlation`: Pearson correlation coefficient (-1 to 1)

**Example:**
```sio
let x = Knowledge::measured(10.0, 1.0, "sensor_A")
let y = Knowledge::measured(20.0, 1.0, "sensor_B")

// Positively correlated (rho = 0.8)
let sum_pos = propagate::sum_correlated(x, y, 0.8)
// Variance = 1 + 1 + 2*0.8*1*1 = 3.6

// Negatively correlated (rho = -0.8)
let sum_neg = propagate::sum_correlated(x, y, -0.8)
// Variance = 1 + 1 - 2*0.8*1*1 = 0.4
```

### `product_correlated`

Product with known correlation.

```sio
pub fn product_correlated(
    x: Knowledge<f64>,
    y: Knowledge<f64>,
    correlation: f64,
) -> Knowledge<f64>
```

**Variance Formula:**
```
Var(XY) = Y^2 * Var(X) + X^2 * Var(Y) + 2*X*Y * rho * sigma_X * sigma_Y
```

## Monte Carlo Propagation

When the delta method is insufficient (non-differentiable functions, complex compositions, or highly nonlinear functions), Monte Carlo provides a numerical estimate.

### `monte_carlo`

Monte Carlo variance propagation for arbitrary univariate functions.

```sio
pub fn monte_carlo<F>(
    x: Knowledge<f64>,
    f: F,
    n_samples: i64,
) -> Knowledge<f64>
where F: fn(f64) -> f64
```

**Parameters:**
- `x`: Input knowledge value
- `f`: Arbitrary function to propagate through
- `n_samples`: Number of Monte Carlo samples

**Algorithm:**
1. Sample `n_samples` values from `N(x.value, sqrt(x.variance))`
2. Apply function `f` to each sample
3. Compute empirical mean and variance of results

**Example:**
```sio
let x = Knowledge::measured(2.0, 0.5, "sensor")

// Complex function that may be non-differentiable
fn complex_fn(v: f64) -> f64 {
    if v < 1.5 {
        v * v
    } else {
        3.0 * v - 2.25
    }
}

let result = propagate::monte_carlo(x, complex_fn, 10000)
```

**When to Use Monte Carlo:**
- Non-differentiable functions
- Functions with discontinuities
- Highly nonlinear functions where linearization fails
- Complex compositions of many functions
- When relative uncertainty exceeds 30%

### `monte_carlo_2d`

Monte Carlo for bivariate functions.

```sio
pub fn monte_carlo_2d<F>(
    x: Knowledge<f64>,
    y: Knowledge<f64>,
    f: F,
    n_samples: i64,
) -> Knowledge<f64>
where F: fn(f64, f64) -> f64
```

**Example:**
```sio
let a = Knowledge::measured(10.0, 1.0, "sensor_A")
let b = Knowledge::measured(5.0, 0.5, "sensor_B")

fn complex_2d(x: f64, y: f64) -> f64 {
    (x * y).sqrt() + (x / y).ln()
}

let result = propagate::monte_carlo_2d(a, b, complex_2d, 10000)
```

## Propagation Rules Summary

| Function | Formula | Var(f(X)) |
|----------|---------|-----------|
| `X + Y` | X + Y | `Var(X) + Var(Y)` |
| `X - Y` | X - Y | `Var(X) + Var(Y)` |
| `X * Y` | XY | `Y^2 Var(X) + X^2 Var(Y)` |
| `X / Y` | X/Y | `Var(X)/Y^2 + X^2 Var(Y)/Y^4` |
| `e^X` | exp(X) | `e^(2X) Var(X)` |
| `ln(X)` | log(X) | `Var(X)/X^2` |
| `sqrt(X)` | sqrt(X) | `Var(X)/(4X)` |
| `X^2` | X*X | `4X^2 Var(X)` |
| `X^n` | pow(X,n) | `n^2 X^(2n-2) Var(X)` |
| `sin(X)` | sin(X) | `cos^2(X) Var(X)` |
| `cos(X)` | cos(X) | `sin^2(X) Var(X)` |
| `1/X` | 1/X | `Var(X)/X^4` |
| `aX + bY` | linear | `a^2 Var(X) + b^2 Var(Y)` |

## GUM Compliance

This module implements uncertainty propagation according to:

- **JCGM 100:2008** - Guide to the Expression of Uncertainty in Measurement (GUM)
- **JCGM 101:2008** - Supplement 1 to the GUM (Monte Carlo methods)

Key GUM principles:
1. Standard uncertainties are expressed as standard deviations
2. Combined uncertainty uses law of propagation of uncertainties
3. Correlation must be considered when inputs are not independent
4. Monte Carlo is the reference method for nonlinear cases

## Example: Complete Uncertainty Budget

```sio
use epistemic::{Knowledge, propagate}

fn calculate_concentration() {
    // Input quantities with their uncertainties
    let mass = Knowledge::measured(12.456, 0.0001, "balance")  // 0.01 mg uncertainty
    let volume = Knowledge::measured(100.0, 0.1, "flask")       // 0.1 mL uncertainty

    // Calculate concentration
    let concentration = mass / volume  // Propagation automatic

    // Explicit propagation for derived quantity
    let ln_conc = propagate::ln(concentration)

    // Report uncertainty budget
    println("Mass: {} +/- {} g", mass.get(), mass.std())
    println("Volume: {} +/- {} mL", volume.get(), volume.std())
    println("Concentration: {} +/- {} g/mL", concentration.get(), concentration.std())
    println("ln(Concentration): {} +/- {}", ln_conc.get(), ln_conc.std())

    // 95% coverage interval
    let (lo, hi) = concentration.ci95()
    println("95% CI: [{}, {}] g/mL", lo, hi)
}
```

## See Also

- [Knowledge<T> API Reference](knowledge.md)
- [MCMC Sampling](mcmc.md) - For posterior inference
- [Meta-Analysis](meta.md) - For combining studies
