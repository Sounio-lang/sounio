//! stdlib/epistemic/propagate.d
//!
//! Variance Propagation: The Delta Method
//!
//! When computing f(X) where X has variance, the result has variance:
//! Var(f(X)) ≈ (df/dx)² · Var(X)
//!
//! This module provides explicit propagation rules for:
//! - Univariate functions (exp, log, sin, cos, etc.)
//! - Multivariate functions (products, quotients, compositions)
//! - Monte Carlo propagation for complex cases
//!
//! # Example
//!
//! ```demetrios
//! use epistemic::{Knowledge, propagate}
//!
//! let x = Knowledge::measured(10.0, 1.0, "sensor")
//!
//! // Delta method propagation
//! let y = propagate::exp(x)  // Var(e^x) ≈ e^(2x) · Var(x)
//!
//! // Monte Carlo for complex functions
//! let z = propagate::monte_carlo(x, |v| complex_function(v), 10000)
//! ```

use core::{Option, Vec}
use epistemic::knowledge::{Knowledge, BetaConfidence, Source, Provenance}

// ============================================================================
// UNIVARIATE PROPAGATION (already in Knowledge, but explicit here)
// ============================================================================

/// Propagate variance through exponential: e^x
/// Var(e^X) ≈ e^(2X) · Var(X)
pub fn exp(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = exp_f64(x.value)
    let variance = exp_f64(2.0 * x.value) * x.variance

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.95),
        provenance: x.provenance.with_step("exp"),
    }
}

/// Propagate variance through natural log: ln(x)
/// Var(ln X) ≈ Var(X) / X²
pub fn ln(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = ln_f64(x.value)
    let variance = x.variance / (x.value * x.value)

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.95),
        provenance: x.provenance.with_step("ln"),
    }
}

/// Propagate variance through square root: √x
/// Var(√X) ≈ Var(X) / (4X)
pub fn sqrt(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = sqrt_f64(x.value)
    let variance = x.variance / (4.0 * x.value)

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.98),
        provenance: x.provenance.with_step("sqrt"),
    }
}

/// Propagate variance through square: x²
/// Var(X²) ≈ 4X² · Var(X)
pub fn square(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = x.value * x.value
    let variance = 4.0 * value * x.variance

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.98),
        provenance: x.provenance.with_step("square"),
    }
}

/// Propagate variance through power: x^n
/// Var(X^n) ≈ n² · X^(2n-2) · Var(X)
pub fn pow(x: Knowledge<f64>, n: f64) -> Knowledge<f64> {
    let value = pow_f64(x.value, n)
    let derivative = n * pow_f64(x.value, n - 1.0)
    let variance = derivative * derivative * x.variance

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.95),
        provenance: x.provenance.with_step("pow"),
    }
}

/// Propagate variance through sine: sin(x)
/// Var(sin X) ≈ cos²(X) · Var(X)
pub fn sin(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = sin_f64(x.value)
    let cos_x = cos_f64(x.value)
    let variance = cos_x * cos_x * x.variance

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.98),
        provenance: x.provenance.with_step("sin"),
    }
}

/// Propagate variance through cosine: cos(x)
/// Var(cos X) ≈ sin²(X) · Var(X)
pub fn cos(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = cos_f64(x.value)
    let sin_x = sin_f64(x.value)
    let variance = sin_x * sin_x * x.variance

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.98),
        provenance: x.provenance.with_step("cos"),
    }
}

/// Propagate variance through tangent: tan(x)
/// Var(tan X) ≈ sec⁴(X) · Var(X)
pub fn tan(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = sin_f64(x.value) / cos_f64(x.value)
    let sec = 1.0 / cos_f64(x.value)
    let sec4 = sec * sec * sec * sec
    let variance = sec4 * x.variance

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.95),
        provenance: x.provenance.with_step("tan"),
    }
}

/// Propagate variance through inverse: 1/x
/// Var(1/X) ≈ Var(X) / X⁴
pub fn inverse(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = 1.0 / x.value
    let x4 = x.value * x.value * x.value * x.value
    let variance = x.variance / x4

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.97),
        provenance: x.provenance.with_step("inverse"),
    }
}

/// Propagate variance through logistic sigmoid: 1/(1 + e^(-x))
/// Var(σ(X)) ≈ σ(X)² · (1 - σ(X))² · Var(X)
pub fn sigmoid(x: Knowledge<f64>) -> Knowledge<f64> {
    let s = 1.0 / (1.0 + exp_f64(0.0 - x.value))
    let derivative = s * (1.0 - s)
    let variance = derivative * derivative * x.variance

    Knowledge {
        value: s,
        variance: variance,
        confidence: x.confidence.decay(0.95),
        provenance: x.provenance.with_step("sigmoid"),
    }
}

// ============================================================================
// BIVARIATE PROPAGATION
// ============================================================================

/// Sum with variance propagation (independent variables)
/// Var(X + Y) = Var(X) + Var(Y)
pub fn sum(x: Knowledge<f64>, y: Knowledge<f64>) -> Knowledge<f64> {
    Knowledge {
        value: x.value + y.value,
        variance: x.variance + y.variance,
        confidence: x.confidence.combine(&y.confidence).decay(0.99),
        provenance: Provenance {
            source: Source::Computed { operation: "sum" },
            steps: Vec::new(),
        },
    }
}

/// Difference with variance propagation
/// Var(X - Y) = Var(X) + Var(Y)
pub fn diff(x: Knowledge<f64>, y: Knowledge<f64>) -> Knowledge<f64> {
    Knowledge {
        value: x.value - y.value,
        variance: x.variance + y.variance,
        confidence: x.confidence.combine(&y.confidence).decay(0.99),
        provenance: Provenance {
            source: Source::Computed { operation: "diff" },
            steps: Vec::new(),
        },
    }
}

/// Product with variance propagation
/// Var(XY) ≈ Y²Var(X) + X²Var(Y)
pub fn product(x: Knowledge<f64>, y: Knowledge<f64>) -> Knowledge<f64> {
    let variance = y.value * y.value * x.variance
                 + x.value * x.value * y.variance

    Knowledge {
        value: x.value * y.value,
        variance: variance,
        confidence: x.confidence.combine(&y.confidence).decay(0.98),
        provenance: Provenance {
            source: Source::Computed { operation: "product" },
            steps: Vec::new(),
        },
    }
}

/// Quotient with variance propagation
/// Var(X/Y) ≈ (1/Y²)Var(X) + (X²/Y⁴)Var(Y)
pub fn quotient(x: Knowledge<f64>, y: Knowledge<f64>) -> Knowledge<f64> {
    let y2 = y.value * y.value
    let y4 = y2 * y2
    let variance = x.variance / y2 + x.value * x.value * y.variance / y4

    Knowledge {
        value: x.value / y.value,
        variance: variance,
        confidence: x.confidence.combine(&y.confidence).decay(0.97),
        provenance: Provenance {
            source: Source::Computed { operation: "quotient" },
            steps: Vec::new(),
        },
    }
}

/// Linear combination: a*X + b*Y
/// Var(aX + bY) = a²Var(X) + b²Var(Y)
pub fn linear_combo(
    a: f64, x: Knowledge<f64>,
    b: f64, y: Knowledge<f64>,
) -> Knowledge<f64> {
    Knowledge {
        value: a * x.value + b * y.value,
        variance: a * a * x.variance + b * b * y.variance,
        confidence: x.confidence.combine(&y.confidence).decay(0.99),
        provenance: Provenance {
            source: Source::Computed { operation: "linear_combo" },
            steps: Vec::new(),
        },
    }
}

// ============================================================================
// CORRELATED PROPAGATION
// ============================================================================

/// Sum with known correlation
/// Var(X + Y) = Var(X) + Var(Y) + 2·ρ·σ_X·σ_Y
pub fn sum_correlated(
    x: Knowledge<f64>,
    y: Knowledge<f64>,
    correlation: f64,
) -> Knowledge<f64> {
    let cov = correlation * sqrt_f64(x.variance) * sqrt_f64(y.variance)
    let variance = x.variance + y.variance + 2.0 * cov

    Knowledge {
        value: x.value + y.value,
        variance: max_f64(variance, 0.0),  // Ensure non-negative
        confidence: x.confidence.combine(&y.confidence).decay(0.98),
        provenance: Provenance {
            source: Source::Computed { operation: "sum_correlated" },
            steps: Vec::new(),
        },
    }
}

/// Product with known correlation
/// Var(XY) ≈ Y²Var(X) + X²Var(Y) + 2XY·ρ·σ_X·σ_Y
pub fn product_correlated(
    x: Knowledge<f64>,
    y: Knowledge<f64>,
    correlation: f64,
) -> Knowledge<f64> {
    let cov = correlation * sqrt_f64(x.variance) * sqrt_f64(y.variance)
    let variance = y.value * y.value * x.variance
                 + x.value * x.value * y.variance
                 + 2.0 * x.value * y.value * cov

    Knowledge {
        value: x.value * y.value,
        variance: max_f64(variance, 0.0),
        confidence: x.confidence.combine(&y.confidence).decay(0.97),
        provenance: Provenance {
            source: Source::Computed { operation: "product_correlated" },
            steps: Vec::new(),
        },
    }
}

// ============================================================================
// MONTE CARLO PROPAGATION
// ============================================================================

/// Monte Carlo variance propagation for arbitrary functions
///
/// When the delta method is insufficient (non-differentiable functions,
/// complex compositions), Monte Carlo provides a numerical estimate.
///
/// Samples from N(value, variance) and computes empirical variance of f(samples).
pub fn monte_carlo<F>(
    x: Knowledge<f64>,
    f: F,
    n_samples: i64,
) -> Knowledge<f64>
where F: fn(f64) -> f64
{
    let std = sqrt_f64(x.variance)

    // Generate samples and apply function
    var sum = 0.0
    var sum_sq = 0.0
    var rng_state = 12345_u64

    var i = 0
    while i < n_samples {
        // Box-Muller for normal sample
        let (u1, u2, new_state) = random_pair(rng_state)
        rng_state = new_state
        let z = sqrt_f64(-2.0 * ln_f64(u1)) * cos_f64(2.0 * 3.141592653589793 * u2)
        let sample = x.value + z * std

        // Apply function
        let y = f(sample)
        sum = sum + y
        sum_sq = sum_sq + y * y

        i = i + 1
    }

    let n = n_samples as f64
    let mean = sum / n
    let variance = (sum_sq / n) - (mean * mean)

    // Confidence based on sample size
    let conf = BetaConfidence::from_rate(0.9, min_f64(n, 1000.0))

    Knowledge {
        value: mean,
        variance: max_f64(variance, 0.0),
        confidence: conf,
        provenance: x.provenance.with_step("monte_carlo"),
    }
}

/// Monte Carlo for bivariate function
pub fn monte_carlo_2d<F>(
    x: Knowledge<f64>,
    y: Knowledge<f64>,
    f: F,
    n_samples: i64,
) -> Knowledge<f64>
where F: fn(f64, f64) -> f64
{
    let std_x = sqrt_f64(x.variance)
    let std_y = sqrt_f64(y.variance)

    var sum = 0.0
    var sum_sq = 0.0
    var rng_state = 67890_u64

    var i = 0
    while i < n_samples {
        // Two independent normals
        let (u1, u2, s1) = random_pair(rng_state)
        let (u3, u4, s2) = random_pair(s1)
        rng_state = s2

        let z1 = sqrt_f64(-2.0 * ln_f64(u1)) * cos_f64(2.0 * 3.141592653589793 * u2)
        let z2 = sqrt_f64(-2.0 * ln_f64(u3)) * cos_f64(2.0 * 3.141592653589793 * u4)

        let sample_x = x.value + z1 * std_x
        let sample_y = y.value + z2 * std_y

        let result = f(sample_x, sample_y)
        sum = sum + result
        sum_sq = sum_sq + result * result

        i = i + 1
    }

    let n = n_samples as f64
    let mean = sum / n
    let variance = (sum_sq / n) - (mean * mean)

    Knowledge {
        value: mean,
        variance: max_f64(variance, 0.0),
        confidence: BetaConfidence::from_rate(0.9, min_f64(n, 1000.0)),
        provenance: Provenance {
            source: Source::Computed { operation: "monte_carlo_2d" },
            steps: Vec::new(),
        },
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn exp_f64(x: f64) -> f64 {
    if x > 10.0 { return 22026.465794806718 * exp_f64(x - 10.0) }
    if x < -10.0 { return 0.00004539992976248485 * exp_f64(x + 10.0) }

    var result = 1.0
    var term = 1.0
    var i = 1
    while i < 20 {
        term = term * x / (i as f64)
        result = result + term
        i = i + 1
    }
    result
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 999999.0 }
    if x == 1.0 { return 0.0 }

    let y = (x - 1.0) / (x + 1.0)
    var result = 0.0
    var term = y
    var i = 1
    while i < 30 {
        result = result + term / (i as f64)
        term = term * y * y
        i = i + 2
    }
    2.0 * result
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y
}

fn pow_f64(base: f64, exp: f64) -> f64 {
    exp_f64(exp * ln_f64(base))
}

fn sin_f64(x: f64) -> f64 {
    // Reduce to [-π, π]
    let pi = 3.141592653589793
    var x_mod = x
    while x_mod > pi { x_mod = x_mod - 2.0 * pi }
    while x_mod < 0.0 - pi { x_mod = x_mod + 2.0 * pi }

    // Taylor series
    var result = x_mod
    var term = x_mod
    var i = 1
    while i < 10 {
        term = 0.0 - term * x_mod * x_mod / ((2.0 * i as f64) * (2.0 * i as f64 + 1.0))
        result = result + term
        i = i + 1
    }
    result
}

fn cos_f64(x: f64) -> f64 {
    let pi = 3.141592653589793
    sin_f64(x + pi / 2.0)
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// Simple LCG random number generator
fn random_pair(state: u64) -> (f64, f64, u64) {
    let a = 6364136223846793005_u64
    let c = 1442695040888963407_u64

    let s1 = state * a + c
    let s2 = s1 * a + c

    let u1 = ((s1 >> 32) as f64) / 4294967296.0
    let u2 = ((s2 >> 32) as f64) / 4294967296.0

    (max_f64(u1, 0.0000001), u2, s2)
}
