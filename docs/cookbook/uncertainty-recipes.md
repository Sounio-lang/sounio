# Uncertainty Recipes

Practical recipes for working with `Knowledge<T>` and epistemic values in Sounio.

## Combining Measurements from Multiple Sources

### Problem

You have measurements of the same quantity from different instruments and want to combine them into a single best estimate with appropriate uncertainty.

### Solution

Use inverse-variance weighting to combine measurements:

```sio
use epistemic::core::*

// Two measurements of the same quantity
let measurement_1 = epistemic_std(100.0, 5.0, 0.90)  // value=100, u=5, conf=0.90
let measurement_2 = epistemic_std(102.0, 4.0, 0.85)  // value=102, u=4, conf=0.85

// Fuse using inverse-variance weighting
let combined = fuse_measurements(measurement_1, measurement_2)

// Result:
// - Value: weighted average (closer to measurement_2 due to lower uncertainty)
// - Uncertainty: reduced (combined evidence is more precise)
// - Confidence: max(0.90, 0.85) = 0.90 (more evidence increases confidence)
```

### Discussion

The `fuse_measurements` function implements inverse-variance weighting:

```
weight_i = 1 / variance_i
combined_value = sum(weight_i * value_i) / sum(weight_i)
combined_variance = 1 / sum(weight_i)
```

This is the optimal weighting when measurements are independent and normally distributed. Key properties:

- The combined value is closer to measurements with lower uncertainty
- The combined uncertainty is always less than either input uncertainty
- Confidence increases because we have more evidence

### Multiple Sources

For more than two measurements, apply `fuse_measurements` iteratively:

```sio
fn fuse_many(measurements: &[EpistemicValue]) -> EpistemicValue {
    if measurements.len() == 0 {
        return epistemic_exact(0.0, 0.0)
    }
    if measurements.len() == 1 {
        return measurements[0]
    }

    var result = measurements[0]
    for i in 1..measurements.len() {
        result = fuse_measurements(result, measurements[i])
    }
    return result
}
```

---

## Checking if Value is Significantly Above Threshold

### Problem

You want to determine if a measured value is significantly above a threshold, accounting for uncertainty.

### Solution

```sio
use epistemic::core::*

// Measurement with uncertainty
let concentration = epistemic_std(52.0, 3.0, 0.95)

// Threshold
let threshold = 50.0

// Method 1: Check if lower confidence bound exceeds threshold
fn is_significantly_above(value: EpistemicValue, threshold: f64) -> bool {
    let lower_bound = get_interval_lo(value)
    return lower_bound > threshold
}

// Method 2: Use probability (requires Knowledge<f64>)
fn probability_above(value: Knowledge<f64>, threshold: f64) -> f64 {
    return value.prob_gt(threshold)
}

// Decision with 95% confidence
let result = is_significantly_above(concentration, threshold)
// Returns false if 52.0 - 2*3.0 = 46.0 < 50.0
```

### Discussion

Two approaches are available:

1. **Interval method**: Check if the lower bound of the uncertainty interval exceeds the threshold. This is conservative: you only say "above" when you're certain.

2. **Probability method**: Calculate P(X > threshold) using the normal approximation. This gives a continuous measure of confidence.

The choice depends on your application:
- Use interval method for safety-critical decisions (false positives are costly)
- Use probability method for risk assessment and continuous optimization

### Probability Calculation Details

For `Knowledge<f64>`, the probability is computed using the normal approximation:

```sio
impl Knowledge<f64> {
    pub fn prob_gt(self: &Knowledge<f64>, threshold: f64) -> f64 {
        if self.variance <= 0.0000001 {
            if self.value > threshold { return 1.0 }
            return 0.0
        }
        let z = (self.value - threshold) / self.std_dev()
        // Phi(z) using error function approximation
        0.5 * (1.0 + erf_approx(z / 1.4142135623730951))
    }
}
```

---

## Propagating Uncertainty Through Custom Functions

### Problem

You have a custom function f(x) and want to propagate uncertainty through it.

### Solution

Use the delta method for smooth functions:

```sio
use epistemic::propagate::*

// For built-in functions, use the propagate module
let x = Knowledge::measured(10.0, 1.0, "sensor")

let y_exp = exp(x)      // Var(e^x) = e^(2x) * Var(x)
let y_log = ln(x)       // Var(ln x) = Var(x) / x^2
let y_sqrt = sqrt(x)    // Var(sqrt(x)) = Var(x) / (4x)
let y_square = square(x) // Var(x^2) = 4x^2 * Var(x)
```

For custom functions, compute the derivative manually:

```sio
// Custom function: f(x) = x^3 - 2x + 1
// Derivative: f'(x) = 3x^2 - 2

fn custom_with_uncertainty(x: Knowledge<f64>) -> Knowledge<f64> {
    let value = x.value * x.value * x.value - 2.0 * x.value + 1.0

    // Compute derivative at x
    let derivative = 3.0 * x.value * x.value - 2.0

    // Delta method: Var(f(x)) = (f'(x))^2 * Var(x)
    let variance = derivative * derivative * x.variance

    Knowledge {
        value: value,
        variance: variance,
        confidence: x.confidence.decay(0.95),
        provenance: x.provenance.with_step("custom_function"),
    }
}
```

### Discussion

The delta method (first-order Taylor expansion) approximates:

```
Var(f(X)) = (df/dx)^2 * Var(X)
```

This is accurate when:
- The function is smooth (differentiable)
- The uncertainty is small relative to the scale of variation
- The function is approximately linear over the uncertainty range

For highly nonlinear functions or large uncertainties, use Monte Carlo propagation instead.

---

## Handling Correlated Uncertainties

### Problem

Two measurements share a common source of error (e.g., same instrument calibration) and their uncertainties are correlated.

### Solution

Use the correlation module to track shared sources:

```sio
use epistemic::correlation::*

// Create a shared source ID for correlated measurements
let calibration_source: i64 = 42

// Both measurements affected by calibration uncertainty
let mass_1 = correlated_from_source(100.0, calibration_source, 2.0, 0.95)
let mass_2 = correlated_from_source(150.0, calibration_source, 2.0, 0.95)

// When we add correlated values, uncertainty INCREASES
let sum = add_correlated(mass_1, mass_2)
// sum.total_u > sqrt(2^2 + 2^2) because of positive correlation

// When we subtract correlated values, uncertainty DECREASES
let diff = sub_correlated(mass_1, mass_2)
// diff.total_u < sqrt(2^2 + 2^2) because errors cancel

// Check correlation coefficient
let r = correlation_coefficient(mass_1, mass_2)  // r = 1.0 (perfect correlation)
```

### Discussion

GUM Equation 14 accounts for correlation:

```
u^2(y) = u^2(a) + u^2(b) + 2 * u(a,b)
```

where u(a,b) is the covariance. For correlated sources:

- **Addition**: Errors compound, uncertainty increases
- **Subtraction**: Errors cancel, uncertainty decreases

This is critical for:
- Measuring differences with the same instrument (calibration errors cancel)
- Ratio measurements (many systematic errors cancel)
- Any calculation involving repeated use of the same measured quantity

### Independent vs Correlated

For independent measurements:

```sio
let a = correlated_independent(100.0, 2.0, 0.95)
let b = correlated_independent(150.0, 3.0, 0.90)

let r = correlation_coefficient(a, b)  // r = 0.0
```

---

## Monte Carlo Uncertainty for Complex Calculations

### Problem

You have a complex function where the delta method is inadequate (nonlinear, non-differentiable, or large uncertainties).

### Solution

Use Monte Carlo propagation:

```sio
use epistemic::montecarlo::*
use epistemic::propagate::monte_carlo

// Define input distribution
let x = mc_input_normal(10.0, 2.0)  // mean=10, std=2

// Propagate through exponential (nonlinear)
let result = mc_exp(x, 10000, 12345)  // 10000 samples, seed=12345

// Result includes:
// - result.mean: Monte Carlo estimate of E[exp(X)]
// - result.std: Monte Carlo estimate of sqrt(Var[exp(X)])
// - result.gum_std: GUM first-order estimate for comparison
// - result.is_nonlinear: true if MC differs significantly from GUM
```

For bivariate functions:

```sio
let a = mc_input_normal(10.0, 1.0)
let b = mc_input_normal(5.0, 0.5)

// Multiplication
let product = mc_mul(a, b, 10000, 54321)

// Division (handles near-zero safely)
let ratio = mc_div(a, b, 10000, 98765)
```

### Discussion

Monte Carlo propagation:

1. Samples N values from the input distribution(s)
2. Evaluates the function at each sample
3. Computes mean and variance of the outputs

Advantages:
- Works for any function (no derivatives needed)
- Handles non-normal distributions
- Automatically captures nonlinear effects
- Jensen's inequality: E[f(X)] > f(E[X]) for convex f

Disadvantages:
- Computationally expensive
- Requires careful choice of N
- Random seed affects results

### When to Use Monte Carlo

Use the `should_use_mc` function to decide:

```sio
fn should_use_mc(input: MCInput) -> bool {
    if abs_f64(input.mean) < 1.0e-15 { return true }
    let rel_u = input.std / abs_f64(input.mean)
    // If relative uncertainty > 30%, linearization may be poor
    return rel_u > 0.3
}
```

---

## Computing Confidence Intervals

### Problem

You want to compute a 95% confidence interval for an epistemic value.

### Solution

```sio
use epistemic::knowledge::*
use epistemic::gum::*

// Method 1: Using Knowledge<f64>
let measurement: Knowledge<f64> = Knowledge::measured(100.0, 25.0, "lab")

let (lower, upper) = measurement.ci95()
// Uses normal approximation: value +/- 1.96 * std

// Method 2: Using GUM with degrees of freedom
let result = gum_type_a(100.0, 5.0, 10)  // value=100, s=5, n=10
// Accounts for t-distribution with n-1 = 9 degrees of freedom

let (lo, hi) = gum_interval_95(result)
// Uses proper coverage factor k from t-distribution
```

### Discussion

For large sample sizes (n > 30), the normal approximation is adequate:
- 95% CI: value +/- 1.96 * standard_uncertainty

For small sample sizes, use the t-distribution:
- 95% CI: value +/- k * standard_uncertainty
- where k depends on degrees of freedom

The GUM module provides proper coverage factors:

```sio
let k = coverage_factor_95(9.0)  // k = 2.262 for v=9
```

---

## Combining Confidence from Multiple Sources

### Problem

You have multiple evidence sources with different confidence levels and want to combine them.

### Solution

```sio
use epistemic::combine::*

// Create confidence representations
let source_1 = beta_confidence_strong(0.9)  // 90% confidence, strong
let source_2 = beta_confidence_strong(0.8)  // 80% confidence, strong

// Different combination rules
let c_mult = combine_confidence(source_1, source_2, rule_multiplicative())
// Independent errors: 0.9 * 0.8 = 0.72

let c_quad = combine_confidence(source_1, source_2, rule_quadrature())
// Correlated errors: uses quadrature of uncertainties

let c_min = combine_confidence(source_1, source_2, rule_minimum())
// Conservative: min(0.9, 0.8) = 0.8

let c_ds = combine_confidence(source_1, source_2, rule_dempster())
// Dempster-Shafer: accounts for conflict
```

### Discussion

Choice of combination rule depends on the relationship between sources:

| Rule | Use When | Formula |
|------|----------|---------|
| Multiplicative | Independent errors | conf1 * conf2 |
| Quadrature | Correlated errors | 1 - sqrt((1-c1)^2 + (1-c2)^2) |
| Minimum | Very conservative | min(conf1, conf2) |
| Dempster-Shafer | Conflicting evidence | Normalized product with conflict |

The Dempster-Shafer rule is appropriate when sources might conflict. High conflict (K near 1) indicates inconsistent evidence.

---

## See Also

- [Epistemic Types Documentation](../epistemic/core.md)
- [GUM Compliance](../epistemic/gum.md)
- [PK Recipes](pk-recipes.md) for pharmacokinetic applications
