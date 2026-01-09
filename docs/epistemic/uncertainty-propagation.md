---
title: Uncertainty Propagation
description: GUM-compliant automatic uncertainty propagation through arithmetic and mathematical operations
prerequisites: [docs/epistemic/knowledge-type.md]
reading_time: 20 minutes
---

# Uncertainty Propagation

When you perform arithmetic on `Knowledge<T>` values, Sounio automatically propagates uncertainty through the computation following the GUM (Guide to the Expression of Uncertainty in Measurement) standard.

**You write the physics. The compiler handles the statistics.**

## Why Automatic Propagation Matters

Manual uncertainty propagation is:
- **Tedious** - Every operation requires explicit calculation
- **Error-prone** - Easy to forget correlations or use wrong formulas
- **Incomplete** - Often omitted under time pressure

The result? Downstream analyses treat uncertain values as exact. Conclusions are drawn that the original uncertainty would have precluded.

Sounio makes uncertainty *infectious*. You cannot accidentally drop it. The type system ensures uncertainty flows through every computation.

## GUM Compliance

Sounio implements the **JCGM 100:2008** (commonly known as GUM) standard for uncertainty propagation. This international standard defines how to:

1. Evaluate Type A uncertainty (statistical analysis)
2. Evaluate Type B uncertainty (other methods)
3. Combine uncertainty components
4. Calculate expanded uncertainty with coverage factors
5. Report results with appropriate significant figures

### Type A vs Type B Uncertainty

| Type | Source | Degrees of Freedom |
|------|--------|-------------------|
| Type A | Statistical analysis of repeated observations | n - 1 |
| Type B | Prior knowledge, calibration certificates, specifications | Effectively infinite |

```sio
use epistemic::gum::{type_a_uncertainty, type_b_uncertainty, type_b_uniform}

// Type A: from repeated measurements
let temp_a = type_a_uncertainty(
    std_dev: 0.3,  // Sample standard deviation
    n: 10          // Number of observations
)
// Standard uncertainty = 0.3 / sqrt(10) = 0.095
// Degrees of freedom = 9

// Type B: from calibration certificate
let temp_b = type_b_uncertainty(0.1)  // Stated as 0.1 degrees
// Degrees of freedom = infinity (1e9)

// Type B: from uniform distribution (e.g., digitization error)
let resolution = type_b_uniform(0.05)  // +/- 0.05 resolution
// Standard uncertainty = 0.05 / sqrt(3) = 0.029
```

## Propagation Rules

### The Delta Method (Law of Propagation of Uncertainty)

For a function y = f(x1, x2, ..., xn), the variance of y is:

```
Var(y) = sum_i (df/dxi)^2 * Var(xi) + 2 * sum_{i<j} (df/dxi)(df/dxj) * Cov(xi, xj)
```

For independent inputs (zero covariance), this simplifies to:

```
Var(y) = sum_i (df/dxi)^2 * Var(xi)
```

### Basic Arithmetic Operations

| Operation | Formula | Propagation Rule |
|-----------|---------|------------------|
| y = a + b | y = a + b | Var(y) = Var(a) + Var(b) |
| y = a - b | y = a - b | Var(y) = Var(a) + Var(b) |
| y = a * b | y = a * b | Var(y) = b^2*Var(a) + a^2*Var(b) |
| y = a / b | y = a / b | Var(y) = (1/b)^2*Var(a) + (a/b^2)^2*Var(b) |

**Note:** For subtraction, variances still ADD. This often surprises newcomers - but it makes sense: both inputs contribute uncertainty to the result.

### Code Examples

```sio
let a = Knowledge::measured(100.0, variance: 4.0, instrument: "A")  // std = 2
let b = Knowledge::measured(50.0, variance: 1.0, instrument: "B")   // std = 1

// Addition: Var(a+b) = Var(a) + Var(b)
let sum = a + b
// sum.value = 150.0
// sum.variance = 4.0 + 1.0 = 5.0
// sum.std() = 2.236

// Subtraction: Var(a-b) = Var(a) + Var(b)  (NOT subtraction!)
let diff = a - b
// diff.value = 50.0
// diff.variance = 4.0 + 1.0 = 5.0
// diff.std() = 2.236

// Multiplication: Var(ab) = b^2*Var(a) + a^2*Var(b)
let prod = a * b
// prod.value = 5000.0
// prod.variance = 50^2 * 4.0 + 100^2 * 1.0 = 10000 + 10000 = 20000
// prod.std() = 141.4

// Division: Var(a/b) = (1/b)^2*Var(a) + (a/b^2)^2*Var(b)
let quot = a / b
// quot.value = 2.0
// quot.variance = (1/50)^2 * 4.0 + (100/2500)^2 * 1.0
//              = 0.0016 + 0.0016 = 0.0032
// quot.std() = 0.057
```

### Relative Uncertainty Form

For multiplication and division, the relative uncertainty form is often more intuitive:

```
(sigma_y / y)^2 = (sigma_a / a)^2 + (sigma_b / b)^2
```

This means relative uncertainties add in quadrature.

```sio
// Relative uncertainty in multiplication/division
let mass = Knowledge::measured(100.0, variance: 4.0, instrument: "balance")
let volume = Knowledge::measured(50.0, variance: 0.25, instrument: "pipette")

// Relative uncertainties
let rel_mass = mass.std() / mass.value()      // 2/100 = 0.02 (2%)
let rel_volume = volume.std() / volume.value() // 0.5/50 = 0.01 (1%)

let density = mass / volume
// density.value = 2.0
// Relative uncertainty = sqrt(0.02^2 + 0.01^2) = sqrt(0.0005) = 0.0224 (2.24%)
// density.std() = 2.0 * 0.0224 = 0.045
```

## Mathematical Functions

### Univariate Functions

For y = f(x), the propagation rule is:

```
Var(y) = (df/dx)^2 * Var(x)
```

| Function | Derivative | Propagation Rule |
|----------|------------|------------------|
| y = sqrt(x) | 1/(2*sqrt(x)) | Var(y) = Var(x) / (4*x) |
| y = x^2 | 2*x | Var(y) = 4*x^2 * Var(x) |
| y = x^n | n*x^(n-1) | Var(y) = n^2 * x^(2n-2) * Var(x) |
| y = exp(x) | exp(x) | Var(y) = exp(2*x) * Var(x) |
| y = ln(x) | 1/x | Var(y) = Var(x) / x^2 |
| y = sin(x) | cos(x) | Var(y) = cos^2(x) * Var(x) |
| y = cos(x) | -sin(x) | Var(y) = sin^2(x) * Var(x) |
| y = 1/x | -1/x^2 | Var(y) = Var(x) / x^4 |

### Code Examples

```sio
let x = Knowledge::measured(4.0, variance: 0.16, instrument: "sensor")

// Square root: Var(sqrt(x)) = Var(x) / (4*x)
let sqrt_x = x.sqrt()
// sqrt_x.value = 2.0
// sqrt_x.variance = 0.16 / 16 = 0.01
// sqrt_x.std() = 0.1

// Square: Var(x^2) = 4*x^2*Var(x)
let x_squared = x.square()
// x_squared.value = 16.0
// x_squared.variance = 4 * 16 * 0.16 = 10.24
// x_squared.std() = 3.2

// Exponential: Var(e^x) = e^(2x) * Var(x)
let exp_x = x.exp()
// exp_x.value = e^4 = 54.6
// exp_x.variance = e^8 * 0.16 = 476.6
// exp_x.std() = 21.8

// Natural log: Var(ln(x)) = Var(x) / x^2
let ln_x = x.ln()
// ln_x.value = ln(4) = 1.386
// ln_x.variance = 0.16 / 16 = 0.01
// ln_x.std() = 0.1
```

### Using the Propagation Module

For explicit propagation (when you need more control):

```sio
use epistemic::propagate

let x = Knowledge::measured(2.0, variance: 0.01, instrument: "sensor")

// Explicit propagation functions
let exp_x = propagate::exp(x)     // e^x
let ln_x = propagate::ln(x)       // ln(x)
let sqrt_x = propagate::sqrt(x)   // sqrt(x)
let sin_x = propagate::sin(x)     // sin(x)
let cos_x = propagate::cos(x)     // cos(x)
let tan_x = propagate::tan(x)     // tan(x)
let inv_x = propagate::inverse(x) // 1/x

// Power function
let x_cubed = propagate::pow(x, 3.0)

// Linear combination: a*x + b*y
let y = Knowledge::measured(3.0, variance: 0.04, instrument: "sensor")
let combo = propagate::linear_combo(2.0, x, 3.0, y)
// Var(2x + 3y) = 4*Var(x) + 9*Var(y)
```

## Correlation Handling

When inputs are correlated, the covariance term cannot be ignored:

```
Var(a + b) = Var(a) + Var(b) + 2*Cov(a, b)
           = Var(a) + Var(b) + 2*rho*sigma_a*sigma_b
```

where rho is the correlation coefficient (-1 to +1).

### Correlated Propagation

```sio
use epistemic::propagate::{sum_correlated, product_correlated}

let x = Knowledge::measured(10.0, variance: 1.0, instrument: "sensor")
let y = Knowledge::measured(5.0, variance: 0.25, instrument: "sensor")

// If x and y are from the same instrument, they may be correlated
let correlation = 0.8  // Strong positive correlation

// Correlated sum
let sum = sum_correlated(x, y, correlation)
// Var = Var(x) + Var(y) + 2*rho*sigma_x*sigma_y
//     = 1.0 + 0.25 + 2*0.8*1.0*0.5 = 2.05

// Correlated product
let prod = product_correlated(x, y, correlation)
// Var = y^2*Var(x) + x^2*Var(y) + 2*x*y*rho*sigma_x*sigma_y
```

### Special Cases

**Perfect positive correlation (rho = 1):**
```sio
// When a = b (same measurement used twice)
let a = Knowledge::measured(10.0, variance: 1.0, instrument: "sensor")

// WRONG: treating a - a as difference of independent values
let wrong = a - a  // Would give Var = 2, but should be 0!

// RIGHT: acknowledge perfect correlation
let correct = sum_correlated(a, -a, correlation: 1.0)  // Var = 0
```

**Perfect negative correlation (rho = -1):**
```sio
// Variables that move opposite ways
let sum = sum_correlated(x, y, correlation: -1.0)
// Variance can be less than either input!
```

## Monte Carlo Propagation

When the delta method is insufficient (non-linear functions, non-Gaussian distributions), Monte Carlo provides a numerical estimate.

```sio
use epistemic::propagate::monte_carlo

let x = Knowledge::measured(2.0, variance: 0.1, instrument: "sensor")

// Monte Carlo for complex function
let result = monte_carlo(
    x,
    |v| complex_nonlinear_function(v),
    n_samples: 10000
)
// Samples from N(x.value, x.variance), applies function, computes statistics
```

### When to Use Monte Carlo

Use Monte Carlo when:
- The function is highly non-linear
- The function is not differentiable
- The input distribution is non-Gaussian
- You need the full output distribution, not just variance

```sio
use epistemic::propagate::{monte_carlo, monte_carlo_2d}

// Single input
let x = Knowledge::measured(1.0, variance: 0.04, instrument: "sensor")
let y = monte_carlo(x, |v| if v > 0.0 { v.ln() } else { 0.0 }, 10000)

// Two inputs (bivariate)
let a = Knowledge::measured(2.0, variance: 0.1, instrument: "A")
let b = Knowledge::measured(3.0, variance: 0.2, instrument: "B")

let z = monte_carlo_2d(a, b, |x, y| x.pow(y), n_samples: 10000)
```

## Coverage Factors and Expanded Uncertainty

The GUM defines **expanded uncertainty** U = k * u, where:
- u is the standard uncertainty
- k is the coverage factor based on desired confidence level

| Coverage Probability | Normal (infinite dof) | t-distribution (dof=5) |
|---------------------|----------------------|------------------------|
| 68% (1-sigma) | k = 1.00 | k = 1.00 |
| 90% | k = 1.645 | k = 2.02 |
| 95% | k = 1.96 | k = 2.57 |
| 99% | k = 2.576 | k = 4.03 |
| 99.73% (3-sigma) | k = 3.00 | k = 4.77 |

### Welch-Satterthwaite Approximation

When combining Type A and Type B uncertainties, the effective degrees of freedom determine the appropriate coverage factor:

```sio
use epistemic::gum::{
    gum_add, gum_mul, gum_div,
    coverage_factor_95, welch_satterthwaite_2
}

// Create GUM results
let mass = gum_type_a(75.0, std_dev: 0.5, n: 10)   // Type A
let volume = gum_simple(50.0, std_u: 0.2)          // Type B

// Combined result
let density = gum_div(mass, volume)

// Effective degrees of freedom via Welch-Satterthwaite
let dof_eff = density.degrees_of_freedom
// Coverage factor depends on dof_eff
let k95 = coverage_factor_95(dof_eff)

// Expanded uncertainty at 95% confidence
let U95 = density.expanded_uncertainty_95
// U95 = k95 * density.std_uncertainty
```

### GUM Result Type

```sio
pub struct GUMResult {
    value: f64,                    // Best estimate
    std_uncertainty: f64,          // Combined standard uncertainty u_c
    degrees_of_freedom: f64,       // Effective degrees of freedom
    coverage_factor_95: f64,       // k for 95% expanded uncertainty
    expanded_uncertainty_95: f64,  // U = k * u_c
}

// Get 95% confidence interval
let (lo, hi) = gum_interval_95(density)
// lo = value - U95
// hi = value + U95

// Relative uncertainty
let rel_u = relative_uncertainty_percent(density)
// 100 * std_uncertainty / |value|
```

## Complete Example

A pharmacokinetic calculation with full uncertainty propagation:

```sio
use epistemic::{Knowledge, BetaConfidence}
use epistemic::gum::{gum_type_a, gum_simple, gum_div, gum_mul}
use units::{mg, L, h, L_per_h}

fn calculate_clearance() -> Knowledge<L_per_h> {
    // Dose administered (from analytical balance, Type A)
    let dose = Knowledge::measured(
        500.0,             // mg
        variance: 625.0,   // std = 25 mg from 10 weighings
        instrument: "mettler_toledo_xpe205"
    )

    // AUC from population PK (Type B, from model)
    let auc = Knowledge::new(
        value: 100.0,      // mg*h/L
        variance: 225.0,   // std = 15 from inter-individual variability
        confidence: BetaConfidence::from_rate(0.90, 50.0),
        source: Source::Computed { operation: "population_pk_model" }
    )

    // Bioavailability (Type B, from literature)
    let bioavailability = Knowledge::new(
        value: 0.85,
        variance: 0.0025,  // std = 0.05
        confidence: BetaConfidence::from_rate(0.95, 200.0),
        source: Source::External {
            source: "FDA_NDA_Review",
            url: "https://www.accessdata.fda.gov/..."
        }
    )

    // Calculate clearance: CL = F * Dose / AUC
    let numerator = dose * bioavailability  // Variance propagates
    let clearance = numerator / auc         // Variance propagates again

    // Report
    let (lo, hi) = clearance.ci95()
    println("Clearance: " + clearance.value().to_string() + " L/h")
    println("95% CI: [" + lo.to_string() + ", " + hi.to_string() + "]")
    println("Relative uncertainty: " +
        (100.0 * clearance.std() / clearance.value()).to_string() + "%")
    println("Provenance: " + clearance.prov().to_string())

    return clearance
}
```

## Best Practices

### 1. Start with Proper Uncertainty Assessment

```sio
// BAD: guessing uncertainty
let mass = Knowledge::measured(75.0, variance: 1.0, instrument: "scale")

// GOOD: use actual calibration data
let mass = Knowledge::new(
    value: 75.0,
    variance: 0.25,  // From balance specification: 0.5g resolution
    confidence: BetaConfidence::from_rate(0.99, 1000.0),  // 1000 checks
    source: Source::Measurement {
        instrument: "Mettler_AE200_SN12345",
        timestamp: calibration_date
    }
)
```

### 2. Check Final Uncertainty

```sio
let result = complex_calculation(a, b, c, d)

// Is the uncertainty acceptable for our purpose?
if result.std() / result.value() > 0.10 {
    println("WARNING: Relative uncertainty exceeds 10%")
    println("Consider:")
    println("  - More precise instruments")
    println("  - More measurements")
    println("  - Reducing intermediate calculations")
}
```

### 3. Document Assumptions

```sio
// When assuming independence
let combined = a + b  // ASSUMES: a and b are independent

// When correlation is known
let combined = sum_correlated(a, b, correlation: 0.0)  // Explicitly independent
```

### 4. Use Monte Carlo for Complex Cases

```sio
// Non-linear function where delta method may be inaccurate
let result = monte_carlo(x, |v| sigmoid(v), n_samples: 10000)

// Compare with delta method
let result_delta = propagate::sigmoid(x)

// If they differ significantly, trust Monte Carlo
```

## See Also

- [Knowledge Type](knowledge-type.md) - The core `Knowledge<T>` structure
- [Confidence Gates](confidence-gates.md) - Control flow based on confidence
- [stdlib/epistemic/propagate.sio](/stdlib/epistemic/propagate.sio) - Propagation implementation
- [stdlib/epistemic/gum.sio](/stdlib/epistemic/gum.sio) - GUM compliance module
- [JCGM 100:2008 (GUM)](https://www.bipm.org/en/publications/guides/gum.html) - ISO uncertainty standard
- Taylor, J.R. "An Introduction to Error Analysis" - Classic textbook on uncertainty
