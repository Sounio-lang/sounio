# Uncertainty Models in Sounio

Sounio provides multiple mathematical frameworks for representing and propagating uncertainty. Different problems require different models -- a simple measurement might need only a standard deviation, while a complex simulation with correlated errors requires affine arithmetic or Monte Carlo methods.

This document covers the available uncertainty models, when to use each, and how Sounio's KEC (Knowledge-Entropy-Complexity) system automatically selects the optimal model.

## Overview of Uncertainty Models

| Model | Representation | Best For | Computational Cost |
|-------|---------------|----------|-------------------|
| **Interval** | `[min, max]` | Bounded uncertainty, worst-case analysis | Low |
| **Gaussian** | `mu +/- sigma` | Measurement uncertainty, central limit | Low |
| **Beta** | `Beta(alpha, beta)` | Proportions, bounded [0,1] values | Low |
| **Affine** | `x0 + sum(xi*ei)` | Correlated uncertainties | Medium |
| **Monte Carlo** | N samples | Complex nonlinear propagation | High |
| **Sequential MC** | Weighted particles | Dynamic systems, state estimation | Very High |

## Interval Arithmetic

Interval arithmetic represents uncertainty as bounded ranges `[a, b]`, guaranteeing the true value lies within the computed interval.

### IEEE 1788 Compliance

Sounio's interval arithmetic follows IEEE 1788-2015 standard:

```sio
// Create intervals
let x = interval_new(1.0, 2.0)       // [1.0, 2.0]
let y = interval_point(5.0)          // [5.0, 5.0]
let z = interval_symmetric(10.0, 0.5) // [9.5, 10.5]

// Special intervals
let empty = interval_empty()          // {} (impossible)
let entire = interval_entire()        // [-inf, +inf] (unknown)
```

### Basic Operations

```sio
// Addition: [a,b] + [c,d] = [a+c, b+d]
let sum = interval_add(x, y)  // [6.0, 7.0]

// Subtraction: [a,b] - [c,d] = [a-d, b-c]
let diff = interval_sub(y, x)  // [3.0, 4.0]

// Multiplication: finds min/max of all four products
let prod = interval_mul(x, y)  // [5.0, 10.0]

// Division with zero handling
let div_result = interval_div(y, x)
if div_result.exception == 0 {
    // Normal division
    let quotient = div_result.result  // [2.5, 5.0]
} else if div_result.exception == 1 {
    // Division by [0,0] -> empty
} else if div_result.exception == 2 {
    // Division by interval spanning zero -> entire
}
```

### Decorated Intervals

Decorations track validity through computations (IEEE 1788 Section 9):

```sio
// Decoration levels:
// com (4): Common - bounded, continuous
// dac (3): Defined and continuous
// def (2): Defined but possibly unbounded
// trv (1): Trivial - could be any value
// ill (0): Invalid/undefined

let decorated = decorated_common(1.0, 2.0)  // [1,2] with com decoration

// Square root preserves decoration for positive intervals
let sqrt_result = interval_sqrt(decorated.interval)
// sqrt_result.decoration = com if interval >= 0
// sqrt_result.decoration = def if interval spans 0 (clamped)
// sqrt_result.decoration = ill if interval < 0 (empty result)
```

### When to Use Interval Arithmetic

**Ideal for:**
- Safety-critical computations requiring guaranteed bounds
- Worst-case analysis
- Bounded physical constraints (e.g., concentrations must be positive)
- Sensitivity analysis

**Limitations:**
- **Dependency problem**: Repeated use of the same variable leads to overestimation
  ```sio
  let x = interval_new(0.0, 1.0)
  let y = interval_sub(x, x)  // Gives [-1, 1], not [0, 0]!
  ```
- **Wrapping effect**: Intervals grow with each operation
- Does not track correlations

## Gaussian (Normal) Distribution

The most common model for measurement uncertainty, assuming errors are normally distributed around a central value.

### Standard Uncertainty Representation

```sio
// Create epistemic value with standard uncertainty
let mass = epistemic_std(75.0, 0.5, 0.95)
// value: 75.0, uncertainty: 0.5, confidence: 0.95

// Access components
let value = get_value(mass)              // 75.0
let std_u = get_std_uncertainty(mass)    // 0.5
let conf = get_confidence(mass)          // 0.95
```

### GUM-Compliant Propagation

Per JCGM 100:2008 (GUM), uncertainties propagate through operations:

```sio
// Addition/Subtraction: variances add
// u(y)^2 = u(a)^2 + u(b)^2
let a = epistemic_std(100.0, 2.0, 0.95)
let b = epistemic_std(50.0, 1.0, 0.95)
let sum = add_epistemic(a, b)
// sum.uncertainty = sqrt(4 + 1) = 2.24

// Multiplication/Division: relative uncertainties add in quadrature
// (u(y)/y)^2 = (u(a)/a)^2 + (u(b)/b)^2
let product = mul_epistemic(a, b)
// rel_a = 2/100 = 0.02, rel_b = 1/50 = 0.02
// rel_product = sqrt(0.02^2 + 0.02^2) = 0.0283
// product.uncertainty = 5000 * 0.0283 = 141.4
```

### Coverage Factors and Expanded Uncertainty

The coverage factor `k` relates standard uncertainty to expanded uncertainty at a given confidence level:

```sio
// Standard coverage factors for normal distribution
fn k_normal_68() -> f64 { 1.0 }    // 68% confidence
fn k_normal_90() -> f64 { 1.645 }  // 90% confidence
fn k_normal_95() -> f64 { 1.96 }   // 95% confidence (common default)
fn k_normal_99() -> f64 { 2.576 }  // 99% confidence
fn k_normal_9973() -> f64 { 3.0 }  // 99.73% (3-sigma)

// For finite degrees of freedom, use t-distribution
let k_95 = coverage_factor_95(dof: 10.0)  // 2.228 (t-distribution)
```

### Type A vs Type B Uncertainty

```sio
// Type A: Statistical analysis of repeated measurements
let type_a = type_a_uncertainty(std_dev: 1.5, n: 10)
// u = s/sqrt(n) = 1.5/sqrt(10) = 0.474
// dof = n - 1 = 9

// Type B: A priori knowledge (specs, certificates, judgment)
let type_b = type_b_uncertainty(std_u: 0.5)
// dof = infinity (1e9)

// Type B from uniform distribution (e.g., resolution)
let resolution = type_b_uniform(half_width: 0.05)
// u = 0.05 / sqrt(3) = 0.029

// Type B from triangular distribution
let triangular = type_b_triangular(half_width: 0.05)
// u = 0.05 / sqrt(6) = 0.020
```

### Welch-Satterthwaite Equation

When combining uncertainties with different degrees of freedom:

```sio
// Effective degrees of freedom for combined uncertainty
let dof_eff = welch_satterthwaite_2(u1, u2)
// nu_eff = u_c^4 / (u1^4/nu1 + u2^4/nu2)

// Use effective DoF for coverage factor
let k = coverage_factor_95(dof_eff)
```

### When to Use Gaussian Model

**Ideal for:**
- Measurement data following normal distribution
- Large sample sizes (central limit theorem)
- GUM-compliant uncertainty budgets
- Laboratory measurements with calibrated instruments

**Limitations:**
- Assumes symmetric errors
- May underestimate uncertainty for skewed distributions
- First-order approximation fails for highly nonlinear functions

## Beta Distribution

For proportions and values bounded to [0, 1], the Beta distribution is the natural choice.

### Creating Beta-Distributed Values

```sio
// Beta(alpha, beta) represents a proportion
// Mean = alpha / (alpha + beta)
// Variance depends on both parameters

// From success/failure counts (conjugate prior for binomial)
let proportion = Knowledge::beta(successes: 80, failures: 20)
// Represents observed proportion of 80%

// From mean and sample size equivalent
let efficacy = Knowledge::beta_from_mean(mean: 0.85, sample_size: 100)
```

### Properties

```sio
// Mean of Beta(a, b)
fn beta_mean(alpha: f64, beta: f64) -> f64 {
    alpha / (alpha + beta)
}

// Mode (for alpha, beta > 1)
fn beta_mode(alpha: f64, beta: f64) -> f64 {
    (alpha - 1.0) / (alpha + beta - 2.0)
}

// Variance
fn beta_variance(alpha: f64, beta: f64) -> f64 {
    let sum = alpha + beta
    alpha * beta / (sum * sum * (sum + 1.0))
}
```

### When to Use Beta Distribution

**Ideal for:**
- Probabilities and proportions
- Clinical trial response rates
- Quality metrics (defect rates)
- Bayesian updating with binomial data

## Affine Arithmetic

Affine arithmetic tracks correlations between uncertainties, reducing the overestimation problem of standard interval arithmetic.

### Representation

Values are represented as:
```
x = x0 + x1*e1 + x2*e2 + ... + xn*en
```

Where:
- `x0` is the central value
- `xi` are noise coefficients
- `ei` are noise symbols in [-1, 1]

```sio
// Affine form representation
struct AffineForm {
    center: f64,
    noise_terms: Vec<(u32, f64)>,  // (symbol_id, coefficient)
}

// Create from interval
let x = affine_from_interval(1.0, 2.0)
// x = 1.5 + 0.5*e1

// Correlated values share noise symbols
let y = affine_from_interval(3.0, 4.0)
// y = 3.5 + 0.5*e2

// Independent uncertainties have different symbols
```

### Operations with Correlation Tracking

```sio
// Addition: noise terms combine for same symbols
let a = UncertainValue::Affine {
    center: 10.0,
    noise_terms: vec![(1, 0.5)]  // 10.0 + 0.5*e1
}
let b = UncertainValue::Affine {
    center: 5.0,
    noise_terms: vec![(1, 0.3)]  // 5.0 + 0.3*e1 (SAME symbol!)
}

// a + b = 15.0 + 0.8*e1 (correlated noise adds)
// Interval would give [14.2, 15.8]
// Without correlation tracking: [14.2, 15.8] (wider!)

// a - b = 5.0 + 0.2*e1 (correlated noise subtracts)
// THIS IS THE KEY BENEFIT: x - x = 0 exactly!
```

### Multiplication and Nonlinearity

```sio
// Multiplication introduces new noise for nonlinear part
// (a + e1)(b + e2) = ab + a*e2 + b*e1 + e1*e2
// The e1*e2 term creates a new independent noise symbol

let product = propagate_affine(&a, &b, BinaryOp::Mul, &config)
// New noise term captures nonlinear contribution
```

### Condensation Strategies

When noise symbols proliferate, Sounio condenses them:

```sio
// Configuration for affine arithmetic
let config = AffineConfig {
    max_noise_symbols: 64,
    condensation: CondensationStrategy::MergeSmallest,
    track_correlations: true,
}

// Condensation strategies:
// MergeSmallest: Combine smallest coefficients
// ToInterval: Convert to interval (loses correlation)
// KeepRecent: Prioritize recent symbols
```

### When to Use Affine Arithmetic

**Ideal for:**
- Correlated uncertainties from shared sources
- Repeated use of variables (avoids dependency problem)
- Moderate nonlinearity
- Numerical analysis with error tracking

**Limitations:**
- Memory grows with noise symbols
- Division and transcendental functions need approximation
- More complex than interval arithmetic

## Monte Carlo Methods

When first-order (GUM) approximation fails, Monte Carlo simulation provides the reference method.

### When GUM Fails

Per JCGM 101:2008, use Monte Carlo when:
1. The measurement function is significantly nonlinear
2. Input distributions are non-Gaussian
3. Combined uncertainty is not well-approximated by normal distribution
4. Degrees of freedom are small

### Basic Monte Carlo Propagation

```sio
// Define input distributions
let a = mc_input_normal(mean: 10.0, std: 1.0)
let b = mc_input_normal(mean: 5.0, std: 0.5)

// Or uniform inputs
let c = mc_input_uniform(lo: 1.0, hi: 2.0)

// Propagate through operations
let result = mc_add(a, b, n: 10000, seed: 12345)

// Result contains:
// result.mean: Sample mean
// result.std: Sample standard deviation
// result.p5, result.p95: 90% confidence interval
// result.gum_std: GUM first-order estimate (for comparison)
// result.is_nonlinear: True if MC differs significantly from GUM
```

### Detecting Nonlinearity

```sio
// Compare MC to GUM estimate
let exp_result = mc_exp(a, n: 10000, seed: 54321)

if exp_result.is_nonlinear {
    // MC std differs from GUM by > 10%
    // Use MC result, not GUM
    print("Nonlinear effects detected: use Monte Carlo")
    print("MC std: {}, GUM std: {}", exp_result.std, exp_result.gum_std)
}
```

### Adaptive Sample Size

```sio
// Check if more samples needed for desired precision
fn should_use_mc(input: MCInput) -> bool {
    if abs_f64(input.mean) < 1.0e-15 { return true }
    let rel_u = input.std / abs_f64(input.mean)
    // If relative uncertainty > 30%, linearization may be poor
    rel_u > 0.3
}

// Policy selection
let mode = if should_use_mc(input) {
    propagation_mc(n: 10000)
} else {
    propagation_gum()
}
```

### When to Use Monte Carlo

**Ideal for:**
- Highly nonlinear functions (exp, log, power)
- Non-Gaussian input distributions
- Complex multi-variate propagation
- Validation of other methods

**Limitations:**
- Computationally expensive
- Requires random number generation
- Results have sampling uncertainty

## Sequential Monte Carlo (SMC)

For dynamic systems where uncertainty evolves over time, SMC (particle filters) provide state estimation.

### State Estimation with Particles

```sio
// Each particle represents a possible state
struct Particle {
    state: f64,
    weight: f64,
}

// Initialize particles from prior
fn smc_init(n_particles: i32, prior_mean: f64, prior_std: f64) -> Vec<Particle> {
    var particles = Vec::new()
    var rng = rng_new(seed)
    for i in 0..n_particles {
        let (state, rng2) = rng_normal_params(rng, prior_mean, prior_std)
        rng = rng2
        particles.push(Particle { state, weight: 1.0 / n_particles as f64 })
    }
    particles
}

// Predict step: propagate particles through dynamics
fn smc_predict(particles: &!Vec<Particle>, dynamics: fn(f64) -> f64, noise_std: f64) {
    for p in particles {
        p.state = dynamics(p.state) + sample_noise(noise_std)
    }
}

// Update step: reweight based on observation
fn smc_update(particles: &!Vec<Particle>, observation: f64, obs_std: f64) {
    var total_weight = 0.0
    for p in particles {
        // Likelihood of observation given particle state
        let likelihood = gaussian_pdf(observation, p.state, obs_std)
        p.weight = p.weight * likelihood
        total_weight = total_weight + p.weight
    }
    // Normalize
    for p in particles {
        p.weight = p.weight / total_weight
    }
}
```

### When to Use SMC

**Ideal for:**
- Time-series state estimation
- Non-Gaussian, nonlinear dynamics
- Multi-modal posterior distributions
- Real-time filtering with streaming data

## KEC Auto-Selection

Sounio's KEC (Knowledge-Entropy-Complexity) system automatically selects the optimal uncertainty model based on:

- **K** (Knowledge): Measurement quality, prior information available
- **E** (Entropy): Information content of uncertainty
- **C** (Complexity): Computational constraints

### Configuration Presets

```sio
// Scientific computing: high accuracy, moderate complexity
let config = KECConfig::scientific()
// max_complexity: 10000, min_confidence: 0.99
// entropy_threshold_distribution: 1.5, cv_threshold_interval: 0.05

// Real-time: fast execution, acceptable approximations
let config = KECConfig::realtime()
// max_complexity: 100, min_confidence: 0.90

// PKPD modeling: balanced for pharmacometrics
let config = KECConfig::pkpd()

// Safety-critical: maximum accuracy, no shortcuts
let config = KECConfig::safety_critical()
// max_complexity: infinity, min_confidence: 0.999
```

### Automatic Selection

```sio
// Analyze uncertainty characteristics
let metrics = UncertaintyMetrics::from_samples(data)
// entropy, cv, skewness, kurtosis, multimodal, correlations

// Analyze computational requirements
let complexity = ComplexityMetrics {
    operation_count: 50,
    graph_depth: 10,
    nonlinearity: 0.3,
    ...
}

// Get recommendation
let selector = KECSelector::new(config)
let result = selector.select(&metrics, &complexity)

print("Recommended: {}", result.recommended)  // e.g., UncertaintyLevel::Affine
print("Confidence: {}", result.confidence)
print("Reasoning:")
for reason in result.reasoning {
    print("  - {}", reason)
}
if !result.warnings.is_empty() {
    print("Warnings:")
    for warning in result.warnings {
        print("  ! {}", warning)
    }
}
```

### Selection Logic

```sio
// The selector scores each level based on:
// 1. Adequacy: How well does this level capture the uncertainty?
// 2. Complexity: Can we afford this level within budget?
// 3. Simplicity: Prefer simpler models when equivalent

fn adequacy_score(level: UncertaintyLevel, uncertainty: &UncertaintyMetrics) -> f64 {
    match level {
        Point => if uncertainty.cv < 0.01 { 1.0 } else { 0.1 },
        Interval => if bounded && symmetric { 0.9 } else { 0.4 },
        Affine => if has_correlations { 0.95 } else { 0.8 },
        Distribution => if low_entropy { 0.9 } else { 0.5 },
        Particles => if multimodal { 1.0 } else { 0.7 },
        ...
    }
}
```

### Operation-Specific Selection

```sio
// Certain operations may require upgrading the model
let level = select_for_operation(
    op: "divide",
    input_levels: &[UncertaintyLevel::Interval],
    config: Some(KECConfig::scientific())
)
// Division with interval may upgrade to Affine to avoid explosion

// ODE solving benefits from affine or particles
let ode_level = select_for_operation(
    op: "solve_ode",
    input_levels: &[UncertaintyLevel::Interval],
    config: Some(KECConfig::scientific())
)
// Returns Affine or Particles based on complexity budget
```

## Conversion Between Models

Sounio allows converting between uncertainty representations:

```sio
// Interval to Gaussian (approximate)
fn interval_to_gaussian(interval: Interval) -> GUMResult {
    let value = (interval.lo + interval.hi) / 2.0
    let half_width = (interval.hi - interval.lo) / 2.0
    // Assume uniform -> std = half_width / sqrt(3)
    let std_u = half_width / sqrt(3.0)
    gum_simple(value, std_u)
}

// Gaussian to Interval (using coverage factor)
fn gaussian_to_interval(gum: GUMResult, coverage: f64) -> Interval {
    let half_width = gum.std_uncertainty * coverage
    interval_new(gum.value - half_width, gum.value + half_width)
}

// Samples to any model
fn samples_to_model(samples: &[f64]) -> UncertaintyMetrics {
    UncertaintyMetrics::from_samples(samples)
    // Can then be used with KEC to select appropriate model
}
```

## Summary: Choosing the Right Model

| Scenario | Recommended Model |
|----------|------------------|
| Simple measurements, normal errors | Gaussian (GUM) |
| Safety-critical, need guaranteed bounds | Interval (IEEE 1788) |
| Repeated variables, correlated errors | Affine |
| Proportions, probabilities | Beta |
| Highly nonlinear, non-Gaussian | Monte Carlo |
| Time-series, dynamic state | Sequential MC |
| Unknown characteristics | Use KEC auto-selection |

## References

- JCGM 100:2008 - Guide to the Expression of Uncertainty in Measurement (GUM)
- JCGM 101:2008 - Supplement 1: Propagation of distributions using a Monte Carlo method
- IEEE 1788-2015 - Standard for Interval Arithmetic
- IEEE 1788.1-2017 - Standard for Interval Arithmetic (Simplified)
- Stolfi & de Figueiredo (2003) - Self-Validated Numerical Methods and Applications (Affine Arithmetic)
- Doucet, de Freitas & Gordon (2001) - Sequential Monte Carlo Methods in Practice
