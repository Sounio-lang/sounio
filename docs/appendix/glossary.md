# Glossary

Key terms and concepts in the Sounio programming language.

## A

### Affine Type
A type whose values can be used *at most once*. Unlike linear types, affine values may be silently dropped. Used for resources where cleanup is optional.

```sio
affine struct Buffer {
    ptr: *u8,
    len: usize,
}
// Buffer can be used once or dropped without use
```

### Algebraic Effect
A structured way to handle side effects in a program. Effects can be declared, performed, and handled. Sounio uses algebraic effects for I/O, mutation, randomness, and more.

```sio
fn read_data() -> string with IO {
    // IO effect declared in signature
}
```

## B

### Beta Confidence
A representation of confidence using the Beta distribution, parameterized by alpha and beta (success and failure counts). Provides a principled way to reason about confidence that accounts for sample size.

```sio
let confidence = BetaConfidence::from_evidence(successes: 8, failures: 2)
let mean = beta_mean(confidence)  // 0.8
```

### Bidirectional Type Inference
A type inference algorithm that propagates type information both from expressions to their contexts (synthesis) and from contexts to expressions (checking). Allows Sounio to infer types with minimal annotations.

## C

### Confidence
A measure of how certain we are about an epistemic value, typically expressed as a probability between 0 and 1. Higher confidence means more certainty about the value's accuracy.

```sio
let measurement = epistemic_std(100.0, 5.0, 0.95)  // 95% confidence
```

### Confidence Gate
A control flow construct that branches based on confidence level. Allows programs to require higher confidence for critical decisions.

```sio
if measurement.confidence > 0.95 {
    proceed(measurement)
} else {
    require_confirmation(measurement)
}
```

### Coverage Factor (k)
A multiplier used to expand standard uncertainty to a confidence interval. For 95% confidence with large sample sizes, k = 1.96. For smaller samples, k depends on degrees of freedom (t-distribution).

### Covariance
A measure of how two random variables change together. In uncertainty propagation, covariance affects how uncertainties combine when values are correlated.

```
u(x,y) = r(x,y) * u(x) * u(y)
```

## D

### Degrees of Freedom
A parameter affecting the coverage factor for confidence intervals. For Type A uncertainty (statistical), degrees of freedom = n - 1 where n is sample size. Type B uncertainty (prior knowledge) has effectively infinite degrees of freedom.

### Delta Method
A technique for propagating variance through functions using first-order Taylor expansion:

```
Var(f(X)) = (df/dx)^2 * Var(X)
```

### Dempster-Shafer Theory
A mathematical theory of evidence that allows explicit representation of uncertainty and ignorance. Used in Sounio's fusion module for combining belief functions.

## E

### Effect
A computational side effect that a function may produce (I/O, mutation, allocation, etc.). Sounio tracks effects in function signatures.

Common effects:
- `IO` - File/network I/O
- `Mut` - State mutation
- `Alloc` - Heap allocation
- `Panic` - May panic
- `Prob` - Probabilistic operations
- `GPU` - GPU kernel execution

### Epistemic
Relating to knowledge or the degree of certainty about facts. In Sounio, epistemic types carry not just values but information about uncertainty, confidence, and provenance.

### Epistemic Value / EpistemicValue
The fundamental type for uncertain values in Sounio, containing:
- Value (central estimate)
- Uncertainty (standard deviation or variance)
- Confidence (how certain we are)
- Optionally: provenance, degrees of freedom

```sio
struct EpistemicValue {
    value: f64,
    variance: f64,
    conf: f64,
}
```

### Exclusive Reference (`&!`)
Sounio's syntax for a mutable/exclusive reference. While held, no other references to the data may exist. Equivalent to Rust's `&mut`.

```sio
fn increment(x: &!i32) {
    *x = *x + 1
}
```

### Expanded Uncertainty (U)
Standard uncertainty multiplied by coverage factor: U = k * u. Represents the half-width of a confidence interval.

## F

### Fixed Effect
In population modeling, parameters that represent typical values in a population (as opposed to random effects which represent individual variation).

### Fusion
The process of combining multiple measurements or evidence sources into a single estimate. Methods include inverse-variance weighting, Dempster-Shafer combination, and Bayesian updating.

## G

### GUM (Guide to the Expression of Uncertainty in Measurement)
International standard (JCGM 100:2008) defining how to evaluate and express measurement uncertainty. Sounio's epistemic module implements GUM-compliant uncertainty propagation.

### GUM Supplement 1
JCGM 101:2008, which describes Monte Carlo methods for uncertainty propagation when GUM's linear approximation is inadequate.

## H

### Handler
A construct for handling algebraic effects. Determines what happens when an effect is performed.

```sio
handle {
    risky_operation()
} with {
    fail(msg) => default_value,
}
```

## I

### Inverse-Variance Weighting
A method for combining measurements where weights are proportional to 1/variance. Measurements with lower uncertainty contribute more to the combined estimate.

```
combined = sum(value_i / var_i) / sum(1 / var_i)
```

## K

### Knowledge<T>
The generic epistemic type in Sounio, parameterized by the underlying value type. Carries value, variance, confidence, and provenance.

```sio
let measurement: Knowledge<f64> = Knowledge::new(
    value: 42.0,
    variance: 0.25,
    confidence: BetaConfidence::strong(0.95, 100.0),
    provenance: Source::Measurement { instrument: "scale" }
)
```

## L

### Linear Type
A type whose values must be used *exactly once*. Ensures resources are properly consumed (not leaked or double-used).

```sio
linear struct FileHandle {
    fd: i32,
}
// FileHandle must be explicitly closed - cannot be dropped silently
```

### Likelihood Ratio
The ratio P(evidence|H) / P(evidence|not H). Used in Bayesian updating to adjust beliefs based on new evidence.

## M

### Monte Carlo Propagation
A numerical method for uncertainty propagation that samples from input distributions, evaluates the function, and computes statistics on outputs. Used when the delta method is inadequate.

## O

### Omega (omega)
In population PK, the standard deviation of random effects (inter-individual variability), often expressed as coefficient of variation.

## P

### Plausibility
In Dempster-Shafer theory, the upper bound on the probability of a hypothesis. Pl(H) = 1 - Bel(not H).

### Provenance
Information about the origin and processing history of a value. Tracks where data came from and what transformations were applied.

```sio
enum Source {
    Measurement { instrument: string, timestamp: i64 },
    Computed { operation: string },
    External { source: string, url: string },
    Assertion { author: string },
}
```

## R

### Random Effect (eta)
In population modeling, parameters representing individual deviation from population typical values. Usually assumed to follow a normal distribution.

### Relative Uncertainty
Standard uncertainty divided by value: u_rel = u / |x|. Often expressed as percentage or coefficient of variation (CV).

## S

### Sensitivity Coefficient
The partial derivative of output with respect to an input: c_i = df/dx_i. Determines how input uncertainty contributes to output uncertainty.

### Standard Uncertainty (u)
A measure of uncertainty expressed as a standard deviation. The fundamental uncertainty measure in GUM.

## T

### Type A Uncertainty
Uncertainty evaluated by statistical analysis of observations. Standard uncertainty of the mean = s / sqrt(n).

### Type B Uncertainty
Uncertainty evaluated by other means (prior knowledge, calibration certificates, manufacturer specs). Assigned based on available information.

## U

### Uncertainty Budget
A table showing all uncertainty components, their values, sensitivity coefficients, and contributions to combined uncertainty.

### Uncertainty Propagation
The process of calculating output uncertainty from input uncertainties when values are combined through mathematical operations.

## V

### Variance
The square of standard uncertainty: variance = u^2. Variances add for independent uncertainties.

### VarID
A unique identifier for a source of uncertainty. Used to track correlations between values that share common uncertainty sources.

## W

### Welch-Satterthwaite
An approximation for the effective degrees of freedom when combining uncertainty components with different degrees of freedom:

```
v_eff = u_c^4 / sum(u_i^4 / v_i)
```

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| u(x) | Standard uncertainty of x |
| U | Expanded uncertainty |
| k | Coverage factor |
| v | Degrees of freedom |
| r(x,y) | Correlation coefficient between x and y |
| u(x,y) | Covariance of x and y |
| c_i | Sensitivity coefficient for input i |
| Bel(H) | Belief in hypothesis H (Dempster-Shafer) |
| Pl(H) | Plausibility of hypothesis H |
| K | Conflict factor in Dempster-Shafer combination |

## Unit Abbreviations

| Unit | Meaning | Dimension |
|------|---------|-----------|
| mg | milligram | Mass |
| mL | milliliter | Volume |
| L | liter | Volume |
| h | hour | Time |
| L/h | liters per hour | Clearance |
| mg/mL | milligrams per milliliter | Concentration |
| 1/h | per hour | Rate constant |

## See Also

- [Epistemic Types](../epistemic/core.md)
- [GUM Compliance](../epistemic/gum.md)
- [Uncertainty Recipes](../cookbook/uncertainty-recipes.md)
- [Language Reference](../language/index.md)
