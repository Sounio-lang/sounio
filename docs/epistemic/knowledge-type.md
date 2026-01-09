---
title: The Knowledge<T> Type
description: Sounio's core epistemic primitive - wrapping values with uncertainty, confidence, and provenance
prerequisites: [docs/epistemic/index.md]
reading_time: 15 minutes
---

# The Knowledge<T> Type

`Knowledge<T>` is the fundamental epistemic type in Sounio. It wraps any value with complete epistemic metadata: how uncertain it is, how confident we are in that uncertainty, and where the value came from.

This is not a wrapper around `T` with metadata bolted on. This is a **fundamental change in how computation works**.

## Structure

```sio
pub struct Knowledge<T> {
    /// The point estimate (the actual value)
    value: T,

    /// Variance (uncertainty squared) - how precisely do we know this?
    variance: f64,

    /// Confidence as a Beta posterior - how reliable is our uncertainty estimate?
    confidence: BetaConfidence,

    /// Where this knowledge came from and how it was transformed
    provenance: Provenance,
}
```

### The Four Components

| Component | Type | Meaning |
|-----------|------|---------|
| `value` | `T` | The point estimate - what we think the value is |
| `variance` | `f64` | Uncertainty squared - how much the value could vary |
| `confidence` | `BetaConfidence` | Meta-uncertainty - how sure are we about our uncertainty? |
| `provenance` | `Provenance` | Complete history - where did this come from? |

**Key insight:** Uncertainty and Confidence are **orthogonal** concepts:
- **Uncertainty** (variance): How precisely do we know the VALUE? (metrology)
- **Confidence**: How much do we TRUST the claim? (epistemology)

A value can have low uncertainty but low confidence (precise but unverified), or high uncertainty but high confidence (we are very sure about being unsure).

## Constructors

### Knowledge::new

The full constructor with all parameters:

```sio
let measurement = Knowledge::new(
    value: 42.0,
    variance: 4.0,           // Standard deviation = 2.0
    confidence: BetaConfidence::from_rate(0.95, 100.0),
    source: Source::Measurement {
        instrument: "lab_analyzer_001",
        timestamp: 1704067200
    }
)
```

### Knowledge::measured

For laboratory/sensor measurements - the most common case:

```sio
// Create from direct measurement
let mass = Knowledge::measured(
    75.0,                    // Value
    variance: 0.25,          // Variance (std = 0.5)
    instrument: "balance_A"  // Instrument ID
)
```

This sets confidence to a uniform prior (maximum uncertainty about the confidence itself) and records the instrument as provenance.

### Knowledge::constant

For constants with zero uncertainty and maximum confidence:

```sio
// Physical constants, defined values
let speed_of_light = Knowledge::constant(299792458.0)  // m/s, exact by definition
let avogadro = Knowledge::constant(6.02214076e23)      // mol^-1, exact since 2019
```

### Knowledge::asserted

For user assertions - values declared without independent verification:

```sio
// User says this is true, but we have no verification
let estimated_age = Knowledge::asserted(
    35.0,
    variance: 25.0,  // Could be off by 5 years
    author: "patient_self_report"
)
```

### Knowledge::exact

Alias for `constant` - for values known exactly:

```sio
let pi = Knowledge::exact(3.14159265358979)
```

### Knowledge::estimated

For estimated values with specified confidence level:

```sio
// Expert estimate with 80% confidence
let prevalence = Knowledge::estimated(
    value: 0.15,
    variance: 0.0025,
    confidence: 0.80
)
```

## Accessing Components

### Value Access

```sio
let k = Knowledge::measured(42.0, variance: 4.0, instrument: "sensor")

// Get the point estimate
let val: f64 = k.value()        // 42.0
let val_ref: &f64 = k.get()     // Reference to value

// Get variance and standard deviation
let var: f64 = k.var()          // 4.0
let std: f64 = k.std()          // 2.0 (sqrt of variance)
let std: f64 = k.std_dev()      // 2.0 (alias)
```

### Confidence Access

```sio
// Get confidence object
let conf: &BetaConfidence = k.conf()

// Get confidence mean (point estimate of confidence)
let conf_mean: f64 = k.conf().mean()  // e.g., 0.95

// Get confidence variance (uncertainty about confidence)
let conf_var: f64 = k.conf().variance()

// Get concentration (effective sample size)
let n_eff: f64 = k.conf().concentration()
```

### Provenance Access

```sio
// Get provenance chain
let prov: &Provenance = k.prov()

// Format as string
let trail: string = k.prov().to_string()
// "balance_A -> normalize -> log_transform"

// Count transformation depth
let depth: i64 = k.prov().depth()
```

## Statistical Queries

For `Knowledge<f64>`, additional statistical methods are available:

### Confidence Intervals

```sio
let measurement = Knowledge::measured(100.0, variance: 25.0, instrument: "sensor")

// 95% confidence interval (normal approximation)
let (lo, hi): (f64, f64) = measurement.ci95()
// (90.2, 109.8) for value=100, std=5

// Custom confidence level
let (lo, hi) = measurement.ci(0.99)  // 99% CI
```

### Probability Queries

```sio
// P(X > threshold)
let p_above: f64 = measurement.prob_gt(95.0)
// Probability that true value exceeds 95

// P(X < threshold)
let p_below: f64 = measurement.prob_lt(105.0)
// Probability that true value is below 105

// P(lo < X < hi)
let p_between: f64 = measurement.prob_between(90.0, 110.0)
// Probability that true value is in range
```

### Exploration Needs

```sio
// Should we acquire more data?
let needs_more: bool = measurement.conf().needs_exploration(0.01)
// True if variance of confidence exceeds threshold

// Uncertainty score (inverse of concentration)
let explore_priority: f64 = measurement.conf().uncertainty()
// Higher = more exploration needed
```

## Type Conversions

### From Raw Values

```sio
// Implicit conversion (zero uncertainty, maximum confidence)
let k: Knowledge<f64> = 42.0.into()

// Explicit with uncertainty
let k = Knowledge::from(42.0, uncertainty: 2.0)
```

### To Raw Values

Extracting raw values requires explicit acknowledgment of uncertainty:

```sio
let measurement = Knowledge::measured(42.0, variance: 4.0, instrument: "sensor")

// Direct extraction is a COMPILE ERROR
// let raw = measurement.value  // ERROR: cannot drop uncertainty

// Option 1: Require high confidence
let raw = measurement.unwrap_certain()  // Panics if confidence < 0.99

// Option 2: Explicit acknowledgment
let raw = measurement.acknowledge_uncertainty()  // Logs warning, returns value

// Option 3: Get with threshold
match measurement.try_unwrap(confidence_threshold: 0.95) {
    Some(v) => use_value(v),
    None => request_more_data(),
}
```

## Integration with Units

`Knowledge<T>` composes with Sounio's unit system:

```sio
use units::{mg, mL, mg_per_mL}

// Knowledge with units
let dose: Knowledge<mg> = Knowledge::measured(
    500.0_mg,
    variance: 625.0,  // (25 mg)^2
    instrument: "analytical_balance"
)

let volume: Knowledge<mL> = Knowledge::measured(
    10.0_mL,
    variance: 0.01,   // (0.1 mL)^2
    instrument: "volumetric_pipette"
)

// Division preserves both units AND propagates uncertainty
let concentration: Knowledge<mg_per_mL> = dose / volume
// concentration.value = 50.0 mg/mL
// concentration.variance computed via GUM
// Units are type-checked at compile time
```

## The BetaConfidence Type

Confidence is not a single number - it is a **distribution**. `BetaConfidence` uses the Beta distribution to model uncertainty about confidence itself.

```sio
pub struct BetaConfidence {
    dist: Beta,  // Beta(alpha, beta) distribution
}
```

### Why Beta Distribution?

The Beta distribution is the conjugate prior for Bernoulli/binomial data, making it ideal for modeling confidence:
- Bounded to [0, 1] - perfect for probabilities
- Flexible shape - can represent uniform, peaked, or skewed beliefs
- Easy updating - new evidence naturally updates the posterior

### Constructors

```sio
// Uniform prior (maximum ignorance)
let conf = BetaConfidence::uniform()  // Beta(1, 1)

// Jeffreys prior (uninformative)
let conf = BetaConfidence::jeffreys()  // Beta(0.5, 0.5)

// From success rate and sample size
let conf = BetaConfidence::from_rate(0.8, 100.0)  // 80 successes in 100 trials

// From observations
let conf = BetaConfidence::from_observations(80, 20)  // 80 successes, 20 failures

// Strong confidence centered at value
let conf = BetaConfidence::strong(0.95, 1000.0)  // High concentration at 95%
```

### Methods

```sio
let conf = BetaConfidence::from_rate(0.9, 100.0)

// Point estimates
conf.mean()        // Expected confidence: 0.9
conf.variance()    // Variance of confidence: ~0.0009

// Concentration (effective sample size)
conf.concentration()  // alpha + beta: ~102

// Exploration priority
conf.uncertainty()              // 1/concentration: ~0.01
conf.needs_exploration(0.001)   // variance > threshold?

// Update with new evidence
let updated = conf.update(successes: 9, failures: 1)

// Combine independent evidence
let combined = conf1.combine(&conf2)

// Decay through transformation
let decayed = conf.decay(0.95)  // Reduce effective sample size by 5%
```

## The Provenance Type

Provenance tracks the complete history of a value:

```sio
pub struct Provenance {
    source: Source,              // Original origin
    steps: Vec<ProvenanceStep>,  // Transformation chain
}

pub enum Source {
    Measurement { instrument: string, timestamp: i64 },
    Computed { operation: string },
    Assertion { author: string },
    External { source: string, url: string },
    Unknown,
}

pub struct ProvenanceStep {
    operation: string,    // What was done
    timestamp: i64,       // When
    decay_factor: f64,    // Confidence decay from this step
}
```

### Querying Provenance

```sio
let measurement = Knowledge::measured(42.0, variance: 4.0, instrument: "sensor_001")
let processed = measurement.sqrt().scale(2.0).ln()

// Full provenance trail
let trail = processed.prov().to_string()
// "sensor_001 -> sqrt -> scale -> ln"

// Transformation depth
let depth = processed.prov().depth()  // 3

// Original source
match processed.prov().source {
    Source::Measurement { instrument, .. } => {
        println("Original instrument: " + instrument)
    },
    _ => {}
}
```

### Adding Provenance

```sio
// Add a provenance step without changing the value
let annotated = measurement.with_provenance("quality_check")
```

## Transformations

### map

Transform the inner value while preserving epistemic metadata:

```sio
let temp_celsius = Knowledge::measured(25.0, variance: 0.25, instrument: "thermometer")

let temp_fahrenheit = temp_celsius.map(
    |c| c * 9.0 / 5.0 + 32.0,
    operation: "celsius_to_fahrenheit"
)
// Note: map preserves variance naively - for proper propagation, use arithmetic ops
```

### Arithmetic Operations

Arithmetic operators automatically propagate uncertainty (see [Uncertainty Propagation](uncertainty-propagation.md)):

```sio
let a = Knowledge::measured(10.0, variance: 1.0, instrument: "A")
let b = Knowledge::measured(5.0, variance: 0.25, instrument: "B")

let sum = a + b       // Var(sum) = Var(a) + Var(b)
let diff = a - b      // Var(diff) = Var(a) + Var(b)
let prod = a * b      // Var(prod) = b^2*Var(a) + a^2*Var(b)
let quot = a / b      // Var(quot) = (1/b)^2*Var(a) + (a/b^2)^2*Var(b)
```

### Mathematical Functions

```sio
let x = Knowledge::measured(4.0, variance: 0.16, instrument: "sensor")

x.sqrt()    // Var(sqrt(x)) = Var(x) / (4*x)
x.square()  // Var(x^2) = 4*x^2*Var(x)
x.exp()     // Var(e^x) = e^(2x)*Var(x)
x.ln()      // Var(ln(x)) = Var(x) / x^2
x.scale(c)  // Var(c*x) = c^2*Var(x)
x.shift(c)  // Var(x+c) = Var(x)
```

## Design Principles

### Variance Over Error Bars

We track variance (sigma^2) not standard deviation (sigma) because:
- Variance is additive for independent variables
- Propagation formulas are cleaner with variance
- No sign ambiguity

### Confidence is a Distribution

We use `BetaConfidence` instead of a single number because:
- A point estimate of confidence is epistemically incomplete
- We need to know "how sure are we about being sure?"
- Beta distributions naturally update with Bayesian evidence

### Provenance is First-Class

Every `Knowledge` value knows its history because:
- Regulatory compliance requires audit trails
- Debugging requires knowing where data came from
- Reproducibility requires complete lineage

### Decay is Explicit

Transformations decay confidence because:
- Derived values are inherently less certain than measurements
- Each computation step can introduce errors
- The decay factor is recorded in provenance

## See Also

- [Uncertainty Propagation](uncertainty-propagation.md) - How uncertainty flows through computations
- [Confidence Gates](confidence-gates.md) - Control flow based on confidence
- [stdlib/epistemic/knowledge.sio](/stdlib/epistemic/knowledge.sio) - Implementation source
- [stdlib/epistemic/core.sio](/stdlib/epistemic/core.sio) - Core epistemic value type
