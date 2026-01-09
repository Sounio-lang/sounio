# Knowledge<T> API Reference

`Knowledge<T>` is the fundamental epistemic type in Sounio. It wraps a value with complete uncertainty metadata: variance, confidence, and provenance. This is not a simple wrapper - it is a fundamental epistemic primitive that changes how we compute.

## Type Definition

```sio
pub struct Knowledge<T> {
    /// The point estimate
    value: T,

    /// Variance (uncertainty in the value)
    variance: f64,

    /// Confidence as Beta posterior
    confidence: BetaConfidence,

    /// Where this knowledge came from
    provenance: Provenance,
}
```

## Constructors

### `Knowledge::new`

Create new knowledge with full specification.

```sio
pub fn new(
    value: T,
    variance: f64,
    confidence: BetaConfidence,
    source: Source,
) -> Knowledge<T>
```

**Parameters:**
- `value`: The point estimate
- `variance`: Uncertainty as variance (sigma squared)
- `confidence`: Beta posterior representing trust in the claim
- `source`: Origin of the knowledge

**Example:**
```sio
let measurement = Knowledge::new(
    42.0,
    0.25,
    BetaConfidence::from_rate(0.95, 100.0),
    Source::Measurement { instrument: "sensor_A", timestamp: 1234567890 }
)
```

### `Knowledge::measured`

Create from measurement - the typical laboratory scenario.

```sio
pub fn measured(value: T, variance: f64, instrument: string) -> Knowledge<T>
```

**Parameters:**
- `value`: The measured value
- `variance`: Measurement variance
- `instrument`: Instrument identifier for provenance

**Example:**
```sio
let mass = Knowledge::measured(75.5, 0.25, "scale_001")
let temp = Knowledge::measured(37.2, 0.01, "thermometer_A")
```

**Notes:**
- Confidence defaults to uniform (maximum ignorance)
- Provenance source is set to `Source::Measurement`

### `Knowledge::constant`

Create constant value with zero variance and maximum confidence.

```sio
pub fn constant(value: T) -> Knowledge<T>
```

**Example:**
```sio
let pi = Knowledge::constant(3.14159265359)
let avogadro = Knowledge::constant(6.02214076e23)
```

**Notes:**
- Variance is 0.0 (exact value)
- Confidence is very high (alpha = 1000, centered at 0.99)
- Provenance source is `Source::Assertion { author: "constant" }`

### `Knowledge::asserted`

Create from user assertion (not independently verified).

```sio
pub fn asserted(value: T, variance: f64, author: string) -> Knowledge<T>
```

**Example:**
```sio
let estimate = Knowledge::asserted(100.0, 25.0, "domain_expert")
```

**Notes:**
- Confidence defaults to uniform
- Provenance marks this as an assertion, not measurement

## Accessors

### `get`

Get the point estimate.

```sio
pub fn get(self: &Knowledge<T>) -> &T
```

### `var`

Get the variance.

```sio
pub fn var(self: &Knowledge<T>) -> f64
```

### `std`

Get the standard deviation (square root of variance).

```sio
pub fn std(self: &Knowledge<T>) -> f64
```

### `conf`

Get the confidence distribution.

```sio
pub fn conf(self: &Knowledge<T>) -> &BetaConfidence
```

### `prov`

Get the provenance chain.

```sio
pub fn prov(self: &Knowledge<T>) -> &Provenance
```

## Knowledge<f64> Methods

These methods are specific to `Knowledge<f64>`:

### `std_dev`

Get standard deviation (alias for `std`).

```sio
pub fn std_dev(self: &Knowledge<f64>) -> f64
```

### `ci95`

Get 95% confidence interval using normal approximation.

```sio
pub fn ci95(self: &Knowledge<f64>) -> (f64, f64)
```

**Returns:** Tuple of (lower_bound, upper_bound)

**Example:**
```sio
let measurement = Knowledge::measured(100.0, 4.0, "sensor")  // std = 2.0
let (lo, hi) = measurement.ci95()
// lo = 100.0 - 1.96 * 2.0 = 96.08
// hi = 100.0 + 1.96 * 2.0 = 103.92
```

### `prob_gt`

Probability that the true value exceeds a threshold.

```sio
pub fn prob_gt(self: &Knowledge<f64>, threshold: f64) -> f64
```

**Example:**
```sio
let conc = Knowledge::measured(50.0, 4.0, "analyzer")
let prob = conc.prob_gt(45.0)
// P(conc > 45) using normal CDF
```

**Notes:**
- Uses normal approximation
- For zero variance, returns 1.0 if value > threshold, else 0.0

### `prob_lt`

Probability that the true value is below a threshold.

```sio
pub fn prob_lt(self: &Knowledge<f64>, threshold: f64) -> f64
```

### `prob_between`

Probability that the true value falls within an interval.

```sio
pub fn prob_between(self: &Knowledge<f64>, lo: f64, hi: f64) -> f64
```

**Example:**
```sio
let dose = Knowledge::measured(500.0, 100.0, "scale")
let prob_therapeutic = dose.prob_between(450.0, 550.0)
```

## Transformations

### `map`

Map the inner value while preserving epistemic metadata.

```sio
pub fn map<U, F>(self: Knowledge<T>, f: F, operation: string) -> Knowledge<U>
where F: fn(T) -> U
```

**Parameters:**
- `f`: The transformation function
- `operation`: Name for provenance tracking

**Notes:**
- Variance is preserved (may need manual adjustment for non-linear functions)
- Confidence decays by 0.95 factor
- Provenance step is recorded

### `with_provenance`

Add a provenance step without changing the value.

```sio
pub fn with_provenance(self: Knowledge<T>, operation: string) -> Knowledge<T>
```

## Scalar Operations (Knowledge<f64>)

### `scale`

Multiply by a scalar constant.

```sio
pub fn scale(self: Knowledge<f64>, c: f64) -> Knowledge<f64>
```

**Variance Formula:** `Var(cX) = c^2 Var(X)`

**Example:**
```sio
let mass_kg = Knowledge::measured(75.0, 1.0, "scale")
let mass_lb = mass_kg.scale(2.205)  // Convert to pounds
// variance = 1.0 * 2.205^2 = 4.862
```

### `shift`

Add a scalar constant.

```sio
pub fn shift(self: Knowledge<f64>, c: f64) -> Knowledge<f64>
```

**Variance Formula:** `Var(X + c) = Var(X)` (constants add no variance)

### `square`

Square the value.

```sio
pub fn square(self: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(X^2) = 4X^2 Var(X)` (delta method)

### `sqrt`

Square root of the value.

```sio
pub fn sqrt(self: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(sqrt(X)) = Var(X) / (4X)` (delta method)

### `exp`

Exponential function.

```sio
pub fn exp(self: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(e^X) = e^(2X) Var(X)` (delta method)

### `ln`

Natural logarithm.

```sio
pub fn ln(self: Knowledge<f64>) -> Knowledge<f64>
```

**Variance Formula:** `Var(ln(X)) = Var(X) / X^2` (delta method)

## Arithmetic Operators

### Addition (`+`)

```sio
impl Add for Knowledge<f64> {
    fn add(self: Knowledge<f64>, other: Knowledge<f64>) -> Knowledge<f64>
}
```

**Variance Formula:** `Var(X + Y) = Var(X) + Var(Y)` (independent variables)

**Confidence:** Combined and decayed by 0.99

**Example:**
```sio
let a = Knowledge::measured(10.0, 1.0, "sensor_A")
let b = Knowledge::measured(20.0, 4.0, "sensor_B")
let sum = a + b
// sum.value = 30.0
// sum.variance = 1.0 + 4.0 = 5.0
```

### Subtraction (`-`)

```sio
impl Sub for Knowledge<f64> {
    fn sub(self: Knowledge<f64>, other: Knowledge<f64>) -> Knowledge<f64>
}
```

**Variance Formula:** `Var(X - Y) = Var(X) + Var(Y)` (variances add, not subtract!)

### Multiplication (`*`)

```sio
impl Mul for Knowledge<f64> {
    fn mul(self: Knowledge<f64>, other: Knowledge<f64>) -> Knowledge<f64>
}
```

**Variance Formula (delta method):**
```
Var(XY) = Y^2 Var(X) + X^2 Var(Y)
```

**Confidence:** Combined and decayed by 0.98

### Division (`/`)

```sio
impl Div for Knowledge<f64> {
    fn div(self: Knowledge<f64>, other: Knowledge<f64>) -> Knowledge<f64>
}
```

**Variance Formula (delta method):**
```
Var(X/Y) = (1/Y^2) Var(X) + (X^2/Y^4) Var(Y)
```

**Confidence:** Combined and decayed by 0.97

**Example:**
```sio
let dose = Knowledge::measured(500.0, 25.0, "scale")
let volume = Knowledge::measured(10.0, 0.01, "pipette")
let concentration = dose / volume
// Variance propagated using delta method
```

## BetaConfidence Type

`BetaConfidence` represents confidence as a Beta distribution, capturing uncertainty about our uncertainty.

### Constructors

#### `BetaConfidence::new`

```sio
pub fn new(alpha: f64, b: f64) -> BetaConfidence
```

#### `BetaConfidence::uniform`

Uniform prior (maximum ignorance).

```sio
pub fn uniform() -> BetaConfidence
// Returns Beta(1, 1)
```

#### `BetaConfidence::jeffreys`

Jeffreys uninformative prior.

```sio
pub fn jeffreys() -> BetaConfidence
// Returns Beta(0.5, 0.5)
```

#### `BetaConfidence::from_rate`

From observed success rate and sample size.

```sio
pub fn from_rate(rate: f64, n: f64) -> BetaConfidence
```

**Example:**
```sio
// Observed 80% success rate in 100 trials
let conf = BetaConfidence::from_rate(0.8, 100.0)
```

#### `BetaConfidence::from_observations`

From success and failure counts.

```sio
pub fn from_observations(successes: i64, failures: i64) -> BetaConfidence
```

#### `BetaConfidence::strong`

Strong confidence centered at a value.

```sio
pub fn strong(center: f64, strength: f64) -> BetaConfidence
```

### Methods

#### `mean`

Mean confidence (point estimate).

```sio
pub fn mean(self: &BetaConfidence) -> f64
```

#### `variance`

Variance (uncertainty about confidence).

```sio
pub fn variance(self: &BetaConfidence) -> f64
```

#### `concentration`

Effective sample size (alpha + beta).

```sio
pub fn concentration(self: &BetaConfidence) -> f64
```

#### `uncertainty`

Uncertainty score (inverse of concentration).

```sio
pub fn uncertainty(self: &BetaConfidence) -> f64
```

#### `needs_exploration`

Should we acquire more data?

```sio
pub fn needs_exploration(self: &BetaConfidence, threshold: f64) -> bool
```

#### `update`

Update with new evidence.

```sio
pub fn update(self: &BetaConfidence, successes: i64, failures: i64) -> BetaConfidence
```

#### `combine`

Combine two independent confidences.

```sio
pub fn combine(self: &BetaConfidence, other: &BetaConfidence) -> BetaConfidence
```

#### `decay`

Decay confidence for propagation.

```sio
pub fn decay(self: &BetaConfidence, factor: f64) -> BetaConfidence
```

## Provenance Types

### Source

```sio
pub enum Source {
    /// Direct measurement from instrument
    Measurement { instrument: string, timestamp: i64 },

    /// Computed from other values
    Computed { operation: string },

    /// User assertion (not independently verified)
    Assertion { author: string },

    /// External data source
    External { source: string, url: string },

    /// Unknown origin
    Unknown,
}
```

### ProvenanceStep

```sio
pub struct ProvenanceStep {
    /// What operation was performed
    operation: string,

    /// When (unix timestamp)
    timestamp: i64,

    /// Confidence decay from this operation
    decay_factor: f64,
}
```

### Provenance

```sio
pub struct Provenance {
    /// Original source
    source: Source,

    /// Chain of transformations
    steps: Vec<ProvenanceStep>,
}
```

#### Methods

##### `to_string`

Format provenance as human-readable string.

```sio
pub fn to_string(self: &Provenance) -> string
```

**Example:**
```sio
let x = Knowledge::measured(10.0, 0.1, "sensor_A")
let y = x.sqrt().ln()
println("{}", y.prov().to_string())
// Output: "sensor_A -> sqrt -> ln"
```

##### `depth`

Count transformation depth.

```sio
pub fn depth(self: &Provenance) -> i64
```

## Invariants

`Knowledge<T>` maintains these invariants:

1. **Confidence bounds:** `0.0 <= confidence <= 1.0`
2. **Non-negative variance:** `variance >= 0.0`
3. **Confidence monotonicity:** Confidence never increases through pure computation
4. **Provenance immutability:** Steps are append-only

## Integration with Units

```sio
use epistemic::Knowledge
use units::{mg, mL}

let dose: Knowledge<mg> = Knowledge::measured(500.0_mg, 25.0, "scale")
let volume: Knowledge<mL> = Knowledge::measured(10.0_mL, 0.01, "pipette")
let concentration: Knowledge<mg/mL> = dose / volume
```

## Complete Example

```sio
use epistemic::{Knowledge, BetaConfidence, Source}
use units::{mg, mL, h}

// Pharmacokinetic calculation with full uncertainty tracking
fn calculate_pk() {
    // Drug dose
    let dose = Knowledge::measured(500.0, 25.0, "scale_A")

    // Volume of distribution
    let vd = Knowledge::measured(50.0, 10.0, "population_estimate")

    // Initial concentration
    let c0 = dose / vd  // Variance propagates automatically

    // Check therapeutic window
    let prob_therapeutic = c0.prob_between(8.0, 12.0)

    // Report with uncertainty
    let (lo, hi) = c0.ci95()
    println("C0 = {} +/- {} mg/L", c0.get(), c0.std())
    println("95% CI: [{}, {}]", lo, hi)
    println("P(therapeutic) = {}", prob_therapeutic)
    println("Provenance: {}", c0.prov().to_string())
}
```

## See Also

- [Epistemic Module Overview](index.md)
- [Variance Propagation](propagate.md)
- [MCMC Sampling](mcmc.md)
