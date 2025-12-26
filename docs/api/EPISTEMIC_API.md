# Sounio Epistemic Gaps — API Documentation

## Module Overview

This documentation covers the four main components added to close gaps in Sounio's epistemic computing system.

---

## 1. Promotion Lattice (`promotion.rs`)

### Purpose

Provides a formal mathematical lattice structure for uncertainty representations, enabling principled promotion (widening) between different uncertainty models.

### Core Types

#### `UncertaintyLevel`

Enumeration of uncertainty model types with lattice ordering.

```rust
pub enum UncertaintyLevel {
    Point = 0,          // Deterministic value
    Interval = 1,       // Closed interval [a, b]
    Fuzzy = 2,          // Fuzzy set with membership
    Affine = 3,         // Affine arithmetic
    DempsterShafer = 4, // Belief functions
    Distribution = 5,   // Full probability distribution
    Particles = 6,      // Sequential Monte Carlo
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `height()` | `fn height(&self) -> u8` | Lattice height (0-4) |
| `info_capacity()` | `fn info_capacity(&self) -> u32` | Information capacity in bits |
| `cost_multiplier()` | `fn cost_multiplier(&self) -> f64` | Computational cost relative to Point |
| `can_promote_to()` | `fn can_promote_to(&self, target: Self) -> bool` | Check if promotion is valid |
| `promotable_targets()` | `fn promotable_targets(&self) -> Vec<Self>` | All valid promotion targets |
| `parse()` | `fn parse(s: &str) -> Option<Self>` | Parse from string |

**Example:**

```rust
use sounio::epistemic::promotion::UncertaintyLevel;

let level = UncertaintyLevel::Interval;
assert_eq!(level.height(), 1);
assert!(level.can_promote_to(UncertaintyLevel::Distribution));
assert!(!level.can_promote_to(UncertaintyLevel::Point)); // Cannot demote
```

#### `PromotionLattice`

The lattice structure with meet and join operations.

```rust
pub struct PromotionLattice;
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new()` | `fn new() -> Self` | Create lattice instance |
| `meet()` | `fn meet(&self, a: UncertaintyLevel, b: UncertaintyLevel) -> UncertaintyLevel` | Greatest lower bound (⊓) |
| `join()` | `fn join(&self, a: UncertaintyLevel, b: UncertaintyLevel) -> UncertaintyLevel` | Least upper bound (⊔) |
| `is_subtype()` | `fn is_subtype(&self, sub: UncertaintyLevel, sup: UncertaintyLevel) -> bool` | Check subtyping relation |
| `meet_all()` | `fn meet_all(&self, levels: &[UncertaintyLevel]) -> UncertaintyLevel` | Meet of multiple levels |
| `join_all()` | `fn join_all(&self, levels: &[UncertaintyLevel]) -> UncertaintyLevel` | Join of multiple levels |
| `between()` | `fn between(&self, lower: UncertaintyLevel, upper: UncertaintyLevel) -> Vec<UncertaintyLevel>` | All levels between bounds |
| `ascii_diagram()` | `fn ascii_diagram(&self) -> String` | ASCII visualization |
| `mermaid_diagram()` | `fn mermaid_diagram(&self) -> String` | Mermaid diagram code |

**Lattice Properties:**

- **Reflexivity:** `∀a. a ≤ a`
- **Antisymmetry:** `a ≤ b ∧ b ≤ a → a = b`
- **Transitivity:** `a ≤ b ∧ b ≤ c → a ≤ c`

**Example:**

```rust
use sounio::epistemic::promotion::{PromotionLattice, UncertaintyLevel};

let lattice = PromotionLattice::new();

// Meet of incomparable elements
let meet = lattice.meet(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy);
assert_eq!(meet, UncertaintyLevel::Point);

// Join finds common supertype
let join = lattice.join(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy);
assert_eq!(join, UncertaintyLevel::Affine);
```

#### `Promoter`

Performs actual value conversions between uncertainty levels.

```rust
pub struct Promoter {
    pub default_samples: usize,
    pub default_particles: usize,
    pub seed: Option<u64>,
}
```

**Builder Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new()` | `fn new() -> Self` | Create with defaults |
| `with_samples()` | `fn with_samples(self, n: usize) -> Self` | Set Monte Carlo samples |
| `with_particles()` | `fn with_particles(self, n: usize) -> Self` | Set SMC particles |
| `with_seed()` | `fn with_seed(self, seed: u64) -> Self` | Set random seed |

**Promotion Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `promote_point()` | `fn promote_point(&self, value: f64, confidence: f64, target: UncertaintyLevel) -> Result<PromotedValue, PromotionError>` | Promote from Point |
| `promote_interval()` | `fn promote_interval(&self, lower: f64, upper: f64, target: UncertaintyLevel) -> Result<PromotedValue, PromotionError>` | Promote from Interval |

**Example:**

```rust
use sounio::epistemic::promotion::*;

let promoter = Promoter::new()
    .with_samples(10000)
    .with_seed(42);

// Promote point to distribution
let result = promoter.promote_point(10.0, 0.95, UncertaintyLevel::Distribution)?;

if let PromotedValue::Distribution { samples, mean, variance } = result {
    println!("Mean: {}, Variance: {}", mean, variance);
}
```

---

## 2. KEC Auto-Selection (`kec.rs`)

### Purpose

Automatically selects the optimal uncertainty propagation backend based on Knowledge, Entropy, and Complexity analysis.

### Core Types

#### `KECConfig`

Configuration for the KEC selector.

```rust
pub struct KECConfig {
    pub max_cost_multiplier: f64,
    pub required_precision: f64,
    pub min_confidence: f64,
    pub prefer_guaranteed_bounds: bool,
    pub prefer_full_distribution: bool,
    pub max_particles: usize,
    pub max_samples: usize,
    pub entropy_threshold_interval: f64,
    pub entropy_threshold_distribution: f64,
    pub complexity_threshold_mc: f64,
    pub enable_adaptive: bool,
}
```

**Preset Constructors:**

| Method | Description | Key Characteristics |
|--------|-------------|---------------------|
| `default()` | Balanced defaults | Cost ≤ 1000x, precision 1%, confidence 95% |
| `scientific()` | High-precision research | Cost ≤ 10000x, precision 0.1%, confidence 99% |
| `realtime()` | Low-latency systems | Cost ≤ 10x, guaranteed bounds |
| `pkpd()` | Pharmacokinetic modeling | Full distribution, moderate cost |
| `safety_critical()` | Safety-critical systems | Guaranteed bounds, confidence 99.9% |

#### `KECSelector`

Main selector engine.

```rust
pub struct KECSelector {
    config: KECConfig,
    lattice: PromotionLattice,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new()` | `fn new() -> Self` | Create with defaults |
| `with_config()` | `fn with_config(config: KECConfig) -> Self` | Create with custom config |
| `select()` | `fn select(&self, uncertainty: &UncertaintyMetrics, complexity: &ComplexityMetrics) -> KECResult` | Perform selection |

#### `UncertaintyAnalyzer`

Builder for analyzing input uncertainty.

```rust
pub struct UncertaintyAnalyzer {
    inputs: Vec<InputInfo>,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new()` | `fn new() -> Self` | Create empty analyzer |
| `add_point()` | `fn add_point(&mut self, name: &str, value: f64) -> &mut Self` | Add point input |
| `add_gaussian()` | `fn add_gaussian(&mut self, name: &str, mean: f64, std_dev: f64) -> &mut Self` | Add Gaussian input |
| `add_uniform()` | `fn add_uniform(&mut self, name: &str, lower: f64, upper: f64) -> &mut Self` | Add uniform input |
| `add_interval()` | `fn add_interval(&mut self, name: &str, lower: f64, upper: f64) -> &mut Self` | Add interval input |
| `analyze()` | `fn analyze(&self) -> UncertaintyMetrics` | Compute metrics |

#### `ComplexityAnalyzer`

Builder for analyzing computation complexity.

```rust
pub struct ComplexityAnalyzer {
    operations: Vec<OpInfo>,
    depth: usize,
}
```

**Operation Types:**

```rust
pub enum OpKind {
    Add, Sub, Mul, Div, Pow, Sqrt,
    Exp, Log, Sin, Cos, Tan,
    Conditional, Loop, FunctionCall,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new()` | `fn new() -> Self` | Create empty analyzer |
| `add_op()` | `fn add_op(&mut self, kind: OpKind) -> &mut Self` | Add operation |
| `enter_scope()` | `fn enter_scope(&mut self) -> &mut Self` | Increase depth |
| `exit_scope()` | `fn exit_scope(&mut self) -> &mut Self` | Decrease depth |
| `analyze()` | `fn analyze(&self) -> ComplexityMetrics` | Compute metrics |

#### `KECResult`

Result of KEC analysis.

```rust
pub struct KECResult {
    pub recommended_model: UncertaintyLevel,
    pub confidence: f64,
    pub alternatives: Vec<(UncertaintyLevel, f64)>,
    pub estimated_cost: f64,
    pub estimated_accuracy: f64,
    pub reasoning: Vec<String>,
    pub warnings: Vec<String>,
    pub suggested_params: ModelParameters,
}
```

**Example:**

```rust
use sounio::epistemic::kec::*;

// Define inputs
let mut uncertainty = UncertaintyAnalyzer::new();
uncertainty
    .add_gaussian("CL", 10.0, 3.0)    // Clearance
    .add_gaussian("Vd", 50.0, 15.0);  // Volume

// Define computation
let mut complexity = ComplexityAnalyzer::new();
complexity
    .add_op(OpKind::Div)   // ke = CL/Vd
    .add_op(OpKind::Exp);  // exp(-ke*t)

// Select model
let selector = KECSelector::with_config(KECConfig::pkpd());
let result = selector.select(&uncertainty.analyze(), &complexity.analyze());

println!("Recommended: {} (confidence: {:.0}%)",
    result.recommended_model,
    result.confidence * 100.0);

for reason in &result.reasoning {
    println!("  • {}", reason);
}
```

---

## 3. Sequential Monte Carlo (`smc.d`)

### Purpose

Provides particle-based uncertainty propagation for state-space models and sequential inference.

### Core Types

#### `Particle<T>`

A single weighted particle.

```sounio
pub struct Particle<T> {
    value: T,
    log_weight: f64,
    weight: f64,
    ancestor: Option<usize>,
}
```

#### `ParticleCloud<T>`

Collection of particles approximating a distribution.

```sounio
pub struct ParticleCloud<T> {
    particles: Vec<Particle<T>>,
    n_particles: usize,
    ess: f64,
    log_likelihood: f64,
    generation: usize,
}
```

**Methods:**

| Method | Description |
|--------|-------------|
| `from_prior(n, sampler)` | Create from prior samples |
| `len()` | Number of particles |
| `effective_sample_size()` | ESS |
| `ess_ratio()` | ESS / N |
| `mean(f)` | Weighted mean |
| `variance(f)` | Weighted variance |
| `std_dev(f)` | Weighted standard deviation |
| `credible_interval(f, alpha)` | Credible interval |
| `values()` | Extract particle values |
| `weights()` | Extract particle weights |
| `to_uncertain_value()` | Convert to UncertainValue |
| `to_knowledge(confidence)` | Convert to Knowledge |

#### `ResamplingStrategy`

Resampling algorithms.

```sounio
pub enum ResamplingStrategy {
    Multinomial,   // Simple but high variance
    Stratified,    // Lower variance
    Systematic,    // Lowest variance, deterministic
    Residual,      // Combines deterministic + stochastic
}
```

#### `BootstrapFilter<T>`

Bootstrap particle filter for sequential state estimation.

```sounio
pub struct BootstrapFilter<T> {
    cloud: ParticleCloud<T>,
    resampler: Resampler,
    config: ParticleFilterConfig,
    generation: usize,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new()` | `fn new<F>(config: ParticleFilterConfig, prior_sampler: F) -> Self` | Initialize filter |
| `predict()` | `fn predict<F>(&mut self, transition: F)` | Prediction step |
| `update()` | `fn update<F>(&mut self, log_likelihood: F)` | Update with observation |
| `step()` | `fn step<P, L>(&mut self, transition: P, log_likelihood: L)` | Single predict+update |
| `cloud()` | `fn cloud(&self) -> &ParticleCloud<T>` | Get current cloud |
| `log_likelihood()` | `fn log_likelihood(&self) -> f64` | Marginal log-likelihood |

**Example:**

```sounio
use sio.epistemic.smc.*

// Define state-space model
let transition = |x: f64| x + 0.1  // Random walk with drift
let observation = |x: f64, y: f64| -(x - y).pow(2) / 2.0  // Gaussian observation

// Initialize filter
let config = ParticleFilterConfig {
    n_particles: 1000,
    ess_threshold: 0.5,
    resampling: ResamplingStrategy::Systematic,
    seed: 42,
}

var filter = BootstrapFilter::new(config, |_| 0.0)  // Prior: δ(0)

// Process observations
for obs in observations {
    filter.step(
        |x, _| transition(x),
        |x| observation(x, obs),
    )
}

// Get estimate
let estimate = filter.cloud().mean(|x| x)
let (lower, upper) = filter.cloud().credible_interval(|x| x, 0.05)
```

---

## 4. Adaptive SMC (`adaptive_smc.d`)

### Purpose

Automatically determines the temperature schedule for SMC samplers based on ESS criterion.

### Core Types

#### `TemperatureSchedule`

A temperature schedule for SMC.

```sounio
pub struct TemperatureSchedule {
    temperatures: Vec<f64>,
    ess_values: Vec<f64>,
    resample_count: usize,
    total_steps: usize,
}
```

**Constructors:**

| Method | Description |
|--------|-------------|
| `linear(n_steps)` | Fixed linear schedule |
| `geometric(n_steps, base)` | Geometric schedule (more steps near 0) |
| `adaptive()` | Empty schedule for adaptive filling |

#### `AdaptiveTemperatureSelector`

Finds next temperature using bisection on ESS.

```sounio
pub struct AdaptiveTemperatureSelector {
    config: AdaptiveConfig,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new()` | `fn new(config: AdaptiveConfig) -> Self` | Create selector |
| `with_ess_threshold()` | `fn with_ess_threshold(self, threshold: f64) -> Self` | Set ESS threshold |
| `find_next_temperature()` | `fn find_next_temperature(&self, current_temp: f64, log_likelihoods: &[f64], current_weights: &[f64], n_particles: usize) -> f64` | Find next temperature |

#### `AdaptiveSMCScheduler`

Full adaptive scheduler combining temperature selection with resampling.

```sounio
pub struct AdaptiveSMCScheduler {
    temp_selector: AdaptiveTemperatureSelector,
    schedule: TemperatureSchedule,
    resample_threshold: f64,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new()` | `fn new(ess_threshold: f64) -> Self` | Create scheduler |
| `initialize()` | `fn initialize(&mut self)` | Reset to initial state |
| `advance()` | `fn advance(&mut self, log_likelihoods: &[f64], current_weights: &[f64], n_particles: usize) -> (f64, bool)` | Advance to next temperature, returns (new_temp, should_resample) |
| `is_complete()` | `fn is_complete(&self) -> bool` | Check if β = 1 reached |
| `summary()` | `fn summary(&self) -> ScheduleSummary` | Get summary statistics |

**Example:**

```sounio
use sio.epistemic.smc.adaptive.*

var scheduler = AdaptiveSMCScheduler::new(0.5)  // ESS threshold = 50%
scheduler.initialize()

let n = 1000
var weights = vec![1.0 / n as f64; n]

while !scheduler.is_complete() {
    // Compute log-likelihoods at current particles
    let log_likelihoods: Vec<f64> = particles.iter()
        .map(|x| target_log_density(x))
        .collect()

    let (new_temp, should_resample) = scheduler.advance(
        &log_likelihoods,
        &weights,
        n,
    )

    if should_resample {
        // Resample particles
        particles = resample(&particles, &weights)
        weights = vec![1.0 / n as f64; n]
    } else {
        // Just reweight
        weights = reweight(&weights, &log_likelihoods, new_temp - current_temp)
    }
}

let summary = scheduler.summary()
println!("Completed in {} steps with {} resamples",
    summary.n_temperatures,
    summary.n_resamples)
```

---

## Error Types

### `PromotionError`

Errors during value promotion.

```rust
pub enum PromotionError {
    CannotDemote { from: UncertaintyLevel, to: UncertaintyLevel },
    IncompatiblePath { from: UncertaintyLevel, to: UncertaintyLevel },
    InsufficientInfo { from: UncertaintyLevel, to: UncertaintyLevel, reason: String },
}
```

---

## Integration with Existing Sounio Types

### With `Knowledge<T>`

```sounio
// Convert ParticleCloud to Knowledge
let cloud: ParticleCloud<f64> = ...
let knowledge: Knowledge<f64> = cloud.to_knowledge(0.95)
    .with_source("smc")
    .with_domain(ChEBI::PlasmaConcentration)
```

### With `UncertainValue`

```sounio
// Convert ParticleCloud to UncertainValue
let uncertain: UncertainValue = cloud.to_uncertain_value()
```

### With `QuantifiedKnowledge<N, U>`

```rust
// Uncertainty in value with units
let Cmax: QuantifiedKnowledge<ParticleCloud<f64>, MilligramPerLiter> = ...
```

---

## Performance Considerations

| Method | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| Point propagation | O(1) | O(1) |
| Interval propagation | O(ops) | O(1) |
| Affine propagation | O(ops × symbols) | O(symbols) |
| Monte Carlo | O(samples × ops) | O(samples) |
| SMC | O(particles × temps × ops) | O(particles) |

**Recommendations:**

- Use Point for < 0.1% uncertainty
- Use Interval for guaranteed bounds with simple computations
- Use Affine for correlated computations up to ~50 operations
- Use Monte Carlo for complex nonlinear functions
- Use SMC for sequential/state-space problems
