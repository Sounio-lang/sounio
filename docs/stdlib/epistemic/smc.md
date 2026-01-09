# Sequential Monte Carlo (SMC) API

Sequential Monte Carlo methods, also known as particle filters, provide uncertainty propagation for state-space models and sequential inference problems. SMC is essential when the posterior changes over time or when MCMC is impractical.

## Overview

SMC represents probability distributions using weighted particles:

```
p(x) = sum_i w_i * delta(x - x_i)
```

Where each particle `x_i` has weight `w_i` and the weights sum to 1.

**Key advantages over MCMC:**
- Naturally handles sequential data
- Provides marginal likelihood estimates
- Can track time-varying posteriors
- Embarrassingly parallel

## Core Types

### `Particle<T>`

A single weighted particle.

```sio
pub struct Particle<T> {
    /// The particle state
    value: T,

    /// Log of unnormalized weight
    log_weight: f64,

    /// Normalized weight (sum to 1 across cloud)
    weight: f64,

    /// Index of parent particle (for ancestry tracking)
    ancestor: Option<usize>,
}
```

### `ParticleCloud<T>`

Collection of particles approximating a distribution.

```sio
pub struct ParticleCloud<T> {
    /// All particles
    particles: Vec<Particle<T>>,

    /// Number of particles
    n_particles: usize,

    /// Effective sample size
    ess: f64,

    /// Accumulated log-likelihood
    log_likelihood: f64,

    /// Generation counter
    generation: usize,
}
```

## ParticleCloud Methods

### Construction

#### `from_prior`

Create particle cloud by sampling from prior.

```sio
pub fn from_prior<F>(n: usize, sampler: F) -> ParticleCloud<T>
where F: fn(usize) -> T
```

**Parameters:**
- `n`: Number of particles
- `sampler`: Function returning prior samples (given particle index)

**Example:**
```sio
// Sample from N(0, 1) prior
let cloud = ParticleCloud::from_prior(1000, |_| random_normal())
```

### Statistics

#### `len`

Number of particles.

```sio
pub fn len(self: &ParticleCloud<T>) -> usize
```

#### `effective_sample_size`

Effective sample size: ESS = 1 / sum(w_i^2).

```sio
pub fn effective_sample_size(self: &ParticleCloud<T>) -> f64
```

**Interpretation:**
- ESS = N: All particles have equal weight
- ESS = 1: One particle dominates (degeneracy)
- ESS < N/2: Typically triggers resampling

#### `ess_ratio`

ESS as fraction of total particles.

```sio
pub fn ess_ratio(self: &ParticleCloud<T>) -> f64
```

#### `mean`

Weighted mean using a projection function.

```sio
pub fn mean<F>(self: &ParticleCloud<T>, f: F) -> f64
where F: fn(&T) -> f64
```

**Example:**
```sio
let mean_x = cloud.mean(|state| state.x)
```

#### `variance`

Weighted variance.

```sio
pub fn variance<F>(self: &ParticleCloud<T>, f: F) -> f64
where F: fn(&T) -> f64
```

#### `std_dev`

Weighted standard deviation.

```sio
pub fn std_dev<F>(self: &ParticleCloud<T>, f: F) -> f64
where F: fn(&T) -> f64
```

#### `credible_interval`

Credible interval from weighted quantiles.

```sio
pub fn credible_interval<F>(self: &ParticleCloud<T>, f: F, alpha: f64) -> (f64, f64)
where F: fn(&T) -> f64
```

**Parameters:**
- `f`: Projection function
- `alpha`: Tail probability (e.g., 0.05 for 95% CI)

**Example:**
```sio
let (lower, upper) = cloud.credible_interval(|s| s.x, 0.05)
// 95% credible interval for x
```

### Extraction

#### `values`

Extract all particle values.

```sio
pub fn values(self: &ParticleCloud<T>) -> Vec<T>
```

#### `weights`

Extract all normalized weights.

```sio
pub fn weights(self: &ParticleCloud<T>) -> Vec<f64>
```

### Conversion

#### `to_knowledge`

Convert to `Knowledge<f64>` with specified confidence.

```sio
pub fn to_knowledge<F>(self: &ParticleCloud<T>, f: F, confidence: f64) -> Knowledge<f64>
where F: fn(&T) -> f64
```

**Example:**
```sio
let estimate = cloud.to_knowledge(|s| s.x, 0.95)
println("Estimate: {} +/- {}", estimate.get(), estimate.std())
```

## Resampling Strategies

### `ResamplingStrategy`

```sio
pub enum ResamplingStrategy {
    /// Simple multinomial resampling
    /// High variance but unbiased
    Multinomial,

    /// Stratified resampling
    /// Lower variance than multinomial
    Stratified,

    /// Systematic resampling
    /// Lowest variance, deterministic given uniform
    Systematic,

    /// Residual resampling
    /// Deterministic integer part + stochastic residual
    Residual,
}
```

**Comparison:**

| Strategy | Variance | Determinism | Best For |
|----------|----------|-------------|----------|
| Multinomial | Highest | None | Theoretical analysis |
| Stratified | Medium | Partial | General use |
| Systematic | Lowest | High | Most applications |
| Residual | Low | Partial | Large particle counts |

**Recommendation:** Use `Systematic` for most applications.

### `Resampler`

Resampling engine.

```sio
pub struct Resampler {
    strategy: ResamplingStrategy,
    seed: u64,
}
```

#### `resample`

Resample particles based on weights.

```sio
pub fn resample<T>(self: &!Resampler, cloud: &!ParticleCloud<T>)
```

**Effect:**
- Particles with high weight are duplicated
- Particles with low weight are eliminated
- All weights reset to 1/N
- ESS restored to N

## Bootstrap Particle Filter

### `BootstrapFilter<T>`

Standard bootstrap particle filter for sequential state estimation.

```sio
pub struct BootstrapFilter<T> {
    /// Current particle cloud
    cloud: ParticleCloud<T>,

    /// Resampling engine
    resampler: Resampler,

    /// Configuration
    config: ParticleFilterConfig,

    /// Current generation
    generation: usize,
}
```

### `ParticleFilterConfig`

```sio
pub struct ParticleFilterConfig {
    /// Number of particles
    n_particles: usize,

    /// ESS threshold for resampling (as fraction of N)
    ess_threshold: f64,

    /// Resampling strategy
    resampling: ResamplingStrategy,

    /// Random seed
    seed: u64,
}
```

### Methods

#### `new`

Initialize filter with prior.

```sio
pub fn new<F>(config: ParticleFilterConfig, prior_sampler: F) -> BootstrapFilter<T>
where F: fn(usize) -> T
```

#### `predict`

Prediction step: propagate particles through transition model.

```sio
pub fn predict<F>(self: &!BootstrapFilter<T>, transition: F)
where F: fn(&T, usize) -> T
```

**Parameters:**
- `transition`: State transition function `(current_state, particle_index) -> next_state`

#### `update`

Update step: reweight particles based on observation likelihood.

```sio
pub fn update<F>(self: &!BootstrapFilter<T>, log_likelihood: F)
where F: fn(&T) -> f64
```

**Parameters:**
- `log_likelihood`: Log-likelihood of observation given state

**Algorithm:**
1. Compute log-likelihood for each particle
2. Update log-weights
3. Normalize weights
4. Compute ESS
5. Resample if ESS < threshold * N

#### `step`

Combined predict + update step.

```sio
pub fn step<P, L>(self: &!BootstrapFilter<T>, transition: P, log_likelihood: L)
where
    P: fn(&T, usize) -> T,
    L: fn(&T) -> f64
```

#### `cloud`

Get current particle cloud.

```sio
pub fn cloud(self: &BootstrapFilter<T>) -> &ParticleCloud<T>
```

#### `log_likelihood`

Get accumulated log marginal likelihood (model evidence).

```sio
pub fn log_likelihood(self: &BootstrapFilter<T>) -> f64
```

**Note:** This is a key output for model comparison.

## Complete Example: Tracking with Noise

```sio
use epistemic::smc::*

// State: position and velocity
struct State {
    x: f64,
    v: f64,
}

fn main() with Prob, Alloc {
    // Observations (noisy position measurements)
    let observations = [1.1, 2.3, 3.2, 4.5, 5.1, 6.8, 7.2, 8.9]

    // Transition model: constant velocity with noise
    fn transition(s: &State, _: usize) -> State {
        State {
            x: s.x + s.v + random_normal() * 0.1,
            v: s.v + random_normal() * 0.05,
        }
    }

    // Observation model: Gaussian likelihood
    fn log_likelihood(s: &State, obs: f64) -> f64 {
        let sigma = 0.5
        -0.5 * (s.x - obs).pow(2) / (sigma * sigma)
    }

    // Configure filter
    let config = ParticleFilterConfig {
        n_particles: 1000,
        ess_threshold: 0.5,
        resampling: ResamplingStrategy::Systematic,
        seed: 42,
    }

    // Initialize with prior: x ~ N(0, 1), v ~ N(1, 0.5)
    var filter = BootstrapFilter::new(config, |_| State {
        x: random_normal(),
        v: 1.0 + random_normal() * 0.5,
    })

    println("Tracking with {} particles", config.n_particles)
    println("")

    // Process observations
    for t in 0..observations.len() {
        let obs = observations[t]

        // Predict and update
        filter.step(
            |s, _| transition(s),
            |s| log_likelihood(s, obs),
        )

        // Get estimates
        let x_mean = filter.cloud().mean(|s| s.x)
        let x_std = filter.cloud().std_dev(|s| s.x)
        let (x_lo, x_hi) = filter.cloud().credible_interval(|s| s.x, 0.05)

        println("t={}: obs={:.1}, est={:.2} +/- {:.2}, 95% CI: [{:.2}, {:.2}]",
            t, obs, x_mean, x_std, x_lo, x_hi)
    }

    println("")
    println("Final log-likelihood: {:.2}", filter.log_likelihood())

    // Convert final estimate to Knowledge
    let final_x = filter.cloud().to_knowledge(|s| s.x, 0.95)
    println("Final position: {} +/- {}", final_x.get(), final_x.std())
}
```

**Example Output:**
```
Tracking with 1000 particles

t=0: obs=1.1, est=1.05 +/- 0.42, 95% CI: [0.31, 1.82]
t=1: obs=2.3, est=2.21 +/- 0.38, 95% CI: [1.52, 2.91]
t=2: obs=3.2, est=3.18 +/- 0.35, 95% CI: [2.55, 3.84]
t=3: obs=4.5, est=4.42 +/- 0.33, 95% CI: [3.81, 5.02]
t=4: obs=5.1, est=5.15 +/- 0.32, 95% CI: [4.56, 5.75]
t=5: obs=6.8, est=6.62 +/- 0.35, 95% CI: [5.98, 7.28]
t=6: obs=7.2, est=7.35 +/- 0.33, 95% CI: [6.74, 7.97]
t=7: obs=8.9, est=8.71 +/- 0.35, 95% CI: [8.05, 9.36]

Final log-likelihood: -8.42
Final position: 8.71 +/- 0.35
```

## Adaptive Temperature SMC

For SMC samplers (as opposed to filters), adaptive tempering is crucial.

### `TemperatureSchedule`

Temperature schedule for annealed SMC.

```sio
pub struct TemperatureSchedule {
    temperatures: Vec<f64>,
    ess_values: Vec<f64>,
    resample_count: usize,
    total_steps: usize,
}
```

#### Constructors

```sio
/// Fixed linear schedule
fn linear(n_steps: usize) -> TemperatureSchedule

/// Geometric schedule (more resolution near 0)
fn geometric(n_steps: usize, base: f64) -> TemperatureSchedule

/// Empty schedule for adaptive filling
fn adaptive() -> TemperatureSchedule
```

### `AdaptiveTemperatureSelector`

Automatically selects next temperature to maintain target ESS.

```sio
pub struct AdaptiveTemperatureSelector {
    config: AdaptiveConfig,
}
```

#### `find_next_temperature`

Find next temperature using bisection on ESS.

```sio
pub fn find_next_temperature(
    self: &AdaptiveTemperatureSelector,
    current_temp: f64,
    log_likelihoods: &[f64],
    current_weights: &[f64],
    n_particles: usize
) -> f64
```

**Algorithm:**
1. Start with beta = 1.0 (target)
2. Compute ESS at new temperature
3. If ESS too low, reduce beta
4. Use bisection to find beta giving target ESS
5. Return new temperature

### `AdaptiveSMCScheduler`

Full adaptive scheduler.

```sio
pub struct AdaptiveSMCScheduler {
    temp_selector: AdaptiveTemperatureSelector,
    schedule: TemperatureSchedule,
    resample_threshold: f64,
}
```

#### Methods

```sio
/// Create scheduler with ESS threshold
fn new(ess_threshold: f64) -> AdaptiveSMCScheduler

/// Reset to initial state
fn initialize(self: &!AdaptiveSMCScheduler)

/// Advance to next temperature
/// Returns (new_temperature, should_resample)
fn advance(
    self: &!AdaptiveSMCScheduler,
    log_likelihoods: &[f64],
    current_weights: &[f64],
    n_particles: usize
) -> (f64, bool)

/// Check if beta = 1 reached
fn is_complete(self: &AdaptiveSMCScheduler) -> bool

/// Get summary statistics
fn summary(self: &AdaptiveSMCScheduler) -> ScheduleSummary
```

## Particle Degeneracy

**Problem:** Over time, particle weights become concentrated on few particles, reducing effective sample size.

**Solution:** Resample when ESS drops below threshold.

**Best Practices:**
1. Use systematic resampling (lowest variance)
2. Resample when ESS < N/2
3. Use more particles for high-dimensional problems
4. Consider auxiliary particle filters for severe degeneracy

## Integration with Knowledge<T>

```sio
// Convert final particle cloud to Knowledge
let cloud: ParticleCloud<f64> = filter.cloud()

// Extract as Knowledge with 95% confidence
let estimate: Knowledge<f64> = cloud.to_knowledge(|x| x, 0.95)

// Now has full epistemic metadata
println("Value: {}", estimate.get())
println("Uncertainty: {}", estimate.std())
println("Confidence: {}", estimate.conf().mean())
println("Provenance: {}", estimate.prov().to_string())
```

## Performance Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Particles** | Start with 1000, increase if ESS too low |
| **Resampling** | Use Systematic, threshold at 0.5 |
| **Parallelism** | Particles are independent, parallelize evaluation |
| **Memory** | O(N) where N = number of particles |
| **Time** | O(N * T) where T = number of time steps |

## When to Use SMC vs MCMC

| Use SMC When... | Use MCMC When... |
|-----------------|------------------|
| Data arrives sequentially | All data available upfront |
| Need marginal likelihood | Only need posterior samples |
| Posterior changes over time | Static posterior |
| Multimodal posteriors | Unimodal posteriors (usually) |
| Need online inference | Batch inference acceptable |

## References

- Doucet, A., et al. "Sequential Monte Carlo Methods in Practice" (2001)
- Chopin, N. "A sequential particle filter method for static models" (2002)
- Del Moral, P., et al. "Sequential Monte Carlo samplers" (2006)
- Gordon, N., et al. "Novel approach to nonlinear/non-Gaussian Bayesian state estimation" (1993)

## See Also

- [Knowledge<T> API Reference](knowledge.md)
- [MCMC Sampling](mcmc.md)
- [Variance Propagation](propagate.md)
