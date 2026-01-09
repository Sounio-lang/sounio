# MCMC Sampling API

Markov Chain Monte Carlo (MCMC) provides production-grade posterior inference with full epistemic tracking. This module implements state-of-the-art samplers with automatic diagnostics and uncertainty quantification.

## Overview

MCMC samples from posterior distributions, but the samples themselves carry uncertainty. Sounio's MCMC module tracks:

- **Within-chain variance**: Sampling noise
- **Between-chain variance**: Convergence uncertainty
- **Effective sample size**: Information content
- **Posterior uncertainty**: Target distribution spread

## Quick Start

```sio
use epistemic::mcmc::{sample, nuts_options_default, summarize_posterior}

// Define log-posterior (unnormalized)
fn log_posterior(theta: &EVector<2>) -> f64 {
    let mu = theta.values[0]
    let sigma = theta.values[1]
    // Normal likelihood + priors
    -0.5 * (mu * mu) / (sigma * sigma) - log(sigma)
}

// Define gradient
fn grad_log_posterior(theta: &EVector<2>) -> EVector<2> {
    // Gradient of log-posterior
    evec_new([-mu / (sigma * sigma), 1.0 / sigma - mu * mu / (sigma * sigma * sigma)])
}

// Sample with NUTS
let summary = sample(log_posterior, grad_log_posterior, 4, 1000)

// Check convergence
if summary.diagnostics.converged.value {
    println("Converged! Mean = {}", summary.parameters[0].mean.get())
}
```

## Samplers

### NUTS (No-U-Turn Sampler)

The recommended sampler for most problems. NUTS automatically tunes the number of leapfrog steps using the no-U-turn criterion.

#### `nuts_sample`

```sio
pub fn nuts_sample<const N: usize>(
    log_prob_fn: fn(&EVector<N>) -> f64,
    grad_fn: fn(&EVector<N>) -> EVector<N>,
    init: EVector<N>,
    num_samples: i32,
    num_warmup: i32,
    opts: NUTSOptions
) -> Chain<N> with Prob, Alloc
```

**Parameters:**
- `log_prob_fn`: Log-probability function (unnormalized posterior)
- `grad_fn`: Gradient of log-probability
- `init`: Initial parameter values
- `num_samples`: Number of post-warmup samples
- `num_warmup`: Number of warmup (adaptation) samples
- `opts`: NUTS configuration options

**Returns:** `Chain<N>` containing samples and diagnostics

**Example:**
```sio
let opts = nuts_options_default()
let init = evec_new([0.0, 1.0])
let chain = nuts_sample(log_posterior, grad_posterior, init, 1000, 500, opts)

println("Accept rate: {}", chain.accept_rate)
println("Final step size: {}", chain.final_step_size)
```

#### `nuts_sample_chains`

Run multiple chains in parallel for convergence diagnostics.

```sio
pub fn nuts_sample_chains<const N: usize>(
    log_prob_fn: fn(&EVector<N>) -> f64,
    grad_fn: fn(&EVector<N>) -> EVector<N>,
    inits: [EVector<N>],
    num_samples: i32,
    num_warmup: i32,
    opts: NUTSOptions
) -> ChainCollection<N> with Prob, Alloc
```

**Best Practice:** Run at least 4 chains from dispersed initial values.

### Metropolis-Hastings

Classic random-walk Metropolis-Hastings sampler. Useful when gradients are unavailable.

#### `mh_sample`

```sio
pub fn mh_sample<const N: usize>(
    log_prob_fn: fn(&EVector<N>) -> f64,
    init: EVector<N>,
    num_samples: i32,
    num_warmup: i32,
    opts: MHOptions
) -> Chain<N> with Prob, Alloc
```

**Parameters:**
- `log_prob_fn`: Log-probability function
- `init`: Initial parameter values
- `num_samples`: Number of samples
- `num_warmup`: Warmup iterations for proposal adaptation
- `opts`: MH configuration

**Example:**
```sio
let opts = mh_options_default()
let chain = mh_sample(log_posterior, init, 10000, 2000, opts)
```

## Configuration Types

### `NUTSOptions`

```sio
struct NUTSOptions {
    max_tree_depth: i32,      // Maximum tree depth (default: 10)
    target_accept: f64,        // Target acceptance rate (default: 0.8)
    adapt_delta: f64,          // Step size adaptation rate
    adapt_mass_matrix: bool,   // Whether to adapt mass matrix
    init_step_size: f64,       // Initial step size (0 = auto)
    dense_mass: bool,          // Use dense vs diagonal mass matrix
    warmup_stages: i32,        // Number of warmup adaptation stages
}
```

#### Constructors

```sio
/// Default options (recommended for most problems)
fn nuts_options_default() -> NUTSOptions

/// Options with custom target acceptance
fn nuts_options_with_target(target: f64) -> NUTSOptions
```

**Default Values:**
- `max_tree_depth`: 10 (prevents runaway tree building)
- `target_accept`: 0.8 (optimal for NUTS)
- `adapt_mass_matrix`: true
- `dense_mass`: false (diagonal is more robust)

### `MHOptions`

```sio
struct MHOptions {
    proposal_scale: f64,       // Scale for proposal distribution
    adapt_proposal: bool,      // Whether to adapt proposal
    target_accept: f64,        // Target acceptance rate (0.234 optimal)
    adapt_interval: i32,       // Iterations between adaptation
}
```

#### Constructors

```sio
fn mh_options_default() -> MHOptions
```

**Default Values:**
- `proposal_scale`: 1.0
- `target_accept`: 0.234 (optimal for random walk MH)
- `adapt_interval`: 100

## Chain Types

### `Sample<N>`

Single MCMC sample.

```sio
struct Sample<const N: usize> {
    theta: EVector<N>,         // Parameter values
    log_prob: f64,             // Log probability
    accept: bool,              // Whether this was an accept
    divergent: bool,           // Whether transition was divergent
    tree_depth: i32,           // Tree depth (NUTS only)
    energy: f64,               // Hamiltonian energy
}
```

### `Chain<N>`

Full chain with samples and metadata.

```sio
struct Chain<const N: usize> {
    samples: [Sample<N>],      // All samples
    warmup_samples: i32,       // Number of warmup samples
    final_step_size: f64,      // Final adapted step size
    final_mass_matrix: MassMatrix,
    accept_rate: f64,          // Overall acceptance rate
    divergence_rate: f64,      // Rate of divergent transitions
    max_tree_depth_rate: f64,  // Rate of max tree depth hits
}
```

### `ChainCollection<N>`

Multiple chains for multi-chain inference.

```sio
struct ChainCollection<const N: usize> {
    chains: [Chain<N>],
    num_chains: i32,
    num_samples: i32,
    num_warmup: i32,
}
```

## Convergence Diagnostics

### R-hat (Gelman-Rubin Statistic)

Compares between-chain and within-chain variance.

```sio
fn compute_rhat<const N: usize>(chains: &ChainCollection<N>, param_idx: usize) -> f64
```

**Interpretation:**
- R-hat < 1.01: Chains have converged
- R-hat < 1.05: Acceptable for most purposes
- R-hat >= 1.1: Chains have NOT converged

### Effective Sample Size (ESS)

Accounts for autocorrelation in chains.

```sio
fn compute_ess_bulk<const N: usize>(chains: &ChainCollection<N>, param_idx: usize) -> f64
fn compute_ess_tail<const N: usize>(chains: &ChainCollection<N>, param_idx: usize) -> f64
```

**Interpretation:**
- ESS > 400: Generally sufficient for reliable estimates
- ESS < 100: Increase samples or improve sampler

**Bulk ESS**: For central estimates (mean, median)
**Tail ESS**: For extreme quantiles (crucial for credible intervals)

### Monte Carlo Standard Error

Standard error of the mean estimate.

```sio
fn compute_mcse_mean<const N: usize>(chains: &ChainCollection<N>, param_idx: usize) -> f64
```

**Formula:** `MCSE = sqrt(Var/ESS)`

## Posterior Summary

### `ParameterSummary`

Summary statistics for a single parameter with epistemic uncertainty.

```sio
struct ParameterSummary {
    mean: Knowledge<f64>,          // Posterior mean with MCSE
    std: Knowledge<f64>,           // Posterior std with uncertainty
    median: Knowledge<f64>,        // Posterior median
    q025: f64,                     // 2.5th percentile
    q25: f64,                      // 25th percentile
    q75: f64,                      // 75th percentile
    q975: f64,                     // 97.5th percentile
    hdi_low: f64,                  // Highest density interval low
    hdi_high: f64,                 // Highest density interval high
    hdi_prob: f64,                 // HDI probability mass
    rhat: f64,                     // Gelman-Rubin statistic
    ess_bulk: f64,                 // Effective sample size (bulk)
    ess_tail: f64,                 // Effective sample size (tail)
    mcse_mean: f64,                // Monte Carlo standard error of mean
    mcse_std: f64,                 // Monte Carlo standard error of std
}
```

**Note:** The `mean`, `std`, and `median` fields are `Knowledge<f64>` values that include:
- The point estimate as the value
- MCSE squared as the variance
- ESS-based confidence
- MCMC provenance

### `ConvergenceDiagnostics`

Overall convergence assessment.

```sio
struct ConvergenceDiagnostics {
    all_rhat_ok: bool,             // All R-hat < 1.01
    min_ess_bulk: f64,             // Minimum bulk ESS
    min_ess_tail: f64,             // Minimum tail ESS
    max_rhat: f64,                 // Maximum R-hat across parameters
    total_divergences: i32,        // Total divergent transitions
    max_treedepth_rate: f64,       // Rate of max tree depth hits
    converged: Knowledge<bool>,    // Overall convergence with confidence
}
```

### `PosteriorSummary<N>`

Complete posterior summary.

```sio
struct PosteriorSummary<const N: usize> {
    parameters: [ParameterSummary; N],
    diagnostics: ConvergenceDiagnostics,
    log_evidence: Knowledge<f64>,  // Log marginal likelihood estimate
    waic: Knowledge<f64>,          // WAIC model comparison
    loo: Knowledge<f64>,           // LOO-CV estimate
}
```

### `summarize_posterior`

Compute full posterior summary.

```sio
fn summarize_posterior<const N: usize>(
    chains: &ChainCollection<N>
) -> PosteriorSummary<N> with Alloc
```

## High-Level API

### `sample`

Sample with sensible defaults.

```sio
fn sample<const N: usize>(
    log_prob_fn: fn(&EVector<N>) -> f64,
    grad_fn: fn(&EVector<N>) -> EVector<N>,
    num_chains: i32,
    num_samples: i32
) -> PosteriorSummary<N> with Prob, Alloc
```

**Example:**
```sio
// Simple 2-parameter model
let summary = sample(log_posterior, grad_posterior, 4, 1000)

// Access results
let mu_mean = summary.parameters[0].mean
let sigma_mean = summary.parameters[1].mean

println("mu = {} +/- {}", mu_mean.get(), mu_mean.std())
```

### `check_convergence`

Quick convergence check.

```sio
fn check_convergence<const N: usize>(summary: &PosteriorSummary<N>) -> bool
```

### `print_diagnostics`

Print convergence warnings.

```sio
fn print_diagnostics<const N: usize>(summary: &PosteriorSummary<N>) with IO
```

**Example Output:**
```
WARNING: Some R-hat values > 1.01 (max: 1.03)
WARNING: Low bulk ESS (min: 312.5)
WARNING: 5 divergent transitions
```

## Highest Density Interval

### `compute_hdi`

Compute the Highest Density Interval (HDI) - the narrowest interval containing a given probability mass.

```sio
fn compute_hdi(sorted: &[f64], prob: f64) -> (f64, f64)
```

**Parameters:**
- `sorted`: Sorted array of samples
- `prob`: Probability mass (e.g., 0.95 for 95% HDI)

**Returns:** Tuple of (lower, upper) bounds

**Why HDI over Equal-Tailed Intervals:**
- HDI is the shortest interval containing the given probability
- More intuitive interpretation
- Better for asymmetric distributions

## Divergent Transitions

Divergent transitions indicate that the sampler encountered regions where the numerical integration broke down. This often signals:

1. **Highly curved posterior** - Need smaller step size
2. **Multimodal posterior** - May need different parameterization
3. **Model misspecification** - Check priors and likelihood

**Handling Divergences:**
```sio
let opts = nuts_options_with_target(0.9)  // Higher target accept
// Or increase max_tree_depth
// Or reparameterize the model
```

## Complete Example

```sio
use epistemic::mcmc::*
use epistemic::linalg::{EVector, evec_new, evec_zeros}

// Bayesian linear regression
fn main() with Prob, Alloc, IO {
    // Simulated data
    let x = [1.0, 2.0, 3.0, 4.0, 5.0]
    let y = [2.1, 4.2, 5.8, 8.1, 10.1]

    // Log-posterior: y ~ Normal(beta0 + beta1*x, sigma)
    fn log_posterior(theta: &EVector<3>) -> f64 {
        let beta0 = theta.values[0]
        let beta1 = theta.values[1]
        let log_sigma = theta.values[2]
        let sigma = exp(log_sigma)

        // Priors
        var lp = -0.5 * beta0 * beta0 / 100.0  // N(0, 10)
        lp = lp - 0.5 * beta1 * beta1 / 100.0  // N(0, 10)
        lp = lp - log_sigma                     // half-Cauchy via Jacobian

        // Likelihood
        for i in 0..5 {
            let mu = beta0 + beta1 * x[i]
            let resid = y[i] - mu
            lp = lp - log_sigma - 0.5 * resid * resid / (sigma * sigma)
        }

        lp
    }

    fn grad_log_posterior(theta: &EVector<3>) -> EVector<3> {
        // Numerical gradient (or derive analytically)
        numerical_gradient(theta, log_posterior, 1e-6)
    }

    // Sample
    let summary = sample(log_posterior, grad_log_posterior, 4, 2000)

    // Check convergence
    print_diagnostics(&summary)

    if check_convergence(&summary) {
        println("\nPosterior Summary:")
        println("beta0 = {} (95% CI: [{}, {}])",
            summary.parameters[0].mean.get(),
            summary.parameters[0].q025,
            summary.parameters[0].q975)
        println("beta1 = {} (95% CI: [{}, {}])",
            summary.parameters[1].mean.get(),
            summary.parameters[1].q025,
            summary.parameters[1].q975)
        println("sigma = {} (95% HDI: [{}, {}])",
            exp(summary.parameters[2].mean.get()),
            summary.parameters[2].hdi_low,
            summary.parameters[2].hdi_high)
    } else {
        println("WARNING: Chains did not converge!")
    }
}
```

## Performance Considerations

| Sampler | Gradient Required | Optimal Acceptance | Scaling |
|---------|-------------------|-------------------|---------|
| NUTS | Yes | 0.8 | O(d^1.5) |
| HMC | Yes | 0.65 | O(d^1.25) |
| MH | No | 0.234 | O(d^2) |

**Recommendations:**
- Use NUTS for most problems (< 100 parameters)
- Use MH only when gradients are unavailable
- Run at least 4 chains for reliable R-hat
- Target ESS > 400 for each parameter

## See Also

- [Knowledge<T> API Reference](knowledge.md)
- [Variance Propagation](propagate.md)
- [Meta-Analysis](meta.md)
- [Sequential Monte Carlo](smc.md)
