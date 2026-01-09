# MCMC: Markov Chain Monte Carlo

Markov Chain Monte Carlo (MCMC) is a family of algorithms for sampling from probability distributions when direct sampling is intractable. Sounio's MCMC implementation tracks epistemic uncertainty throughout the sampling process.

## Why MCMC?

For most Bayesian models, the posterior distribution cannot be computed analytically:

```sio
// We want: P(theta | data) = P(data | theta) P(theta) / P(data)
// Problem: P(data) = integral of P(data | theta) P(theta) d_theta
//          This integral is usually intractable!

// Solution: Sample from posterior without computing P(data)
```

MCMC constructs a Markov chain whose stationary distribution is the target posterior. Running the chain produces samples that (asymptotically) come from the posterior.

## Metropolis-Hastings

The foundational MCMC algorithm.

### Algorithm

1. Start at some initial state theta_0
2. For each iteration:
   - Propose new state theta' from proposal distribution Q(theta' | theta)
   - Compute acceptance probability: alpha = min(1, P(theta') Q(theta|theta') / P(theta) Q(theta'|theta))
   - Accept with probability alpha; otherwise stay at theta

```sio
use std::epistemic::mcmc::{MHOptions, mh_sample, mh_options_default}

/// Metropolis-Hastings sampler
fn metropolis_hastings(
    log_posterior: fn(&[f64]) -> f64,
    init: [f64],
    n_samples: i64,
    n_warmup: i64,
    proposal_scale: f64
) -> MHResult with Prob {
    var state = init
    var current_log_p = log_posterior(&state)
    var samples: [[f64]] = []
    var n_accepted: i64 = 0

    var iter: i64 = 0
    while iter < n_warmup + n_samples {
        // Propose new state (symmetric random walk proposal)
        var proposed = state
        var j: i64 = 0
        while j < state.len() {
            proposed[j] = state[j] + proposal_scale * sample_normal(0.0, 1.0)
            j = j + 1
        }

        // Compute log acceptance probability
        let proposed_log_p = log_posterior(&proposed)
        let log_alpha = proposed_log_p - current_log_p

        // Accept/reject
        if ln_f64(sample_uniform(0.0, 1.0)) < log_alpha {
            state = proposed
            current_log_p = proposed_log_p
            n_accepted = n_accepted + 1
        }

        // Store sample (after warmup)
        if iter >= n_warmup {
            samples = samples ++ [state]
        }

        iter = iter + 1
    }

    return MHResult {
        samples: samples,
        acceptance_rate: (n_accepted as f64) / ((n_warmup + n_samples) as f64),
    }
}
```

### Tuning the Proposal

The proposal scale critically affects performance:

```sio
/// Adaptive Metropolis-Hastings
fn adaptive_mh(
    log_posterior: fn(&[f64]) -> f64,
    init: [f64],
    n_samples: i64,
    n_warmup: i64
) -> MHResult with Prob {
    var proposal_scale = 1.0
    var state = init
    var current_log_p = log_posterior(&state)
    var samples: [[f64]] = []
    var recent_accepts: i64 = 0
    let adapt_interval: i64 = 100
    let target_accept_rate = 0.234  // Optimal for high-dimensional targets

    var iter: i64 = 0
    while iter < n_warmup + n_samples {
        // MH step
        var proposed = propose(state, proposal_scale)
        let proposed_log_p = log_posterior(&proposed)

        if ln_f64(sample_uniform(0.0, 1.0)) < proposed_log_p - current_log_p {
            state = proposed
            current_log_p = proposed_log_p
            recent_accepts = recent_accepts + 1
        }

        // Adapt during warmup
        if iter < n_warmup && (iter + 1) % adapt_interval == 0 {
            let accept_rate = (recent_accepts as f64) / (adapt_interval as f64)
            if accept_rate > target_accept_rate + 0.05 {
                proposal_scale = proposal_scale * 1.1
            } else if accept_rate < target_accept_rate - 0.05 {
                proposal_scale = proposal_scale * 0.9
            }
            recent_accepts = 0
        }

        if iter >= n_warmup {
            samples = samples ++ [state]
        }
        iter = iter + 1
    }

    return MHResult { samples: samples, acceptance_rate: 0.0 }
}
```

## Hamiltonian Monte Carlo (HMC)

HMC uses gradient information to make efficient proposals, dramatically reducing random walk behavior.

### The Physics Analogy

HMC treats the parameter space as a physical system:
- Parameters theta are "position"
- Auxiliary "momentum" p is introduced
- The Hamiltonian H(theta, p) = -log P(theta) + ||p||^2/2
- Simulated dynamics conserve H, enabling distant proposals

```sio
use std::epistemic::mcmc::{HMCOptions, hmc_options_default}

/// Single leapfrog integration step
fn leapfrog_step(
    position: [f64],
    momentum: [f64],
    gradient: [f64],
    step_size: f64
) -> ([f64], [f64]) {
    // Half step for momentum
    var new_momentum = momentum
    var j: i64 = 0
    while j < momentum.len() {
        new_momentum[j] = momentum[j] + 0.5 * step_size * gradient[j]
        j = j + 1
    }

    // Full step for position
    var new_position = position
    j = 0
    while j < position.len() {
        new_position[j] = position[j] + step_size * new_momentum[j]
        j = j + 1
    }

    return (new_position, new_momentum)
}

/// HMC transition
fn hmc_transition(
    position: [f64],
    log_prob: f64,
    gradient_fn: fn(&[f64]) -> [f64],
    log_prob_fn: fn(&[f64]) -> f64,
    step_size: f64,
    n_leapfrog: i64
) -> ([f64], f64, bool) with Prob {
    // Sample momentum
    var momentum: [f64] = []
    var j: i64 = 0
    while j < position.len() {
        momentum = momentum ++ [sample_normal(0.0, 1.0)]
        j = j + 1
    }

    // Initial Hamiltonian
    let initial_h = -log_prob + 0.5 * dot_product(momentum, momentum)

    // Leapfrog integration
    var pos = position
    var mom = momentum
    var i: i64 = 0
    while i < n_leapfrog {
        let grad = gradient_fn(&pos)
        // Half step momentum
        j = 0
        while j < mom.len() {
            mom[j] = mom[j] + 0.5 * step_size * grad[j]
            j = j + 1
        }
        // Full step position
        j = 0
        while j < pos.len() {
            pos[j] = pos[j] + step_size * mom[j]
            j = j + 1
        }
        // Half step momentum
        let new_grad = gradient_fn(&pos)
        j = 0
        while j < mom.len() {
            mom[j] = mom[j] + 0.5 * step_size * new_grad[j]
            j = j + 1
        }
        i = i + 1
    }

    // Final Hamiltonian
    let new_log_prob = log_prob_fn(&pos)
    let final_h = -new_log_prob + 0.5 * dot_product(mom, mom)

    // MH acceptance
    let log_accept = initial_h - final_h
    let accept = ln_f64(sample_uniform(0.0, 1.0)) < log_accept

    if accept {
        return (pos, new_log_prob, true)
    } else {
        return (position, log_prob, false)
    }
}
```

## NUTS (No-U-Turn Sampler)

NUTS automatically tunes the number of leapfrog steps, eliminating a key HMC tuning parameter.

### The U-Turn Criterion

NUTS builds a binary tree of leapfrog steps, stopping when the trajectory begins to "turn around":

```sio
use std::epistemic::mcmc::{
    NUTSOptions, nuts_sample, nuts_options_default, nuts_options_with_target
}

/// Check for U-turn: trajectory reversing direction
fn check_uturn(
    pos_backward: [f64],
    pos_forward: [f64],
    mom_backward: [f64],
    mom_forward: [f64]
) -> bool {
    // Compute delta position
    var delta: [f64] = []
    var j: i64 = 0
    while j < pos_forward.len() {
        delta = delta ++ [pos_forward[j] - pos_backward[j]]
        j = j + 1
    }

    // U-turn if either momentum is pointing "backward"
    let forward_dot = dot_product(delta, mom_forward)
    let backward_dot = dot_product(delta, mom_backward)

    return forward_dot < 0.0 || backward_dot < 0.0
}
```

### Using NUTS in Sounio

```sio
use std::epistemic::mcmc::{
    sample, NUTSOptions, nuts_options_default,
    PosteriorSummary, check_convergence, print_diagnostics
}

fn bayesian_inference_example() -> i32 with Prob, Alloc, IO {
    // Define log posterior (unnormalized)
    fn log_posterior(theta: &[f64]) -> f64 {
        let mu = theta[0]
        let log_sigma = theta[1]
        let sigma = exp_f64(log_sigma)

        // Prior: mu ~ N(0, 10), log_sigma ~ N(0, 2)
        let log_prior = -0.5 * (mu * mu / 100.0 + log_sigma * log_sigma / 4.0)

        // Likelihood: data ~ N(mu, sigma)
        let data: [f64] = [4.5, 5.2, 4.8, 5.5, 4.9, 5.1]
        var log_lik = 0.0
        var i: i64 = 0
        while i < data.len() {
            let diff = data[i] - mu
            log_lik = log_lik - log_sigma - 0.5 * diff * diff / (sigma * sigma)
            i = i + 1
        }

        return log_prior + log_lik
    }

    // Gradient (computed or automatic differentiation)
    fn gradient(theta: &[f64]) -> [f64] {
        // Numerical gradient for simplicity
        let eps = 0.0001
        var grad: [f64] = []
        var j: i64 = 0
        while j < theta.len() {
            var theta_plus = theta
            var theta_minus = theta
            theta_plus[j] = theta_plus[j] + eps
            theta_minus[j] = theta_minus[j] - eps
            let g = (log_posterior(&theta_plus) - log_posterior(&theta_minus)) / (2.0 * eps)
            grad = grad ++ [g]
            j = j + 1
        }
        return grad
    }

    print("=== Bayesian Inference with NUTS ===\n\n")

    // Configure NUTS
    let opts = nuts_options_default()
    // opts.target_accept = 0.8  // Target acceptance rate
    // opts.max_tree_depth = 10  // Maximum tree depth

    // Run sampler (4 chains, 1000 samples each)
    let summary = sample(log_posterior, gradient, 4, 1000)

    // Check convergence
    print("Checking convergence...\n")
    print_diagnostics(&summary)

    if check_convergence(&summary) {
        print("\nAll chains converged!\n\n")
    } else {
        print("\nWarning: Some chains may not have converged.\n\n")
    }

    // Report results
    print("Posterior Summary:\n")
    print("  mu: ")
    print(summary.parameters[0].mean.value)
    print(" +/- ")
    print(summary.parameters[0].std.value)
    print("\n    95% CI: [")
    print(summary.parameters[0].q025)
    print(", ")
    print(summary.parameters[0].q975)
    print("]\n")

    print("  log_sigma: ")
    print(summary.parameters[1].mean.value)
    print(" +/- ")
    print(summary.parameters[1].std.value)
    print("\n")

    return 0
}
```

## Convergence Diagnostics

### R-hat (Gelman-Rubin Statistic)

R-hat compares within-chain and between-chain variance:

```sio
/// Compute R-hat for a parameter across chains
fn compute_rhat(chains: [[f64]]) -> f64 {
    let m = chains.len() as f64      // Number of chains
    let n = chains[0].len() as f64   // Samples per chain

    // Compute chain means
    var chain_means: [f64] = []
    var c: i64 = 0
    while c < chains.len() {
        var sum = 0.0
        var i: i64 = 0
        while i < chains[c].len() {
            sum = sum + chains[c][i]
            i = i + 1
        }
        chain_means = chain_means ++ [sum / n]
        c = c + 1
    }

    // Overall mean
    var grand_mean = 0.0
    c = 0
    while c < chain_means.len() {
        grand_mean = grand_mean + chain_means[c]
        c = c + 1
    }
    grand_mean = grand_mean / m

    // Between-chain variance B
    var B = 0.0
    c = 0
    while c < chain_means.len() {
        let diff = chain_means[c] - grand_mean
        B = B + diff * diff
        c = c + 1
    }
    B = n * B / (m - 1.0)

    // Within-chain variance W
    var W = 0.0
    c = 0
    while c < chains.len() {
        var chain_var = 0.0
        var i: i64 = 0
        while i < chains[c].len() {
            let diff = chains[c][i] - chain_means[c]
            chain_var = chain_var + diff * diff
            i = i + 1
        }
        W = W + chain_var / (n - 1.0)
        c = c + 1
    }
    W = W / m

    // Pooled variance estimate
    let var_plus = (n - 1.0) / n * W + B / n

    // R-hat
    return sqrt_f64(var_plus / W)
}

fn interpret_rhat(rhat: f64) with IO {
    if rhat < 1.01 {
        print("R-hat < 1.01: Excellent convergence\n")
    } else if rhat < 1.05 {
        print("R-hat < 1.05: Acceptable convergence\n")
    } else if rhat < 1.1 {
        print("R-hat < 1.1: Marginal convergence, consider more samples\n")
    } else {
        print("R-hat >= 1.1: Chains have NOT converged!\n")
    }
}
```

### Effective Sample Size (ESS)

ESS accounts for autocorrelation in MCMC samples:

```sio
/// Compute effective sample size
fn compute_ess(samples: [f64]) -> f64 {
    let n = samples.len() as f64

    // Compute mean
    var mean = 0.0
    var i: i64 = 0
    while i < samples.len() {
        mean = mean + samples[i]
        i = i + 1
    }
    mean = mean / n

    // Compute variance
    var var_sum = 0.0
    i = 0
    while i < samples.len() {
        let diff = samples[i] - mean
        var_sum = var_sum + diff * diff
        i = i + 1
    }
    let variance = var_sum / (n - 1.0)

    // Compute autocorrelation sum
    var sum_rho = 0.0
    let max_lag = min_i64(100, (samples.len() / 2) as i64)

    var lag: i64 = 1
    while lag < max_lag {
        var cov = 0.0
        i = 0
        while i < samples.len() - lag {
            cov = cov + (samples[i] - mean) * (samples[i + lag] - mean)
            i = i + 1
        }
        let rho = cov / var_sum

        // Stop when autocorrelation becomes negligible
        if rho < 0.05 {
            break
        }
        sum_rho = sum_rho + rho
        lag = lag + 1
    }

    // ESS = n / (1 + 2 * sum of autocorrelations)
    return n / (1.0 + 2.0 * sum_rho)
}

fn interpret_ess(ess: f64, n_samples: i64) with IO {
    let efficiency = ess / (n_samples as f64)
    print("ESS: ")
    print(ess)
    print(" (efficiency: ")
    print(efficiency * 100.0)
    print("%)\n")

    if ess < 100.0 {
        print("Warning: Very low ESS. Consider:\n")
        print("  - Running longer chains\n")
        print("  - Reparameterizing the model\n")
        print("  - Using a better sampler\n")
    } else if ess < 400.0 {
        print("Note: ESS is moderate. Results may have elevated MCSE.\n")
    } else {
        print("ESS is adequate for reliable inference.\n")
    }
}
```

## Posterior Summaries with Epistemic Types

Sounio produces posterior summaries as epistemic values:

```sio
use std::epistemic::{Knowledge, Confidence, Provenance}

/// Parameter summary with full uncertainty tracking
struct ParameterSummary {
    mean: Knowledge<f64>,     // Mean with MCSE uncertainty
    std: Knowledge<f64>,      // Std with uncertainty
    median: Knowledge<f64>,   // Median with uncertainty
    q025: f64,                // 2.5th percentile
    q975: f64,                // 97.5th percentile
    hdi_low: f64,             // 94% HDI lower
    hdi_high: f64,            // 94% HDI upper
    rhat: f64,                // Convergence diagnostic
    ess_bulk: f64,            // Bulk ESS
    ess_tail: f64,            // Tail ESS
    mcse_mean: f64,           // Monte Carlo SE of mean
}

/// Compute MCSE (Monte Carlo Standard Error)
fn compute_mcse(samples: [f64], ess: f64) -> f64 {
    let n = samples.len() as f64

    // Compute sample std
    var mean = 0.0
    var i: i64 = 0
    while i < samples.len() {
        mean = mean + samples[i]
        i = i + 1
    }
    mean = mean / n

    var var_sum = 0.0
    i = 0
    while i < samples.len() {
        let diff = samples[i] - mean
        var_sum = var_sum + diff * diff
        i = i + 1
    }
    let std = sqrt_f64(var_sum / (n - 1.0))

    // MCSE = std / sqrt(ESS)
    return std / sqrt_f64(ess)
}

fn posterior_mean_to_knowledge(
    samples: [f64],
    ess: f64,
    rhat: f64,
    n_chains: i64,
    n_samples: i64
) -> Knowledge<f64> {
    // Compute mean
    var mean = 0.0
    var i: i64 = 0
    while i < samples.len() {
        mean = mean + samples[i]
        i = i + 1
    }
    mean = mean / (samples.len() as f64)

    // MCSE
    let mcse = compute_mcse(samples, ess)

    Knowledge {
        value: mean,
        variance: mcse * mcse,  // MCSE squared is variance of the estimator
        confidence: Confidence::Bayesian {
            prior_weight: 0.1,
            data_weight: 0.9,
        },
        provenance: Provenance::MCMC {
            sampler: "NUTS",
            chains: n_chains as i32,
            samples: n_samples as i32,
            rhat: rhat,
        },
    }
}
```

## Complete Example: Hierarchical Model

```sio
use std::epistemic::mcmc::{
    sample, NUTSOptions, nuts_options_default,
    PosteriorSummary, print_diagnostics
}

fn hierarchical_inference() -> i32 with Prob, Alloc, IO {
    print("=== Hierarchical Bayesian Model ===\n\n")

    // Data: scores from 5 schools
    let school_data: [([f64], i64)] = [
        ([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], 0),
        ([12.0, 7.0, 8.0, 10.0, 5.0], 1),
        ([15.0, 11.0, 9.0, 13.0, 8.0, 10.0], 2),
        ([8.0, 6.0, 7.0, 10.0, 9.0, 8.0, 11.0], 3),
        ([10.0, 9.0, 11.0, 8.0, 10.0], 4),
    ]
    let n_schools: i64 = 5

    // Parameters: [mu, log_tau, theta_1, ..., theta_5]
    // mu: global mean
    // tau: between-school std
    // theta_j: school j effect

    fn log_posterior(params: &[f64]) -> f64 {
        let mu = params[0]
        let log_tau = params[1]
        let tau = exp_f64(log_tau)

        // Hyperprior
        let log_prior = -0.5 * mu * mu / 100.0  // mu ~ N(0, 10)
        let log_prior = log_prior - 0.5 * log_tau * log_tau / 4.0  // log_tau ~ N(0, 2)

        // Group-level priors
        var log_group_prior = 0.0
        var j: i64 = 0
        while j < 5 {
            let theta_j = params[2 + j]
            let diff = theta_j - mu
            log_group_prior = log_group_prior - 0.5 * diff * diff / (tau * tau) - log_tau
            j = j + 1
        }

        // Likelihood (simplified - using school means)
        let school_means: [f64] = [8.75, 8.4, 11.0, 8.43, 9.6]
        let school_se: [f64] = [3.2, 2.8, 2.5, 2.9, 2.6]
        var log_lik = 0.0
        j = 0
        while j < 5 {
            let theta_j = params[2 + j]
            let diff = school_means[j] - theta_j
            log_lik = log_lik - 0.5 * diff * diff / (school_se[j] * school_se[j])
            j = j + 1
        }

        return log_prior + log_group_prior + log_lik
    }

    fn gradient(params: &[f64]) -> [f64] {
        // Numerical gradient
        let eps = 0.0001
        var grad: [f64] = []
        var j: i64 = 0
        while j < params.len() {
            var plus = params
            var minus = params
            plus[j] = plus[j] + eps
            minus[j] = minus[j] - eps
            grad = grad ++ [(log_posterior(&plus) - log_posterior(&minus)) / (2.0 * eps)]
            j = j + 1
        }
        return grad
    }

    // Run NUTS
    print("Running NUTS sampler (4 chains, 2000 samples)...\n\n")
    let summary = sample(log_posterior, gradient, 4, 2000)

    // Report results
    print_diagnostics(&summary)

    print("\nHyperparameters:\n")
    print("  mu (global mean): ")
    print(summary.parameters[0].mean.value)
    print(" [")
    print(summary.parameters[0].q025)
    print(", ")
    print(summary.parameters[0].q975)
    print("]\n")

    print("  tau (between-school std): ")
    let tau_mean = exp_f64(summary.parameters[1].mean.value)
    print(tau_mean)
    print("\n\n")

    print("School Effects:\n")
    var j: i64 = 0
    while j < 5 {
        print("  School ")
        print(j + 1)
        print(": ")
        print(summary.parameters[2 + j].mean.value)
        print(" [")
        print(summary.parameters[2 + j].q025)
        print(", ")
        print(summary.parameters[2 + j].q975)
        print("]\n")
        j = j + 1
    }

    return 0
}

fn main() -> i32 {
    return hierarchical_inference()
}
```

## Summary

| Sampler | Requires Gradient | Tuning | Best For |
|---------|------------------|--------|----------|
| Metropolis-Hastings | No | Proposal scale | Low-dimensional, simple models |
| HMC | Yes | Step size, # steps | High-dimensional, continuous |
| NUTS | Yes | Step size (auto-tuned) | General purpose, production |

| Diagnostic | Target | Interpretation |
|------------|--------|----------------|
| R-hat | < 1.01 | Chains have mixed/converged |
| ESS bulk | > 400 | Enough samples for means |
| ESS tail | > 400 | Enough samples for quantiles |
| Divergences | 0 | No numerical issues |

Key takeaways:

1. **NUTS is the default** - Use NUTS for most problems
2. **Run multiple chains** - 4 chains is standard for diagnosing convergence
3. **Check diagnostics** - R-hat and ESS are essential
4. **Warmup matters** - Use at least 500-1000 warmup iterations
5. **Epistemic tracking** - Sounio propagates uncertainty through the entire inference pipeline
