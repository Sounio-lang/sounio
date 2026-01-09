# Bayesian Computing in Sounio

Sounio provides comprehensive support for Bayesian inference and probabilistic programming, integrating seamlessly with its epistemic type system. Every posterior distribution, every credible interval, every model comparison carries quantified uncertainty.

## Bayesian Inference Overview

Bayesian inference provides a principled framework for updating beliefs given data:

```
Posterior = (Likelihood x Prior) / Evidence
P(theta | data) = P(data | theta) P(theta) / P(data)
```

### Why Bayesian?

1. **Principled uncertainty quantification** - Posteriors represent full uncertainty, not point estimates
2. **Prior knowledge incorporation** - Domain expertise naturally integrates
3. **Coherent decision making** - Bayesian decision theory minimizes expected loss
4. **Small sample validity** - Works even with limited data (with appropriate priors)
5. **Model comparison** - Bayes factors and model evidence for principled selection

### The Bayesian Workflow

```sio
// 1. Specify prior
let prior = Normal { mu: 0.0, sigma: 10.0 }

// 2. Specify likelihood (data model)
let likelihood = Normal { mu: theta, sigma: known_sigma }

// 3. Observe data
let data: [f64] = [4.5, 5.2, 4.8, 5.5, 4.9]

// 4. Compute posterior (via conjugacy, MCMC, or VI)
let posterior = infer_posterior(prior, likelihood, data)

// 5. Summarize and decide
let mean = posterior.mean
let ci_95 = posterior.credible_interval(0.95)
```

## The Prob Effect

Sounio tracks probabilistic computations through the `Prob` effect:

```sio
/// Sample from a distribution
fn sample_normal(mu: f64, sigma: f64) -> f64 with Prob {
    // Sampling requires the Prob effect
    let z = sample_standard_normal()
    return mu + sigma * z
}

/// Observe data (condition on likelihood)
fn observe(dist: Distribution, value: f64) -> f64 with Prob {
    // Returns log probability contribution
    return dist.log_pdf(value)
}

/// A simple probabilistic program
fn coin_bias_model(flips: [bool]) -> f64 with Prob {
    // Prior: uniform over [0, 1]
    let p = sample_beta(1.0, 1.0)

    // Likelihood: each flip is Bernoulli(p)
    var i: i64 = 0
    while i < flips.len() {
        observe(Bernoulli { p: p }, flips[i] as f64)
        i = i + 1
    }

    return p
}
```

### Effect Composition

The `Prob` effect composes with other Sounio effects:

```sio
/// Probabilistic computation with memory allocation
fn sample_many(n: i64) -> [f64] with Prob, Alloc {
    var samples: [f64] = []
    var i: i64 = 0
    while i < n {
        let x = sample_normal(0.0, 1.0)
        samples = samples ++ [x]
        i = i + 1
    }
    return samples
}

/// Probabilistic I/O
fn report_posterior(posterior: Posterior) with Prob, IO {
    print("Posterior mean: ")
    print(posterior.mean)
    print("\n95% CI: [")
    print(posterior.ci_lower)
    print(", ")
    print(posterior.ci_upper)
    print("]\n")
}
```

## Prior, Likelihood, Posterior

### Prior Distributions

Priors encode beliefs before seeing data:

```sio
use std::prob::{Normal, Beta, Gamma, Uniform}

// Uninformative priors
let uniform_prior = Uniform { a: 0.0, b: 1.0 }          // No preference
let vague_normal = Normal { mu: 0.0, sigma: 100.0 }    // Wide normal

// Informative priors (encode domain knowledge)
let expert_prior = Normal { mu: 5.0, sigma: 1.0 }      // Expert believes mu ~ 5
let success_rate = Beta { alpha: 8.0, beta: 2.0 }      // Prior: ~80% success

// Weakly informative priors (regularization)
let regularizing = Normal { mu: 0.0, sigma: 2.5 }      // Shrinkage toward 0
```

### Likelihood Functions

The likelihood connects parameters to data:

```sio
/// Log-likelihood for Normal data
fn normal_log_likelihood(
    data: [f64],
    mu: f64,
    sigma: f64
) -> f64 {
    var log_lik = 0.0
    var i: i64 = 0
    while i < data.len() {
        let x = data[i]
        let diff = x - mu
        log_lik = log_lik - 0.5 * ln_f64(2.0 * pi())
        log_lik = log_lik - ln_f64(sigma)
        log_lik = log_lik - (diff * diff) / (2.0 * sigma * sigma)
        i = i + 1
    }
    return log_lik
}

/// Log-likelihood for Bernoulli data
fn bernoulli_log_likelihood(
    data: [bool],
    p: f64
) -> f64 {
    var log_lik = 0.0
    var i: i64 = 0
    while i < data.len() {
        if data[i] {
            log_lik = log_lik + ln_f64(p + 0.0000001)
        } else {
            log_lik = log_lik + ln_f64(1.0 - p + 0.0000001)
        }
        i = i + 1
    }
    return log_lik
}
```

### Posterior Computation

#### Conjugate Priors

When prior and likelihood are conjugate, posteriors are analytical:

```sio
use std::prob::{Beta, Normal, beta_bernoulli_posterior, normal_normal_posterior}

/// Beta-Bernoulli conjugate posterior
fn coin_posterior(
    prior_alpha: f64,
    prior_beta: f64,
    n_heads: i64,
    n_tails: i64
) -> Beta {
    // Posterior is Beta(alpha + heads, beta + tails)
    Beta {
        alpha: prior_alpha + (n_heads as f64),
        beta_param: prior_beta + (n_tails as f64),
    }
}

/// Normal-Normal conjugate posterior (known variance)
fn mean_posterior(
    prior_mu: f64,
    prior_sigma: f64,
    data_mean: f64,
    data_sigma: f64,
    n: i64
) -> NormalPosterior {
    let prior_prec = 1.0 / (prior_sigma * prior_sigma)
    let lik_prec = (n as f64) / (data_sigma * data_sigma)

    let post_prec = prior_prec + lik_prec
    let post_var = 1.0 / post_prec
    let post_mu = post_var * (prior_prec * prior_mu + lik_prec * data_mean)

    NormalPosterior {
        mu: post_mu,
        sigma: sqrt_f64(post_var),
    }
}
```

#### Non-Conjugate Cases

For non-conjugate models, use MCMC or variational inference:

```sio
use std::epistemic::mcmc::{nuts_sample, summarize_posterior}
use std::prob::inference::{importance_sample, mh_sample}

// See mcmc.md and variational-inference.md for details
```

## Probabilistic Programming in Sounio

### Defining Models

```sio
/// Bayesian linear regression model
fn linear_regression_model(
    X: [[f64]],      // Features (n x p)
    y: [f64],        // Outcomes (n)
    sigma: f64       // Known noise std
) -> [f64] with Prob {
    let n = y.len()
    let p = X[0].len()

    // Priors on coefficients
    var beta: [f64] = []
    var j: i64 = 0
    while j < p {
        let b = sample_normal(0.0, 10.0)  // Vague prior
        beta = beta ++ [b]
        j = j + 1
    }

    // Likelihood
    var i: i64 = 0
    while i < n {
        // Compute predicted value
        var mu = 0.0
        j = 0
        while j < p {
            mu = mu + beta[j] * X[i][j]
            j = j + 1
        }

        // Observe
        observe(Normal { mu: mu, sigma: sigma }, y[i])
        i = i + 1
    }

    return beta
}

/// Hierarchical model
fn hierarchical_model(
    group_data: [([f64], i64)],  // (data, group_id) pairs
    n_groups: i64
) -> (f64, f64, [f64]) with Prob {
    // Hyperpriors
    let global_mu = sample_normal(0.0, 10.0)
    let global_sigma = sample_gamma(2.0, 0.5)

    // Group-level parameters
    var group_means: [f64] = []
    var g: i64 = 0
    while g < n_groups {
        let mu_g = sample_normal(global_mu, global_sigma)
        group_means = group_means ++ [mu_g]
        g = g + 1
    }

    // Data likelihood
    var i: i64 = 0
    while i < group_data.len() {
        let data = group_data[i].0
        let group = group_data[i].1
        var j: i64 = 0
        while j < data.len() {
            observe(Normal { mu: group_means[group], sigma: 1.0 }, data[j])
            j = j + 1
        }
        i = i + 1
    }

    return (global_mu, global_sigma, group_means)
}
```

### Running Inference

```sio
use std::epistemic::mcmc::{sample, PosteriorSummary, check_convergence}

fn run_bayesian_analysis() -> i32 with Prob, Alloc, IO {
    // Define data
    let X: [[f64]] = [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
    let y: [f64] = [2.5, 3.8, 5.1, 6.2]

    // Define log posterior
    fn log_posterior(beta: [f64]) -> f64 {
        // Prior
        var log_prior = 0.0
        var j: i64 = 0
        while j < beta.len() {
            log_prior = log_prior + logpdf_normal(Normal { mu: 0.0, sigma: 10.0 }, beta[j])
            j = j + 1
        }

        // Likelihood
        let log_lik = normal_regression_log_lik(X, y, beta, 1.0)

        return log_prior + log_lik
    }

    // Run NUTS sampler
    print("Running MCMC inference...\n")
    let summary = sample(log_posterior, gradient_fn, 4, 1000)  // 4 chains, 1000 samples

    // Check convergence
    if check_convergence(&summary) {
        print("Convergence achieved!\n\n")
    } else {
        print("Warning: Chains may not have converged\n\n")
    }

    // Report results
    print("Posterior Summary:\n")
    var p: i64 = 0
    while p < summary.parameters.len() {
        let param = summary.parameters[p]
        print("  beta[")
        print(p)
        print("]: ")
        print(param.mean.value)
        print(" +/- ")
        print(param.std.value)
        print(" [")
        print(param.q025)
        print(", ")
        print(param.q975)
        print("]\n")
        print("    R-hat: ")
        print(param.rhat)
        print(", ESS: ")
        print(param.ess_bulk)
        print("\n")
        p = p + 1
    }

    return 0
}
```

## Key Modules

### std::prob

Core probability distributions and operations:

```sio
use std::prob::{
    // Distributions
    Normal, Uniform, Bernoulli, Beta, Gamma, Poisson, Exponential,

    // Distribution creation
    normal_new, uniform_new, bernoulli_new, beta_new, gamma_new,

    // Sampling
    sample_normal, sample_uniform, sample_bernoulli, sample_beta,

    // Density functions
    logpdf_normal, logpdf_uniform, logpdf_bernoulli, logpdf_beta,

    // Observe (conditioning)
    observe_normal, observe_uniform, observe_bernoulli,

    // Random number generation
    RNG, rng_new, rng_uniform, rng_normal,
}
```

### std::epistemic::mcmc

Markov Chain Monte Carlo samplers:

```sio
use std::epistemic::mcmc::{
    // Samplers
    nuts_sample, mh_sample, hmc_sample,

    // Configuration
    NUTSOptions, MHOptions, HMCOptions,
    nuts_options_default, mh_options_default,

    // Chain management
    Chain, ChainCollection, nuts_sample_chains,

    // Diagnostics
    compute_rhat, compute_ess_bulk, compute_ess_tail,

    // Posterior analysis
    PosteriorSummary, ParameterSummary, summarize_posterior,
    check_convergence, print_diagnostics,

    // High-level API
    sample, sample_with_options,
}
```

### std::prob::inference

Inference algorithms:

```sio
use std::prob::inference::{
    // Rejection sampling
    rejection_sample, RejectionConfig,

    // Importance sampling
    importance_sample, ImportanceConfig, effective_sample_size,

    // Weighted samples
    weighted_mean, weighted_variance, SampleCollection,

    // Model comparison
    waic, loo_cv, bayes_factor,
}
```

## Integration with Epistemic Types

Bayesian posteriors naturally produce `Knowledge<T>` values:

```sio
use std::epistemic::{Knowledge, Confidence, Provenance}

/// Convert posterior summary to Knowledge
fn posterior_to_knowledge(summary: ParameterSummary) -> Knowledge<f64> {
    Knowledge {
        value: summary.mean.value,
        variance: summary.std.value * summary.std.value,
        confidence: Confidence::Bayesian {
            prior_weight: 0.1,
            data_weight: 0.9,
        },
        provenance: Provenance::MCMC {
            sampler: "NUTS",
            chains: 4,
            samples: 1000,
            rhat: summary.rhat,
        },
    }
}

/// Use in downstream computation
fn make_decision(posterior: Knowledge<f64>, threshold: f64) -> Decision {
    // Posterior includes full uncertainty
    if posterior.confidence_above(threshold, 0.95) {
        Decision::Accept
    } else if posterior.confidence_below(threshold, 0.95) {
        Decision::Reject
    } else {
        Decision::Uncertain
    }
}
```

## Learning Path

### Beginner

1. Understanding priors, likelihoods, and posteriors
2. Conjugate models (Beta-Binomial, Normal-Normal)
3. Basic MCMC concepts

### Intermediate

4. [MCMC Methods](mcmc.md) - Metropolis-Hastings, HMC, NUTS
5. Model checking and diagnostics
6. Hierarchical models

### Advanced

7. [Variational Inference](variational-inference.md) - ELBO, mean-field, ADVI
8. Model comparison and selection
9. Advanced probabilistic programming patterns

## References

- Gelman, A., et al. (2013). *Bayesian Data Analysis*. 3rd ed. Chapman & Hall/CRC.
- McElreath, R. (2020). *Statistical Rethinking*. 2nd ed. CRC Press.
- Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo."
- van de Meent, J.W., et al. (2018). "An Introduction to Probabilistic Programming."
- Blei, D.M., Kucukelbir, A., & McAuliffe, J.D. (2017). "Variational Inference: A Review for Statisticians."
