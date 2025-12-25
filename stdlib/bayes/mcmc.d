// bayes::mcmc — Markov Chain Monte Carlo Sampling
//
// Metropolis-Hastings and related MCMC algorithms for posterior sampling.
// Foundation for Bayesian inference in Demetrios.
//
// References:
// - Metropolis et al. (1953): "Equation of state calculations..."
// - Hastings (1970): "Monte Carlo sampling methods using Markov chains"
// - Gelman & Rubin (1992): "Inference from iterative simulation"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// MCMC CONFIGURATION
// ============================================================================

/// MCMC sampler configuration
struct MCMCConfig {
    n_samples: i64,     // Number of posterior samples
    n_warmup: i64,      // Warmup/burn-in samples
    n_thin: i64,        // Thinning interval
    proposal_sd: f64,   // Proposal distribution std dev
    adapt_proposal: bool, // Adapt proposal during warmup
    seed: i64,          // Random seed
}

fn mcmc_config_default() -> MCMCConfig {
    MCMCConfig {
        n_samples: 1000,
        n_warmup: 500,
        n_thin: 1,
        proposal_sd: 1.0,
        adapt_proposal: true,
        seed: 12345,
    }
}

/// MCMC chain result
struct MCMCChain {
    samples: [f64; 100],    // Posterior samples
    n_samples: i64,         // Number of samples stored
    acceptance_rate: f64,   // Proportion accepted
    mean: f64,              // Posterior mean
    std: f64,               // Posterior std dev
    median: f64,            // Posterior median
    ci_lower: f64,          // 95% credible interval lower
    ci_upper: f64,          // 95% credible interval upper
}

fn mcmc_chain_new() -> MCMCChain {
    MCMCChain {
        samples: [0.0; 100],
        n_samples: 0,
        acceptance_rate: 0.0,
        mean: 0.0,
        std: 0.0,
        median: 0.0,
        ci_lower: 0.0,
        ci_upper: 0.0,
    }
}

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/// LCG random state
struct RNG {
    seed: i64,
    last: f64,
}

fn rng_new(seed: i64) -> RNG {
    RNG { seed: seed, last: 0.0 }
}

/// Generate uniform random in [0, 1)
fn rng_uniform(rng: RNG) -> RNG {
    let a: i64 = 1103515245
    let c: i64 = 12345
    let m: i64 = 2147483648

    let new_seed = (a * rng.seed + c) % m
    let u = (new_seed as f64) / (m as f64)
    RNG { seed: new_seed, last: u }
}

/// Generate standard normal using Box-Muller
fn rng_normal(rng: RNG) -> RNG {
    var r = rng

    // Generate two uniforms
    r = rng_uniform(r)
    var u1 = r.last
    if u1 < 1e-10 { u1 = 1e-10 }

    r = rng_uniform(r)
    let u2 = r.last

    // Box-Muller transform
    let z = sqrt(-2.0 * log(u1)) * cos_approx(2.0 * pi() * u2)
    RNG { seed: r.seed, last: z }
}

/// Cosine approximation
fn cos_approx(x: f64) -> f64 {
    // Reduce to [0, 2*pi]
    var t = x
    let two_pi = 2.0 * pi()
    while t < 0.0 { t = t + two_pi }
    while t >= two_pi { t = t - two_pi }

    // Taylor series approximation
    let x2 = t * t
    let x4 = x2 * x2
    let x6 = x4 * x2
    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0
}

// ============================================================================
// LOG-POSTERIOR FUNCTION TYPE
// ============================================================================

/// Simple log-posterior for single parameter models
/// Takes parameter value, returns log-posterior (up to constant)
struct LogPosterior {
    // For demonstration: Normal likelihood with Normal prior
    // Data mean and variance
    data_mean: f64,
    data_var: f64,
    data_n: i64,
    // Prior parameters
    prior_mean: f64,
    prior_var: f64,
}

fn log_posterior_new(data_mean: f64, data_var: f64, data_n: i64, prior_mean: f64, prior_var: f64) -> LogPosterior {
    LogPosterior {
        data_mean: data_mean,
        data_var: data_var,
        data_n: data_n,
        prior_mean: prior_mean,
        prior_var: prior_var,
    }
}

/// Compute log-posterior at theta
fn log_posterior_compute(lp: LogPosterior, theta: f64) -> f64 {
    // Log-likelihood: sum of log N(x_i | theta, sigma^2)
    // = -n/2 * log(2*pi*sigma^2) - sum((x_i - theta)^2) / (2*sigma^2)
    // Sufficient statistic form: -n * (theta - x_bar)^2 / (2*sigma^2)
    let ll = -(lp.data_n as f64) * (theta - lp.data_mean) * (theta - lp.data_mean) / (2.0 * lp.data_var)

    // Log-prior: log N(theta | prior_mean, prior_var)
    let lpr = -(theta - lp.prior_mean) * (theta - lp.prior_mean) / (2.0 * lp.prior_var)

    ll + lpr
}

// ============================================================================
// METROPOLIS-HASTINGS SAMPLER
// ============================================================================

/// Run Metropolis-Hastings sampler
fn metropolis_hastings(lp: LogPosterior, config: MCMCConfig) -> MCMCChain {
    var chain = mcmc_chain_new()
    var rng = rng_new(config.seed)

    // Initialize at prior mean
    var theta = lp.prior_mean
    var log_p_current = log_posterior_compute(lp, theta)

    var n_accepted: i64 = 0
    var n_total: i64 = 0
    var proposal_sd = config.proposal_sd

    // Total iterations
    let total_iter = config.n_warmup + config.n_samples * config.n_thin
    var sample_idx: i64 = 0

    var iter: i64 = 0
    while iter < total_iter {
        // Propose new value
        rng = rng_normal(rng)
        let theta_proposed = theta + proposal_sd * rng.last

        // Compute acceptance probability
        let log_p_proposed = log_posterior_compute(lp, theta_proposed)
        let log_alpha = log_p_proposed - log_p_current

        // Accept/reject
        rng = rng_uniform(rng)
        let u = rng.last

        if log(u) < log_alpha {
            theta = theta_proposed
            log_p_current = log_p_proposed
            n_accepted = n_accepted + 1
        }
        n_total = n_total + 1

        // Adapt proposal during warmup
        if iter < config.n_warmup && config.adapt_proposal && iter > 0 && iter % 50 == 0 {
            let accept_rate = (n_accepted as f64) / (n_total as f64)
            if accept_rate < 0.2 {
                proposal_sd = proposal_sd * 0.8
            } else if accept_rate > 0.5 {
                proposal_sd = proposal_sd * 1.2
            }
        }

        // Store sample after warmup
        if iter >= config.n_warmup && (iter - config.n_warmup) % config.n_thin == 0 {
            if sample_idx < 100 {
                chain.samples[sample_idx as usize] = theta
                sample_idx = sample_idx + 1
            }
        }

        iter = iter + 1
    }

    chain.n_samples = sample_idx
    chain.acceptance_rate = (n_accepted as f64) / (n_total as f64)

    // Compute posterior summaries
    chain = compute_chain_summaries(chain)

    chain
}

// ============================================================================
// CHAIN SUMMARIES
// ============================================================================

/// Compute summary statistics for MCMC chain
fn compute_chain_summaries(chain: MCMCChain) -> MCMCChain {
    var result = chain
    let n = chain.n_samples

    if n == 0 {
        return result
    }

    // Mean
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + chain.samples[i as usize]
        i = i + 1
    }
    result.mean = sum / (n as f64)

    // Std dev
    var sum_sq = 0.0
    i = 0
    while i < n {
        let diff = chain.samples[i as usize] - result.mean
        sum_sq = sum_sq + diff * diff
        i = i + 1
    }
    result.std = sqrt(sum_sq / ((n - 1) as f64))

    // Sort for quantiles
    var sorted: [f64; 100] = chain.samples
    i = 0
    while i < n - 1 {
        var j: i64 = 0
        while j < n - 1 - i {
            if sorted[j as usize] > sorted[(j + 1) as usize] {
                let temp = sorted[j as usize]
                sorted[j as usize] = sorted[(j + 1) as usize]
                sorted[(j + 1) as usize] = temp
            }
            j = j + 1
        }
        i = i + 1
    }

    // Median
    if n % 2 == 1 {
        result.median = sorted[(n / 2) as usize]
    } else {
        result.median = (sorted[(n / 2 - 1) as usize] + sorted[(n / 2) as usize]) / 2.0
    }

    // 95% credible interval (2.5% and 97.5% quantiles)
    let lower_idx = ((n as f64) * 0.025) as i64
    let upper_idx = ((n as f64) * 0.975) as i64
    result.ci_lower = sorted[lower_idx as usize]
    result.ci_upper = sorted[upper_idx as usize]

    result
}

// ============================================================================
// ANALYTICAL POSTERIOR (FOR COMPARISON)
// ============================================================================

/// Compute analytical posterior for normal-normal model
/// Prior: theta ~ N(mu0, sigma0^2)
/// Likelihood: x_bar ~ N(theta, sigma^2/n)
/// Posterior: theta | data ~ N(mu_post, sigma_post^2)
struct AnalyticalPosterior {
    mean: f64,
    variance: f64,
}

fn normal_normal_posterior(prior_mean: f64, prior_var: f64,
                           data_mean: f64, data_var: f64, data_n: i64) -> AnalyticalPosterior {
    let prior_prec = 1.0 / prior_var
    let lik_prec = (data_n as f64) / data_var

    let post_prec = prior_prec + lik_prec
    let post_var = 1.0 / post_prec
    let post_mean = post_var * (prior_prec * prior_mean + lik_prec * data_mean)

    AnalyticalPosterior { mean: post_mean, variance: post_var }
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_metropolis_simple() -> bool {
    // Normal-normal conjugate model
    // Data: mean = 5.0, var = 1.0, n = 10
    // Prior: mean = 0.0, var = 10.0

    let lp = log_posterior_new(5.0, 1.0, 10, 0.0, 10.0)
    var config = mcmc_config_default()
    config.n_samples = 100
    config.n_warmup = 100

    let chain = metropolis_hastings(lp, config)

    // Posterior mean should be close to data mean (strong likelihood)
    // Analytical: post_prec = 0.1 + 10 = 10.1
    // post_mean = (0.1 * 0 + 10 * 5) / 10.1 = 4.95
    abs_f64(chain.mean - 4.95) < 0.5 && chain.acceptance_rate > 0.1
}

fn test_acceptance_rate() -> bool {
    let lp = log_posterior_new(0.0, 1.0, 10, 0.0, 1.0)
    var config = mcmc_config_default()
    config.n_samples = 100
    config.n_warmup = 200

    let chain = metropolis_hastings(lp, config)

    // Acceptance rate should be reasonable (20-50%)
    chain.acceptance_rate > 0.15 && chain.acceptance_rate < 0.7
}

fn test_credible_interval() -> bool {
    let lp = log_posterior_new(0.0, 1.0, 100, 0.0, 10.0)
    var config = mcmc_config_default()
    config.n_samples = 100
    config.n_warmup = 100

    let chain = metropolis_hastings(lp, config)

    // CI should contain true posterior mean (near 0)
    chain.ci_lower < 0.0 && chain.ci_upper > 0.0
}

fn main() -> i32 {
    print("Testing bayes::mcmc module...\n")

    if !test_metropolis_simple() {
        print("FAIL: metropolis_simple\n")
        return 1
    }
    print("PASS: metropolis_simple\n")

    if !test_acceptance_rate() {
        print("FAIL: acceptance_rate\n")
        return 2
    }
    print("PASS: acceptance_rate\n")

    if !test_credible_interval() {
        print("FAIL: credible_interval\n")
        return 3
    }
    print("PASS: credible_interval\n")

    print("All bayes::mcmc tests PASSED\n")
    0
}
