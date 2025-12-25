// bayes::prior — Prior Distributions for Bayesian Inference
//
// Common prior distributions with log-probability evaluation.
// Designed for use with MCMC and variational inference.
//
// References:
// - Gelman et al. (2013): "Bayesian Data Analysis"
// - Stan Development Team: "Stan Prior Choice Recommendations"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// PRIOR RESULT TYPES
// ============================================================================

/// Result of prior evaluation
struct PriorEval {
    log_prob: f64,      // Log-probability density
    valid: bool,        // Whether value is in support
}

fn prior_eval_new(lp: f64, valid: bool) -> PriorEval {
    PriorEval { log_prob: lp, valid: valid }
}

/// Prior specification with hyperparameters
struct Prior {
    dist_type: i64,     // 0=Normal, 1=LogNormal, 2=Uniform, 3=Beta, 4=Gamma, 5=HalfNormal, 6=Exponential
    param1: f64,        // Mean/location/lower/alpha/shape
    param2: f64,        // Std/scale/upper/beta/rate
}

fn prior_new(dist_type: i64, p1: f64, p2: f64) -> Prior {
    Prior { dist_type: dist_type, param1: p1, param2: p2 }
}

// ============================================================================
// NORMAL PRIOR
// ============================================================================

/// Normal(mu, sigma) prior
fn prior_normal(mu: f64, sigma: f64) -> Prior {
    prior_new(0, mu, sigma)
}

/// Log-probability of Normal prior
fn log_prob_normal(x: f64, mu: f64, sigma: f64) -> PriorEval {
    if sigma <= 0.0 {
        return prior_eval_new(-1e100, false)
    }

    let z = (x - mu) / sigma
    let lp = -0.5 * log(2.0 * pi()) - log(sigma) - 0.5 * z * z
    prior_eval_new(lp, true)
}

// ============================================================================
// LOG-NORMAL PRIOR
// ============================================================================

/// LogNormal(mu, sigma) prior - for positive parameters
fn prior_lognormal(mu: f64, sigma: f64) -> Prior {
    prior_new(1, mu, sigma)
}

/// Log-probability of LogNormal prior
fn log_prob_lognormal(x: f64, mu: f64, sigma: f64) -> PriorEval {
    if x <= 0.0 || sigma <= 0.0 {
        return prior_eval_new(-1e100, false)
    }

    let log_x = log(x)
    let z = (log_x - mu) / sigma
    let lp = -0.5 * log(2.0 * pi()) - log(sigma) - log_x - 0.5 * z * z
    prior_eval_new(lp, true)
}

// ============================================================================
// UNIFORM PRIOR
// ============================================================================

/// Uniform(lower, upper) prior
fn prior_uniform(lower: f64, upper: f64) -> Prior {
    prior_new(2, lower, upper)
}

/// Log-probability of Uniform prior
fn log_prob_uniform(x: f64, lower: f64, upper: f64) -> PriorEval {
    if x < lower || x > upper {
        return prior_eval_new(-1e100, false)
    }

    let lp = -log(upper - lower)
    prior_eval_new(lp, true)
}

// ============================================================================
// BETA PRIOR
// ============================================================================

/// Beta(alpha, beta) prior - for probabilities in [0, 1]
fn prior_beta(alpha: f64, beta: f64) -> Prior {
    prior_new(3, alpha, beta)
}

/// Log-probability of Beta prior (simplified, no log-gamma)
fn log_prob_beta(x: f64, alpha: f64, beta: f64) -> PriorEval {
    if x <= 0.0 || x >= 1.0 || alpha <= 0.0 || beta <= 0.0 {
        return prior_eval_new(-1e100, false)
    }

    // log(x^(a-1) * (1-x)^(b-1)) - log(B(a,b))
    // We omit the normalizing constant for MCMC purposes
    let lp = (alpha - 1.0) * log(x) + (beta - 1.0) * log(1.0 - x)
    prior_eval_new(lp, true)
}

// ============================================================================
// GAMMA PRIOR
// ============================================================================

/// Gamma(shape, rate) prior - for positive parameters
fn prior_gamma(shape: f64, rate: f64) -> Prior {
    prior_new(4, shape, rate)
}

/// Log-probability of Gamma prior (simplified)
fn log_prob_gamma(x: f64, shape: f64, rate: f64) -> PriorEval {
    if x <= 0.0 || shape <= 0.0 || rate <= 0.0 {
        return prior_eval_new(-1e100, false)
    }

    // (shape - 1) * log(x) - rate * x + shape * log(rate) - log(Gamma(shape))
    // Omit normalizing constant
    let lp = (shape - 1.0) * log(x) - rate * x
    prior_eval_new(lp, true)
}

// ============================================================================
// HALF-NORMAL PRIOR
// ============================================================================

/// HalfNormal(sigma) prior - for positive scale parameters
fn prior_half_normal(sigma: f64) -> Prior {
    prior_new(5, 0.0, sigma)
}

/// Log-probability of HalfNormal prior
fn log_prob_half_normal(x: f64, sigma: f64) -> PriorEval {
    if x < 0.0 || sigma <= 0.0 {
        return prior_eval_new(-1e100, false)
    }

    let z = x / sigma
    let lp = log(2.0) - 0.5 * log(2.0 * pi()) - log(sigma) - 0.5 * z * z
    prior_eval_new(lp, true)
}

// ============================================================================
// EXPONENTIAL PRIOR
// ============================================================================

/// Exponential(rate) prior - for positive waiting times
fn prior_exponential(rate: f64) -> Prior {
    prior_new(6, rate, 0.0)
}

/// Log-probability of Exponential prior
fn log_prob_exponential(x: f64, rate: f64) -> PriorEval {
    if x < 0.0 || rate <= 0.0 {
        return prior_eval_new(-1e100, false)
    }

    let lp = log(rate) - rate * x
    prior_eval_new(lp, true)
}

// ============================================================================
// CAUCHY PRIOR (HEAVY-TAILED)
// ============================================================================

/// Cauchy(location, scale) prior - robust, heavy-tailed
fn prior_cauchy(location: f64, scale: f64) -> Prior {
    prior_new(7, location, scale)
}

/// Log-probability of Cauchy prior
fn log_prob_cauchy(x: f64, location: f64, scale: f64) -> PriorEval {
    if scale <= 0.0 {
        return prior_eval_new(-1e100, false)
    }

    let z = (x - location) / scale
    let lp = -log(pi()) - log(scale) - log(1.0 + z * z)
    prior_eval_new(lp, true)
}

// ============================================================================
// HALF-CAUCHY PRIOR
// ============================================================================

/// HalfCauchy(scale) prior - for variance parameters (recommended by Gelman)
fn prior_half_cauchy(scale: f64) -> Prior {
    prior_new(8, 0.0, scale)
}

/// Log-probability of HalfCauchy prior
fn log_prob_half_cauchy(x: f64, scale: f64) -> PriorEval {
    if x < 0.0 || scale <= 0.0 {
        return prior_eval_new(-1e100, false)
    }

    let z = x / scale
    let lp = log(2.0) - log(pi()) - log(scale) - log(1.0 + z * z)
    prior_eval_new(lp, true)
}

// ============================================================================
// GENERIC PRIOR EVALUATION
// ============================================================================

/// Evaluate log-probability of any prior at value x
fn prior_log_prob(prior: Prior, x: f64) -> PriorEval {
    if prior.dist_type == 0 {
        return log_prob_normal(x, prior.param1, prior.param2)
    }
    if prior.dist_type == 1 {
        return log_prob_lognormal(x, prior.param1, prior.param2)
    }
    if prior.dist_type == 2 {
        return log_prob_uniform(x, prior.param1, prior.param2)
    }
    if prior.dist_type == 3 {
        return log_prob_beta(x, prior.param1, prior.param2)
    }
    if prior.dist_type == 4 {
        return log_prob_gamma(x, prior.param1, prior.param2)
    }
    if prior.dist_type == 5 {
        return log_prob_half_normal(x, prior.param2)
    }
    if prior.dist_type == 6 {
        return log_prob_exponential(x, prior.param1)
    }
    if prior.dist_type == 7 {
        return log_prob_cauchy(x, prior.param1, prior.param2)
    }
    if prior.dist_type == 8 {
        return log_prob_half_cauchy(x, prior.param2)
    }
    // Unknown prior type
    prior_eval_new(-1e100, false)
}

// ============================================================================
// PRIOR RECOMMENDATIONS
// ============================================================================

/// Weakly informative prior for regression coefficients
fn prior_weakly_informative_coef() -> Prior {
    // Normal(0, 2.5) as recommended by Stan
    prior_normal(0.0, 2.5)
}

/// Weakly informative prior for intercepts
fn prior_weakly_informative_intercept() -> Prior {
    // Student-t approximated by Cauchy
    prior_cauchy(0.0, 10.0)
}

/// Weakly informative prior for scale parameters
fn prior_weakly_informative_scale() -> Prior {
    // Half-Cauchy(0, 2.5) as recommended by Gelman
    prior_half_cauchy(2.5)
}

/// Jeffreys prior for variance (improper, use with caution)
fn prior_jeffreys_variance() -> Prior {
    // Corresponds to p(sigma^2) prop to 1/sigma^2
    // Approximated as Gamma(0.001, 0.001)
    prior_gamma(0.001, 0.001)
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_normal_prior() -> bool {
    let prior = prior_normal(0.0, 1.0)
    let eval = prior_log_prob(prior, 0.0)

    // log(N(0|0,1)) = -0.5 * log(2*pi) ≈ -0.919
    eval.valid && abs_f64(eval.log_prob - (-0.919)) < 0.01
}

fn test_lognormal_prior() -> bool {
    let prior = prior_lognormal(0.0, 1.0)

    // Value must be positive
    let eval_neg = prior_log_prob(prior, -1.0)
    let eval_pos = prior_log_prob(prior, 1.0)

    !eval_neg.valid && eval_pos.valid
}

fn test_uniform_prior() -> bool {
    let prior = prior_uniform(0.0, 1.0)

    let eval_in = prior_log_prob(prior, 0.5)
    let eval_out = prior_log_prob(prior, 1.5)

    // log(U(0.5|0,1)) = log(1) = 0
    eval_in.valid && abs_f64(eval_in.log_prob) < 0.01 && !eval_out.valid
}

fn test_beta_prior() -> bool {
    let prior = prior_beta(2.0, 2.0)

    let eval = prior_log_prob(prior, 0.5)

    // Beta(2,2) is symmetric, mode at 0.5
    // log((0.5)^1 * (0.5)^1) = 2 * log(0.5) ≈ -1.39
    eval.valid && eval.log_prob > -2.0
}

fn test_half_normal_support() -> bool {
    let prior = prior_half_normal(1.0)

    let eval_neg = prior_log_prob(prior, -0.5)
    let eval_pos = prior_log_prob(prior, 0.5)

    !eval_neg.valid && eval_pos.valid
}

fn main() -> i32 {
    print("Testing bayes::prior module...\n")

    if !test_normal_prior() {
        print("FAIL: normal_prior\n")
        return 1
    }
    print("PASS: normal_prior\n")

    if !test_lognormal_prior() {
        print("FAIL: lognormal_prior\n")
        return 2
    }
    print("PASS: lognormal_prior\n")

    if !test_uniform_prior() {
        print("FAIL: uniform_prior\n")
        return 3
    }
    print("PASS: uniform_prior\n")

    if !test_beta_prior() {
        print("FAIL: beta_prior\n")
        return 4
    }
    print("PASS: beta_prior\n")

    if !test_half_normal_support() {
        print("FAIL: half_normal_support\n")
        return 5
    }
    print("PASS: half_normal_support\n")

    print("All bayes::prior tests PASSED\n")
    0
}
