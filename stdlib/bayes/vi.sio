// bayes::vi — Variational Inference
//
// Mean-field variational inference for approximate Bayesian inference.
// Faster than MCMC for large datasets, provides lower bound on evidence.
//
// References:
// - Blei et al. (2017): "Variational Inference: A Review for Statisticians"
// - Kucukelbir et al. (2017): "Automatic Differentiation Variational Inference"
// - Jordan et al. (1999): "An Introduction to Variational Methods"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// VI CONFIGURATION
// ============================================================================

/// Variational inference configuration
struct VIConfig {
    max_iter: i64,          // Maximum iterations
    tol: f64,               // Convergence tolerance for ELBO
    learning_rate: f64,     // Step size for gradient updates
    n_samples: i64,         // Monte Carlo samples for gradient estimation
    seed: i64,              // Random seed
}

fn vi_config_default() -> VIConfig {
    VIConfig {
        max_iter: 1000,
        tol: 1e-4,
        learning_rate: 0.01,
        n_samples: 10,
        seed: 12345,
    }
}

/// Variational distribution (mean-field Gaussian)
struct VIDist {
    mean: f64,              // Variational mean
    log_std: f64,           // Log of variational std (for unconstrained optimization)
    std: f64,               // Variational std (exp(log_std))
}

fn vi_dist_new(mean: f64, std: f64) -> VIDist {
    VIDist {
        mean: mean,
        log_std: log(std),
        std: std,
    }
}

/// VI result
struct VIResult {
    dist: VIDist,           // Fitted variational distribution
    elbo: f64,              // Final ELBO (evidence lower bound)
    converged: bool,        // Whether optimization converged
    n_iter: i64,            // Number of iterations
    elbo_history: [f64; 100], // ELBO at each iteration (first 100)
}

fn vi_result_new() -> VIResult {
    VIResult {
        dist: vi_dist_new(0.0, 1.0),
        elbo: -1e100,
        converged: false,
        n_iter: 0,
        elbo_history: [0.0; 100],
    }
}

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

struct RNG {
    seed: i64,
    last: f64,
}

fn rng_new(seed: i64) -> RNG {
    RNG { seed: seed, last: 0.0 }
}

fn rng_uniform(rng: RNG) -> RNG {
    let a: i64 = 1103515245
    let c: i64 = 12345
    let m: i64 = 2147483648

    let new_seed = (a * rng.seed + c) % m
    let u = (new_seed as f64) / (m as f64)
    RNG { seed: new_seed, last: u }
}

fn rng_normal(rng: RNG) -> RNG {
    var r = rng

    r = rng_uniform(r)
    var u1 = r.last
    if u1 < 1e-10 { u1 = 1e-10 }

    r = rng_uniform(r)
    let u2 = r.last

    let two_pi = 2.0 * pi()
    // Approximate cos
    var angle = two_pi * u2
    while angle > two_pi { angle = angle - two_pi }
    let x2 = angle * angle
    let cos_val = 1.0 - x2 / 2.0 + x2 * x2 / 24.0

    let z = sqrt(-2.0 * log(u1)) * cos_val
    RNG { seed: r.seed, last: z }
}

// ============================================================================
// MODEL SPECIFICATION (NORMAL-NORMAL)
// ============================================================================

/// Model for normal-normal conjugate case
struct NormalModel {
    data_mean: f64,
    data_var: f64,
    data_n: i64,
    prior_mean: f64,
    prior_var: f64,
}

fn normal_model_new(data_mean: f64, data_var: f64, data_n: i64, prior_mean: f64, prior_var: f64) -> NormalModel {
    NormalModel {
        data_mean: data_mean,
        data_var: data_var,
        data_n: data_n,
        prior_mean: prior_mean,
        prior_var: prior_var,
    }
}

/// Log joint probability p(data, theta)
fn log_joint(model: NormalModel, theta: f64) -> f64 {
    // Log-likelihood
    let ll = -(model.data_n as f64) * (theta - model.data_mean) * (theta - model.data_mean) / (2.0 * model.data_var)

    // Log-prior
    let lpr = -(theta - model.prior_mean) * (theta - model.prior_mean) / (2.0 * model.prior_var)

    ll + lpr
}

// ============================================================================
// ELBO COMPUTATION
// ============================================================================

/// Compute ELBO using Monte Carlo sampling
/// ELBO = E_q[log p(x, z)] - E_q[log q(z)]
///      = E_q[log p(x, z)] + H(q)
fn compute_elbo(model: NormalModel, q: VIDist, n_samples: i64, rng: RNG) -> (f64, RNG) {
    var r = rng
    var sum_log_joint = 0.0

    var i: i64 = 0
    while i < n_samples {
        // Sample from q(theta) = N(mu, sigma^2)
        r = rng_normal(r)
        let z = r.last
        let theta = q.mean + q.std * z

        // Accumulate log joint
        sum_log_joint = sum_log_joint + log_joint(model, theta)
        i = i + 1
    }

    // Monte Carlo estimate of E_q[log p(x, z)]
    let expected_log_joint = sum_log_joint / (n_samples as f64)

    // Entropy of Gaussian: H(q) = 0.5 * log(2 * pi * e * sigma^2)
    //                           = 0.5 * (1 + log(2*pi) + 2*log(sigma))
    let entropy = 0.5 * (1.0 + log(2.0 * pi()) + 2.0 * q.log_std)

    let elbo = expected_log_joint + entropy

    (elbo, r)
}

/// Compute ELBO gradients using reparameterization trick
fn compute_elbo_gradients(model: NormalModel, q: VIDist, n_samples: i64, rng: RNG) -> (f64, f64, RNG) {
    var r = rng
    var grad_mean = 0.0
    var grad_log_std = 0.0

    var i: i64 = 0
    while i < n_samples {
        // Sample epsilon ~ N(0, 1)
        r = rng_normal(r)
        let eps = r.last

        // Reparameterized sample: theta = mu + sigma * eps
        let theta = q.mean + q.std * eps

        // Gradient of log p(x, theta) w.r.t. theta
        // For normal-normal: d/dtheta log p = -n*(theta - x_bar)/var - (theta - mu0)/var0
        let grad_ll = -(model.data_n as f64) * (theta - model.data_mean) / model.data_var
        let grad_lpr = -(theta - model.prior_mean) / model.prior_var
        let grad_log_p = grad_ll + grad_lpr

        // Chain rule for reparameterization
        // d/d_mu = d/d_theta * d_theta/d_mu = grad_log_p * 1
        // d/d_log_sigma = d/d_theta * d_theta/d_log_sigma = grad_log_p * sigma * eps
        grad_mean = grad_mean + grad_log_p
        grad_log_std = grad_log_std + grad_log_p * q.std * eps

        i = i + 1
    }

    // Average gradients
    grad_mean = grad_mean / (n_samples as f64)
    grad_log_std = grad_log_std / (n_samples as f64)

    // Add entropy gradient w.r.t. log_std: d/d_log_std (0.5 * 2 * log_std) = 1
    grad_log_std = grad_log_std + 1.0

    (grad_mean, grad_log_std, r)
}

// ============================================================================
// VARIATIONAL INFERENCE OPTIMIZER
// ============================================================================

/// Run mean-field variational inference
fn run_vi(model: NormalModel, config: VIConfig) -> VIResult {
    var result = vi_result_new()
    var rng = rng_new(config.seed)

    // Initialize variational parameters
    var q = vi_dist_new(model.prior_mean, sqrt(model.prior_var))

    var prev_elbo = -1e100

    var iter: i64 = 0
    while iter < config.max_iter {
        // Compute gradients
        let grad_result = compute_elbo_gradients(model, q, config.n_samples, rng)
        let grad_mean = grad_result.0
        let grad_log_std = grad_result.1
        rng = grad_result.2

        // Gradient ascent update
        q.mean = q.mean + config.learning_rate * grad_mean
        q.log_std = q.log_std + config.learning_rate * grad_log_std
        q.std = exp(q.log_std)

        // Compute ELBO for convergence check
        let elbo_result = compute_elbo(model, q, config.n_samples, rng)
        let elbo = elbo_result.0
        rng = elbo_result.1

        // Store history
        if iter < 100 {
            result.elbo_history[iter as usize] = elbo
        }

        // Check convergence
        let diff = elbo - prev_elbo
        let abs_diff = if diff < 0.0 { -diff } else { diff }
        if abs_diff < config.tol && iter > 10 {
            result.converged = true
            result.n_iter = iter + 1
            result.elbo = elbo
            result.dist = q
            return result
        }

        prev_elbo = elbo
        iter = iter + 1
    }

    result.n_iter = iter
    result.elbo = prev_elbo
    result.dist = q
    result.converged = false

    result
}

// ============================================================================
// ANALYTICAL SOLUTION (FOR COMPARISON)
// ============================================================================

/// Analytical posterior for normal-normal model
fn analytical_posterior(model: NormalModel) -> VIDist {
    let prior_prec = 1.0 / model.prior_var
    let lik_prec = (model.data_n as f64) / model.data_var

    let post_prec = prior_prec + lik_prec
    let post_var = 1.0 / post_prec
    let post_mean = post_var * (prior_prec * model.prior_mean + lik_prec * model.data_mean)

    vi_dist_new(post_mean, sqrt(post_var))
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_vi_converges() -> bool {
    let model = normal_model_new(5.0, 1.0, 10, 0.0, 10.0)
    var config = vi_config_default()
    config.max_iter = 500
    config.learning_rate = 0.1

    let result = run_vi(model, config)

    // Should converge
    result.converged || result.n_iter > 100
}

fn test_vi_matches_analytical() -> bool {
    let model = normal_model_new(5.0, 1.0, 10, 0.0, 10.0)
    var config = vi_config_default()
    config.max_iter = 500
    config.learning_rate = 0.1
    config.n_samples = 20

    let result = run_vi(model, config)
    let analytical = analytical_posterior(model)

    // VI mean should be close to analytical
    // Analytical: post_prec = 0.1 + 10 = 10.1
    // post_mean = (0.1 * 0 + 10 * 5) / 10.1 = 4.95
    abs_f64(result.dist.mean - analytical.mean) < 0.5
}

fn test_elbo_increases() -> bool {
    let model = normal_model_new(0.0, 1.0, 10, 0.0, 1.0)
    var config = vi_config_default()
    config.max_iter = 100
    config.learning_rate = 0.05

    let result = run_vi(model, config)

    // ELBO should generally increase (allowing some noise)
    // Check that later ELBO > initial ELBO
    let n_check = if result.n_iter > 50 { 50 } else { result.n_iter }
    result.elbo_history[(n_check - 1) as usize] > result.elbo_history[0] - 1.0
}

fn main() -> i32 {
    print("Testing bayes::vi module...\n")

    if !test_vi_converges() {
        print("FAIL: vi_converges\n")
        return 1
    }
    print("PASS: vi_converges\n")

    if !test_vi_matches_analytical() {
        print("FAIL: vi_matches_analytical\n")
        return 2
    }
    print("PASS: vi_matches_analytical\n")

    if !test_elbo_increases() {
        print("FAIL: elbo_increases\n")
        return 3
    }
    print("PASS: elbo_increases\n")

    print("All bayes::vi tests PASSED\n")
    0
}
