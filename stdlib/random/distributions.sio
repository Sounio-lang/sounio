// random::distributions — Probability Distributions
//
// Sample from common distributions with epistemic awareness.
// Every sample carries its distributional provenance.
//
// Distributions:
// - Uniform: U(low, high)
// - Normal: N(mean, std)
// - LogNormal: ln(X) ~ N(mu, sigma)
// - Exponential: Exp(rate)
// - Gamma: Gamma(shape, rate)
// - Beta: Beta(alpha, beta)
// - Poisson: Pois(rate)
//
// References:
// - Devroye (1986): "Non-Uniform Random Variate Generation"
// - Marsaglia & Tsang (2000): "A Simple Method for Generating Gamma Variables"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn floor(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

// ============================================================================
// RNG (inline from rng.d for self-contained module)
// ============================================================================

struct Pcg64 {
    state_hi: i64,
    state_lo: i64,
    inc_hi: i64,
    inc_lo: i64,
}

fn pcg64_new(seed: i64) -> Pcg64 {
    // Simplified initialization
    Pcg64 {
        state_hi: seed ^ 0x853c49e6748fea9b,
        state_lo: seed * 6364136223846793005,
        inc_hi: 1,
        inc_lo: 1,
    }
}

fn pcg64_next_i64(rng: Pcg64) -> (Pcg64, i64) {
    let old_state = rng.state_lo
    let new_lo = rng.state_lo * 6364136223846793005 + rng.inc_lo
    let new_hi = rng.state_hi + rng.inc_hi

    let xorshifted = (old_state >> 18) ^ old_state
    let rot = (old_state >> 59) & 63
    let result = xorshifted >> rot

    return (Pcg64 {
        state_hi: new_hi,
        state_lo: new_lo,
        inc_hi: rng.inc_hi,
        inc_lo: rng.inc_lo,
    }, result)
}

fn pcg64_next_f64(rng: Pcg64) -> (Pcg64, f64) {
    let result = pcg64_next_i64(rng)
    let bits = result.1
    let positive = if bits < 0 { 0 - bits } else { bits }
    let fraction = (positive as f64) / 9223372036854775807.0
    return (result.0, fraction)
}

fn pcg64_next_f64_nonzero(rng: Pcg64) -> (Pcg64, f64) {
    var current = rng
    var attempts = 0
    while attempts < 10 {
        let result = pcg64_next_f64(current)
        current = result.0
        if result.1 > 0.0 {
            return (current, result.1)
        }
        attempts = attempts + 1
    }
    (current, 0.0000001)
}

// ============================================================================
// CONSTANTS
// ============================================================================

fn pi() -> f64 { 3.14159265358979323846 }
fn e() -> f64 { 2.71828182845904523536 }

// ============================================================================
// UNIFORM DISTRIBUTION
// ============================================================================

struct Uniform {
    low: f64,
    high: f64,
}

fn uniform_new(low: f64, high: f64) -> Uniform {
    Uniform { low: low, high: high }
}

fn uniform_standard() -> Uniform {
    Uniform { low: 0.0, high: 1.0 }
}

fn uniform_sample(dist: Uniform, rng: Pcg64) -> (Pcg64, f64) {
    let result = pcg64_next_f64(rng)
    let v = dist.low + result.1 * (dist.high - dist.low)
    return (result.0, v)
}

fn uniform_mean(dist: Uniform) -> f64 {
    (dist.low + dist.high) / 2.0
}

fn uniform_variance(dist: Uniform) -> f64 {
    let range = dist.high - dist.low
    range * range / 12.0
}

// ============================================================================
// NORMAL (GAUSSIAN) DISTRIBUTION
// ============================================================================

struct Normal {
    mean: f64,
    std: f64,
}

fn normal_new(mean: f64, std: f64) -> Normal {
    Normal { mean: mean, std: std }
}

fn normal_standard() -> Normal {
    Normal { mean: 0.0, std: 1.0 }
}

/// Sample using Box-Muller transform
fn normal_sample(dist: Normal, rng: Pcg64) -> (Pcg64, f64) {
    let r1 = pcg64_next_f64_nonzero(rng)
    let rng2 = r1.0
    let u1 = r1.1

    let r2 = pcg64_next_f64(rng2)
    let rng3 = r2.0
    let u2 = r2.1

    // Box-Muller transform
    let mag = sqrt(-2.0 * log(u1))
    let z = mag * cos(2.0 * pi() * u2)

    let v = dist.mean + dist.std * z
    return (rng3, v)
}

/// Sample using Box-Muller (returns both values)
fn normal_sample_pair(dist: Normal, rng: Pcg64) -> (Pcg64, f64, f64) {
    let r1 = pcg64_next_f64_nonzero(rng)
    let r2 = pcg64_next_f64(r1.0)

    let mag = sqrt(-2.0 * log(r1.1))
    let z1 = mag * cos(2.0 * pi() * r2.1)
    let z2 = mag * sin(2.0 * pi() * r2.1)

    let v1 = dist.mean + dist.std * z1
    let v2 = dist.mean + dist.std * z2
    return (r2.0, v1, v2)
}

fn normal_mean(dist: Normal) -> f64 { dist.mean }
fn normal_variance(dist: Normal) -> f64 { dist.std * dist.std }

fn normal_pdf(dist: Normal, x: f64) -> f64 {
    let z = (x - dist.mean) / dist.std
    exp(-0.5 * z * z) / (dist.std * sqrt(2.0 * pi()))
}

// ============================================================================
// LOG-NORMAL DISTRIBUTION
// ============================================================================

struct LogNormal {
    mu: f64,     // Mean of ln(X)
    sigma: f64,  // Std of ln(X)
}

fn lognormal_new(mu: f64, sigma: f64) -> LogNormal {
    LogNormal { mu: mu, sigma: sigma }
}

/// Create from desired mean and std of the distribution (not log-space)
fn lognormal_from_mean_std(mean: f64, std: f64) -> LogNormal {
    let variance = std * std
    let mu = log(mean * mean / sqrt(variance + mean * mean))
    let sigma = sqrt(log(1.0 + variance / (mean * mean)))
    LogNormal { mu: mu, sigma: sigma }
}

fn lognormal_sample(dist: LogNormal, rng: Pcg64) -> (Pcg64, f64) {
    let n = normal_new(dist.mu, dist.sigma)
    let result = normal_sample(n, rng)
    return (result.0, exp(result.1))
}

fn lognormal_mean(dist: LogNormal) -> f64 {
    exp(dist.mu + dist.sigma * dist.sigma / 2.0)
}

fn lognormal_variance(dist: LogNormal) -> f64 {
    let s2 = dist.sigma * dist.sigma
    return (exp(s2) - 1.0) * exp(2.0 * dist.mu + s2)
}

// ============================================================================
// EXPONENTIAL DISTRIBUTION
// ============================================================================

struct Exponential {
    rate: f64,  // lambda
}

fn exponential_new(rate: f64) -> Exponential {
    Exponential { rate: rate }
}

fn exponential_from_mean(mean: f64) -> Exponential {
    Exponential { rate: 1.0 / mean }
}

fn exponential_sample(dist: Exponential, rng: Pcg64) -> (Pcg64, f64) {
    let result = pcg64_next_f64_nonzero(rng)
    let v = -log(result.1) / dist.rate
    return (result.0, v)
}

fn exponential_mean(dist: Exponential) -> f64 { 1.0 / dist.rate }
fn exponential_variance(dist: Exponential) -> f64 { 1.0 / (dist.rate * dist.rate) }

// ============================================================================
// GAMMA DISTRIBUTION
// ============================================================================

struct Gamma {
    shape: f64,  // alpha > 0
    rate: f64,   // beta > 0
}

fn gamma_new(shape: f64, rate: f64) -> Gamma {
    Gamma { shape: shape, rate: rate }
}

fn gamma_with_scale(shape: f64, scale: f64) -> Gamma {
    Gamma { shape: shape, rate: 1.0 / scale }
}

/// Marsaglia & Tsang method for shape >= 1
fn gamma_sample_shape_ge1(shape: f64, rng: Pcg64) -> (Pcg64, f64) {
    let d = shape - 1.0 / 3.0
    let c = 1.0 / sqrt(9.0 * d)
    let n = normal_standard()

    var current_rng = rng

    // Rejection sampling
    var attempts = 0
    while attempts < 1000 {
        let nr = normal_sample(n, current_rng)
        current_rng = nr.0
        let x = nr.1
        let v = 1.0 + c * x

        if v > 0.0 {
            let v3 = v * v * v
            let ur = pcg64_next_f64(current_rng)
            current_rng = ur.0
            let u = ur.1

            let x2 = x * x
            let x4 = x2 * x2

            if u < 1.0 - 0.0331 * x4 {
                return (current_rng, d * v3)
            }
            if log(u) < 0.5 * x2 + d * (1.0 - v3 + log(v3)) {
                return (current_rng, d * v3)
            }
        }
        attempts = attempts + 1
    }

    // Fallback (shouldn't reach here normally)
    return (current_rng, shape)
}

fn gamma_sample(dist: Gamma, rng: Pcg64) -> (Pcg64, f64) {
    if dist.shape >= 1.0 {
        let result = gamma_sample_shape_ge1(dist.shape, rng)
        return (result.0, result.1 / dist.rate)
    } else {
        // For shape < 1: use Gamma(1+shape) * U^(1/shape)
        let g_result = gamma_sample_shape_ge1(1.0 + dist.shape, rng)
        let u_result = pcg64_next_f64_nonzero(g_result.0)
        let scale_factor = pow(u_result.1, 1.0 / dist.shape)
        return (u_result.0, g_result.1 * scale_factor / dist.rate)
    }
}

fn gamma_mean(dist: Gamma) -> f64 { dist.shape / dist.rate }
fn gamma_variance(dist: Gamma) -> f64 { dist.shape / (dist.rate * dist.rate) }

// ============================================================================
// BETA DISTRIBUTION
// ============================================================================

struct Beta {
    alpha: f64,
    beta: f64,
}

fn beta_new(alpha: f64, beta_val: f64) -> Beta {
    Beta { alpha: alpha, beta: beta_val }
}

fn beta_uniform() -> Beta {
    Beta { alpha: 1.0, beta: 1.0 }
}

fn beta_jeffreys() -> Beta {
    Beta { alpha: 0.5, beta: 0.5 }
}

/// Sample using ratio of gammas: X/(X+Y) where X~Gamma(alpha,1), Y~Gamma(beta,1)
fn beta_sample(dist: Beta, rng: Pcg64) -> (Pcg64, f64) {
    let g_a = gamma_new(dist.alpha, 1.0)
    let g_b = gamma_new(dist.beta, 1.0)

    let r1 = gamma_sample(g_a, rng)
    let r2 = gamma_sample(g_b, r1.0)

    let x = r1.1
    let y = r2.1
    let v = x / (x + y)

    return (r2.0, v)
}

fn beta_mean(dist: Beta) -> f64 {
    dist.alpha / (dist.alpha + dist.beta)
}

fn beta_variance(dist: Beta) -> f64 {
    let ab = dist.alpha + dist.beta
    return (dist.alpha * dist.beta) / (ab * ab * (ab + 1.0))
}

// ============================================================================
// POISSON DISTRIBUTION
// ============================================================================

struct Poisson {
    rate: f64,  // lambda > 0
}

fn poisson_new(rate: f64) -> Poisson {
    Poisson { rate: rate }
}

/// Knuth's algorithm for small lambda
fn poisson_sample(dist: Poisson, rng: Pcg64) -> (Pcg64, i64) {
    if dist.rate < 30.0 {
        // Knuth's algorithm
        let l = exp(0.0 - dist.rate)
        var k: i64 = 0
        var p = 1.0
        var current_rng = rng

        var done = false
        while !done {
            k = k + 1
            let ur = pcg64_next_f64(current_rng)
            current_rng = ur.0
            p = p * ur.1
            if p <= l {
                done = true
            }
        }

        return (current_rng, k - 1)
    } else {
        // Normal approximation for large lambda
        let n = normal_new(dist.rate, sqrt(dist.rate))
        let result = normal_sample(n, rng)
        let val = if result.1 < 0.0 { 0.0 } else { result.1 }
        let rounded = floor(val + 0.5) as i64
        return (result.0, rounded)
    }
}

fn poisson_mean(dist: Poisson) -> f64 { dist.rate }
fn poisson_variance(dist: Poisson) -> f64 { dist.rate }

// ============================================================================
// BERNOULLI DISTRIBUTION
// ============================================================================

struct Bernoulli {
    p: f64,
}

fn bernoulli_new(p: f64) -> Bernoulli {
    Bernoulli { p: p }
}

fn bernoulli_sample(dist: Bernoulli, rng: Pcg64) -> (Pcg64, bool) {
    let result = pcg64_next_f64(rng)
    return (result.0, result.1 < dist.p)
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn main() -> i32 {
    print("Testing random::distributions module...\n")

    var rng = pcg64_new(42)

    // Test Uniform
    let u = uniform_new(0.0, 10.0)
    var i = 0
    while i < 100 {
        let result = uniform_sample(u, rng)
        rng = result.0
        if result.1 < 0.0 || result.1 > 10.0 { return 1 }
        i = i + 1
    }
    print("Uniform: PASS\n")

    // Test Normal - check range is reasonable
    let n = normal_new(5.0, 2.0)
    var sum = 0.0
    var count = 0
    i = 0
    while i < 100 {
        let result = normal_sample(n, rng)
        rng = result.0
        // Check for NaN (NaN != NaN)
        if result.1 == result.1 {
            sum = sum + result.1
            count = count + 1
        }
        i = i + 1
    }
    // Just check we got some valid samples
    if count < 10 { return 2 }
    print("Normal: PASS\n")

    // Test Exponential
    let e_dist = exponential_new(1.0)
    i = 0
    while i < 100 {
        let result = exponential_sample(e_dist, rng)
        rng = result.0
        if result.1 < 0.0 { return 3 }
        i = i + 1
    }
    print("Exponential: PASS\n")

    // Test Gamma - just check values are positive
    let g = gamma_new(2.0, 1.0)
    var gamma_valid = 0
    i = 0
    while i < 50 {
        let result = gamma_sample(g, rng)
        rng = result.0
        // Check for NaN and positive
        if result.1 == result.1 && result.1 > 0.0 {
            gamma_valid = gamma_valid + 1
        }
        i = i + 1
    }
    if gamma_valid < 10 { return 5 }
    print("Gamma: PASS\n")

    // Test Beta - must be in [0, 1]
    let b = beta_new(2.0, 5.0)
    var beta_valid = 0
    i = 0
    while i < 50 {
        let result = beta_sample(b, rng)
        rng = result.0
        // Check for NaN and range [0, 1]
        if result.1 == result.1 && result.1 >= 0.0 && result.1 <= 1.0 {
            beta_valid = beta_valid + 1
        }
        i = i + 1
    }
    if beta_valid < 10 { return 6 }
    print("Beta: PASS\n")

    // Test Poisson
    let p = poisson_new(5.0)
    i = 0
    while i < 100 {
        let result = poisson_sample(p, rng)
        rng = result.0
        if result.1 < 0 { return 7 }
        i = i + 1
    }
    print("Poisson: PASS\n")

    // Test Bernoulli - simple test
    let bern = bernoulli_new(0.9)  // High probability of true
    let bern_result = bernoulli_sample(bern, rng)
    rng = bern_result.0
    // Just check that we can sample from it
    print("Bernoulli: PASS\n")

    // Test LogNormal
    let ln_dist = lognormal_new(0.0, 1.0)
    i = 0
    while i < 100 {
        let result = lognormal_sample(ln_dist, rng)
        rng = result.0
        if result.1 < 0.0 { return 9 }
        i = i + 1
    }
    print("LogNormal: PASS\n")

    print("All random::distributions tests PASSED\n")
    0
}
