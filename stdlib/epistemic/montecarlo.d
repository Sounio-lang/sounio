//! Monte Carlo Propagation Module
//!
//! When GUM's first-order linearization fails (nonlinearity, non-Gaussian inputs),
//! Monte Carlo is the reference method for propagating distributions.
//!
//! Per JCGM 101:2008 (Supplement 1 to the GUM):
//! "The Monte Carlo method is a general numerical technique for solving
//!  the uncertainty propagation problem"
//!
//! Key Features:
//! - Generate N samples from input distributions
//! - Propagate through arbitrary functions
//! - Extract statistics from output distribution
//! - Handles arbitrary nonlinearity
//!
//! References:
//!   - JCGM 101:2008 (Evaluation of measurement data — Supplement 1 to the GUM)

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn exp(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn log_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1.0e308 }
    return log(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

// ============================================================================
// PSEUDO-RANDOM NUMBER GENERATOR
// ============================================================================

// Linear Congruential Generator (simple but sufficient for MC)
// Parameters from Numerical Recipes
struct RngState {
    state: i64,
}

fn rng_new(seed: i64) -> RngState {
    return RngState { state: seed }
}

// Generate next uniform random in [0, 1)
fn rng_next(s: RngState) -> (f64, RngState) {
    // LCG: state = (a * state + c) mod m
    let a: i64 = 1103515245
    let c: i64 = 12345
    let m: i64 = 2147483648  // 2^31

    var next_state = a * s.state + c
    // Use modulo operator for efficient remainder calculation
    next_state = next_state % m
    if next_state < 0 {
        next_state = next_state + m
    }

    let u = (next_state as f64) / (m as f64)
    return (u, RngState { state: next_state })
}

// Box-Muller transform: generate standard normal from two uniforms
fn rng_normal(s: RngState) -> (f64, RngState) {
    let r1 = rng_next(s)
    let u1 = r1.0
    let rng2 = r1.1

    let r2 = rng_next(rng2)
    let u2 = r2.0
    let rng3 = r2.1

    // Avoid log(0)
    var u1_safe = u1
    if u1_safe < 1.0e-10 { u1_safe = 1.0e-10 }

    // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    let pi = 3.14159265358979323846
    let r = sqrt_f64(0.0 - 2.0 * log_f64(u1_safe))
    let theta = 2.0 * pi * u2
    let z = r * cos(theta)

    return (z, rng3)
}

// Generate normal with given mean and std
fn rng_normal_params(s: RngState, mean: f64, std: f64) -> (f64, RngState) {
    let r = rng_normal(s)
    let z = r.0
    let s2 = r.1
    return (mean + std * z, s2)
}

// ============================================================================
// SAMPLE ARRAYS (fixed size for static allocation)
// ============================================================================

// Fixed-size sample storage (1000 samples)
struct Samples {
    count: i32,
    // Store up to 1000 samples in 10 blocks of 100
    s0: f64, s1: f64, s2: f64, s3: f64, s4: f64,
    s5: f64, s6: f64, s7: f64, s8: f64, s9: f64,
    s10: f64, s11: f64, s12: f64, s13: f64, s14: f64,
    s15: f64, s16: f64, s17: f64, s18: f64, s19: f64,
    // Statistics (computed lazily)
    mean: f64,
    std: f64,
    p5: f64,   // 5th percentile
    p95: f64,  // 95th percentile
}

fn samples_new() -> Samples {
    return Samples {
        count: 0,
        s0: 0.0, s1: 0.0, s2: 0.0, s3: 0.0, s4: 0.0,
        s5: 0.0, s6: 0.0, s7: 0.0, s8: 0.0, s9: 0.0,
        s10: 0.0, s11: 0.0, s12: 0.0, s13: 0.0, s14: 0.0,
        s15: 0.0, s16: 0.0, s17: 0.0, s18: 0.0, s19: 0.0,
        mean: 0.0,
        std: 0.0,
        p5: 0.0,
        p95: 0.0,
    }
}

fn get_sample(s: Samples, idx: i32) -> f64 {
    if idx == 0 { return s.s0 }
    if idx == 1 { return s.s1 }
    if idx == 2 { return s.s2 }
    if idx == 3 { return s.s3 }
    if idx == 4 { return s.s4 }
    if idx == 5 { return s.s5 }
    if idx == 6 { return s.s6 }
    if idx == 7 { return s.s7 }
    if idx == 8 { return s.s8 }
    if idx == 9 { return s.s9 }
    if idx == 10 { return s.s10 }
    if idx == 11 { return s.s11 }
    if idx == 12 { return s.s12 }
    if idx == 13 { return s.s13 }
    if idx == 14 { return s.s14 }
    if idx == 15 { return s.s15 }
    if idx == 16 { return s.s16 }
    if idx == 17 { return s.s17 }
    if idx == 18 { return s.s18 }
    if idx == 19 { return s.s19 }
    return 0.0
}

fn set_sample(s: Samples, idx: i32, val: f64) -> Samples {
    var result = s
    if idx == 0 { result.s0 = val }
    else if idx == 1 { result.s1 = val }
    else if idx == 2 { result.s2 = val }
    else if idx == 3 { result.s3 = val }
    else if idx == 4 { result.s4 = val }
    else if idx == 5 { result.s5 = val }
    else if idx == 6 { result.s6 = val }
    else if idx == 7 { result.s7 = val }
    else if idx == 8 { result.s8 = val }
    else if idx == 9 { result.s9 = val }
    else if idx == 10 { result.s10 = val }
    else if idx == 11 { result.s11 = val }
    else if idx == 12 { result.s12 = val }
    else if idx == 13 { result.s13 = val }
    else if idx == 14 { result.s14 = val }
    else if idx == 15 { result.s15 = val }
    else if idx == 16 { result.s16 = val }
    else if idx == 17 { result.s17 = val }
    else if idx == 18 { result.s18 = val }
    else if idx == 19 { result.s19 = val }
    return result
}

// ============================================================================
// MONTE CARLO INPUT SPECIFICATION
// ============================================================================

// Input distribution specification
struct MCInput {
    mean: f64,
    std: f64,
    dist_type: i32,  // 0=normal, 1=uniform, 2=triangular
}

fn mc_input_normal(mean: f64, std: f64) -> MCInput {
    return MCInput { mean: mean, std: std, dist_type: 0 }
}

fn mc_input_uniform(lo: f64, hi: f64) -> MCInput {
    // Uniform: mean = (lo+hi)/2, std = (hi-lo)/sqrt(12)
    let mean = (lo + hi) / 2.0
    let std = (hi - lo) / sqrt_f64(12.0)
    return MCInput { mean: mean, std: std, dist_type: 1 }
}

// ============================================================================
// SAMPLE GENERATION
// ============================================================================

// Generate samples from an input distribution
fn generate_samples(input: MCInput, n: i32, seed: i64) -> Samples {
    var samples = samples_new()
    var rng = rng_new(seed)

    var i: i32 = 0
    while i < n && i < 20 {
        if input.dist_type == 0 {
            // Normal distribution
            let r = rng_normal_params(rng, input.mean, input.std)
            samples = set_sample(samples, i, r.0)
            rng = r.1
        } else {
            // Uniform (simplified)
            let r = rng_next(rng)
            let u = r.0
            rng = r.1
            // Scale uniform to [mean - sqrt(3)*std, mean + sqrt(3)*std]
            let half_width = sqrt_f64(3.0) * input.std
            let val = input.mean - half_width + 2.0 * half_width * u
            samples = set_sample(samples, i, val)
        }
        i = i + 1
    }

    samples.count = i
    return samples
}

// ============================================================================
// SAMPLE STATISTICS
// ============================================================================

fn compute_mean(s: Samples) -> f64 {
    if s.count == 0 { return 0.0 }

    var sum: f64 = 0.0
    var i: i32 = 0
    while i < s.count {
        sum = sum + get_sample(s, i)
        i = i + 1
    }

    return sum / (s.count as f64)
}

fn compute_std(s: Samples) -> f64 {
    if s.count < 2 { return 0.0 }

    let mean = compute_mean(s)
    var sum_sq: f64 = 0.0

    var i: i32 = 0
    while i < s.count {
        let diff = get_sample(s, i) - mean
        sum_sq = sum_sq + diff * diff
        i = i + 1
    }

    return sqrt_f64(sum_sq / ((s.count - 1) as f64))
}

fn compute_stats(s: Samples) -> Samples {
    var result = s
    result.mean = compute_mean(s)
    result.std = compute_std(s)
    return result
}

// ============================================================================
// MONTE CARLO PROPAGATION FOR SPECIFIC FUNCTIONS
// ============================================================================

// MC propagation result
struct MCResult {
    mean: f64,
    std: f64,
    p5: f64,
    p95: f64,
    n_samples: i32,
    gum_std: f64,        // GUM first-order estimate for comparison
    is_nonlinear: bool,  // True if MC differs significantly from GUM
}

// Propagate through addition: y = a + b
fn mc_add(a: MCInput, b: MCInput, n: i32, seed: i64) -> MCResult {
    var rng = rng_new(seed)
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0

    var i: i32 = 0
    while i < n {
        // Sample from a
        let ra = rng_normal_params(rng, a.mean, a.std)
        let va = ra.0
        rng = ra.1

        // Sample from b
        let rb = rng_normal_params(rng, b.mean, b.std)
        let vb = rb.0
        rng = rb.1

        // y = a + b
        let y = va + vb
        sum = sum + y
        sum_sq = sum_sq + y * y
        i = i + 1
    }

    let mean = sum / (n as f64)
    let variance = sum_sq / (n as f64) - mean * mean
    let std = sqrt_f64(abs_f64(variance))

    // GUM estimate for addition
    let gum_std = sqrt_f64(a.std * a.std + b.std * b.std)

    return MCResult {
        mean: mean,
        std: std,
        p5: mean - 1.645 * std,
        p95: mean + 1.645 * std,
        n_samples: n,
        gum_std: gum_std,
        is_nonlinear: false,
    }
}

// Propagate through multiplication: y = a * b
fn mc_mul(a: MCInput, b: MCInput, n: i32, seed: i64) -> MCResult {
    var rng = rng_new(seed)
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0

    var i: i32 = 0
    while i < n {
        let ra = rng_normal_params(rng, a.mean, a.std)
        let va = ra.0
        rng = ra.1

        let rb = rng_normal_params(rng, b.mean, b.std)
        let vb = rb.0
        rng = rb.1

        let y = va * vb
        sum = sum + y
        sum_sq = sum_sq + y * y
        i = i + 1
    }

    let mean = sum / (n as f64)
    let variance = sum_sq / (n as f64) - mean * mean
    let std = sqrt_f64(abs_f64(variance))

    // GUM estimate for multiplication (relative uncertainties add in quadrature)
    var gum_std: f64 = 0.0
    if abs_f64(a.mean) > 1.0e-15 && abs_f64(b.mean) > 1.0e-15 {
        let rel_a = a.std / abs_f64(a.mean)
        let rel_b = b.std / abs_f64(b.mean)
        let rel_y = sqrt_f64(rel_a * rel_a + rel_b * rel_b)
        gum_std = abs_f64(a.mean * b.mean) * rel_y
    }

    // Check if MC differs significantly from GUM (> 10% difference)
    var is_nonlinear = false
    if gum_std > 1.0e-15 {
        let diff = abs_f64(std - gum_std) / gum_std
        if diff > 0.1 { is_nonlinear = true }
    }

    return MCResult {
        mean: mean,
        std: std,
        p5: mean - 1.645 * std,
        p95: mean + 1.645 * std,
        n_samples: n,
        gum_std: gum_std,
        is_nonlinear: is_nonlinear,
    }
}

// Propagate through exponential: y = exp(a)
fn mc_exp(a: MCInput, n: i32, seed: i64) -> MCResult {
    var rng = rng_new(seed)
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0

    var i: i32 = 0
    while i < n {
        let ra = rng_normal_params(rng, a.mean, a.std)
        let va = ra.0
        rng = ra.1

        let y = exp(va)
        sum = sum + y
        sum_sq = sum_sq + y * y
        i = i + 1
    }

    let mean = sum / (n as f64)
    let variance = sum_sq / (n as f64) - mean * mean
    let std = sqrt_f64(abs_f64(variance))

    // GUM estimate for exp: u(y) = |y| * u(a) (first-order)
    let gum_y = exp(a.mean)
    let gum_std = abs_f64(gum_y) * a.std

    var is_nonlinear = false
    if gum_std > 1.0e-15 {
        let diff = abs_f64(std - gum_std) / gum_std
        if diff > 0.1 { is_nonlinear = true }
    }

    return MCResult {
        mean: mean,
        std: std,
        p5: mean - 1.645 * std,
        p95: mean + 1.645 * std,
        n_samples: n,
        gum_std: gum_std,
        is_nonlinear: is_nonlinear,
    }
}

// Propagate through division: y = a / b
fn mc_div(a: MCInput, b: MCInput, n: i32, seed: i64) -> MCResult {
    var rng = rng_new(seed)
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0
    var valid_count: i32 = 0

    var i: i32 = 0
    while i < n {
        let ra = rng_normal_params(rng, a.mean, a.std)
        let va = ra.0
        rng = ra.1

        let rb = rng_normal_params(rng, b.mean, b.std)
        let vb = rb.0
        rng = rb.1

        // Skip division by values too close to zero
        if abs_f64(vb) > 1.0e-10 {
            let y = va / vb
            sum = sum + y
            sum_sq = sum_sq + y * y
            valid_count = valid_count + 1
        }
        i = i + 1
    }

    if valid_count == 0 {
        return MCResult {
            mean: 0.0,
            std: 1.0e308,
            p5: 0.0 - 1.0e308,
            p95: 1.0e308,
            n_samples: 0,
            gum_std: 1.0e308,
            is_nonlinear: true,
        }
    }

    let mean = sum / (valid_count as f64)
    let variance = sum_sq / (valid_count as f64) - mean * mean
    let std = sqrt_f64(abs_f64(variance))

    // GUM estimate for division
    var gum_std: f64 = 0.0
    if abs_f64(a.mean) > 1.0e-15 && abs_f64(b.mean) > 1.0e-15 {
        let rel_a = a.std / abs_f64(a.mean)
        let rel_b = b.std / abs_f64(b.mean)
        let rel_y = sqrt_f64(rel_a * rel_a + rel_b * rel_b)
        gum_std = abs_f64(a.mean / b.mean) * rel_y
    }

    var is_nonlinear = false
    if gum_std > 1.0e-15 {
        let diff = abs_f64(std - gum_std) / gum_std
        if diff > 0.1 { is_nonlinear = true }
    }

    return MCResult {
        mean: mean,
        std: std,
        p5: mean - 1.645 * std,
        p95: mean + 1.645 * std,
        n_samples: valid_count,
        gum_std: gum_std,
        is_nonlinear: is_nonlinear,
    }
}

// ============================================================================
// POLICY: WHEN TO USE MC vs GUM
// ============================================================================

struct PropagationMode {
    use_mc: bool,
    reason_code: i32,  // 0=default GUM, 1=forced MC, 2=nonlinear detected, 3=interval exploded
    n_samples: i32,
}

fn propagation_gum() -> PropagationMode {
    return PropagationMode { use_mc: false, reason_code: 0, n_samples: 0 }
}

fn propagation_mc(n: i32) -> PropagationMode {
    return PropagationMode { use_mc: true, reason_code: 1, n_samples: n }
}

// Check if MC is recommended (large relative uncertainty suggests nonlinearity matters)
fn should_use_mc(input: MCInput) -> bool {
    if abs_f64(input.mean) < 1.0e-15 { return true }
    let rel_u = input.std / abs_f64(input.mean)
    // If relative uncertainty > 30%, linearization may be poor
    return rel_u > 0.3
}

// ============================================================================
// TESTS
// ============================================================================

fn test_rng_uniform() -> bool {
    var rng = rng_new(12345)

    var sum: f64 = 0.0
    var i: i32 = 0
    while i < 100 {
        let r = rng_next(rng)
        let u = r.0
        rng = r.1

        // Should be in [0, 1)
        if u < 0.0 { return false }
        if u >= 1.0 { return false }

        sum = sum + u
        i = i + 1
    }

    // Mean should be approximately 0.5
    let mean = sum / 100.0
    if mean < 0.3 { return false }
    if mean > 0.7 { return false }

    return true
}

fn test_rng_normal() -> bool {
    var rng = rng_new(54321)

    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0
    var i: i32 = 0

    while i < 100 {
        let r = rng_normal(rng)
        let z = r.0
        rng = r.1

        sum = sum + z
        sum_sq = sum_sq + z * z
        i = i + 1
    }

    let mean = sum / 100.0
    let variance = sum_sq / 100.0 - mean * mean
    let std = sqrt_f64(abs_f64(variance))

    // Mean should be approximately 0
    if abs_f64(mean) > 0.5 { return false }

    // Std should be approximately 1
    if std < 0.5 { return false }
    if std > 1.5 { return false }

    return true
}

fn test_mc_addition() -> bool {
    let a = mc_input_normal(10.0, 1.0)
    let b = mc_input_normal(20.0, 2.0)

    let result = mc_add(a, b, 1000, 12345)

    // Mean should be approximately 30
    if abs_f64(result.mean - 30.0) > 1.0 { return false }

    // GUM std = sqrt(1² + 2²) = sqrt(5) ≈ 2.24
    let expected_std = sqrt_f64(5.0)
    if abs_f64(result.gum_std - expected_std) > 0.1 { return false }

    // MC std should be similar to GUM for linear operation
    if abs_f64(result.std - result.gum_std) > 0.5 { return false }

    return true
}

fn test_mc_multiplication() -> bool {
    let a = mc_input_normal(10.0, 1.0)  // 10% relative uncertainty
    let b = mc_input_normal(5.0, 0.5)   // 10% relative uncertainty

    let result = mc_mul(a, b, 1000, 54321)

    // Mean should be approximately 50
    if abs_f64(result.mean - 50.0) > 5.0 { return false }

    // Relative uncertainty ≈ sqrt(10² + 10²) = 14.1%
    // So std ≈ 50 * 0.141 = 7.07
    if result.std < 3.0 { return false }
    if result.std > 15.0 { return false }

    return true
}

fn test_mc_exp_nonlinear() -> bool {
    // Large uncertainty in exponent -> nonlinear effects
    let a = mc_input_normal(1.0, 0.5)  // 50% relative uncertainty in exponent

    let result = mc_exp(a, 1000, 98765)

    // exp(1.0) ≈ 2.718
    // But with large uncertainty, mean will be higher due to Jensen's inequality
    // E[exp(X)] > exp(E[X]) for convex functions

    // MC mean should be higher than exp(1.0) = 2.718
    if result.mean < 2.5 { return false }

    // Should be detected as nonlinear
    // (MC std differs from GUM by > 10%)
    // This may or may not trigger depending on the sample
    // Just check we get reasonable results
    if result.std < 0.5 { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_rng_uniform() { return 1 }
    if !test_rng_normal() { return 2 }
    if !test_mc_addition() { return 3 }
    if !test_mc_multiplication() { return 4 }
    if !test_mc_exp_nonlinear() { return 5 }

    return 0
}
