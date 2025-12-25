// random::sampling — Random Sampling Utilities
//
// Functions for sampling from collections, shuffling, and PK/PD variability.
//
// Features:
// - sample: Random selection from arrays
// - sample_weighted: Weighted random selection
// - shuffle: Random permutation
// - resample: Bootstrap resampling
// - pk_variability: PK parameter variability (IIV, IOV)
//
// References:
// - Knuth (1997): "The Art of Computer Programming, Vol. 2"
// - Lavielle (2014): "Mixed Effects Models for the Population Approach"

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

/// Generate integer in [0, n)
fn pcg64_bounded(rng: Pcg64, n: i64) -> (Pcg64, i64) {
    let result = pcg64_next_i64(rng)
    let x = result.1
    let positive = if x < 0 { 0 - x } else { x }
    let bounded = positive % n
    return (result.0, bounded)
}

// ============================================================================
// NORMAL DISTRIBUTION (for variability)
// ============================================================================

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn cos(x: f64) -> f64;
}

fn pi() -> f64 { 3.14159265358979323846 }

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
    return (current, 0.0000001)
}

fn sample_normal(mean: f64, std: f64, rng: Pcg64) -> (Pcg64, f64) {
    let r1 = pcg64_next_f64_nonzero(rng)
    let r2 = pcg64_next_f64(r1.0)

    let mag = sqrt(-2.0 * log(r1.1))
    let z = mag * cos(2.0 * pi() * r2.1)

    return (r2.0, mean + std * z)
}

// ============================================================================
// BASIC SAMPLING
// ============================================================================

/// Sample a single random index from [0, n)
fn sample_index(n: i64, rng: Pcg64) -> (Pcg64, i64) {
    return pcg64_bounded(rng, n)
}

/// Sample a single element from an f64 array
fn sample_one_f64(arr: [f64], rng: Pcg64) -> (Pcg64, f64) {
    let n = arr.len() as i64
    let idx_result = pcg64_bounded(rng, n)
    return (idx_result.0, arr[idx_result.1 as usize])
}

/// Sample a single element from an i64 array
fn sample_one_i64(arr: [i64], rng: Pcg64) -> (Pcg64, i64) {
    let n = arr.len() as i64
    let idx_result = pcg64_bounded(rng, n)
    return (idx_result.0, arr[idx_result.1 as usize])
}

// ============================================================================
// WEIGHTED SAMPLING
// ============================================================================

/// Sample index with weights (weights don't need to sum to 1)
fn sample_weighted_index(weights: [f64], rng: Pcg64) -> (Pcg64, i64) {
    // Compute total weight
    var total = 0.0
    var i: usize = 0
    while i < weights.len() {
        total = total + weights[i]
        i = i + 1
    }

    // Sample uniform [0, total)
    let u_result = pcg64_next_f64(rng)
    let threshold = u_result.1 * total

    // Find which bucket
    var cumsum = 0.0
    i = 0
    while i < weights.len() {
        cumsum = cumsum + weights[i]
        if cumsum > threshold {
            return (u_result.0, i as i64)
        }
        i = i + 1
    }

    // Fallback to last index
    return (u_result.0, (weights.len() - 1) as i64)
}

// ============================================================================
// SHUFFLING (Fisher-Yates)
// ============================================================================

/// Shuffle an array in place using Fisher-Yates
fn shuffle_f64(arr: [f64], rng: Pcg64) -> (Pcg64, [f64]) {
    var result = arr
    var current_rng = rng
    var i = result.len()

    while i > 1 {
        i = i - 1
        let j_result = pcg64_bounded(current_rng, (i + 1) as i64)
        current_rng = j_result.0
        let j = j_result.1 as usize

        // Swap result[i] and result[j]
        let temp = result[i]
        result[i] = result[j]
        result[j] = temp
    }

    return (current_rng, result)
}

/// Shuffle an i64 array
fn shuffle_i64(arr: [i64], rng: Pcg64) -> (Pcg64, [i64]) {
    var result = arr
    var current_rng = rng
    var i = result.len()

    while i > 1 {
        i = i - 1
        let j_result = pcg64_bounded(current_rng, (i + 1) as i64)
        current_rng = j_result.0
        let j = j_result.1 as usize

        let temp = result[i]
        result[i] = result[j]
        result[j] = temp
    }

    return (current_rng, result)
}

// ============================================================================
// RESAMPLING (BOOTSTRAP)
// ============================================================================

/// Bootstrap resample: sample n items with replacement
fn resample_f64(arr: [f64], n: i64, rng: Pcg64) -> (Pcg64, [f64]) {
    var result: [f64] = []
    var current_rng = rng
    let arr_len = arr.len() as i64

    var i: i64 = 0
    while i < n {
        let idx_result = pcg64_bounded(current_rng, arr_len)
        current_rng = idx_result.0
        result.push(arr[idx_result.1 as usize])
        i = i + 1
    }

    return (current_rng, result)
}

// ============================================================================
// PK/PD VARIABILITY
// ============================================================================

/// Inter-Individual Variability (IIV) parameters
struct IIV {
    omega_cl: f64,   // CV for clearance
    omega_vc: f64,   // CV for central volume
    omega_ka: f64,   // CV for absorption rate
}

fn iiv_new(omega_cl: f64, omega_vc: f64, omega_ka: f64) -> IIV {
    IIV {
        omega_cl: omega_cl,
        omega_vc: omega_vc,
        omega_ka: omega_ka,
    }
}

fn iiv_typical() -> IIV {
    // Typical PK variability (30% CV)
    IIV {
        omega_cl: 0.3,
        omega_vc: 0.3,
        omega_ka: 0.5,
    }
}

/// Generate individual PK parameters with IIV
struct IndividualPK {
    cl: f64,   // Individual clearance
    vc: f64,   // Individual central volume
    ka: f64,   // Individual absorption rate constant
}

fn generate_individual_pk(
    pop_cl: f64, pop_vc: f64, pop_ka: f64,
    iiv: IIV, rng: Pcg64
) -> (Pcg64, IndividualPK) {
    // Sample eta values (random effects)
    let eta_cl_result = sample_normal(0.0, iiv.omega_cl, rng)
    let eta_vc_result = sample_normal(0.0, iiv.omega_vc, eta_cl_result.0)
    let eta_ka_result = sample_normal(0.0, iiv.omega_ka, eta_vc_result.0)

    // Apply exponential transformation
    let ind = IndividualPK {
        cl: pop_cl * exp(eta_cl_result.1),
        vc: pop_vc * exp(eta_vc_result.1),
        ka: pop_ka * exp(eta_ka_result.1),
    }

    return (eta_ka_result.0, ind)
}

/// Generate a virtual population
fn generate_population(
    pop_cl: f64, pop_vc: f64, pop_ka: f64,
    iiv: IIV, n_subjects: i64, rng: Pcg64
) -> (Pcg64, [IndividualPK]) {
    var population: [IndividualPK] = []
    var current_rng = rng

    var i: i64 = 0
    while i < n_subjects {
        let result = generate_individual_pk(pop_cl, pop_vc, pop_ka, iiv, current_rng)
        current_rng = result.0
        population.push(result.1)
        i = i + 1
    }

    return (current_rng, population)
}

// ============================================================================
// RESIDUAL ERROR MODELS
// ============================================================================

/// Add proportional residual error
fn add_proportional_error(value: f64, sigma: f64, rng: Pcg64) -> (Pcg64, f64) {
    let eps_result = sample_normal(0.0, sigma, rng)
    return (eps_result.0, value * (1.0 + eps_result.1))
}

/// Add additive residual error
fn add_additive_error(value: f64, sigma: f64, rng: Pcg64) -> (Pcg64, f64) {
    let eps_result = sample_normal(0.0, sigma, rng)
    return (eps_result.0, value + eps_result.1)
}

/// Add combined (additive + proportional) residual error
fn add_combined_error(value: f64, sigma_add: f64, sigma_prop: f64, rng: Pcg64) -> (Pcg64, f64) {
    let eps1_result = sample_normal(0.0, sigma_add, rng)
    let eps2_result = sample_normal(0.0, sigma_prop, eps1_result.0)
    return (eps2_result.0, value * (1.0 + eps2_result.1) + eps1_result.1)
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn main() -> i32 {
    print("Testing random::sampling module...\n")

    var rng = pcg64_new(42)

    // Test sample_index
    var i = 0
    while i < 100 {
        let result = sample_index(10, rng)
        rng = result.0
        if result.1 < 0 || result.1 >= 10 { return 1 }
        i = i + 1
    }
    print("sample_index: PASS\n")

    // Test sample_one_f64
    let arr: [f64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    i = 0
    while i < 50 {
        let result = sample_one_f64(arr, rng)
        rng = result.0
        if result.1 < 1.0 || result.1 > 5.0 { return 2 }
        i = i + 1
    }
    print("sample_one_f64: PASS\n")

    // Test weighted sampling
    let weights: [f64] = [1.0, 0.0, 0.0]  // Should always pick index 0
    i = 0
    while i < 20 {
        let result = sample_weighted_index(weights, rng)
        rng = result.0
        if result.1 != 0 { return 3 }
        i = i + 1
    }
    print("sample_weighted_index: PASS\n")

    // Test shuffle
    let to_shuffle: [f64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    let shuffled_result = shuffle_f64(to_shuffle, rng)
    rng = shuffled_result.0
    // Check same length
    if shuffled_result.1.len() != 5 { return 4 }
    print("shuffle_f64: PASS\n")

    // Test resample
    let to_resample: [f64] = [1.0, 2.0, 3.0]
    let resampled_result = resample_f64(to_resample, 10, rng)
    rng = resampled_result.0
    if resampled_result.1.len() != 10 { return 5 }
    print("resample_f64: PASS\n")

    // Test IIV
    let iiv = iiv_typical()
    let pk_result = generate_individual_pk(10.0, 50.0, 1.0, iiv, rng)
    rng = pk_result.0
    // Individual params should be positive
    if pk_result.1.cl <= 0.0 { return 6 }
    if pk_result.1.vc <= 0.0 { return 7 }
    if pk_result.1.ka <= 0.0 { return 8 }
    print("IIV generation: PASS\n")

    // Test population generation
    let pop_result = generate_population(10.0, 50.0, 1.0, iiv, 5, rng)
    rng = pop_result.0
    if pop_result.1.len() != 5 { return 9 }
    print("Population generation: PASS\n")

    // Test residual error
    let err_result = add_proportional_error(100.0, 0.1, rng)
    rng = err_result.0
    // Value should be roughly around 100 with 10% error
    if err_result.1 < 50.0 || err_result.1 > 200.0 { return 10 }
    print("Residual error: PASS\n")

    print("All random::sampling tests PASSED\n")
    0
}
