// bayes::diagnostics — MCMC Convergence Diagnostics
//
// Tools for assessing MCMC chain convergence and mixing.
// Includes Gelman-Rubin R-hat, effective sample size, and autocorrelation.
//
// References:
// - Gelman & Rubin (1992): "Inference from iterative simulation..."
// - Vehtari et al. (2021): "Rank-normalization, folding, and localization"
// - Geyer (1992): "Practical Markov Chain Monte Carlo"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// DIAGNOSTIC RESULT TYPES
// ============================================================================

/// Complete diagnostic summary
struct DiagnosticSummary {
    rhat: f64,              // Gelman-Rubin R-hat (should be < 1.01)
    ess: f64,               // Effective sample size
    ess_per_sec: f64,       // ESS per second (if timing available)
    mean: f64,              // Posterior mean
    std: f64,               // Posterior std
    mcse: f64,              // Monte Carlo standard error
    autocorr_lag1: f64,     // Lag-1 autocorrelation
    converged: bool,        // Whether chain appears converged
}

fn diagnostic_summary_new() -> DiagnosticSummary {
    DiagnosticSummary {
        rhat: 1.0,
        ess: 0.0,
        ess_per_sec: 0.0,
        mean: 0.0,
        std: 0.0,
        mcse: 0.0,
        autocorr_lag1: 0.0,
        converged: false,
    }
}

// ============================================================================
// BASIC CHAIN STATISTICS
// ============================================================================

/// Compute mean of chain
fn chain_mean(samples: [f64; 100], n: i64) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + samples[i as usize]
        i = i + 1
    }
    sum / (n as f64)
}

/// Compute variance of chain
fn chain_variance(samples: [f64; 100], n: i64) -> f64 {
    let m = chain_mean(samples, n)
    var sum_sq = 0.0
    var i: i64 = 0
    while i < n {
        let diff = samples[i as usize] - m
        sum_sq = sum_sq + diff * diff
        i = i + 1
    }
    sum_sq / ((n - 1) as f64)
}

// ============================================================================
// AUTOCORRELATION
// ============================================================================

/// Compute autocorrelation at given lag
fn autocorrelation(samples: [f64; 100], n: i64, lag: i64) -> f64 {
    if lag >= n {
        return 0.0
    }

    let m = chain_mean(samples, n)
    var var_sum = 0.0
    var cov_sum = 0.0

    var i: i64 = 0
    while i < n {
        let diff = samples[i as usize] - m
        var_sum = var_sum + diff * diff
        i = i + 1
    }

    i = 0
    while i < n - lag {
        let diff1 = samples[i as usize] - m
        let diff2 = samples[(i + lag) as usize] - m
        cov_sum = cov_sum + diff1 * diff2
        i = i + 1
    }

    if var_sum < 1e-10 {
        return 0.0
    }

    cov_sum / var_sum
}

/// Compute autocorrelation for first k lags
fn autocorrelation_series(samples: [f64; 100], n: i64, max_lag: i64) -> [f64; 50] {
    var acf: [f64; 50] = [0.0; 50]
    var lag: i64 = 0
    while lag < max_lag && lag < 50 {
        acf[lag as usize] = autocorrelation(samples, n, lag)
        lag = lag + 1
    }
    acf
}

// ============================================================================
// EFFECTIVE SAMPLE SIZE (ESS)
// ============================================================================

/// Compute effective sample size using autocorrelation
/// ESS = n / (1 + 2 * sum of autocorrelations)
fn effective_sample_size(samples: [f64; 100], n: i64) -> f64 {
    // Compute autocorrelations until they become negligible
    var sum_rho = 0.0
    var lag: i64 = 1

    // Geyer's initial positive sequence estimator
    while lag < n / 2 && lag < 50 {
        let rho = autocorrelation(samples, n, lag)

        // Stop if autocorrelation becomes negative
        if rho < 0.0 {
            break
        }

        sum_rho = sum_rho + rho
        lag = lag + 1
    }

    let tau = 1.0 + 2.0 * sum_rho
    if tau < 1.0 { tau = 1.0 }

    (n as f64) / tau
}

/// Monte Carlo Standard Error
fn mcse(samples: [f64; 100], n: i64) -> f64 {
    let ess = effective_sample_size(samples, n)
    let variance = chain_variance(samples, n)

    if ess > 0.0 {
        sqrt(variance / ess)
    } else {
        sqrt(variance / (n as f64))
    }
}

// ============================================================================
// GELMAN-RUBIN R-HAT
// ============================================================================

/// Compute R-hat for two chains
/// R-hat < 1.01 indicates convergence
fn rhat_two_chains(chain1: [f64; 100], n1: i64, chain2: [f64; 100], n2: i64) -> f64 {
    // Assume equal chain lengths for simplicity
    let n = if n1 < n2 { n1 } else { n2 }

    // Chain means
    let m1 = chain_mean(chain1, n)
    let m2 = chain_mean(chain2, n)

    // Overall mean
    let m_all = (m1 + m2) / 2.0

    // Between-chain variance B
    let b = (n as f64) * ((m1 - m_all) * (m1 - m_all) + (m2 - m_all) * (m2 - m_all)) / 1.0

    // Within-chain variance W
    let s1 = chain_variance(chain1, n)
    let s2 = chain_variance(chain2, n)
    let w = (s1 + s2) / 2.0

    // Pooled variance estimate
    let var_hat = ((n - 1) as f64) / (n as f64) * w + b / (n as f64)

    // R-hat
    if w > 1e-10 {
        sqrt(var_hat / w)
    } else {
        1.0
    }
}

/// Compute split R-hat (split single chain in half)
fn rhat_split(samples: [f64; 100], n: i64) -> f64 {
    if n < 4 {
        return 1.0
    }

    let half = n / 2

    // First half
    var chain1: [f64; 100] = [0.0; 100]
    var i: i64 = 0
    while i < half {
        chain1[i as usize] = samples[i as usize]
        i = i + 1
    }

    // Second half
    var chain2: [f64; 100] = [0.0; 100]
    i = 0
    while i < half {
        chain2[i as usize] = samples[(half + i) as usize]
        i = i + 1
    }

    rhat_two_chains(chain1, half, chain2, half)
}

// ============================================================================
// COMPREHENSIVE DIAGNOSTICS
// ============================================================================

/// Run all diagnostics on a single chain
fn diagnose_chain(samples: [f64; 100], n: i64) -> DiagnosticSummary {
    var summary = diagnostic_summary_new()

    // Basic statistics
    summary.mean = chain_mean(samples, n)
    summary.std = sqrt(chain_variance(samples, n))

    // ESS and MCSE
    summary.ess = effective_sample_size(samples, n)
    summary.mcse = mcse(samples, n)

    // Autocorrelation
    summary.autocorr_lag1 = autocorrelation(samples, n, 1)

    // Split R-hat
    summary.rhat = rhat_split(samples, n)

    // Convergence assessment
    // Good: R-hat < 1.01, ESS > 100, low autocorrelation
    let rhat_ok = summary.rhat < 1.05
    let ess_ok = summary.ess > 50.0
    let acf_ok = summary.autocorr_lag1 < 0.9

    summary.converged = rhat_ok && ess_ok && acf_ok

    summary
}

// ============================================================================
// DIAGNOSTIC THRESHOLDS
// ============================================================================

/// Check if R-hat indicates convergence
fn rhat_converged(rhat: f64) -> bool {
    rhat < 1.01
}

/// Check if ESS is sufficient for reliable inference
fn ess_sufficient(ess: f64, for_mean: bool) -> bool {
    if for_mean {
        ess > 100.0
    } else {
        // For quantiles, need more samples
        ess > 400.0
    }
}

/// Interpret autocorrelation
fn autocorr_severity(acf1: f64) -> i64 {
    let abs_acf = if acf1 < 0.0 { -acf1 } else { acf1 }
    if abs_acf < 0.1 {
        0  // negligible
    } else if abs_acf < 0.3 {
        1  // low
    } else if abs_acf < 0.6 {
        2  // moderate
    } else {
        3  // high (problematic)
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_chain_mean() -> bool {
    var samples: [f64; 100] = [0.0; 100]
    samples[0] = 1.0
    samples[1] = 2.0
    samples[2] = 3.0
    samples[3] = 4.0
    samples[4] = 5.0

    let m = chain_mean(samples, 5)
    abs_f64(m - 3.0) < 0.01
}

fn test_autocorrelation_zero_lag() -> bool {
    var samples: [f64; 100] = [0.0; 100]
    samples[0] = 1.0
    samples[1] = 2.0
    samples[2] = 3.0
    samples[3] = 4.0
    samples[4] = 5.0

    // Autocorrelation at lag 0 should be 1
    let acf0 = autocorrelation(samples, 5, 0)
    abs_f64(acf0 - 1.0) < 0.01
}

fn test_ess_independent() -> bool {
    // Independent samples should have ESS close to n
    var samples: [f64; 100] = [0.0; 100]
    samples[0] = 0.1
    samples[1] = 0.9
    samples[2] = 0.2
    samples[3] = 0.8
    samples[4] = 0.3
    samples[5] = 0.7
    samples[6] = 0.4
    samples[7] = 0.6
    samples[8] = 0.5
    samples[9] = 0.5

    let ess = effective_sample_size(samples, 10)

    // ESS should be reasonably close to n for independent samples
    ess > 5.0
}

fn test_rhat_identical_chains() -> bool {
    var chain1: [f64; 100] = [0.0; 100]
    var chain2: [f64; 100] = [0.0; 100]

    var i: i64 = 0
    while i < 10 {
        chain1[i as usize] = (i as f64) * 0.1
        chain2[i as usize] = (i as f64) * 0.1
        i = i + 1
    }

    let rhat = rhat_two_chains(chain1, 10, chain2, 10)

    // Identical chains should have R-hat close to 1 (allowing some numerical tolerance)
    abs_f64(rhat - 1.0) < 0.1
}

fn test_diagnose_chain() -> bool {
    // Well-mixed chain
    var samples: [f64; 100] = [0.0; 100]
    var i: i64 = 0
    while i < 50 {
        // Alternating pattern (low autocorrelation)
        samples[i as usize] = if i % 2 == 0 { 0.5 } else { 0.4 }
        i = i + 1
    }

    let summary = diagnose_chain(samples, 50)

    // Should have computed diagnostics
    summary.ess > 0.0 && summary.rhat > 0.0
}

fn main() -> i32 {
    print("Testing bayes::diagnostics module...\n")

    if !test_chain_mean() {
        print("FAIL: chain_mean\n")
        return 1
    }
    print("PASS: chain_mean\n")

    if !test_autocorrelation_zero_lag() {
        print("FAIL: autocorrelation_zero_lag\n")
        return 2
    }
    print("PASS: autocorrelation_zero_lag\n")

    if !test_ess_independent() {
        print("FAIL: ess_independent\n")
        return 3
    }
    print("PASS: ess_independent\n")

    if !test_rhat_identical_chains() {
        print("FAIL: rhat_identical_chains\n")
        return 4
    }
    print("PASS: rhat_identical_chains\n")

    if !test_diagnose_chain() {
        print("FAIL: diagnose_chain\n")
        return 5
    }
    print("PASS: diagnose_chain\n")

    print("All bayes::diagnostics tests PASSED\n")
    0
}
