// error_models.d - Residual Error Models for Population PK/PBPK
//
// Implements observation/measurement error models following
// MedLang Track D Pharmacometrics QSP Specification.
//
// Key features:
// - Proportional error (CV-based)
// - Additive error (absolute SD)
// - Combined error (proportional + additive)
// - Log-additive error
// - Likelihood calculations for NLME
//
// Reference: FDA Guidance on Population PK, NONMEM User Guide
// Inspired by MedLang (github.com/agourakis82/medlang)
//
// Module: pbpk::error_models (for future module system)

// =============================================================================
// MATH HELPERS (must be defined first - no forward declarations)
// =============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut guess = x / 2.0
    if guess < 1.0 { guess = 1.0 }
    let mut i = 0
    while i < 20 {
        guess = (guess + x / guess) / 2.0
        i = i + 1
    }
    return guess
}

fn exp_f64(x: f64) -> f64 {
    if x > 20.0 { return exp_f64(x / 2.0) * exp_f64(x / 2.0) }
    if x < 0.0 - 20.0 { return 1.0 / exp_f64(0.0 - x) }
    let mut sum = 1.0
    let mut term = 1.0
    let mut i = 1
    while i <= 15 {
        term = term * x / i
        sum = sum + term
        i = i + 1
    }
    return sum
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 10000000000.0 }
    let e = 2.718281828459045
    let mut val = x
    let mut k = 0.0
    while val > e {
        val = val / e
        k = k + 1.0
    }
    while val < 1.0 / e {
        val = val * e
        k = k - 1.0
    }
    let u = (val - 1.0) / (val + 1.0)
    let u2 = u * u
    let mut sum = u
    let mut term = u
    term = term * u2
    sum = sum + term / 3.0
    term = term * u2
    sum = sum + term / 5.0
    term = term * u2
    sum = sum + term / 7.0
    term = term * u2
    sum = sum + term / 9.0
    return 2.0 * sum + k
}

// Standard normal CDF approximation (Abramowitz & Stegun)
fn normal_cdf(z: f64) -> f64 {
    let a1 = 0.254829592
    let a2 = 0.0 - 0.284496736
    let a3 = 1.421413741
    let a4 = 0.0 - 1.453152027
    let a5 = 1.061405429
    let p = 0.3275911

    let sign = if z < 0.0 { 0.0 - 1.0 } else { 1.0 }
    let x = abs_f64(z) / sqrt_f64(2.0)

    let t = 1.0 / (1.0 + p * x)
    let t2 = t * t
    let t3 = t2 * t
    let t4 = t3 * t
    let t5 = t4 * t

    let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp_f64(0.0 - x * x)

    return 0.5 * (1.0 + sign * y)
}

// =============================================================================
// ERROR MODEL TYPES
// =============================================================================

// Proportional error model
// Y = F * (1 + eps)
// CV = sigma_prop
struct ProportionalError {
    sigma: f64   // Proportional SD (CV as fraction)
}

// Additive error model
// Y = F + eps
// SD = sigma_add (in concentration units)
struct AdditiveError {
    sigma: f64   // Additive SD (mg/L)
}

// Combined error model (most common in PK)
// Y = F * (1 + eps_prop) + eps_add
// Variance = (sigma_prop * F)^2 + sigma_add^2
struct CombinedError {
    sigma_prop: f64,   // Proportional SD
    sigma_add: f64     // Additive SD
}

// Log-additive error model
// ln(Y) = ln(F) + eps
// Equivalent to exponential error on original scale
struct LogAdditiveError {
    sigma: f64   // SD on log scale
}

// Exponential error model
// Y = F * exp(eps)
// Same as log-additive, different parameterization
struct ExponentialError {
    sigma: f64   // SD of exponent
}

// =============================================================================
// ERROR MODEL CONSTRUCTORS
// =============================================================================

fn proportional_error(cv: f64) -> ProportionalError {
    return ProportionalError { sigma: cv }
}

fn additive_error(sd: f64) -> AdditiveError {
    return AdditiveError { sigma: sd }
}

fn combined_error(cv: f64, sd: f64) -> CombinedError {
    return CombinedError { sigma_prop: cv, sigma_add: sd }
}

fn log_additive_error(sd: f64) -> LogAdditiveError {
    return LogAdditiveError { sigma: sd }
}

fn exponential_error(sd: f64) -> ExponentialError {
    return ExponentialError { sigma: sd }
}

// =============================================================================
// VARIANCE CALCULATIONS
// =============================================================================

// Variance for proportional error at prediction F
fn proportional_variance(err: ProportionalError, f_pred: f64) -> f64 {
    return (err.sigma * f_pred) * (err.sigma * f_pred)
}

// Variance for additive error (constant)
fn additive_variance(err: AdditiveError, f_pred: f64) -> f64 {
    return err.sigma * err.sigma
}

// Variance for combined error at prediction F
// Var = (sigma_prop * F)^2 + sigma_add^2
fn combined_variance(err: CombinedError, f_pred: f64) -> f64 {
    let prop_var = (err.sigma_prop * f_pred) * (err.sigma_prop * f_pred)
    let add_var = err.sigma_add * err.sigma_add
    return prop_var + add_var
}

// Variance for log-additive error
// On log scale, variance is sigma^2 (constant)
fn log_additive_variance(err: LogAdditiveError, f_pred: f64) -> f64 {
    return err.sigma * err.sigma
}

// Variance for exponential error at prediction F
// On original scale: Var ≈ (sigma * F)^2 for small sigma
fn exponential_variance(err: ExponentialError, f_pred: f64) -> f64 {
    // Exact: Var = F^2 * (exp(sigma^2) - 1)
    let sig2 = err.sigma * err.sigma
    let factor = exp_f64(sig2) - 1.0
    return f_pred * f_pred * factor
}

// =============================================================================
// STANDARD DEVIATION AT PREDICTION
// =============================================================================

fn proportional_sd(err: ProportionalError, f_pred: f64) -> f64 {
    return abs_f64(err.sigma * f_pred)
}

fn additive_sd(err: AdditiveError, f_pred: f64) -> f64 {
    return err.sigma
}

fn combined_sd(err: CombinedError, f_pred: f64) -> f64 {
    let variance = combined_variance(err, f_pred)
    return sqrt_f64(variance)
}

fn log_additive_sd(err: LogAdditiveError, f_pred: f64) -> f64 {
    return err.sigma
}

fn exponential_sd(err: ExponentialError, f_pred: f64) -> f64 {
    let variance = exponential_variance(err, f_pred)
    return sqrt_f64(variance)
}

// =============================================================================
// RESIDUAL CALCULATIONS (IWRES, WRES, etc.)
// =============================================================================

// Individual weighted residual (IWRES)
// IWRES = (DV - PRED) / SD
fn iwres_proportional(obs: f64, pred: f64, err: ProportionalError) -> f64 {
    let sd = proportional_sd(err, pred)
    if sd < 0.0000000001 { return 0.0 }
    return (obs - pred) / sd
}

fn iwres_additive(obs: f64, pred: f64, err: AdditiveError) -> f64 {
    let sd = additive_sd(err, pred)
    if sd < 0.0000000001 { return 0.0 }
    return (obs - pred) / sd
}

fn iwres_combined(obs: f64, pred: f64, err: CombinedError) -> f64 {
    let sd = combined_sd(err, pred)
    if sd < 0.0000000001 { return 0.0 }
    return (obs - pred) / sd
}

fn iwres_log_additive(obs: f64, pred: f64, err: LogAdditiveError) -> f64 {
    if obs <= 0.0 { return 0.0 }
    if pred <= 0.0 { return 0.0 }
    let log_obs = ln_f64(obs)
    let log_pred = ln_f64(pred)
    return (log_obs - log_pred) / err.sigma
}

// =============================================================================
// LIKELIHOOD CALCULATIONS
// =============================================================================

// Log-likelihood contribution for a single observation
// -2LL contribution = ln(2*pi*var) + (obs - pred)^2 / var
fn log_likelihood_normal(obs: f64, pred: f64, variance: f64) -> f64 {
    let pi = 3.14159265358979
    let resid = obs - pred
    let ll = 0.0 - 0.5 * (ln_f64(2.0 * pi * variance) + resid * resid / variance)
    return ll
}

// -2 Log-likelihood (for NONMEM-style objective function)
fn minus_2ll_normal(obs: f64, pred: f64, variance: f64) -> f64 {
    let ll = log_likelihood_normal(obs, pred, variance)
    return 0.0 - 2.0 * ll
}

// Log-likelihood for proportional error
fn log_likelihood_proportional(obs: f64, pred: f64, err: ProportionalError) -> f64 {
    let variance = proportional_variance(err, pred)
    return log_likelihood_normal(obs, pred, variance)
}

// Log-likelihood for additive error
fn log_likelihood_additive(obs: f64, pred: f64, err: AdditiveError) -> f64 {
    let variance = additive_variance(err, pred)
    return log_likelihood_normal(obs, pred, variance)
}

// Log-likelihood for combined error
fn log_likelihood_combined(obs: f64, pred: f64, err: CombinedError) -> f64 {
    let variance = combined_variance(err, pred)
    return log_likelihood_normal(obs, pred, variance)
}

// Log-likelihood for log-additive error (log-normal distribution)
fn log_likelihood_log_additive(obs: f64, pred: f64, err: LogAdditiveError) -> f64 {
    if obs <= 0.0 { return 0.0 - 10000000000.0 }  // Very negative for invalid
    if pred <= 0.0 { return 0.0 - 10000000000.0 }
    let log_obs = ln_f64(obs)
    let log_pred = ln_f64(pred)
    let variance = err.sigma * err.sigma
    // Log-normal: add -ln(obs) term for Jacobian
    let ll = log_likelihood_normal(log_obs, log_pred, variance) - ln_f64(obs)
    return ll
}

// =============================================================================
// SIMULATION WITH ERROR (FOR VPC)
// =============================================================================

// Simulate observation with proportional error
fn simulate_proportional(pred: f64, err: ProportionalError, eps: f64) -> f64 {
    // Y = F * (1 + sigma * eps) where eps ~ N(0,1)
    return pred * (1.0 + err.sigma * eps)
}

// Simulate observation with additive error
fn simulate_additive(pred: f64, err: AdditiveError, eps: f64) -> f64 {
    // Y = F + sigma * eps
    return pred + err.sigma * eps
}

// Simulate observation with combined error
fn simulate_combined(pred: f64, err: CombinedError, eps1: f64, eps2: f64) -> f64 {
    // Y = F * (1 + sigma_prop * eps1) + sigma_add * eps2
    return pred * (1.0 + err.sigma_prop * eps1) + err.sigma_add * eps2
}

// Simulate observation with log-additive error
fn simulate_log_additive(pred: f64, err: LogAdditiveError, eps: f64) -> f64 {
    // Y = F * exp(sigma * eps)
    return pred * exp_f64(err.sigma * eps)
}

// =============================================================================
// BELOW QUANTIFICATION LIMIT (BQL) HANDLING
// =============================================================================

// Check if observation is BQL
fn is_bql(obs: f64, lloq: f64) -> bool {
    return obs < lloq
}

// M3 method: Likelihood contribution for BQL observation
// P(Y < LLOQ) = Phi((LLOQ - PRED) / SD)
fn m3_bql_likelihood(pred: f64, lloq: f64, sd: f64) -> f64 {
    let z = (lloq - pred) / sd
    let prob = normal_cdf(z)
    if prob < 0.0000000001 { return 0.0 - 10000000000.0 }
    return ln_f64(prob)
}

// M4 method: Treat BQL as LLOQ/2 with additive error at LLOQ
fn m4_bql_imputation(lloq: f64) -> f64 {
    return lloq / 2.0
}

// =============================================================================
// GOODNESS OF FIT METRICS
// =============================================================================

// Calculate conditional weighted residuals squared (CWRESI^2)
fn cwresi_squared(obs: f64, pred: f64, sd: f64) -> f64 {
    if sd < 0.0000000001 { return 0.0 }
    let resid = (obs - pred) / sd
    return resid * resid
}

// Mean prediction error (MPE) - bias indicator
fn mean_prediction_error_3(obs1: f64, pred1: f64, obs2: f64, pred2: f64, obs3: f64, pred3: f64) -> f64 {
    let sum = (obs1 - pred1) + (obs2 - pred2) + (obs3 - pred3)
    return sum / 3.0
}

// Root mean squared error (RMSE) for 3 observations
fn rmse_3(obs1: f64, pred1: f64, obs2: f64, pred2: f64, obs3: f64, pred3: f64) -> f64 {
    let d1 = obs1 - pred1
    let d2 = obs2 - pred2
    let d3 = obs3 - pred3
    let sum_sq = d1 * d1 + d2 * d2 + d3 * d3
    return sqrt_f64(sum_sq / 3.0)
}

// =============================================================================
// TESTS
// =============================================================================

fn main() -> i32 {
    println("=== Error Model Tests ===")
    println("")

    // Test 1: Proportional variance
    println("Test 1: Proportional error variance")
    let err_prop = proportional_error(0.2)  // 20% CV
    let var_prop = proportional_variance(err_prop, 10.0)
    println("  CV=20%, PRED=10: Variance = ")
    println(var_prop)
    // Expected: (0.2 * 10)^2 = 4.0
    let err1 = abs_f64(var_prop - 4.0)
    println("")

    // Test 2: Combined variance
    println("Test 2: Combined error variance")
    let err_comb = combined_error(0.1, 0.5)  // 10% prop + 0.5 additive
    let var_comb = combined_variance(err_comb, 10.0)
    println("  CV=10%, SD=0.5, PRED=10: Variance = ")
    println(var_comb)
    // Expected: (0.1*10)^2 + 0.5^2 = 1.0 + 0.25 = 1.25
    let err2 = abs_f64(var_comb - 1.25)
    println("")

    // Test 3: IWRES calculation
    println("Test 3: IWRES (combined)")
    let iwres = iwres_combined(12.0, 10.0, err_comb)
    println("  OBS=12, PRED=10: IWRES = ")
    println(iwres)
    // Expected: (12-10) / sqrt(1.25) = 2 / 1.118 = 1.789
    let expected_iwres = 2.0 / sqrt_f64(1.25)
    let err3 = abs_f64(iwres - expected_iwres)
    println("")

    // Test 4: Log-likelihood
    println("Test 4: Log-likelihood (normal)")
    let ll = log_likelihood_normal(10.5, 10.0, 1.0)
    println("  OBS=10.5, PRED=10, VAR=1: LL = ")
    println(ll)
    // Expected: -0.5 * (ln(2*pi*1) + 0.25) = -0.5 * (1.838 + 0.25) = -1.044
    let pi = 3.14159265358979
    let expected_ll = 0.0 - 0.5 * (ln_f64(2.0 * pi) + 0.25)
    let err4 = abs_f64(ll - expected_ll)
    println("")

    // Test 5: Simulation with error
    println("Test 5: Simulation with proportional error")
    let pred = 10.0
    let eps = 1.0  // 1 SD above mean
    let sim = simulate_proportional(pred, err_prop, eps)
    println("  PRED=10, CV=20%, eps=1: Y = ")
    println(sim)
    // Expected: 10 * (1 + 0.2 * 1) = 12.0
    let err5 = abs_f64(sim - 12.0)
    println("")

    // Test 6: Normal CDF
    println("Test 6: Normal CDF")
    let cdf_0 = normal_cdf(0.0)
    let cdf_2 = normal_cdf(2.0)
    println("  Phi(0) = ")
    println(cdf_0)
    println("  Phi(2) = ")
    println(cdf_2)
    // Expected: Phi(0) = 0.5, Phi(2) ≈ 0.9772
    let err6a = abs_f64(cdf_0 - 0.5)
    let err6b = abs_f64(cdf_2 - 0.9772)
    println("")

    // Validation
    if err1 < 0.01 {
        if err2 < 0.01 {
            if err3 < 0.01 {
                if err4 < 0.1 {
                    if err5 < 0.01 {
                        if err6a < 0.01 {
                            if err6b < 0.01 {
                                println("ALL TESTS PASSED")
                                return 0
                            }
                        }
                    }
                }
            }
        }
    }

    println("SOME TESTS FAILED")
    println("  err1 = ")
    println(err1)
    println("  err2 = ")
    println(err2)
    println("  err3 = ")
    println(err3)
    println("  err4 = ")
    println(err4)
    println("  err5 = ")
    println(err5)
    println("  err6a = ")
    println(err6a)
    println("  err6b = ")
    println(err6b)
    return 1
}
