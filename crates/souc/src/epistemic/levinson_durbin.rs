//! Levinson-Durbin Recursion and PACF
//!
//! Implements the Levinson-Durbin algorithm (Durbin 1960, Levinson 1947) for
//! efficient AR parameter estimation from autocovariance sequences, plus the
//! partial autocorrelation function (PACF) needed for AR order identification.
//!
//! This fills the gap in `stdlib/stats/timeseries/arima.sio` where AR(p≥3)
//! fitting was marked TODO. The Rust implementation here can be called from
//! the Sounio runtime via FFI, or used directly in the compiler's epistemic
//! analysis pipeline.
//!
//! # References
//!
//! - Levinson, N. (1947). The Wiener RMS error criterion in filter design and
//!   prediction. J. Math. Phys., 25, 261-278.
//! - Durbin, J. (1960). The fitting of time-series models. Rev. Inst. Int. Stat.,
//!   28, 233-244.
//! - Burg, J. P. (1975). Maximum entropy spectral analysis. PhD thesis, Stanford.

/// Result of Levinson-Durbin recursion.
#[derive(Debug, Clone, PartialEq)]
pub struct LevinsonDurbinResult {
    /// AR coefficients φ₁, φ₂, ..., φ_p
    pub ar_coeffs: Vec<f64>,
    /// Reflection coefficients k₁, k₂, ..., k_p (these are the PACF values)
    pub reflection_coeffs: Vec<f64>,
    /// Prediction error variance at each stage: σ²₁, σ²₂, ..., σ²_p
    pub prediction_errors: Vec<f64>,
    /// Final prediction error variance σ²_p
    pub final_variance: f64,
}

/// Result of AR model fitting.
#[derive(Debug, Clone, PartialEq)]
pub struct ArFitResult {
    /// AR coefficients φ₁, ..., φ_p
    pub coeffs: Vec<f64>,
    /// Selected AR order
    pub order: usize,
    /// Innovation (residual) variance
    pub innovation_variance: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
}

/// Levinson-Durbin recursion for solving the Yule-Walker equations.
///
/// Given autocovariance values `r(0), r(1), ..., r(p)` (where `acf[0] = r(0)`
/// is the variance), computes AR coefficients for an AR(p) model.
///
/// The algorithm runs in O(p²) time and O(p) space, versus O(p³) for direct
/// matrix inversion of the Toeplitz system.
///
/// # Panics
///
/// Returns `None` if `acf` has fewer than 2 elements or if `acf[0]` is zero
/// (constant series).
pub fn levinson_durbin(acf: &[f64]) -> Option<LevinsonDurbinResult> {
    if acf.len() < 2 {
        return None;
    }

    let p = acf.len() - 1; // order = number of lags provided
    let r0 = acf[0];

    if r0.abs() < 1e-15 {
        return None; // constant series, zero variance
    }

    let mut ar_coeffs = vec![0.0; p];
    let mut ar_prev = vec![0.0; p];
    let mut reflection_coeffs = Vec::with_capacity(p);
    let mut prediction_errors = Vec::with_capacity(p);

    // Stage 1: k₁ = r(1) / r(0)
    let k1 = acf[1] / r0;
    ar_coeffs[0] = k1;
    let sigma_sq = r0 * (1.0 - k1 * k1);

    reflection_coeffs.push(k1);
    prediction_errors.push(sigma_sq);

    if sigma_sq.abs() < 1e-15 {
        // Perfect prediction at order 1
        return Some(LevinsonDurbinResult {
            ar_coeffs: vec![k1],
            reflection_coeffs,
            prediction_errors,
            final_variance: sigma_sq,
        });
    }

    let mut sigma_sq = sigma_sq;

    // Stages 2..p
    for m in 1..p {
        // Compute k_{m+1} = (r(m+1) - Σ_{j=0}^{m-1} φ_{m,j+1} * r(m+1-j-1)) / σ²_m
        //                  = (r(m+1) - Σ_{j=1}^{m} φ_{m,j} * r(m+1-j)) / σ²_m
        let mut num = acf[m + 1];
        for j in 0..m {
            num -= ar_coeffs[j] * acf[m - j];
        }
        let k = num / sigma_sq;

        // Update AR coefficients: φ_{m+1,j} = φ_{m,j} - k * φ_{m, m+1-j}
        ar_prev[..m].copy_from_slice(&ar_coeffs[..m]);
        for j in 0..m {
            ar_coeffs[j] = ar_prev[j] - k * ar_prev[m - 1 - j];
        }
        ar_coeffs[m] = k;

        sigma_sq *= 1.0 - k * k;
        reflection_coeffs.push(k);
        prediction_errors.push(sigma_sq);

        if sigma_sq.abs() < 1e-15 {
            // Perfect prediction achieved
            let final_coeffs = ar_coeffs[..=m].to_vec();
            return Some(LevinsonDurbinResult {
                ar_coeffs: final_coeffs,
                reflection_coeffs,
                prediction_errors,
                final_variance: sigma_sq,
            });
        }
    }

    Some(LevinsonDurbinResult {
        ar_coeffs,
        reflection_coeffs,
        prediction_errors,
        final_variance: sigma_sq,
    })
}

/// Compute the Partial Autocorrelation Function (PACF) from autocovariance.
///
/// The PACF at lag k equals the k-th reflection coefficient from the
/// Levinson-Durbin recursion. Useful for AR order identification:
/// for a true AR(p) process, PACF cuts off after lag p.
pub fn pacf(acf: &[f64]) -> Vec<f64> {
    match levinson_durbin(acf) {
        Some(result) => result.reflection_coeffs,
        None => Vec::new(),
    }
}

/// Compute sample autocovariance from a time series.
///
/// Returns `r(0), r(1), ..., r(max_lag)` where `r(k) = (1/n) Σ (x_t - x̄)(x_{t+k} - x̄)`.
pub fn sample_autocovariance(series: &[f64], max_lag: usize) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return vec![0.0];
    }

    let mean = series.iter().sum::<f64>() / n as f64;
    let max_lag = max_lag.min(n - 1);
    let mut acv = Vec::with_capacity(max_lag + 1);

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for t in 0..(n - lag) {
            sum += (series[t] - mean) * (series[t + lag] - mean);
        }
        acv.push(sum / n as f64);
    }

    acv
}

/// Fit an AR(p) model to a time series using Levinson-Durbin.
///
/// Computes the sample autocovariance, runs Levinson-Durbin, and returns
/// AR coefficients with information criteria for model selection.
pub fn fit_ar_levinson(series: &[f64], order: usize) -> Option<ArFitResult> {
    let n = series.len();
    if n <= order || order == 0 {
        return None;
    }

    let acv = sample_autocovariance(series, order);
    let ld = levinson_durbin(&acv)?;

    let sigma_sq = ld.final_variance;
    let nf = n as f64;

    // Log-likelihood for Gaussian AR model
    let ll = -0.5 * nf * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma_sq.ln());

    // Information criteria
    let k = (order + 1) as f64; // p AR coeffs + variance parameter
    let aic = -2.0 * ll + 2.0 * k;
    let bic = -2.0 * ll + k * nf.ln();

    Some(ArFitResult {
        coeffs: ld.ar_coeffs,
        order,
        innovation_variance: sigma_sq,
        aic,
        bic,
        log_likelihood: ll,
    })
}

/// Select AR order by minimizing AIC over AR(1) through AR(max_order).
///
/// Returns the order with the lowest AIC. If all fits fail, returns 1.
pub fn ar_order_select(series: &[f64], max_order: usize) -> usize {
    let max_order = max_order.min(series.len().saturating_sub(1)).max(1);

    let mut best_order = 1;
    let mut best_aic = f64::INFINITY;

    for p in 1..=max_order {
        if let Some(fit) = fit_ar_levinson(series, p) {
            if fit.aic < best_aic {
                best_aic = fit.aic;
                best_order = p;
            }
        }
    }

    best_order
}

// =============================================================================
// C-callable FFI for Sounio runtime
// =============================================================================

/// Compute AR coefficients via Levinson-Durbin from autocovariance.
///
/// # Safety
///
/// `acf_ptr` must point to `acf_len` valid f64 values.
/// `out_coeffs` must point to space for at least `acf_len - 1` f64 values.
/// `out_variance` must be a valid pointer.
/// Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_levinson_durbin(
    acf_ptr: *const f64,
    acf_len: usize,
    out_coeffs: *mut f64,
    out_variance: *mut f64,
) -> i32 {
    if acf_ptr.is_null() || out_coeffs.is_null() || out_variance.is_null() || acf_len < 2 {
        return -1;
    }

    let acf = unsafe {
        // SAFETY: pointers were checked for null above, `acf_len >= 2`, and the caller
        // guarantees `acf_ptr` points to `acf_len` readable `f64` values.
        std::slice::from_raw_parts(acf_ptr, acf_len)
    };

    match levinson_durbin(acf) {
        Some(result) => {
            for (i, &c) in result.ar_coeffs.iter().enumerate() {
                unsafe {
                    // SAFETY: caller guarantees `out_coeffs` has space for at least
                    // `acf_len - 1` `f64` values, and `result.ar_coeffs.len() <= acf_len - 1`.
                    *out_coeffs.add(i) = c;
                }
            }
            unsafe {
                // SAFETY: `out_variance` is non-null (validated above) and the caller
                // guarantees it is a valid writable pointer.
                *out_variance = result.final_variance;
            }
            0
        }
        None => -1,
    }
}

/// Compute PACF from autocovariance.
///
/// # Safety
///
/// `acf_ptr` must point to `acf_len` valid f64 values.
/// `out_pacf` must point to space for at least `acf_len - 1` f64 values.
/// Returns the number of PACF values written, or -1 on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_pacf(
    acf_ptr: *const f64,
    acf_len: usize,
    out_pacf: *mut f64,
) -> i32 {
    if acf_ptr.is_null() || out_pacf.is_null() || acf_len < 2 {
        return -1;
    }

    let acf = unsafe {
        // SAFETY: pointers were checked for null above, `acf_len >= 2`, and the caller
        // guarantees `acf_ptr` points to `acf_len` readable `f64` values.
        std::slice::from_raw_parts(acf_ptr, acf_len)
    };
    let result = pacf(acf);

    for (i, &v) in result.iter().enumerate() {
        unsafe {
            // SAFETY: caller guarantees `out_pacf` has space for at least `acf_len - 1`
            // values, and `result.len() <= acf_len - 1`.
            *out_pacf.add(i) = v;
        }
    }

    result.len() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AR(1) with φ₁ = 0.8: autocovariance is r(k) = σ² · 0.8^k
    /// With σ² = 1/(1-0.64) = 1/0.36:
    /// r(0) = 1/0.36, r(1) = 0.8/0.36, etc.
    #[test]
    fn test_ar1_recovery() {
        let rho: f64 = 0.8;
        let var = 1.0 / (1.0 - rho * rho);
        let acf = vec![var, var * rho, var * rho.powi(2), var * rho.powi(3)];

        let result = levinson_durbin(&acf).unwrap();

        assert!(
            (result.ar_coeffs[0] - 0.8).abs() < 1e-10,
            "AR(1) φ₁ should be 0.8, got {}",
            result.ar_coeffs[0]
        );
        // Reflection coefficients beyond lag 1 should be ~0 for AR(1)
        for i in 1..result.reflection_coeffs.len() {
            assert!(
                result.reflection_coeffs[i].abs() < 1e-10,
                "PACF({}) should be ~0 for AR(1), got {}",
                i + 1,
                result.reflection_coeffs[i]
            );
        }
    }

    /// AR(2) with φ₁ = 0.5, φ₂ = -0.3
    /// Yule-Walker: r(1) = φ₁·r(0) + φ₂·r(1)  →  ρ(1) = φ₁/(1-φ₂)
    ///              r(2) = φ₁·r(1) + φ₂·r(0)  →  ρ(2) = φ₁·ρ(1) + φ₂
    #[test]
    fn test_ar2_recovery() {
        let phi1 = 0.5;
        let phi2 = -0.3;

        // Compute theoretical autocorrelation
        let rho1 = phi1 / (1.0 - phi2);
        let rho2 = phi1 * rho1 + phi2;
        let rho3 = phi1 * rho2 + phi2 * rho1;

        // Autocovariance (with r(0) = 1.0 for simplicity — the Yule-Walker
        // system for normalized ACF recovers the same coefficients)
        let acf = vec![1.0, rho1, rho2, rho3];

        let result = levinson_durbin(&acf).unwrap();

        assert!(
            (result.ar_coeffs[0] - phi1).abs() < 1e-10,
            "φ₁ should be {phi1}, got {}",
            result.ar_coeffs[0]
        );
        assert!(
            (result.ar_coeffs[1] - phi2).abs() < 1e-10,
            "φ₂ should be {phi2}, got {}",
            result.ar_coeffs[1]
        );
    }

    /// Verify Levinson-Durbin matches manual Yule-Walker 2x2 solution
    #[test]
    fn test_matches_yule_walker_2x2() {
        let r0 = 1.0;
        let r1 = 0.6;
        let r2 = 0.2;

        // Manual Yule-Walker for p=2:
        // [[r0, r1], [r1, r0]] · [φ₁, φ₂] = [r1, r2]
        // [[1, 0.6], [0.6, 1]] · [φ₁, φ₂] = [0.6, 0.2]
        let det = r0 * r0 - r1 * r1; // 1 - 0.36 = 0.64
        let yw_phi1 = (r0 * r1 - r1 * r2) / det; // (0.6 - 0.12) / 0.64
        let yw_phi2 = (r0 * r2 - r1 * r1) / det; // (0.2 - 0.36) / 0.64

        let acf = vec![r0, r1, r2];
        let result = levinson_durbin(&acf).unwrap();

        assert!(
            (result.ar_coeffs[0] - yw_phi1).abs() < 1e-10,
            "LD φ₁={} should match YW φ₁={yw_phi1}",
            result.ar_coeffs[0]
        );
        assert!(
            (result.ar_coeffs[1] - yw_phi2).abs() < 1e-10,
            "LD φ₂={} should match YW φ₂={yw_phi2}",
            result.ar_coeffs[1]
        );
    }

    /// PACF of AR(1): only lag 1 is nonzero
    #[test]
    fn test_pacf_ar1() {
        let rho: f64 = 0.9;
        let var = 1.0 / (1.0 - rho * rho);
        let acf: Vec<f64> = (0..6).map(|k| var * rho.powi(k)).collect();

        let p = pacf(&acf);

        assert!((p[0] - rho).abs() < 1e-10, "PACF(1) should be {rho}");
        for i in 1..p.len() {
            assert!(
                p[i].abs() < 1e-10,
                "PACF({}) should be ~0, got {}",
                i + 1,
                p[i]
            );
        }
    }

    /// PACF of AR(2): lags 1 and 2 are nonzero, rest ~0
    #[test]
    fn test_pacf_ar2() {
        let phi1 = 0.6;
        let phi2 = 0.2;

        // Theoretical ACF via recursion: ρ(k) = φ₁·ρ(k-1) + φ₂·ρ(k-2)
        let rho1 = phi1 / (1.0 - phi2);
        let mut rho = vec![1.0, rho1];
        for k in 2..8 {
            rho.push(phi1 * rho[k - 1] + phi2 * rho[k - 2]);
        }

        let p = pacf(&rho);

        // First two should be nonzero
        assert!(p[0].abs() > 0.01, "PACF(1) should be nonzero");
        assert!(p[1].abs() > 0.01, "PACF(2) should be nonzero");

        // Rest should be ~0
        for i in 2..p.len() {
            assert!(
                p[i].abs() < 1e-10,
                "PACF({}) should be ~0 for AR(2), got {}",
                i + 1,
                p[i]
            );
        }
    }

    /// Order selection should pick p=2 for an AR(2) process
    #[test]
    fn test_order_selection() {
        // Generate AR(2) data: x_t = 0.5·x_{t-1} - 0.3·x_{t-2} + noise
        let n = 2000;
        let phi1 = 0.5;
        let phi2 = -0.3;
        let mut series = vec![0.0; n];

        // Deterministic pseudo-random noise (same as mcmc_diagnostics)
        let mut z = 0.5;
        for t in 2..n {
            z = (z * std::f64::consts::PI + std::f64::consts::E).fract();
            let noise = (z - 0.5) * 0.5; // small noise
            series[t] = phi1 * series[t - 1] + phi2 * series[t - 2] + noise;
        }

        let selected = ar_order_select(&series, 6);
        assert!(
            selected == 2,
            "Order selection should pick p=2, got p={selected}"
        );
    }

    /// Constant series returns None
    #[test]
    fn test_constant_series() {
        let acf = vec![0.0, 0.0, 0.0];
        assert!(levinson_durbin(&acf).is_none());
    }

    /// Single lag returns None (need at least 2 elements)
    #[test]
    fn test_too_short() {
        let acf = vec![1.0];
        assert!(levinson_durbin(&acf).is_none());
    }

    /// AR(1) fit from series data
    #[test]
    fn test_fit_ar_from_series() {
        let n = 1000;
        let rho = 0.7;
        let mut series = vec![0.0; n];

        let mut z = 0.42;
        for t in 1..n {
            z = (z * std::f64::consts::PI + std::f64::consts::E).fract();
            let noise = (z - 0.5) * 0.3;
            series[t] = rho * series[t - 1] + noise;
        }

        let fit = fit_ar_levinson(&series, 1).unwrap();

        assert!(
            (fit.coeffs[0] - rho).abs() < 0.1,
            "Estimated φ₁={} should be near {rho}",
            fit.coeffs[0]
        );
        assert!(fit.innovation_variance > 0.0);
        assert!(fit.aic.is_finite());
        assert!(fit.bic.is_finite());
    }

    /// Higher-order AR(4) recovery
    #[test]
    fn test_ar4_recovery() {
        // AR(4): φ = [0.4, -0.2, 0.1, -0.05]
        // Compute theoretical autocovariance via recursion
        let phi = [0.4, -0.2, 0.1, -0.05];

        // Use Yule-Walker equations to get theoretical ACF
        // ρ(k) = Σ φ_j · ρ(k-j) for k ≥ 1
        // For k=0: ρ(0) = 1
        // Start with approximate recursion for lags 0..8
        let mut rho = vec![1.0; 9];

        // Iterate the Yule-Walker system to convergence
        for _ in 0..200 {
            let mut new_rho = vec![1.0; 9];
            for k in 1..9 {
                let mut val = 0.0;
                for (j, &p) in phi.iter().enumerate() {
                    let lag = if k > j + 1 {
                        k - j - 1
                    } else {
                        j + 1 - k
                    };
                    val += p * rho[lag.min(8)];
                }
                new_rho[k] = val;
            }
            rho = new_rho;
        }

        let result = levinson_durbin(&rho[..5]).unwrap();

        for (i, &expected) in phi.iter().enumerate() {
            assert!(
                (result.ar_coeffs[i] - expected).abs() < 1e-4,
                "φ_{} should be {expected}, got {}",
                i + 1,
                result.ar_coeffs[i]
            );
        }
    }

    /// Prediction error decreases with model order for AR(2) process
    #[test]
    fn test_prediction_error_monotone() {
        let rho: f64 = 0.8;
        let var = 1.0 / (1.0 - rho * rho);
        let acf: Vec<f64> = (0..6).map(|k| var * rho.powi(k)).collect();

        let result = levinson_durbin(&acf).unwrap();

        // Each prediction error should be ≤ the previous one
        for i in 1..result.prediction_errors.len() {
            assert!(
                result.prediction_errors[i] <= result.prediction_errors[i - 1] + 1e-15,
                "σ²_{} = {} should be ≤ σ²_{} = {}",
                i + 1,
                result.prediction_errors[i],
                i,
                result.prediction_errors[i - 1]
            );
        }
    }

    /// Reflection coefficients bounded in [-1, 1] for stationary process
    #[test]
    fn test_reflection_bounded() {
        let phi1 = 0.3;
        let phi2 = 0.2;
        let rho1 = phi1 / (1.0 - phi2);
        let mut rho = vec![1.0, rho1];
        for k in 2..8 {
            rho.push(phi1 * rho[k - 1] + phi2 * rho[k - 2]);
        }

        let result = levinson_durbin(&rho).unwrap();

        for (i, &k) in result.reflection_coeffs.iter().enumerate() {
            assert!(
                k.abs() <= 1.0 + 1e-10,
                "Reflection coeff k_{} = {} should be in [-1,1]",
                i + 1,
                k
            );
        }
    }
}
