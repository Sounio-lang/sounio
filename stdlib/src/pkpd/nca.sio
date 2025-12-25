//! Non-compartmental analysis for pharmacokinetics

use linalg::Vector
use units::{mg, L, h, mg_L, L_h, h_inv}

/// Non-compartmental analysis result
pub struct NCAResult {
    /// Area under the curve (0 to last)
    pub auc_last: f64: mg_L * h,
    
    /// Area under the curve (0 to infinity)
    pub auc_inf: f64: mg_L * h,
    
    /// Maximum concentration
    pub cmax: f64: mg_L,
    
    /// Time to maximum concentration
    pub tmax: f64: h,
    
    /// Terminal elimination rate constant
    pub lambda_z: f64: h_inv,
    
    /// Terminal half-life
    pub t_half: f64: h,
    
    /// Clearance
    pub cl: f64: L_h,
    
    /// Volume of distribution (steady state)
    pub vss: f64: L,
    
    /// Volume of distribution (terminal phase)
    pub vz: f64: L,
    
    /// Mean residence time
    pub mrt: f64: h,
    
    /// Fraction of AUC extrapolated
    pub auc_extrap_percent: f64,
    
    /// R-squared for terminal phase
    pub r_squared: f64,
}

/// Perform non-compartmental analysis
pub fn nca_analysis(
    time: &Vector<f64>,
    concentration: &Vector<f64>,
    dose: f64: mg,
    dose_time: f64: h,
    n_terminal_points: usize,
) -> NCAResult {
    let n = time.len();
    assert!(n == concentration.len(), "time and concentration vectors must have same length");
    assert!(n >= 3, "need at least 3 data points");
    
    // Find Cmax and Tmax
    let (cmax, tmax) = find_cmax_tmax(time, concentration);
    
    // Calculate AUC to last measurable concentration
    let auc_last = calculate_auc_trapezoid(time, concentration);
    
    // Calculate terminal elimination parameters
    let (lambda_z, r_squared) = calculate_lambda_z(time, concentration, n_terminal_points);
    let t_half = if lambda_z > 0.0 { 0.693 / lambda_z } else { f64::NAN };
    
    // Calculate AUC to infinity
    let c_last = concentration[n - 1];
    let auc_extrap = if lambda_z > 0.0 && c_last > 0.0 {
        c_last / lambda_z
    } else {
        0.0
    };
    let auc_inf = auc_last + auc_extrap;
    let auc_extrap_percent = if auc_inf > 0.0 {
        100.0 * auc_extrap / auc_inf
    } else {
        f64::NAN
    };
    
    // Calculate clearance
    let cl = if auc_inf > 0.0 { dose / auc_inf } else { f64::NAN };
    
    // Calculate volume of distribution (terminal phase)
    let vz = if lambda_z > 0.0 && cl.is_finite() {
        cl / lambda_z
    } else {
        f64::NAN
    };
    
    // Calculate AUMC (area under first moment curve)
    let aumc_last = calculate_aumc_trapezoid(time, concentration);
    let aumc_extrap = if lambda_z > 0.0 && c_last > 0.0 {
        let t_last = time[n - 1];
        c_last * t_last / lambda_z + c_last / (lambda_z * lambda_z)
    } else {
        0.0
    };
    let aumc_inf = aumc_last + aumc_extrap;
    
    // Calculate mean residence time
    let mrt = if auc_inf > 0.0 { aumc_inf / auc_inf } else { f64::NAN };
    
    // Calculate volume of distribution at steady state
    let vss = if cl.is_finite() && mrt.is_finite() {
        cl * mrt
    } else {
        f64::NAN
    };
    
    NCAResult {
        auc_last,
        auc_inf,
        cmax,
        tmax,
        lambda_z,
        t_half,
        cl,
        vss,
        vz,
        mrt,
        auc_extrap_percent,
        r_squared,
    }
}

/// Find maximum concentration and time to maximum
fn find_cmax_tmax(time: &Vector<f64>, concentration: &Vector<f64>) -> (f64, f64) {
    let mut cmax = 0.0;
    let mut tmax = 0.0;
    
    for i in 0..concentration.len() {
        if concentration[i] > cmax {
            cmax = concentration[i];
            tmax = time[i];
        }
    }
    
    (cmax, tmax)
}

/// Calculate AUC using trapezoidal rule
fn calculate_auc_trapezoid(time: &Vector<f64>, concentration: &Vector<f64>) -> f64 {
    let mut auc = 0.0;
    
    for i in 1..time.len() {
        let dt = time[i] - time[i-1];
        let avg_conc = 0.5 * (concentration[i-1] + concentration[i]);
        auc += dt * avg_conc;
    }
    
    auc
}

/// Calculate AUMC (area under first moment curve) using trapezoidal rule
fn calculate_aumc_trapezoid(time: &Vector<f64>, concentration: &Vector<f64>) -> f64 {
    let mut aumc = 0.0;
    
    for i in 1..time.len() {
        let dt = time[i] - time[i-1];
        let avg_tc = 0.5 * (time[i-1] * concentration[i-1] + time[i] * concentration[i]);
        aumc += dt * avg_tc;
    }
    
    aumc
}

/// Calculate terminal elimination rate constant using linear regression
fn calculate_lambda_z(
    time: &Vector<f64>,
    concentration: &Vector<f64>,
    n_points: usize,
) -> (f64, f64) {
    let n = time.len();
    if n < n_points {
        return (f64::NAN, f64::NAN);
    }
    
    // Use last n_points for regression
    let start_idx = n - n_points;
    
    let mut sum_t = 0.0;
    let mut sum_log_c = 0.0;
    let mut sum_t_log_c = 0.0;
    let mut sum_t2 = 0.0;
    let mut sum_log_c2 = 0.0;
    let mut count = 0;
    
    for i in start_idx..n {
        if concentration[i] > 0.0 {
            let t = time[i];
            let log_c = concentration[i].ln();
            
            sum_t += t;
            sum_log_c += log_c;
            sum_t_log_c += t * log_c;
            sum_t2 += t * t;
            sum_log_c2 += log_c * log_c;
            count += 1;
        }
    }
    
    if count < 2 {
        return (f64::NAN, f64::NAN);
    }
    
    let n_f = count as f64;
    
    // Linear regression: log(C) = intercept + slope * t
    let slope = (n_f * sum_t_log_c - sum_t * sum_log_c) / (n_f * sum_t2 - sum_t * sum_t);
    let intercept = (sum_log_c - slope * sum_t) / n_f;
    
    // R-squared
    let ss_tot = sum_log_c2 - sum_log_c * sum_log_c / n_f;
    let ss_res = sum_log_c2 - intercept * sum_log_c - slope * sum_t_log_c;
    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { f64::NAN };
    
    let lambda_z = -slope; // Elimination rate constant is negative slope
    
    (lambda_z, r_squared)
}

/// Calculate bioavailability and bioequivalence parameters
pub struct BioequivalenceResult {
    /// Ratio of test/reference AUC
    pub auc_ratio: f64,
    
    /// 90% confidence interval for AUC ratio
    pub auc_ci_lower: f64,
    pub auc_ci_upper: f64,
    
    /// Ratio of test/reference Cmax
    pub cmax_ratio: f64,
    
    /// 90% confidence interval for Cmax ratio
    pub cmax_ci_lower: f64,
    pub cmax_ci_upper: f64,
    
    /// Bioequivalent (both AUC and Cmax ratios in 80-125% range)
    pub bioequivalent: bool,
}

/// Perform bioequivalence analysis
pub fn bioequivalence_analysis(
    test_results: &[NCAResult],
    reference_results: &[NCAResult],
) -> BioequivalenceResult {
    assert!(test_results.len() == reference_results.len(), 
            "test and reference must have same number of subjects");
    
    let n = test_results.len();
    
    // Calculate log-transformed ratios
    let mut log_auc_ratios = Vec::new();
    let mut log_cmax_ratios = Vec::new();
    
    for i in 0..n {
        if test_results[i].auc_inf > 0.0 && reference_results[i].auc_inf > 0.0 {
            log_auc_ratios.push((test_results[i].auc_inf / reference_results[i].auc_inf).ln());
        }
        if test_results[i].cmax > 0.0 && reference_results[i].cmax > 0.0 {
            log_cmax_ratios.push((test_results[i].cmax / reference_results[i].cmax).ln());
        }
    }
    
    // Calculate means and confidence intervals
    let (auc_ratio, auc_ci_lower, auc_ci_upper) = calculate_ratio_ci(&log_auc_ratios);
    let (cmax_ratio, cmax_ci_lower, cmax_ci_upper) = calculate_ratio_ci(&log_cmax_ratios);
    
    // Check bioequivalence (80-125% rule)
    let bioequivalent = auc_ci_lower >= 0.8 && auc_ci_upper <= 1.25 &&
                       cmax_ci_lower >= 0.8 && cmax_ci_upper <= 1.25;
    
    BioequivalenceResult {
        auc_ratio,
        auc_ci_lower,
        auc_ci_upper,
        cmax_ratio,
        cmax_ci_lower,
        cmax_ci_upper,
        bioequivalent,
    }
}

/// Calculate geometric mean ratio and 90% confidence interval
fn calculate_ratio_ci(log_ratios: &[f64]) -> (f64, f64, f64) {
    if log_ratios.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    
    let n = log_ratios.len() as f64;
    let mean: f64 = log_ratios.iter().sum::<f64>() / n;
    let variance: f64 = log_ratios.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    let se = (variance / n).sqrt();
    
    // t-value for 90% CI (two-tailed, alpha = 0.1)
    let t_value = 1.645; // Approximate for large n, should use t-distribution
    
    let ratio = mean.exp();
    let ci_lower = (mean - t_value * se).exp();
    let ci_upper = (mean + t_value * se).exp();
    
    (ratio, ci_lower, ci_upper)
}
