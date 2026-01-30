//! Runtime intrinsics for Sounio
//!
//! These functions are called by compiled Sounio code via FFI.

use std::ffi::CStr;
use std::os::raw::c_char;

// ============================================================================
// Print intrinsics
// ============================================================================

/// Print a string to stdout
///
/// # Safety
/// `s` must be a valid null-terminated C string pointer
#[no_mangle]
pub unsafe extern "C" fn sounio_print_str(s: *const c_char) {
    if s.is_null() {
        return;
    }

    let cstr = CStr::from_ptr(s);
    if let Ok(s) = cstr.to_str() {
        print!("{}", s);
    }
}

/// Print an integer
#[no_mangle]
pub extern "C" fn sounio_print_i64(value: i64) {
    print!("{}", value);
}

/// Print a float
#[no_mangle]
pub extern "C" fn sounio_print_f64(value: f64) {
    print!("{}", value);
}

/// Print a boolean
#[no_mangle]
pub extern "C" fn sounio_print_bool(value: bool) {
    print!("{}", value);
}

/// Print newline
#[no_mangle]
pub extern "C" fn sounio_println() {
    println!();
}

// ============================================================================
// Memory intrinsics
// ============================================================================

/// Allocate memory
#[no_mangle]
pub extern "C" fn sounio_alloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8;
    }

    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        std::alloc::alloc(layout)
    }
}

/// Allocate zeroed memory
#[no_mangle]
pub extern "C" fn sounio_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8;
    }

    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        std::alloc::alloc_zeroed(layout)
    }
}

/// Deallocate memory
///
/// # Safety
/// `ptr` must be a valid pointer previously allocated with the same size and alignment
#[no_mangle]
pub unsafe extern "C" fn sounio_dealloc(ptr: *mut u8, size: usize, align: usize) {
    if size == 0 || ptr.is_null() {
        return;
    }

    let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
    std::alloc::dealloc(ptr, layout);
}

/// Reallocate memory
///
/// # Safety
/// `ptr` must be a valid pointer previously allocated with the old size and alignment
#[no_mangle]
pub unsafe extern "C" fn sounio_realloc(
    ptr: *mut u8,
    old_size: usize,
    align: usize,
    new_size: usize,
) -> *mut u8 {
    if ptr.is_null() || old_size == 0 {
        return sounio_alloc(new_size, align);
    }

    if new_size == 0 {
        sounio_dealloc(ptr, old_size, align);
        return std::ptr::null_mut();
    }

    let layout = std::alloc::Layout::from_size_align_unchecked(old_size, align);
    std::alloc::realloc(ptr, layout, new_size)
}

// ============================================================================
// Panic / Assert intrinsics
// ============================================================================

/// Panic with message
///
/// # Safety
/// `msg` must be a valid null-terminated C string pointer if not null
#[no_mangle]
pub unsafe extern "C" fn sounio_panic(msg: *const c_char) -> ! {
    let message = if msg.is_null() {
        "panic".to_string()
    } else {
        CStr::from_ptr(msg).to_str().unwrap_or("panic").to_string()
    };

    panic!("{}", message);
}

/// Assert with message
///
/// # Safety
/// `msg` must be a valid null-terminated C string pointer if not null
#[no_mangle]
pub unsafe extern "C" fn sounio_assert(condition: bool, msg: *const c_char) {
    if !condition {
        sounio_panic(msg);
    }
}

// ============================================================================
// Math intrinsics
// ============================================================================

#[no_mangle]
pub extern "C" fn sounio_sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}

#[no_mangle]
pub extern "C" fn sounio_sin_f64(x: f64) -> f64 {
    x.sin()
}

#[no_mangle]
pub extern "C" fn sounio_cos_f64(x: f64) -> f64 {
    x.cos()
}

#[no_mangle]
pub extern "C" fn sounio_tan_f64(x: f64) -> f64 {
    x.tan()
}

#[no_mangle]
pub extern "C" fn sounio_asin_f64(x: f64) -> f64 {
    x.asin()
}

#[no_mangle]
pub extern "C" fn sounio_acos_f64(x: f64) -> f64 {
    x.acos()
}

#[no_mangle]
pub extern "C" fn sounio_atan_f64(x: f64) -> f64 {
    x.atan()
}

#[no_mangle]
pub extern "C" fn sounio_atan2_f64(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

#[no_mangle]
pub extern "C" fn sounio_exp_f64(x: f64) -> f64 {
    x.exp()
}

#[no_mangle]
pub extern "C" fn sounio_ln_f64(x: f64) -> f64 {
    x.ln()
}

#[no_mangle]
pub extern "C" fn sounio_log10_f64(x: f64) -> f64 {
    x.log10()
}

#[no_mangle]
pub extern "C" fn sounio_log2_f64(x: f64) -> f64 {
    x.log2()
}

#[no_mangle]
pub extern "C" fn sounio_pow_f64(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

#[no_mangle]
pub extern "C" fn sounio_abs_f64(x: f64) -> f64 {
    x.abs()
}

#[no_mangle]
pub extern "C" fn sounio_floor_f64(x: f64) -> f64 {
    x.floor()
}

#[no_mangle]
pub extern "C" fn sounio_ceil_f64(x: f64) -> f64 {
    x.ceil()
}

#[no_mangle]
pub extern "C" fn sounio_round_f64(x: f64) -> f64 {
    x.round()
}

#[no_mangle]
pub extern "C" fn sounio_trunc_f64(x: f64) -> f64 {
    x.trunc()
}

#[no_mangle]
pub extern "C" fn sounio_min_f64(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[no_mangle]
pub extern "C" fn sounio_max_f64(a: f64, b: f64) -> f64 {
    a.max(b)
}

#[no_mangle]
pub extern "C" fn sounio_clamp_f64(x: f64, min: f64, max: f64) -> f64 {
    x.clamp(min, max)
}

// ============================================================================
// Knowledge/Uncertainty intrinsics
// ============================================================================

/// Knowledge value with uncertainty
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Knowledge {
    pub value: f64,
    pub uncertainty: f64,
    pub confidence: f64,
}

#[no_mangle]
pub extern "C" fn sounio_knowledge_new(value: f64, uncertainty: f64, confidence: f64) -> Knowledge {
    Knowledge {
        value,
        uncertainty: uncertainty.abs(),
        confidence: confidence.clamp(0.0, 1.0),
    }
}

#[no_mangle]
pub extern "C" fn sounio_knowledge_exact(value: f64) -> Knowledge {
    Knowledge {
        value,
        uncertainty: 0.0,
        confidence: 1.0,
    }
}

/// Add two Knowledge values (uncertainty propagation)
#[no_mangle]
pub extern "C" fn sounio_knowledge_add(a: Knowledge, b: Knowledge) -> Knowledge {
    Knowledge {
        value: a.value + b.value,
        // Quadrature sum for independent uncertainties
        uncertainty: (a.uncertainty.powi(2) + b.uncertainty.powi(2)).sqrt(),
        // Minimum confidence
        confidence: a.confidence.min(b.confidence),
    }
}

/// Subtract two Knowledge values
#[no_mangle]
pub extern "C" fn sounio_knowledge_sub(a: Knowledge, b: Knowledge) -> Knowledge {
    Knowledge {
        value: a.value - b.value,
        uncertainty: (a.uncertainty.powi(2) + b.uncertainty.powi(2)).sqrt(),
        confidence: a.confidence.min(b.confidence),
    }
}

/// Multiply two Knowledge values
#[no_mangle]
pub extern "C" fn sounio_knowledge_mul(a: Knowledge, b: Knowledge) -> Knowledge {
    let value = a.value * b.value;
    // Relative uncertainty propagation
    let rel_a = if a.value != 0.0 {
        a.uncertainty / a.value.abs()
    } else {
        0.0
    };
    let rel_b = if b.value != 0.0 {
        b.uncertainty / b.value.abs()
    } else {
        0.0
    };
    let rel_result = (rel_a.powi(2) + rel_b.powi(2)).sqrt();

    Knowledge {
        value,
        uncertainty: value.abs() * rel_result,
        confidence: a.confidence.min(b.confidence),
    }
}

/// Divide two Knowledge values
#[no_mangle]
pub extern "C" fn sounio_knowledge_div(a: Knowledge, b: Knowledge) -> Knowledge {
    if b.value == 0.0 {
        return Knowledge {
            value: f64::NAN,
            uncertainty: f64::INFINITY,
            confidence: 0.0,
        };
    }

    let value = a.value / b.value;
    let rel_a = if a.value != 0.0 {
        a.uncertainty / a.value.abs()
    } else {
        0.0
    };
    let rel_b = b.uncertainty / b.value.abs();
    let rel_result = (rel_a.powi(2) + rel_b.powi(2)).sqrt();

    Knowledge {
        value,
        uncertainty: value.abs() * rel_result,
        confidence: a.confidence.min(b.confidence),
    }
}

/// Power function for Knowledge
#[no_mangle]
pub extern "C" fn sounio_knowledge_pow(k: Knowledge, n: f64) -> Knowledge {
    let value = k.value.powf(n);
    let rel_uncertainty = n.abs() * (k.uncertainty / k.value.abs());

    Knowledge {
        value,
        uncertainty: value.abs() * rel_uncertainty,
        confidence: k.confidence,
    }
}

/// Print Knowledge value
#[no_mangle]
pub extern "C" fn sounio_knowledge_print(k: Knowledge) {
    let ci_width = k.uncertainty * 1.96; // 95% CI for Gaussian
    print!(
        "{:.4} +/- {:.4} ({}% CI)",
        k.value,
        ci_width,
        (k.confidence * 100.0) as i32
    );
}

/// Get 95% confidence interval lower bound
#[no_mangle]
pub extern "C" fn sounio_knowledge_ci_lower(k: Knowledge) -> f64 {
    k.value - k.uncertainty * 1.96
}

/// Get 95% confidence interval upper bound
#[no_mangle]
pub extern "C" fn sounio_knowledge_ci_upper(k: Knowledge) -> f64 {
    k.value + k.uncertainty * 1.96
}

// ============================================================================
// Bootstrap Uncertainty intrinsics
// ============================================================================

/// Bootstrap resampling-based uncertainty
///
/// Represents uncertainty via percentile confidence intervals from
/// bootstrap samples. Samples are stored externally; this struct holds
/// summary statistics for FFI efficiency.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BootstrapKnowledge {
    /// Original point estimate
    pub estimate: f64,
    /// Lower percentile bound (e.g., 2.5th percentile for 95% CI)
    pub percentile_lower: f64,
    /// Upper percentile bound (e.g., 97.5th percentile for 95% CI)
    pub percentile_upper: f64,
    /// Bootstrap standard error (std dev of bootstrap samples)
    pub bootstrap_se: f64,
    /// Bias estimate: mean(samples) - estimate
    pub bias: f64,
    /// Number of bootstrap samples used
    pub n_samples: u32,
}

/// Helper: compute percentile from sorted slice
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Helper: compute mean of slice
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Helper: compute std dev of slice
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

/// Create BootstrapKnowledge from bootstrap samples
///
/// # Arguments
/// * `estimate` - Original point estimate
/// * `samples_ptr` - Pointer to bootstrap sample array
/// * `n_samples` - Number of samples
/// * `alpha` - Significance level (e.g., 0.05 for 95% CI)
///
/// # Safety
/// `samples_ptr` must be a valid pointer to an array of n_samples f64 values
#[no_mangle]
pub unsafe extern "C" fn sounio_bootstrap_from_samples(
    estimate: f64,
    samples_ptr: *const f64,
    n_samples: usize,
    alpha: f64,
) -> BootstrapKnowledge {
    if samples_ptr.is_null() || n_samples == 0 {
        return BootstrapKnowledge {
            estimate,
            percentile_lower: estimate,
            percentile_upper: estimate,
            bootstrap_se: 0.0,
            bias: 0.0,
            n_samples: 0,
        };
    }

    let samples = std::slice::from_raw_parts(samples_ptr, n_samples);
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sample_mean = mean(samples);
    let lower_p = alpha / 2.0;
    let upper_p = 1.0 - alpha / 2.0;

    BootstrapKnowledge {
        estimate,
        percentile_lower: percentile(&sorted, lower_p),
        percentile_upper: percentile(&sorted, upper_p),
        bootstrap_se: std_dev(samples),
        bias: sample_mean - estimate,
        n_samples: n_samples as u32,
    }
}

/// Create BootstrapKnowledge directly from summary statistics
#[no_mangle]
pub extern "C" fn sounio_bootstrap_new(
    estimate: f64,
    percentile_lower: f64,
    percentile_upper: f64,
    bootstrap_se: f64,
    bias: f64,
    n_samples: u32,
) -> BootstrapKnowledge {
    BootstrapKnowledge {
        estimate,
        percentile_lower,
        percentile_upper,
        bootstrap_se,
        bias,
        n_samples,
    }
}

/// Add two Bootstrap values (sample-wise propagation using summary stats)
///
/// Note: For full accuracy, operate on raw samples. This uses normal approximation.
#[no_mangle]
pub extern "C" fn sounio_bootstrap_add(
    a: BootstrapKnowledge,
    b: BootstrapKnowledge,
) -> BootstrapKnowledge {
    let estimate = a.estimate + b.estimate;
    // Quadrature sum for SE (assuming independence)
    let bootstrap_se = (a.bootstrap_se.powi(2) + b.bootstrap_se.powi(2)).sqrt();
    // CI widths add in quadrature for independent bootstrap CIs
    let a_half_width = (a.percentile_upper - a.percentile_lower) / 2.0;
    let b_half_width = (b.percentile_upper - b.percentile_lower) / 2.0;
    let combined_half_width = (a_half_width.powi(2) + b_half_width.powi(2)).sqrt();
    let center = (a.estimate + a.bias) + (b.estimate + b.bias);

    BootstrapKnowledge {
        estimate,
        percentile_lower: center - combined_half_width,
        percentile_upper: center + combined_half_width,
        bootstrap_se,
        bias: a.bias + b.bias,
        n_samples: a.n_samples.min(b.n_samples),
    }
}

/// Subtract two Bootstrap values
#[no_mangle]
pub extern "C" fn sounio_bootstrap_sub(
    a: BootstrapKnowledge,
    b: BootstrapKnowledge,
) -> BootstrapKnowledge {
    let estimate = a.estimate - b.estimate;
    let bootstrap_se = (a.bootstrap_se.powi(2) + b.bootstrap_se.powi(2)).sqrt();
    let a_half_width = (a.percentile_upper - a.percentile_lower) / 2.0;
    let b_half_width = (b.percentile_upper - b.percentile_lower) / 2.0;
    let combined_half_width = (a_half_width.powi(2) + b_half_width.powi(2)).sqrt();
    let center = (a.estimate + a.bias) - (b.estimate + b.bias);

    BootstrapKnowledge {
        estimate,
        percentile_lower: center - combined_half_width,
        percentile_upper: center + combined_half_width,
        bootstrap_se,
        bias: a.bias - b.bias,
        n_samples: a.n_samples.min(b.n_samples),
    }
}

/// Multiply two Bootstrap values (using relative uncertainty approximation)
#[no_mangle]
pub extern "C" fn sounio_bootstrap_mul(
    a: BootstrapKnowledge,
    b: BootstrapKnowledge,
) -> BootstrapKnowledge {
    let estimate = a.estimate * b.estimate;
    // Relative SE propagation
    let rel_a = if a.estimate != 0.0 {
        a.bootstrap_se / a.estimate.abs()
    } else {
        0.0
    };
    let rel_b = if b.estimate != 0.0 {
        b.bootstrap_se / b.estimate.abs()
    } else {
        0.0
    };
    let rel_combined = (rel_a.powi(2) + rel_b.powi(2)).sqrt();
    let bootstrap_se = estimate.abs() * rel_combined;

    // Relative CI widths
    let a_rel_half = if a.estimate != 0.0 {
        (a.percentile_upper - a.percentile_lower) / (2.0 * a.estimate.abs())
    } else {
        0.0
    };
    let b_rel_half = if b.estimate != 0.0 {
        (b.percentile_upper - b.percentile_lower) / (2.0 * b.estimate.abs())
    } else {
        0.0
    };
    let combined_rel_half = (a_rel_half.powi(2) + b_rel_half.powi(2)).sqrt();
    let half_width = estimate.abs() * combined_rel_half;

    BootstrapKnowledge {
        estimate,
        percentile_lower: estimate - half_width,
        percentile_upper: estimate + half_width,
        bootstrap_se,
        bias: 0.0, // Bias doesn't propagate simply through multiplication
        n_samples: a.n_samples.min(b.n_samples),
    }
}

/// Divide two Bootstrap values
#[no_mangle]
pub extern "C" fn sounio_bootstrap_div(
    a: BootstrapKnowledge,
    b: BootstrapKnowledge,
) -> BootstrapKnowledge {
    if b.estimate == 0.0 {
        return BootstrapKnowledge {
            estimate: f64::NAN,
            percentile_lower: f64::NEG_INFINITY,
            percentile_upper: f64::INFINITY,
            bootstrap_se: f64::INFINITY,
            bias: 0.0,
            n_samples: 0,
        };
    }

    let estimate = a.estimate / b.estimate;
    let rel_a = if a.estimate != 0.0 {
        a.bootstrap_se / a.estimate.abs()
    } else {
        0.0
    };
    let rel_b = b.bootstrap_se / b.estimate.abs();
    let rel_combined = (rel_a.powi(2) + rel_b.powi(2)).sqrt();
    let bootstrap_se = estimate.abs() * rel_combined;

    let a_rel_half = if a.estimate != 0.0 {
        (a.percentile_upper - a.percentile_lower) / (2.0 * a.estimate.abs())
    } else {
        0.0
    };
    let b_rel_half = (b.percentile_upper - b.percentile_lower) / (2.0 * b.estimate.abs());
    let combined_rel_half = (a_rel_half.powi(2) + b_rel_half.powi(2)).sqrt();
    let half_width = estimate.abs() * combined_rel_half;

    BootstrapKnowledge {
        estimate,
        percentile_lower: estimate - half_width,
        percentile_upper: estimate + half_width,
        bootstrap_se,
        bias: 0.0,
        n_samples: a.n_samples.min(b.n_samples),
    }
}

/// Convert BootstrapKnowledge to simple Knowledge
#[no_mangle]
pub extern "C" fn sounio_bootstrap_to_knowledge(boot: BootstrapKnowledge) -> Knowledge {
    // Use bootstrap SE as uncertainty, derive confidence from CI width
    let ci_width = boot.percentile_upper - boot.percentile_lower;
    // Assume 95% CI -> confidence 0.95, scale if CI is narrower/wider
    let expected_95_width = boot.bootstrap_se * 2.0 * 1.96;
    let confidence = if expected_95_width > 0.0 {
        (0.95 * ci_width / expected_95_width).clamp(0.0, 1.0)
    } else {
        0.95
    };

    Knowledge {
        value: boot.estimate,
        uncertainty: boot.bootstrap_se,
        confidence,
    }
}

/// Print BootstrapKnowledge value
#[no_mangle]
pub extern "C" fn sounio_bootstrap_print(boot: BootstrapKnowledge) {
    print!(
        "{:.4} [{:.4}, {:.4}] (SE={:.4}, bias={:.4}, n={})",
        boot.estimate,
        boot.percentile_lower,
        boot.percentile_upper,
        boot.bootstrap_se,
        boot.bias,
        boot.n_samples
    );
}

// ============================================================================
// Cross-Validation Uncertainty intrinsics
// ============================================================================

/// K-fold cross-validation uncertainty
///
/// Represents model performance uncertainty from K-fold CV.
/// Stores summary statistics; fold values stored externally.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CVKnowledge {
    /// Mean across folds
    pub mean: f64,
    /// Standard error of the mean (SE = std / sqrt(k))
    pub std_error: f64,
    /// Standard deviation across folds
    pub std_dev: f64,
    /// Number of folds
    pub k: u32,
}

/// Create CVKnowledge from fold values
///
/// # Safety
/// `folds_ptr` must be a valid pointer to an array of k f64 values
#[no_mangle]
pub unsafe extern "C" fn sounio_cv_from_folds(folds_ptr: *const f64, k: usize) -> CVKnowledge {
    if folds_ptr.is_null() || k == 0 {
        return CVKnowledge {
            mean: 0.0,
            std_error: 0.0,
            std_dev: 0.0,
            k: 0,
        };
    }

    let folds = std::slice::from_raw_parts(folds_ptr, k);
    let fold_mean = mean(folds);
    let fold_std = std_dev(folds);
    let std_error = fold_std / (k as f64).sqrt();

    CVKnowledge {
        mean: fold_mean,
        std_error,
        std_dev: fold_std,
        k: k as u32,
    }
}

/// Create CVKnowledge directly from summary statistics
#[no_mangle]
pub extern "C" fn sounio_cv_new(mean: f64, std_error: f64, std_dev: f64, k: u32) -> CVKnowledge {
    CVKnowledge {
        mean,
        std_error,
        std_dev,
        k,
    }
}

/// Combine two CV results (pooled statistics)
///
/// Useful when combining CV results from different model evaluations.
#[no_mangle]
pub extern "C" fn sounio_cv_combine(a: CVKnowledge, b: CVKnowledge) -> CVKnowledge {
    // Pooled mean (weighted by k)
    let total_k = a.k + b.k;
    let pooled_mean = if total_k > 0 {
        (a.mean * a.k as f64 + b.mean * b.k as f64) / total_k as f64
    } else {
        0.0
    };

    // Pooled variance
    let a_var = a.std_dev.powi(2);
    let b_var = b.std_dev.powi(2);
    let pooled_var = if total_k > 0 {
        ((a.k as f64 - 1.0) * a_var + (b.k as f64 - 1.0) * b_var) / (total_k as f64 - 2.0).max(1.0)
    } else {
        0.0
    };
    let pooled_std = pooled_var.sqrt();
    let pooled_se = pooled_std / (total_k as f64).sqrt();

    CVKnowledge {
        mean: pooled_mean,
        std_error: pooled_se,
        std_dev: pooled_std,
        k: total_k,
    }
}

/// Convert CVKnowledge to simple Knowledge
#[no_mangle]
pub extern "C" fn sounio_cv_to_knowledge(cv: CVKnowledge) -> Knowledge {
    // Use std_error as uncertainty, confidence based on k
    // More folds -> higher confidence
    let confidence = if cv.k >= 10 {
        0.95
    } else if cv.k >= 5 {
        0.90
    } else {
        0.80
    };

    Knowledge {
        value: cv.mean,
        uncertainty: cv.std_error,
        confidence,
    }
}

/// Get 95% confidence interval for CV (using t-distribution approximation)
#[no_mangle]
pub extern "C" fn sounio_cv_ci_lower(cv: CVKnowledge) -> f64 {
    // t-value for 95% CI with k-1 degrees of freedom (approximate)
    let t_value = if cv.k <= 2 {
        12.71
    } else if cv.k <= 5 {
        2.78
    } else if cv.k <= 10 {
        2.26
    } else {
        1.96
    };
    cv.mean - t_value * cv.std_error
}

#[no_mangle]
pub extern "C" fn sounio_cv_ci_upper(cv: CVKnowledge) -> f64 {
    let t_value = if cv.k <= 2 {
        12.71
    } else if cv.k <= 5 {
        2.78
    } else if cv.k <= 10 {
        2.26
    } else {
        1.96
    };
    cv.mean + t_value * cv.std_error
}

/// Print CVKnowledge value
#[no_mangle]
pub extern "C" fn sounio_cv_print(cv: CVKnowledge) {
    print!(
        "{:.4} +/- {:.4} (k={}, std={:.4})",
        cv.mean, cv.std_error, cv.k, cv.std_dev
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_add() {
        let a = sounio_knowledge_new(10.0, 0.5, 0.95);
        let b = sounio_knowledge_new(20.0, 0.3, 0.95);
        let result = sounio_knowledge_add(a, b);

        assert!((result.value - 30.0).abs() < 1e-10);
        assert!((result.uncertainty - (0.5_f64.powi(2) + 0.3_f64.powi(2)).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_knowledge_mul() {
        let a = sounio_knowledge_new(10.0, 1.0, 0.95); // 10% relative uncertainty
        let b = sounio_knowledge_new(5.0, 0.5, 0.95); // 10% relative uncertainty
        let result = sounio_knowledge_mul(a, b);

        assert!((result.value - 50.0).abs() < 1e-10);
        // Relative uncertainty should be ~14.14% (sqrt(0.1^2 + 0.1^2))
    }

    #[test]
    fn test_math_intrinsics() {
        assert!((sounio_sqrt_f64(4.0) - 2.0).abs() < 1e-10);
        assert!((sounio_sin_f64(0.0)).abs() < 1e-10);
        assert!((sounio_cos_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((sounio_exp_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((sounio_ln_f64(1.0)).abs() < 1e-10);
    }

    // ========================================================================
    // Bootstrap tests
    // ========================================================================

    #[test]
    fn test_bootstrap_from_samples() {
        let samples = [10.1, 10.3, 9.8, 10.0, 10.2, 9.9, 10.4, 10.1, 9.7, 10.5];
        let boot =
            unsafe { sounio_bootstrap_from_samples(10.0, samples.as_ptr(), samples.len(), 0.05) };

        assert_eq!(boot.n_samples, 10);
        assert!((boot.estimate - 10.0).abs() < 1e-10);
        // Percentiles should bracket the samples
        assert!(boot.percentile_lower < boot.estimate);
        assert!(boot.percentile_upper > boot.estimate);
        // SE should be positive
        assert!(boot.bootstrap_se > 0.0);
    }

    #[test]
    fn test_bootstrap_add() {
        let a = sounio_bootstrap_new(10.0, 9.5, 10.5, 0.25, 0.0, 100);
        let b = sounio_bootstrap_new(20.0, 19.0, 21.0, 0.5, 0.0, 100);
        let result = sounio_bootstrap_add(a, b);

        assert!((result.estimate - 30.0).abs() < 1e-10);
        // SE should combine in quadrature
        let expected_se = (0.25_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
        assert!((result.bootstrap_se - expected_se).abs() < 1e-10);
    }

    #[test]
    fn test_bootstrap_mul() {
        let a = sounio_bootstrap_new(10.0, 9.0, 11.0, 0.5, 0.0, 100);
        let b = sounio_bootstrap_new(5.0, 4.5, 5.5, 0.25, 0.0, 100);
        let result = sounio_bootstrap_mul(a, b);

        assert!((result.estimate - 50.0).abs() < 1e-10);
        // Result should have wider CI than either input
        assert!(result.percentile_upper > result.estimate);
        assert!(result.percentile_lower < result.estimate);
    }

    #[test]
    fn test_bootstrap_to_knowledge() {
        let boot = sounio_bootstrap_new(10.0, 9.5, 10.5, 0.25, 0.0, 100);
        let k = sounio_bootstrap_to_knowledge(boot);

        assert!((k.value - 10.0).abs() < 1e-10);
        assert!((k.uncertainty - 0.25).abs() < 1e-10);
        assert!(k.confidence > 0.0 && k.confidence <= 1.0);
    }

    // ========================================================================
    // CrossValidation tests
    // ========================================================================

    #[test]
    fn test_cv_from_folds() {
        let folds = [0.85, 0.87, 0.83, 0.86, 0.84];
        let cv = unsafe { sounio_cv_from_folds(folds.as_ptr(), folds.len()) };

        assert_eq!(cv.k, 5);
        assert!((cv.mean - 0.85).abs() < 0.01);
        assert!(cv.std_error > 0.0);
        assert!(cv.std_dev > 0.0);
        // SE should be std_dev / sqrt(k)
        let expected_se = cv.std_dev / (5.0_f64).sqrt();
        assert!((cv.std_error - expected_se).abs() < 1e-10);
    }

    #[test]
    fn test_cv_combine() {
        let a = sounio_cv_new(0.85, 0.01, 0.02, 5);
        let b = sounio_cv_new(0.87, 0.015, 0.03, 5);
        let result = sounio_cv_combine(a, b);

        assert_eq!(result.k, 10);
        // Pooled mean should be between a.mean and b.mean
        assert!(result.mean >= 0.85 && result.mean <= 0.87);
    }

    #[test]
    fn test_cv_to_knowledge() {
        let cv = sounio_cv_new(0.85, 0.01, 0.02, 5);
        let k = sounio_cv_to_knowledge(cv);

        assert!((k.value - 0.85).abs() < 1e-10);
        assert!((k.uncertainty - 0.01).abs() < 1e-10);
        assert!(k.confidence == 0.90); // 5 folds -> 90% confidence
    }

    #[test]
    fn test_cv_confidence_intervals() {
        let cv = sounio_cv_new(0.85, 0.01, 0.02, 10);
        let lower = sounio_cv_ci_lower(cv);
        let upper = sounio_cv_ci_upper(cv);

        assert!(lower < cv.mean);
        assert!(upper > cv.mean);
        // CI should be symmetric around mean
        assert!((cv.mean - lower - (upper - cv.mean)).abs() < 1e-10);
    }
}
