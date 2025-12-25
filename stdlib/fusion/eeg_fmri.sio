// fusion::eeg_fmri — EEG-fMRI Multimodal Integration
//
// Combines EEG's temporal precision with fMRI's spatial resolution.
// Essential for computational psychiatry: network dynamics in disorders.
//
// Methods:
// - EEG-informed fMRI: Use EEG features as fMRI regressors
// - fMRI-informed EEG: Use fMRI activation as source priors
// - Representational Similarity Analysis (RSA)
//
// References:
// - Debener et al. (2006): "Trial-by-trial coupling of EEG and fMRI"
// - Cichy et al. (2016): "Similarity-based fusion of MEG and fMRI"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// HEMODYNAMIC RESPONSE FUNCTION (HRF)
// ============================================================================

/// HRF parameters (Canonical double-gamma)
struct HRFParams {
    peak_delay: f64,        // Time to peak (seconds), ~6
    undershoot_delay: f64,  // Undershoot time (seconds), ~16
    peak_undershoot_ratio: f64, // Amplitude ratio, ~6
}

fn hrf_params_canonical() -> HRFParams {
    HRFParams {
        peak_delay: 6.0,
        undershoot_delay: 16.0,
        peak_undershoot_ratio: 6.0,
    }
}

/// Gamma PDF approximation
fn gamma_pdf(t: f64, shape: f64, scale: f64) -> f64 {
    if t <= 0.0 {
        return 0.0
    }
    let x = t / scale
    // Simplified gamma: x^(shape-1) * exp(-x)
    pow(x, shape - 1.0) * exp(-x)
}

/// Compute HRF value at time t
fn hrf_value(t: f64, params: HRFParams) -> f64 {
    let peak = gamma_pdf(t, params.peak_delay, 1.0)
    let undershoot = gamma_pdf(t, params.undershoot_delay, 1.0)
    peak - undershoot / params.peak_undershoot_ratio
}

// ============================================================================
// REPRESENTATIONAL DISSIMILARITY MATRIX (RDM)
// ============================================================================

/// Compute dissimilarity (1 - correlation) between two patterns
fn pattern_dissim(p1: [f64; 10], p2: [f64; 10], n: i64) -> f64 {
    // Pearson correlation
    var sum_x = 0.0
    var sum_y = 0.0
    var sum_xx = 0.0
    var sum_yy = 0.0
    var sum_xy = 0.0

    var i: i64 = 0
    while i < n {
        sum_x = sum_x + p1[i as usize]
        sum_y = sum_y + p2[i as usize]
        sum_xx = sum_xx + p1[i as usize] * p1[i as usize]
        sum_yy = sum_yy + p2[i as usize] * p2[i as usize]
        sum_xy = sum_xy + p1[i as usize] * p2[i as usize]
        i = i + 1
    }

    let nf = n as f64
    let cov = sum_xy - sum_x * sum_y / nf
    let var_x = sum_xx - sum_x * sum_x / nf
    let var_y = sum_yy - sum_y * sum_y / nf

    let denom = sqrt(var_x * var_y)
    let r = if denom > 1e-10 { cov / denom } else { 0.0 }

    1.0 - r  // Dissimilarity
}

/// Compare two RDMs (correlation)
fn compare_rdms(rdm1: [[f64; 4]; 4], rdm2: [[f64; 4]; 4]) -> f64 {
    // Extract upper triangle and correlate
    var v1: [f64; 10] = [0.0; 10]
    var v2: [f64; 10] = [0.0; 10]
    var n: i64 = 0

    var i: i64 = 0
    while i < 4 {
        var j = i + 1
        while j < 4 {
            v1[n as usize] = rdm1[i as usize][j as usize]
            v2[n as usize] = rdm2[i as usize][j as usize]
            n = n + 1
            j = j + 1
        }
        i = i + 1
    }

    // Pearson correlation
    var sum_x = 0.0
    var sum_y = 0.0
    var sum_xx = 0.0
    var sum_yy = 0.0
    var sum_xy = 0.0

    i = 0
    while i < n {
        sum_x = sum_x + v1[i as usize]
        sum_y = sum_y + v2[i as usize]
        sum_xx = sum_xx + v1[i as usize] * v1[i as usize]
        sum_yy = sum_yy + v2[i as usize] * v2[i as usize]
        sum_xy = sum_xy + v1[i as usize] * v2[i as usize]
        i = i + 1
    }

    let nf = n as f64
    let cov = sum_xy - sum_x * sum_y / nf
    let var_x = sum_xx - sum_x * sum_x / nf
    let var_y = sum_yy - sum_y * sum_y / nf

    let denom = sqrt(var_x * var_y)
    if denom > 1e-10 { cov / denom } else { 0.0 }
}

// ============================================================================
// SOURCE PRIOR
// ============================================================================

/// fMRI-based source prior for EEG localization
struct SourcePrior {
    weights: [f64; 100],
    n_sources: i64,
    threshold: f64,
}

fn source_prior_new() -> SourcePrior {
    SourcePrior {
        weights: [0.0; 100],
        n_sources: 0,
        threshold: 0.0,
    }
}

/// Create source prior from t-map values (soft thresholding)
fn create_prior_soft(tmap: [f64; 100], n: i64, threshold: f64) -> SourcePrior {
    var prior = source_prior_new()
    prior.n_sources = n
    prior.threshold = threshold

    var sum = 0.0
    var i: i64 = 0
    while i < n {
        // Sigmoid transform
        prior.weights[i as usize] = 1.0 / (1.0 + exp(-(tmap[i as usize] - threshold)))
        sum = sum + prior.weights[i as usize]
        i = i + 1
    }

    // Normalize
    if sum > 0.0 {
        i = 0
        while i < n {
            prior.weights[i as usize] = prior.weights[i as usize] / sum
            i = i + 1
        }
    }

    prior
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_hrf() -> bool {
    let params = hrf_params_canonical()

    // HRF should peak around 5-6 seconds
    let h5 = hrf_value(5.0, params)
    let h6 = hrf_value(6.0, params)
    let h10 = hrf_value(10.0, params)

    // Peak should be near 5-6s
    h5 > h10 && h6 > h10
}

fn test_rdm_compare_identical() -> bool {
    var rdm: [[f64; 4]; 4] = [[0.0; 4]; 4]
    rdm[0][1] = 0.5
    rdm[1][0] = 0.5
    rdm[0][2] = 0.3
    rdm[2][0] = 0.3
    rdm[1][2] = 0.8
    rdm[2][1] = 0.8

    let sim = compare_rdms(rdm, rdm)
    abs_f64(sim - 1.0) < 0.01
}

fn test_source_prior() -> bool {
    var tmap: [f64; 100] = [0.0; 100]
    tmap[0] = 5.0  // High activation
    tmap[1] = 0.0  // Low activation

    let prior = create_prior_soft(tmap, 2, 2.0)

    // First source should have higher weight
    prior.weights[0] > prior.weights[1]
}

fn main() -> i32 {
    print("Testing fusion::eeg_fmri module...\n")

    if !test_hrf() {
        print("FAIL: hrf\n")
        return 1
    }
    print("PASS: hrf\n")

    if !test_rdm_compare_identical() {
        print("FAIL: rdm_compare\n")
        return 2
    }
    print("PASS: rdm_compare\n")

    if !test_source_prior() {
        print("FAIL: source_prior\n")
        return 3
    }
    print("PASS: source_prior\n")

    print("All fusion::eeg_fmri tests PASSED\n")
    0
}
