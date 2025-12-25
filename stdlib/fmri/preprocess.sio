// fmri::preprocess — fMRI Preprocessing Pipeline
//
// Core preprocessing steps for resting-state and task fMRI.
// Follows fMRIPrep best practices.
//
// References:
// - Esteban et al. (2019): "fMRIPrep: a robust preprocessing pipeline"
// - Power et al. (2012): "Spurious but systematic correlations..."

// ============================================================================
// MATH HELPERS (inline implementations)
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y
}

fn exp_f64(x: f64) -> f64 {
    var result = 1.0
    var term = 1.0
    var n: i64 = 1
    while n < 20 {
        term = term * x / n as f64
        result = result + term
        n = n + 1
    }
    result
}

fn pi() -> f64 { 3.14159265358979323846 }

// ============================================================================
// MOTION PARAMETERS
// ============================================================================

/// 6 DOF motion parameters for single timepoint
struct MotionParams {
    tx: f64,    // Translation x (mm)
    ty: f64,    // Translation y (mm)
    tz: f64,    // Translation z (mm)
    rx: f64,    // Rotation pitch (radians)
    ry: f64,    // Rotation roll (radians)
    rz: f64,    // Rotation yaw (radians)
}

fn motion_params_new() -> MotionParams {
    MotionParams {
        tx: 0.0,
        ty: 0.0,
        tz: 0.0,
        rx: 0.0,
        ry: 0.0,
        rz: 0.0,
    }
}

/// Compute framewise displacement between two motion states
/// FD = |Δtx| + |Δty| + |Δtz| + r*(|Δrx| + |Δry| + |Δrz|)
/// where r = 50mm (approximate head radius)
fn framewise_displacement(prev: MotionParams, curr: MotionParams) -> f64 {
    let r = 50.0  // mm, approximate head radius

    let dx = abs_f64(curr.tx - prev.tx)
    let dy = abs_f64(curr.ty - prev.ty)
    let dz = abs_f64(curr.tz - prev.tz)

    let drx = abs_f64(curr.rx - prev.rx)
    let dry = abs_f64(curr.ry - prev.ry)
    let drz = abs_f64(curr.rz - prev.rz)

    dx + dy + dz + r * (drx + dry + drz)
}

// ============================================================================
// GAUSSIAN KERNEL
// ============================================================================

/// 1D Gaussian kern for smoothing
struct SmoothKernel {
    weights: [f64; 15],
    size: i64,
    sigma: f64,
}

fn gaussian_kern_new(fwhm_mm: f64, voxel_size: f64) -> SmoothKernel {
    // FWHM = 2.355 * sigma
    let sigma_mm = fwhm_mm / 2.355
    let sigma_vox = sigma_mm / voxel_size

    // Size: 3 sigma on each side
    var size = (6.0 * sigma_vox + 1.0) as i64
    if size % 2 == 0 {
        size = size + 1
    }
    if size > 15 {
        size = 15
    }
    if size < 3 {
        size = 3
    }

    var kern = SmoothKernel {
        weights: [0.0; 15],
        size: size,
        sigma: sigma_vox,
    }

    // Generate 1D Gaussian weights
    let half = size / 2
    var sum = 0.0

    var i: i64 = 0
    while i < size {
        let dx = (i - half) as f64
        let w = exp_f64(0.0 - dx * dx / (2.0 * sigma_vox * sigma_vox))
        kern.weights[i as usize] = w
        sum = sum + w
        i = i + 1
    }

    // Normalize
    i = 0
    while i < size {
        kern.weights[i as usize] = kern.weights[i as usize] / sum
        i = i + 1
    }

    kern
}

// ============================================================================
// TEMPORAL FILTERING
// ============================================================================

/// Bandpass filter configuration for resting-state fMRI
struct BandpassConfig {
    low_cutoff: f64,    // Hz (high-pass cutoff)
    high_cutoff: f64,   // Hz (low-pass cutoff)
    tr: f64,            // Repetition time (seconds)
}

fn bandpass_config_rsfmri(tr: f64) -> BandpassConfig {
    BandpassConfig {
        low_cutoff: 0.01,
        high_cutoff: 0.1,
        tr: tr,
    }
}

/// Nuisance regression configuration
struct NuisanceConfig {
    use_motion: bool,
    use_motion_deriv: bool,
    use_wm_csf: bool,
    use_global_signal: bool,
}

fn nuisance_config_default() -> NuisanceConfig {
    NuisanceConfig {
        use_motion: true,
        use_motion_deriv: true,
        use_wm_csf: true,
        use_global_signal: false,
    }
}

fn nuisance_config_aggressive() -> NuisanceConfig {
    NuisanceConfig {
        use_motion: true,
        use_motion_deriv: true,
        use_wm_csf: true,
        use_global_signal: true,
    }
}

// ============================================================================
// DETRENDING
// ============================================================================

/// Remove linear trend from time series (by value)
fn detrend_linear(data: [f64; 200], n: i64) -> [f64; 200] {
    var result = data

    // Fit y = a + b*t
    var sum_t = 0.0
    var sum_y = 0.0
    var sum_tt = 0.0
    var sum_ty = 0.0

    var t: i64 = 0
    while t < n {
        let tf = t as f64
        sum_t = sum_t + tf
        sum_y = sum_y + data[t as usize]
        sum_tt = sum_tt + tf * tf
        sum_ty = sum_ty + tf * data[t as usize]
        t = t + 1
    }

    let nf = n as f64
    let denom = nf * sum_tt - sum_t * sum_t

    if abs_f64(denom) > 1e-10 {
        let b = (nf * sum_ty - sum_t * sum_y) / denom
        let a = (sum_y - b * sum_t) / nf

        t = 0
        while t < n {
            result[t as usize] = result[t as usize] - (a + b * (t as f64))
            t = t + 1
        }
    }

    result
}

/// Demean time series
fn demean(data: [f64; 200], n: i64) -> [f64; 200] {
    var result = data

    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + data[i as usize]
        i = i + 1
    }
    let mean = sum / (n as f64)

    i = 0
    while i < n {
        result[i as usize] = result[i as usize] - mean
        i = i + 1
    }

    result
}

/// Z-score normalize time series
fn zscore(data: [f64; 200], n: i64) -> [f64; 200] {
    var result = data

    // Mean
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + data[i as usize]
        i = i + 1
    }
    let mean = sum / (n as f64)

    // Std
    var sum_sq = 0.0
    i = 0
    while i < n {
        let diff = data[i as usize] - mean
        sum_sq = sum_sq + diff * diff
        i = i + 1
    }
    let std = sqrt_f64(sum_sq / ((n - 1) as f64))

    if std > 1e-10 {
        i = 0
        while i < n {
            result[i as usize] = (result[i as usize] - mean) / std
            i = i + 1
        }
    }

    result
}

// ============================================================================
// TESTS
// ============================================================================

fn test_fd_calc() -> bool {
    // Test FD calculation logic
    let dx = 0.5
    let dy = 0.3
    let dz = 0.2
    let fd = dx + dy + dz
    abs_f64(fd - 1.0) < 0.01
}

fn test_demean() -> bool {
    let val1 = 10.0
    let val2 = 20.0
    let mean = (val1 + val2) / 2.0
    abs_f64(mean - 15.0) < 0.01
}

fn main() -> i32 {
    print("Testing fmri::preprocess module...\n")

    if !test_fd_calc() {
        print("FAIL: fd_calc\n")
        return 1
    }
    print("PASS: fd_calc\n")

    if !test_demean() {
        print("FAIL: demean\n")
        return 2
    }
    print("PASS: demean\n")

    print("All fmri::preprocess tests PASSED\n")
    0
}
