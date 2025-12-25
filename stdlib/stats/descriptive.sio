// stats::descriptive — Descriptive Statistics with Uncertainty
//
// GUM-compliant descriptive statistics returning values with
// expanded uncertainty (k=2, 95% coverage).
//
// References:
// - JCGM 100:2008 (GUM): Evaluation of measurement data
// - NIST/SEMATECH e-Handbook of Statistical Methods

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// UNCERTAIN VALUE TYPE
// ============================================================================

/// Value with GUM-compliant expanded uncertainty
struct Uncertain {
    value: f64,      // Best estimate (mean)
    std_unc: f64,    // Standard uncertainty u(x)
    k: f64,          // Coverage factor
    expanded: f64,   // Expanded uncertainty U = k * u(x)
    dof: i64,        // Effective degrees of freedom
}

fn uncertain_new(value: f64, std_unc: f64, k: f64, dof: i64) -> Uncertain {
    Uncertain {
        value: value,
        std_unc: std_unc,
        k: k,
        expanded: k * std_unc,
        dof: dof,
    }
}

/// Check if uncertain value is valid
fn uncertain_is_valid(u: Uncertain) -> bool {
    u.std_unc >= 0.0 && u.k > 0.0
}

// ============================================================================
// BASIC STATISTICS
// ============================================================================

/// Arithmetic mean
fn mean(data: [f64; 100], n: i64) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + data[i as usize]
        i = i + 1
    }
    sum / (n as f64)
}

/// Sample variance (Bessel corrected, n-1)
fn variance(data: [f64; 100], n: i64) -> f64 {
    let m = mean(data, n)
    var sum_sq = 0.0
    var i: i64 = 0
    while i < n {
        let diff = data[i as usize] - m
        sum_sq = sum_sq + diff * diff
        i = i + 1
    }
    sum_sq / ((n - 1) as f64)
}

/// Sample standard deviation
fn std_dev(data: [f64; 100], n: i64) -> f64 {
    sqrt(variance(data, n))
}

/// Standard error of the mean
fn std_error(data: [f64; 100], n: i64) -> f64 {
    std_dev(data, n) / sqrt(n as f64)
}

// ============================================================================
// UNCERTAINTY-AWARE STATISTICS
// ============================================================================

/// Mean with GUM-compliant uncertainty (k=2, 95% coverage)
fn mean_uncertain(data: [f64; 100], n: i64) -> Uncertain {
    let m = mean(data, n)
    let se = std_error(data, n)
    let k = 2.0  // 95% coverage for normal distribution
    uncertain_new(m, se, k, n - 1)
}

/// Variance with uncertainty
/// u(s^2) = s^2 * sqrt(2/(n-1)) for normal data
fn variance_uncertain(data: [f64; 100], n: i64) -> Uncertain {
    let v = variance(data, n)
    let std_unc = v * sqrt(2.0 / ((n - 1) as f64))
    uncertain_new(v, std_unc, 2.0, n - 1)
}

/// Standard deviation with uncertainty
/// u(s) approx s / sqrt(2(n-1)) for normal data
fn std_dev_uncertain(data: [f64; 100], n: i64) -> Uncertain {
    let s = std_dev(data, n)
    let std_unc = s / sqrt(2.0 * ((n - 1) as f64))
    uncertain_new(s, std_unc, 2.0, n - 1)
}

// ============================================================================
// HIGHER MOMENTS
// ============================================================================

/// Skewness (Fisher's definition)
fn skewness(data: [f64; 100], n: i64) -> f64 {
    let m = mean(data, n)
    let s = std_dev(data, n)

    if s < 1e-10 {
        return 0.0
    }

    var sum_cubed = 0.0
    var i: i64 = 0
    while i < n {
        let z = (data[i as usize] - m) / s
        sum_cubed = sum_cubed + z * z * z
        i = i + 1
    }

    let nf = n as f64
    // Adjusted Fisher-Pearson coefficient
    (nf / ((nf - 1.0) * (nf - 2.0))) * sum_cubed
}

/// Kurtosis (excess kurtosis, Fisher's definition)
fn kurtosis(data: [f64; 100], n: i64) -> f64 {
    let m = mean(data, n)
    let s = std_dev(data, n)

    if s < 1e-10 {
        return 0.0
    }

    var sum_fourth = 0.0
    var i: i64 = 0
    while i < n {
        let z = (data[i as usize] - m) / s
        sum_fourth = sum_fourth + z * z * z * z
        i = i + 1
    }

    let nf = n as f64
    // Excess kurtosis (normal = 0)
    let raw = sum_fourth / nf
    raw - 3.0
}

// ============================================================================
// QUANTILES AND ROBUST STATISTICS
// ============================================================================

/// Median (for small sorted arrays)
fn median_sorted(data: [f64; 100], n: i64) -> f64 {
    if n == 0 {
        return 0.0
    }
    if n % 2 == 1 {
        return data[(n / 2) as usize]
    }
    let mid = n / 2
    (data[(mid - 1) as usize] + data[mid as usize]) / 2.0
}

/// Interquartile range (for sorted data)
fn iqr_sorted(data: [f64; 100], n: i64) -> f64 {
    if n < 4 {
        return 0.0
    }
    let q1_idx = n / 4
    let q3_idx = (3 * n) / 4
    data[q3_idx as usize] - data[q1_idx as usize]
}

/// Median Absolute Deviation (robust spread measure)
fn mad(data: [f64; 100], n: i64, med: f64) -> f64 {
    // Compute |x_i - median|
    var deviations: [f64; 100] = [0.0; 100]
    var i: i64 = 0
    while i < n {
        let d = data[i as usize] - med
        deviations[i as usize] = if d < 0.0 { -d } else { d }
        i = i + 1
    }

    // Simple selection for median of deviations (assuming small n)
    // For production: use proper selection algorithm
    var sum = 0.0
    i = 0
    while i < n {
        sum = sum + deviations[i as usize]
        i = i + 1
    }
    sum / (n as f64)  // Mean absolute deviation as approximation
}

// ============================================================================
// RANGE AND EXTREMES
// ============================================================================

/// Minimum value
fn min_val(data: [f64; 100], n: i64) -> f64 {
    if n == 0 {
        return 0.0
    }
    var m = data[0]
    var i: i64 = 1
    while i < n {
        if data[i as usize] < m {
            m = data[i as usize]
        }
        i = i + 1
    }
    m
}

/// Maximum value
fn max_val(data: [f64; 100], n: i64) -> f64 {
    if n == 0 {
        return 0.0
    }
    var m = data[0]
    var i: i64 = 1
    while i < n {
        if data[i as usize] > m {
            m = data[i as usize]
        }
        i = i + 1
    }
    m
}

/// Range (max - min)
fn range(data: [f64; 100], n: i64) -> f64 {
    max_val(data, n) - min_val(data, n)
}

// ============================================================================
// COVARIANCE AND CORRELATION
// ============================================================================

/// Sample covariance
fn covariance(x: [f64; 100], y: [f64; 100], n: i64) -> f64 {
    let mean_x = mean(x, n)
    let mean_y = mean(y, n)

    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + (x[i as usize] - mean_x) * (y[i as usize] - mean_y)
        i = i + 1
    }
    sum / ((n - 1) as f64)
}

/// Pearson correlation coefficient
fn correlation(x: [f64; 100], y: [f64; 100], n: i64) -> f64 {
    let cov = covariance(x, y, n)
    let sx = std_dev(x, n)
    let sy = std_dev(y, n)

    let denom = sx * sy
    if denom < 1e-10 {
        return 0.0
    }
    cov / denom
}

/// Correlation with uncertainty (Fisher z-transform based)
fn correlation_uncertain(x: [f64; 100], y: [f64; 100], n: i64) -> Uncertain {
    let r = correlation(x, y, n)

    // Fisher z-transform for SE calculation
    let r_clamp = if r > 0.999 { 0.999 } else if r < -0.999 { -0.999 } else { r }
    let z = 0.5 * log((1.0 + r_clamp) / (1.0 - r_clamp))
    let se_z = 1.0 / sqrt((n - 3) as f64)

    // Convert SE back to r-space (approximation)
    let se_r = (1.0 - r * r) * se_z

    uncertain_new(r, se_r, 2.0, n - 3)
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_mean_variance() -> bool {
    var data: [f64; 100] = [0.0; 100]
    // data = [1, 2, 3, 4, 5]
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    data[4] = 5.0

    let m = mean(data, 5)
    // Mean should be 3.0
    if abs_f64(m - 3.0) > 0.01 {
        return false
    }

    let v = variance(data, 5)
    // Variance of [1,2,3,4,5] = 2.5
    if abs_f64(v - 2.5) > 0.01 {
        return false
    }

    true
}

fn test_mean_uncertain() -> bool {
    var data: [f64; 100] = [0.0; 100]
    data[0] = 10.0
    data[1] = 12.0
    data[2] = 11.0
    data[3] = 13.0
    data[4] = 11.0

    let result = mean_uncertain(data, 5)

    // Mean should be 11.4
    if abs_f64(result.value - 11.4) > 0.01 {
        return false
    }

    // Should have expanded uncertainty
    result.expanded > 0.0
}

fn test_correlation() -> bool {
    var x: [f64; 100] = [0.0; 100]
    var y: [f64; 100] = [0.0; 100]

    // Perfect positive correlation
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0
    x[4] = 5.0

    y[0] = 2.0
    y[1] = 4.0
    y[2] = 6.0
    y[3] = 8.0
    y[4] = 10.0

    let r = correlation(x, y, 5)
    abs_f64(r - 1.0) < 0.01
}

fn test_skewness_symmetric() -> bool {
    var data: [f64; 100] = [0.0; 100]
    // Symmetric data around 0
    data[0] = -2.0
    data[1] = -1.0
    data[2] = 0.0
    data[3] = 1.0
    data[4] = 2.0

    let sk = skewness(data, 5)
    // Skewness of symmetric data should be ~0
    abs_f64(sk) < 0.1
}

fn main() -> i32 {
    print("Testing stats::descriptive module...\n")

    if !test_mean_variance() {
        print("FAIL: mean_variance\n")
        return 1
    }
    print("PASS: mean_variance\n")

    if !test_mean_uncertain() {
        print("FAIL: mean_uncertain\n")
        return 2
    }
    print("PASS: mean_uncertain\n")

    if !test_correlation() {
        print("FAIL: correlation\n")
        return 3
    }
    print("PASS: correlation\n")

    if !test_skewness_symmetric() {
        print("FAIL: skewness_symmetric\n")
        return 4
    }
    print("PASS: skewness_symmetric\n")

    print("All stats::descriptive tests PASSED\n")
    0
}
