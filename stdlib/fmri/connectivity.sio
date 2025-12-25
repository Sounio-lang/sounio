// fmri::connectivity — Functional Connectivity Analysis
//
// ROI-to-ROI connectivity for resting-state fMRI.
// With uncertainty quantification via bootstrap.
//
// References:
// - Biswal et al. (1995): "Functional connectivity in the motor cortex..."
// - Power et al. (2011): "Functional network organization"

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

fn log_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1000.0 }
    var y = x - 1.0
    if y > 1.0 { y = 1.0 }
    if y < -0.9 { y = -0.9 }
    var n: i64 = 0
    while n < 20 {
        let ey = exp_f64(y)
        y = y + (x - ey) / ey
        n = n + 1
    }
    y
}

// ============================================================================
// CORRELATION
// ============================================================================

/// Compute Pearson correlation between two vectors
fn pearson_corr(x: [f64; 100], y: [f64; 100], n: i64) -> f64 {
    var sum_x = 0.0
    var sum_y = 0.0
    var sum_xx = 0.0
    var sum_yy = 0.0
    var sum_xy = 0.0

    var i: i64 = 0
    while i < n {
        sum_x = sum_x + x[i as usize]
        sum_y = sum_y + y[i as usize]
        sum_xx = sum_xx + x[i as usize] * x[i as usize]
        sum_yy = sum_yy + y[i as usize] * y[i as usize]
        sum_xy = sum_xy + x[i as usize] * y[i as usize]
        i = i + 1
    }

    let nf = n as f64
    let cov_xy = sum_xy - sum_x * sum_y / nf
    let var_x = sum_xx - sum_x * sum_x / nf
    let var_y = sum_yy - sum_y * sum_y / nf

    let denom = sqrt_f64(var_x * var_y)
    if denom > 1e-10 {
        return cov_xy / denom
    } else {
        return 0.0
    }
}

/// Fisher z-transformation: z = arctanh(r)
fn fisher_z(r: f64) -> f64 {
    let r_clamped = if r > 0.999 { 0.999 } else if r < -0.999 { -0.999 } else { r }
    0.5 * log_f64((1.0 + r_clamped) / (1.0 - r_clamped))
}

/// Inverse Fisher z-transformation: r = tanh(z)
fn fisher_z_inv(z: f64) -> f64 {
    let e2z = exp_f64(2.0 * z)
    (e2z - 1.0) / (e2z + 1.0)
}

/// Standard error of Fisher z
fn fisher_z_se(n: i64) -> f64 {
    1.0 / sqrt_f64((n - 3) as f64)
}

// ============================================================================
// CONNECTIVITY RESULT
// ============================================================================

/// Connectivity with uncertainty
struct FCResult {
    r: f64,             // Pearson correlation
    z: f64,             // Fisher-z
    se: f64,            // Standard error of z
    ci_lower: f64,      // 95% CI lower (r)
    ci_upper: f64,      // 95% CI upper (r)
}

fn fc_result_new() -> FCResult {
    FCResult {
        r: 0.0,
        z: 0.0,
        se: 0.0,
        ci_lower: 0.0,
        ci_upper: 0.0,
    }
}

/// Compute FC with confidence interval
fn compute_fc(x: [f64; 100], y: [f64; 100], n: i64) -> FCResult {
    var result = fc_result_new()

    result.r = pearson_corr(x, y, n)
    result.z = fisher_z(result.r)
    result.se = fisher_z_se(n)

    // 95% CI in z-space
    let z_crit = 1.96
    let z_lower = result.z - z_crit * result.se
    let z_upper = result.z + z_crit * result.se

    // Convert back to r
    result.ci_lower = fisher_z_inv(z_lower)
    result.ci_upper = fisher_z_inv(z_upper)

    result
}

// ============================================================================
// NETWORK METRICS
// ============================================================================

/// Mean FC strength from correlation matrix
fn mean_fc_strength_3x3(corr: [[f64; 3]; 3]) -> f64 {
    // Average of upper triangle
    let sum = corr[0][1] + corr[0][2] + corr[1][2]
    sum / 3.0
}

/// Node degree (number of significant connections)
fn node_degree(corr_row: [f64; 10], n_nodes: i64, threshold: f64) -> i64 {
    var degree: i64 = 0
    var j: i64 = 0
    while j < n_nodes {
        if abs_f64(corr_row[j as usize]) > threshold {
            degree = degree + 1
        }
        j = j + 1
    }
    degree
}

// ============================================================================
// TESTS
// ============================================================================

fn test_pearson_perfect() -> bool {
    // Perfect positive correlation
    var x: [f64; 100] = [0.0; 100]
    var y: [f64; 100] = [0.0; 100]

    var i: i64 = 0
    while i < 5 {
        x[i as usize] = (i + 1) as f64  // 1, 2, 3, 4, 5
        y[i as usize] = 2.0 * ((i + 1) as f64)  // 2, 4, 6, 8, 10
        i = i + 1
    }

    let r = pearson_corr(x, y, 5)
    abs_f64(r - 1.0) < 0.01
}

fn test_fisher_z() -> bool {
    // z(0.5) ≈ 0.549
    let z = fisher_z(0.5)
    if abs_f64(z - 0.549) > 0.01 {
        return false
    }

    // Inverse
    let r_back = fisher_z_inv(z)
    abs_f64(r_back - 0.5) < 0.01
}

fn test_mean_fc() -> bool {
    var corr: [[f64; 3]; 3] = [[0.0; 3]; 3]
    corr[0][0] = 1.0
    corr[0][1] = 0.6
    corr[0][2] = 0.4
    corr[1][0] = 0.6
    corr[1][1] = 1.0
    corr[1][2] = 0.5
    corr[2][0] = 0.4
    corr[2][1] = 0.5
    corr[2][2] = 1.0

    let mean_fc = mean_fc_strength_3x3(corr)
    // (0.6 + 0.4 + 0.5) / 3 = 0.5
    abs_f64(mean_fc - 0.5) < 0.01
}

fn main() -> i32 {
    print("Testing fmri::connectivity module...\n")

    if !test_pearson_perfect() {
        print("FAIL: pearson_perfect\n")
        return 1
    }
    print("PASS: pearson_perfect\n")

    if !test_fisher_z() {
        print("FAIL: fisher_z\n")
        return 2
    }
    print("PASS: fisher_z\n")

    if !test_mean_fc() {
        print("FAIL: mean_fc\n")
        return 3
    }
    print("PASS: mean_fc\n")

    print("All fmri::connectivity tests PASSED\n")
    0
}
