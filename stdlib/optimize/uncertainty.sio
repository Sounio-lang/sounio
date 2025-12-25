// optimize::uncertainty — GUM-Compliant Parameter Uncertainty Quantification
//
// Implements uncertainty propagation following the Guide to the Expression
// of Uncertainty in Measurement (GUM). Provides covariance estimation,
// confidence intervals, and uncertainty propagation through models.
//
// References:
// - JCGM 100:2008 "Evaluation of measurement data - GUM"
// - JCGM 101:2008 "Propagation of distributions using Monte Carlo"
// - Bard (1974): "Nonlinear Parameter Estimation"

extern "C" {
    fn sqrt(x: f64) -> f64;
}

// ============================================================================
// UNCERTAINTY STRUCTURES
// ============================================================================

struct ParamUncertainty {
    estimate: [f64; 20],
    std_error: [f64; 20],
    ci_lower: [f64; 20],
    ci_upper: [f64; 20],
    covariance: [f64; 400],
    correlation: [f64; 400],
    n_params: i64,
    confidence_level: f64,
    dof: i64,  // degrees of freedom
}

fn uncertainty_new(n: i64) -> ParamUncertainty {
    ParamUncertainty {
        estimate: [0.0; 20],
        std_error: [1e308; 20],
        ci_lower: [-1e308; 20],
        ci_upper: [1e308; 20],
        covariance: [0.0; 400],
        correlation: [0.0; 400],
        n_params: n,
        confidence_level: 0.95,
        dof: 0,
    }
}

// ============================================================================
// T-DISTRIBUTION QUANTILES (approximation)
// ============================================================================

fn t_quantile_95(dof: i64) -> f64 {
    // Approximate t-distribution quantile for 95% confidence
    // For large dof, approaches 1.96
    if dof < 1 {
        return 12.706  // dof=1
    } else if dof == 1 {
        return 12.706
    } else if dof == 2 {
        return 4.303
    } else if dof == 3 {
        return 3.182
    } else if dof == 4 {
        return 2.776
    } else if dof == 5 {
        return 2.571
    } else if dof <= 10 {
        return 2.228
    } else if dof <= 20 {
        return 2.086
    } else if dof <= 30 {
        return 2.042
    } else if dof <= 50 {
        return 2.009
    } else if dof <= 100 {
        return 1.984
    } else {
        return 1.96
    }
}

// ============================================================================
// MATRIX OPERATIONS FOR COVARIANCE
// ============================================================================

fn mat_transpose(A: [f64; 2000], m: i64, n: i64) -> [f64; 2000] {
    // A is m x n, result is n x m
    var result: [f64; 2000] = [0.0; 2000]
    var i: i64 = 0
    while i < m {
        var j: i64 = 0
        while j < n {
            result[(j * m + i) as usize] = A[(i * n + j) as usize]
            j = j + 1
        }
        i = i + 1
    }
    return result
}

fn mat_mult_AtA(A: [f64; 2000], m: i64, n: i64) -> [f64; 400] {
    // Compute A^T * A where A is m x n, result is n x n
    var result: [f64; 400] = [0.0; 400]
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            var sum = 0.0
            var k: i64 = 0
            while k < m {
                sum = sum + A[(k * n + i) as usize] * A[(k * n + j) as usize]
                k = k + 1
            }
            result[(i * n + j) as usize] = sum
            j = j + 1
        }
        i = i + 1
    }
    return result
}

fn cholesky_lower(A: [f64; 400], n: i64) -> [f64; 400] {
    var L: [f64; 400] = [0.0; 400]
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j <= i {
            var sum = 0.0
            if j == i {
                var k: i64 = 0
                while k < j {
                    sum = sum + L[(j * n + k) as usize] * L[(j * n + k) as usize]
                    k = k + 1
                }
                let diag = A[(j * n + j) as usize] - sum
                L[(i * n + j) as usize] = if diag > 0.0 { sqrt(diag) } else { 0.0 }
            } else {
                var k: i64 = 0
                while k < j {
                    sum = sum + L[(i * n + k) as usize] * L[(j * n + k) as usize]
                    k = k + 1
                }
                if L[(j * n + j) as usize] > 1e-15 {
                    L[(i * n + j) as usize] = (A[(i * n + j) as usize] - sum) / L[(j * n + j) as usize]
                }
            }
            j = j + 1
        }
        i = i + 1
    }
    return L
}

fn forward_solve(L: [f64; 400], b: [f64; 20], n: i64) -> [f64; 20] {
    var x: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < n {
        var sum = b[i as usize]
        var j: i64 = 0
        while j < i {
            sum = sum - L[(i * n + j) as usize] * x[j as usize]
            j = j + 1
        }
        if L[(i * n + i) as usize] > 1e-15 {
            x[i as usize] = sum / L[(i * n + i) as usize]
        }
        i = i + 1
    }
    return x
}

fn backward_solve(U: [f64; 400], b: [f64; 20], n: i64) -> [f64; 20] {
    // U is upper triangular (L^T)
    var x: [f64; 20] = [0.0; 20]
    var i: i64 = n - 1
    while i >= 0 {
        var sum = b[i as usize]
        var j: i64 = i + 1
        while j < n {
            sum = sum - U[(i * n + j) as usize] * x[j as usize]
            j = j + 1
        }
        if U[(i * n + i) as usize] > 1e-15 {
            x[i as usize] = sum / U[(i * n + i) as usize]
        }
        i = i - 1
    }
    return x
}

fn invert_via_cholesky(A: [f64; 400], n: i64) -> [f64; 400] {
    // Invert symmetric positive definite matrix via Cholesky
    let L = cholesky_lower(A, n)

    // Build L^T
    var Lt: [f64; 400] = [0.0; 400]
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            Lt[(i * n + j) as usize] = L[(j * n + i) as usize]
            j = j + 1
        }
        i = i + 1
    }

    // Solve for each column of inverse
    var inv: [f64; 400] = [0.0; 400]
    i = 0
    while i < n {
        // Solve L * y = e_i
        var e: [f64; 20] = [0.0; 20]
        e[i as usize] = 1.0

        let y = forward_solve(L, e, n)
        let x = backward_solve(Lt, y, n)

        var j: i64 = 0
        while j < n {
            inv[(j * n + i) as usize] = x[j as usize]
            j = j + 1
        }
        i = i + 1
    }

    return inv
}

// ============================================================================
// COVARIANCE ESTIMATION FROM JACOBIAN
// ============================================================================

fn estimate_covariance_from_jacobian(
    jac: [f64; 2000],  // m x n Jacobian
    residuals: [f64; 100],  // m residuals
    m: i64,  // number of observations
    n: i64   // number of parameters
) -> ParamUncertainty {
    var unc = uncertainty_new(n)
    unc.dof = m - n

    // Estimate residual variance: s^2 = SSE / (m - n)
    var sse = 0.0
    var i: i64 = 0
    while i < m {
        sse = sse + residuals[i as usize] * residuals[i as usize]
        i = i + 1
    }
    let s2 = if unc.dof > 0 { sse / (unc.dof as f64) } else { sse }

    // Compute J^T * J
    let JtJ = mat_mult_AtA(jac, m, n)

    // Covariance = s^2 * (J^T J)^{-1}
    let JtJ_inv = invert_via_cholesky(JtJ, n)

    i = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            unc.covariance[(i * n + j) as usize] = s2 * JtJ_inv[(i * n + j) as usize]
            j = j + 1
        }
        i = i + 1
    }

    // Standard errors
    i = 0
    while i < n {
        let var_i = unc.covariance[(i * n + i) as usize]
        unc.std_error[i as usize] = if var_i > 0.0 { sqrt(var_i) } else { 0.0 }
        i = i + 1
    }

    // Correlation matrix
    i = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            let si = unc.std_error[i as usize]
            let sj = unc.std_error[j as usize]
            if si > 1e-15 && sj > 1e-15 {
                unc.correlation[(i * n + j) as usize] = unc.covariance[(i * n + j) as usize] / (si * sj)
            } else if i == j {
                unc.correlation[(i * n + j) as usize] = 1.0
            }
            j = j + 1
        }
        i = i + 1
    }

    // Confidence intervals
    let t = t_quantile_95(unc.dof)
    i = 0
    while i < n {
        let hw = t * unc.std_error[i as usize]
        unc.ci_lower[i as usize] = unc.estimate[i as usize] - hw
        unc.ci_upper[i as usize] = unc.estimate[i as usize] + hw
        i = i + 1
    }

    return unc
}

// ============================================================================
// PROPAGATE UNCERTAINTY (linear approximation)
// ============================================================================

fn propagate_uncertainty_linear(
    unc: ParamUncertainty,
    sensitivity: [f64; 20],  // df/d(param_i)
    n: i64
) -> (f64, f64) {
    // Returns (variance of f, standard error of f)
    // Var(f) = sum_i sum_j (df/dxi) * Cov(xi, xj) * (df/dxj)
    var var_f = 0.0
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            var_f = var_f + sensitivity[i as usize] * unc.covariance[(i * n + j) as usize] * sensitivity[j as usize]
            j = j + 1
        }
        i = i + 1
    }
    let se_f = if var_f > 0.0 { sqrt(var_f) } else { 0.0 }
    return (var_f, se_f)
}

// ============================================================================
// RELATIVE STANDARD ERROR (%CV)
// ============================================================================

fn relative_se(estimate: f64, se: f64) -> f64 {
    let abs_est = if estimate < 0.0 { -estimate } else { estimate }
    if abs_est > 1e-15 {
        return 100.0 * se / abs_est
    }
    return 0.0
}

// ============================================================================
// TESTS
// ============================================================================

fn test_covariance_estimation() -> bool {
    // Simple 2-parameter linear model: y = a + b*x
    // Data: x = [1, 2, 3, 4], y = [2.1, 3.9, 6.1, 8.0]
    // True: a=0, b=2

    // Jacobian for y = a + b*x at x=[1,2,3,4]
    // dy/da = 1, dy/db = x
    var jac: [f64; 2000] = [0.0; 2000]
    // Row 0: x=1
    jac[0] = 1.0
    jac[1] = 1.0
    // Row 1: x=2
    jac[2] = 1.0
    jac[3] = 2.0
    // Row 2: x=3
    jac[4] = 1.0
    jac[5] = 3.0
    // Row 3: x=4
    jac[6] = 1.0
    jac[7] = 4.0

    // Fitted values: a=0.05, b=1.97
    // y_fit = [2.02, 3.99, 5.96, 7.93]
    // Residuals = y - y_fit
    var residuals: [f64; 100] = [0.0; 100]
    residuals[0] = 2.1 - 2.02   // 0.08
    residuals[1] = 3.9 - 3.99   // -0.09
    residuals[2] = 6.1 - 5.96   // 0.14
    residuals[3] = 8.0 - 7.93   // 0.07

    var unc = estimate_covariance_from_jacobian(jac, residuals, 4, 2)
    unc.estimate[0] = 0.05
    unc.estimate[1] = 1.97

    // Check that standard errors are reasonable
    // For this simple case, SE should be small
    return unc.std_error[0] < 1.0 && unc.std_error[1] < 1.0
}

fn test_uncertainty_propagation() -> bool {
    var unc = uncertainty_new(2)
    unc.estimate[0] = 10.0
    unc.estimate[1] = 5.0
    unc.std_error[0] = 0.5
    unc.std_error[1] = 0.2

    // Covariance: assume independent
    unc.covariance[0] = 0.25  // var(x0)
    unc.covariance[1] = 0.0   // cov(x0, x1)
    unc.covariance[2] = 0.0   // cov(x1, x0)
    unc.covariance[3] = 0.04  // var(x1)

    // f = x0 + x1, sensitivity = [1, 1]
    var sens: [f64; 20] = [0.0; 20]
    sens[0] = 1.0
    sens[1] = 1.0

    let result = propagate_uncertainty_linear(unc, sens, 2)
    let var_f = result.0
    let se_f = result.1

    // Var(f) = Var(x0) + Var(x1) = 0.25 + 0.04 = 0.29
    // SE(f) = sqrt(0.29) ≈ 0.539
    let expected_var = 0.29
    let err = if var_f - expected_var < 0.0 { expected_var - var_f } else { var_f - expected_var }

    return err < 0.01
}

fn test_relative_se() -> bool {
    let rse1 = relative_se(100.0, 5.0)  // Should be 5%
    let rse2 = relative_se(50.0, 5.0)   // Should be 10%

    let err1 = if rse1 - 5.0 < 0.0 { 5.0 - rse1 } else { rse1 - 5.0 }
    let err2 = if rse2 - 10.0 < 0.0 { 10.0 - rse2 } else { rse2 - 10.0 }

    return err1 < 0.01 && err2 < 0.01
}

fn main() -> i32 {
    print("Testing optimize::uncertainty module...\n")

    if !test_covariance_estimation() {
        print("FAIL: covariance_estimation\n")
        return 1
    }
    print("PASS: covariance_estimation\n")

    if !test_uncertainty_propagation() {
        print("FAIL: uncertainty_propagation\n")
        return 2
    }
    print("PASS: uncertainty_propagation\n")

    if !test_relative_se() {
        print("FAIL: relative_se\n")
        return 3
    }
    print("PASS: relative_se\n")

    print("All optimize::uncertainty tests PASSED\n")
    0
}
