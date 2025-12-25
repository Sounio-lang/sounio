// optimize::levenberg_marquardt — Levenberg-Marquardt Algorithm
//
// The workhorse of nonlinear least squares optimization.
// Interpolates between Gauss-Newton and gradient descent.
//
// Algorithm:
// 1. Compute Jacobian J and residuals r
// 2. Solve (J'J + λI) δ = -J'r for step δ
// 3. If f(x + δ) < f(x): accept, decrease λ
//    Else: reject, increase λ
// 4. Repeat until convergence
//
// References:
// - Levenberg (1944): "A Method for the Solution of Certain Problems in Least Squares"
// - Marquardt (1963): "An Algorithm for Least-Squares Estimation of Nonlinear Parameters"
// - Moré (1978): "The Levenberg-Marquardt Algorithm: Implementation and Theory"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
    fn exp(x: f64) -> f64;
}

// ============================================================================
// LEVENBERG-MARQUARDT CONFIG
// ============================================================================

struct LMConfig {
    initial_lambda: f64,        // Initial damping parameter
    lambda_up: f64,             // Factor to increase λ on reject
    lambda_down: f64,           // Factor to decrease λ on accept
    min_lambda: f64,            // Minimum λ (nearly Gauss-Newton)
    max_lambda: f64,            // Maximum λ (nearly gradient descent)
    max_iterations: i64,
    func_tol: f64,              // Convergence on residual change
    param_tol: f64,             // Convergence on parameter change
    grad_tol: f64,              // Convergence on gradient norm
    finite_diff_step: f64,      // Step size for Jacobian
}

fn lm_config_default() -> LMConfig {
    LMConfig {
        initial_lambda: 1.0,
        lambda_up: 10.0,
        lambda_down: 3.0,
        min_lambda: 1e-12,
        max_lambda: 1e12,
        max_iterations: 200,
        func_tol: 1e-8,
        param_tol: 1e-8,
        grad_tol: 1e-6,
        finite_diff_step: 1.4901161193847656e-8,
    }
}

fn lm_config_tight() -> LMConfig {
    var c = lm_config_default()
    c.max_iterations = 500
    c.func_tol = 1e-12
    c.param_tol = 1e-12
    c.grad_tol = 1e-10
    return c
}

fn lm_config_fast() -> LMConfig {
    var c = lm_config_default()
    c.max_iterations = 100
    c.func_tol = 1e-6
    c.param_tol = 1e-6
    c.grad_tol = 1e-4
    return c
}

// ============================================================================
// TERMINATION REASONS
// ============================================================================

fn TERM_MAX_ITER() -> i32 { 1 }
fn TERM_FUNC_TOL() -> i32 { 2 }
fn TERM_PARAM_TOL() -> i32 { 3 }
fn TERM_GRAD_TOL() -> i32 { 4 }
fn TERM_STALLED() -> i32 { 5 }

fn termination_success(reason: i32) -> bool {
    reason >= 2 && reason <= 4
}

// ============================================================================
// PARAMETER UNCERTAINTY
// ============================================================================

struct ParamUncertainty {
    value: f64,
    std_uncertainty: f64,
    expanded_k2: f64,
    coverage: f64,
    df: i64,
}

fn param_uncertainty_new(value: f64, std_unc: f64, df: i64) -> ParamUncertainty {
    let k = if df < 10 { 2.0 + 2.0 / (df as f64) } else { 2.0 }
    ParamUncertainty {
        value: value,
        std_uncertainty: std_unc,
        expanded_k2: k * std_unc,
        coverage: k,
        df: df,
    }
}

// ============================================================================
// OPTIMIZE RESULT
// ============================================================================

struct OptimizeResult {
    params: [f64; 20],
    n_params: i64,
    value: f64,
    converged: bool,
    termination: i32,
    iterations: i64,
    func_evals: i64,
    grad_evals: i64,
    has_uncertainty: bool,
    uncertainty: [ParamUncertainty; 20],
    covariance: [[f64; 20]; 20],
    mse: f64,
    dof: i64,
}

fn optimize_result_new() -> OptimizeResult {
    OptimizeResult {
        params: [0.0; 20],
        n_params: 0,
        value: 1e308,
        converged: false,
        termination: 6,
        iterations: 0,
        func_evals: 0,
        grad_evals: 0,
        has_uncertainty: false,
        uncertainty: [param_uncertainty_new(0.0, 0.0, -1); 20],
        covariance: [[0.0; 20]; 20],
        mse: 0.0,
        dof: 0,
    }
}

fn optimize_result_set_params(result: OptimizeResult, params: [f64; 20], n: i64) -> OptimizeResult {
    var r = result
    var i: i64 = 0
    while i < n && i < 20 {
        r.params[i as usize] = params[i as usize]
        i = i + 1
    }
    r.n_params = n
    return r
}

// ============================================================================
// VECTOR UTILITIES
// ============================================================================

fn vec_sum_sq(v: [f64; 100], n: i64) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + v[i as usize] * v[i as usize]
        i = i + 1
    }
    return sum
}

fn vec_norm_20(v: [f64; 20], n: i64) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + v[i as usize] * v[i as usize]
        i = i + 1
    }
    return sqrt(sum)
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

/// Compute J'J (Gram matrix)
fn compute_jtj(j: [f64; 2000], n_res: i64, n_par: i64, jtj: [f64; 400]) -> [f64; 400] {
    var result = jtj
    var i: i64 = 0
    while i < n_par {
        var jj: i64 = 0
        while jj < n_par {
            var sum = 0.0
            var k: i64 = 0
            while k < n_res {
                let ji = j[(k * n_par + i) as usize]
                let jk = j[(k * n_par + jj) as usize]
                sum = sum + ji * jk
                k = k + 1
            }
            result[(i * n_par + jj) as usize] = sum
            jj = jj + 1
        }
        i = i + 1
    }
    return result
}

/// Compute J'r
fn compute_jtr(j: [f64; 2000], r: [f64; 100], n_res: i64, n_par: i64) -> [f64; 20] {
    var jtr: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < n_par {
        var sum = 0.0
        var k: i64 = 0
        while k < n_res {
            sum = sum + j[(k * n_par + i) as usize] * r[k as usize]
            k = k + 1
        }
        jtr[i as usize] = sum
        i = i + 1
    }
    return jtr
}

/// Solve (A + λI)x = b using Cholesky decomposition
fn solve_damped_system(
    a: [f64; 400],
    b: [f64; 20],
    lambda: f64,
    n: i64
) -> ([f64; 20], bool) {
    var L: [f64; 400] = [0.0; 400]
    var y: [f64; 20] = [0.0; 20]
    var x: [f64; 20] = [0.0; 20]

    // Form A + λI and compute Cholesky: L * L' = A + λI
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j <= i {
            var sum = a[(i * n + j) as usize]
            if i == j {
                sum = sum + lambda
            }

            var k: i64 = 0
            while k < j {
                sum = sum - L[(i * n + k) as usize] * L[(j * n + k) as usize]
                k = k + 1
            }

            if i == j {
                if sum <= 0.0 {
                    return (x, false)
                }
                L[(i * n + j) as usize] = sqrt(sum)
            } else {
                let Ljj = L[(j * n + j) as usize]
                if Ljj == 0.0 {
                    return (x, false)
                }
                L[(i * n + j) as usize] = sum / Ljj
            }
            j = j + 1
        }
        i = i + 1
    }

    // Forward solve: L * y = b
    i = 0
    while i < n {
        var sum = b[i as usize]
        var j: i64 = 0
        while j < i {
            sum = sum - L[(i * n + j) as usize] * y[j as usize]
            j = j + 1
        }
        let Lii = L[(i * n + i) as usize]
        if Lii == 0.0 {
            return (x, false)
        }
        y[i as usize] = sum / Lii
        i = i + 1
    }

    // Backward solve: L' * x = y
    i = n - 1
    while i >= 0 {
        var sum = y[i as usize]
        var j = i + 1
        while j < n {
            sum = sum - L[(j * n + i) as usize] * x[j as usize]
            j = j + 1
        }
        let Lii = L[(i * n + i) as usize]
        if Lii == 0.0 {
            return (x, false)
        }
        x[i as usize] = sum / Lii
        i = i - 1
    }

    return (x, true)
}

// ============================================================================
// FINITE DIFFERENCES
// ============================================================================

fn finite_diff_jacobian_lm(
    params: [f64; 20],
    n_params: i64,
    n_residuals: i64,
    h: f64,
    // Inline residuals computation for exponential model
    x_data: [f64; 5],
    y_data: [f64; 5]
) -> [f64; 2000] {
    var jac: [f64; 2000] = [0.0; 2000]
    var work_params: [f64; 20] = [0.0; 20]
    var r_plus: [f64; 100] = [0.0; 100]
    var r_minus: [f64; 100] = [0.0; 100]

    var i: i64 = 0
    while i < n_params {
        work_params[i as usize] = params[i as usize]
        i = i + 1
    }

    var j: i64 = 0
    while j < n_params {
        let pj = params[j as usize]
        let abs_pj = if pj < 0.0 { -pj } else { pj }
        let step = h * (1.0 + abs_pj)

        // Compute r_plus
        work_params[j as usize] = pj + step
        let a_plus = work_params[0]
        let b_plus = work_params[1]
        var k: i64 = 0
        while k < n_residuals {
            let x = x_data[k as usize]
            r_plus[k as usize] = y_data[k as usize] - a_plus * exp(-b_plus * x)
            k = k + 1
        }

        // Compute r_minus
        work_params[j as usize] = pj - step
        let a_minus = work_params[0]
        let b_minus = work_params[1]
        k = 0
        while k < n_residuals {
            let x = x_data[k as usize]
            r_minus[k as usize] = y_data[k as usize] - a_minus * exp(-b_minus * x)
            k = k + 1
        }

        // Compute derivative
        var ii: i64 = 0
        while ii < n_residuals {
            let idx = (ii * n_params + j) as usize
            jac[idx] = (r_plus[ii as usize] - r_minus[ii as usize]) / (2.0 * step)
            ii = ii + 1
        }

        work_params[j as usize] = pj
        j = j + 1
    }

    return jac
}

// ============================================================================
// LEVENBERG-MARQUARDT (SIMPLIFIED)
// ============================================================================

/// Simplified LM for exponential fit test
fn lm_fit_exponential(
    initial_params: [f64; 20],
    n_params: i64,
    x_data: [f64; 5],
    y_data: [f64; 5],
    n_residuals: i64,
    config: LMConfig
) -> OptimizeResult {
    var result = optimize_result_new()
    result.n_params = n_params

    var params = initial_params
    var lambda = config.initial_lambda

    // Compute initial residuals and cost
    var residuals: [f64; 100] = [0.0; 100]
    var i: i64 = 0
    while i < n_residuals {
        let x = x_data[i as usize]
        residuals[i as usize] = y_data[i as usize] - params[0] * exp(-params[1] * x)
        i = i + 1
    }
    var cost = 0.5 * vec_sum_sq(residuals, n_residuals)
    result.func_evals = 1

    // Compute initial Jacobian
    var jac = finite_diff_jacobian_lm(params, n_params, n_residuals, config.finite_diff_step, x_data, y_data)
    result.grad_evals = 1

    var termination = TERM_MAX_ITER()
    var iterations: i64 = 0

    while iterations < config.max_iterations {
        iterations = iterations + 1

        // Compute J'J and J'r
        var jtj: [f64; 400] = [0.0; 400]
        jtj = compute_jtj(jac, n_residuals, n_params, jtj)
        let jtr = compute_jtr(jac, residuals, n_residuals, n_params)

        // Check gradient convergence
        let grad_norm = vec_norm_20(jtr, n_params)
        if grad_norm < config.grad_tol {
            termination = TERM_GRAD_TOL()
            break
        }

        // Negate J'r
        var neg_jtr: [f64; 20] = [0.0; 20]
        i = 0
        while i < n_params {
            neg_jtr[i as usize] = -jtr[i as usize]
            i = i + 1
        }

        // Try to find acceptable step
        var step_accepted = false
        var inner_iters = 0
        var delta: [f64; 20] = [0.0; 20]

        while !step_accepted && inner_iters < 20 {
            inner_iters = inner_iters + 1

            // Solve (J'J + λI) δ = -J'r
            let solve_result = solve_damped_system(jtj, neg_jtr, lambda, n_params)
            delta = solve_result.0
            let solved = solve_result.1

            if !solved {
                lambda = lambda * config.lambda_up
                if lambda > config.max_lambda {
                    lambda = config.max_lambda
                }
                continue
            }

            // Compute trial point
            var trial_params: [f64; 20] = [0.0; 20]
            i = 0
            while i < n_params {
                trial_params[i as usize] = params[i as usize] + delta[i as usize]
                i = i + 1
            }

            // Evaluate at trial
            var trial_residuals: [f64; 100] = [0.0; 100]
            i = 0
            while i < n_residuals {
                let x = x_data[i as usize]
                trial_residuals[i as usize] = y_data[i as usize] - trial_params[0] * exp(-trial_params[1] * x)
                i = i + 1
            }
            result.func_evals = result.func_evals + 1

            let trial_cost = 0.5 * vec_sum_sq(trial_residuals, n_residuals)

            if trial_cost < cost {
                step_accepted = true

                // Check param convergence
                let param_change = vec_norm_20(delta, n_params)
                let param_scale = vec_norm_20(params, n_params)
                if param_change < config.param_tol * (1.0 + param_scale) {
                    termination = TERM_PARAM_TOL()
                }

                // Check function convergence
                let cost_change = cost - trial_cost
                if cost_change < config.func_tol * cost {
                    if termination == TERM_MAX_ITER() {
                        termination = TERM_FUNC_TOL()
                    }
                }

                // Update state
                params = trial_params
                residuals = trial_residuals
                cost = trial_cost

                // Decrease lambda
                lambda = lambda / config.lambda_down
                if lambda < config.min_lambda {
                    lambda = config.min_lambda
                }

                // Update Jacobian
                jac = finite_diff_jacobian_lm(params, n_params, n_residuals, config.finite_diff_step, x_data, y_data)
                result.grad_evals = result.grad_evals + 1
            } else {
                lambda = lambda * config.lambda_up
                if lambda > config.max_lambda {
                    termination = TERM_STALLED()
                    step_accepted = true
                }
            }
        }

        if termination != TERM_MAX_ITER() {
            break
        }
    }

    // Build result
    result = optimize_result_set_params(result, params, n_params)
    result.value = 2.0 * cost
    result.converged = termination_success(termination)
    result.termination = termination
    result.iterations = iterations

    // Compute uncertainty
    let dof = n_residuals - n_params
    if dof > 0 {
        let mse = 2.0 * cost / (dof as f64)
        result.mse = mse
        result.dof = dof

        // Compute (J'J)^(-1)
        var jtj: [f64; 400] = [0.0; 400]
        jtj = compute_jtj(jac, n_residuals, n_params, jtj)

        // Invert by solving for identity columns
        var cov_success = true
        i = 0
        while i < n_params && cov_success {
            var identity_col: [f64; 20] = [0.0; 20]
            identity_col[i as usize] = 1.0

            let solve_result = solve_damped_system(jtj, identity_col, 0.0, n_params)
            let cov_col = solve_result.0
            cov_success = solve_result.1

            if cov_success {
                var j: i64 = 0
                while j < n_params {
                    result.covariance[j as usize][i as usize] = mse * cov_col[j as usize]
                    j = j + 1
                }
            }
            i = i + 1
        }

        if cov_success {
            result.has_uncertainty = true
            i = 0
            while i < n_params {
                let variance = result.covariance[i as usize][i as usize]
                let std_unc = if variance > 0.0 { sqrt(variance) } else { 0.0 }
                result.uncertainty[i as usize] = param_uncertainty_new(
                    params[i as usize],
                    std_unc,
                    dof
                )
                i = i + 1
            }
        }
    }

    return result
}

// ============================================================================
// TESTS
// ============================================================================

fn test_lm_exponential_fit() -> bool {
    let x_data: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0]
    let y_data: [f64; 5] = [2.0, 1.2131, 0.7358, 0.4463, 0.2707]

    var initial: [f64; 20] = [0.0; 20]
    initial[0] = 1.0
    initial[1] = 1.0

    let result = lm_fit_exponential(initial, 2, x_data, y_data, 5, lm_config_default())

    if !result.converged {
        print("LM did not converge\n")
        return false
    }

    let a = result.params[0]
    let b = result.params[1]

    let a_err = if a - 2.0 < 0.0 { 2.0 - a } else { a - 2.0 }
    let b_err = if b - 0.5 < 0.0 { 0.5 - b } else { b - 0.5 }

    if a_err > 0.01 {
        print("Parameter a is wrong\n")
        return false
    }
    if b_err > 0.01 {
        print("Parameter b is wrong\n")
        return false
    }

    if !result.has_uncertainty {
        print("No uncertainty computed\n")
        return false
    }

    return true
}

fn main() -> i32 {
    print("Testing optimize::levenberg_marquardt module...\n")

    if !test_lm_exponential_fit() {
        print("FAIL: lm_exponential_fit\n")
        return 1
    }
    print("PASS: lm_exponential_fit\n")

    print("All optimize::levenberg_marquardt tests PASSED\n")
    0
}
