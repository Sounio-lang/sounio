// optimize::bfgs — BFGS Quasi-Newton Method
//
// The BFGS algorithm builds an approximation to the inverse Hessian
// using gradient information. Superlinear convergence for smooth problems.
//
// References:
// - Broyden (1970), Fletcher (1970), Goldfarb (1970), Shanno (1970)
// - Nocedal & Wright (2006): "Numerical Optimization"

extern "C" {
    fn sqrt(x: f64) -> f64;
}

// ============================================================================
// CONFIG AND RESULT TYPES
// ============================================================================

struct BFGSConfig {
    max_iterations: i64,
    grad_tol: f64,
    func_tol: f64,
    param_tol: f64,
    c1: f64,
    alpha_init: f64,
    alpha_min: f64,
    max_linesearch: i64,
    finite_diff_step: f64,
}

fn bfgs_config_default() -> BFGSConfig {
    BFGSConfig {
        max_iterations: 1000,
        grad_tol: 1e-6,
        func_tol: 1e-8,
        param_tol: 1e-8,
        c1: 1e-4,
        alpha_init: 1.0,
        alpha_min: 1e-20,
        max_linesearch: 40,
        finite_diff_step: 1.4901161193847656e-8,
    }
}

fn TERM_MAX_ITER() -> i32 { 1 }
fn TERM_GRAD_TOL() -> i32 { 4 }
fn TERM_FUNC_TOL() -> i32 { 2 }
fn TERM_PARAM_TOL() -> i32 { 3 }
fn TERM_LINESEARCH_FAIL() -> i32 { 5 }

fn termination_success(reason: i32) -> bool {
    reason >= 2 && reason <= 4
}

struct BFGSResult {
    params: [f64; 20],
    n_params: i64,
    value: f64,
    converged: bool,
    termination: i32,
    iterations: i64,
    func_evals: i64,
    grad_evals: i64,
}

fn bfgs_result_new() -> BFGSResult {
    BFGSResult {
        params: [0.0; 20],
        n_params: 0,
        value: 1e308,
        converged: false,
        termination: TERM_MAX_ITER(),
        iterations: 0,
        func_evals: 0,
        grad_evals: 0,
    }
}

// ============================================================================
// VECTOR UTILITIES
// ============================================================================

fn vec_norm(v: [f64; 20], n: i64) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + v[i as usize] * v[i as usize]
        i = i + 1
    }
    return sqrt(sum)
}

fn vec_dot(a: [f64; 20], b: [f64; 20], n: i64) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + a[i as usize] * b[i as usize]
        i = i + 1
    }
    return sum
}

fn mat_vec(H: [f64; 400], x: [f64; 20], n: i64) -> [f64; 20] {
    var y: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < n {
        var sum = 0.0
        var j: i64 = 0
        while j < n {
            sum = sum + H[(i * n + j) as usize] * x[j as usize]
            j = j + 1
        }
        y[i as usize] = sum
        i = i + 1
    }
    return y
}

fn mat_identity(n: i64) -> [f64; 400] {
    var H: [f64; 400] = [0.0; 400]
    var i: i64 = 0
    while i < n {
        H[(i * n + i) as usize] = 1.0
        i = i + 1
    }
    return H
}

// ============================================================================
// TEST FUNCTIONS (inline)
// ============================================================================

fn quadratic(params: [f64; 20]) -> f64 {
    let dx = params[0] - 3.0
    let dy = params[1] - 2.0
    return dx * dx + dy * dy
}

fn rosenbrock(params: [f64; 20]) -> f64 {
    let x = params[0]
    let y = params[1]
    let a = 1.0 - x
    let b = y - x * x
    return a * a + 100.0 * b * b
}

// ============================================================================
// BFGS FOR QUADRATIC
// ============================================================================

fn compute_gradient_quad(x: [f64; 20], n: i64, h: f64) -> [f64; 20] {
    var gr: [f64; 20] = [0.0; 20]
    var work = x

    var i: i64 = 0
    while i < n {
        let xi = x[i as usize]
        let step = h * (1.0 + if xi < 0.0 { -xi } else { xi })

        work[i as usize] = xi + step
        let fp = quadratic(work)

        work[i as usize] = xi - step
        let fm = quadratic(work)

        gr[i as usize] = (fp - fm) / (2.0 * step)
        work[i as usize] = xi
        i = i + 1
    }
    return gr
}

fn bfgs_update(H: [f64; 400], s: [f64; 20], y: [f64; 20], n: i64) -> [f64; 400] {
    let ys = vec_dot(y, s, n)
    if ys <= 1e-10 {
        return H
    }

    let rho = 1.0 / ys
    let Hy = mat_vec(H, y, n)
    let yHy = vec_dot(y, Hy, n)
    let factor = (1.0 + yHy * rho) * rho

    var H_new = H
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            let idx = (i * n + j) as usize
            H_new[idx] = H[idx]
                - rho * (Hy[i as usize] * s[j as usize] + s[i as usize] * Hy[j as usize])
                + factor * s[i as usize] * s[j as usize]
            j = j + 1
        }
        i = i + 1
    }
    return H_new
}

fn bfgs_quadratic(initial: [f64; 20], n_params: i64, config: BFGSConfig) -> BFGSResult {
    var result = bfgs_result_new()
    result.n_params = n_params

    var x = initial
    var H = mat_identity(n_params)

    var f_x = quadratic(x)
    result.func_evals = 1

    var g = compute_gradient_quad(x, n_params, config.finite_diff_step)
    result.grad_evals = result.grad_evals + 2 * n_params

    var termination = TERM_MAX_ITER()

    while result.iterations < config.max_iterations {
        result.iterations = result.iterations + 1

        let grad_norm = vec_norm(g, n_params)
        if grad_norm < config.grad_tol {
            termination = TERM_GRAD_TOL()
            break
        }

        // d = -H * g
        var d = mat_vec(H, g, n_params)
        var i: i64 = 0
        while i < n_params {
            d[i as usize] = -d[i as usize]
            i = i + 1
        }

        let slope = vec_dot(g, d, n_params)
        if slope >= 0.0 {
            termination = TERM_LINESEARCH_FAIL()
            break
        }

        // Line search
        var alpha = config.alpha_init
        var work: [f64; 20] = [0.0; 20]
        var f_new = f_x

        var ls_iter: i64 = 0
        var ls_done = false
        while ls_iter < config.max_linesearch && !ls_done {
            i = 0
            while i < n_params {
                work[i as usize] = x[i as usize] + alpha * d[i as usize]
                i = i + 1
            }

            f_new = quadratic(work)
            result.func_evals = result.func_evals + 1

            if f_new <= f_x + config.c1 * alpha * slope {
                ls_done = true
            } else {
                alpha = alpha * 0.5
                if alpha < config.alpha_min {
                    ls_done = true
                }
            }
            ls_iter = ls_iter + 1
        }

        if alpha < config.alpha_min {
            termination = TERM_LINESEARCH_FAIL()
            break
        }

        // Compute step
        var s: [f64; 20] = [0.0; 20]
        i = 0
        while i < n_params {
            s[i as usize] = alpha * d[i as usize]
            i = i + 1
        }

        let step_norm = vec_norm(s, n_params)
        let x_norm = vec_norm(x, n_params)
        if step_norm < config.param_tol * (1.0 + x_norm) {
            termination = TERM_PARAM_TOL()
            i = 0
            while i < n_params {
                x[i as usize] = x[i as usize] + s[i as usize]
                i = i + 1
            }
            f_x = f_new
            break
        }

        let f_change = f_x - f_new
        let abs_f_change = if f_change < 0.0 { -f_change } else { f_change }
        if abs_f_change < config.func_tol * (1.0 + f_x) {
            termination = TERM_FUNC_TOL()
        }

        // Update x
        i = 0
        while i < n_params {
            x[i as usize] = x[i as usize] + s[i as usize]
            i = i + 1
        }

        let g_new = compute_gradient_quad(x, n_params, config.finite_diff_step)
        result.grad_evals = result.grad_evals + 2 * n_params

        // y = g_new - g
        var y: [f64; 20] = [0.0; 20]
        i = 0
        while i < n_params {
            y[i as usize] = g_new[i as usize] - g[i as usize]
            i = i + 1
        }

        H = bfgs_update(H, s, y, n_params)
        g = g_new
        f_x = f_new

        if termination != TERM_MAX_ITER() {
            break
        }
    }

    result.params = x
    result.value = f_x
    result.termination = termination
    result.converged = termination_success(termination)

    return result
}

// ============================================================================
// BFGS FOR ROSENBROCK
// ============================================================================

fn compute_gradient_rosen(x: [f64; 20], n: i64, h: f64) -> [f64; 20] {
    var gr: [f64; 20] = [0.0; 20]
    var work = x

    var i: i64 = 0
    while i < n {
        let xi = x[i as usize]
        let step = h * (1.0 + if xi < 0.0 { -xi } else { xi })

        work[i as usize] = xi + step
        let fp = rosenbrock(work)

        work[i as usize] = xi - step
        let fm = rosenbrock(work)

        gr[i as usize] = (fp - fm) / (2.0 * step)
        work[i as usize] = xi
        i = i + 1
    }
    return gr
}

fn bfgs_rosenbrock(initial: [f64; 20], n_params: i64, config: BFGSConfig) -> BFGSResult {
    var result = bfgs_result_new()
    result.n_params = n_params

    var x = initial
    var H = mat_identity(n_params)

    var f_x = rosenbrock(x)
    result.func_evals = 1

    var g = compute_gradient_rosen(x, n_params, config.finite_diff_step)
    result.grad_evals = result.grad_evals + 2 * n_params

    var termination = TERM_MAX_ITER()

    while result.iterations < config.max_iterations {
        result.iterations = result.iterations + 1

        let grad_norm = vec_norm(g, n_params)
        if grad_norm < config.grad_tol {
            termination = TERM_GRAD_TOL()
            break
        }

        var d = mat_vec(H, g, n_params)
        var i: i64 = 0
        while i < n_params {
            d[i as usize] = -d[i as usize]
            i = i + 1
        }

        let slope = vec_dot(g, d, n_params)
        if slope >= 0.0 {
            termination = TERM_LINESEARCH_FAIL()
            break
        }

        var alpha = config.alpha_init
        var work: [f64; 20] = [0.0; 20]
        var f_new = f_x

        var ls_iter: i64 = 0
        var ls_done = false
        while ls_iter < config.max_linesearch && !ls_done {
            i = 0
            while i < n_params {
                work[i as usize] = x[i as usize] + alpha * d[i as usize]
                i = i + 1
            }

            f_new = rosenbrock(work)
            result.func_evals = result.func_evals + 1

            if f_new <= f_x + config.c1 * alpha * slope {
                ls_done = true
            } else {
                alpha = alpha * 0.5
                if alpha < config.alpha_min {
                    ls_done = true
                }
            }
            ls_iter = ls_iter + 1
        }

        if alpha < config.alpha_min {
            termination = TERM_LINESEARCH_FAIL()
            break
        }

        var s: [f64; 20] = [0.0; 20]
        i = 0
        while i < n_params {
            s[i as usize] = alpha * d[i as usize]
            i = i + 1
        }

        let step_norm = vec_norm(s, n_params)
        let x_norm = vec_norm(x, n_params)
        if step_norm < config.param_tol * (1.0 + x_norm) {
            termination = TERM_PARAM_TOL()
            i = 0
            while i < n_params {
                x[i as usize] = x[i as usize] + s[i as usize]
                i = i + 1
            }
            f_x = f_new
            break
        }

        let f_change = f_x - f_new
        let abs_f_change = if f_change < 0.0 { -f_change } else { f_change }
        if abs_f_change < config.func_tol * (1.0 + f_x) {
            termination = TERM_FUNC_TOL()
        }

        i = 0
        while i < n_params {
            x[i as usize] = x[i as usize] + s[i as usize]
            i = i + 1
        }

        let g_new = compute_gradient_rosen(x, n_params, config.finite_diff_step)
        result.grad_evals = result.grad_evals + 2 * n_params

        var y: [f64; 20] = [0.0; 20]
        i = 0
        while i < n_params {
            y[i as usize] = g_new[i as usize] - g[i as usize]
            i = i + 1
        }

        H = bfgs_update(H, s, y, n_params)
        g = g_new
        f_x = f_new

        if termination != TERM_MAX_ITER() {
            break
        }
    }

    result.params = x
    result.value = f_x
    result.termination = termination
    result.converged = termination_success(termination)

    return result
}

// ============================================================================
// TESTS
// ============================================================================

fn test_bfgs_quadratic() -> bool {
    var initial: [f64; 20] = [0.0; 20]
    let result = bfgs_quadratic(initial, 2, bfgs_config_default())

    if !result.converged {
        return false
    }

    let x_err = if result.params[0] - 3.0 < 0.0 { 3.0 - result.params[0] } else { result.params[0] - 3.0 }
    let y_err = if result.params[1] - 2.0 < 0.0 { 2.0 - result.params[1] } else { result.params[1] - 2.0 }

    return x_err < 0.001 && y_err < 0.001
}

fn test_bfgs_rosenbrock() -> bool {
    var initial: [f64; 20] = [0.0; 20]
    initial[0] = -1.0
    initial[1] = 1.0

    var config = bfgs_config_default()
    config.max_iterations = 2000

    let result = bfgs_rosenbrock(initial, 2, config)

    let x_err = if result.params[0] - 1.0 < 0.0 { 1.0 - result.params[0] } else { result.params[0] - 1.0 }
    let y_err = if result.params[1] - 1.0 < 0.0 { 1.0 - result.params[1] } else { result.params[1] - 1.0 }

    return x_err < 0.1 && y_err < 0.1
}

fn main() -> i32 {
    print("Testing optimize::bfgs module...\n")

    if !test_bfgs_quadratic() {
        print("FAIL: bfgs_quadratic\n")
        return 1
    }
    print("PASS: bfgs_quadratic\n")

    if !test_bfgs_rosenbrock() {
        print("FAIL: bfgs_rosenbrock\n")
        return 2
    }
    print("PASS: bfgs_rosenbrock\n")

    print("All optimize::bfgs tests PASSED\n")
    0
}
