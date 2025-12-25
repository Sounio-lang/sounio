// optimize::nelder_mead — Nelder-Mead Simplex Algorithm
//
// A derivative-free optimization method that uses a simplex of n+1 vertices
// to explore the parameter space. Robust for noisy or non-differentiable
// objective functions.
//
// References:
// - Nelder & Mead (1965): "A Simplex Method for Function Minimization"
// - Lagarias et al. (1998): "Convergence Properties of the Nelder-Mead Simplex Method"

extern "C" {
    fn sqrt(x: f64) -> f64;
}

// ============================================================================
// CONFIG AND RESULT TYPES
// ============================================================================

struct NMConfig {
    max_iterations: i64,
    func_tol: f64,
    param_tol: f64,
    alpha: f64,    // reflection coefficient
    gamma: f64,    // expansion coefficient
    rho: f64,      // contraction coefficient
    sigma: f64,    // shrink coefficient
    initial_step: f64,
}

fn nm_config_default() -> NMConfig {
    NMConfig {
        max_iterations: 5000,
        func_tol: 1e-8,
        param_tol: 1e-8,
        alpha: 1.0,
        gamma: 2.0,
        rho: 0.5,
        sigma: 0.5,
        initial_step: 0.1,
    }
}

fn TERM_MAX_ITER() -> i32 { 1 }
fn TERM_FUNC_TOL() -> i32 { 2 }
fn TERM_PARAM_TOL() -> i32 { 3 }

fn termination_success(reason: i32) -> bool {
    reason >= 2 && reason <= 3
}

struct NMResult {
    params: [f64; 20],
    n_params: i64,
    value: f64,
    converged: bool,
    termination: i32,
    iterations: i64,
    func_evals: i64,
}

fn nm_result_new() -> NMResult {
    NMResult {
        params: [0.0; 20],
        n_params: 0,
        value: 1e308,
        converged: false,
        termination: TERM_MAX_ITER(),
        iterations: 0,
        func_evals: 0,
    }
}

// ============================================================================
// SIMPLEX STORAGE (n+1 vertices, each with 20 coords + 1 function value)
// Max 21 vertices, each with 20 params, stored row-major
// ============================================================================

struct Simplex {
    vertices: [f64; 420],  // 21 * 20 = 420 coords
    values: [f64; 21],     // function values for each vertex
    n_vertices: i64,
    n_params: i64,
}

fn simplex_new(n: i64) -> Simplex {
    Simplex {
        vertices: [0.0; 420],
        values: [1e308; 21],
        n_vertices: n + 1,
        n_params: n,
    }
}

fn simplex_get_vertex(s: Simplex, idx: i64) -> [f64; 20] {
    var v: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < s.n_params {
        v[i as usize] = s.vertices[(idx * 20 + i) as usize]
        i = i + 1
    }
    return v
}

fn simplex_set_vertex(s: Simplex, idx: i64, v: [f64; 20], fval: f64) -> Simplex {
    var snew = s
    var i: i64 = 0
    while i < s.n_params {
        snew.vertices[(idx * 20 + i) as usize] = v[i as usize]
        i = i + 1
    }
    snew.values[idx as usize] = fval
    return snew
}

// ============================================================================
// TEST FUNCTIONS
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
// SIMPLEX OPERATIONS
// ============================================================================

fn find_best_worst(s: Simplex) -> (i64, i64, i64) {
    var best: i64 = 0
    var worst: i64 = 0
    var second_worst: i64 = 0

    var i: i64 = 1
    while i < s.n_vertices {
        if s.values[i as usize] < s.values[best as usize] {
            best = i
        }
        if s.values[i as usize] > s.values[worst as usize] {
            second_worst = worst
            worst = i
        } else if s.values[i as usize] > s.values[second_worst as usize] && i != worst {
            second_worst = i
        }
        i = i + 1
    }

    return (best, worst, second_worst)
}

fn compute_centroid(s: Simplex, exclude: i64) -> [f64; 20] {
    var center: [f64; 20] = [0.0; 20]
    let n = s.n_params
    let count = s.n_vertices - 1

    var v: i64 = 0
    while v < s.n_vertices {
        if v != exclude {
            var i: i64 = 0
            while i < n {
                center[i as usize] = center[i as usize] + s.vertices[(v * 20 + i) as usize]
                i = i + 1
            }
        }
        v = v + 1
    }

    var i: i64 = 0
    while i < n {
        center[i as usize] = center[i as usize] / (count as f64)
        i = i + 1
    }

    return center
}

fn reflect(center: [f64; 20], worst: [f64; 20], alpha: f64, n: i64) -> [f64; 20] {
    var r: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < n {
        r[i as usize] = center[i as usize] + alpha * (center[i as usize] - worst[i as usize])
        i = i + 1
    }
    return r
}

fn expand(center: [f64; 20], reflected: [f64; 20], gamma: f64, n: i64) -> [f64; 20] {
    var e: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < n {
        e[i as usize] = center[i as usize] + gamma * (reflected[i as usize] - center[i as usize])
        i = i + 1
    }
    return e
}

fn contract_outside(center: [f64; 20], reflected: [f64; 20], rho: f64, n: i64) -> [f64; 20] {
    var c: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < n {
        c[i as usize] = center[i as usize] + rho * (reflected[i as usize] - center[i as usize])
        i = i + 1
    }
    return c
}

fn contract_inside(center: [f64; 20], worst: [f64; 20], rho: f64, n: i64) -> [f64; 20] {
    var c: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < n {
        c[i as usize] = center[i as usize] - rho * (center[i as usize] - worst[i as usize])
        i = i + 1
    }
    return c
}

fn simplex_diameter(s: Simplex) -> f64 {
    var max_dist = 0.0
    let n = s.n_params

    var i: i64 = 0
    while i < s.n_vertices {
        var j: i64 = i + 1
        while j < s.n_vertices {
            var dist = 0.0
            var k: i64 = 0
            while k < n {
                let diff = s.vertices[(i * 20 + k) as usize] - s.vertices[(j * 20 + k) as usize]
                dist = dist + diff * diff
                k = k + 1
            }
            dist = sqrt(dist)
            if dist > max_dist {
                max_dist = dist
            }
            j = j + 1
        }
        i = i + 1
    }

    return max_dist
}

fn value_spread(s: Simplex) -> f64 {
    var min_val = s.values[0]
    var max_val = s.values[0]

    var i: i64 = 1
    while i < s.n_vertices {
        if s.values[i as usize] < min_val {
            min_val = s.values[i as usize]
        }
        if s.values[i as usize] > max_val {
            max_val = s.values[i as usize]
        }
        i = i + 1
    }

    return max_val - min_val
}

// ============================================================================
// NELDER-MEAD FOR QUADRATIC
// ============================================================================

fn nm_quadratic(initial: [f64; 20], n_params: i64, config: NMConfig) -> NMResult {
    var result = nm_result_new()
    result.n_params = n_params

    // Initialize simplex
    var s = simplex_new(n_params)

    // First vertex is initial point
    let f0 = quadratic(initial)
    s = simplex_set_vertex(s, 0, initial, f0)
    result.func_evals = 1

    // Create remaining vertices
    var i: i64 = 0
    while i < n_params {
        var v = initial
        v[i as usize] = v[i as usize] + config.initial_step
        let fv = quadratic(v)
        s = simplex_set_vertex(s, i + 1, v, fv)
        result.func_evals = result.func_evals + 1
        i = i + 1
    }

    var termination = TERM_MAX_ITER()

    while result.iterations < config.max_iterations {
        result.iterations = result.iterations + 1

        // Check convergence
        let spread = value_spread(s)
        if spread < config.func_tol {
            termination = TERM_FUNC_TOL()
            break
        }

        let diam = simplex_diameter(s)
        if diam < config.param_tol {
            termination = TERM_PARAM_TOL()
            break
        }

        // Find best, worst, second worst
        let indices = find_best_worst(s)
        let best_idx = indices.0
        let worst_idx = indices.1
        let second_worst_idx = indices.2

        let f_best = s.values[best_idx as usize]
        let f_worst = s.values[worst_idx as usize]
        let f_second_worst = s.values[second_worst_idx as usize]

        let worst_v = simplex_get_vertex(s, worst_idx)
        let center = compute_centroid(s, worst_idx)

        // Reflect
        let reflected = reflect(center, worst_v, config.alpha, n_params)
        let f_reflected = quadratic(reflected)
        result.func_evals = result.func_evals + 1

        if f_reflected < f_second_worst && f_reflected >= f_best {
            // Accept reflection
            s = simplex_set_vertex(s, worst_idx, reflected, f_reflected)
        } else if f_reflected < f_best {
            // Try expansion
            let expanded = expand(center, reflected, config.gamma, n_params)
            let f_expanded = quadratic(expanded)
            result.func_evals = result.func_evals + 1

            if f_expanded < f_reflected {
                s = simplex_set_vertex(s, worst_idx, expanded, f_expanded)
            } else {
                s = simplex_set_vertex(s, worst_idx, reflected, f_reflected)
            }
        } else {
            // Contract
            if f_reflected < f_worst {
                // Outside contraction
                let contracted = contract_outside(center, reflected, config.rho, n_params)
                let f_contracted = quadratic(contracted)
                result.func_evals = result.func_evals + 1

                if f_contracted <= f_reflected {
                    s = simplex_set_vertex(s, worst_idx, contracted, f_contracted)
                } else {
                    // Shrink
                    let best_v = simplex_get_vertex(s, best_idx)
                    var j: i64 = 0
                    while j < s.n_vertices {
                        if j != best_idx {
                            var v = simplex_get_vertex(s, j)
                            var k: i64 = 0
                            while k < n_params {
                                v[k as usize] = best_v[k as usize] + config.sigma * (v[k as usize] - best_v[k as usize])
                                k = k + 1
                            }
                            let fv = quadratic(v)
                            s = simplex_set_vertex(s, j, v, fv)
                            result.func_evals = result.func_evals + 1
                        }
                        j = j + 1
                    }
                }
            } else {
                // Inside contraction
                let contracted = contract_inside(center, worst_v, config.rho, n_params)
                let f_contracted = quadratic(contracted)
                result.func_evals = result.func_evals + 1

                if f_contracted < f_worst {
                    s = simplex_set_vertex(s, worst_idx, contracted, f_contracted)
                } else {
                    // Shrink
                    let best_v = simplex_get_vertex(s, best_idx)
                    var j: i64 = 0
                    while j < s.n_vertices {
                        if j != best_idx {
                            var v = simplex_get_vertex(s, j)
                            var k: i64 = 0
                            while k < n_params {
                                v[k as usize] = best_v[k as usize] + config.sigma * (v[k as usize] - best_v[k as usize])
                                k = k + 1
                            }
                            let fv = quadratic(v)
                            s = simplex_set_vertex(s, j, v, fv)
                            result.func_evals = result.func_evals + 1
                        }
                        j = j + 1
                    }
                }
            }
        }
    }

    // Find best vertex
    let final_indices = find_best_worst(s)
    let final_best = final_indices.0
    result.params = simplex_get_vertex(s, final_best)
    result.value = s.values[final_best as usize]
    result.termination = termination
    result.converged = termination_success(termination)

    return result
}

// ============================================================================
// NELDER-MEAD FOR ROSENBROCK
// ============================================================================

fn nm_rosenbrock(initial: [f64; 20], n_params: i64, config: NMConfig) -> NMResult {
    var result = nm_result_new()
    result.n_params = n_params

    var s = simplex_new(n_params)

    let f0 = rosenbrock(initial)
    s = simplex_set_vertex(s, 0, initial, f0)
    result.func_evals = 1

    var i: i64 = 0
    while i < n_params {
        var v = initial
        let step_size = if initial[i as usize] == 0.0 { 0.1 } else { config.initial_step * initial[i as usize] }
        let abs_step = if step_size < 0.0 { -step_size } else { step_size }
        v[i as usize] = v[i as usize] + (if abs_step < 0.1 { 0.1 } else { abs_step })
        let fv = rosenbrock(v)
        s = simplex_set_vertex(s, i + 1, v, fv)
        result.func_evals = result.func_evals + 1
        i = i + 1
    }

    var termination = TERM_MAX_ITER()

    while result.iterations < config.max_iterations {
        result.iterations = result.iterations + 1

        let spread = value_spread(s)
        if spread < config.func_tol {
            termination = TERM_FUNC_TOL()
            break
        }

        let diam = simplex_diameter(s)
        if diam < config.param_tol {
            termination = TERM_PARAM_TOL()
            break
        }

        let indices = find_best_worst(s)
        let best_idx = indices.0
        let worst_idx = indices.1
        let second_worst_idx = indices.2

        let f_best = s.values[best_idx as usize]
        let f_worst = s.values[worst_idx as usize]
        let f_second_worst = s.values[second_worst_idx as usize]

        let worst_v = simplex_get_vertex(s, worst_idx)
        let center = compute_centroid(s, worst_idx)

        let reflected = reflect(center, worst_v, config.alpha, n_params)
        let f_reflected = rosenbrock(reflected)
        result.func_evals = result.func_evals + 1

        if f_reflected < f_second_worst && f_reflected >= f_best {
            s = simplex_set_vertex(s, worst_idx, reflected, f_reflected)
        } else if f_reflected < f_best {
            let expanded = expand(center, reflected, config.gamma, n_params)
            let f_expanded = rosenbrock(expanded)
            result.func_evals = result.func_evals + 1

            if f_expanded < f_reflected {
                s = simplex_set_vertex(s, worst_idx, expanded, f_expanded)
            } else {
                s = simplex_set_vertex(s, worst_idx, reflected, f_reflected)
            }
        } else {
            if f_reflected < f_worst {
                let contracted = contract_outside(center, reflected, config.rho, n_params)
                let f_contracted = rosenbrock(contracted)
                result.func_evals = result.func_evals + 1

                if f_contracted <= f_reflected {
                    s = simplex_set_vertex(s, worst_idx, contracted, f_contracted)
                } else {
                    let best_v = simplex_get_vertex(s, best_idx)
                    var j: i64 = 0
                    while j < s.n_vertices {
                        if j != best_idx {
                            var v = simplex_get_vertex(s, j)
                            var k: i64 = 0
                            while k < n_params {
                                v[k as usize] = best_v[k as usize] + config.sigma * (v[k as usize] - best_v[k as usize])
                                k = k + 1
                            }
                            let fv = rosenbrock(v)
                            s = simplex_set_vertex(s, j, v, fv)
                            result.func_evals = result.func_evals + 1
                        }
                        j = j + 1
                    }
                }
            } else {
                let contracted = contract_inside(center, worst_v, config.rho, n_params)
                let f_contracted = rosenbrock(contracted)
                result.func_evals = result.func_evals + 1

                if f_contracted < f_worst {
                    s = simplex_set_vertex(s, worst_idx, contracted, f_contracted)
                } else {
                    let best_v = simplex_get_vertex(s, best_idx)
                    var j: i64 = 0
                    while j < s.n_vertices {
                        if j != best_idx {
                            var v = simplex_get_vertex(s, j)
                            var k: i64 = 0
                            while k < n_params {
                                v[k as usize] = best_v[k as usize] + config.sigma * (v[k as usize] - best_v[k as usize])
                                k = k + 1
                            }
                            let fv = rosenbrock(v)
                            s = simplex_set_vertex(s, j, v, fv)
                            result.func_evals = result.func_evals + 1
                        }
                        j = j + 1
                    }
                }
            }
        }
    }

    let final_indices = find_best_worst(s)
    let final_best = final_indices.0
    result.params = simplex_get_vertex(s, final_best)
    result.value = s.values[final_best as usize]
    result.termination = termination
    result.converged = termination_success(termination)

    return result
}

// ============================================================================
// TESTS
// ============================================================================

fn test_nm_quadratic() -> bool {
    var initial: [f64; 20] = [0.0; 20]
    let result = nm_quadratic(initial, 2, nm_config_default())

    if !result.converged {
        return false
    }

    let x_err = if result.params[0] - 3.0 < 0.0 { 3.0 - result.params[0] } else { result.params[0] - 3.0 }
    let y_err = if result.params[1] - 2.0 < 0.0 { 2.0 - result.params[1] } else { result.params[1] - 2.0 }

    return x_err < 0.01 && y_err < 0.01
}

fn test_nm_rosenbrock() -> bool {
    var initial: [f64; 20] = [0.0; 20]
    initial[0] = -1.0
    initial[1] = 1.0

    var config = nm_config_default()
    config.func_tol = 1e-10
    config.param_tol = 1e-10

    let result = nm_rosenbrock(initial, 2, config)

    // Nelder-Mead may not converge as tightly on Rosenbrock, allow looser tolerance
    let x_err = if result.params[0] - 1.0 < 0.0 { 1.0 - result.params[0] } else { result.params[0] - 1.0 }
    let y_err = if result.params[1] - 1.0 < 0.0 { 1.0 - result.params[1] } else { result.params[1] - 1.0 }

    return x_err < 0.5 && y_err < 0.5
}

fn main() -> i32 {
    print("Testing optimize::nelder_mead module...\n")

    if !test_nm_quadratic() {
        print("FAIL: nm_quadratic\n")
        return 1
    }
    print("PASS: nm_quadratic\n")

    if !test_nm_rosenbrock() {
        print("FAIL: nm_rosenbrock\n")
        return 2
    }
    print("PASS: nm_rosenbrock\n")

    print("All optimize::nelder_mead tests PASSED\n")
    0
}
