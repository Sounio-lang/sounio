// optimize::differential_evolution — Differential Evolution Global Optimizer
//
// A population-based stochastic optimization algorithm for finding global
// optima. Particularly effective for non-convex, multi-modal, and high-
// dimensional problems.
//
// References:
// - Storn & Price (1997): "Differential Evolution - A Simple and Efficient Heuristic"
// - Price, Storn & Lampinen (2005): "Differential Evolution: A Practical Approach"

extern "C" {
    fn sqrt(x: f64) -> f64;
}

// ============================================================================
// CONFIG AND RESULT TYPES
// ============================================================================

struct DEConfig {
    pop_size: i64,
    max_generations: i64,
    F: f64,              // mutation scale factor [0,2]
    CR: f64,             // crossover probability [0,1]
    func_tol: f64,
    strategy: i32,       // 0=rand/1, 1=best/1
}

fn de_config_default() -> DEConfig {
    DEConfig {
        pop_size: 50,
        max_generations: 1000,
        F: 0.8,
        CR: 0.9,
        func_tol: 1e-8,
        strategy: 0,
    }
}

fn TERM_MAX_GEN() -> i32 { 1 }
fn TERM_FUNC_TOL() -> i32 { 2 }

fn termination_success(reason: i32) -> bool {
    reason >= 2
}

struct DEResult {
    params: [f64; 20],
    n_params: i64,
    value: f64,
    converged: bool,
    termination: i32,
    generations: i64,
    func_evals: i64,
}

fn de_result_new() -> DEResult {
    DEResult {
        params: [0.0; 20],
        n_params: 0,
        value: 1e308,
        converged: false,
        termination: TERM_MAX_GEN(),
        generations: 0,
        func_evals: 0,
    }
}

// ============================================================================
// SIMPLE LINEAR CONGRUENTIAL RNG
// ============================================================================

struct SimpleRng {
    state: i64,
}

fn rng_new(seed: i64) -> SimpleRng {
    SimpleRng { state: seed }
}

fn rng_next(r: SimpleRng) -> (SimpleRng, i64) {
    // LCG parameters (same as glibc)
    let a: i64 = 1103515245
    let c: i64 = 12345
    let m: i64 = 2147483648  // 2^31
    let new_state = (a * r.state + c) % m
    return (SimpleRng { state: new_state }, new_state)
}

fn rng_uniform(r: SimpleRng) -> (SimpleRng, f64) {
    let result = rng_next(r)
    let new_r = result.0
    let val = result.1
    let u = (val as f64) / 2147483648.0
    return (new_r, u)
}

fn rng_int_range(r: SimpleRng, low: i64, high: i64) -> (SimpleRng, i64) {
    let result = rng_uniform(r)
    let new_r = result.0
    let u = result.1
    let range = high - low
    let idx = low + (u * (range as f64)) as i64
    return (new_r, if idx >= high { high - 1 } else { idx })
}

// ============================================================================
// POPULATION STORAGE
// Max 100 individuals, each with 20 params
// ============================================================================

struct Population {
    individuals: [f64; 2000],  // 100 * 20
    fitness: [f64; 100],
    size: i64,
    n_params: i64,
    best_idx: i64,
}

fn pop_new(size: i64, n: i64) -> Population {
    Population {
        individuals: [0.0; 2000],
        fitness: [1e308; 100],
        size: size,
        n_params: n,
        best_idx: 0,
    }
}

fn pop_get(p: Population, idx: i64) -> [f64; 20] {
    var v: [f64; 20] = [0.0; 20]
    var i: i64 = 0
    while i < p.n_params {
        v[i as usize] = p.individuals[(idx * 20 + i) as usize]
        i = i + 1
    }
    return v
}

fn pop_set(p: Population, idx: i64, v: [f64; 20], fit: f64) -> Population {
    var pnew = p
    var i: i64 = 0
    while i < p.n_params {
        pnew.individuals[(idx * 20 + i) as usize] = v[i as usize]
        i = i + 1
    }
    pnew.fitness[idx as usize] = fit

    // Update best if necessary
    if fit < pnew.fitness[pnew.best_idx as usize] {
        pnew.best_idx = idx
    }

    return pnew
}

fn pop_find_best(p: Population) -> i64 {
    var best: i64 = 0
    var i: i64 = 1
    while i < p.size {
        if p.fitness[i as usize] < p.fitness[best as usize] {
            best = i
        }
        i = i + 1
    }
    return best
}

fn pop_best_fitness(p: Population) -> f64 {
    let best = pop_find_best(p)
    return p.fitness[best as usize]
}

fn pop_worst_fitness(p: Population) -> f64 {
    var worst = p.fitness[0]
    var i: i64 = 1
    while i < p.size {
        if p.fitness[i as usize] > worst {
            worst = p.fitness[i as usize]
        }
        i = i + 1
    }
    return worst
}

// ============================================================================
// TEST FUNCTIONS
// ============================================================================

fn rastrigin(params: [f64; 20], n: i64) -> f64 {
    let pi = 3.14159265358979323846
    var sum = 10.0 * (n as f64)
    var i: i64 = 0
    while i < n {
        let x = params[i as usize]
        sum = sum + x * x - 10.0 * cos_approx(2.0 * pi * x)
        i = i + 1
    }
    return sum
}

fn cos_approx(x: f64) -> f64 {
    // Cosine approximation using Taylor series
    // Reduce x to [-pi, pi]
    let pi = 3.14159265358979323846
    let two_pi = 2.0 * pi
    var y = x
    while y > pi {
        y = y - two_pi
    }
    while y < -pi {
        y = y + two_pi
    }
    // Taylor series for cos
    let y2 = y * y
    let y4 = y2 * y2
    let y6 = y4 * y2
    let y8 = y6 * y2
    return 1.0 - y2/2.0 + y4/24.0 - y6/720.0 + y8/40320.0
}

fn sphere(params: [f64; 20], n: i64) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        let x = params[i as usize]
        sum = sum + x * x
        i = i + 1
    }
    return sum
}

// ============================================================================
// DIFFERENTIAL EVOLUTION FOR SPHERE
// ============================================================================

fn de_sphere(lower: [f64; 20], upper: [f64; 20], n_params: i64, config: DEConfig) -> DEResult {
    var result = de_result_new()
    result.n_params = n_params

    let pop_size = if config.pop_size > 100 { 100 } else { config.pop_size }
    var pop = pop_new(pop_size, n_params)
    var rng = rng_new(12345)

    // Initialize population randomly
    var i: i64 = 0
    while i < pop_size {
        var v: [f64; 20] = [0.0; 20]
        var j: i64 = 0
        while j < n_params {
            let r = rng_uniform(rng)
            rng = r.0
            let u = r.1
            v[j as usize] = lower[j as usize] + u * (upper[j as usize] - lower[j as usize])
            j = j + 1
        }
        let fit = sphere(v, n_params)
        pop = pop_set(pop, i, v, fit)
        result.func_evals = result.func_evals + 1
        i = i + 1
    }

    var termination = TERM_MAX_GEN()

    while result.generations < config.max_generations {
        result.generations = result.generations + 1

        // Check convergence
        let best_fit = pop_best_fitness(pop)
        let worst_fit = pop_worst_fitness(pop)
        if worst_fit - best_fit < config.func_tol {
            termination = TERM_FUNC_TOL()
            break
        }

        // For each individual
        i = 0
        while i < pop_size {
            let target = pop_get(pop, i)

            // Select 3 distinct random individuals
            var r1: i64 = i
            while r1 == i {
                let sel = rng_int_range(rng, 0, pop_size)
                rng = sel.0
                r1 = sel.1
            }

            var r2: i64 = i
            while r2 == i || r2 == r1 {
                let sel = rng_int_range(rng, 0, pop_size)
                rng = sel.0
                r2 = sel.1
            }

            var r3: i64 = i
            while r3 == i || r3 == r1 || r3 == r2 {
                let sel = rng_int_range(rng, 0, pop_size)
                rng = sel.0
                r3 = sel.1
            }

            let x1 = pop_get(pop, r1)
            let x2 = pop_get(pop, r2)
            let x3 = pop_get(pop, r3)

            // Mutation: v = x1 + F * (x2 - x3)
            var mutant: [f64; 20] = [0.0; 20]
            var j: i64 = 0
            while j < n_params {
                mutant[j as usize] = x1[j as usize] + config.F * (x2[j as usize] - x3[j as usize])
                // Bound
                if mutant[j as usize] < lower[j as usize] {
                    mutant[j as usize] = lower[j as usize]
                }
                if mutant[j as usize] > upper[j as usize] {
                    mutant[j as usize] = upper[j as usize]
                }
                j = j + 1
            }

            // Crossover
            var trial: [f64; 20] = [0.0; 20]
            let jrand_sel = rng_int_range(rng, 0, n_params)
            rng = jrand_sel.0
            let jrand = jrand_sel.1

            j = 0
            while j < n_params {
                let r = rng_uniform(rng)
                rng = r.0
                let u = r.1

                if u < config.CR || j == jrand {
                    trial[j as usize] = mutant[j as usize]
                } else {
                    trial[j as usize] = target[j as usize]
                }
                j = j + 1
            }

            // Selection
            let fit_trial = sphere(trial, n_params)
            result.func_evals = result.func_evals + 1

            if fit_trial <= pop.fitness[i as usize] {
                pop = pop_set(pop, i, trial, fit_trial)
            }

            i = i + 1
        }
    }

    // Extract best
    let best_idx = pop_find_best(pop)
    result.params = pop_get(pop, best_idx)
    result.value = pop.fitness[best_idx as usize]
    result.termination = termination
    result.converged = termination_success(termination)

    return result
}

// ============================================================================
// DIFFERENTIAL EVOLUTION FOR RASTRIGIN
// ============================================================================

fn de_rastrigin(lower: [f64; 20], upper: [f64; 20], n_params: i64, config: DEConfig) -> DEResult {
    var result = de_result_new()
    result.n_params = n_params

    let pop_size = if config.pop_size > 100 { 100 } else { config.pop_size }
    var pop = pop_new(pop_size, n_params)
    var rng = rng_new(54321)

    // Initialize population
    var i: i64 = 0
    while i < pop_size {
        var v: [f64; 20] = [0.0; 20]
        var j: i64 = 0
        while j < n_params {
            let r = rng_uniform(rng)
            rng = r.0
            let u = r.1
            v[j as usize] = lower[j as usize] + u * (upper[j as usize] - lower[j as usize])
            j = j + 1
        }
        let fit = rastrigin(v, n_params)
        pop = pop_set(pop, i, v, fit)
        result.func_evals = result.func_evals + 1
        i = i + 1
    }

    var termination = TERM_MAX_GEN()

    while result.generations < config.max_generations {
        result.generations = result.generations + 1

        let best_fit = pop_best_fitness(pop)
        let worst_fit = pop_worst_fitness(pop)
        if worst_fit - best_fit < config.func_tol {
            termination = TERM_FUNC_TOL()
            break
        }

        i = 0
        while i < pop_size {
            let target = pop_get(pop, i)

            var r1: i64 = i
            while r1 == i {
                let sel = rng_int_range(rng, 0, pop_size)
                rng = sel.0
                r1 = sel.1
            }

            var r2: i64 = i
            while r2 == i || r2 == r1 {
                let sel = rng_int_range(rng, 0, pop_size)
                rng = sel.0
                r2 = sel.1
            }

            var r3: i64 = i
            while r3 == i || r3 == r1 || r3 == r2 {
                let sel = rng_int_range(rng, 0, pop_size)
                rng = sel.0
                r3 = sel.1
            }

            let x1 = pop_get(pop, r1)
            let x2 = pop_get(pop, r2)
            let x3 = pop_get(pop, r3)

            var mutant: [f64; 20] = [0.0; 20]
            var j: i64 = 0
            while j < n_params {
                mutant[j as usize] = x1[j as usize] + config.F * (x2[j as usize] - x3[j as usize])
                if mutant[j as usize] < lower[j as usize] {
                    mutant[j as usize] = lower[j as usize]
                }
                if mutant[j as usize] > upper[j as usize] {
                    mutant[j as usize] = upper[j as usize]
                }
                j = j + 1
            }

            var trial: [f64; 20] = [0.0; 20]
            let jrand_sel = rng_int_range(rng, 0, n_params)
            rng = jrand_sel.0
            let jrand = jrand_sel.1

            j = 0
            while j < n_params {
                let r = rng_uniform(rng)
                rng = r.0
                let u = r.1

                if u < config.CR || j == jrand {
                    trial[j as usize] = mutant[j as usize]
                } else {
                    trial[j as usize] = target[j as usize]
                }
                j = j + 1
            }

            let fit_trial = rastrigin(trial, n_params)
            result.func_evals = result.func_evals + 1

            if fit_trial <= pop.fitness[i as usize] {
                pop = pop_set(pop, i, trial, fit_trial)
            }

            i = i + 1
        }
    }

    let best_idx = pop_find_best(pop)
    result.params = pop_get(pop, best_idx)
    result.value = pop.fitness[best_idx as usize]
    result.termination = termination
    result.converged = termination_success(termination)

    return result
}

// ============================================================================
// TESTS
// ============================================================================

fn test_de_sphere() -> bool {
    var lower: [f64; 20] = [-5.0; 20]
    var upper: [f64; 20] = [5.0; 20]

    var config = de_config_default()
    config.pop_size = 30
    config.max_generations = 200

    let result = de_sphere(lower, upper, 2, config)

    // Should find minimum near (0, 0)
    let x_err = if result.params[0] < 0.0 { -result.params[0] } else { result.params[0] }
    let y_err = if result.params[1] < 0.0 { -result.params[1] } else { result.params[1] }

    return x_err < 0.1 && y_err < 0.1
}

fn test_de_rastrigin() -> bool {
    var lower: [f64; 20] = [-5.12; 20]
    var upper: [f64; 20] = [5.12; 20]

    var config = de_config_default()
    config.pop_size = 50
    config.max_generations = 500

    let result = de_rastrigin(lower, upper, 2, config)

    // Rastrigin global min is at (0, 0) with value 0
    // Allow larger tolerance since it's a hard multi-modal function
    return result.value < 1.0
}

fn main() -> i32 {
    print("Testing optimize::differential_evolution module...\n")

    if !test_de_sphere() {
        print("FAIL: de_sphere\n")
        return 1
    }
    print("PASS: de_sphere\n")

    if !test_de_rastrigin() {
        print("FAIL: de_rastrigin\n")
        return 2
    }
    print("PASS: de_rastrigin\n")

    print("All optimize::differential_evolution tests PASSED\n")
    0
}
