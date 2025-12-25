// refutation.d - Causal Refutation & Sensitivity Analysis
//
// Every causal claim must survive refutation.
// Every effect estimate needs sensitivity bounds.
// Every conclusion requires robustness checks.
//
// Philosophy:
// Causal inference is fundamentally about ruling out alternatives.
// Identification assumptions are untestable but can be probed:
// - Placebo tests check for spurious effects
// - Bootstrap quantifies sampling uncertainty
// - Sensitivity analysis bounds unmeasured confounding
// - Refutation tests stress-test causal claims
//
// References:
// - Rosenbaum: "Observational Studies"
// - Cinelli & Hazlett: "Making Sense of Sensitivity"
// - Ding & VanderWeele: "Sensitivity Analysis Without Assumptions"

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
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
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    return y
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1000000.0 }
    // Use Taylor series for ln(1+y) where y = x - 1
    var y = x - 1.0
    if abs_f64(y) < 1.0 {
        var result = y
        var term = y
        var i: i64 = 2
        while i < 20 {
            term = 0.0 - term * y
            result = result + term / (i as f64)
            i = i + 1
        }
        return result
    }
    // For larger values, use Newton iteration
    var guess = 1.0
    var i: i64 = 0
    while i < 30 {
        let exp_guess = exp_f64(guess)
        guess = guess + (x - exp_guess) / exp_guess
        i = i + 1
    }
    return guess
}

fn exp_f64(x: f64) -> f64 {
    // Taylor series for exp
    var result = 1.0
    var term = 1.0
    var i: i64 = 1
    while i < 30 {
        term = term * x / (i as f64)
        result = result + term
        if abs_f64(term) < 0.000000000000001 { i = 30 }
        i = i + 1
    }
    return result
}

fn tanh_f64(x: f64) -> f64 {
    let e2x = exp_f64(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)
}

fn erf_f64(x: f64) -> f64 {
    let a1 = 0.254829592
    let a2 = 0.0 - 0.284496736
    let a3 = 1.421413741
    let a4 = 0.0 - 1.453152027
    let a5 = 1.061405429
    let p = 0.3275911

    var sign = 1.0
    if x < 0.0 { sign = 0.0 - 1.0 }
    let x_abs = abs_f64(x)
    let t = 1.0 / (1.0 + p * x_abs)
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp_f64(0.0 - x_abs * x_abs)
    return sign * y
}

fn normal_cdf(x: f64) -> f64 {
    return 0.5 * (1.0 + erf_f64(x / 1.4142135623730951))
}

fn min_i64(a: i64, b: i64) -> i64 {
    if a < b { return a }
    return b
}


// ============================================================================
// TREATMENT EFFECT TYPE
// ============================================================================

// Simple treatment effect estimate for refutation testing
struct ATEstimate {
    ate: f64,          // Average treatment effect
    ate_se: f64        // Standard error
}

fn ate_estimate_new(ate: f64, se: f64) -> ATEstimate {
    return ATEstimate { ate: ate, ate_se: se }
}

fn treatment_effect_t_stat(eff: ATEstimate) -> f64 {
    if eff.ate_se < 0.0000000001 { return 0.0 }
    return eff.ate / eff.ate_se
}

fn treatment_effect_ci_lower(eff: ATEstimate) -> f64 {
    return eff.ate - 1.96 * eff.ate_se
}

fn treatment_effect_ci_upper(eff: ATEstimate) -> f64 {
    return eff.ate + 1.96 * eff.ate_se
}

// ============================================================================
// REFUTATION RESULT TYPES
// ============================================================================

// Result of a single refutation test
struct RefutationResult {
    passed: i64,              // 1 if passed, 0 if failed
    original_eff: f64,
    refutation_eff: f64,
    p_value: f64,             // -1 if not applicable
    confidence_alpha: f64,
    confidence_beta: f64
}

fn refutation_result_new() -> RefutationResult {
    return RefutationResult {
        passed: 0,
        original_eff: 0.0,
        refutation_eff: 0.0,
        p_value: 0.0 - 1.0,
        confidence_alpha: 1.0,
        confidence_beta: 1.0
    }
}

fn refutation_result_is_robust(result: RefutationResult) -> i64 {
    let conf_mean = result.confidence_alpha / (result.confidence_alpha + result.confidence_beta)
    if result.passed > 0 && conf_mean > 0.8 { return 1 }
    return 0
}

fn refutation_result_effect_ratio(result: RefutationResult) -> f64 {
    if abs_f64(result.original_eff) < 0.0000000001 {
        return 1000000000.0
    }
    return result.refutation_eff / result.original_eff
}

// ============================================================================
// REFUTATION DATA
// ============================================================================

// Data container for refutation tests
struct RefutationData {
    treatment: [f64],
    outcome: [f64],
    n: i64
}

fn refutation_data_new(treatment: [f64], outcome: [f64]) -> RefutationData {
    let n_obs = len(treatment) as i64
    return RefutationData {
        treatment: treatment,
        outcome: outcome,
        n: n_obs
    }
}

// ============================================================================
// SIMPLE EFFECT ESTIMATOR
// ============================================================================

// Estimate ATE from data using difference in means
fn estimate_ate(data: RefutationData) -> ATEstimate {
    var t_sum = 0.0
    var c_sum = 0.0
    var t_n: i64 = 0
    var c_n: i64 = 0
    var t_sq = 0.0
    var c_sq = 0.0

    var i: i64 = 0
    while i < data.n {
        if data.treatment[i] > 0.5 {
            t_sum = t_sum + data.outcome[i]
            t_sq = t_sq + data.outcome[i] * data.outcome[i]
            t_n = t_n + 1
        } else {
            c_sum = c_sum + data.outcome[i]
            c_sq = c_sq + data.outcome[i] * data.outcome[i]
            c_n = c_n + 1
        }
        i = i + 1
    }

    var t_mean = 0.0
    if t_n > 0 { t_mean = t_sum / (t_n as f64) }

    var c_mean = 0.0
    if c_n > 0 { c_mean = c_sum / (c_n as f64) }

    let ate = t_mean - c_mean

    // Variance of difference in means
    var t_var = 0.0
    if t_n > 1 { t_var = (t_sq - t_sum * t_sum / (t_n as f64)) / ((t_n - 1) as f64) }

    var c_var = 0.0
    if c_n > 1 { c_var = (c_sq - c_sum * c_sum / (c_n as f64)) / ((c_n - 1) as f64) }

    // Avoid division by zero
    var se = 0.0
    if t_n > 0 && c_n > 0 {
        se = sqrt_f64(t_var / (t_n as f64) + c_var / (c_n as f64))
    }

    return ate_estimate_new(ate, se)
}

// ============================================================================
// PERMUTATION (FOR PLACEBO TEST)
// ============================================================================

// Permute treatment assignment using LCG random number generator
fn permute_treatment(data: RefutationData, seed: i64) -> RefutationData {
    var permuted: [f64] = []
    var i: i64 = 0
    while i < data.n {
        permuted = permuted ++ [data.treatment[i]]
        i = i + 1
    }

    var rng = seed
    i = data.n - 1
    while i > 0 {
        // LCG random number
        rng = (rng * 1103515245 + 12345) % 2147483648
        var j = rng % (i + 1)
        if j < 0 { j = 0 - j }

        // Swap permuted[i] and permuted[j]
        let tmp = permuted[i]
        var new_permuted: [f64] = []
        var k: i64 = 0
        while k < data.n {
            if k == i {
                new_permuted = new_permuted ++ [permuted[j]]
            } else if k == j {
                new_permuted = new_permuted ++ [tmp]
            } else {
                new_permuted = new_permuted ++ [permuted[k]]
            }
            k = k + 1
        }
        permuted = new_permuted
        i = i - 1
    }

    return RefutationData {
        treatment: permuted,
        outcome: data.outcome,
        n: data.n
    }
}

// ============================================================================
// BOOTSTRAP SAMPLE
// ============================================================================

fn bootstrap_sample(data: RefutationData, seed: i64) -> RefutationData {
    var new_treatment: [f64] = []
    var new_outcome: [f64] = []

    var rng = seed
    var i: i64 = 0
    while i < data.n {
        // Random index
        rng = (rng * 1103515245 + 12345) % 2147483648
        var idx = rng % data.n
        if idx < 0 { idx = 0 - idx }

        new_treatment = new_treatment ++ [data.treatment[idx]]
        new_outcome = new_outcome ++ [data.outcome[idx]]
        i = i + 1
    }

    return RefutationData {
        treatment: new_treatment,
        outcome: new_outcome,
        n: data.n
    }
}

// ============================================================================
// RANDOM SUBSET
// ============================================================================

fn random_subset(data: RefutationData, fraction: f64, seed: i64) -> RefutationData {
    let subset_n = (data.n as f64 * fraction) as i64

    // Generate indices and shuffle
    var indices: [i64] = []
    var i: i64 = 0
    while i < data.n {
        indices = indices ++ [i]
        i = i + 1
    }

    var rng = seed
    i = data.n - 1
    while i > 0 {
        rng = (rng * 1103515245 + 12345) % 2147483648
        var j = rng % (i + 1)
        if j < 0 { j = 0 - j }

        // Swap indices
        let tmp = indices[i]
        var new_indices: [i64] = []
        var k: i64 = 0
        while k < data.n {
            if k == i {
                new_indices = new_indices ++ [indices[j]]
            } else if k == j {
                new_indices = new_indices ++ [tmp]
            } else {
                new_indices = new_indices ++ [indices[k]]
            }
            k = k + 1
        }
        indices = new_indices
        i = i - 1
    }

    // Take first subset_n indices
    var new_treatment: [f64] = []
    var new_outcome: [f64] = []
    i = 0
    while i < subset_n {
        let idx = indices[i]
        new_treatment = new_treatment ++ [data.treatment[idx]]
        new_outcome = new_outcome ++ [data.outcome[idx]]
        i = i + 1
    }

    return RefutationData {
        treatment: new_treatment,
        outcome: new_outcome,
        n: subset_n
    }
}

// ============================================================================
// SORT FLOATS (INSERTION SORT)
// ============================================================================

fn sort_floats(arr: [f64]) -> [f64] {
    var result: [f64] = []
    var i: i64 = 0
    while i < len(arr) {
        result = result ++ [arr[i]]
        i = i + 1
    }

    let n = len(result)
    i = 1
    while i < n {
        let key = result[i]
        var j = i - 1

        // Find insertion point
        while j >= 0 && result[j] > key {
            j = j - 1
        }

        // Rebuild array with key inserted
        var new_result: [f64] = []
        var k: i64 = 0
        while k < n {
            if k <= j || k > i {
                new_result = new_result ++ [result[k]]
            } else if k == j + 1 {
                new_result = new_result ++ [key]
            } else {
                new_result = new_result ++ [result[k - 1]]
            }
            k = k + 1
        }
        result = new_result
        i = i + 1
    }

    return result
}

// ============================================================================
// PLACEBO TREATMENT TEST
// ============================================================================

// Placebo treatment test (permutation test)
// Randomly permutes treatment assignment and re-estimates effect.
// Original effect should be larger than placebo effects.
fn placebo_treatment_test(
    data: RefutationData,
    eff: ATEstimate,
    n_permutations: i64,
    alpha: f64
) -> RefutationResult {
    let original_ate = eff.ate
    var placebo_effects: [f64] = []

    var perm: i64 = 0
    while perm < n_permutations {
        let permuted = permute_treatment(data, perm * 12345 + 1)
        let placebo_effect = estimate_ate(permuted)
        placebo_effects = placebo_effects ++ [placebo_effect.ate]
        perm = perm + 1
    }

    // Compute mean of placebo effects
    var sum = 0.0
    var i: i64 = 0
    while i < n_permutations {
        sum = sum + placebo_effects[i]
        i = i + 1
    }
    let placebo_mean = sum / (n_permutations as f64)

    // Count how many placebo effects are at least as extreme
    var n_extreme: i64 = 0
    i = 0
    while i < n_permutations {
        if abs_f64(placebo_effects[i]) >= abs_f64(original_ate) {
            n_extreme = n_extreme + 1
        }
        i = i + 1
    }
    let p_value = (n_extreme as f64) / (n_permutations as f64)

    var passed: i64 = 0
    if p_value < alpha { passed = 1 }

    var conf_alpha = 2.0
    var conf_beta = 8.0 + p_value * 10.0
    if passed > 0 {
        conf_alpha = 8.0 + (1.0 - p_value) * 10.0
        conf_beta = 2.0
    }

    return RefutationResult {
        passed: passed,
        original_eff: original_ate,
        refutation_eff: placebo_mean,
        p_value: p_value,
        confidence_alpha: conf_alpha,
        confidence_beta: conf_beta
    }
}

// ============================================================================
// BOOTSTRAP CI TEST
// ============================================================================

// Bootstrap confidence interval test
// Resamples data with replacement to estimate CI.
fn bootstrap_ci_test(
    data: RefutationData,
    eff: ATEstimate,
    n_bootstrap: i64,
    ci_level: f64
) -> RefutationResult {
    var effects: [f64] = []

    var b: i64 = 0
    while b < n_bootstrap {
        let boot_data = bootstrap_sample(data, b * 54321 + 1)
        let boot_effect = estimate_ate(boot_data)
        effects = effects ++ [boot_effect.ate]
        b = b + 1
    }

    // Sort effects for percentile CI
    effects = sort_floats(effects)

    let alpha_ci = 1.0 - ci_level
    let lower_idx = ((alpha_ci / 2.0) * (n_bootstrap as f64)) as i64
    var upper_idx = (((1.0 - alpha_ci / 2.0) * (n_bootstrap as f64)) as i64)
    upper_idx = min_i64(upper_idx, n_bootstrap - 1)

    let ci_lower = effects[lower_idx]
    let ci_upper = effects[upper_idx]

    let original_ate = eff.ate
    var excludes_zero: i64 = 0
    if ci_lower > 0.0 || ci_upper < 0.0 { excludes_zero = 1 }

    var original_in_ci: i64 = 0
    if original_ate >= ci_lower && original_ate <= ci_upper { original_in_ci = 1 }

    var conf_alpha = 5.0
    var conf_beta = 5.0
    if excludes_zero > 0 {
        conf_alpha = 8.0 + ci_level * 10.0
        conf_beta = 2.0
    }

    return RefutationResult {
        passed: original_in_ci,
        original_eff: original_ate,
        refutation_eff: (ci_lower + ci_upper) / 2.0,
        p_value: 0.0 - 1.0,  // Not applicable
        confidence_alpha: conf_alpha,
        confidence_beta: conf_beta
    }
}

// ============================================================================
// SUBSET REFUTATION TEST
// ============================================================================

// Random subset refutation test
// Estimates effect on random subsets to check stability.
fn subset_refutation_test(
    data: RefutationData,
    eff: ATEstimate,
    n_subsets: i64,
    subset_fraction: f64
) -> RefutationResult {
    var effects: [f64] = []

    var s: i64 = 0
    while s < n_subsets {
        let subset = random_subset(data, subset_fraction, s * 13579 + 1)
        let subset_effect = estimate_ate(subset)
        effects = effects ++ [subset_effect.ate]
        s = s + 1
    }

    // Compute mean and std
    var sum = 0.0
    var i: i64 = 0
    while i < n_subsets {
        sum = sum + effects[i]
        i = i + 1
    }
    let mean = sum / (n_subsets as f64)

    var var_sum = 0.0
    i = 0
    while i < n_subsets {
        let diff = effects[i] - mean
        var_sum = var_sum + diff * diff
        i = i + 1
    }
    let std = sqrt_f64(var_sum / ((n_subsets - 1) as f64))

    let original_ate = eff.ate
    var passed: i64 = 0
    if abs_f64(original_ate - mean) < 2.0 * std { passed = 1 }

    var conf_alpha = 3.0
    var conf_beta = 7.0
    if passed > 0 {
        conf_alpha = 9.0
        conf_beta = 1.0
    }

    return RefutationResult {
        passed: passed,
        original_eff: original_ate,
        refutation_eff: mean,
        p_value: 0.0 - 1.0,
        confidence_alpha: conf_alpha,
        confidence_beta: conf_beta
    }
}

// ============================================================================
// SENSITIVITY ANALYSIS: ROSENBAUM BOUNDS
// ============================================================================

// Sensitivity analysis result
struct SensitivityResult {
    gamma_values: [f64],
    upper_bounds: [f64],
    lower_bounds: [f64],
    p_values: [f64],
    critical_gamma: f64,   // -1 if not found
    robust: i64            // 1 if robust to all tested gamma
}

// Rosenbaum bounds for sensitivity to unmeasured confounding
// Computes how much hidden bias (Gamma) would be needed to
// explain away the observed effect.
fn rosenbaum_bounds(
    eff: ATEstimate,
    gamma_max: f64,
    gamma_steps: i64
) -> SensitivityResult {
    var gamma_values: [f64] = []
    var upper_bounds: [f64] = []
    var lower_bounds: [f64] = []
    var p_values: [f64] = []

    let ate = eff.ate
    let se = eff.ate_se

    var i: i64 = 0
    while i < gamma_steps {
        let gamma = 1.0 + (gamma_max - 1.0) * ((i as f64) / ((gamma_steps - 1) as f64))
        gamma_values = gamma_values ++ [gamma]

        // Bias adjustment based on gamma
        let bias = tanh_f64(ln_f64(gamma) / 2.0)
        upper_bounds = upper_bounds ++ [ate + bias * se * 2.0]
        lower_bounds = lower_bounds ++ [ate - bias * se * 2.0]

        // Adjusted p-value
        let z = abs_f64(ate) / (se * sqrt_f64(gamma))
        let p = 2.0 * (1.0 - normal_cdf(z))
        p_values = p_values ++ [p]

        i = i + 1
    }

    // Find critical gamma where p > 0.05
    var critical_gamma = 0.0 - 1.0
    var robust: i64 = 1
    i = 0
    while i < gamma_steps {
        if p_values[i] > 0.05 {
            critical_gamma = gamma_values[i]
            robust = 0
            i = gamma_steps  // break
        }
        i = i + 1
    }

    return SensitivityResult {
        gamma_values: gamma_values,
        upper_bounds: upper_bounds,
        lower_bounds: lower_bounds,
        p_values: p_values,
        critical_gamma: critical_gamma,
        robust: robust
    }
}

// ============================================================================
// OMITTED VARIABLE BIAS ANALYSIS
// ============================================================================

// Omitted variable bias analysis result
struct OVBResult {
    robustness_value: f64,
    original_eff: f64,
    original_se: f64
}

// Compute omitted variable bias analysis
// Estimates robustness value: how much R² would a confounder need
// with both treatment and outcome to explain away the effect.
fn omitted_variable_bias(
    data: RefutationData,
    eff: ATEstimate
) -> OVBResult {
    let ate = eff.ate
    let se = eff.ate_se
    let t_stat = ate / se
    let n = data.n as f64

    // Robustness value (partial R² needed to nullify effect)
    let rv = t_stat * t_stat / (t_stat * t_stat + n - 2.0)

    return OVBResult {
        robustness_value: rv,
        original_eff: ate,
        original_se: se
    }
}

// ============================================================================
// REFUTATION SUITE
// ============================================================================

// Suite of refutation test results
struct RefutationSuite {
    placebo_result: RefutationResult,
    bootstrap_result: RefutationResult,
    subset_result: RefutationResult,
    n_passed: i64,
    n_total: i64,
    overall_robustness: f64
}

// Run full refutation suite with default settings
fn run_refutation_suite(
    data: RefutationData,
    eff: ATEstimate
) -> RefutationSuite {

    // Placebo treatment test
    let placebo = placebo_treatment_test(data, eff, 100, 0.05)

    // Bootstrap CI test
    let bootstrap = bootstrap_ci_test(data, eff, 500, 0.95)

    // Subset refutation test
    let subset = subset_refutation_test(data, eff, 100, 0.8)

    // Count passed tests
    var n_passed: i64 = 0
    if placebo.passed > 0 { n_passed = n_passed + 1 }
    if bootstrap.passed > 0 { n_passed = n_passed + 1 }
    if subset.passed > 0 { n_passed = n_passed + 1 }

    let overall = (n_passed as f64) / 3.0

    return RefutationSuite {
        placebo_result: placebo,
        bootstrap_result: bootstrap,
        subset_result: subset,
        n_passed: n_passed,
        n_total: 3,
        overall_robustness: overall
    }
}

fn refutation_suite_is_robust(suite: RefutationSuite) -> i64 {
    if suite.n_passed == suite.n_total && suite.overall_robustness > 0.8 {
        return 1
    }
    return 0
}

// ============================================================================
// PRINTING UTILITIES
// ============================================================================

fn print_refutation_result(name: [u8], result: RefutationResult) -> i64 {
    print("  ")
    print(name)
    print(": ")
    if result.passed > 0 { println("PASSED") } else { println("FAILED") }

    print("    Original: ")
    println(result.original_eff)

    print("    Refutation: ")
    println(result.refutation_eff)

    if result.p_value >= 0.0 {
        print("    p-value: ")
        println(result.p_value)
    }

    return 0
}

fn print_refutation_suite(suite: RefutationSuite) -> i64 {
    println("=== Refutation Suite Results ===")
    print("Tests passed: ")
    print(suite.n_passed)
    print("/")
    println(suite.n_total)

    print("Overall robustness: ")
    print(suite.overall_robustness * 100.0)
    println("%")

    println("")
    println("Individual results:")
    print_refutation_result("Placebo Treatment", suite.placebo_result)
    print_refutation_result("Bootstrap CI", suite.bootstrap_result)
    print_refutation_result("Random Subset", suite.subset_result)

    return 0
}

fn print_sensitivity_result(result: SensitivityResult) -> i64 {
    println("=== Sensitivity Analysis (Rosenbaum Bounds) ===")

    println("Gamma values and p-values:")
    var i: i64 = 0
    while i < len(result.gamma_values) {
        print("  Gamma=")
        print(result.gamma_values[i])
        print(": p=")
        println(result.p_values[i])
        i = i + 1
    }

    if result.critical_gamma >= 0.0 {
        print("Critical Gamma: ")
        println(result.critical_gamma)
        println("Effect is sensitive to hidden bias")
    } else {
        println("Effect robust to all tested Gamma values")
    }

    return 0
}

// ============================================================================
// DEMONSTRATION
// ============================================================================

fn main() -> i32 {
    println("=== epistemic::refutation — Causal Refutation Demo ===")
    println("")

    // Create synthetic data
    let n: i64 = 100
    var treatment: [f64] = []
    var outcome: [f64] = []

    var i: i64 = 0
    var seed: i64 = 42
    while i < n {
        // Generate treatment (0 or 1) - assign treatment to roughly half
        var t = 0.0
        if i < n / 2 { t = 1.0 }
        treatment = treatment ++ [t]

        // Generate outcome with treatment effect
        seed = (seed * 1103515245 + 12345) % 2147483648
        var noise_raw = seed % 1000
        if noise_raw < 0 { noise_raw = 0 - noise_raw }
        let noise = (noise_raw as f64) / 1000.0 - 0.5
        let y = 2.0 + 0.5 * t + noise  // True ATE = 0.5
        outcome = outcome ++ [y]

        i = i + 1
    }

    // Create refutation data
    let data = refutation_data_new(treatment, outcome)

    // Estimate ATE
    let est = estimate_ate(data)
    println("--- Estimated Causal ATE ---")
    print("ATE: ")
    println(est.ate)
    print("SE: ")
    println(est.ate_se)
    println("")

    // Run refutation suite
    println("--- Running Refutation Suite ---")
    let suite = run_refutation_suite(data, est)
    print_refutation_suite(suite)
    println("")

    // Sensitivity analysis
    println("--- Sensitivity Analysis ---")
    let sensitivity = rosenbaum_bounds(est, 2.0, 5)
    print_sensitivity_result(sensitivity)
    println("")

    // Omitted variable bias
    println("--- Omitted Variable Bias ---")
    let ovb = omitted_variable_bias(data, est)
    print("Robustness value: ")
    println(ovb.robustness_value)
    println("")

    println("=== Demo Complete ===")

    return 0
}
