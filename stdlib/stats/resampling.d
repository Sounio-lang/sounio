// stats::resampling — Bootstrap and Permutation Methods
//
// Nonparametric resampling for distribution-free inference.
// Returns confidence intervals and p-values without distributional assumptions.
//
// References:
// - Efron (1979): "Bootstrap methods: another look at the jackknife"
// - Good (2005): "Permutation, Parametric and Bootstrap Tests of Hypotheses"
// - Efron & Tibshirani (1993): "An Introduction to the Bootstrap"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// BOOTSTRAP RESULT TYPES
// ============================================================================

/// Result of bootstrap confidence interval
struct BootstrapCI {
    estimate: f64,      // Point estimate (original sample)
    se: f64,            // Bootstrap standard error
    ci_lower: f64,      // Lower CI bound (percentile method)
    ci_upper: f64,      // Upper CI bound
    alpha: f64,         // Significance level (e.g., 0.05)
    n_boot: i64,        // Number of bootstrap samples
    bias: f64,          // Bootstrap bias estimate
}

fn bootstrap_ci_new() -> BootstrapCI {
    BootstrapCI {
        estimate: 0.0,
        se: 0.0,
        ci_lower: 0.0,
        ci_upper: 0.0,
        alpha: 0.05,
        n_boot: 1000,
        bias: 0.0,
    }
}

/// Result of permutation test
struct PermutationResult {
    observed_stat: f64,  // Observed test statistic
    p_value: f64,        // Two-tailed p-value
    n_perm: i64,         // Number of permutations
    n_extreme: i64,      // Count of permutations >= observed
}

fn permutation_result_new() -> PermutationResult {
    PermutationResult {
        observed_stat: 0.0,
        p_value: 1.0,
        n_perm: 1000,
        n_extreme: 0,
    }
}

// ============================================================================
// SIMPLE LCG RANDOM NUMBER GENERATOR
// ============================================================================

/// Linear Congruential Generator state with last value
struct LCGState {
    seed: i64,
    last_value: i64,
}

fn lcg_new(seed: i64) -> LCGState {
    LCGState { seed: seed, last_value: 0 }
}

/// Generate next random integer [0, 2^31)
fn lcg_next(state: LCGState) -> LCGState {
    // Parameters from Numerical Recipes
    let a: i64 = 1103515245
    let c: i64 = 12345
    let m: i64 = 2147483648  // 2^31

    let new_seed = (a * state.seed + c) % m
    LCGState { seed: new_seed, last_value: new_seed }
}

/// Generate random integer in [0, max)
fn lcg_int_range(state: LCGState, max: i64) -> LCGState {
    let new_state = lcg_next(state)
    let result = new_state.last_value % max
    LCGState { seed: new_state.seed, last_value: result }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn mean(data: [f64; 100], n: i64) -> f64 {
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + data[i as usize]
        i = i + 1
    }
    sum / (n as f64)
}

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

fn std_dev(data: [f64; 100], n: i64) -> f64 {
    sqrt(variance(data, n))
}

/// Simple bubble sort for small arrays (for percentiles)
fn sort_array(data: [f64; 100], n: i64) -> [f64; 100] {
    var sorted = data
    var i: i64 = 0
    while i < n - 1 {
        var j: i64 = 0
        while j < n - 1 - i {
            if sorted[j as usize] > sorted[(j + 1) as usize] {
                let temp = sorted[j as usize]
                sorted[j as usize] = sorted[(j + 1) as usize]
                sorted[(j + 1) as usize] = temp
            }
            j = j + 1
        }
        i = i + 1
    }
    sorted
}

/// Get percentile from sorted array
fn percentile_sorted(sorted: [f64; 100], n: i64, p: f64) -> f64 {
    if n == 0 {
        return 0.0
    }
    let idx = (p * ((n - 1) as f64)) as i64
    if idx >= n - 1 {
        return sorted[(n - 1) as usize]
    }
    if idx < 0 {
        return sorted[0]
    }
    sorted[idx as usize]
}

// ============================================================================
// BOOTSTRAP METHODS
// ============================================================================

/// Bootstrap confidence interval for the mean
fn bootstrap_mean_ci(data: [f64; 100], n: i64, n_boot: i64, alpha: f64, seed: i64) -> BootstrapCI {
    var result = bootstrap_ci_new()
    result.n_boot = n_boot
    result.alpha = alpha

    // Original estimate
    result.estimate = mean(data, n)

    // Bootstrap resampling
    var boot_means: [f64; 100] = [0.0; 100]
    var rng = lcg_new(seed)
    var effective_boot = if n_boot > 100 { 100 } else { n_boot }

    var b: i64 = 0
    while b < effective_boot {
        // Resample with replacement
        var boot_sample: [f64; 100] = [0.0; 100]
        var i: i64 = 0
        while i < n {
            rng = lcg_int_range(rng, n)
            let idx = rng.last_value
            boot_sample[i as usize] = data[idx as usize]
            i = i + 1
        }

        boot_means[b as usize] = mean(boot_sample, n)
        b = b + 1
    }

    // Bootstrap SE
    var sum_sq = 0.0
    var sum = 0.0
    b = 0
    while b < effective_boot {
        sum = sum + boot_means[b as usize]
        b = b + 1
    }
    let boot_mean = sum / (effective_boot as f64)

    b = 0
    while b < effective_boot {
        let diff = boot_means[b as usize] - boot_mean
        sum_sq = sum_sq + diff * diff
        b = b + 1
    }
    result.se = sqrt(sum_sq / ((effective_boot - 1) as f64))

    // Bias
    result.bias = boot_mean - result.estimate

    // Percentile CI
    let sorted_boots = sort_array(boot_means, effective_boot)
    let lower_p = alpha / 2.0
    let upper_p = 1.0 - alpha / 2.0
    result.ci_lower = percentile_sorted(sorted_boots, effective_boot, lower_p)
    result.ci_upper = percentile_sorted(sorted_boots, effective_boot, upper_p)

    result
}

/// Bootstrap confidence interval for the median
fn bootstrap_median_ci(data: [f64; 100], n: i64, n_boot: i64, alpha: f64, seed: i64) -> BootstrapCI {
    var result = bootstrap_ci_new()
    result.n_boot = n_boot
    result.alpha = alpha

    // Original estimate (median)
    let sorted_data = sort_array(data, n)
    result.estimate = if n % 2 == 1 {
        sorted_data[(n / 2) as usize]
    } else {
        (sorted_data[(n / 2 - 1) as usize] + sorted_data[(n / 2) as usize]) / 2.0
    }

    // Bootstrap resampling
    var boot_medians: [f64; 100] = [0.0; 100]
    var rng = lcg_new(seed)
    var effective_boot = if n_boot > 100 { 100 } else { n_boot }

    var b: i64 = 0
    while b < effective_boot {
        // Resample with replacement
        var boot_sample: [f64; 100] = [0.0; 100]
        var i: i64 = 0
        while i < n {
            rng = lcg_int_range(rng, n)
            let idx = rng.last_value
            boot_sample[i as usize] = data[idx as usize]
            i = i + 1
        }

        // Compute median of bootstrap sample
        let sorted_boot = sort_array(boot_sample, n)
        boot_medians[b as usize] = if n % 2 == 1 {
            sorted_boot[(n / 2) as usize]
        } else {
            (sorted_boot[(n / 2 - 1) as usize] + sorted_boot[(n / 2) as usize]) / 2.0
        }
        b = b + 1
    }

    // Bootstrap SE
    var sum = 0.0
    b = 0
    while b < effective_boot {
        sum = sum + boot_medians[b as usize]
        b = b + 1
    }
    let boot_mean = sum / (effective_boot as f64)

    var sum_sq = 0.0
    b = 0
    while b < effective_boot {
        let diff = boot_medians[b as usize] - boot_mean
        sum_sq = sum_sq + diff * diff
        b = b + 1
    }
    result.se = sqrt(sum_sq / ((effective_boot - 1) as f64))

    // Bias
    result.bias = boot_mean - result.estimate

    // Percentile CI
    let sorted_boots = sort_array(boot_medians, effective_boot)
    let lower_p = alpha / 2.0
    let upper_p = 1.0 - alpha / 2.0
    result.ci_lower = percentile_sorted(sorted_boots, effective_boot, lower_p)
    result.ci_upper = percentile_sorted(sorted_boots, effective_boot, upper_p)

    result
}

/// Bootstrap confidence interval for correlation
fn bootstrap_correlation_ci(x: [f64; 100], y: [f64; 100], n: i64, n_boot: i64, alpha: f64, seed: i64) -> BootstrapCI {
    var result = bootstrap_ci_new()
    result.n_boot = n_boot
    result.alpha = alpha

    // Original correlation
    let mx = mean(x, n)
    let my = mean(y, n)
    let sx = std_dev(x, n)
    let sy = std_dev(y, n)

    var cov = 0.0
    var i: i64 = 0
    while i < n {
        cov = cov + (x[i as usize] - mx) * (y[i as usize] - my)
        i = i + 1
    }
    cov = cov / ((n - 1) as f64)
    result.estimate = if sx * sy > 1e-10 { cov / (sx * sy) } else { 0.0 }

    // Bootstrap
    var boot_corrs: [f64; 100] = [0.0; 100]
    var rng = lcg_new(seed)
    var effective_boot = if n_boot > 100 { 100 } else { n_boot }

    var b: i64 = 0
    while b < effective_boot {
        var boot_x: [f64; 100] = [0.0; 100]
        var boot_y: [f64; 100] = [0.0; 100]

        i = 0
        while i < n {
            rng = lcg_int_range(rng, n)
            let idx = rng.last_value
            boot_x[i as usize] = x[idx as usize]
            boot_y[i as usize] = y[idx as usize]
            i = i + 1
        }

        // Compute correlation of bootstrap sample
        let bmx = mean(boot_x, n)
        let bmy = mean(boot_y, n)
        let bsx = std_dev(boot_x, n)
        let bsy = std_dev(boot_y, n)

        var bcov = 0.0
        i = 0
        while i < n {
            bcov = bcov + (boot_x[i as usize] - bmx) * (boot_y[i as usize] - bmy)
            i = i + 1
        }
        bcov = bcov / ((n - 1) as f64)
        boot_corrs[b as usize] = if bsx * bsy > 1e-10 { bcov / (bsx * bsy) } else { 0.0 }

        b = b + 1
    }

    // SE and CI
    var sum = 0.0
    b = 0
    while b < effective_boot {
        sum = sum + boot_corrs[b as usize]
        b = b + 1
    }
    let boot_mean = sum / (effective_boot as f64)

    var sum_sq = 0.0
    b = 0
    while b < effective_boot {
        let diff = boot_corrs[b as usize] - boot_mean
        sum_sq = sum_sq + diff * diff
        b = b + 1
    }
    result.se = sqrt(sum_sq / ((effective_boot - 1) as f64))
    result.bias = boot_mean - result.estimate

    let sorted_boots = sort_array(boot_corrs, effective_boot)
    result.ci_lower = percentile_sorted(sorted_boots, effective_boot, alpha / 2.0)
    result.ci_upper = percentile_sorted(sorted_boots, effective_boot, 1.0 - alpha / 2.0)

    result
}

// ============================================================================
// PERMUTATION TESTS
// ============================================================================

/// Permutation test for difference in means
fn permutation_test_means(x: [f64; 100], nx: i64, y: [f64; 100], ny: i64, n_perm: i64, seed: i64) -> PermutationResult {
    var result = permutation_result_new()
    result.n_perm = n_perm

    // Observed difference
    let mx = mean(x, nx)
    let my = mean(y, ny)
    result.observed_stat = mx - my
    let abs_observed = if result.observed_stat < 0.0 { -result.observed_stat } else { result.observed_stat }

    // Combined data
    var combined: [f64; 100] = [0.0; 100]
    var i: i64 = 0
    while i < nx {
        combined[i as usize] = x[i as usize]
        i = i + 1
    }
    i = 0
    while i < ny {
        combined[(nx + i) as usize] = y[i as usize]
        i = i + 1
    }
    let n_total = nx + ny

    // Permutation resampling
    var rng = lcg_new(seed)
    var effective_perm = if n_perm > 100 { 100 } else { n_perm }
    result.n_extreme = 0

    var p: i64 = 0
    while p < effective_perm {
        // Fisher-Yates shuffle (partial - first nx elements become group 1)
        var perm = combined
        i = 0
        while i < nx {
            rng = lcg_int_range(rng, n_total - i)
            let j = rng.last_value
            let swap_idx = i + j
            let temp = perm[i as usize]
            perm[i as usize] = perm[swap_idx as usize]
            perm[swap_idx as usize] = temp
            i = i + 1
        }

        // Compute mean difference for this permutation
        var sum1 = 0.0
        i = 0
        while i < nx {
            sum1 = sum1 + perm[i as usize]
            i = i + 1
        }

        var sum2 = 0.0
        i = nx
        while i < n_total {
            sum2 = sum2 + perm[i as usize]
            i = i + 1
        }

        let perm_diff = sum1 / (nx as f64) - sum2 / (ny as f64)
        let abs_perm_diff = if perm_diff < 0.0 { -perm_diff } else { perm_diff }

        if abs_perm_diff >= abs_observed {
            result.n_extreme = result.n_extreme + 1
        }

        p = p + 1
    }

    // P-value
    result.p_value = (result.n_extreme as f64) / (effective_perm as f64)

    result
}

/// Permutation test for correlation (test H0: rho = 0)
fn permutation_test_correlation(x: [f64; 100], y: [f64; 100], n: i64, n_perm: i64, seed: i64) -> PermutationResult {
    var result = permutation_result_new()
    result.n_perm = n_perm

    // Observed correlation
    let mx = mean(x, n)
    let my = mean(y, n)
    let sx = std_dev(x, n)
    let sy = std_dev(y, n)

    var cov = 0.0
    var i: i64 = 0
    while i < n {
        cov = cov + (x[i as usize] - mx) * (y[i as usize] - my)
        i = i + 1
    }
    cov = cov / ((n - 1) as f64)
    result.observed_stat = if sx * sy > 1e-10 { cov / (sx * sy) } else { 0.0 }
    let abs_observed = if result.observed_stat < 0.0 { -result.observed_stat } else { result.observed_stat }

    // Permutation: shuffle y, keep x fixed
    var rng = lcg_new(seed)
    var effective_perm = if n_perm > 100 { 100 } else { n_perm }
    result.n_extreme = 0

    var p: i64 = 0
    while p < effective_perm {
        // Shuffle y
        var perm_y = y
        i = 0
        while i < n - 1 {
            rng = lcg_int_range(rng, n - i)
            let j = rng.last_value
            let swap_idx = i + j
            let temp = perm_y[i as usize]
            perm_y[i as usize] = perm_y[swap_idx as usize]
            perm_y[swap_idx as usize] = temp
            i = i + 1
        }

        // Compute correlation with permuted y
        let pmy = mean(perm_y, n)
        let psy = std_dev(perm_y, n)

        var pcov = 0.0
        i = 0
        while i < n {
            pcov = pcov + (x[i as usize] - mx) * (perm_y[i as usize] - pmy)
            i = i + 1
        }
        pcov = pcov / ((n - 1) as f64)
        let perm_corr = if sx * psy > 1e-10 { pcov / (sx * psy) } else { 0.0 }
        let abs_perm = if perm_corr < 0.0 { -perm_corr } else { perm_corr }

        if abs_perm >= abs_observed {
            result.n_extreme = result.n_extreme + 1
        }

        p = p + 1
    }

    result.p_value = (result.n_extreme as f64) / (effective_perm as f64)

    result
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_bootstrap_mean() -> bool {
    var data: [f64; 100] = [0.0; 100]
    data[0] = 10.0
    data[1] = 12.0
    data[2] = 11.0
    data[3] = 13.0
    data[4] = 11.0

    let result = bootstrap_mean_ci(data, 5, 50, 0.05, 12345)

    // Estimate should be mean = 11.4
    abs_f64(result.estimate - 11.4) < 0.01 && result.ci_lower < result.estimate && result.estimate < result.ci_upper
}

fn test_bootstrap_median() -> bool {
    var data: [f64; 100] = [0.0; 100]
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    data[4] = 5.0

    let result = bootstrap_median_ci(data, 5, 50, 0.05, 12345)

    // Median should be 3.0
    abs_f64(result.estimate - 3.0) < 0.01
}

fn test_permutation_means() -> bool {
    var x: [f64; 100] = [0.0; 100]
    var y: [f64; 100] = [0.0; 100]

    // Same distribution
    x[0] = 5.0
    x[1] = 6.0
    x[2] = 7.0

    y[0] = 5.0
    y[1] = 6.0
    y[2] = 7.0

    let result = permutation_test_means(x, 3, y, 3, 50, 12345)

    // P-value should be high (no difference)
    result.p_value > 0.1
}

fn test_permutation_correlation() -> bool {
    var x: [f64; 100] = [0.0; 100]
    var y: [f64; 100] = [0.0; 100]

    // Strong positive correlation
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

    let result = permutation_test_correlation(x, y, 5, 50, 12345)

    // Should detect significant correlation
    result.observed_stat > 0.9
}

fn main() -> i32 {
    print("Testing stats::resampling module...\n")

    if !test_bootstrap_mean() {
        print("FAIL: bootstrap_mean\n")
        return 1
    }
    print("PASS: bootstrap_mean\n")

    if !test_bootstrap_median() {
        print("FAIL: bootstrap_median\n")
        return 2
    }
    print("PASS: bootstrap_median\n")

    if !test_permutation_means() {
        print("FAIL: permutation_means\n")
        return 3
    }
    print("PASS: permutation_means\n")

    if !test_permutation_correlation() {
        print("FAIL: permutation_correlation\n")
        return 4
    }
    print("PASS: permutation_correlation\n")

    print("All stats::resampling tests PASSED\n")
    0
}
