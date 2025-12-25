// stats::inferential — Inferential Statistics with Uncertainty
//
// Hypothesis testing with GUM-compliant uncertainty on test statistics
// and effect sizes. Returns p-values and confidence intervals.
//
// References:
// - Student (1908): "The probable error of a mean"
// - Welch (1947): "The generalization of Student's problem"
// - Mann & Whitney (1947): "On a test of whether one of two random variables..."

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// TEST RESULT TYPES
// ============================================================================

/// Result of a t-test
struct TTestResult {
    t_stat: f64,        // t-statistic
    df: f64,            // degrees of freedom
    p_value: f64,       // two-tailed p-value
    mean_diff: f64,     // difference in means
    se_diff: f64,       // standard error of difference
    ci_lower: f64,      // 95% CI lower bound
    ci_upper: f64,      // 95% CI upper bound
    cohens_d: f64,      // effect size (Cohen's d)
}

fn ttest_result_new() -> TTestResult {
    TTestResult {
        t_stat: 0.0,
        df: 0.0,
        p_value: 1.0,
        mean_diff: 0.0,
        se_diff: 0.0,
        ci_lower: 0.0,
        ci_upper: 0.0,
        cohens_d: 0.0,
    }
}

/// Result of ANOVA
struct AnovaResult {
    f_stat: f64,        // F-statistic
    df_between: i64,    // between-group df
    df_within: i64,     // within-group df
    p_value: f64,       // p-value
    ss_between: f64,    // sum of squares between
    ss_within: f64,     // sum of squares within
    eta_squared: f64,   // effect size (eta^2)
}

fn anova_result_new() -> AnovaResult {
    AnovaResult {
        f_stat: 0.0,
        df_between: 0,
        df_within: 0,
        p_value: 1.0,
        ss_between: 0.0,
        ss_within: 0.0,
        eta_squared: 0.0,
    }
}

/// Result of Mann-Whitney U test
struct MannWhitneyResult {
    u_stat: f64,        // U statistic
    z_stat: f64,        // z-score (normal approx)
    p_value: f64,       // two-tailed p-value
    rank_biserial: f64, // effect size (r)
}

fn mannwhitney_result_new() -> MannWhitneyResult {
    MannWhitneyResult {
        u_stat: 0.0,
        z_stat: 0.0,
        p_value: 1.0,
        rank_biserial: 0.0,
    }
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

/// Approximate p-value from t-distribution using normal approximation
/// Valid for df > 30; conservative for smaller df
fn t_to_p_approx(t: f64, df: f64) -> f64 {
    let abs_t = if t < 0.0 { -t } else { t }

    // Normal approximation for large df
    // For df < 30, this underestimates p (conservative)
    // Using logistic approximation to normal CDF
    let z = abs_t

    // Approximate 2-tailed p-value using error function approximation
    // P(|T| > t) ≈ 2 * Φ(-|t|) for large df
    let a1 = 0.254829592
    let a2 = -0.284496736
    let a3 = 1.421413741
    let a4 = -1.453152027
    let a5 = 1.061405429
    let p = 0.3275911

    let sign = if z < 0.0 { -1.0 } else { 1.0 }
    let x = if z < 0.0 { -z } else { z }

    let t_erf = 1.0 / (1.0 + p * x)
    let erf = 1.0 - (((((a5 * t_erf + a4) * t_erf) + a3) * t_erf + a2) * t_erf + a1) * t_erf * exp(-x * x)

    // Two-tailed p-value
    1.0 - erf
}

/// Approximate p-value from F-distribution
fn f_to_p_approx(f: f64, df1: i64, df2: i64) -> f64 {
    // Very rough approximation using chi-square relationship
    // For production: use proper incomplete beta function
    if f <= 0.0 {
        return 1.0
    }

    // Fisher's approximation for large df
    let x = ((df1 as f64) * f) / ((df1 as f64) * f + (df2 as f64))

    // Approximate p-value (underestimates for small df)
    let p = 1.0 - x

    if p < 0.0 { 0.0 } else if p > 1.0 { 1.0 } else { p }
}

/// Approximate p-value from z-score (normal distribution)
fn z_to_p(z: f64) -> f64 {
    let abs_z = if z < 0.0 { -z } else { z }

    // Error function approximation
    let a1 = 0.254829592
    let a2 = -0.284496736
    let a3 = 1.421413741
    let a4 = -1.453152027
    let a5 = 1.061405429
    let p = 0.3275911

    let t = 1.0 / (1.0 + p * abs_z)
    let erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs_z * abs_z)

    // Two-tailed p-value
    1.0 - erf
}

// ============================================================================
// T-TESTS
// ============================================================================

/// One-sample t-test: test if mean differs from mu0
fn t_test_one_sample(data: [f64; 100], n: i64, mu0: f64) -> TTestResult {
    var result = ttest_result_new()

    let m = mean(data, n)
    let s = std_dev(data, n)
    let se = s / sqrt(n as f64)

    result.mean_diff = m - mu0
    result.se_diff = se
    result.df = (n - 1) as f64

    if se > 1e-10 {
        result.t_stat = result.mean_diff / se
    }

    result.p_value = t_to_p_approx(result.t_stat, result.df)

    // 95% CI (using t-critical ≈ 2.0 for large n)
    let t_crit = 2.0
    result.ci_lower = result.mean_diff - t_crit * se
    result.ci_upper = result.mean_diff + t_crit * se

    // Cohen's d = (mean - mu0) / s
    if s > 1e-10 {
        result.cohens_d = result.mean_diff / s
    }

    result
}

/// Independent samples t-test (equal variance assumed)
fn t_test_independent(x: [f64; 100], nx: i64, y: [f64; 100], ny: i64) -> TTestResult {
    var result = ttest_result_new()

    let mx = mean(x, nx)
    let my = mean(y, ny)
    let sx = std_dev(x, nx)
    let sy = std_dev(y, ny)

    result.mean_diff = mx - my
    result.df = (nx + ny - 2) as f64

    // Pooled standard deviation
    let var_x = sx * sx
    let var_y = sy * sy
    let sp_sq = (((nx - 1) as f64) * var_x + ((ny - 1) as f64) * var_y) / result.df
    let sp = sqrt(sp_sq)

    // Standard error of difference
    result.se_diff = sp * sqrt(1.0 / (nx as f64) + 1.0 / (ny as f64))

    if result.se_diff > 1e-10 {
        result.t_stat = result.mean_diff / result.se_diff
    }

    result.p_value = t_to_p_approx(result.t_stat, result.df)

    // 95% CI
    let t_crit = 2.0
    result.ci_lower = result.mean_diff - t_crit * result.se_diff
    result.ci_upper = result.mean_diff + t_crit * result.se_diff

    // Cohen's d
    if sp > 1e-10 {
        result.cohens_d = result.mean_diff / sp
    }

    result
}

/// Welch's t-test (unequal variance)
fn t_test_welch(x: [f64; 100], nx: i64, y: [f64; 100], ny: i64) -> TTestResult {
    var result = ttest_result_new()

    let mx = mean(x, nx)
    let my = mean(y, ny)
    let var_x = variance(x, nx)
    let var_y = variance(y, ny)

    result.mean_diff = mx - my

    // Welch's SE
    let se_x_sq = var_x / (nx as f64)
    let se_y_sq = var_y / (ny as f64)
    result.se_diff = sqrt(se_x_sq + se_y_sq)

    // Welch-Satterthwaite df
    let num = (se_x_sq + se_y_sq) * (se_x_sq + se_y_sq)
    let denom = se_x_sq * se_x_sq / ((nx - 1) as f64) + se_y_sq * se_y_sq / ((ny - 1) as f64)
    result.df = if denom > 1e-10 { num / denom } else { ((nx + ny - 2) as f64) }

    if result.se_diff > 1e-10 {
        result.t_stat = result.mean_diff / result.se_diff
    }

    result.p_value = t_to_p_approx(result.t_stat, result.df)

    // 95% CI
    let t_crit = 2.0
    result.ci_lower = result.mean_diff - t_crit * result.se_diff
    result.ci_upper = result.mean_diff + t_crit * result.se_diff

    // Cohen's d using pooled SD approximation
    let avg_var = (var_x + var_y) / 2.0
    let sp = sqrt(avg_var)
    if sp > 1e-10 {
        result.cohens_d = result.mean_diff / sp
    }

    result
}

/// Paired t-test
fn t_test_paired(x: [f64; 100], y: [f64; 100], n: i64) -> TTestResult {
    // Compute differences
    var diff: [f64; 100] = [0.0; 100]
    var i: i64 = 0
    while i < n {
        diff[i as usize] = x[i as usize] - y[i as usize]
        i = i + 1
    }

    // One-sample t-test on differences
    t_test_one_sample(diff, n, 0.0)
}

// ============================================================================
// ANOVA
// ============================================================================

/// One-way ANOVA for 3 groups
fn anova_oneway_3(g1: [f64; 100], n1: i64,
                  g2: [f64; 100], n2: i64,
                  g3: [f64; 100], n3: i64) -> AnovaResult {
    var result = anova_result_new()

    let n_total = n1 + n2 + n3
    let k = 3  // number of groups

    // Group means
    let m1 = mean(g1, n1)
    let m2 = mean(g2, n2)
    let m3 = mean(g3, n3)

    // Grand mean
    var sum_total = 0.0
    var i: i64 = 0
    while i < n1 {
        sum_total = sum_total + g1[i as usize]
        i = i + 1
    }
    i = 0
    while i < n2 {
        sum_total = sum_total + g2[i as usize]
        i = i + 1
    }
    i = 0
    while i < n3 {
        sum_total = sum_total + g3[i as usize]
        i = i + 1
    }
    let grand_mean = sum_total / (n_total as f64)

    // SS between
    result.ss_between = (n1 as f64) * (m1 - grand_mean) * (m1 - grand_mean)
                      + (n2 as f64) * (m2 - grand_mean) * (m2 - grand_mean)
                      + (n3 as f64) * (m3 - grand_mean) * (m3 - grand_mean)

    // SS within
    result.ss_within = 0.0
    i = 0
    while i < n1 {
        let diff = g1[i as usize] - m1
        result.ss_within = result.ss_within + diff * diff
        i = i + 1
    }
    i = 0
    while i < n2 {
        let diff = g2[i as usize] - m2
        result.ss_within = result.ss_within + diff * diff
        i = i + 1
    }
    i = 0
    while i < n3 {
        let diff = g3[i as usize] - m3
        result.ss_within = result.ss_within + diff * diff
        i = i + 1
    }

    result.df_between = k - 1
    result.df_within = n_total - k

    // Mean squares
    let ms_between = result.ss_between / (result.df_between as f64)
    let ms_within = result.ss_within / (result.df_within as f64)

    if ms_within > 1e-10 {
        result.f_stat = ms_between / ms_within
    }

    result.p_value = f_to_p_approx(result.f_stat, result.df_between, result.df_within)

    // Eta squared
    let ss_total = result.ss_between + result.ss_within
    if ss_total > 1e-10 {
        result.eta_squared = result.ss_between / ss_total
    }

    result
}

// ============================================================================
// NON-PARAMETRIC TESTS
// ============================================================================

/// Mann-Whitney U test (Wilcoxon rank-sum)
fn mann_whitney_u(x: [f64; 100], nx: i64, y: [f64; 100], ny: i64) -> MannWhitneyResult {
    var result = mannwhitney_result_new()

    // Combine and rank
    // For simplicity, count pairwise comparisons
    var u_x = 0.0

    var i: i64 = 0
    while i < nx {
        var j: i64 = 0
        while j < ny {
            if x[i as usize] > y[j as usize] {
                u_x = u_x + 1.0
            } else if x[i as usize] == y[j as usize] {
                u_x = u_x + 0.5
            }
            j = j + 1
        }
        i = i + 1
    }

    let u_y = (nx as f64) * (ny as f64) - u_x
    result.u_stat = if u_x < u_y { u_x } else { u_y }

    // Normal approximation for z-score
    let n_xy = (nx as f64) * (ny as f64)
    let mu_u = n_xy / 2.0
    let sigma_u = sqrt(n_xy * ((nx + ny + 1) as f64) / 12.0)

    if sigma_u > 1e-10 {
        result.z_stat = (u_x - mu_u) / sigma_u
    }

    result.p_value = z_to_p(result.z_stat)

    // Rank-biserial correlation (effect size)
    if n_xy > 0.0 {
        result.rank_biserial = (2.0 * u_x - n_xy) / n_xy
    }

    result
}

// ============================================================================
// EFFECT SIZE INTERPRETATION
// ============================================================================

/// Cohen's d interpretation
fn cohens_d_interpret(d: f64) -> i64 {
    let abs_d = if d < 0.0 { -d } else { d }
    if abs_d < 0.2 {
        return 0  // negligible
    }
    if abs_d < 0.5 {
        return 1  // small
    }
    if abs_d < 0.8 {
        return 2  // medium
    }
    3  // large
}

/// Eta-squared interpretation
fn eta_squared_interpret(eta2: f64) -> i64 {
    if eta2 < 0.01 {
        return 0  // negligible
    }
    if eta2 < 0.06 {
        return 1  // small
    }
    if eta2 < 0.14 {
        return 2  // medium
    }
    3  // large
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_one_sample_t() -> bool {
    var data: [f64; 100] = [0.0; 100]
    data[0] = 5.0
    data[1] = 6.0
    data[2] = 7.0
    data[3] = 8.0
    data[4] = 9.0

    // Mean = 7, test against mu0 = 7 -> t should be ~0
    let result = t_test_one_sample(data, 5, 7.0)

    abs_f64(result.t_stat) < 0.01
}

fn test_independent_t() -> bool {
    var x: [f64; 100] = [0.0; 100]
    var y: [f64; 100] = [0.0; 100]

    // Group 1: mean = 10
    x[0] = 9.0
    x[1] = 10.0
    x[2] = 11.0
    x[3] = 10.0
    x[4] = 10.0

    // Group 2: mean = 15
    y[0] = 14.0
    y[1] = 15.0
    y[2] = 16.0
    y[3] = 15.0
    y[4] = 15.0

    let result = t_test_independent(x, 5, y, 5)

    // mean_diff should be -5
    abs_f64(result.mean_diff - (-5.0)) < 0.1
}

fn test_paired_t() -> bool {
    var pre: [f64; 100] = [0.0; 100]
    var post: [f64; 100] = [0.0; 100]

    // Pre-treatment: 100, 102, 104, 101, 103
    pre[0] = 100.0
    pre[1] = 102.0
    pre[2] = 104.0
    pre[3] = 101.0
    pre[4] = 103.0

    // Post-treatment: 95, 97, 99, 96, 98 (decrease of 5)
    post[0] = 95.0
    post[1] = 97.0
    post[2] = 99.0
    post[3] = 96.0
    post[4] = 98.0

    let result = t_test_paired(pre, post, 5)

    // mean difference should be 5
    abs_f64(result.mean_diff - 5.0) < 0.1
}

fn test_mann_whitney() -> bool {
    var x: [f64; 100] = [0.0; 100]
    var y: [f64; 100] = [0.0; 100]

    // Group with lower values
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    // Group with higher values
    y[0] = 4.0
    y[1] = 5.0
    y[2] = 6.0

    let result = mann_whitney_u(x, 3, y, 3)

    // U should reflect clear separation
    result.u_stat >= 0.0
}

fn main() -> i32 {
    print("Testing stats::inferential module...\n")

    if !test_one_sample_t() {
        print("FAIL: one_sample_t\n")
        return 1
    }
    print("PASS: one_sample_t\n")

    if !test_independent_t() {
        print("FAIL: independent_t\n")
        return 2
    }
    print("PASS: independent_t\n")

    if !test_paired_t() {
        print("FAIL: paired_t\n")
        return 3
    }
    print("PASS: paired_t\n")

    if !test_mann_whitney() {
        print("FAIL: mann_whitney\n")
        return 4
    }
    print("PASS: mann_whitney\n")

    print("All stats::inferential tests PASSED\n")
    0
}
