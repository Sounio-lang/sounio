// stats::multiple_testing — Multiple Comparison Corrections
//
// Control false discovery and family-wise error rates when testing
// multiple hypotheses simultaneously.
//
// References:
// - Bonferroni (1936): "Teoria statistica delle classi..."
// - Benjamini & Hochberg (1995): "Controlling the false discovery rate"
// - Holm (1979): "A simple sequentially rejective multiple test procedure"

extern "C" {
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// RESULT TYPES
// ============================================================================

/// Result of multiple testing correction
struct MTCResult {
    n_tests: i64,           // Number of tests
    n_significant: i64,     // Number significant after correction
    alpha: f64,             // Nominal significance level
    method: i64,            // 0=Bonferroni, 1=Holm, 2=FDR_BH
}

fn mtc_result_new() -> MTCResult {
    MTCResult {
        n_tests: 0,
        n_significant: 0,
        alpha: 0.05,
        method: 0,
    }
}

/// P-value with significance status
struct AdjustedPValue {
    original_p: f64,        // Original p-value
    adjusted_p: f64,        // Adjusted p-value
    is_significant: bool,   // After correction
    rank: i64,              // Rank (for ordering)
}

fn adjusted_p_new() -> AdjustedPValue {
    AdjustedPValue {
        original_p: 1.0,
        adjusted_p: 1.0,
        is_significant: false,
        rank: 0,
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Simple sort p-values (bubble sort for small n)
fn sort_indices_by_pvalue(pvals: [f64; 50], n: i64) -> [i64; 50] {
    var indices: [i64; 50] = [0; 50]
    var i: i64 = 0
    while i < n {
        indices[i as usize] = i
        i = i + 1
    }

    // Bubble sort by p-value
    i = 0
    while i < n - 1 {
        var j: i64 = 0
        while j < n - 1 - i {
            let idx_j = indices[j as usize]
            let idx_j1 = indices[(j + 1) as usize]
            if pvals[idx_j as usize] > pvals[idx_j1 as usize] {
                indices[j as usize] = idx_j1
                indices[(j + 1) as usize] = idx_j
            }
            j = j + 1
        }
        i = i + 1
    }

    indices
}

/// Compute inverse rank (for reverse ordering)
fn compute_ranks(sorted_indices: [i64; 50], n: i64) -> [i64; 50] {
    var ranks: [i64; 50] = [0; 50]
    var i: i64 = 0
    while i < n {
        let orig_idx = sorted_indices[i as usize]
        ranks[orig_idx as usize] = i + 1  // 1-based rank
        i = i + 1
    }
    ranks
}

// ============================================================================
// BONFERRONI CORRECTION
// ============================================================================

/// Bonferroni correction: p_adj = p * n
/// Most conservative, controls FWER
fn bonferroni_correct(pvals: [f64; 50], n: i64, alpha: f64) -> [f64; 50] {
    var adjusted: [f64; 50] = [1.0; 50]

    var i: i64 = 0
    while i < n {
        var adj_p = pvals[i as usize] * (n as f64)

        // Cap at 1.0
        if adj_p > 1.0 {
            adj_p = 1.0
        }

        adjusted[i as usize] = adj_p
        i = i + 1
    }

    adjusted
}

/// Count significant results after Bonferroni correction
fn bonferroni_count(pvals: [f64; 50], n: i64, alpha: f64) -> MTCResult {
    var result = mtc_result_new()
    result.n_tests = n
    result.alpha = alpha
    result.method = 0

    let adjusted = bonferroni_correct(pvals, n, alpha)
    var count: i64 = 0
    var i: i64 = 0
    while i < n {
        if adjusted[i as usize] < alpha {
            count = count + 1
        }
        i = i + 1
    }
    result.n_significant = count

    result
}

// ============================================================================
// HOLM-BONFERRONI (STEP-DOWN)
// ============================================================================

/// Holm step-down procedure
/// Less conservative than Bonferroni, still controls FWER
fn holm_correct(pvals: [f64; 50], n: i64, alpha: f64) -> [f64; 50] {
    var adjusted: [f64; 50] = [1.0; 50]

    // Sort p-values
    let sorted_indices = sort_indices_by_pvalue(pvals, n)

    // Step-down: adjusted p = p * (n - rank + 1), with running max
    var running_max = 0.0
    var i: i64 = 0
    while i < n {
        let sorted_idx = sorted_indices[i as usize]
        let p_val = pvals[sorted_idx as usize]

        // Adjusted p-value = p * (n - rank + 1)
        var adj_p = p_val * ((n - i) as f64)
        if adj_p > 1.0 {
            adj_p = 1.0
        }

        // Take running maximum to ensure monotonicity
        if adj_p < running_max {
            adj_p = running_max
        } else {
            running_max = adj_p
        }

        adjusted[sorted_idx as usize] = adj_p
        i = i + 1
    }

    adjusted
}

/// Count significant results after Holm correction
fn holm_count(pvals: [f64; 50], n: i64, alpha: f64) -> MTCResult {
    var result = mtc_result_new()
    result.n_tests = n
    result.alpha = alpha
    result.method = 1

    let adjusted = holm_correct(pvals, n, alpha)
    var count: i64 = 0
    var i: i64 = 0
    while i < n {
        if adjusted[i as usize] < alpha {
            count = count + 1
        }
        i = i + 1
    }
    result.n_significant = count

    result
}

// ============================================================================
// BENJAMINI-HOCHBERG FDR
// ============================================================================

/// Benjamini-Hochberg FDR correction
/// Controls expected proportion of false discoveries
fn fdr_bh_correct(pvals: [f64; 50], n: i64, alpha: f64) -> [f64; 50] {
    var adjusted: [f64; 50] = [1.0; 50]

    // Sort p-values
    let sorted_indices = sort_indices_by_pvalue(pvals, n)

    // Step-up: start from largest p-value
    var running_min = 1.0
    var i = n - 1
    while i >= 0 {
        let sorted_idx = sorted_indices[i as usize]
        let p_val = pvals[sorted_idx as usize]
        let rank = i + 1

        // q_i = p_i * n / rank
        var adj_p = p_val * (n as f64) / (rank as f64)
        if adj_p > 1.0 {
            adj_p = 1.0
        }

        // Take running minimum going backwards
        if adj_p < running_min {
            running_min = adj_p
        } else {
            adj_p = running_min
        }

        adjusted[sorted_idx as usize] = adj_p
        i = i - 1
    }

    adjusted
}

/// Count significant results after FDR correction
fn fdr_bh_count(pvals: [f64; 50], n: i64, alpha: f64) -> MTCResult {
    var result = mtc_result_new()
    result.n_tests = n
    result.alpha = alpha
    result.method = 2

    let adjusted = fdr_bh_correct(pvals, n, alpha)
    var count: i64 = 0
    var i: i64 = 0
    while i < n {
        if adjusted[i as usize] < alpha {
            count = count + 1
        }
        i = i + 1
    }
    result.n_significant = count

    result
}

// ============================================================================
// SIDAK CORRECTION
// ============================================================================

/// Sidak correction: p_adj = 1 - (1 - p)^n
/// Slightly less conservative than Bonferroni for independent tests
fn sidak_adjusted_p(p: f64, n: i64) -> f64 {
    if p >= 1.0 {
        return 1.0
    }
    if p <= 0.0 {
        return 0.0
    }

    // 1 - (1 - p)^n using log for numerical stability
    // = 1 - exp(n * log(1 - p))
    let log_1mp = if 1.0 - p > 0.0 {
        // Approximate log(1-p) for small p
        let x = 1.0 - p
        // Simple series: log(x) for x near 1
        -p - p * p / 2.0 - p * p * p / 3.0
    } else {
        -100.0  // Very negative for p ~ 1
    }

    // Use (n as f64) directly in multiplication
    let result = 1.0 - (log_1mp * (n as f64))
    // Clamp result
    if result < 0.0 { 0.0 } else if result > 1.0 { 1.0 } else { result }
}

// ============================================================================
// EFFECTIVE NUMBER OF TESTS
// ============================================================================

/// Estimate effective number of tests for correlated variables
/// Simplified: use eigenvalue approach approximation
fn effective_n_tests(correlation_matrix: [[f64; 10]; 10], n: i64) -> f64 {
    // Li & Ji (2005) approximation:
    // M_eff = sum(I(lambda_i >= 1)) + sum(lambda_i - floor(lambda_i)) for lambda_i < 1

    // For simplicity, estimate from sum of squared correlations
    var sum_r2 = 0.0
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            if i != j {
                let r = correlation_matrix[i as usize][j as usize]
                sum_r2 = sum_r2 + r * r
            }
            j = j + 1
        }
        i = i + 1
    }

    // Average r^2
    let n_pairs = (n * (n - 1)) as f64
    let avg_r2 = if n_pairs > 0.0 { sum_r2 / n_pairs } else { 0.0 }

    // Effective tests: n / (1 + avg_r2 * (n - 1))
    let nf = n as f64
    nf / (1.0 + avg_r2 * (nf - 1.0))
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_bonferroni() -> bool {
    var pvals: [f64; 50] = [1.0; 50]
    pvals[0] = 0.01
    pvals[1] = 0.03
    pvals[2] = 0.06

    let adjusted = bonferroni_correct(pvals, 3, 0.05)

    // p = 0.01 -> 0.01 * 3 = 0.03 (significant at alpha=0.05)
    // p = 0.03 -> 0.03 * 3 = 0.09 (not significant at alpha=0.05)
    let first_sig = adjusted[0] < 0.05
    let second_not_sig = adjusted[1] >= 0.05

    first_sig && second_not_sig
}

fn test_holm() -> bool {
    var pvals: [f64; 50] = [1.0; 50]
    pvals[0] = 0.01
    pvals[1] = 0.02
    pvals[2] = 0.03

    let result = holm_count(pvals, 3, 0.05)

    // Holm: sorted [0.01, 0.02, 0.03]
    // 0.01 < 0.05/3 = 0.0167 -> sig
    // 0.02 < 0.05/2 = 0.025 -> sig
    // 0.03 < 0.05/1 = 0.05 -> sig
    result.n_significant == 3
}

fn test_fdr_bh() -> bool {
    var pvals: [f64; 50] = [1.0; 50]
    pvals[0] = 0.01
    pvals[1] = 0.02
    pvals[2] = 0.50

    let result = fdr_bh_count(pvals, 3, 0.05)

    // BH: sorted [0.01, 0.02, 0.50]
    // rank 1: 0.01 * 3/1 = 0.03 < 0.05 -> sig
    // rank 2: 0.02 * 3/2 = 0.03 < 0.05 -> sig
    // rank 3: 0.50 * 3/3 = 0.50 >= 0.05 -> not sig
    result.n_significant == 2
}

fn test_fdr_vs_bonferroni() -> bool {
    // FDR should be less conservative
    var pvals: [f64; 50] = [1.0; 50]
    pvals[0] = 0.01
    pvals[1] = 0.03
    pvals[2] = 0.04
    pvals[3] = 0.20

    let bonf = bonferroni_count(pvals, 4, 0.05)
    let fdr = fdr_bh_count(pvals, 4, 0.05)

    // FDR should find >= as many significant as Bonferroni
    fdr.n_significant >= bonf.n_significant
}

fn main() -> i32 {
    print("Testing stats::multiple_testing module...\n")

    if !test_bonferroni() {
        print("FAIL: bonferroni\n")
        return 1
    }
    print("PASS: bonferroni\n")

    if !test_holm() {
        print("FAIL: holm\n")
        return 2
    }
    print("PASS: holm\n")

    if !test_fdr_bh() {
        print("FAIL: fdr_bh\n")
        return 3
    }
    print("PASS: fdr_bh\n")

    if !test_fdr_vs_bonferroni() {
        print("FAIL: fdr_vs_bonferroni\n")
        return 4
    }
    print("PASS: fdr_vs_bonferroni\n")

    print("All stats::multiple_testing tests PASSED\n")
    0
}
