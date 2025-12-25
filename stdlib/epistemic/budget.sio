//! Uncertainty Budget Ledger
//!
//! Scientists don't just want a final uncertainty — they want the decomposition.
//! This module implements GUM-style uncertainty budgets.
//!
//! References:
//!   - GUM (JCGM 100:2008) Section 5: Determining Combined Uncertainty

extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

// ============================================================================
// BUDGET ENTRY
// ============================================================================

struct BudgetEntry {
    source_type: i32,        // 0=Type A, 1=Type B
    estimate: f64,
    std_uncert: f64,
    sensitivity: f64,
    contribution: f64,       // (c_i * u_i)²
    dof: f64,
}

fn budget_entry_a(estimate: f64, std_uncert: f64, sensitivity: f64, dof: f64) -> BudgetEntry {
    let contrib = sensitivity * sensitivity * std_uncert * std_uncert
    return BudgetEntry {
        source_type: 0,
        estimate: estimate,
        std_uncert: std_uncert,
        sensitivity: sensitivity,
        contribution: contrib,
        dof: dof,
    }
}

fn budget_entry_b(estimate: f64, std_uncert: f64, sensitivity: f64) -> BudgetEntry {
    let contrib = sensitivity * sensitivity * std_uncert * std_uncert
    return BudgetEntry {
        source_type: 1,
        estimate: estimate,
        std_uncert: std_uncert,
        sensitivity: sensitivity,
        contribution: contrib,
        dof: 1.0e30,
    }
}

fn empty_entry() -> BudgetEntry {
    return BudgetEntry {
        source_type: 0,
        estimate: 0.0,
        std_uncert: 0.0,
        sensitivity: 0.0,
        contribution: 0.0,
        dof: 1.0,
    }
}

// ============================================================================
// UNCERTAINTY BUDGET (4 entries max for simplicity)
// ============================================================================

struct UncertaintyBudget {
    result_value: f64,
    entry_count: i32,
    e0: BudgetEntry,
    e1: BudgetEntry,
    e2: BudgetEntry,
    e3: BudgetEntry,
    combined_variance: f64,
    combined_std_uncert: f64,
    effective_dof: f64,
    confidence: f64,
}

fn budget_new(result_value: f64, confidence: f64) -> UncertaintyBudget {
    return UncertaintyBudget {
        result_value: result_value,
        entry_count: 0,
        e0: empty_entry(),
        e1: empty_entry(),
        e2: empty_entry(),
        e3: empty_entry(),
        combined_variance: 0.0,
        combined_std_uncert: 0.0,
        effective_dof: 0.0,
        confidence: confidence,
    }
}

fn get_entry(budget: UncertaintyBudget, idx: i32) -> BudgetEntry {
    if idx == 0 { return budget.e0 }
    if idx == 1 { return budget.e1 }
    if idx == 2 { return budget.e2 }
    if idx == 3 { return budget.e3 }
    return empty_entry()
}

fn add_entry(budget: UncertaintyBudget, entry: BudgetEntry) -> UncertaintyBudget {
    var result = budget
    let idx = budget.entry_count

    if idx == 0 { result.e0 = entry }
    else if idx == 1 { result.e1 = entry }
    else if idx == 2 { result.e2 = entry }
    else if idx == 3 { result.e3 = entry }

    if idx < 4 {
        result.entry_count = idx + 1
    }

    return result
}

// ============================================================================
// BUDGET COMPUTATION
// ============================================================================

fn compute_combined(budget: UncertaintyBudget) -> UncertaintyBudget {
    var result = budget
    var total_var: f64 = 0.0

    if budget.entry_count > 0 {
        total_var = total_var + budget.e0.contribution
    }
    if budget.entry_count > 1 {
        total_var = total_var + budget.e1.contribution
    }
    if budget.entry_count > 2 {
        total_var = total_var + budget.e2.contribution
    }
    if budget.entry_count > 3 {
        total_var = total_var + budget.e3.contribution
    }

    result.combined_variance = total_var
    result.combined_std_uncert = sqrt_f64(total_var)

    return result
}

fn compute_dof(budget: UncertaintyBudget) -> UncertaintyBudget {
    var result = budget
    let u_c4 = budget.combined_variance * budget.combined_variance
    var denom: f64 = 0.0

    if budget.entry_count > 0 && budget.e0.dof < 1.0e20 {
        let c4 = budget.e0.contribution * budget.e0.contribution
        denom = denom + c4 / budget.e0.dof
    }
    if budget.entry_count > 1 && budget.e1.dof < 1.0e20 {
        let c4 = budget.e1.contribution * budget.e1.contribution
        denom = denom + c4 / budget.e1.dof
    }
    if budget.entry_count > 2 && budget.e2.dof < 1.0e20 {
        let c4 = budget.e2.contribution * budget.e2.contribution
        denom = denom + c4 / budget.e2.dof
    }
    if budget.entry_count > 3 && budget.e3.dof < 1.0e20 {
        let c4 = budget.e3.contribution * budget.e3.contribution
        denom = denom + c4 / budget.e3.dof
    }

    if denom > 1.0e-15 {
        result.effective_dof = u_c4 / denom
    } else {
        result.effective_dof = 1.0e30
    }

    return result
}

fn finalize_budget(budget: UncertaintyBudget) -> UncertaintyBudget {
    var result = compute_combined(budget)
    result = compute_dof(result)
    return result
}

// ============================================================================
// BUDGET ANALYSIS
// ============================================================================

fn percent_contribution(budget: UncertaintyBudget, idx: i32) -> f64 {
    if budget.combined_variance < 1.0e-15 { return 0.0 }
    let entry = get_entry(budget, idx)
    return 100.0 * entry.contribution / budget.combined_variance
}

struct BudgetSummary {
    result_value: f64,
    combined_std_uncert: f64,
    relative_uncert_percent: f64,
    effective_dof: f64,
    entry_count: i32,
}

fn budget_summary(budget: UncertaintyBudget) -> BudgetSummary {
    var rel_uncert: f64 = 0.0
    if abs_f64(budget.result_value) > 1.0e-15 {
        rel_uncert = 100.0 * budget.combined_std_uncert / abs_f64(budget.result_value)
    }

    return BudgetSummary {
        result_value: budget.result_value,
        combined_std_uncert: budget.combined_std_uncert,
        relative_uncert_percent: rel_uncert,
        effective_dof: budget.effective_dof,
        entry_count: budget.entry_count,
    }
}

// ============================================================================
// EXPANDED UNCERTAINTY
// ============================================================================

fn coverage_factor(dof: f64, conf: f64) -> f64 {
    // Approximate t-values for 95% confidence
    if conf < 0.95 {
        if dof < 5.0 { return 2.13 }
        if dof < 10.0 { return 1.83 }
        return 1.65
    }
    // 95% confidence
    if dof < 3.0 { return 4.30 }
    if dof < 5.0 { return 2.78 }
    if dof < 10.0 { return 2.26 }
    if dof < 30.0 { return 2.04 }
    return 1.96
}

fn expanded_uncertainty(budget: UncertaintyBudget, conf: f64) -> f64 {
    let k = coverage_factor(budget.effective_dof, conf)
    return k * budget.combined_std_uncert
}

struct ConfInterval {
    lower: f64,
    upper: f64,
    coverage_k: f64,
}

fn conf_interval(budget: UncertaintyBudget, conf: f64) -> ConfInterval {
    let k = coverage_factor(budget.effective_dof, conf)
    let u_exp = k * budget.combined_std_uncert
    return ConfInterval {
        lower: budget.result_value - u_exp,
        upper: budget.result_value + u_exp,
        coverage_k: k,
    }
}

// ============================================================================
// BUILDER FUNCTIONS
// ============================================================================

fn add_type_a(budget: UncertaintyBudget, estimate: f64, std_uncert: f64, sensitivity: f64, n_samples: f64) -> UncertaintyBudget {
    let dof = n_samples - 1.0
    let entry = budget_entry_a(estimate, std_uncert, sensitivity, dof)
    return add_entry(budget, entry)
}

fn add_type_b(budget: UncertaintyBudget, estimate: f64, std_uncert: f64, sensitivity: f64) -> UncertaintyBudget {
    let entry = budget_entry_b(estimate, std_uncert, sensitivity)
    return add_entry(budget, entry)
}

// ============================================================================
// TESTS
// ============================================================================

fn test_simple_budget() -> bool {
    var budget = budget_new(100.0, 0.9)

    // Source 1: u=1.0, c=1.0 → contribution = 1.0
    budget = add_type_a(budget, 50.0, 1.0, 1.0, 10.0)

    // Source 2: u=2.0, c=1.0 → contribution = 4.0
    budget = add_type_a(budget, 50.0, 2.0, 1.0, 10.0)

    budget = finalize_budget(budget)

    // Combined variance = 1 + 4 = 5
    let diff = abs_f64(budget.combined_variance - 5.0)
    if diff > 0.01 { return false }

    // Combined std = sqrt(5) ≈ 2.236
    let diff2 = abs_f64(budget.combined_std_uncert - sqrt_f64(5.0))
    if diff2 > 0.01 { return false }

    return true
}

fn test_percent_contribution() -> bool {
    var budget = budget_new(100.0, 0.9)

    // Source 1: contribution = 1.0 (20%)
    budget = add_type_b(budget, 50.0, 1.0, 1.0)

    // Source 2: contribution = 4.0 (80%)
    budget = add_type_b(budget, 50.0, 2.0, 1.0)

    budget = finalize_budget(budget)

    let pct1 = percent_contribution(budget, 0)
    let pct2 = percent_contribution(budget, 1)

    if abs_f64(pct1 - 20.0) > 1.0 { return false }
    if abs_f64(pct2 - 80.0) > 1.0 { return false }

    return true
}

fn test_expanded_uncertainty() -> bool {
    var budget = budget_new(100.0, 0.95)
    budget = add_type_a(budget, 100.0, 2.0, 1.0, 10.0)
    budget = finalize_budget(budget)

    // Combined u_c = 2.0
    if abs_f64(budget.combined_std_uncert - 2.0) > 0.01 { return false }

    // Effective DOF = 9 (n-1)
    if abs_f64(budget.effective_dof - 9.0) > 0.1 { return false }

    // 95% confidence interval
    let ci = conf_interval(budget, 0.95)

    // k should be about 2.26 for ν=9
    if ci.coverage_k < 2.0 { return false }
    if ci.coverage_k > 2.5 { return false }

    return true
}

fn test_summary() -> bool {
    var budget = budget_new(100.0, 0.9)
    budget = add_type_b(budget, 100.0, 5.0, 1.0)
    budget = finalize_budget(budget)

    let summary = budget_summary(budget)

    if summary.entry_count != 1 { return false }
    if abs_f64(summary.combined_std_uncert - 5.0) > 0.01 { return false }
    if abs_f64(summary.relative_uncert_percent - 5.0) > 0.1 { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_simple_budget() { return 1 }
    if !test_percent_contribution() { return 2 }
    if !test_expanded_uncertainty() { return 3 }
    if !test_summary() { return 4 }

    return 0
}
