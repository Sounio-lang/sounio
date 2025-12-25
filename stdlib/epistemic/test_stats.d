/// test_stats.d — Comprehensive tests for epistemic::stats
///
/// Tests all core functionality with numerical verification.

// Import the stats module
use stats

// Test tolerance
fn tol() -> f64 { return 0.0001 }

// ============================================================================
// TEST HELPERS
// ============================================================================

fn assert_near(actual: f64, expected: f64, name: [u8]) -> i64 {
    let diff = abs_f64(actual - expected)
    if diff < tol() {
        print("  PASS: ")
        println(name)
        return 1
    } else {
        print("  FAIL: ")
        print(name)
        print(" — expected ")
        print(expected)
        print(", got ")
        println(actual)
        return 0
    }
}

fn assert_true(condition: i64, name: [u8]) -> i64 {
    if condition > 0 {
        print("  PASS: ")
        println(name)
        return 1
    } else {
        print("  FAIL: ")
        println(name)
        return 0
    }
}

// ============================================================================
// SECTION 1: BETA CORE TESTS
// ============================================================================

fn test_beta_construction() -> i64 {
    println("=== Beta Construction Tests ===")
    var passed: i64 = 0

    // Test uniform prior
    let uniform = beta_uniform()
    passed = passed + assert_near(uniform.alpha, 1.0, "uniform alpha")
    passed = passed + assert_near(uniform.beta, 1.0, "uniform beta")

    // Test Jeffreys prior
    let jeffreys = beta_jeffreys()
    passed = passed + assert_near(jeffreys.alpha, 0.5, "jeffreys alpha")
    passed = passed + assert_near(jeffreys.beta, 0.5, "jeffreys beta")

    // Test informative prior
    let info = beta_informative(0.7, 10.0)
    passed = passed + assert_near(info.alpha, 7.0, "informative alpha")
    passed = passed + assert_near(info.beta, 3.0, "informative beta")

    println("")
    return passed
}

fn test_beta_moments() -> i64 {
    println("=== Beta Moments Tests ===")
    var passed: i64 = 0

    // Beta(8, 4) from uniform + 7 successes, 3 failures
    let b = beta_update(beta_uniform(), 7.0, 3.0)

    // Mean = 8/12 ≈ 0.6667
    let expected_mean = 8.0 / 12.0
    passed = passed + assert_near(beta_mean(b), expected_mean, "mean")

    // Variance = (8*4) / (12^2 * 13) = 32/1872 ≈ 0.0171
    let expected_var = 32.0 / 1872.0
    passed = passed + assert_near(beta_variance(b), expected_var, "variance")

    // Mode = (8-1)/(12-2) = 7/10 = 0.7
    let expected_mode = 7.0 / 10.0
    passed = passed + assert_near(beta_mode(b), expected_mode, "mode")

    // Sample size = 12
    passed = passed + assert_near(beta_sample_size(b), 12.0, "sample size")

    println("")
    return passed
}

fn test_beta_update() -> i64 {
    println("=== Bayesian Update Tests ===")
    var passed: i64 = 0

    let prior = beta_uniform()

    // Single update
    let post1 = beta_update(prior, 1.0, 0.0)
    passed = passed + assert_near(post1.alpha, 2.0, "single success alpha")
    passed = passed + assert_near(post1.beta, 1.0, "single success beta")

    // Multiple updates should be equivalent to batch
    let post_seq = beta_update(beta_update(prior, 3.0, 0.0), 2.0, 1.0)
    let post_batch = beta_update(prior, 5.0, 1.0)
    passed = passed + assert_near(post_seq.alpha, post_batch.alpha, "sequential=batch alpha")
    passed = passed + assert_near(post_seq.beta, post_batch.beta, "sequential=batch beta")

    println("")
    return passed
}

// ============================================================================
// SECTION 2: VARIANCE PROPAGATION TESTS
// ============================================================================

fn test_epistemic_value() -> i64 {
    println("=== Epistemic Value Tests ===")
    var passed: i64 = 0

    let b = beta_update(beta_uniform(), 7.0, 3.0)
    let e = epistemic_from_beta(b)

    passed = passed + assert_near(e.value, beta_mean(b), "value matches mean")
    passed = passed + assert_near(e.variance, beta_variance(b), "variance matches")
    passed = passed + assert_near(e.alpha, b.alpha, "alpha preserved")
    passed = passed + assert_near(e.beta, b.beta, "beta preserved")

    println("")
    return passed
}

fn test_epistemic_arithmetic() -> i64 {
    println("=== Epistemic Arithmetic Tests ===")
    var passed: i64 = 0

    let e1 = epistemic_point(0.6, 0.02)
    let e2 = epistemic_point(0.4, 0.01)

    // Scale
    let scaled = epistemic_scale(e1, 2.0)
    passed = passed + assert_near(scaled.value, 1.2, "scaled value")
    passed = passed + assert_near(scaled.variance, 0.08, "scaled variance")

    // Multiply
    let prod = epistemic_mul(e1, e2)
    passed = passed + assert_near(prod.value, 0.24, "product value")
    // Var(XY) ≈ 0.6²*0.01 + 0.4²*0.02 + 0.02*0.01
    let expected_prod_var = 0.36 * 0.01 + 0.16 * 0.02 + 0.02 * 0.01
    passed = passed + assert_near(prod.variance, expected_prod_var, "product variance")

    println("")
    return passed
}

// ============================================================================
// SECTION 3: ACTIVE INFERENCE TESTS
// ============================================================================

fn test_info_gain() -> i64 {
    println("=== Information Gain Tests ===")
    var passed: i64 = 0

    // Uniform prior: maximum ignorance, high info gain
    let uniform = beta_uniform()
    let ig_uniform = expected_info_gain(uniform)
    passed = passed + assert_true(if ig_uniform > 0.2 { 1 } else { 0 }, "uniform has high info gain")

    // Strong prior: low ignorance, low info gain
    let strong = beta_update(beta_uniform(), 49.0, 1.0)
    let ig_strong = expected_info_gain(strong)
    passed = passed + assert_true(if ig_strong < 0.02 { 1 } else { 0 }, "strong has low info gain")

    // Info gain decreases with evidence
    passed = passed + assert_true(if ig_uniform > ig_strong { 1 } else { 0 }, "info gain decreases with evidence")

    println("")
    return passed
}

fn test_variance_reduction() -> i64 {
    println("=== Variance Reduction Tests ===")
    var passed: i64 = 0

    let b = beta_uniform()
    let vr = expected_variance_reduction(b)

    // Variance reduction should be positive
    passed = passed + assert_true(if vr > 0.0 { 1 } else { 0 }, "variance reduction positive")

    // More observations = more reduction
    let vr1 = expected_variance_reduction_k(b, 1.0)
    let vr10 = expected_variance_reduction_k(b, 10.0)
    passed = passed + assert_true(if vr10 > vr1 { 1 } else { 0 }, "more obs = more reduction")

    println("")
    return passed
}

fn test_exploration_exploitation() -> i64 {
    println("=== Exploration-Exploitation Tests ===")
    var passed: i64 = 0

    // High variance = explore
    let uncertain = beta_uniform()
    let ee_uncertain = exploration_exploitation_score(uncertain, 0.7)
    passed = passed + assert_near(ee_uncertain, 0.0, "uncertain -> explore")

    // Low variance = exploit
    let certain = beta_update(beta_uniform(), 99.0, 1.0)
    let ee_certain = exploration_exploitation_score(certain, 0.7)
    passed = passed + assert_near(ee_certain, 1.0, "certain -> exploit")

    println("")
    return passed
}

// ============================================================================
// SECTION 4: ML INTEGRATION TESTS
// ============================================================================

fn test_variance_penalty() -> i64 {
    println("=== Variance Penalty Tests ===")
    var passed: i64 = 0

    let base_loss = 0.5
    let variance = 0.1
    let lambda = 0.5

    let penalty = variance_penalty_loss(base_loss, variance, lambda)

    passed = passed + assert_near(penalty.base_loss, 0.5, "base loss preserved")
    passed = passed + assert_near(penalty.variance_term, 0.05, "variance term correct")
    passed = passed + assert_near(penalty.total_loss, 0.55, "total loss correct")

    // Higher variance = higher penalty
    let high_var = variance_penalty_loss(base_loss, 0.2, lambda)
    passed = passed + assert_true(if high_var.total_loss > penalty.total_loss { 1 } else { 0 },
                                   "higher variance = higher loss")

    println("")
    return passed
}

fn test_curriculum_weight() -> i64 {
    println("=== Curriculum Weight Tests ===")
    var passed: i64 = 0

    // Low variance = lower weight
    let low_weight = curriculum_sample_weight(0.01, 1.0)
    // High variance = higher weight
    let high_weight = curriculum_sample_weight(0.2, 1.0)

    passed = passed + assert_true(if high_weight > low_weight { 1 } else { 0 },
                                   "high variance = higher weight")

    // Zero variance = base weight
    let base_weight = curriculum_sample_weight(0.0, 1.0)
    passed = passed + assert_near(base_weight, 1.0, "zero variance = base weight")

    println("")
    return passed
}

// ============================================================================
// SECTION 5: HIERARCHICAL COMBINATION TESTS
// ============================================================================

fn test_hierarchical_combine() -> i64 {
    println("=== Hierarchical Combination Tests ===")
    var passed: i64 = 0

    // Three similar distributions
    let d1 = beta_update(beta_uniform(), 7.0, 3.0)  // mean 0.67
    let d2 = beta_update(beta_uniform(), 6.0, 4.0)  // mean 0.60
    let d3 = beta_update(beta_uniform(), 8.0, 2.0)  // mean 0.73

    let combined = beta_hierarchical_combine([d1, d2, d3])

    // Combined mean should be between individual means
    let cm = beta_mean(combined)
    passed = passed + assert_true(if cm > 0.5 { 1 } else { 0 }, "combined mean > 0.5")
    passed = passed + assert_true(if cm < 0.8 { 1 } else { 0 }, "combined mean < 0.8")

    // Combined variance should be lower than average
    let cv = beta_variance(combined)
    let avg_var = (beta_variance(d1) + beta_variance(d2) + beta_variance(d3)) / 3.0
    passed = passed + assert_true(if cv < avg_var { 1 } else { 0 }, "combined variance < average")

    println("")
    return passed
}

fn test_precision_weighted() -> i64 {
    println("=== Precision-Weighted Combination Tests ===")
    var passed: i64 = 0

    // One precise, one imprecise
    let precise = beta_update(beta_uniform(), 99.0, 1.0)   // Low variance
    let imprecise = beta_update(beta_uniform(), 1.0, 1.0)  // High variance

    let combined = beta_precision_weighted([precise, imprecise])

    // Combined should be closer to precise estimate
    let cm = beta_mean(combined)
    let pm = beta_mean(precise)
    let im = beta_mean(imprecise)

    let diff_to_precise = abs_f64(cm - pm)
    let diff_to_imprecise = abs_f64(cm - im)

    passed = passed + assert_true(if diff_to_precise < diff_to_imprecise { 1 } else { 0 },
                                   "combined closer to precise")

    println("")
    return passed
}

// ============================================================================
// SECTION 6: INFORMATION THEORY TESTS
// ============================================================================

fn test_kl_divergence() -> i64 {
    println("=== KL Divergence Tests ===")
    var passed: i64 = 0

    let p = beta_update(beta_uniform(), 7.0, 3.0)
    let q = beta_update(beta_uniform(), 6.0, 4.0)

    // KL(p||p) = 0
    let kl_self = beta_kl_divergence(p, p)
    passed = passed + assert_near(kl_self, 0.0, "KL(p||p) = 0")

    // KL(p||q) >= 0
    let kl_pq = beta_kl_divergence(p, q)
    passed = passed + assert_true(if kl_pq >= 0.0 { 1 } else { 0 }, "KL(p||q) >= 0")

    // KL(p||q) != KL(q||p) in general
    let kl_qp = beta_kl_divergence(q, p)
    // They should be close but not necessarily equal
    passed = passed + assert_true(if abs_f64(kl_pq - kl_qp) < 0.1 { 1 } else { 0 },
                                   "KL is approximately symmetric for similar distributions")

    println("")
    return passed
}

// ============================================================================
// SECTION 7: NOVELTY TESTS
// ============================================================================

fn test_observations_for_variance() -> i64 {
    println("=== Observations for Target Variance Tests ===")
    var passed: i64 = 0

    let current = beta_uniform()  // Variance ≈ 0.083
    let current_var = beta_variance(current)

    // Target = current (need 0 more observations)
    let n0 = observations_for_target_variance(current, current_var)
    passed = passed + assert_near(n0, 0.0, "0 obs needed for current variance")

    // Target < current (need more observations)
    let n_low = observations_for_target_variance(current, 0.01)
    passed = passed + assert_true(if n_low > 0.0 { 1 } else { 0 }, "need obs for lower variance")

    println("")
    return passed
}

fn test_data_collection_priority() -> i64 {
    println("=== Data Collection Priority Tests ===")
    var passed: i64 = 0

    // High variance = high priority
    let uncertain = beta_uniform()
    let p_uncertain = data_collection_priority(uncertain, 1.0)

    // Low variance = low priority
    let certain = beta_update(beta_uniform(), 99.0, 1.0)
    let p_certain = data_collection_priority(certain, 1.0)

    passed = passed + assert_true(if p_uncertain > p_certain { 1 } else { 0 },
                                   "uncertain has higher priority")

    // Importance scales priority
    let p_important = data_collection_priority(uncertain, 2.0)
    passed = passed + assert_true(if p_important > p_uncertain { 1 } else { 0 },
                                   "importance scales priority")

    println("")
    return passed
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

fn main() -> i32 {
    println("╔════════════════════════════════════════════════════════════╗")
    println("║  epistemic::stats Test Suite                               ║")
    println("╚════════════════════════════════════════════════════════════╝")
    println("")

    var total_passed: i64 = 0

    // Section 1: Core Beta
    total_passed = total_passed + test_beta_construction()
    total_passed = total_passed + test_beta_moments()
    total_passed = total_passed + test_beta_update()

    // Section 2: Variance Propagation
    total_passed = total_passed + test_epistemic_value()
    total_passed = total_passed + test_epistemic_arithmetic()

    // Section 3: Active Inference
    total_passed = total_passed + test_info_gain()
    total_passed = total_passed + test_variance_reduction()
    total_passed = total_passed + test_exploration_exploitation()

    // Section 4: ML Integration
    total_passed = total_passed + test_variance_penalty()
    total_passed = total_passed + test_curriculum_weight()

    // Section 5: Hierarchical
    total_passed = total_passed + test_hierarchical_combine()
    total_passed = total_passed + test_precision_weighted()

    // Section 6: Information Theory
    total_passed = total_passed + test_kl_divergence()

    // Section 7: Novelty
    total_passed = total_passed + test_observations_for_variance()
    total_passed = total_passed + test_data_collection_priority()

    println("═══════════════════════════════════════════════════════════════")
    print("Total tests passed: ")
    println(total_passed)
    println("")

    if total_passed >= 30 {
        println("TEST SUITE PASSED")
        return 0
    } else {
        println("TEST SUITE FAILED — check individual tests above")
        return 1
    }
}
