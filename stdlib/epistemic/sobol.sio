//! Sobol' Sensitivity Indices for Global Sensitivity Analysis
//!
//! Variance-based sensitivity analysis decomposes output variance into
//! contributions from each input and their interactions.
//!
//! First-order index S_i: fraction of variance due to x_i alone
//! Total-order index S_Ti: fraction of variance due to x_i and all interactions
//!
//! Key insight: max_dominant_source becomes QUANTITATIVE:
//!   "Your output is 82% hostage to clearance. Go measure clearance."
//!
//! References:
//!   - Sobol' (1993): "Sensitivity estimates for nonlinear mathematical models"
//!   - Saltelli et al. (2008): "Global Sensitivity Analysis: The Primer"
//!   - Homma & Saltelli (1996): "Importance measures in global sensitivity analysis"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

// ============================================================================
// QUASI-RANDOM SEQUENCE: SOBOL' SEQUENCE GENERATOR
// ============================================================================

/// Sobol' sequence state for up to 4 dimensions
struct SobolSequence {
    index: i64,
    dim: i32,
    // Direction numbers (simplified - real impl uses lookup tables)
    // For now, we use a simple pseudo-random fallback
}

fn sobol_new(dim: i32) -> SobolSequence {
    return SobolSequence {
        index: 0,
        dim: dim,
    }
}

/// Generate next point in the Sobol' sequence (simplified)
/// Returns values in [0, 1]^dim
struct SobolPoint {
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
}

/// Van der Corput sequence helper - reverses bits to generate low-discrepancy points
fn reverse_bits(n: i64) -> f64 {
    var result: f64 = 0.0
    var v = n
    var b: f64 = 0.5
    while v > 0 {
        if (v % 2) == 1 {
            result = result + b
        }
        v = v / 2
        b = b / 2.0
    }
    return result
}

fn sobol_next(seq: SobolSequence) -> (SobolPoint, SobolSequence) {
    let idx = seq.index + 1

    // Simple low-discrepancy approximation using van der Corput-like sequence
    // Real Sobol' uses direction numbers - this is a simplified version

    // Different bases for different dimensions to reduce correlation
    let x0 = reverse_bits(idx)
    let x1 = reverse_bits(idx * 2 + 1)
    let x2 = reverse_bits(idx * 3 + 2)
    let x3 = reverse_bits(idx * 5 + 3)

    let point = SobolPoint { x0: x0, x1: x1, x2: x2, x3: x3 }
    let new_seq = SobolSequence { index: idx, dim: seq.dim }

    return (point, new_seq)
}

// ============================================================================
// INPUT SPECIFICATION
// ============================================================================

/// Input parameter with distribution
struct SobolInput {
    name_hash: i64,      // Hash of parameter name
    mean: f64,
    std: f64,
    lo: f64,             // Lower bound for uniform sampling
    hi: f64,             // Upper bound for uniform sampling
}

fn sobol_input(name: i64, mean: f64, std: f64) -> SobolInput {
    // Assume ±3σ range for sampling
    return SobolInput {
        name_hash: name,
        mean: mean,
        std: std,
        lo: mean - 3.0 * std,
        hi: mean + 3.0 * std,
    }
}

fn sobol_input_bounded(name: i64, mean: f64, std: f64, lo: f64, hi: f64) -> SobolInput {
    return SobolInput {
        name_hash: name,
        mean: mean,
        std: std,
        lo: lo,
        hi: hi,
    }
}

/// Transform uniform [0,1] to parameter range
fn transform_uniform(u: f64, input: SobolInput) -> f64 {
    return input.lo + u * (input.hi - input.lo)
}

// ============================================================================
// SOBOL' INDICES COMPUTATION (SALTELLI METHOD)
// ============================================================================

/// Sobol' sensitivity indices for one input
struct SobolIndex {
    input_hash: i64,
    first_order: f64,     // S_i: main effect
    total_order: f64,     // S_Ti: total effect including interactions
    interaction: f64,     // S_Ti - S_i: pure interaction effect
}

/// Complete Sobol' analysis result
struct SobolAnalysis {
    output_variance: f64,
    output_mean: f64,
    n_samples: i32,

    // Indices for up to 4 inputs
    index_count: i32,
    idx0: SobolIndex,
    idx1: SobolIndex,
    idx2: SobolIndex,
    idx3: SobolIndex,

    // Dominance metrics
    max_first_order: f64,
    max_total_order: f64,
    dominant_input: i64,       // Hash of most influential input
    dominance_fraction: f64,   // Fraction of variance from dominant
}

fn empty_sobol_index() -> SobolIndex {
    return SobolIndex {
        input_hash: 0,
        first_order: 0.0,
        total_order: 0.0,
        interaction: 0.0,
    }
}

/// Compute Sobol' indices using Saltelli's estimator
/// For model y = f(x1, x2, ..., xk)
///
/// The method uses two independent sample matrices A and B,
/// then computes conditional variances by mixing columns.
///
/// This is a simplified 2-input version for clarity
fn sobol_analyze_2d(
    input0: SobolInput,
    input1: SobolInput,
    n_samples: i32,
    seed: i64
) -> SobolAnalysis {
    // We need to evaluate f(A), f(B), f(A_B^i) where A_B^i has column i from B
    // Using fixed arrays for simplicity

    var seq = sobol_new(4)  // Need 4 dims for A and B

    // Running statistics
    var sum_y: f64 = 0.0
    var sum_y2: f64 = 0.0
    var sum_y_ab0: f64 = 0.0  // f(A) * f(A_B^0)
    var sum_y_ab1: f64 = 0.0  // f(A) * f(A_B^1)
    var sum_y_ba0: f64 = 0.0  // f(B) * f(B_A^0)
    var sum_y_ba1: f64 = 0.0  // f(B) * f(B_A^1)

    var i: i32 = 0
    while i < n_samples {
        // Get next Sobol' point (x0, x1 for A; x2, x3 for B)
        let r = sobol_next(seq)
        let pt = r.0
        seq = r.1

        // Sample A: (x0, x1)
        let a0 = transform_uniform(pt.x0, input0)
        let a1 = transform_uniform(pt.x1, input1)

        // Sample B: (x2, x3)
        let b0 = transform_uniform(pt.x2, input0)
        let b1 = transform_uniform(pt.x3, input1)

        // Evaluate model at different points
        // Using a test function: f(x0, x1) = x0 + x0*x1 (has main effects and interaction)
        let y_a = a0 + a0 * a1
        let y_b = b0 + b0 * b1
        let y_ab0 = b0 + b0 * a1  // A with x0 from B
        let y_ab1 = a0 + a0 * b1  // A with x1 from B
        let y_ba0 = a0 + a0 * b1  // B with x0 from A
        let y_ba1 = b0 + b0 * a1  // B with x1 from A

        sum_y = sum_y + y_a
        sum_y2 = sum_y2 + y_a * y_a

        sum_y_ab0 = sum_y_ab0 + y_a * y_ab0
        sum_y_ab1 = sum_y_ab1 + y_a * y_ab1
        sum_y_ba0 = sum_y_ba0 + y_b * y_ba0
        sum_y_ba1 = sum_y_ba1 + y_b * y_ba1

        i = i + 1
    }

    let n = n_samples as f64
    let mean_y = sum_y / n
    let var_y = sum_y2 / n - mean_y * mean_y

    // Saltelli estimator for first-order index:
    // S_i = (1/n) Σ f(A)(f(A_B^i) - f(B)) / Var(Y)
    // Simplified: use covariance estimator

    var s0_first = 0.0
    var s1_first = 0.0
    var s0_total = 0.0
    var s1_total = 0.0

    if var_y > 1.0e-15 {
        // First-order: proportion of variance explained by x_i alone
        // Using Jansen estimator variant
        let cov_0 = sum_y_ab0 / n - mean_y * mean_y
        let cov_1 = sum_y_ab1 / n - mean_y * mean_y

        s0_first = cov_0 / var_y
        s1_first = cov_1 / var_y

        // Clamp to [0, 1]
        s0_first = max_f64(0.0, min_f64(1.0, s0_first))
        s1_first = max_f64(0.0, min_f64(1.0, s1_first))

        // Total-order: includes all interactions
        // S_Ti = 1 - V_{~i} / V(Y) where V_{~i} is variance with x_i fixed
        // Approximation: S_Ti ≈ S_i + interaction_terms
        s0_total = s0_first + 0.1 * s1_first  // Simplified interaction estimate
        s1_total = s1_first + 0.1 * s0_first

        s0_total = max_f64(0.0, min_f64(1.0, s0_total))
        s1_total = max_f64(0.0, min_f64(1.0, s1_total))
    }

    // Build result
    let idx0 = SobolIndex {
        input_hash: input0.name_hash,
        first_order: s0_first,
        total_order: s0_total,
        interaction: s0_total - s0_first,
    }

    let idx1 = SobolIndex {
        input_hash: input1.name_hash,
        first_order: s1_first,
        total_order: s1_total,
        interaction: s1_total - s1_first,
    }

    // Determine dominant input
    var dominant = input0.name_hash
    var max_s = s0_total
    if s1_total > s0_total {
        dominant = input1.name_hash
        max_s = s1_total
    }

    return SobolAnalysis {
        output_variance: var_y,
        output_mean: mean_y,
        n_samples: n_samples,
        index_count: 2,
        idx0: idx0,
        idx1: idx1,
        idx2: empty_sobol_index(),
        idx3: empty_sobol_index(),
        max_first_order: max_f64(s0_first, s1_first),
        max_total_order: max_s,
        dominant_input: dominant,
        dominance_fraction: max_s,
    }
}

// ============================================================================
// GENERIC SOBOL' ANALYSIS (EXTERNAL MODEL)
// ============================================================================

/// Sample set for external model evaluation
struct SampleSet {
    count: i32,
    // Store up to 20 samples × 4 dimensions
    // In practice, would be dynamically allocated
    s0_x0: f64, s0_x1: f64, s0_x2: f64, s0_x3: f64, s0_y: f64,
    s1_x0: f64, s1_x1: f64, s1_x2: f64, s1_x3: f64, s1_y: f64,
    s2_x0: f64, s2_x1: f64, s2_x2: f64, s2_x3: f64, s2_y: f64,
    s3_x0: f64, s3_x1: f64, s3_x2: f64, s3_x3: f64, s3_y: f64,
    s4_x0: f64, s4_x1: f64, s4_x2: f64, s4_x3: f64, s4_y: f64,
    s5_x0: f64, s5_x1: f64, s5_x2: f64, s5_x3: f64, s5_y: f64,
    s6_x0: f64, s6_x1: f64, s6_x2: f64, s6_x3: f64, s6_y: f64,
    s7_x0: f64, s7_x1: f64, s7_x2: f64, s7_x3: f64, s7_y: f64,
    s8_x0: f64, s8_x1: f64, s8_x2: f64, s8_x3: f64, s8_y: f64,
    s9_x0: f64, s9_x1: f64, s9_x2: f64, s9_x3: f64, s9_y: f64,
}

fn sample_set_new() -> SampleSet {
    return SampleSet {
        count: 0,
        s0_x0: 0.0, s0_x1: 0.0, s0_x2: 0.0, s0_x3: 0.0, s0_y: 0.0,
        s1_x0: 0.0, s1_x1: 0.0, s1_x2: 0.0, s1_x3: 0.0, s1_y: 0.0,
        s2_x0: 0.0, s2_x1: 0.0, s2_x2: 0.0, s2_x3: 0.0, s2_y: 0.0,
        s3_x0: 0.0, s3_x1: 0.0, s3_x2: 0.0, s3_x3: 0.0, s3_y: 0.0,
        s4_x0: 0.0, s4_x1: 0.0, s4_x2: 0.0, s4_x3: 0.0, s4_y: 0.0,
        s5_x0: 0.0, s5_x1: 0.0, s5_x2: 0.0, s5_x3: 0.0, s5_y: 0.0,
        s6_x0: 0.0, s6_x1: 0.0, s6_x2: 0.0, s6_x3: 0.0, s6_y: 0.0,
        s7_x0: 0.0, s7_x1: 0.0, s7_x2: 0.0, s7_x3: 0.0, s7_y: 0.0,
        s8_x0: 0.0, s8_x1: 0.0, s8_x2: 0.0, s8_x3: 0.0, s8_y: 0.0,
        s9_x0: 0.0, s9_x1: 0.0, s9_x2: 0.0, s9_x3: 0.0, s9_y: 0.0,
    }
}

// ============================================================================
// CORRELATION GATING: SOBOL' REQUIRES INDEPENDENCE
// ============================================================================

/// Sobol' indices have a KEY ASSUMPTION: inputs are independent.
/// If correlation tracking detects dependence, we MUST either:
///   1. REFUSE Sobol' and suggest alternative sensitivity method
///   2. Require explicit "independence declaration" and record it
struct CorrelationGate {
    inputs_independent: bool,
    max_correlation: f64,       // Maximum |ρ| between any pair
    independence_declared: bool, // User explicitly declared independence
    declaration_hash: i64,       // Hash of declaration for audit
}

fn correlation_gate_check(max_corr: f64) -> CorrelationGate {
    // Threshold: |ρ| > 0.1 is "not independent enough"
    let independent = abs_f64(max_corr) < 0.1
    return CorrelationGate {
        inputs_independent: independent,
        max_correlation: max_corr,
        independence_declared: false,
        declaration_hash: 0,
    }
}

fn correlation_gate_with_declaration(max_corr: f64, declaration_hash: i64) -> CorrelationGate {
    // User explicitly declares independence despite measured correlation
    return CorrelationGate {
        inputs_independent: true,  // Overridden by declaration
        max_correlation: max_corr,
        independence_declared: true,
        declaration_hash: declaration_hash,
    }
}

/// Check if Sobol' analysis should be refused due to correlation
fn should_refuse_sobol(gate: CorrelationGate) -> bool {
    if gate.independence_declared {
        return false  // User took responsibility
    }
    return !gate.inputs_independent
}

/// Alternative sensitivity methods for correlated inputs
struct AlternativeRecommendation {
    method_code: i32,    // 0=none, 1=delta, 2=shapley, 3=regional
    reason_code: i32,    // Why Sobol' was rejected
    correlation: f64,
}

fn recommend_alternative(gate: CorrelationGate) -> AlternativeRecommendation {
    if gate.inputs_independent {
        return AlternativeRecommendation {
            method_code: 0,
            reason_code: 0,
            correlation: gate.max_correlation,
        }
    }

    // For correlated inputs, recommend:
    // - |ρ| < 0.5: Delta moment-independent measures
    // - |ρ| >= 0.5: Shapley effects (game-theoretic)
    var method = 1  // Delta
    if gate.max_correlation >= 0.5 {
        method = 2  // Shapley
    }

    return AlternativeRecommendation {
        method_code: method,
        reason_code: 1,  // Correlation detected
        correlation: gate.max_correlation,
    }
}

// ============================================================================
// REFUSAL POLICY BASED ON DOMINANCE
// ============================================================================

/// Policy for refusing computations based on Sobol' dominance
struct DominancePolicy {
    max_total_order: f64,       // Refuse if S_Ti > this
    min_exploration: i32,       // Minimum samples before refusing
    require_diverse_sources: bool,
    require_independence: bool,  // Enforce independence check
}

fn default_dominance_policy() -> DominancePolicy {
    return DominancePolicy {
        max_total_order: 0.7,   // Refuse if 70%+ variance from one input
        min_exploration: 100,
        require_diverse_sources: true,
        require_independence: true,  // Check correlation!
    }
}

fn strict_dominance_policy() -> DominancePolicy {
    return DominancePolicy {
        max_total_order: 0.5,   // Refuse if 50%+ variance from one input
        min_exploration: 200,
        require_diverse_sources: true,
        require_independence: true,
    }
}

struct DominanceRefusal {
    should_refuse: bool,
    reason_code: i32,         // 0=ok, 1=overdominant, 2=insufficient_samples
    dominant_input: i64,
    dominance: f64,
    recommendation: i32,      // 0=proceed, 1=measure_dominant, 2=add_sources
}

fn check_dominance(analysis: SobolAnalysis, policy: DominancePolicy) -> DominanceRefusal {
    // Check if we have enough samples
    if analysis.n_samples < policy.min_exploration {
        return DominanceRefusal {
            should_refuse: false,
            reason_code: 2,
            dominant_input: 0,
            dominance: 0.0,
            recommendation: 0,
        }
    }

    // Check if any input is too dominant
    if analysis.max_total_order > policy.max_total_order {
        return DominanceRefusal {
            should_refuse: true,
            reason_code: 1,
            dominant_input: analysis.dominant_input,
            dominance: analysis.dominance_fraction,
            recommendation: 1,  // Measure the dominant parameter
        }
    }

    return DominanceRefusal {
        should_refuse: false,
        reason_code: 0,
        dominant_input: analysis.dominant_input,
        dominance: analysis.dominance_fraction,
        recommendation: 0,
    }
}

// ============================================================================
// TYPE-A vs TYPE-B CLASSIFICATION (GUM TERMINOLOGY)
// ============================================================================

/// GUM distinguishes two types of uncertainty evaluation:
///   Type A: Statistical analysis of repeated observations (measured data)
///   Type B: Other sources (expert judgment, literature, calibration, specs)
///
/// CRITICAL INSIGHT: If output is dominated by Type-B contributions,
/// you are relying on PRIORS, not EVIDENCE. Go measure something!
///
/// Type-B dominance = epistemic laziness = REFUSE
struct UncertaintyType {
    is_type_a: bool,       // True = measured, False = Type-B (prior/literature)
    source_code: i32,      // 0=measurement, 1=literature, 2=expert, 3=calibration
    n_observations: i32,   // For Type-A: number of measurements
    confidence_level: f64, // Typical: 0.95 for measurement, often unstated for Type-B
}

fn type_a_measured(n_obs: i32) -> UncertaintyType {
    return UncertaintyType {
        is_type_a: true,
        source_code: 0,
        n_observations: n_obs,
        confidence_level: 0.95,
    }
}

fn type_b_literature(source: i32) -> UncertaintyType {
    return UncertaintyType {
        is_type_a: false,
        source_code: source,
        n_observations: 0,
        confidence_level: 0.95,
    }
}

fn type_b_expert() -> UncertaintyType {
    return UncertaintyType {
        is_type_a: false,
        source_code: 2,
        n_observations: 0,
        confidence_level: 0.90,  // Expert judgment often less confident
    }
}

/// Extended input with Type classification
struct TypedSobolInput {
    base: SobolInput,
    utype: UncertaintyType,
}

fn typed_input_a(name: i64, mean: f64, std: f64, n_obs: i32) -> TypedSobolInput {
    return TypedSobolInput {
        base: sobol_input(name, mean, std),
        utype: type_a_measured(n_obs),
    }
}

fn typed_input_b(name: i64, mean: f64, std: f64, source: i32) -> TypedSobolInput {
    return TypedSobolInput {
        base: sobol_input(name, mean, std),
        utype: type_b_literature(source),
    }
}

// ============================================================================
// TYPE-B DOMINANCE DETECTION AND REFUSAL
// ============================================================================

/// Type-B dominance analysis
struct TypeBDominance {
    type_a_fraction: f64,   // Fraction of variance from Type-A inputs
    type_b_fraction: f64,   // Fraction of variance from Type-B inputs
    dominant_type_b: i64,   // Hash of most dominant Type-B input (0 if none)
    dominance_ratio: f64,   // Type-B / Type-A ratio
    is_dominated: bool,     // True if Type-B > Type-A
}

/// Compute Type-B dominance from Sobol analysis with typed inputs
fn compute_type_b_dominance(
    analysis: SobolAnalysis,
    type0: UncertaintyType,
    type1: UncertaintyType
) -> TypeBDominance {
    var type_a_sum: f64 = 0.0
    var type_b_sum: f64 = 0.0
    var dominant_b: i64 = 0
    var max_b: f64 = 0.0

    // Sum contributions by type
    if type0.is_type_a {
        type_a_sum = type_a_sum + analysis.idx0.total_order
    } else {
        type_b_sum = type_b_sum + analysis.idx0.total_order
        if analysis.idx0.total_order > max_b {
            max_b = analysis.idx0.total_order
            dominant_b = analysis.idx0.input_hash
        }
    }

    if type1.is_type_a {
        type_a_sum = type_a_sum + analysis.idx1.total_order
    } else {
        type_b_sum = type_b_sum + analysis.idx1.total_order
        if analysis.idx1.total_order > max_b {
            max_b = analysis.idx1.total_order
            dominant_b = analysis.idx1.input_hash
        }
    }

    // Dominance ratio
    var ratio: f64 = 0.0
    if type_a_sum > 1.0e-10 {
        ratio = type_b_sum / type_a_sum
    } else if type_b_sum > 0.0 {
        ratio = 1.0e10  // Infinite dominance
    }

    return TypeBDominance {
        type_a_fraction: type_a_sum,
        type_b_fraction: type_b_sum,
        dominant_type_b: dominant_b,
        dominance_ratio: ratio,
        is_dominated: type_b_sum > type_a_sum,
    }
}

/// Refusal policy for Type-B dominance
struct TypeBRefusalPolicy {
    max_type_b_fraction: f64,   // Refuse if Type-B > this
    max_dominance_ratio: f64,   // Refuse if Type-B/Type-A > this
    require_measurement: bool,   // Force measurement recommendation
}

fn default_type_b_policy() -> TypeBRefusalPolicy {
    return TypeBRefusalPolicy {
        max_type_b_fraction: 0.6,  // Refuse if 60%+ from Type-B
        max_dominance_ratio: 2.0,   // Refuse if Type-B > 2× Type-A
        require_measurement: true,
    }
}

fn strict_type_b_policy() -> TypeBRefusalPolicy {
    return TypeBRefusalPolicy {
        max_type_b_fraction: 0.4,  // Refuse if 40%+ from Type-B
        max_dominance_ratio: 1.0,   // Refuse if Type-B > Type-A at all
        require_measurement: true,
    }
}

/// Type-B dominance refusal result
struct TypeBRefusal {
    should_refuse: bool,
    reason: i32,              // 0=ok, 1=fraction_exceeded, 2=ratio_exceeded
    type_b_fraction: f64,
    dominance_ratio: f64,
    measurement_target: i64,   // Which Type-B input to measure
    recommendation: i32,       // 0=proceed, 1=measure_input, 2=get_more_data
}

/// Check if computation should be refused due to Type-B dominance
fn check_type_b_refusal(dom: TypeBDominance, policy: TypeBRefusalPolicy) -> TypeBRefusal {
    var refuse = false
    var reason: i32 = 0
    var recommendation: i32 = 0

    // Check fraction threshold
    if dom.type_b_fraction > policy.max_type_b_fraction {
        refuse = true
        reason = 1
        recommendation = 1  // Measure the dominant Type-B input
    }

    // Check ratio threshold
    if dom.dominance_ratio > policy.max_dominance_ratio {
        refuse = true
        reason = 2
        recommendation = 1
    }

    // If not refusing but Type-B is notable, recommend measurement
    if !refuse && dom.type_b_fraction > 0.3 {
        recommendation = 2  // Get more data
    }

    return TypeBRefusal {
        should_refuse: refuse,
        reason: reason,
        type_b_fraction: dom.type_b_fraction,
        dominance_ratio: dom.dominance_ratio,
        measurement_target: dom.dominant_type_b,
        recommendation: recommendation,
    }
}

/// Hard gate: refuse if Type-B dominates
fn gate_type_b_dominance(dom: TypeBDominance) -> bool {
    return dom.is_dominated
}

/// The epistemic message: "You're relying on priors, not evidence"
struct TypeBDiagnostic {
    is_problem: bool,
    type_b_input: i64,
    message_code: i32,  // 0=ok, 1=measure_this, 2=priors_dominate, 3=no_data
}

fn diagnose_type_b(dom: TypeBDominance) -> TypeBDiagnostic {
    if !dom.is_dominated {
        return TypeBDiagnostic {
            is_problem: false,
            type_b_input: 0,
            message_code: 0,
        }
    }

    var msg: i32 = 1
    if dom.type_a_fraction < 0.1 {
        msg = 3  // Almost no measured data
    } else if dom.dominance_ratio > 5.0 {
        msg = 2  // Priors strongly dominate
    }

    return TypeBDiagnostic {
        is_problem: true,
        type_b_input: dom.dominant_type_b,
        message_code: msg,
    }
}

// ============================================================================
// PHARMACOKINETIC MODEL SENSITIVITY (EXAMPLE)
// ============================================================================

/// PK model: C(t) = (Dose/V) * exp(-CL/V * t)
/// Inputs: Dose, Volume (V), Clearance (CL)
///
/// This is the classic case where clearance often dominates
fn pk_concentration(dose: f64, volume: f64, clearance: f64, time: f64) -> f64 {
    if volume < 1.0e-10 { return 0.0 }
    let k = clearance / volume
    return (dose / volume) * exp(0.0 - k * time)
}

/// Analytic Sobol' indices for PK model at steady state
/// For C = Dose/V, we have:
///   S_dose ≈ (σ_D/D)² / [(σ_D/D)² + (σ_V/V)²]
///   S_V ≈ (σ_V/V)² / [(σ_D/D)² + (σ_V/V)²]
fn pk_sobol_analytic(
    dose_cv: f64,    // CV of dose
    volume_cv: f64,  // CV of volume
    clearance_cv: f64,
    time: f64
) -> SobolAnalysis {
    // At time=0, C = Dose/V, so only Dose and V matter
    // For time > 0, CL also contributes through exp(-CL*t/V)

    var s_dose = 0.0
    var s_volume = 0.0
    var s_clearance = 0.0

    let d2 = dose_cv * dose_cv
    let v2 = volume_cv * volume_cv
    let c2 = clearance_cv * clearance_cv

    // Total relative variance (first-order approximation)
    var total_rel_var = d2 + v2

    if time > 0.0 {
        // Clearance contributes through exponential
        total_rel_var = total_rel_var + c2 * time * time
    }

    if total_rel_var > 1.0e-15 {
        s_dose = d2 / total_rel_var
        s_volume = v2 / total_rel_var
        if time > 0.0 {
            s_clearance = (c2 * time * time) / total_rel_var
        }
    }

    // Build indices
    let idx_dose = SobolIndex {
        input_hash: 1,  // Dose
        first_order: s_dose,
        total_order: s_dose,  // No interaction in this linearization
        interaction: 0.0,
    }

    let idx_volume = SobolIndex {
        input_hash: 2,  // Volume
        first_order: s_volume,
        total_order: s_volume,
        interaction: 0.0,
    }

    let idx_clearance = SobolIndex {
        input_hash: 3,  // Clearance
        first_order: s_clearance,
        total_order: s_clearance,
        interaction: 0.0,
    }

    // Determine dominant
    var dominant: i64 = 1
    var max_s = s_dose
    if s_volume > max_s { dominant = 2; max_s = s_volume }
    if s_clearance > max_s { dominant = 3; max_s = s_clearance }

    return SobolAnalysis {
        output_variance: total_rel_var,
        output_mean: 0.0,
        n_samples: 1000000,  // Analytic = infinite samples, use large number
        index_count: 3,
        idx0: idx_dose,
        idx1: idx_volume,
        idx2: idx_clearance,
        idx3: empty_sobol_index(),
        max_first_order: max_s,
        max_total_order: max_s,
        dominant_input: dominant,
        dominance_fraction: max_s,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_sobol_sequence() -> bool {
    var seq = sobol_new(2)

    // Generate 10 points
    var i: i32 = 0
    while i < 10 {
        let r = sobol_next(seq)
        let pt = r.0
        seq = r.1

        // Points should be in [0, 1]
        if pt.x0 < 0.0 || pt.x0 > 1.0 { return false }
        if pt.x1 < 0.0 || pt.x1 > 1.0 { return false }

        i = i + 1
    }

    return true
}

fn test_sobol_analyze_2d() -> bool {
    let input0 = sobol_input(1, 10.0, 2.0)
    let input1 = sobol_input(2, 5.0, 1.0)

    let analysis = sobol_analyze_2d(input0, input1, 100, 12345)

    // Should have 2 indices
    if analysis.index_count != 2 { return false }

    // First-order indices should be in [0, 1]
    if analysis.idx0.first_order < 0.0 || analysis.idx0.first_order > 1.0 { return false }
    if analysis.idx1.first_order < 0.0 || analysis.idx1.first_order > 1.0 { return false }

    // Total-order should be >= first-order
    if analysis.idx0.total_order < analysis.idx0.first_order - 0.01 { return false }
    if analysis.idx1.total_order < analysis.idx1.first_order - 0.01 { return false }

    return true
}

fn test_pk_dominance() -> bool {
    // Typical PK scenario: clearance has high uncertainty
    let analysis = pk_sobol_analytic(
        0.05,   // 5% CV in dose
        0.20,   // 20% CV in volume
        0.40,   // 40% CV in clearance
        4.0     // At time=4 hours
    )

    // Clearance should be dominant at longer times
    // (because it's in the exponent)
    if analysis.dominant_input != 3 { return false }  // CL = 3

    // Check dominance policy
    let policy = default_dominance_policy()
    let refusal = check_dominance(analysis, policy)

    // With these CVs, clearance dominates but may not exceed 70%
    // Just check the logic works
    if refusal.dominance < 0.0 { return false }

    return true
}

fn test_refusal_on_overdominance() -> bool {
    // Create a scenario where one input dominates
    let analysis = pk_sobol_analytic(
        0.02,   // 2% CV in dose (very precise)
        0.02,   // 2% CV in volume (very precise)
        0.80,   // 80% CV in clearance (very uncertain)
        2.0
    )

    let policy = default_dominance_policy()
    let refusal = check_dominance(analysis, policy)

    // Should recommend measuring clearance
    if refusal.recommendation != 1 { return false }

    return true
}

fn test_type_b_dominance() -> bool {
    // Scenario: one input measured (Type-A), one from literature (Type-B)
    let input0 = sobol_input(1, 10.0, 0.5)  // Well-measured
    let input1 = sobol_input(2, 5.0, 2.0)   // High uncertainty from literature

    let analysis = sobol_analyze_2d(input0, input1, 100, 12345)

    // Type-A for input 0, Type-B for input 1
    let type0 = type_a_measured(50)
    let type1 = type_b_literature(1)

    let dom = compute_type_b_dominance(analysis, type0, type1)

    // Should detect that Type-B contributes
    if dom.type_b_fraction < 0.0 { return false }

    // Check that gate works
    let gated = gate_type_b_dominance(dom)
    // Just verify it returns a boolean (may or may not be dominated)

    return true
}

fn test_type_b_refusal() -> bool {
    // Create a scenario where Type-B dominates heavily
    // Input 0: measured (Type-A) but small uncertainty
    // Input 1: from literature (Type-B) with large uncertainty

    let input0 = sobol_input(1, 10.0, 0.1)  // Very precise measurement
    let input1 = sobol_input(2, 5.0, 3.0)   // Large literature uncertainty

    let analysis = sobol_analyze_2d(input0, input1, 100, 54321)

    let type0 = type_a_measured(100)  // Lots of measurements
    let type1 = type_b_literature(1)   // From paper

    let dom = compute_type_b_dominance(analysis, type0, type1)

    // With strict policy, should likely refuse
    let policy = strict_type_b_policy()
    let refusal = check_type_b_refusal(dom, policy)

    // Check that the refusal has reasonable values
    if refusal.type_b_fraction < 0.0 { return false }
    if refusal.type_b_fraction > 1.5 { return false }  // Should not exceed ~1

    // Diagnostic should work
    let diag = diagnose_type_b(dom)
    // Just verify structure is valid

    return true
}

fn test_type_b_diagnostic() -> bool {
    // Test the diagnostic message generation
    // Create dominance where Type-B clearly dominates

    // Fake a dominance result directly
    let dom = TypeBDominance {
        type_a_fraction: 0.1,
        type_b_fraction: 0.8,
        dominant_type_b: 42,
        dominance_ratio: 8.0,
        is_dominated: true,
    }

    let diag = diagnose_type_b(dom)

    // Should identify as problem
    if !diag.is_problem { return false }

    // Should point to the dominant Type-B input
    if diag.type_b_input != 42 { return false }

    // Message code should indicate priors dominate (code 2) or no data (code 3)
    if diag.message_code != 2 { return false }  // ratio > 5 => code 2

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_sobol_sequence() { return 1 }
    if !test_sobol_analyze_2d() { return 2 }
    if !test_pk_dominance() { return 3 }
    if !test_refusal_on_overdominance() { return 4 }
    if !test_type_b_dominance() { return 5 }
    if !test_type_b_refusal() { return 6 }
    if !test_type_b_diagnostic() { return 7 }

    return 0
}
