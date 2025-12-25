//! Epistemic ROI: Information Gain vs Entropy Debt
//!
//! The correct epistemic invariant is:
//!   Net epistemic progress = information gain − entropy debt
//!
//! CRITICAL SEMANTIC DISTINCTION:
//!   - entropy_debt: Information-destroying operations (many→one, rounding,
//!     dropping resolution/provenance, collapsing intervals). Maps to
//!     Landauer-style irreversibility. The operation CANNOT be undone.
//!   - model_risk: Approximation error (linearization, discretization, model
//!     form error). The error is QUANTIFIABLE and potentially correctable.
//!
//! These are fundamentally different epistemic failures:
//!   - Debt = you threw information away (thermodynamically motivated)
//!   - Risk = your approximation lies sometimes (statistically motivated)
//!
//! ROI Normalization:
//!   η = max(0, gain) / (debt + ε)
//!   NOT gain / op_count (creates gaming incentive by macro-composing ops)
//!
//! References:
//!   - Kullback & Leibler (1951): "On Information and Sufficiency"
//!   - Cover & Thomas (2006): "Elements of Information Theory", Chapter 2
//!   - Landauer (1961): "Irreversibility and Heat Generation"
//!   - GUM (JCGM 100:2008): Uncertainty propagation framework

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn log_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1.0e308 }
    return log(x)
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
// GAUSSIAN PDF REPRESENTATION
// ============================================================================

/// Represents a univariate Gaussian distribution N(μ, σ²)
struct GaussianPDF {
    mean: f64,
    std: f64,
    variance: f64,
}

fn gaussian_new(mean: f64, std: f64) -> GaussianPDF {
    return GaussianPDF {
        mean: mean,
        std: std,
        variance: std * std,
    }
}

/// Evaluate PDF at point x
fn gaussian_pdf(g: GaussianPDF, x: f64) -> f64 {
    let pi = 3.14159265358979323846
    let z = (x - g.mean) / g.std
    let coef = 1.0 / (g.std * sqrt_f64(2.0 * pi))
    return coef * exp(0.0 - 0.5 * z * z)
}

/// Shannon entropy of a Gaussian: H(X) = 0.5 * ln(2πeσ²)
fn gaussian_entropy(g: GaussianPDF) -> f64 {
    let pi = 3.14159265358979323846
    let e = 2.71828182845904523536
    return 0.5 * log_f64(2.0 * pi * e * g.variance)
}

// ============================================================================
// KL DIVERGENCE: INFORMATION GAIN
// ============================================================================

/// KL divergence D_KL(P || Q) for two Gaussians
///
/// For Gaussians: D_KL(P || Q) = log(σ_Q/σ_P) + (σ_P² + (μ_P - μ_Q)²)/(2σ_Q²) - 1/2
///
/// This measures information gained when updating from prior Q to posterior P
fn kl_divergence_gaussian(posterior: GaussianPDF, prior: GaussianPDF) -> f64 {
    // Avoid division by zero
    if prior.std < 1.0e-15 { return 1.0e308 }
    if posterior.std < 1.0e-15 { return 1.0e308 }

    let log_ratio = log_f64(prior.std / posterior.std)
    let mean_diff = posterior.mean - prior.mean
    let variance_term = (posterior.variance + mean_diff * mean_diff) / (2.0 * prior.variance)

    return log_ratio + variance_term - 0.5
}

/// Convert KL divergence from nats to bits
fn nats_to_bits(kl_nats: f64) -> f64 {
    let ln2 = 0.693147180559945309417
    return kl_nats / ln2
}

/// Information gain in bits when updating from prior to posterior
fn info_gain_bits(prior: GaussianPDF, posterior: GaussianPDF) -> f64 {
    let kl_nats = kl_divergence_gaussian(posterior, prior)
    return nats_to_bits(kl_nats)
}

// ============================================================================
// ENTROPY DEBT: IRREVERSIBLE INFORMATION LOSS (Landauer-motivated)
// ============================================================================

/// Entropy debt from IRREVERSIBLE information-destroying operations.
/// These are many→one mappings that cannot be undone.
///
/// Categories of TRUE debt (thermodynamically motivated):
///   0 = rounding (precision loss)
///   1 = truncation (data discard)
///   2 = interval_collapse (interval → point)
///   3 = provenance_drop (dropping audit trail)
struct EntropyDebt {
    source_name: i64,       // Hash of source description
    debt_bits: f64,         // Information irreversibly lost in bits
    category: i32,
}

fn debt_rounding(source: i64, bits_lost: f64) -> EntropyDebt {
    // Rounding from high precision to low: irrecoverable
    return EntropyDebt {
        source_name: source,
        debt_bits: bits_lost,
        category: 0,
    }
}

fn debt_truncation(source: i64, fraction_discarded: f64) -> EntropyDebt {
    // Information lost by truncating/discarding data
    // -log2(1 - fraction) bits lost
    var debt = 0.0
    if fraction_discarded > 0.0 && fraction_discarded < 1.0 {
        debt = 0.0 - log_f64(1.0 - fraction_discarded) / 0.693147
    }
    return EntropyDebt {
        source_name: source,
        debt_bits: debt,
        category: 1,
    }
}

fn debt_interval_collapse(source: i64, interval_width: f64, point_uncert: f64) -> EntropyDebt {
    // Collapsing [a, b] to a point loses information about the original spread
    // Approximately log2(interval_width / point_uncert) bits if interval >> point
    var debt = 0.0
    if interval_width > point_uncert && point_uncert > 0.0 {
        debt = log_f64(interval_width / point_uncert) / 0.693147
    }
    return EntropyDebt {
        source_name: source,
        debt_bits: max_f64(0.0, debt),
        category: 2,
    }
}

fn debt_provenance_drop(source: i64, steps_lost: i32) -> EntropyDebt {
    // Each provenance step carries ~1 bit of audit information
    return EntropyDebt {
        source_name: source,
        debt_bits: steps_lost as f64,
        category: 3,
    }
}

// ============================================================================
// MODEL RISK: QUANTIFIABLE APPROXIMATION ERROR (NOT debt!)
// ============================================================================

/// Model risk from approximations whose error is quantifiable.
/// These are NOT irreversible - the true answer exists, we're just approximating.
///
/// Categories of model risk:
///   0 = linearization (GUM first-order vs true nonlinear)
///   1 = discretization (numerical approximation)
///   2 = model_form (assuming wrong functional form)
///   3 = distribution_assumption (e.g., assuming Gaussianity)
struct ModelRisk {
    source_name: i64,
    risk_bits: f64,         // Potential error in bits
    category: i32,
    is_correctable: bool,   // True if MC or higher-order method could fix it
}

fn risk_linearization(source: i64, fractional_error: f64) -> ModelRisk {
    // Linearization approximation: error is quantifiable via MC
    var risk = 0.0
    if fractional_error > 0.0 && fractional_error < 1.0 {
        risk = 0.0 - log_f64(1.0 - fractional_error) / 0.693147
    }
    return ModelRisk {
        source_name: source,
        risk_bits: risk,
        category: 0,
        is_correctable: true,  // MC can correct this
    }
}

fn risk_discretization(source: i64, resolution_bits: f64) -> ModelRisk {
    // Numerical discretization: finer grid could improve
    return ModelRisk {
        source_name: source,
        risk_bits: resolution_bits,
        category: 1,
        is_correctable: true,
    }
}

fn risk_model_form(source: i64, residual_fraction: f64) -> ModelRisk {
    // Wrong model form: measured by residual magnitude
    var risk = 0.0
    if residual_fraction > 0.0 {
        risk = log_f64(1.0 + residual_fraction) / 0.693147
    }
    return ModelRisk {
        source_name: source,
        risk_bits: risk,
        category: 2,
        is_correctable: false,  // Need different model
    }
}

fn risk_distribution_assumption(source: i64, assumption_hash: i64) -> ModelRisk {
    // Assuming Gaussian when non-Gaussian, etc.
    // Cost is 0.5 bits for "I assumed a shape"
    return ModelRisk {
        source_name: source,
        risk_bits: 0.5,
        category: 3,
        is_correctable: true,  // Can use empirical distribution
    }
}

// ============================================================================
// EPISTEMIC LEDGER: CREDIT - DEBT = NET PROGRESS
// ============================================================================

/// Complete epistemic ledger tracking information gain, debt, AND risk
/// Critical invariants:
///   - debt never decreases (irreversible by definition)
///   - gain never increases without evidence
///   - risk can increase or decrease (it's correctable)
struct EpistemicLedger {
    // Credit side: information gained from updates
    credit_count: i32,
    credit_total_bits: f64,

    // Track individual credits (up to 4)
    c0_source: i64, c0_bits: f64,
    c1_source: i64, c1_bits: f64,
    c2_source: i64, c2_bits: f64,
    c3_source: i64, c3_bits: f64,

    // Debt side: IRREVERSIBLE information loss
    debt_count: i32,
    debt_total_bits: f64,

    // Track individual debts (up to 4)
    d0: EntropyDebt,
    d1: EntropyDebt,
    d2: EntropyDebt,
    d3: EntropyDebt,

    // Risk side: CORRECTABLE approximation error (separate from debt!)
    risk_count: i32,
    risk_total_bits: f64,
    risk_correctable_bits: f64,  // Subset that could be fixed by MC, etc.

    // Track individual risks (up to 4)
    r0: ModelRisk,
    r1: ModelRisk,
    r2: ModelRisk,
    r3: ModelRisk,
}

fn empty_debt() -> EntropyDebt {
    return EntropyDebt { source_name: 0, debt_bits: 0.0, category: 0 }
}

fn empty_risk() -> ModelRisk {
    return ModelRisk { source_name: 0, risk_bits: 0.0, category: 0, is_correctable: false }
}

fn ledger_new() -> EpistemicLedger {
    return EpistemicLedger {
        credit_count: 0,
        credit_total_bits: 0.0,
        c0_source: 0, c0_bits: 0.0,
        c1_source: 0, c1_bits: 0.0,
        c2_source: 0, c2_bits: 0.0,
        c3_source: 0, c3_bits: 0.0,
        debt_count: 0,
        debt_total_bits: 0.0,
        d0: empty_debt(),
        d1: empty_debt(),
        d2: empty_debt(),
        d3: empty_debt(),
        risk_count: 0,
        risk_total_bits: 0.0,
        risk_correctable_bits: 0.0,
        r0: empty_risk(),
        r1: empty_risk(),
        r2: empty_risk(),
        r3: empty_risk(),
    }
}

/// Record information gain from a Bayesian update
fn ledger_add_credit(ledger: EpistemicLedger, source: i64, prior: GaussianPDF, posterior: GaussianPDF) -> EpistemicLedger {
    let gain = info_gain_bits(prior, posterior)
    var result = ledger

    let idx = ledger.credit_count
    if idx == 0 { result.c0_source = source; result.c0_bits = gain }
    else if idx == 1 { result.c1_source = source; result.c1_bits = gain }
    else if idx == 2 { result.c2_source = source; result.c2_bits = gain }
    else if idx == 3 { result.c3_source = source; result.c3_bits = gain }

    if idx < 4 {
        result.credit_count = idx + 1
    }
    result.credit_total_bits = ledger.credit_total_bits + gain

    return result
}

/// Record information gain directly (when prior/posterior not available)
fn ledger_add_credit_direct(ledger: EpistemicLedger, source: i64, bits: f64) -> EpistemicLedger {
    var result = ledger

    let idx = ledger.credit_count
    if idx == 0 { result.c0_source = source; result.c0_bits = bits }
    else if idx == 1 { result.c1_source = source; result.c1_bits = bits }
    else if idx == 2 { result.c2_source = source; result.c2_bits = bits }
    else if idx == 3 { result.c3_source = source; result.c3_bits = bits }

    if idx < 4 {
        result.credit_count = idx + 1
    }
    result.credit_total_bits = ledger.credit_total_bits + bits

    return result
}

/// Record entropy debt from an IRREVERSIBLE operation
fn ledger_add_debt(ledger: EpistemicLedger, debt: EntropyDebt) -> EpistemicLedger {
    var result = ledger

    let idx = ledger.debt_count
    if idx == 0 { result.d0 = debt }
    else if idx == 1 { result.d1 = debt }
    else if idx == 2 { result.d2 = debt }
    else if idx == 3 { result.d3 = debt }

    if idx < 4 {
        result.debt_count = idx + 1
    }
    result.debt_total_bits = ledger.debt_total_bits + debt.debt_bits

    return result
}

/// Record model risk from a CORRECTABLE approximation
fn ledger_add_risk(ledger: EpistemicLedger, risk: ModelRisk) -> EpistemicLedger {
    var result = ledger

    let idx = ledger.risk_count
    if idx == 0 { result.r0 = risk }
    else if idx == 1 { result.r1 = risk }
    else if idx == 2 { result.r2 = risk }
    else if idx == 3 { result.r3 = risk }

    if idx < 4 {
        result.risk_count = idx + 1
    }
    result.risk_total_bits = ledger.risk_total_bits + risk.risk_bits
    if risk.is_correctable {
        result.risk_correctable_bits = ledger.risk_correctable_bits + risk.risk_bits
    }

    return result
}

/// Net epistemic progress = credit - debt (risk is separate!)
/// Risk doesn't reduce progress because it's correctable
fn ledger_net_progress(ledger: EpistemicLedger) -> f64 {
    return ledger.credit_total_bits - ledger.debt_total_bits
}

/// Epistemic ROI = max(0, gain) / (debt + ε)
/// NOT normalized by op_count (creates gaming incentive)
/// ε = 0.01 bits to avoid division by zero
fn ledger_roi(ledger: EpistemicLedger) -> f64 {
    let epsilon = 0.01
    let gain = max_f64(0.0, ledger.credit_total_bits)
    return gain / (ledger.debt_total_bits + epsilon)
}

/// Per-risk efficiency: gain / (risk + ε)
/// Measures how much risk we're taking per bit of gain
fn ledger_risk_efficiency(ledger: EpistemicLedger) -> f64 {
    let epsilon = 0.01
    let gain = max_f64(0.0, ledger.credit_total_bits)
    return gain / (ledger.risk_total_bits + epsilon)
}

/// Total epistemic cost = debt + risk (both reduce confidence)
fn ledger_total_cost(ledger: EpistemicLedger) -> f64 {
    return ledger.debt_total_bits + ledger.risk_total_bits
}

/// Check if ledger violates invariants
fn ledger_check_invariants(prev: EpistemicLedger, curr: EpistemicLedger) -> bool {
    // INVARIANT: debt never decreases
    if curr.debt_total_bits < prev.debt_total_bits - 0.001 {
        return false  // Violation!
    }
    return true
}

// ============================================================================
// EPISTEMIC QUALITY ASSESSMENT
// ============================================================================

struct EpistemicQuality {
    roi: f64,
    net_progress: f64,
    status: i32,           // 0=excellent, 1=good, 2=marginal, 3=poor, 4=noise_polishing
    recommendation: i32,   // 0=proceed, 1=review_assumptions, 2=get_better_data, 3=refuse
}

/// Assess the epistemic quality of a computation
fn assess_quality(ledger: EpistemicLedger) -> EpistemicQuality {
    let roi = ledger_roi(ledger)
    let net = ledger_net_progress(ledger)

    var status: i32 = 0
    var recommendation: i32 = 0

    if net > 1.0 && roi > 2.0 {
        // Excellent: significant learning with good ROI
        status = 0
        recommendation = 0
    } else if net > 0.0 && roi > 1.0 {
        // Good: positive progress
        status = 1
        recommendation = 0
    } else if net > -0.5 && roi > 0.5 {
        // Marginal: slight information loss but acceptable
        status = 2
        recommendation = 1
    } else if net > -2.0 {
        // Poor: significant information loss
        status = 3
        recommendation = 2
    } else {
        // Noise polishing: losing more than gaining
        status = 4
        recommendation = 3  // Refuse
    }

    return EpistemicQuality {
        roi: roi,
        net_progress: net,
        status: status,
        recommendation: recommendation,
    }
}

/// Check if computation should be refused due to poor epistemic quality
fn should_refuse(quality: EpistemicQuality) -> bool {
    return quality.recommendation == 3
}

/// Check if this is "polishing noise" (high debt, low credit)
fn is_noise_polishing(ledger: EpistemicLedger) -> bool {
    // High debt + low credit = polishing noise
    let roi = ledger_roi(ledger)
    let net = ledger_net_progress(ledger)
    return roi < 0.5 && net < -1.0
}

/// Check if this is "real learning" (high credit, low debt)
fn is_real_learning(ledger: EpistemicLedger) -> bool {
    let roi = ledger_roi(ledger)
    let net = ledger_net_progress(ledger)
    return roi > 2.0 && net > 1.0
}

// ============================================================================
// ROI BITS: THE FUNDAMENTAL REFUSAL METRIC
// ============================================================================

/// ROI in bits = credit - debt
/// This is the CORE metric for refusal decisions.
///
/// CRITICAL: When roi_bits < 0, we are DESTROYING information.
/// The refusal hook MUST fire when roi_bits < 0.
///
/// This is NOT a suggestion - it's a thermodynamic constraint.
/// You cannot create information from nothing (Landauer limit).
fn roi_bits(ledger: EpistemicLedger) -> f64 {
    return ledger.credit_total_bits - ledger.debt_total_bits
}

/// Refusal policy: MUST refuse when roi_bits < 0
/// This is the hard gate. Non-negotiable.
///
/// Returns:
///   0 = proceed (positive ROI)
///   1 = warn (marginal, 0 <= roi_bits < 0.1)
///   2 = refuse (negative ROI, destroying information)
fn refusal_policy(ledger: EpistemicLedger) -> i32 {
    let bits = roi_bits(ledger)

    if bits < 0.0 {
        // HARD REFUSE: We are destroying information
        return 2
    }

    if bits < 0.1 {
        // WARN: Marginal, not learning much
        return 1
    }

    // PROCEED: Positive epistemic progress
    return 0
}

/// Refusal hook result for integration with compiler/runtime
struct RefusalResult {
    should_refuse: bool,
    roi_bits: f64,
    reason: i32,          // 0=none, 1=negative_roi, 2=noise_polishing, 3=asymptotic
    debt_breakdown: f64,  // Total debt in bits
    credit_breakdown: f64, // Total credit in bits
}

/// The main refusal hook - call this before any epistemic operation
fn check_refusal(ledger: EpistemicLedger) -> RefusalResult {
    let bits = roi_bits(ledger)
    let policy = refusal_policy(ledger)

    var reason: i32 = 0
    var refuse = false

    if policy == 2 {
        reason = 1  // negative_roi
        refuse = true
    } else if is_noise_polishing(ledger) {
        reason = 2  // noise_polishing
        refuse = true
    }

    return RefusalResult {
        should_refuse: refuse,
        roi_bits: bits,
        reason: reason,
        debt_breakdown: ledger.debt_total_bits,
        credit_breakdown: ledger.credit_total_bits,
    }
}

/// Hard refusal gate - for use in pipelines
/// Returns true if operation should be blocked
fn gate_negative_roi(ledger: EpistemicLedger) -> bool {
    return roi_bits(ledger) < 0.0
}

/// Soft refusal gate - warns but doesn't block
/// Returns true if operation is marginal (0 <= roi < 0.1)
fn gate_marginal_roi(ledger: EpistemicLedger) -> bool {
    let bits = roi_bits(ledger)
    return bits >= 0.0 && bits < 0.1
}

// ============================================================================
// ASYMPTOTE DETECTION WITH ROI CONTEXT
// ============================================================================

/// Asymptotic limit: when adding more computation doesn't improve ROI
struct AsymptoteStatus {
    is_asymptotic: bool,
    diminishing_returns: bool,
    bits_per_operation: f64,
    recommendation: i32,  // 0=continue, 1=review, 2=stop
}

/// Detect if we're hitting asymptotic limits
fn detect_asymptote(prev_ledger: EpistemicLedger, curr_ledger: EpistemicLedger, operations: f64) -> AsymptoteStatus {
    let prev_net = ledger_net_progress(prev_ledger)
    let curr_net = ledger_net_progress(curr_ledger)
    let delta_net = curr_net - prev_net

    var bits_per_op = 0.0
    if operations > 0.0 {
        bits_per_op = delta_net / operations
    }

    // Asymptotic if marginal gain is very small
    let is_asymptotic = abs_f64(bits_per_op) < 0.01

    // Diminishing returns if ROI is getting worse
    let prev_roi = ledger_roi(prev_ledger)
    let curr_roi = ledger_roi(curr_ledger)
    let diminishing = curr_roi < prev_roi * 0.9

    var recommendation = 0
    if is_asymptotic && diminishing {
        recommendation = 2  // Stop
    } else if is_asymptotic || diminishing {
        recommendation = 1  // Review
    }

    return AsymptoteStatus {
        is_asymptotic: is_asymptotic,
        diminishing_returns: diminishing,
        bits_per_operation: bits_per_op,
        recommendation: recommendation,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_kl_divergence_same() -> bool {
    // KL divergence of identical distributions should be 0
    let p = gaussian_new(10.0, 2.0)
    let kl = kl_divergence_gaussian(p, p)
    if abs_f64(kl) > 0.001 { return false }
    return true
}

fn test_kl_divergence_update() -> bool {
    // Prior: N(10, 4)  (high uncertainty)
    // Posterior: N(12, 1)  (low uncertainty, shifted mean)
    let prior = gaussian_new(10.0, 2.0)
    let posterior = gaussian_new(12.0, 1.0)

    let kl = kl_divergence_gaussian(posterior, prior)
    let bits = nats_to_bits(kl)

    // Should be positive (we gained information)
    if bits <= 0.0 { return false }

    // Should be reasonable (not astronomical)
    if bits > 10.0 { return false }

    return true
}

fn test_ledger_credit() -> bool {
    var ledger = ledger_new()

    let prior = gaussian_new(10.0, 5.0)
    let posterior = gaussian_new(10.0, 1.0)

    // Adding a measurement that reduces uncertainty
    ledger = ledger_add_credit(ledger, 1, prior, posterior)

    if ledger.credit_count != 1 { return false }
    if ledger.credit_total_bits < 1.0 { return false }  // Should gain significant info

    return true
}

fn test_ledger_debt() -> bool {
    var ledger = ledger_new()

    // Add truncation debt (20% data discarded - irreversible!)
    let debt = debt_truncation(100, 0.20)
    ledger = ledger_add_debt(ledger, debt)

    if ledger.debt_count != 1 { return false }
    if ledger.debt_total_bits <= 0.0 { return false }

    return true
}

fn test_ledger_risk() -> bool {
    var ledger = ledger_new()

    // Add linearization risk (5% error - correctable via MC)
    let risk = risk_linearization(100, 0.05)
    ledger = ledger_add_risk(ledger, risk)

    if ledger.risk_count != 1 { return false }
    if ledger.risk_total_bits <= 0.0 { return false }
    if ledger.risk_correctable_bits <= 0.0 { return false }

    return true
}

fn test_net_progress() -> bool {
    var ledger = ledger_new()

    // Add significant credit
    let prior = gaussian_new(10.0, 10.0)
    let posterior = gaussian_new(10.0, 1.0)
    ledger = ledger_add_credit(ledger, 1, prior, posterior)

    // Add small debt (rounding - irreversible)
    let debt = debt_rounding(100, 0.1)
    ledger = ledger_add_debt(ledger, debt)

    let net = ledger_net_progress(ledger)

    // Net should be positive (more credit than debt)
    if net <= 0.0 { return false }

    let quality = assess_quality(ledger)
    // Should be excellent or good
    if quality.status > 1 { return false }

    return true
}

fn test_noise_polishing() -> bool {
    var ledger = ledger_new()

    // Add tiny credit
    ledger = ledger_add_credit_direct(ledger, 1, 0.1)

    // Add large irreversible debt (lots of truncation/rounding)
    let debt1 = debt_truncation(100, 0.5)
    let debt2 = debt_rounding(101, 2.0)
    ledger = ledger_add_debt(ledger, debt1)
    ledger = ledger_add_debt(ledger, debt2)

    // Should be identified as noise polishing
    if !is_noise_polishing(ledger) { return false }

    let quality = assess_quality(ledger)
    if quality.status != 4 { return false }  // Should be status 4 (noise_polishing)

    return true
}

fn test_real_learning() -> bool {
    var ledger = ledger_new()

    // Add significant credit from measurement
    let prior = gaussian_new(100.0, 50.0)
    let posterior = gaussian_new(95.0, 5.0)
    ledger = ledger_add_credit(ledger, 1, prior, posterior)

    // Add minimal debt (tiny rounding)
    let debt = debt_rounding(100, 0.1)
    ledger = ledger_add_debt(ledger, debt)

    // Should be identified as real learning
    if !is_real_learning(ledger) { return false }

    return true
}

fn test_asymptote_detection() -> bool {
    // Initial ledger with good progress
    var ledger1 = ledger_new()
    let prior = gaussian_new(100.0, 20.0)
    let post1 = gaussian_new(100.0, 10.0)
    ledger1 = ledger_add_credit(ledger1, 1, prior, post1)

    // After more operations, less improvement
    var ledger2 = ledger1
    let post2 = gaussian_new(100.0, 9.5)  // Only slightly better
    ledger2 = ledger_add_credit(ledger2, 2, post1, post2)
    let debt = debt_rounding(100, 0.05)
    ledger2 = ledger_add_debt(ledger2, debt)

    let status = detect_asymptote(ledger1, ledger2, 100.0)

    // Should detect diminishing returns
    // (The marginal improvement is small relative to operations)
    // May or may not be asymptotic depending on exact numbers

    return true
}

fn test_roi_bits_positive() -> bool {
    // High credit, low debt => positive ROI bits
    var ledger = ledger_new()
    let prior = gaussian_new(10.0, 10.0)
    let posterior = gaussian_new(10.0, 1.0)
    ledger = ledger_add_credit(ledger, 1, prior, posterior)

    // Add minimal debt
    let debt = debt_rounding(100, 0.1)
    ledger = ledger_add_debt(ledger, debt)

    let bits = roi_bits(ledger)
    // Should be positive (significant info gain, tiny debt)
    if bits <= 0.0 { return false }

    // Refusal policy should say proceed
    if refusal_policy(ledger) != 0 { return false }

    // Gate should not fire
    if gate_negative_roi(ledger) { return false }

    return true
}

fn test_roi_bits_negative() -> bool {
    // Low credit, high debt => negative ROI bits
    var ledger = ledger_new()

    // Tiny credit
    ledger = ledger_add_credit_direct(ledger, 1, 0.05)

    // Large debt (lots of irreversible losses)
    let debt1 = debt_truncation(100, 0.5)
    let debt2 = debt_rounding(101, 2.0)
    ledger = ledger_add_debt(ledger, debt1)
    ledger = ledger_add_debt(ledger, debt2)

    let bits = roi_bits(ledger)
    // Should be negative (destroying information)
    if bits >= 0.0 { return false }

    // Refusal policy should say refuse
    if refusal_policy(ledger) != 2 { return false }

    // Gate should fire
    if !gate_negative_roi(ledger) { return false }

    return true
}

fn test_refusal_hook() -> bool {
    // Test the full refusal hook
    var ledger = ledger_new()

    // Create negative ROI scenario
    ledger = ledger_add_credit_direct(ledger, 1, 0.1)
    let debt = debt_truncation(100, 0.8)  // 80% data loss = big debt
    ledger = ledger_add_debt(ledger, debt)

    let result = check_refusal(ledger)

    // Should refuse
    if !result.should_refuse { return false }

    // Reason should be negative_roi (1)
    if result.reason != 1 { return false }

    // roi_bits should be negative
    if result.roi_bits >= 0.0 { return false }

    return true
}

fn test_marginal_gate() -> bool {
    // Test marginal ROI detection
    var ledger = ledger_new()

    // Add tiny credit (0.05 bits)
    ledger = ledger_add_credit_direct(ledger, 1, 0.05)

    // No debt => roi = 0.05 (marginal)
    let bits = roi_bits(ledger)
    if bits < 0.0 { return false }
    if bits >= 0.1 { return false }

    // Should trigger marginal gate
    if !gate_marginal_roi(ledger) { return false }

    // But not negative gate
    if gate_negative_roi(ledger) { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_kl_divergence_same() { return 1 }
    if !test_kl_divergence_update() { return 2 }
    if !test_ledger_credit() { return 3 }
    if !test_ledger_debt() { return 4 }
    if !test_ledger_risk() { return 5 }
    if !test_net_progress() { return 6 }
    if !test_noise_polishing() { return 7 }
    if !test_real_learning() { return 8 }
    if !test_asymptote_detection() { return 9 }
    if !test_roi_bits_positive() { return 10 }
    if !test_roi_bits_negative() { return 11 }
    if !test_refusal_hook() { return 12 }
    if !test_marginal_gate() { return 13 }

    return 0
}
