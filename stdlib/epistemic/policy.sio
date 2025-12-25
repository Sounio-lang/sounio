//! Epistemic Policy Module
//!
//! This module enforces "shoreline" behavior: making it harder to lie
//! with numbers than to tell the truth.
//!
//! KEY PRINCIPLE: Dropping epistemic metadata must be:
//! 1. Syntactically noisy (explicit function call)
//! 2. Auditable (recorded in provenance)
//! 3. Policy-gated (can be forbidden by context)
//!
//! NO IMPLICIT UNWRAP. EVER.

// ============================================================================
// UNCERTAINTY MODE
// ============================================================================

// How to represent and propagate uncertainty
enum UncertaintyMode {
    // Standard deviation (GUM fast path, research iteration)
    StdDev,

    // Interval arithmetic (conservative, clinical safety)
    Interval,

    // Both: StdDev for typical, interval for guaranteed bounds
    Dual,
}

// ============================================================================
// EPISTEMIC POLICY
// ============================================================================

// Policy object required at epistemic boundaries
// This prevents "silent defaults" from becoming doctrine
struct EpistemicPolicy {
    // Minimum acceptable confidence (0.0 = accept anything)
    min_conf: f64,

    // How to propagate uncertainty
    mode: UncertaintyMode,

    // Whether boost() is allowed (explicit confidence increase)
    allow_boost: bool,

    // Maximum acceptable DS conflict before error
    max_conflict: f64,

    // Coverage factor for expanded uncertainty (default k=2 for ~95%)
    default_coverage_k: f64,

    // Whether to allow unwrap without reason
    require_unwrap_reason: bool,

    // Whether to track all operations in provenance
    full_provenance: bool,
}

// ============================================================================
// POLICY CONSTRUCTORS
// ============================================================================

// Research mode: permissive, fast iteration
fn policy_research() -> EpistemicPolicy {
    return EpistemicPolicy {
        min_conf: 0.0,
        mode: UncertaintyMode::StdDev,
        allow_boost: true,
        max_conflict: 1.0,  // allow any conflict
        default_coverage_k: 2.0,
        require_unwrap_reason: false,
        full_provenance: false,
    }
}

// Clinical mode: conservative, safety-first
fn policy_clinical() -> EpistemicPolicy {
    return EpistemicPolicy {
        min_conf: 0.7,
        mode: UncertaintyMode::Interval,
        allow_boost: false,  // no artificial confidence
        max_conflict: 0.3,   // reject high-conflict evidence
        default_coverage_k: 2.576,  // 99% coverage
        require_unwrap_reason: true,
        full_provenance: true,
    }
}

// Regulatory mode: strictest, for submission
fn policy_regulatory() -> EpistemicPolicy {
    return EpistemicPolicy {
        min_conf: 0.9,
        mode: UncertaintyMode::Dual,
        allow_boost: false,
        max_conflict: 0.1,
        default_coverage_k: 3.0,  // 99.7% coverage
        require_unwrap_reason: true,
        full_provenance: true,
    }
}

// Default policy (research mode)
fn policy_default() -> EpistemicPolicy {
    return policy_research()
}

// ============================================================================
// UNWRAP RESULTS
// ============================================================================

// Result of attempting to unwrap an epistemic value
struct UnwrapResult {
    value: f64,
    succeeded: bool,
    reason_code: i32,  // 0=ok, 1=below_min_conf, 2=missing_reason, 3=policy_forbids
}

// Risk acknowledgment levels
enum RiskLevel {
    // I understand this loses epistemic information
    Acknowledged,

    // I've validated this is acceptable for my use case
    Validated,

    // This is for display/logging only, not computation
    DisplayOnly,

    // Emergency override (logged with warning)
    Override,
}

// ============================================================================
// EPISTEMIC VALUE WITH POLICY SUPPORT
// ============================================================================

// Import core types (simplified for self-contained module)
struct Uncertainty {
    tag: i32,
    std_u: f64,
    std_k: f64,
    interval_lo: f64,
    interval_hi: f64,
}

struct PolicyValue {
    value: f64,
    uncert: Uncertainty,
    conf: f64,
    source_id: i64,      // VarID for correlation tracking
    provenance_id: i64,
}

fn uncertainty_std(u: f64) -> Uncertainty {
    return Uncertainty {
        tag: 1,
        std_u: if u < 0.0 { 0.0 } else { u },
        std_k: 1.0,
        interval_lo: 0.0,
        interval_hi: 0.0,
    }
}

fn policy_value_new(value: f64, std_u: f64, conf: f64, source_id: i64) -> PolicyValue {
    return PolicyValue {
        value: value,
        uncert: uncertainty_std(std_u),
        conf: conf,
        source_id: source_id,
        provenance_id: 0,
    }
}

// ============================================================================
// UNWRAP FUNCTIONS (THE ONLY WAY TO EXTRACT VALUES)
// ============================================================================

// Unwrap with full acknowledgment (auditable)
// Returns UnwrapResult, caller must check succeeded
fn unwrap_with_policy(
    v: PolicyValue,
    policy: EpistemicPolicy,
    risk: RiskLevel,
    reason_id: i64  // 0 = no reason provided
) -> UnwrapResult {
    // Check if reason is required
    if policy.require_unwrap_reason && reason_id == 0 {
        return UnwrapResult {
            value: 0.0,
            succeeded: false,
            reason_code: 2,  // missing_reason
        }
    }

    // Check minimum confidence
    if v.conf < policy.min_conf {
        return UnwrapResult {
            value: 0.0,
            succeeded: false,
            reason_code: 1,  // below_min_conf
        }
    }

    // Success - in a real impl, would log to provenance
    return UnwrapResult {
        value: v.value,
        succeeded: true,
        reason_code: 0,
    }
}

// Require minimum confidence (returns success flag + value)
fn require_conf(v: PolicyValue, min_conf: f64) -> UnwrapResult {
    if v.conf >= min_conf {
        return UnwrapResult {
            value: v.value,
            succeeded: true,
            reason_code: 0,
        }
    }
    return UnwrapResult {
        value: 0.0,
        succeeded: false,
        reason_code: 1,
    }
}

// Hard fail if below threshold (returns value or panics)
fn refuse_below(v: PolicyValue, min_conf: f64) -> f64 {
    if v.conf < min_conf {
        // In a real impl, this would panic/abort
        // For now, return sentinel
        return -999999.0
    }
    return v.value
}

// Display-only extraction (explicitly marked as non-computational)
fn for_display(v: PolicyValue) -> f64 {
    // Always succeeds, but caller promises not to use for computation
    return v.value
}

// ============================================================================
// POLICY VALIDATION
// ============================================================================

// Check if a value meets policy requirements
fn meets_policy(v: PolicyValue, policy: EpistemicPolicy) -> bool {
    return v.conf >= policy.min_conf
}

// Check DS conflict against policy
fn conflict_acceptable(conflict_k: f64, policy: EpistemicPolicy) -> bool {
    return conflict_k <= policy.max_conflict
}

// ============================================================================
// EXPANDED UNCERTAINTY
// ============================================================================

// Get expanded uncertainty interval with coverage factor
fn expanded_interval(v: PolicyValue, k: f64) -> (f64, f64) {
    let u = v.uncert.std_u
    let half_width = k * u
    return (v.value - half_width, v.value + half_width)
}

// Get expanded uncertainty with policy's default coverage
fn expanded_default(v: PolicyValue, policy: EpistemicPolicy) -> (f64, f64) {
    return expanded_interval(v, policy.default_coverage_k)
}

// Standard coverage factors
fn coverage_68() -> f64 { return 1.0 }      // 68% (1σ)
fn coverage_90() -> f64 { return 1.645 }    // 90%
fn coverage_95() -> f64 { return 1.96 }     // 95%
fn coverage_99() -> f64 { return 2.576 }    // 99%
fn coverage_997() -> f64 { return 3.0 }     // 99.7% (3σ)

// ============================================================================
// POLICY CONTEXT
// ============================================================================

// Note: Global policy state would require runtime support.
// For now, policy is passed explicitly to functions that need it.
// This is actually better: explicit > implicit.

// Get uncertainty mode as integer (for comparisons)
fn policy_mode_tag(policy: EpistemicPolicy) -> i32 {
    if policy.mode == UncertaintyMode::StdDev { return 0 }
    if policy.mode == UncertaintyMode::Interval { return 1 }
    return 2  // Dual
}

// ============================================================================
// TESTS
// ============================================================================

fn test_unwrap_respects_min_conf() -> bool {
    let v = policy_value_new(100.0, 5.0, 0.8, 1)
    let policy = policy_clinical()  // min_conf = 0.7

    let result = unwrap_with_policy(v, policy, RiskLevel::Acknowledged, 1)
    return result.succeeded  // 0.8 >= 0.7, should succeed
}

fn test_unwrap_rejects_low_conf() -> bool {
    let v = policy_value_new(100.0, 5.0, 0.5, 1)
    let policy = policy_clinical()  // min_conf = 0.7

    let result = unwrap_with_policy(v, policy, RiskLevel::Acknowledged, 1)
    return !result.succeeded  // 0.5 < 0.7, should fail
}

fn test_unwrap_requires_reason() -> bool {
    let v = policy_value_new(100.0, 5.0, 0.9, 1)
    let policy = policy_clinical()  // require_unwrap_reason = true

    // No reason provided (reason_id = 0)
    let result = unwrap_with_policy(v, policy, RiskLevel::Acknowledged, 0)
    return !result.succeeded && result.reason_code == 2
}

fn test_research_mode_permissive() -> bool {
    let v = policy_value_new(100.0, 5.0, 0.3, 1)
    let policy = policy_research()  // min_conf = 0.0

    let result = unwrap_with_policy(v, policy, RiskLevel::Acknowledged, 0)
    return result.succeeded  // research mode allows anything
}

fn test_expanded_uncertainty() -> bool {
    let v = policy_value_new(100.0, 5.0, 0.9, 1)

    // 95% coverage (k=1.96)
    let interval = expanded_interval(v, coverage_95())
    let lo = interval.0
    let hi = interval.1

    // Should be approximately [90.2, 109.8]
    let expected_lo = 100.0 - 1.96 * 5.0
    let expected_hi = 100.0 + 1.96 * 5.0

    if lo < expected_lo - 0.1 || lo > expected_lo + 0.1 {
        return false
    }
    if hi < expected_hi - 0.1 || hi > expected_hi + 0.1 {
        return false
    }
    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_unwrap_respects_min_conf() { return 1 }
    if !test_unwrap_rejects_low_conf() { return 2 }
    if !test_unwrap_requires_reason() { return 3 }
    if !test_research_mode_permissive() { return 4 }
    if !test_expanded_uncertainty() { return 5 }

    return 0
}
