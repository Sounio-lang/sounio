//! Epistemic Ledger — Entropic Cost Accounting
//!
//! Landauer's principle: erasing 1 bit costs k_B T ln 2 energy.
//! We track "entropic debt" (bits discarded) separately from "credit" (bits gained).
//!
//! Key insight: compression is NOT evidence. Information loss is debt.
//!
//! debt = bits_discarded (discretization, rounding, approximation)
//! credit = bits_gained (measurements, observations, evidence)
//!
//! References:
//!   - Landauer, "Irreversibility and Heat Generation" (1961)
//!   - Bennett, "Thermodynamics of Computation" (1982)
//!   - PROV-DM W3C Recommendation (2013)

extern "C" {
    fn log(x: f64) -> f64;
    fn sqrt(x: f64) -> f64;
}

fn log2(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1.0e308 }
    return log(x) / 0.693147180559945  // ln(2)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { return a }
    return b
}

// ============================================================================
// ENTROPIC LEDGER ENTRY
// ============================================================================

// Single entry in the entropic ledger
struct LedgerEntry {
    operation_id: i32,      // Sequential operation ID
    operation_type: i32,    // 0=erase, 1=measure, 2=compute, 3=compress
    bits_debt: f64,         // Bits lost (discretization, rounding)
    bits_credit: f64,       // Bits gained (evidence)
    net_entropy: f64,       // debt - credit
    provenance_id: i32,     // Link to PROV-DM entity
}

// Operation types
fn OP_ERASE() -> i32 { return 0 }
fn OP_MEASURE() -> i32 { return 1 }
fn OP_COMPUTE() -> i32 { return 2 }
fn OP_COMPRESS() -> i32 { return 3 }

// ============================================================================
// ENTROPIC LEDGER
// ============================================================================

// The ledger tracks all entropic transactions
struct EpistemicLedger {
    total_debt: f64,        // Σ bits_discarded
    total_credit: f64,      // Σ bits_gained
    net_balance: f64,       // debt - credit
    entry_count: i32,
    max_entries: i32,
    policy_threshold: f64,  // Auto-refuse if net > threshold
    reversible_mode: bool,  // If true, refuse any debt-incurring op
}

fn ledger_new(policy_threshold: f64) -> EpistemicLedger {
    return EpistemicLedger {
        total_debt: 0.0,
        total_credit: 0.0,
        net_balance: 0.0,
        entry_count: 0,
        max_entries: 1000,
        policy_threshold: policy_threshold,
        reversible_mode: false,
    }
}

fn ledger_enable_reversible(ledger: EpistemicLedger) -> EpistemicLedger {
    return EpistemicLedger {
        total_debt: ledger.total_debt,
        total_credit: ledger.total_credit,
        net_balance: ledger.net_balance,
        entry_count: ledger.entry_count,
        max_entries: ledger.max_entries,
        policy_threshold: ledger.policy_threshold,
        reversible_mode: true,
    }
}

// ============================================================================
// INFORMATION-THEORETIC BIT CALCULATIONS
// ============================================================================

// Bits lost when discretizing continuous value to resolution epsilon
// Based on rate-distortion theory: R(D) for Gaussian = -0.5 log2(D/sigma^2)
fn bits_discretized(sigma: f64, epsilon: f64) -> f64 {
    let sigma_val = sigma
    let eps_val = epsilon

    if sigma_val <= 0.0 { return 0.0 }
    if eps_val <= 0.0 { return 0.0 }

    // Information lost = log2(sigma / epsilon)
    // This is the number of distinguishable levels we're collapsing
    let ratio = sigma_val / eps_val
    if ratio <= 1.0 { return 0.0 }

    return log2(ratio)
}

// Bits gained from a measurement with precision sigma
// Shannon: H(X) = 0.5 log2(2*pi*e*sigma^2) for Gaussian
// But relative to prior sigma_prior: I = 0.5 log2(sigma_prior^2 / sigma^2)
fn bits_measured(sigma_prior: f64, sigma_posterior: f64) -> f64 {
    let prior = sigma_prior
    let post = sigma_posterior

    if prior <= 0.0 { return 0.0 }
    if post <= 0.0 { return 0.0 }
    if post >= prior { return 0.0 }  // No information gain

    // Information gain = log2(prior / posterior)
    return log2(prior / post)
}

// Bits lost when rounding to n decimal places
fn bits_rounded(n_decimals: i32) -> f64 {
    // Rounding to n decimals loses log2(10^n) bits relative to full precision
    // But we only count the "lost" resolution
    let n = n_decimals
    if n < 0 { return 0.0 }

    // Approximation: typical f64 has ~15 decimal digits
    // Rounding to n loses (15 - n) * log2(10) bits
    let digits_lost = 15 - n
    if digits_lost <= 0 { return 0.0 }

    let dl_f64 = 0.0 + digits_lost  // convert to f64
    return dl_f64 * 3.321928  // log2(10)
}

// ============================================================================
// EXPLICIT ERASURE OPERATIONS (Landauer cost)
// ============================================================================

// Discharge result with explicit acknowledgment
struct DischargeResult {
    ok: bool,               // True if erasure succeeded
    bits_erased: f64,       // Actual bits erased
    energy_joules: f64,     // Landauer minimum energy at room temp
    ledger_id: i32,         // Entry ID in ledger
}

// Landauer energy per bit at temperature T (Kelvin)
// E = k_B * T * ln(2)
fn landauer_energy(bits: f64, temp_kelvin: f64) -> f64 {
    let b = bits
    let t = temp_kelvin

    // k_B = 1.380649e-23 J/K
    // ln(2) = 0.693147
    let k_b_ln2 = 1.380649e-23 * 0.693147
    return b * k_b_ln2 * t
}

// Explicit discharge: user acknowledges information loss
fn discharge(ledger: EpistemicLedger, epsilon: f64, source_sigma: f64) -> DischargeResult {
    let l = ledger
    let eps = epsilon
    let sigma = source_sigma

    // Calculate bits being erased
    let bits_lost = bits_discretized(sigma, eps)

    // Check if reversible mode blocks this
    if l.reversible_mode && bits_lost > 0.0 {
        return DischargeResult {
            ok: false,
            bits_erased: 0.0,
            energy_joules: 0.0,
            ledger_id: 0 - 1,
        }
    }

    // Check if policy threshold would be exceeded
    let new_balance = l.net_balance + bits_lost
    if new_balance > l.policy_threshold && l.policy_threshold > 0.0 {
        return DischargeResult {
            ok: false,
            bits_erased: 0.0,
            energy_joules: 0.0,
            ledger_id: 0 - 1,
        }
    }

    // Compute Landauer minimum at room temperature (300K)
    let energy = landauer_energy(bits_lost, 300.0)

    return DischargeResult {
        ok: true,
        bits_erased: bits_lost,
        energy_joules: energy,
        ledger_id: l.entry_count,
    }
}

// Record an erasure in the ledger
fn ledger_record_erase(ledger: EpistemicLedger, bits: f64, prov_id: i32) -> EpistemicLedger {
    let l = ledger
    let b = bits
    let p = prov_id

    let new_debt = l.total_debt + b
    let new_balance = l.net_balance + b
    let new_count = l.entry_count + 1

    return EpistemicLedger {
        total_debt: new_debt,
        total_credit: l.total_credit,
        net_balance: new_balance,
        entry_count: new_count,
        max_entries: l.max_entries,
        policy_threshold: l.policy_threshold,
        reversible_mode: l.reversible_mode,
    }
}

// Record a measurement (credit) in the ledger
fn ledger_record_measure(ledger: EpistemicLedger, bits: f64, prov_id: i32) -> EpistemicLedger {
    let l = ledger
    let b = bits

    let new_credit = l.total_credit + b
    let new_balance = l.net_balance - b  // Credit reduces balance
    let new_count = l.entry_count + 1

    return EpistemicLedger {
        total_debt: l.total_debt,
        total_credit: new_credit,
        net_balance: new_balance,
        entry_count: new_count,
        max_entries: l.max_entries,
        policy_threshold: l.policy_threshold,
        reversible_mode: l.reversible_mode,
    }
}

// ============================================================================
// REVERSIBLE OPERATIONS (ZERO DEBT)
// ============================================================================

// Result of a reversible operation
struct ReversibleResult {
    value: f64,
    inverse_exists: bool,
    bits_debt: f64,  // Should be 0 for reversible ops
}

// Reversible addition (always invertible)
fn reversible_add(x: f64, y: f64) -> ReversibleResult {
    // Copy params to locals (workaround for codegen bug)
    let x_val = x
    let y_val = y
    let sum = x_val + y_val

    return ReversibleResult {
        value: sum,
        inverse_exists: true,
        bits_debt: 0.0,
    }
}

// Reversible multiplication (invertible if y != 0)
fn reversible_mul(x: f64, y: f64) -> ReversibleResult {
    // Copy params to locals
    let x_val = x
    let y_val = y
    let product = x_val * y_val
    let invertible = abs_f64(y_val) > 1.0e-15

    return ReversibleResult {
        value: product,
        inverse_exists: invertible,
        bits_debt: 0.0,
    }
}

// Division - reversible if x != 0
fn reversible_div(x: f64, y: f64) -> ReversibleResult {
    // Copy params to locals
    let x_val = x
    let y_val = y

    if abs_f64(y_val) < 1.0e-15 {
        return ReversibleResult {
            value: 0.0,
            inverse_exists: false,
            bits_debt: 0.0,
        }
    }

    let quotient = x_val / y_val
    let invertible = abs_f64(x_val) > 1.0e-15

    return ReversibleResult {
        value: quotient,
        inverse_exists: invertible,
        bits_debt: 0.0,
    }
}

// ============================================================================
// POLICY GATES
// ============================================================================

// Policy gate result
struct PolicyDecision {
    allowed: bool,
    reason_code: i32,   // 0=ok, 1=threshold, 2=reversible, 3=contributor
    excess_bits: f64,   // How many bits over threshold
}

// Check if operation is allowed under current policy
fn policy_check(ledger: EpistemicLedger, proposed_debt: f64) -> PolicyDecision {
    let l = ledger
    let d = proposed_debt

    // Check reversible mode
    if l.reversible_mode && d > 0.0 {
        return PolicyDecision {
            allowed: false,
            reason_code: 2,
            excess_bits: d,
        }
    }

    // Check threshold
    let new_balance = l.net_balance + d
    if l.policy_threshold > 0.0 && new_balance > l.policy_threshold {
        let excess = new_balance - l.policy_threshold
        return PolicyDecision {
            allowed: false,
            reason_code: 1,
            excess_bits: excess,
        }
    }

    return PolicyDecision {
        allowed: true,
        reason_code: 0,
        excess_bits: 0.0,
    }
}

// Identify top contributors to debt
struct DebtContributor {
    operation_id: i32,
    bits_debt: f64,
    percentage: f64,
}

fn top_contributor(ledger: EpistemicLedger) -> DebtContributor {
    // In a real implementation, we'd track individual entries
    // For now, return aggregate info
    let l = ledger

    return DebtContributor {
        operation_id: l.entry_count - 1,
        bits_debt: l.total_debt,
        percentage: 100.0,
    }
}

// ============================================================================
// INVARIANT TESTS
// ============================================================================

// Conservation: debt cannot spontaneously decrease
fn invariant_debt_monotonic(old_debt: f64, new_debt: f64) -> bool {
    return new_debt >= old_debt
}

// Credit cannot spontaneously increase without measurement
fn invariant_credit_bounded(old_credit: f64, new_credit: f64) -> bool {
    return new_credit >= old_credit
}

// Net balance = debt - credit (always)
fn invariant_balance_correct(ledger: EpistemicLedger) -> bool {
    let l = ledger
    let expected = l.total_debt - l.total_credit
    return abs_f64(l.net_balance - expected) < 1.0e-10
}

// ============================================================================
// ENERGY EQUIVALENCE (OPTIONAL VISUALIZATION)
// ============================================================================

// Convert bits to equivalent energy at given temperature
struct EnergyEquivalent {
    joules: f64,
    electron_volts: f64,
    calories: f64,
}

fn bits_to_energy(bits: f64, temp_kelvin: f64) -> EnergyEquivalent {
    let b = bits
    let t = temp_kelvin

    let joules = landauer_energy(b, t)
    let ev = joules / 1.602176634e-19
    let cal = joules / 4.184

    return EnergyEquivalent {
        joules: joules,
        electron_volts: ev,
        calories: cal,
    }
}

// Human-scale metaphor: how many bits = boiling a cup of water?
// 1 cup water (250mL) from 20C to 100C = 250 * 4.184 * 80 = 83680 J
// At 300K: bits = E / (k_B * T * ln(2)) = 83680 / (1.38e-23 * 300 * 0.693)
// = 83680 / 2.87e-21 = 2.9e25 bits
fn bits_to_boil_cup() -> f64 {
    return 2.9e25
}

// ============================================================================
// TESTS
// ============================================================================

fn test_bits_discretized() -> bool {
    // sigma = 1.0, epsilon = 0.1 -> should lose ~3.32 bits (log2(10))
    let bits = bits_discretized(1.0, 0.1)
    if abs_f64(bits - 3.321928) > 0.01 { return false }

    // sigma = 1.0, epsilon = 1.0 -> no loss
    let bits2 = bits_discretized(1.0, 1.0)
    if bits2 != 0.0 { return false }

    return true
}

fn test_bits_measured() -> bool {
    // Prior sigma = 10, posterior = 1 -> gained log2(10) = 3.32 bits
    let bits = bits_measured(10.0, 1.0)
    if abs_f64(bits - 3.321928) > 0.01 { return false }

    // No gain if posterior >= prior
    let bits2 = bits_measured(1.0, 2.0)
    if bits2 != 0.0 { return false }

    return true
}

fn test_landauer_energy() -> bool {
    // 1 bit at 300K = k_B * T * ln(2) = 1.38e-23 * 300 * 0.693 = 2.87e-21 J
    let e = landauer_energy(1.0, 300.0)
    if abs_f64(e - 2.87e-21) > 1.0e-22 { return false }

    return true
}

fn test_ledger_debt_credit() -> bool {
    var ledger = ledger_new(100.0)

    // Record some debt
    ledger = ledger_record_erase(ledger, 5.0, 1)
    if abs_f64(ledger.total_debt - 5.0) > 0.001 { return false }
    if abs_f64(ledger.net_balance - 5.0) > 0.001 { return false }

    // Record some credit
    ledger = ledger_record_measure(ledger, 3.0, 2)
    if abs_f64(ledger.total_credit - 3.0) > 0.001 { return false }
    if abs_f64(ledger.net_balance - 2.0) > 0.001 { return false }

    // Check invariant
    if !invariant_balance_correct(ledger) { return false }

    return true
}

fn test_policy_gate() -> bool {
    var ledger = ledger_new(10.0)  // Threshold = 10 bits

    // Should allow small debt
    let check1 = policy_check(ledger, 5.0)
    if !check1.allowed { return false }

    // Record the debt
    ledger = ledger_record_erase(ledger, 5.0, 1)

    // Should deny exceeding threshold
    let check2 = policy_check(ledger, 10.0)  // Would make total = 15 > 10
    if check2.allowed { return false }
    if check2.reason_code != 1 { return false }

    return true
}

fn test_reversible_mode() -> bool {
    var ledger = ledger_new(100.0)
    ledger = ledger_enable_reversible(ledger)

    // Should deny any debt in reversible mode
    let check = policy_check(ledger, 0.001)
    if check.allowed { return false }
    if check.reason_code != 2 { return false }

    // Zero debt should be allowed
    let check2 = policy_check(ledger, 0.0)
    if !check2.allowed { return false }

    return true
}

fn test_reversible_ops() -> bool {
    let r1 = reversible_add(5.0, 3.0)
    if abs_f64(r1.value - 8.0) > 0.001 { return false }
    if !r1.inverse_exists { return false }
    if r1.bits_debt != 0.0 { return false }

    let r2 = reversible_mul(5.0, 3.0)
    if abs_f64(r2.value - 15.0) > 0.001 { return false }
    if !r2.inverse_exists { return false }

    let r3 = reversible_mul(5.0, 0.0)
    if r3.inverse_exists { return false }  // 0 is NOT invertible, should be false

    return true
}

fn test_discharge() -> bool {
    var ledger = ledger_new(100.0)

    // Discharge with sigma=1, epsilon=0.1 -> ~3.32 bits
    let result = discharge(ledger, 0.1, 1.0)
    if !result.ok { return false }
    if abs_f64(result.bits_erased - 3.321928) > 0.01 { return false }
    if result.energy_joules <= 0.0 { return false }

    return true
}

fn test_invariants() -> bool {
    // Debt monotonic
    if !invariant_debt_monotonic(5.0, 6.0) { return false }
    if invariant_debt_monotonic(6.0, 5.0) { return false }

    // Credit bounded
    if !invariant_credit_bounded(5.0, 6.0) { return false }

    return true
}

fn main() -> i32 {
    if !test_bits_discretized() { return 1 }
    if !test_bits_measured() { return 2 }
    if !test_landauer_energy() { return 3 }
    if !test_ledger_debt_credit() { return 4 }
    if !test_policy_gate() { return 5 }
    if !test_reversible_mode() { return 6 }
    if !test_reversible_ops() { return 7 }
    if !test_discharge() { return 8 }
    if !test_invariants() { return 9 }

    return 0
}
