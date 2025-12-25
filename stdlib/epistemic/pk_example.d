//! Pharmacokinetic Example with Rigorous Epistemic Semantics
//!
//! This example demonstrates epistemic computing on a real pharmacokinetic
//! calculation. Every value carries:
//!
//! - UNCERTAINTY (Channel A): How precisely we know the value (GUM propagation)
//! - CONFIDENCE (Channel B): How much we trust the source (monotone non-increasing)
//!
//! The example computes drug concentration using a one-compartment PK model:
//!
//!   C(t) = (D / V) * exp(-k * t)
//!
//! where:
//!   D = dose (mg)
//!   V = volume of distribution (L)
//!   k = elimination rate constant (1/h)
//!   t = time (h)
//!
//! Each parameter comes from a different source with different uncertainty
//! and confidence levels. The computation properly propagates both.

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn exp_f64(x: f64) -> f64 {
    return exp(x)
}

fn log_f64(x: f64) -> f64 {
    if x <= 0.0 { return -1.0e308 }
    return log(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

// ============================================================================
// EPISTEMIC VALUE (from core.d, simplified here for self-contained example)
// ============================================================================

struct Uncertainty {
    tag: i32,           // 0=Exact, 1=StdDev, 2=Interval
    std_u: f64,
    std_k: f64,
    interval_lo: f64,
    interval_hi: f64,
}

struct EpistemicValue {
    value: f64,
    uncert: Uncertainty,
    conf: f64,
    source: i32,        // 0=unknown, 1=literature, 2=measured, 3=computed
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

fn uncert_to_std(u: Uncertainty) -> f64 {
    if u.tag == 0 { return 0.0 }
    if u.tag == 1 { return u.std_u }
    return (u.interval_hi - u.interval_lo) / 4.0
}

// Create epistemic value from literature source
fn from_literature(value: f64, cv_percent: f64, confidence: f64) -> EpistemicValue {
    // CV = coefficient of variation = std/mean * 100
    let std = value * cv_percent / 100.0
    return EpistemicValue {
        value: value,
        uncert: uncertainty_std(std),
        conf: confidence,
        source: 1,  // literature
    }
}

// Create epistemic value from measurement
fn from_measurement(value: f64, std: f64, confidence: f64) -> EpistemicValue {
    return EpistemicValue {
        value: value,
        uncert: uncertainty_std(std),
        conf: confidence,
        source: 2,  // measured
    }
}

// Create exact epistemic value (e.g., from prescription)
fn exact(value: f64, confidence: f64) -> EpistemicValue {
    return EpistemicValue {
        value: value,
        uncert: Uncertainty { tag: 0, std_u: 0.0, std_k: 1.0, interval_lo: 0.0, interval_hi: 0.0 },
        conf: confidence,
        source: 0,
    }
}

// ============================================================================
// GUM-COMPLIANT ARITHMETIC
// ============================================================================

// Multiplication with GUM propagation
fn mul_ep(a: EpistemicValue, b: EpistemicValue) -> EpistemicValue {
    let value = a.value * b.value
    let conf = min_f64(a.conf, b.conf)

    let ua = uncert_to_std(a.uncert)
    let ub = uncert_to_std(b.uncert)

    var u_rel_a = 0.0
    var u_rel_b = 0.0
    if abs_f64(a.value) > 1.0e-15 {
        u_rel_a = ua / abs_f64(a.value)
    }
    if abs_f64(b.value) > 1.0e-15 {
        u_rel_b = ub / abs_f64(b.value)
    }

    let u_rel = sqrt_f64(u_rel_a * u_rel_a + u_rel_b * u_rel_b)
    let u_combined = abs_f64(value) * u_rel

    return EpistemicValue {
        value: value,
        uncert: uncertainty_std(u_combined),
        conf: conf,
        source: 3,  // computed
    }
}

// Division with GUM propagation
fn div_ep(a: EpistemicValue, b: EpistemicValue) -> EpistemicValue {
    if abs_f64(b.value) < 1.0e-15 {
        return EpistemicValue {
            value: 0.0,
            uncert: Uncertainty { tag: 2, std_u: 0.0, std_k: 1.0, interval_lo: -1.0e308, interval_hi: 1.0e308 },
            conf: 0.0,
            source: 3,
        }
    }

    let value = a.value / b.value
    let conf = min_f64(a.conf, b.conf)

    let ua = uncert_to_std(a.uncert)
    let ub = uncert_to_std(b.uncert)

    var u_rel_a = 0.0
    var u_rel_b = 0.0
    if abs_f64(a.value) > 1.0e-15 {
        u_rel_a = ua / abs_f64(a.value)
    }
    if abs_f64(b.value) > 1.0e-15 {
        u_rel_b = ub / abs_f64(b.value)
    }

    let u_rel = sqrt_f64(u_rel_a * u_rel_a + u_rel_b * u_rel_b)
    let u_combined = abs_f64(value) * u_rel

    return EpistemicValue {
        value: value,
        uncert: uncertainty_std(u_combined),
        conf: conf,
        source: 3,
    }
}

// Negation (for exponential argument)
fn neg_ep(a: EpistemicValue) -> EpistemicValue {
    return EpistemicValue {
        value: 0.0 - a.value,
        uncert: a.uncert,  // uncertainty unchanged
        conf: a.conf,
        source: 3,
    }
}

// Exponential with uncertainty propagation
// For y = exp(x): u(y) = |y| * u(x)
fn exp_ep(a: EpistemicValue) -> EpistemicValue {
    let value = exp_f64(a.value)
    let ua = uncert_to_std(a.uncert)

    // Derivative of exp is exp, so u(y) = |exp(x)| * u(x) = y * u(x)
    let u_combined = abs_f64(value) * ua

    return EpistemicValue {
        value: value,
        uncert: uncertainty_std(u_combined),
        conf: a.conf,  // confidence unchanged for unary ops
        source: 3,
    }
}

// ============================================================================
// PK MODEL
// ============================================================================

// One-compartment PK model: C(t) = (D/V) * exp(-k*t)
fn concentration_at_time(
    dose: EpistemicValue,
    volume: EpistemicValue,
    k_elim: EpistemicValue,
    time: EpistemicValue
) -> EpistemicValue {
    // C0 = D / V
    let c0 = div_ep(dose, volume)

    // k * t
    let kt = mul_ep(k_elim, time)

    // -k * t
    let neg_kt = neg_ep(kt)

    // exp(-k*t)
    let decay = exp_ep(neg_kt)

    // C(t) = C0 * exp(-k*t)
    let concentration = mul_ep(c0, decay)

    return concentration
}

// Calculate half-life: t_half = ln(2) / k
fn half_life(k_elim: EpistemicValue) -> EpistemicValue {
    let ln2 = exact(0.693147, 1.0)  // ln(2) is exact
    return div_ep(ln2, k_elim)
}

// Calculate AUC (area under curve) for one-compartment IV bolus
// AUC = D / (V * k) = D / CL
fn auc_iv_bolus(
    dose: EpistemicValue,
    volume: EpistemicValue,
    k_elim: EpistemicValue
) -> EpistemicValue {
    // CL = V * k
    let clearance = mul_ep(volume, k_elim)

    // AUC = D / CL
    return div_ep(dose, clearance)
}

// ============================================================================
// REPORTING
// ============================================================================

fn relative_uncertainty_percent(e: EpistemicValue) -> f64 {
    if abs_f64(e.value) < 1.0e-15 {
        return 100.0
    }
    return uncert_to_std(e.uncert) / abs_f64(e.value) * 100.0
}

fn get_95_ci_low(e: EpistemicValue) -> f64 {
    let u = uncert_to_std(e.uncert)
    return e.value - 1.96 * u
}

fn get_95_ci_high(e: EpistemicValue) -> f64 {
    let u = uncert_to_std(e.uncert)
    return e.value + 1.96 * u
}

// ============================================================================
// EXAMPLE: THEOPHYLLINE PK
// ============================================================================

fn main() -> i32 {
    // =========================================================
    // PARAMETERS WITH EPISTEMIC PROVENANCE
    // =========================================================

    // Dose: 300 mg IV bolus (from prescription, exact, high confidence)
    let dose = exact(300.0, 0.99)

    // Volume of distribution: 0.5 L/kg * 70 kg = 35 L
    // From literature: CV ~20%, confidence 0.85 (population average)
    let volume = from_literature(35.0, 20.0, 0.85)

    // Elimination rate constant: 0.08 1/h
    // From literature: CV ~30%, confidence 0.80 (high variability)
    let k_elim = from_literature(0.08, 30.0, 0.80)

    // Time point: 4 hours (exact, from study protocol)
    let time = exact(4.0, 1.0)

    // =========================================================
    // COMPUTE CONCENTRATION WITH FULL UNCERTAINTY
    // =========================================================

    let conc = concentration_at_time(dose, volume, k_elim, time)

    // =========================================================
    // VERIFY EPISTEMIC INVARIANTS
    // =========================================================

    // INVARIANT 1: Confidence should be <= min of inputs
    let min_input_conf = min_f64(min_f64(dose.conf, volume.conf),
                                  min_f64(k_elim.conf, time.conf))

    if conc.conf > min_input_conf + 1.0e-10 {
        // VIOLATION: confidence increased during computation
        return 1
    }

    // INVARIANT 2: Uncertainty should grow appropriately
    // Note: For exp(x), the relative uncertainty transformation is:
    // u_rel(exp(x)) ≈ |x| * u(x) / |exp(x)| = |x| * u_rel(x) * |x/exp(x)|
    // This can reduce relative uncertainty when |x| < 1
    // So we check that ABSOLUTE uncertainty hasn't disappeared
    let conc_rel_u = relative_uncertainty_percent(conc)

    // The concentration should still have meaningful uncertainty
    // (at least ~15% given the input uncertainties)
    if conc_rel_u < 15.0 {
        // VIOLATION: uncertainty suspiciously low
        return 2
    }

    // =========================================================
    // COMPUTE DERIVED PK PARAMETERS
    // =========================================================

    let t_half = half_life(k_elim)
    let auc = auc_iv_bolus(dose, volume, k_elim)

    // Verify derived parameters also maintain invariants
    if t_half.conf > k_elim.conf + 1.0e-10 {
        return 3
    }

    if auc.conf > min_input_conf + 1.0e-10 {
        return 4
    }

    // =========================================================
    // SANITY CHECK: VALUES SHOULD BE REASONABLE
    // =========================================================

    // Concentration at 4h should be positive
    if conc.value < 0.0 {
        return 5
    }

    // Half-life should be ~8.7 hours for k=0.08/h
    // t_half = ln(2)/0.08 ≈ 8.66
    if abs_f64(t_half.value - 8.66) > 1.0 {
        return 6
    }

    // AUC should be D/CL = 300/(35*0.08) = 300/2.8 ≈ 107
    if abs_f64(auc.value - 107.0) > 10.0 {
        return 7
    }

    // =========================================================
    // VERIFY CONFIDENCE IS PROPERLY DEGRADED
    // =========================================================

    // The concentration confidence should reflect the least trusted input
    // which is k_elim at 0.80
    if conc.conf > 0.81 {
        return 8
    }

    // =========================================================
    // VERIFY UNCERTAINTY PROPAGATION IS CORRECT
    // =========================================================

    // For theophylline at t=4h:
    // C(4) = (300/35) * exp(-0.08*4)
    //      = 8.57 * exp(-0.32)
    //      = 8.57 * 0.726
    //      = 6.22 mg/L

    let expected_conc = 6.22
    if abs_f64(conc.value - expected_conc) > 0.5 {
        return 9
    }

    // The uncertainty should be substantial given 20% and 30% CVs
    // Note: exponential with small argument reduces relative uncertainty
    // Expected: sqrt(20² + ~10²) ≈ 22%
    if conc_rel_u < 15.0 {
        // Uncertainty suspiciously low
        return 10
    }

    // =========================================================
    // ALL INVARIANTS VERIFIED
    // =========================================================

    return 0
}
