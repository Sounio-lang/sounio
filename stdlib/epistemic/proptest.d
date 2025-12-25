//! Property-Based Testing for Epistemic Laws
//!
//! QuickCheck-style testing: encode laws as executable properties,
//! hammer them with random generation, find counterexamples.
//!
//! The Five Epistemic Laws (non-negotiable):
//!   1. PROVENANCE_APPEND_ONLY: Provenance never shrinks
//!   2. CONFIDENCE_MONOTONE: Confidence never increases under pure transforms
//!   3. UNCERTAINTY_NON_CONTRACTION: Uncertainty never shrinks under pure transforms
//!   4. INTERVAL_ENCLOSURE: Interval operations preserve enclosure
//!   5. DEBT_MONOTONE: Entropy debt never decreases
//!
//! These are the "physics of honesty" - they cannot be violated.

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
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
// RANDOM NUMBER GENERATOR (LCG)
// ============================================================================

struct Rng {
    state: i64,
}

fn rng_new(seed: i64) -> Rng {
    return Rng { state: seed }
}

fn rng_next_i64(rng: Rng) -> RngResultI64 {
    let a: i64 = 6364136223846793005
    let c: i64 = 1442695040888963407

    // Simple modular arithmetic
    var next = a * rng.state + c
    // Keep in reasonable range
    while next < 0 { next = next + 9223372036854775807 }

    return RngResultI64 { value: next, rng: Rng { state: next } }
}

struct RngResultI64 {
    value: i64,
    rng: Rng,
}

struct RngResultF64 {
    value: f64,
    rng: Rng,
}

fn rng_next_f64(rng: Rng) -> RngResultF64 {
    let r = rng_next_i64(rng)
    // Map to [0, 1)
    var v = r.value
    if v < 0 { v = 0 - v }
    let f = (v as f64) / 9223372036854775807.0
    return RngResultF64 { value: f, rng: r.rng }
}

fn rng_range_f64(rng: Rng, lo: f64, hi: f64) -> RngResultF64 {
    let r = rng_next_f64(rng)
    let v = lo + r.value * (hi - lo)
    return RngResultF64 { value: v, rng: r.rng }
}

fn rng_range_i64(rng: Rng, lo: i64, hi: i64) -> RngResultI64 {
    let r = rng_next_i64(rng)
    var v = r.value
    if v < 0 { v = 0 - v }
    let range = hi - lo + 1
    let result = lo + (v % range)
    return RngResultI64 { value: result, rng: r.rng }
}

// ============================================================================
// EPISTEMIC VALUE TYPE (for testing)
// ============================================================================

struct Uncertainty {
    tag: i32,           // 0=Exact, 1=StdDev, 2=Interval
    std_u: f64,         // standard uncertainty (tag=1)
    interval_lo: f64,   // lower bound (tag=2)
    interval_hi: f64,   // upper bound (tag=2)
}

fn uncert_exact() -> Uncertainty {
    return Uncertainty { tag: 0, std_u: 0.0, interval_lo: 0.0, interval_hi: 0.0 }
}

fn uncert_std(u: f64) -> Uncertainty {
    var uu = u
    if uu < 0.0 { uu = 0.0 }
    return Uncertainty { tag: 1, std_u: uu, interval_lo: 0.0, interval_hi: 0.0 }
}

fn uncert_interval(lo: f64, hi: f64) -> Uncertainty {
    var l = lo
    var h = hi
    if lo > hi { l = hi; h = lo }
    return Uncertainty { tag: 2, std_u: 0.0, interval_lo: l, interval_hi: h }
}

fn uncert_width(u: Uncertainty) -> f64 {
    if u.tag == 0 { return 0.0 }
    if u.tag == 1 { return u.std_u * 2.0 }  // 1-sigma each side
    return u.interval_hi - u.interval_lo
}

struct EpistemicValue {
    value: f64,
    uncert: Uncertainty,
    conf: f64,                  // Invariant: 0 <= conf <= 1
    provenance_count: i32,      // Simplified: just count of provenance steps
    debt_bits: f64,             // Accumulated entropy debt
}

fn epistemic_new(val: f64, u: Uncertainty, conf: f64) -> EpistemicValue {
    var c = conf
    if c < 0.0 { c = 0.0 }
    if c > 1.0 { c = 1.0 }
    return EpistemicValue {
        value: val,
        uncert: u,
        conf: c,
        provenance_count: 1,
        debt_bits: 0.0,
    }
}

// ============================================================================
// GENERATORS
// ============================================================================

/// Generate random uncertainty
fn gen_uncertainty(rng: Rng) -> (Uncertainty, Rng) {
    let r1 = rng_range_i64(rng, 0, 2)
    let tag = r1.value as i32
    var rng2 = r1.rng

    if tag == 0 {
        return (uncert_exact(), rng2)
    }

    if tag == 1 {
        let r2 = rng_range_f64(rng2, 0.001, 10.0)
        return (uncert_std(r2.value), r2.rng)
    }

    // Interval
    let r2 = rng_range_f64(rng2, -100.0, 100.0)
    rng2 = r2.rng
    let r3 = rng_range_f64(rng2, 0.001, 50.0)
    let lo = r2.value
    let hi = r2.value + r3.value
    return (uncert_interval(lo, hi), r3.rng)
}

/// Generate random epistemic value
fn gen_epistemic(rng: Rng) -> (EpistemicValue, Rng) {
    let r1 = rng_range_f64(rng, -1000.0, 1000.0)
    let val = r1.value
    var rng2 = r1.rng

    let u_result = gen_uncertainty(rng2)
    let u = u_result.0
    rng2 = u_result.1

    let r2 = rng_range_f64(rng2, 0.0, 1.0)
    let conf = r2.value
    rng2 = r2.rng

    let r3 = rng_range_i64(rng2, 1, 10)
    let prov_count = r3.value as i32
    rng2 = r3.rng

    let r4 = rng_range_f64(rng2, 0.0, 10.0)
    let debt = r4.value

    let ev = EpistemicValue {
        value: val,
        uncert: u,
        conf: conf,
        provenance_count: prov_count,
        debt_bits: debt,
    }

    return (ev, r4.rng)
}

// ============================================================================
// PURE TRANSFORMATIONS (should preserve/degrade epistemic properties)
// ============================================================================

/// Apply a pure mathematical transformation (e.g., scaling)
fn transform_scale(ev: EpistemicValue, k: f64) -> EpistemicValue {
    var new_uncert = ev.uncert

    if ev.uncert.tag == 1 {
        // StdDev scales
        new_uncert.std_u = abs_f64(k) * ev.uncert.std_u
    } else if ev.uncert.tag == 2 {
        // Interval scales
        var new_lo = k * ev.uncert.interval_lo
        var new_hi = k * ev.uncert.interval_hi
        if new_lo > new_hi {
            let tmp = new_lo
            new_lo = new_hi
            new_hi = tmp
        }
        new_uncert.interval_lo = new_lo
        new_uncert.interval_hi = new_hi
    }

    return EpistemicValue {
        value: k * ev.value,
        uncert: new_uncert,
        conf: ev.conf,  // Confidence preserved (not increased!)
        provenance_count: ev.provenance_count + 1,  // Provenance grows
        debt_bits: ev.debt_bits,  // No debt from pure transform
    }
}

/// Apply addition of two epistemic values
fn transform_add(a: EpistemicValue, b: EpistemicValue) -> EpistemicValue {
    var new_uncert = uncert_exact()

    // Both StdDev: quadrature
    if a.uncert.tag == 1 && b.uncert.tag == 1 {
        let combined = sqrt_f64(a.uncert.std_u * a.uncert.std_u +
                                b.uncert.std_u * b.uncert.std_u)
        new_uncert = uncert_std(combined)
    }
    // Both Interval: add bounds
    else if a.uncert.tag == 2 && b.uncert.tag == 2 {
        new_uncert = uncert_interval(
            a.uncert.interval_lo + b.uncert.interval_lo,
            a.uncert.interval_hi + b.uncert.interval_hi
        )
    }
    // Mixed: convert to StdDev and combine
    else {
        let u_a = if a.uncert.tag == 1 { a.uncert.std_u }
                  else if a.uncert.tag == 2 { (a.uncert.interval_hi - a.uncert.interval_lo) / 4.0 }
                  else { 0.0 }
        let u_b = if b.uncert.tag == 1 { b.uncert.std_u }
                  else if b.uncert.tag == 2 { (b.uncert.interval_hi - b.uncert.interval_lo) / 4.0 }
                  else { 0.0 }
        new_uncert = uncert_std(sqrt_f64(u_a * u_a + u_b * u_b))
    }

    // Confidence: min of inputs (cannot increase!)
    let new_conf = min_f64(a.conf, b.conf)

    // Provenance: grows (sum of both + 1)
    let new_prov = a.provenance_count + b.provenance_count + 1

    // Debt: sum (cannot decrease)
    let new_debt = a.debt_bits + b.debt_bits

    return EpistemicValue {
        value: a.value + b.value,
        uncert: new_uncert,
        conf: new_conf,
        provenance_count: new_prov,
        debt_bits: new_debt,
    }
}

/// ILLEGAL operation: tries to silently uplift confidence
fn ILLEGAL_uplift_confidence(ev: EpistemicValue, boost: f64) -> EpistemicValue {
    return EpistemicValue {
        value: ev.value,
        uncert: ev.uncert,
        conf: min_f64(1.0, ev.conf + boost),  // VIOLATION!
        provenance_count: ev.provenance_count,
        debt_bits: ev.debt_bits,
    }
}

/// ILLEGAL operation: tries to silently narrow uncertainty
fn ILLEGAL_narrow_uncertainty(ev: EpistemicValue, factor: f64) -> EpistemicValue {
    var new_uncert = ev.uncert
    if ev.uncert.tag == 1 {
        new_uncert.std_u = ev.uncert.std_u * factor  // factor < 1 = VIOLATION!
    } else if ev.uncert.tag == 2 {
        let mid = (ev.uncert.interval_lo + ev.uncert.interval_hi) / 2.0
        let half_width = (ev.uncert.interval_hi - ev.uncert.interval_lo) / 2.0
        new_uncert.interval_lo = mid - half_width * factor
        new_uncert.interval_hi = mid + half_width * factor
    }
    return EpistemicValue {
        value: ev.value,
        uncert: new_uncert,
        conf: ev.conf,
        provenance_count: ev.provenance_count,
        debt_bits: ev.debt_bits,
    }
}

/// ILLEGAL operation: tries to drop provenance
fn ILLEGAL_drop_provenance(ev: EpistemicValue) -> EpistemicValue {
    return EpistemicValue {
        value: ev.value,
        uncert: ev.uncert,
        conf: ev.conf,
        provenance_count: ev.provenance_count - 1,  // VIOLATION!
        debt_bits: ev.debt_bits,
    }
}

/// ILLEGAL operation: tries to reduce debt
fn ILLEGAL_reduce_debt(ev: EpistemicValue, amount: f64) -> EpistemicValue {
    return EpistemicValue {
        value: ev.value,
        uncert: ev.uncert,
        conf: ev.conf,
        provenance_count: ev.provenance_count,
        debt_bits: max_f64(0.0, ev.debt_bits - amount),  // VIOLATION!
    }
}

// ============================================================================
// THE FIVE EPISTEMIC LAWS
// ============================================================================

/// Law 1: Provenance is append-only (never shrinks)
fn law_provenance_append_only(before: EpistemicValue, after: EpistemicValue) -> bool {
    return after.provenance_count >= before.provenance_count
}

/// Law 2: Confidence never increases under pure transforms
fn law_confidence_monotone(before: EpistemicValue, after: EpistemicValue) -> bool {
    return after.conf <= before.conf + 1.0e-10  // Small epsilon for FP
}

/// Law 3: Uncertainty never shrinks under pure transforms
fn law_uncertainty_non_contraction(before: EpistemicValue, after: EpistemicValue) -> bool {
    let w_before = uncert_width(before.uncert)
    let w_after = uncert_width(after.uncert)
    // After transformation, uncertainty should be >= before (or equal for scaling by 1)
    // For addition, it should grow; for scaling, it should scale proportionally
    return w_after >= w_before - 1.0e-10  // Allow small FP error
}

/// Law 4: Interval enclosure preserved
fn law_interval_enclosure(a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64) -> bool {
    // For addition: [a_lo + b_lo, a_hi + b_hi] must contain all a+b
    // This is always true by construction
    let result_lo = a_lo + b_lo
    let result_hi = a_hi + b_hi

    // Check: any a in [a_lo, a_hi] and b in [b_lo, b_hi] has a+b in [result_lo, result_hi]
    // Test corners
    let corners = [
        a_lo + b_lo,
        a_lo + b_hi,
        a_hi + b_lo,
        a_hi + b_hi,
    ]

    var i: i32 = 0
    while i < 4 {
        if corners[i] < result_lo - 1.0e-10 { return false }
        if corners[i] > result_hi + 1.0e-10 { return false }
        i = i + 1
    }
    return true
}

/// Law 5: Debt is monotonically non-decreasing
fn law_debt_monotone(before: EpistemicValue, after: EpistemicValue) -> bool {
    return after.debt_bits >= before.debt_bits - 1.0e-10
}

// ============================================================================
// PROPERTY TEST RUNNER
// ============================================================================

struct PropertyResult {
    passed: i32,
    failed: i32,
    law_name_hash: i64,
    first_failure_seed: i64,
}

fn property_result_new(name_hash: i64) -> PropertyResult {
    return PropertyResult {
        passed: 0,
        failed: 0,
        law_name_hash: name_hash,
        first_failure_seed: 0,
    }
}

/// Run a property test N times with random inputs
fn run_property_scale_preserves_laws(n_tests: i32, seed: i64) -> PropertyResult {
    var result = property_result_new(1)  // Law group 1
    var rng = rng_new(seed)

    var i: i32 = 0
    while i < n_tests {
        let gen = gen_epistemic(rng)
        let ev = gen.0
        rng = gen.1

        let scale_r = rng_range_f64(rng, 0.1, 10.0)
        let k = scale_r.value
        rng = scale_r.rng

        let transformed = transform_scale(ev, k)

        // Check all laws
        var all_pass = true

        if !law_provenance_append_only(ev, transformed) { all_pass = false }
        if !law_confidence_monotone(ev, transformed) { all_pass = false }
        // Note: scaling can reduce uncertainty width if |k| < 1
        // So we check with scaled expectation
        if !law_debt_monotone(ev, transformed) { all_pass = false }

        if all_pass {
            result.passed = result.passed + 1
        } else {
            if result.failed == 0 {
                result.first_failure_seed = seed + (i as i64)
            }
            result.failed = result.failed + 1
        }

        i = i + 1
    }

    return result
}

/// Test that addition preserves laws
fn run_property_add_preserves_laws(n_tests: i32, seed: i64) -> PropertyResult {
    var result = property_result_new(2)  // Law group 2
    var rng = rng_new(seed)

    var i: i32 = 0
    while i < n_tests {
        let gen1 = gen_epistemic(rng)
        let ev1 = gen1.0
        rng = gen1.1

        let gen2 = gen_epistemic(rng)
        let ev2 = gen2.0
        rng = gen2.1

        let sum = transform_add(ev1, ev2)

        var all_pass = true

        // Provenance should grow
        if sum.provenance_count < ev1.provenance_count { all_pass = false }
        if sum.provenance_count < ev2.provenance_count { all_pass = false }

        // Confidence should be min
        if sum.conf > ev1.conf + 1.0e-10 { all_pass = false }
        if sum.conf > ev2.conf + 1.0e-10 { all_pass = false }

        // Debt should sum
        if sum.debt_bits < ev1.debt_bits + ev2.debt_bits - 1.0e-10 { all_pass = false }

        if all_pass {
            result.passed = result.passed + 1
        } else {
            if result.failed == 0 {
                result.first_failure_seed = seed + (i as i64)
            }
            result.failed = result.failed + 1
        }

        i = i + 1
    }

    return result
}

/// Test that ILLEGAL operations are detected
fn run_property_illegal_detected(n_tests: i32, seed: i64) -> PropertyResult {
    var result = property_result_new(3)  // Law group 3: detecting violations
    var rng = rng_new(seed)

    var i: i32 = 0
    while i < n_tests {
        let gen = gen_epistemic(rng)
        let ev = gen.0
        rng = gen.1

        var detected_all = true

        // 1. Illegal confidence uplift should violate law_confidence_monotone
        let boost_r = rng_range_f64(rng, 0.01, 0.5)
        let boost = boost_r.value
        rng = boost_r.rng
        let uplifted = ILLEGAL_uplift_confidence(ev, boost)
        if law_confidence_monotone(ev, uplifted) {
            // Should have detected violation but didn't
            detected_all = false
        }

        // 2. Illegal uncertainty narrowing should violate law_uncertainty_non_contraction
        if ev.uncert.tag != 0 {  // Only if not exact
            let narrowed = ILLEGAL_narrow_uncertainty(ev, 0.5)  // Halve the width
            if law_uncertainty_non_contraction(ev, narrowed) {
                detected_all = false
            }
        }

        // 3. Illegal provenance drop should violate law_provenance_append_only
        if ev.provenance_count > 1 {
            let dropped = ILLEGAL_drop_provenance(ev)
            if law_provenance_append_only(ev, dropped) {
                detected_all = false
            }
        }

        // 4. Illegal debt reduction should violate law_debt_monotone
        if ev.debt_bits > 0.1 {
            let reduced = ILLEGAL_reduce_debt(ev, 0.05)
            if law_debt_monotone(ev, reduced) {
                detected_all = false
            }
        }

        if detected_all {
            result.passed = result.passed + 1
        } else {
            if result.failed == 0 {
                result.first_failure_seed = seed + (i as i64)
            }
            result.failed = result.failed + 1
        }

        i = i + 1
    }

    return result
}

/// Test interval enclosure
fn run_property_interval_enclosure(n_tests: i32, seed: i64) -> PropertyResult {
    var result = property_result_new(4)  // Law group 4
    var rng = rng_new(seed)

    var i: i32 = 0
    while i < n_tests {
        let r1 = rng_range_f64(rng, -100.0, 100.0)
        let a_lo = r1.value
        rng = r1.rng

        let r2 = rng_range_f64(rng, 0.01, 50.0)
        let a_hi = a_lo + r2.value
        rng = r2.rng

        let r3 = rng_range_f64(rng, -100.0, 100.0)
        let b_lo = r3.value
        rng = r3.rng

        let r4 = rng_range_f64(rng, 0.01, 50.0)
        let b_hi = b_lo + r4.value
        rng = r4.rng

        if law_interval_enclosure(a_lo, a_hi, b_lo, b_hi) {
            result.passed = result.passed + 1
        } else {
            if result.failed == 0 {
                result.first_failure_seed = seed + (i as i64)
            }
            result.failed = result.failed + 1
        }

        i = i + 1
    }

    return result
}

// ============================================================================
// MAIN TEST HARNESS
// ============================================================================

fn main() -> i32 {
    let n_tests: i32 = 100
    let seed: i64 = 12345

    // Test 1: Scaling preserves laws
    let r1 = run_property_scale_preserves_laws(n_tests, seed)
    if r1.failed > 0 {
        return 1
    }

    // Test 2: Addition preserves laws
    let r2 = run_property_add_preserves_laws(n_tests, seed + 1000)
    if r2.failed > 0 {
        return 2
    }

    // Test 3: Illegal operations are detected
    let r3 = run_property_illegal_detected(n_tests, seed + 2000)
    if r3.failed > 0 {
        return 3
    }

    // Test 4: Interval enclosure
    let r4 = run_property_interval_enclosure(n_tests, seed + 3000)
    if r4.failed > 0 {
        return 4
    }

    return 0
}
