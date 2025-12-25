//! IEEE 1788-2015 Interval Arithmetic Semantics
//!
//! This module aligns Demetrios interval arithmetic with IEEE 1788 standard,
//! ensuring reproducibility and well-defined behavior for edge cases.
//!
//! Key IEEE 1788 concepts:
//!   - Empty interval: {} (result of impossible constraints)
//!   - Entire interval: [-∞, +∞] (maximum ignorance)
//!   - Exception handling for division by interval spanning zero
//!   - Decorations for tracking validity
//!
//! References:
//!   - IEEE 1788-2015: "Standard for Interval Arithmetic"
//!   - IEEE 1788.1-2017: "Simplified subset (Level 1)"

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
// IEEE 1788 INTERVAL REPRESENTATION
// ============================================================================

/// IEEE 1788 bare interval [a, b]
/// Special cases:
///   - Empty: is_empty = true
///   - Entire: lo = -inf, hi = +inf
///   - Point: lo == hi
struct Interval {
    lo: f64,
    hi: f64,
    is_empty: bool,
}

fn interval_empty() -> Interval {
    return Interval {
        lo: 0.0,
        hi: 0.0,
        is_empty: true,
    }
}

fn interval_entire() -> Interval {
    return Interval {
        lo: 0.0 - 1.0e308,
        hi: 1.0e308,
        is_empty: false,
    }
}

fn interval_new(lo: f64, hi: f64) -> Interval {
    if lo > hi {
        // Invalid bounds -> empty
        return interval_empty()
    }
    return Interval {
        lo: lo,
        hi: hi,
        is_empty: false,
    }
}

fn interval_point(x: f64) -> Interval {
    return Interval {
        lo: x,
        hi: x,
        is_empty: false,
    }
}

fn interval_symmetric(center: f64, half_width: f64) -> Interval {
    return interval_new(center - half_width, center + half_width)
}

// ============================================================================
// IEEE 1788 DECORATIONS
// ============================================================================

/// Decorations track interval validity through computations
/// Per IEEE 1788 Section 9
/// 0 = ill (undefined/invalid)
/// 1 = trv (trivial - could be any value)
/// 2 = def (defined but possibly unbounded)
/// 3 = dac (defined and continuous)
/// 4 = com (common - bounded and continuous)
struct Decoration {
    value: i32,
}

fn dec_ill() -> Decoration { return Decoration { value: 0 } }  // Invalid
fn dec_trv() -> Decoration { return Decoration { value: 1 } }  // Trivial
fn dec_def() -> Decoration { return Decoration { value: 2 } }  // Defined
fn dec_dac() -> Decoration { return Decoration { value: 3 } }  // DAC
fn dec_com() -> Decoration { return Decoration { value: 4 } }  // Common

/// Decorated interval: interval + decoration
struct DecoratedInterval {
    interval: Interval,
    decoration: Decoration,
}

fn decorated_new(interval: Interval, dec: Decoration) -> DecoratedInterval {
    return DecoratedInterval {
        interval: interval,
        decoration: dec,
    }
}

fn decorated_common(lo: f64, hi: f64) -> DecoratedInterval {
    let interval = interval_new(lo, hi)
    if interval.is_empty {
        return DecoratedInterval { interval: interval, decoration: dec_ill() }
    }
    return DecoratedInterval { interval: interval, decoration: dec_com() }
}

/// Combine decorations (take minimum)
fn dec_combine(a: Decoration, b: Decoration) -> Decoration {
    if a.value < b.value {
        return a
    }
    return b
}

// ============================================================================
// IEEE 1788 BASIC OPERATIONS
// ============================================================================

/// Addition: [a, b] + [c, d] = [a+c, b+d]
fn interval_add(x: Interval, y: Interval) -> Interval {
    if x.is_empty || y.is_empty {
        return interval_empty()
    }
    return interval_new(x.lo + y.lo, x.hi + y.hi)
}

/// Subtraction: [a, b] - [c, d] = [a-d, b-c]
fn interval_sub(x: Interval, y: Interval) -> Interval {
    if x.is_empty || y.is_empty {
        return interval_empty()
    }
    return interval_new(x.lo - y.hi, x.hi - y.lo)
}

/// Multiplication: [a, b] × [c, d]
fn interval_mul(x: Interval, y: Interval) -> Interval {
    if x.is_empty || y.is_empty {
        return interval_empty()
    }

    // All four products
    let p1 = x.lo * y.lo
    let p2 = x.lo * y.hi
    let p3 = x.hi * y.lo
    let p4 = x.hi * y.hi

    let lo = min_f64(min_f64(p1, p2), min_f64(p3, p4))
    let hi = max_f64(max_f64(p1, p2), max_f64(p3, p4))

    return interval_new(lo, hi)
}

// ============================================================================
// IEEE 1788 DIVISION: THE CRITICAL OPERATION
// ============================================================================

/// Division result with exception handling
struct DivisionResult {
    result: Interval,
    exception: i32,    // 0=none, 1=div_by_zero, 2=entire_result
    decoration: Decoration,
}

/// Division: [a, b] / [c, d]
/// Per IEEE 1788, division by interval containing zero is well-defined
/// but requires careful handling:
///   - If 0 ∈ [c, d]: result may be unbounded or split
///   - If [c, d] = [0, 0]: result is empty (not infinity!)
fn interval_div(x: Interval, y: Interval) -> DivisionResult {
    if x.is_empty || y.is_empty {
        return DivisionResult {
            result: interval_empty(),
            exception: 0,
            decoration: dec_ill(),
        }
    }

    // Check if y contains zero
    let y_contains_zero = y.lo <= 0.0 && y.hi >= 0.0

    if y_contains_zero {
        // Special cases per IEEE 1788
        if y.lo == 0.0 && y.hi == 0.0 {
            // [c, d] = [0, 0]: division by zero interval -> empty
            return DivisionResult {
                result: interval_empty(),
                exception: 1,  // Division by zero
                decoration: dec_ill(),
            }
        }

        if y.lo == 0.0 {
            // [0, d] with d > 0: result is [a/d, +∞] or similar
            if x.lo >= 0.0 {
                return DivisionResult {
                    result: interval_new(x.lo / y.hi, 1.0e308),
                    exception: 2,  // Entire result
                    decoration: dec_def(),
                }
            } else if x.hi <= 0.0 {
                return DivisionResult {
                    result: interval_new(0.0 - 1.0e308, x.hi / y.hi),
                    exception: 2,
                    decoration: dec_def(),
                }
            }
        }

        if y.hi == 0.0 {
            // [c, 0] with c < 0: result is [-∞, a/c] or similar
            if x.lo >= 0.0 {
                return DivisionResult {
                    result: interval_new(0.0 - 1.0e308, x.lo / y.lo),
                    exception: 2,
                    decoration: dec_def(),
                }
            } else if x.hi <= 0.0 {
                return DivisionResult {
                    result: interval_new(x.hi / y.lo, 1.0e308),
                    exception: 2,
                    decoration: dec_def(),
                }
            }
        }

        // General case: 0 strictly inside [c, d]
        // Result is entire (two disconnected pieces reunited)
        return DivisionResult {
            result: interval_entire(),
            exception: 2,
            decoration: dec_trv(),
        }
    }

    // Normal division: 0 ∉ [c, d]
    // [a, b] / [c, d] = [a, b] × [1/d, 1/c]
    let inv_lo = 1.0 / y.hi
    let inv_hi = 1.0 / y.lo

    let p1 = x.lo * inv_lo
    let p2 = x.lo * inv_hi
    let p3 = x.hi * inv_lo
    let p4 = x.hi * inv_hi

    let lo = min_f64(min_f64(p1, p2), min_f64(p3, p4))
    let hi = max_f64(max_f64(p1, p2), max_f64(p3, p4))

    return DivisionResult {
        result: interval_new(lo, hi),
        exception: 0,
        decoration: dec_com(),
    }
}

// ============================================================================
// IEEE 1788 ELEMENTARY FUNCTIONS
// ============================================================================

/// Square root: sqrt([a, b])
fn interval_sqrt(x: Interval) -> DecoratedInterval {
    if x.is_empty {
        return decorated_new(interval_empty(), dec_ill())
    }

    if x.hi < 0.0 {
        // Entirely negative -> empty (no real square root)
        return decorated_new(interval_empty(), dec_ill())
    }

    let lo = max_f64(0.0, x.lo)
    return decorated_new(
        interval_new(sqrt_f64(lo), sqrt_f64(x.hi)),
        if x.lo >= 0.0 { dec_com() } else { dec_def() }
    )
}

/// Exponential: exp([a, b])
fn interval_exp(x: Interval) -> DecoratedInterval {
    if x.is_empty {
        return decorated_new(interval_empty(), dec_ill())
    }

    // exp is monotonically increasing
    return decorated_new(
        interval_new(exp(x.lo), exp(x.hi)),
        dec_com()
    )
}

/// Natural logarithm: log([a, b])
fn interval_log(x: Interval) -> DecoratedInterval {
    if x.is_empty {
        return decorated_new(interval_empty(), dec_ill())
    }

    if x.hi <= 0.0 {
        // Entirely non-positive -> undefined
        return decorated_new(interval_empty(), dec_ill())
    }

    var dec = dec_com()
    var lo_val: f64 = 0.0

    if x.lo <= 0.0 {
        // Interval includes non-positive -> clamp, degrade decoration
        lo_val = 0.0 - 1.0e308  // -infinity for log(0+)
        dec = dec_def()
    } else {
        lo_val = log(x.lo)
    }

    return decorated_new(
        interval_new(lo_val, log(x.hi)),
        dec
    )
}

// ============================================================================
// IEEE 1788 SET OPERATIONS
// ============================================================================

/// Intersection: [a, b] ∩ [c, d]
fn interval_intersect(x: Interval, y: Interval) -> Interval {
    if x.is_empty || y.is_empty {
        return interval_empty()
    }

    let lo = max_f64(x.lo, y.lo)
    let hi = min_f64(x.hi, y.hi)

    if lo > hi {
        return interval_empty()
    }
    return interval_new(lo, hi)
}

/// Hull (convex union): hull([a, b], [c, d])
fn interval_hull(x: Interval, y: Interval) -> Interval {
    if x.is_empty { return y }
    if y.is_empty { return x }

    return interval_new(
        min_f64(x.lo, y.lo),
        max_f64(x.hi, y.hi)
    )
}

// ============================================================================
// IEEE 1788 COMPARISONS
// ============================================================================

/// Check if interval is subset of another
fn interval_subset(x: Interval, y: Interval) -> bool {
    if x.is_empty { return true }  // Empty is subset of everything
    if y.is_empty { return false } // Non-empty not subset of empty

    return x.lo >= y.lo && x.hi <= y.hi
}

/// Check if intervals overlap
fn interval_overlaps(x: Interval, y: Interval) -> bool {
    if x.is_empty || y.is_empty { return false }

    return x.lo <= y.hi && y.lo <= x.hi
}

/// Width of interval
fn interval_width(x: Interval) -> f64 {
    if x.is_empty { return 0.0 }
    return x.hi - x.lo
}

/// Midpoint of interval
fn interval_mid(x: Interval) -> f64 {
    if x.is_empty { return 0.0 }
    return (x.lo + x.hi) / 2.0
}

/// Radius of interval
fn interval_rad(x: Interval) -> f64 {
    if x.is_empty { return 0.0 }
    return (x.hi - x.lo) / 2.0
}

// ============================================================================
// IEEE 1788 EXCEPTION HANDLING
// ============================================================================

/// Exception flags per IEEE 1788
struct IntervalException {
    undefined_operation: bool,   // e.g., sqrt of negative
    possibly_undefined: bool,    // e.g., sqrt of interval spanning 0
    division_by_zero: bool,
    overflow: bool,
}

fn exception_none() -> IntervalException {
    return IntervalException {
        undefined_operation: false,
        possibly_undefined: false,
        division_by_zero: false,
        overflow: false,
    }
}

fn exception_from_div(result: DivisionResult) -> IntervalException {
    return IntervalException {
        undefined_operation: result.exception == 1,
        possibly_undefined: result.exception == 2,
        division_by_zero: result.exception == 1,
        overflow: false,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_interval_add() -> bool {
    let a = interval_new(1.0, 2.0)
    let b = interval_new(3.0, 4.0)
    let c = interval_add(a, b)

    if abs_f64(c.lo - 4.0) > 0.001 { return false }
    if abs_f64(c.hi - 6.0) > 0.001 { return false }

    return true
}

fn test_interval_mul() -> bool {
    let a = interval_new(-2.0, 3.0)
    let b = interval_new(1.0, 2.0)
    let c = interval_mul(a, b)

    // Products: -2*1=-2, -2*2=-4, 3*1=3, 3*2=6
    // Min = -4, Max = 6
    if abs_f64(c.lo - (0.0 - 4.0)) > 0.001 { return false }
    if abs_f64(c.hi - 6.0) > 0.001 { return false }

    return true
}

fn test_interval_div_normal() -> bool {
    let a = interval_new(2.0, 4.0)
    let b = interval_new(1.0, 2.0)
    let result = interval_div(a, b)

    // [2,4] / [1,2] = [2/2, 4/1] = [1, 4]
    if result.exception != 0 { return false }
    if abs_f64(result.result.lo - 1.0) > 0.001 { return false }
    if abs_f64(result.result.hi - 4.0) > 0.001 { return false }

    return true
}

fn test_interval_div_by_zero() -> bool {
    let a = interval_new(1.0, 2.0)
    let b = interval_new(0.0, 0.0)
    let result = interval_div(a, b)

    // Division by [0,0] -> empty with exception
    if !result.result.is_empty { return false }
    if result.exception != 1 { return false }

    return true
}

fn test_interval_div_spanning_zero() -> bool {
    let a = interval_new(1.0, 2.0)
    let b = interval_new(-1.0, 1.0)  // Spans zero
    let result = interval_div(a, b)

    // Should be entire or have exception
    if result.exception != 2 { return false }

    return true
}

fn test_interval_sqrt() -> bool {
    let a = interval_new(4.0, 9.0)
    let result = interval_sqrt(a)

    if result.interval.is_empty { return false }
    if abs_f64(result.interval.lo - 2.0) > 0.001 { return false }
    if abs_f64(result.interval.hi - 3.0) > 0.001 { return false }

    // Negative interval -> empty
    let b = interval_new(-4.0, -1.0)
    let result2 = interval_sqrt(b)
    if !result2.interval.is_empty { return false }

    return true
}

fn test_decorated_operations() -> bool {
    let a = decorated_common(1.0, 2.0)
    let b = decorated_common(3.0, 4.0)

    if a.decoration.value != 4 { return false }  // com
    if b.decoration.value != 4 { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_interval_add() { return 1 }
    if !test_interval_mul() { return 2 }
    if !test_interval_div_normal() { return 3 }
    if !test_interval_div_by_zero() { return 4 }
    if !test_interval_div_spanning_zero() { return 5 }
    if !test_interval_sqrt() { return 6 }
    if !test_decorated_operations() { return 7 }

    return 0
}
