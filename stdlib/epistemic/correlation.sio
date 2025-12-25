//! Correlation and Covariance Module for GUM-Compliant Uncertainty
//!
//! Implements GUM Equations 13-14 for correlated inputs.
//!
//! GUM Equation 10 (independent):
//!   u_c²(y) = Σᵢ (∂f/∂xᵢ)² u²(xᵢ)
//!
//! GUM Equation 13 (general):
//!   u_c²(y) = Σᵢ Σⱼ (∂f/∂xᵢ)(∂f/∂xⱼ) u(xᵢ,xⱼ)
//!
//! GUM Equation 14 (with correlation coefficient):
//!   u_c²(y) = Σᵢ (∂f/∂xᵢ)² u²(xᵢ)
//!           + 2 Σᵢ Σⱼ>ᵢ (∂f/∂xᵢ)(∂f/∂xⱼ) u(xᵢ)u(xⱼ)r(xᵢ,xⱼ)
//!
//! KEY CONCEPTS:
//!
//! 1. VarID: Token identifying a source of uncertainty. When two values
//!    share a VarID, they are correlated (possibly perfectly).
//!
//! 2. Covariance: u(x,y) = u(x)·u(y)·r(x,y) where r is correlation [-1,1]
//!
//! 3. Common sources: If x and y share a VarID with the same sensitivity,
//!    their correlation is +1 for that component.

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

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

// ============================================================================
// VARID: SOURCE IDENTIFIER FOR SHARED INFLUENCES
// ============================================================================

// VarID identifies a source of uncertainty. When two quantities share
// a VarID, they are correlated through that common source.
//
// Example: If weight_kg and BMI both depend on the same scale measurement,
// they share a VarID and their uncertainties must include covariance.

struct VarID {
    id: i64,            // Unique identifier for this source
    sensitivity: f64,   // Sensitivity coefficient (partial derivative)
    uncertainty: f64,   // Standard uncertainty from this source
}

// Global counter for VarID generation (using function-local for simplicity)
fn next_var_id(base: i64) -> i64 {
    return base + 1
}

fn var_id_new(id: i64, sensitivity: f64, uncertainty: f64) -> VarID {
    return VarID {
        id: id,
        sensitivity: sensitivity,
        uncertainty: uncertainty,
    }
}

// ============================================================================
// CORRELATED VALUE: VALUE WITH TRACKED CORRELATIONS
// ============================================================================

// Maximum number of tracked correlation sources per value
// (In a real implementation, this would be a Vec)
struct CorrelatedValue {
    value: f64,
    total_u: f64,       // Total combined uncertainty
    conf: f64,          // Confidence (Channel B)

    // Up to 4 tracked sources (simplified - would be Vec in real impl)
    source_count: i32,

    // Source 1
    s1_id: i64,
    s1_sens: f64,
    s1_u: f64,

    // Source 2
    s2_id: i64,
    s2_sens: f64,
    s2_u: f64,

    // Source 3
    s3_id: i64,
    s3_sens: f64,
    s3_u: f64,

    // Source 4
    s4_id: i64,
    s4_sens: f64,
    s4_u: f64,

    // Residual uncertainty (from sources not tracked)
    residual_u2: f64,   // variance
}

fn correlated_empty() -> CorrelatedValue {
    return CorrelatedValue {
        value: 0.0,
        total_u: 0.0,
        conf: 1.0,
        source_count: 0,
        s1_id: 0, s1_sens: 0.0, s1_u: 0.0,
        s2_id: 0, s2_sens: 0.0, s2_u: 0.0,
        s3_id: 0, s3_sens: 0.0, s3_u: 0.0,
        s4_id: 0, s4_sens: 0.0, s4_u: 0.0,
        residual_u2: 0.0,
    }
}

// Create a correlated value from a primary measurement source
fn correlated_from_source(value: f64, var_id: i64, uncertainty: f64, conf: f64) -> CorrelatedValue {
    return CorrelatedValue {
        value: value,
        total_u: uncertainty,
        conf: conf,
        source_count: 1,
        s1_id: var_id,
        s1_sens: 1.0,  // Sensitivity = 1 for primary source
        s1_u: uncertainty,
        s2_id: 0, s2_sens: 0.0, s2_u: 0.0,
        s3_id: 0, s3_sens: 0.0, s3_u: 0.0,
        s4_id: 0, s4_sens: 0.0, s4_u: 0.0,
        residual_u2: 0.0,
    }
}

// Create a correlated value with no tracked sources (independent)
fn correlated_independent(value: f64, uncertainty: f64, conf: f64) -> CorrelatedValue {
    return CorrelatedValue {
        value: value,
        total_u: uncertainty,
        conf: conf,
        source_count: 0,
        s1_id: 0, s1_sens: 0.0, s1_u: 0.0,
        s2_id: 0, s2_sens: 0.0, s2_u: 0.0,
        s3_id: 0, s3_sens: 0.0, s3_u: 0.0,
        s4_id: 0, s4_sens: 0.0, s4_u: 0.0,
        residual_u2: uncertainty * uncertainty,
    }
}

// ============================================================================
// COVARIANCE COMPUTATION
// ============================================================================

// Get sensitivity and uncertainty for a source index
fn get_source_data(cv: CorrelatedValue, idx: i32) -> (i64, f64, f64) {
    if idx == 0 { return (cv.s1_id, cv.s1_sens, cv.s1_u) }
    if idx == 1 { return (cv.s2_id, cv.s2_sens, cv.s2_u) }
    if idx == 2 { return (cv.s3_id, cv.s3_sens, cv.s3_u) }
    if idx == 3 { return (cv.s4_id, cv.s4_sens, cv.s4_u) }
    return (0, 0.0, 0.0)
}

// Compute covariance u(x,y) between two correlated values
// Based on shared sources: u(x,y) = Σₖ cₓₖ·cᵧₖ·u²ₖ
// where cₓₖ is the sensitivity of x to source k
fn covariance(a: CorrelatedValue, b: CorrelatedValue) -> f64 {
    var cov = 0.0

    // Check each source in a against each source in b
    var i = 0
    while i < a.source_count {
        let src_a = get_source_data(a, i)
        let id_a = src_a.0
        let sens_a = src_a.1
        let u_a = src_a.2

        var j = 0
        while j < b.source_count {
            let src_b = get_source_data(b, j)
            let id_b = src_b.0
            let sens_b = src_b.1
            let u_b = src_b.2

            // If same source ID, they're correlated through this source
            if id_a == id_b && id_a != 0 {
                // Covariance contribution = c_a · c_b · u²
                // Since both have same source, u_a and u_b should be equal
                // We use the geometric mean just in case
                let u_shared = sqrt_f64(u_a * u_b)
                cov = cov + sens_a * sens_b * u_shared * u_shared
            }
            j = j + 1
        }
        i = i + 1
    }

    return cov
}

// Correlation coefficient r(x,y) = u(x,y) / (u(x)·u(y))
fn correlation_coefficient(a: CorrelatedValue, b: CorrelatedValue) -> f64 {
    let cov = covariance(a, b)
    let denom = a.total_u * b.total_u

    if abs_f64(denom) < 1.0e-15 {
        return 0.0
    }

    let r = cov / denom

    // Clamp to [-1, 1]
    if r < -1.0 { return -1.0 }
    if r > 1.0 { return 1.0 }
    return r
}

// ============================================================================
// GUM EQUATION 14: ADDITION WITH CORRELATION
// ============================================================================

// Add two correlated values: y = a + b
// GUM Eq 14: u²(y) = u²(a) + u²(b) + 2·u(a,b)
//          = u²(a) + u²(b) + 2·u(a)·u(b)·r(a,b)
fn add_correlated(a: CorrelatedValue, b: CorrelatedValue) -> CorrelatedValue {
    let value = a.value + b.value
    let conf = min_f64(a.conf, b.conf)

    // GUM Equation 14 for addition (∂y/∂a = 1, ∂y/∂b = 1)
    let cov = covariance(a, b)
    let u2 = a.total_u * a.total_u + b.total_u * b.total_u + 2.0 * cov
    let total_u = sqrt_f64(u2)

    // Merge source tracking (simplified: take union up to capacity)
    var result = correlated_empty()
    result.value = value
    result.total_u = total_u
    result.conf = conf

    // Copy sources from a
    result.source_count = a.source_count
    result.s1_id = a.s1_id
    result.s1_sens = a.s1_sens  // Sensitivity unchanged for addition
    result.s1_u = a.s1_u
    result.s2_id = a.s2_id
    result.s2_sens = a.s2_sens
    result.s2_u = a.s2_u
    result.s3_id = a.s3_id
    result.s3_sens = a.s3_sens
    result.s3_u = a.s3_u
    result.s4_id = a.s4_id
    result.s4_sens = a.s4_sens
    result.s4_u = a.s4_u

    // Add sources from b (if room and not already present)
    // For simplicity, just add to residual
    result.residual_u2 = a.residual_u2 + b.residual_u2 + b.total_u * b.total_u

    return result
}

// Subtract two correlated values: y = a - b
// GUM Eq 14: u²(y) = u²(a) + u²(b) - 2·u(a,b)
fn sub_correlated(a: CorrelatedValue, b: CorrelatedValue) -> CorrelatedValue {
    let value = a.value - b.value
    let conf = min_f64(a.conf, b.conf)

    // GUM Equation 14 for subtraction (∂y/∂a = 1, ∂y/∂b = -1)
    let cov = covariance(a, b)
    var u2 = a.total_u * a.total_u + b.total_u * b.total_u - 2.0 * cov
    if u2 < 0.0 { u2 = 0.0 }  // Numerical safety
    let total_u = sqrt_f64(u2)

    var result = correlated_empty()
    result.value = value
    result.total_u = total_u
    result.conf = conf
    result.source_count = a.source_count
    result.s1_id = a.s1_id
    result.s1_sens = a.s1_sens
    result.s1_u = a.s1_u
    result.s2_id = a.s2_id
    result.s2_sens = a.s2_sens
    result.s2_u = a.s2_u
    result.residual_u2 = a.residual_u2 + b.residual_u2

    return result
}

// ============================================================================
// GUM EQUATION 14: MULTIPLICATION WITH CORRELATION
// ============================================================================

// Multiply two correlated values: y = a * b
// For y = a·b: ∂y/∂a = b, ∂y/∂b = a
// u²(y) = b²·u²(a) + a²·u²(b) + 2·a·b·u(a,b)
fn mul_correlated(a: CorrelatedValue, b: CorrelatedValue) -> CorrelatedValue {
    let value = a.value * b.value
    let conf = min_f64(a.conf, b.conf)

    let cov = covariance(a, b)
    let u2 = b.value * b.value * a.total_u * a.total_u
           + a.value * a.value * b.total_u * b.total_u
           + 2.0 * a.value * b.value * cov
    let total_u = sqrt_f64(abs_f64(u2))

    var result = correlated_empty()
    result.value = value
    result.total_u = total_u
    result.conf = conf

    // Update sensitivities: new_sens_k = a·sens_b_k + b·sens_a_k
    if a.source_count > 0 {
        result.source_count = a.source_count
        result.s1_id = a.s1_id
        result.s1_sens = b.value * a.s1_sens
        result.s1_u = a.s1_u
    }

    result.residual_u2 = a.residual_u2 * b.value * b.value
                       + b.residual_u2 * a.value * a.value

    return result
}

// ============================================================================
// SCALE BY CONSTANT (sensitivity scaling)
// ============================================================================

fn scale_correlated(a: CorrelatedValue, k: f64) -> CorrelatedValue {
    return CorrelatedValue {
        value: k * a.value,
        total_u: abs_f64(k) * a.total_u,
        conf: a.conf,
        source_count: a.source_count,
        s1_id: a.s1_id,
        s1_sens: k * a.s1_sens,  // Scale sensitivity by k
        s1_u: a.s1_u,
        s2_id: a.s2_id,
        s2_sens: k * a.s2_sens,
        s2_u: a.s2_u,
        s3_id: a.s3_id,
        s3_sens: k * a.s3_sens,
        s3_u: a.s3_u,
        s4_id: a.s4_id,
        s4_sens: k * a.s4_sens,
        s4_u: a.s4_u,
        residual_u2: k * k * a.residual_u2,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_independent_addition() -> bool {
    // Two independent measurements, uncorrelated
    let a = correlated_independent(10.0, 1.0, 0.9)
    let b = correlated_independent(20.0, 2.0, 0.85)

    let sum = add_correlated(a, b)

    // Value should be 30
    if abs_f64(sum.value - 30.0) > 0.001 { return false }

    // Independent: u² = 1² + 2² = 5, u = √5 ≈ 2.236
    let expected_u = sqrt_f64(5.0)
    if abs_f64(sum.total_u - expected_u) > 0.01 { return false }

    // Confidence = min(0.9, 0.85) = 0.85
    if abs_f64(sum.conf - 0.85) > 0.001 { return false }

    return true
}

fn test_correlated_addition() -> bool {
    // Two measurements from the same source (perfectly correlated)
    let shared_id: i64 = 42
    let a = correlated_from_source(10.0, shared_id, 1.0, 0.9)
    let b = correlated_from_source(20.0, shared_id, 2.0, 0.85)

    // Covariance should be sens_a * sens_b * min(u_a, u_b)²
    // With same source: cov = 1 * 1 * 1 * 1 = 1
    let cov = covariance(a, b)

    // Since both track same source with same u (geometric mean of 1 and 2 ≈ 1.41)
    // cov ≈ 1 * 1 * (√(1*2))² = √2² = 2
    // Actually cov = sens_a * sens_b * u_shared² where u_shared = √(1*2)
    // So cov = 1 * 1 * (√2)² = 2

    let sum = add_correlated(a, b)

    // Value should be 30
    if abs_f64(sum.value - 30.0) > 0.001 { return false }

    // Correlated: u² = 1 + 4 + 2*cov, where cov ≈ 2
    // u² ≈ 1 + 4 + 4 = 9, u = 3
    // This is LARGER than independent case (√5 ≈ 2.24)
    // Positive correlation increases uncertainty
    if sum.total_u < 2.2 { return false }  // Must be at least √5

    return true
}

fn test_correlated_subtraction() -> bool {
    // Subtracting correlated values REDUCES uncertainty
    // Classic example: measuring difference of two things with same instrument

    let shared_id: i64 = 100
    let a = correlated_from_source(15.0, shared_id, 1.0, 0.9)
    let b = correlated_from_source(10.0, shared_id, 1.0, 0.9)

    let diff = sub_correlated(a, b)

    // Value should be 5
    if abs_f64(diff.value - 5.0) > 0.001 { return false }

    // For perfect correlation (same source, same u):
    // u² = 1 + 1 - 2*1 = 0  (uncertainty cancels!)
    // In practice with our geometric mean: u² ≈ 1 + 1 - 2*1 = 0
    // Should be much less than independent case (√2 ≈ 1.41)
    if diff.total_u > 0.5 { return false }  // Should be near zero

    return true
}

fn test_correlation_coefficient() -> bool {
    let shared_id: i64 = 200
    let a = correlated_from_source(10.0, shared_id, 2.0, 0.9)
    let b = correlated_from_source(20.0, shared_id, 2.0, 0.85)

    let r = correlation_coefficient(a, b)

    // Same source with same uncertainty should give r = 1
    if r < 0.99 { return false }

    // Independent values should give r = 0
    let c = correlated_independent(30.0, 3.0, 0.8)
    let r2 = correlation_coefficient(a, c)
    if abs_f64(r2) > 0.01 { return false }

    return true
}

fn test_scaling_preserves_source() -> bool {
    let shared_id: i64 = 300
    let a = correlated_from_source(10.0, shared_id, 1.0, 0.9)

    // Scale by 2
    let scaled = scale_correlated(a, 2.0)

    // Value should double
    if abs_f64(scaled.value - 20.0) > 0.001 { return false }

    // Uncertainty should double
    if abs_f64(scaled.total_u - 2.0) > 0.001 { return false }

    // Source should still be tracked with scaled sensitivity
    if scaled.s1_id != shared_id { return false }
    if abs_f64(scaled.s1_sens - 2.0) > 0.001 { return false }

    return true
}

fn test_multiplication_correlation() -> bool {
    let shared_id: i64 = 400
    let a = correlated_from_source(10.0, shared_id, 1.0, 0.9)
    let b = correlated_from_source(5.0, shared_id, 0.5, 0.85)

    let prod = mul_correlated(a, b)

    // Value should be 50
    if abs_f64(prod.value - 50.0) > 0.001 { return false }

    // Confidence = min(0.9, 0.85) = 0.85
    if abs_f64(prod.conf - 0.85) > 0.001 { return false }

    // Uncertainty should include correlation term
    // Independent: u_rel² = (1/10)² + (0.5/5)² = 0.01 + 0.01 = 0.02
    // u = 50 * √0.02 ≈ 7.07
    // With correlation: extra positive term, should be larger
    if prod.total_u < 5.0 { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_independent_addition() { return 1 }
    if !test_correlated_addition() { return 2 }
    if !test_correlated_subtraction() { return 3 }
    if !test_correlation_coefficient() { return 4 }
    if !test_scaling_preserves_source() { return 5 }
    if !test_multiplication_correlation() { return 6 }

    return 0
}
