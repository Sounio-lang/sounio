// epistemic_dual.d - Uncertainty-Aware Automatic Differentiation
//
// Extends dual numbers with epistemic uncertainty tracking.
// Each value carries:
//   - val: primal value f(x)
//   - dot: tangent (derivative) f'(x)
//   - unc: epistemic uncertainty in value
//   - unc_dot: uncertainty in derivative
//
// This enables:
//   1. Automatic derivative computation (standard AD)
//   2. Uncertainty propagation through computations
//   3. Sensitivity analysis (how uncertainty affects derivatives)
//   4. Confidence tracking for scientific computing
//
// Mathematical Foundation:
//   If x = x₀ ± σₓ (value with uncertainty), then:
//   f(x) ≈ f(x₀) ± |f'(x₀)| * σₓ  (linear error propagation)
//   f'(x) ≈ f'(x₀) ± |f''(x₀)| * σₓ  (derivative uncertainty)
//
// Reference: "Error Analysis" by J.R. Taylor
//            "Automatic Differentiation of Algorithms" by Griewank & Walther

// ============================================================================
// HELPER FUNCTIONS (from dual.d)
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut y = x
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    return y
}

fn exp_f64(x: f64) -> f64 {
    if x > 20.0 { return exp_f64(x / 2.0) * exp_f64(x / 2.0) }
    if x < 0.0 - 20.0 { return 1.0 / exp_f64(0.0 - x) }
    let mut sum = 1.0
    let mut term = 1.0
    term = term * x / 1.0
    sum = sum + term
    term = term * x / 2.0
    sum = sum + term
    term = term * x / 3.0
    sum = sum + term
    term = term * x / 4.0
    sum = sum + term
    term = term * x / 5.0
    sum = sum + term
    term = term * x / 6.0
    sum = sum + term
    term = term * x / 7.0
    sum = sum + term
    term = term * x / 8.0
    sum = sum + term
    term = term * x / 9.0
    sum = sum + term
    term = term * x / 10.0
    sum = sum + term
    term = term * x / 11.0
    sum = sum + term
    term = term * x / 12.0
    sum = sum + term
    term = term * x / 13.0
    sum = sum + term
    term = term * x / 14.0
    sum = sum + term
    term = term * x / 15.0
    sum = sum + term
    return sum
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1000000.0 }
    let mut val = x
    let mut k = 0.0
    let e = 2.718281828459045
    while val > e { val = val / e; k = k + 1.0 }
    while val < 1.0 / e { val = val * e; k = k - 1.0 }
    let u = (val - 1.0) / (val + 1.0)
    let u2 = u * u
    let mut sum = u
    let mut term = u
    term = term * u2; sum = sum + term / 3.0
    term = term * u2; sum = sum + term / 5.0
    term = term * u2; sum = sum + term / 7.0
    term = term * u2; sum = sum + term / 9.0
    term = term * u2; sum = sum + term / 11.0
    term = term * u2; sum = sum + term / 13.0
    term = term * u2; sum = sum + term / 15.0
    term = term * u2; sum = sum + term / 17.0
    term = term * u2; sum = sum + term / 19.0
    return 2.0 * sum + k
}

fn pow_f64(x: f64, n: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    return exp_f64(n * ln_f64(x))
}

fn sin_f64(x: f64) -> f64 {
    let pi = 3.141592653589793
    let mut y = x
    while y > pi { y = y - 2.0 * pi }
    while y < 0.0 - pi { y = y + 2.0 * pi }
    let x2 = y * y
    let mut sum = y
    let mut term = y
    term = term * (0.0 - x2) / (2.0 * 3.0); sum = sum + term
    term = term * (0.0 - x2) / (4.0 * 5.0); sum = sum + term
    term = term * (0.0 - x2) / (6.0 * 7.0); sum = sum + term
    term = term * (0.0 - x2) / (8.0 * 9.0); sum = sum + term
    term = term * (0.0 - x2) / (10.0 * 11.0); sum = sum + term
    term = term * (0.0 - x2) / (12.0 * 13.0); sum = sum + term
    return sum
}

fn cos_f64(x: f64) -> f64 {
    let pi = 3.141592653589793
    let mut y = x
    while y > pi { y = y - 2.0 * pi }
    while y < 0.0 - pi { y = y + 2.0 * pi }
    let x2 = y * y
    let mut sum = 1.0
    let mut term = 1.0
    term = term * (0.0 - x2) / (1.0 * 2.0); sum = sum + term
    term = term * (0.0 - x2) / (3.0 * 4.0); sum = sum + term
    term = term * (0.0 - x2) / (5.0 * 6.0); sum = sum + term
    term = term * (0.0 - x2) / (7.0 * 8.0); sum = sum + term
    term = term * (0.0 - x2) / (9.0 * 10.0); sum = sum + term
    term = term * (0.0 - x2) / (11.0 * 12.0); sum = sum + term
    return sum
}

fn tanh_f64(x: f64) -> f64 {
    let e2x = exp_f64(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)
}

// ============================================================================
// EPISTEMIC DUAL NUMBER TYPE
// ============================================================================
// Epistemic dual: (val, dot, unc, unc_dot)
// - val: primal value f(x)
// - dot: tangent f'(x)
// - unc: uncertainty in value σ_f
// - unc_dot: uncertainty in derivative σ_{f'}

struct EpistemicDual {
    val: f64,      // Primal value
    dot: f64,      // Derivative (tangent)
    unc: f64,      // Uncertainty in value (standard deviation)
    unc_dot: f64   // Uncertainty in derivative
}

// Create a constant (no uncertainty, no derivative)
fn edual_const(val: f64) -> EpistemicDual {
    return EpistemicDual { val: val, dot: 0.0, unc: 0.0, unc_dot: 0.0 }
}

// Create a variable with derivative=1, no uncertainty
fn edual_var(val: f64) -> EpistemicDual {
    return EpistemicDual { val: val, dot: 1.0, unc: 0.0, unc_dot: 0.0 }
}

// Create a variable with uncertainty (derivative=1)
fn edual_var_uncertain(val: f64, uncertainty: f64) -> EpistemicDual {
    return EpistemicDual { val: val, dot: 1.0, unc: uncertainty, unc_dot: 0.0 }
}

// Create from measurement with relative uncertainty
fn edual_measurement(val: f64, rel_unc: f64) -> EpistemicDual {
    return EpistemicDual { val: val, dot: 1.0, unc: abs_f64(val) * rel_unc, unc_dot: 0.0 }
}

// Full constructor
fn edual_new(val: f64, dot: f64, unc: f64, unc_dot: f64) -> EpistemicDual {
    return EpistemicDual { val: val, dot: dot, unc: unc, unc_dot: unc_dot }
}

// ============================================================================
// UNCERTAINTY PROPAGATION
// ============================================================================

// Combine independent uncertainties: σ_combined = sqrt(σ₁² + σ₂²)
fn combine_unc(u1: f64, u2: f64) -> f64 {
    return sqrt_f64(u1 * u1 + u2 * u2)
}

// Combine with correlation: σ = sqrt(σ₁² + σ₂² + 2ρσ₁σ₂)
fn combine_unc_corr(u1: f64, u2: f64, rho: f64) -> f64 {
    return sqrt_f64(u1*u1 + u2*u2 + 2.0*rho*u1*u2)
}

// Relative uncertainty
fn relative_unc(ed: EpistemicDual) -> f64 {
    if abs_f64(ed.val) < 0.000000000000001 { return 0.0 }
    return ed.unc / abs_f64(ed.val)
}

// Confidence (inverse of relative uncertainty, capped at 1.0)
fn confidence(ed: EpistemicDual) -> f64 {
    let rel = relative_unc(ed)
    if rel < 0.000000000000001 { return 1.0 }
    let c = 1.0 / (1.0 + rel)
    if c > 1.0 { return 1.0 }
    return c
}

// ============================================================================
// ARITHMETIC OPERATIONS WITH UNCERTAINTY PROPAGATION
// ============================================================================

fn edual_add(a: EpistemicDual, b: EpistemicDual) -> EpistemicDual {
    // f = a + b
    // f' = a' + b'
    // σ_f = sqrt(σ_a² + σ_b²)  (assuming independence)
    // σ_{f'} = sqrt(σ_{a'}² + σ_{b'}²)
    return EpistemicDual {
        val: a.val + b.val,
        dot: a.dot + b.dot,
        unc: combine_unc(a.unc, b.unc),
        unc_dot: combine_unc(a.unc_dot, b.unc_dot)
    }
}

fn edual_sub(a: EpistemicDual, b: EpistemicDual) -> EpistemicDual {
    // Same uncertainty propagation as addition
    return EpistemicDual {
        val: a.val - b.val,
        dot: a.dot - b.dot,
        unc: combine_unc(a.unc, b.unc),
        unc_dot: combine_unc(a.unc_dot, b.unc_dot)
    }
}

fn edual_mul(a: EpistemicDual, b: EpistemicDual) -> EpistemicDual {
    // f = a * b
    // f' = a'*b + a*b'  (product rule)
    // σ_f = |f| * sqrt((σ_a/a)² + (σ_b/b)²)  (relative uncertainties add)
    // σ_{f'} = sqrt((b*σ_{a'})² + (a*σ_{b'})² + (a'*σ_b)² + (b'*σ_a)²)

    let val = a.val * b.val
    let dot = a.dot * b.val + a.val * b.dot

    // Value uncertainty (linear approximation)
    let unc_a_contrib = abs_f64(b.val) * a.unc
    let unc_b_contrib = abs_f64(a.val) * b.unc
    let unc = combine_unc(unc_a_contrib, unc_b_contrib)

    // Derivative uncertainty (all four terms)
    let ud1 = abs_f64(b.val) * a.unc_dot
    let ud2 = abs_f64(a.val) * b.unc_dot
    let ud3 = abs_f64(a.dot) * b.unc
    let ud4 = abs_f64(b.dot) * a.unc
    let unc_dot = sqrt_f64(ud1*ud1 + ud2*ud2 + ud3*ud3 + ud4*ud4)

    return EpistemicDual { val: val, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_div(a: EpistemicDual, b: EpistemicDual) -> EpistemicDual {
    // f = a / b
    // f' = (a'*b - a*b') / b²  (quotient rule)
    // σ_f = |f| * sqrt((σ_a/a)² + (σ_b/b)²)

    let b2 = b.val * b.val
    let val = a.val / b.val
    let dot = (a.dot * b.val - a.val * b.dot) / b2

    // Value uncertainty
    let unc_a_contrib = a.unc / abs_f64(b.val)
    let unc_b_contrib = abs_f64(a.val) * b.unc / b2
    let unc = combine_unc(unc_a_contrib, unc_b_contrib)

    // Derivative uncertainty (more complex for quotient)
    let ud1 = a.unc_dot / abs_f64(b.val)
    let ud2 = abs_f64(a.val) * b.unc_dot / b2
    let ud3 = abs_f64(a.dot) * b.unc / b2
    let ud4 = abs_f64(b.dot) * a.unc / b2
    let ud5 = 2.0 * abs_f64(dot) * b.unc / abs_f64(b.val)  // Second order
    let unc_dot = sqrt_f64(ud1*ud1 + ud2*ud2 + ud3*ud3 + ud4*ud4 + ud5*ud5)

    return EpistemicDual { val: val, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_neg(a: EpistemicDual) -> EpistemicDual {
    return EpistemicDual {
        val: 0.0 - a.val,
        dot: 0.0 - a.dot,
        unc: a.unc,        // Uncertainty unchanged by negation
        unc_dot: a.unc_dot
    }
}

fn edual_scale(a: EpistemicDual, s: f64) -> EpistemicDual {
    // f = s * a (s is exact constant)
    // f' = s * a'
    // σ_f = |s| * σ_a
    return EpistemicDual {
        val: a.val * s,
        dot: a.dot * s,
        unc: abs_f64(s) * a.unc,
        unc_dot: abs_f64(s) * a.unc_dot
    }
}

fn edual_add_const(a: EpistemicDual, c: f64) -> EpistemicDual {
    // Adding constant doesn't change uncertainty
    return EpistemicDual {
        val: a.val + c,
        dot: a.dot,
        unc: a.unc,
        unc_dot: a.unc_dot
    }
}

// ============================================================================
// MATHEMATICAL FUNCTIONS WITH UNCERTAINTY PROPAGATION
// ============================================================================

fn edual_sqrt(a: EpistemicDual) -> EpistemicDual {
    // f = sqrt(a)
    // f' = a' / (2*sqrt(a))
    // f'' = -a' / (4*a^(3/2)) + a'' / (2*sqrt(a))
    // σ_f = |f'| * σ_a = σ_a / (2*sqrt(a))

    let v = sqrt_f64(a.val)
    let two_v = 2.0 * v
    let dot = a.dot / two_v

    // Value uncertainty
    let unc = a.unc / two_v

    // Derivative uncertainty: includes σ_a contribution to f'
    let unc_dot_1 = a.unc_dot / two_v
    let unc_dot_2 = abs_f64(a.dot) * a.unc / (2.0 * a.val * v)  // From chain rule
    let unc_dot = combine_unc(unc_dot_1, unc_dot_2)

    return EpistemicDual { val: v, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_exp(a: EpistemicDual) -> EpistemicDual {
    // f = exp(a)
    // f' = a' * exp(a)
    // σ_f = |exp(a)| * σ_a = exp(a) * σ_a

    let e = exp_f64(a.val)
    let dot = a.dot * e

    // Value uncertainty
    let unc = e * a.unc

    // Derivative uncertainty
    let unc_dot_1 = e * a.unc_dot
    let unc_dot_2 = abs_f64(a.dot) * e * a.unc  // From exp'(a) * σ_a
    let unc_dot = combine_unc(unc_dot_1, unc_dot_2)

    return EpistemicDual { val: e, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_ln(a: EpistemicDual) -> EpistemicDual {
    // f = ln(a)
    // f' = a' / a
    // σ_f = σ_a / |a|

    let val = ln_f64(a.val)
    let dot = a.dot / a.val

    // Value uncertainty
    let unc = a.unc / abs_f64(a.val)

    // Derivative uncertainty
    let unc_dot_1 = a.unc_dot / abs_f64(a.val)
    let unc_dot_2 = abs_f64(a.dot) * a.unc / (a.val * a.val)
    let unc_dot = combine_unc(unc_dot_1, unc_dot_2)

    return EpistemicDual { val: val, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_pow(a: EpistemicDual, n: f64) -> EpistemicDual {
    // f = a^n
    // f' = n * a^(n-1) * a'
    // σ_f = |n * a^(n-1)| * σ_a

    let v = pow_f64(a.val, n)
    let df_da = n * pow_f64(a.val, n - 1.0)
    let dot = a.dot * df_da

    // Value uncertainty
    let unc = abs_f64(df_da) * a.unc

    // Derivative uncertainty (includes second derivative contribution)
    let d2f_da2 = n * (n - 1.0) * pow_f64(a.val, n - 2.0)
    let unc_dot_1 = abs_f64(df_da) * a.unc_dot
    let unc_dot_2 = abs_f64(a.dot * d2f_da2) * a.unc
    let unc_dot = combine_unc(unc_dot_1, unc_dot_2)

    return EpistemicDual { val: v, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_sin(a: EpistemicDual) -> EpistemicDual {
    // f = sin(a)
    // f' = cos(a) * a'
    // σ_f = |cos(a)| * σ_a

    let s = sin_f64(a.val)
    let c = cos_f64(a.val)
    let dot = a.dot * c

    // Value uncertainty
    let unc = abs_f64(c) * a.unc

    // Derivative uncertainty (f'' = -sin(a))
    let unc_dot_1 = abs_f64(c) * a.unc_dot
    let unc_dot_2 = abs_f64(a.dot * s) * a.unc  // From -sin(a) * a' * σ_a
    let unc_dot = combine_unc(unc_dot_1, unc_dot_2)

    return EpistemicDual { val: s, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_cos(a: EpistemicDual) -> EpistemicDual {
    // f = cos(a)
    // f' = -sin(a) * a'
    // σ_f = |sin(a)| * σ_a

    let s = sin_f64(a.val)
    let c = cos_f64(a.val)
    let dot = 0.0 - a.dot * s

    // Value uncertainty
    let unc = abs_f64(s) * a.unc

    // Derivative uncertainty
    let unc_dot_1 = abs_f64(s) * a.unc_dot
    let unc_dot_2 = abs_f64(a.dot * c) * a.unc
    let unc_dot = combine_unc(unc_dot_1, unc_dot_2)

    return EpistemicDual { val: c, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_tanh(a: EpistemicDual) -> EpistemicDual {
    // f = tanh(a)
    // f' = (1 - tanh²(a)) * a' = sech²(a) * a'
    // σ_f = sech²(a) * σ_a

    let t = tanh_f64(a.val)
    let sech2 = 1.0 - t * t
    let dot = a.dot * sech2

    // Value uncertainty
    let unc = abs_f64(sech2) * a.unc

    // Derivative uncertainty (f'' = -2*tanh*sech²)
    let unc_dot_1 = abs_f64(sech2) * a.unc_dot
    let unc_dot_2 = 2.0 * abs_f64(t * sech2 * a.dot) * a.unc
    let unc_dot = combine_unc(unc_dot_1, unc_dot_2)

    return EpistemicDual { val: t, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_sigmoid(a: EpistemicDual) -> EpistemicDual {
    // f = 1 / (1 + exp(-a)) = σ(a)
    // f' = σ(a) * (1 - σ(a)) * a'

    let s = 1.0 / (1.0 + exp_f64(0.0 - a.val))
    let ds = s * (1.0 - s)  // Derivative of sigmoid
    let dot = a.dot * ds

    // Value uncertainty
    let unc = abs_f64(ds) * a.unc

    // Derivative uncertainty (f'' = σ(1-σ)(1-2σ))
    let d2s = ds * (1.0 - 2.0 * s)
    let unc_dot_1 = abs_f64(ds) * a.unc_dot
    let unc_dot_2 = abs_f64(a.dot * d2s) * a.unc
    let unc_dot = combine_unc(unc_dot_1, unc_dot_2)

    return EpistemicDual { val: s, dot: dot, unc: unc, unc_dot: unc_dot }
}

fn edual_relu(a: EpistemicDual) -> EpistemicDual {
    // f = max(0, a)
    // f' = 1 if a > 0, else 0
    // Uncertainty at boundary is tricky - we preserve it

    if a.val > 0.0 {
        return EpistemicDual { val: a.val, dot: a.dot, unc: a.unc, unc_dot: a.unc_dot }
    }
    // Below zero, output is zero but uncertainty in "when we cross" matters
    // For conservative estimate, we keep some uncertainty
    return EpistemicDual { val: 0.0, dot: 0.0, unc: a.unc, unc_dot: 0.0 }
}

fn edual_abs(a: EpistemicDual) -> EpistemicDual {
    // f = |a|
    // Uncertainty unchanged by absolute value
    if a.val >= 0.0 {
        return a
    }
    return edual_neg(a)
}

// ============================================================================
// SENSITIVITY ANALYSIS
// ============================================================================

// Sensitivity: |∂f/∂x| * σ_x / |f| - how much uncertainty matters
fn sensitivity(ed: EpistemicDual) -> f64 {
    if abs_f64(ed.val) < 0.000000000000001 { return 0.0 }
    return abs_f64(ed.dot) * ed.unc / abs_f64(ed.val)
}

// Derivative-weighted uncertainty
fn derivative_weighted_unc(ed: EpistemicDual) -> f64 {
    return abs_f64(ed.dot) * ed.unc
}

// Total uncertainty (value + derivative contribution)
fn total_uncertainty(ed: EpistemicDual) -> f64 {
    return combine_unc(ed.unc, ed.unc_dot)
}

// ============================================================================
// COMPARISON AND UTILITIES
// ============================================================================

fn edual_max(a: EpistemicDual, b: EpistemicDual) -> EpistemicDual {
    if a.val >= b.val { return a }
    return b
}

fn edual_min(a: EpistemicDual, b: EpistemicDual) -> EpistemicDual {
    if a.val <= b.val { return a }
    return b
}

// Check if values are compatible within uncertainties
fn edual_compatible(a: EpistemicDual, b: EpistemicDual, sigma_mult: f64) -> i64 {
    let diff = abs_f64(a.val - b.val)
    let combined_unc = combine_unc(a.unc, b.unc)
    if diff <= sigma_mult * combined_unc { return 1 }
    return 0
}

// Print epistemic dual (for debugging)
fn edual_print(label: f64, ed: EpistemicDual) -> i64 {
    println("EpistemicDual:")
    println("  val = ")
    println(ed.val)
    println("  dot = ")
    println(ed.dot)
    println("  unc = ")
    println(ed.unc)
    println("  unc_dot = ")
    println(ed.unc_dot)
    println("  confidence = ")
    println(confidence(ed))
    return 0
}

// ============================================================================
// PHARMACOKINETICS EXAMPLE: Uncertainty in Clearance Calculation
// ============================================================================

// One-compartment PK: C(t) = (D/V) * exp(-CL/V * t)
fn pk_concentration(dose: EpistemicDual, volume: EpistemicDual, clearance: EpistemicDual, time: EpistemicDual) -> EpistemicDual {
    // C = (D/V) * exp(-(CL/V)*t)
    let d_over_v = edual_div(dose, volume)
    let cl_over_v = edual_div(clearance, volume)
    let neg_cl_v_t = edual_neg(edual_mul(cl_over_v, time))
    let exp_term = edual_exp(neg_cl_v_t)
    return edual_mul(d_over_v, exp_term)
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    println("=== Demetrios Epistemic Autodiff: Uncertainty-Aware AD ===")
    println("")

    // Test 1: Basic arithmetic with uncertainty
    println("Test 1: f(x) = x^2 with x = 3.0 +/- 0.1")
    let x1 = edual_var_uncertain(3.0, 0.1)
    let y1 = edual_mul(x1, x1)
    println("  f(x) = ")
    println(y1.val)
    println("  f'(x) = ")
    println(y1.dot)
    println("  sigma_f = ")
    println(y1.unc)
    println("  sigma_f' = ")
    println(y1.unc_dot)
    println("  Expected: f=9, f'=6, sigma_f~0.424 (sqrt(2)*3*0.1 for independent)")
    println("  Confidence = ")
    println(confidence(y1))
    println("")

    // Test 2: Chain rule with uncertainty
    println("Test 2: f(x) = exp(x) with x = 1.0 +/- 0.05")
    let x2 = edual_var_uncertain(1.0, 0.05)
    let y2 = edual_exp(x2)
    println("  f(x) = ")
    println(y2.val)
    println("  f'(x) = ")
    println(y2.dot)
    println("  sigma_f = ")
    println(y2.unc)
    println("  Expected: f=e~2.718, f'=e~2.718, sigma_f=e*0.05~0.136")
    println("")

    // Test 3: Pharmacokinetics with uncertainty
    println("Test 3: PK Concentration with measurement uncertainties")
    let dose = edual_measurement(100.0, 0.02)      // 100 mg +/- 2%
    let volume = edual_measurement(50.0, 0.05)     // 50 L +/- 5%
    let clearance = edual_measurement(5.0, 0.10)   // 5 L/h +/- 10%
    let time = edual_const(2.0)                    // 2 hours (exact)

    let conc = pk_concentration(dose, volume, clearance, time)
    println("  Dose = 100 mg +/- 2%")
    println("  Volume = 50 L +/- 5%")
    println("  Clearance = 5 L/h +/- 10%")
    println("  Time = 2 h")
    println("")
    println("  Concentration = ")
    println(conc.val)
    println("  mg/L")
    println("  Uncertainty = +/- ")
    println(conc.unc)
    println("  mg/L")
    println("  Relative uncertainty = ")
    println(relative_unc(conc) * 100.0)
    println("  %")
    println("  Confidence = ")
    println(confidence(conc))
    println("")

    // Test 4: Sensitivity analysis
    println("Test 4: Sensitivity analysis for sin(x) at x=pi/4")
    let pi = 3.141592653589793
    let x4 = edual_var_uncertain(pi / 4.0, 0.01)
    let y4 = edual_sin(x4)
    println("  f(x) = sin(x)")
    println("  x = pi/4 +/- 0.01")
    println("  f(x) = ")
    println(y4.val)
    println("  f'(x) = ")
    println(y4.dot)
    println("  sigma_f = ")
    println(y4.unc)
    println("  Sensitivity = ")
    println(sensitivity(y4))
    println("")

    // Test 5: Compatibility check
    println("Test 5: Checking if two measurements are compatible")
    let m1 = edual_measurement(10.0, 0.05)  // 10 +/- 5%
    let m2 = edual_measurement(10.3, 0.04)  // 10.3 +/- 4%
    let is_compat = edual_compatible(m1, m2, 2.0)
    println("  m1 = 10.0 +/- 5%")
    println("  m2 = 10.3 +/- 4%")
    println("  Compatible within 2-sigma? ")
    if is_compat == 1 {
        println("  YES")
    } else {
        println("  NO")
    }
    println("")

    // Validation
    // For x^2 with x=3+/-0.1: sigma_f = sqrt((3*0.1)^2 + (3*0.1)^2) = sqrt(2)*0.3 = 0.4243
    let err1 = abs_f64(y1.val - 9.0) + abs_f64(y1.dot - 6.0)
    let expected_unc = sqrt_f64(2.0) * 3.0 * 0.1
    let err2 = abs_f64(y1.unc - expected_unc)

    if err1 < 0.01 && err2 < 0.01 {
        println("TEST PASSED: Epistemic autodiff working correctly")
        return 0
    } else {
        println("TEST FAILED: Errors detected")
        println("  err1 = ")
        println(err1)
        println("  err2 = ")
        println(err2)
        return 1
    }
}
