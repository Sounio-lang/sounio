//! Automatic Differentiation for Sensitivity Coefficients
//!
//! GUM's law of propagation requires partial derivatives ("sensitivity coefficients"):
//!   u_c²(y) = Σᵢ (∂f/∂xᵢ)² u²(xᵢ) + 2 Σᵢ Σⱼ>ᵢ (∂f/∂xᵢ)(∂f/∂xⱼ) u(xᵢ,xⱼ)
//!
//! This module implements forward-mode AD using dual numbers:
//!   (value, derivative) pairs that propagate derivatives automatically.
//!
//! Benefits over numerical differentiation:
//!   - Exact derivatives (no truncation error)
//!   - Efficient (single forward pass)
//!   - Handles arbitrary compositions
//!
//! References:
//!   - GUM (JCGM 100:2008) Section 5.1: Law of propagation of uncertainty

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
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

// ============================================================================
// DUAL NUMBER: THE CORE OF FORWARD-MODE AD
// ============================================================================

// A dual number carries both value and derivative
// For computing ∂f/∂xᵢ: set xᵢ = Dual(value, 1.0), all others = Dual(value, 0.0)
struct Dual {
    val: f64,   // The value f(x)
    der: f64,   // The derivative ∂f/∂x
}

fn dual_const(x: f64) -> Dual {
    return Dual { val: x, der: 0.0 }
}

fn dual_var(x: f64) -> Dual {
    // Mark this as the variable we're differentiating with respect to
    return Dual { val: x, der: 1.0 }
}

// ============================================================================
// DUAL ARITHMETIC
// ============================================================================

// Addition: (a, a') + (b, b') = (a+b, a'+b')
fn dual_add(a: Dual, b: Dual) -> Dual {
    return Dual {
        val: a.val + b.val,
        der: a.der + b.der,
    }
}

// Subtraction: (a, a') - (b, b') = (a-b, a'-b')
fn dual_sub(a: Dual, b: Dual) -> Dual {
    return Dual {
        val: a.val - b.val,
        der: a.der - b.der,
    }
}

// Multiplication: (a, a') * (b, b') = (a*b, a*b' + a'*b)
fn dual_mul(a: Dual, b: Dual) -> Dual {
    return Dual {
        val: a.val * b.val,
        der: a.val * b.der + a.der * b.val,
    }
}

// Division: (a, a') / (b, b') = (a/b, (a'*b - a*b') / b²)
fn dual_div(a: Dual, b: Dual) -> Dual {
    if abs_f64(b.val) < 1.0e-15 {
        return Dual { val: 0.0, der: 0.0 }
    }
    return Dual {
        val: a.val / b.val,
        der: (a.der * b.val - a.val * b.der) / (b.val * b.val),
    }
}

// Negation: -(a, a') = (-a, -a')
fn dual_neg(a: Dual) -> Dual {
    return Dual { val: 0.0 - a.val, der: 0.0 - a.der }
}

// Scale by constant: k * (a, a') = (k*a, k*a')
fn dual_scale(k: f64, a: Dual) -> Dual {
    return Dual { val: k * a.val, der: k * a.der }
}

// ============================================================================
// DUAL TRANSCENDENTAL FUNCTIONS
// ============================================================================

// Exponential: exp(a, a') = (exp(a), exp(a) * a')
fn dual_exp(a: Dual) -> Dual {
    let e = exp(a.val)
    return Dual { val: e, der: e * a.der }
}

// Natural log: log(a, a') = (log(a), a'/a)
fn dual_log(a: Dual) -> Dual {
    if a.val <= 0.0 {
        return Dual { val: 0.0 - 1.0e308, der: 0.0 }
    }
    return Dual { val: log(a.val), der: a.der / a.val }
}

// Square root: sqrt(a, a') = (sqrt(a), a' / (2*sqrt(a)))
fn dual_sqrt(a: Dual) -> Dual {
    if a.val < 0.0 {
        return Dual { val: 0.0, der: 0.0 }
    }
    let s = sqrt(a.val)
    if s < 1.0e-15 {
        return Dual { val: 0.0, der: 0.0 }
    }
    return Dual { val: s, der: a.der / (2.0 * s) }
}

// Power: a^n where n is constant
// (a, a')^n = (a^n, n * a^(n-1) * a')
fn dual_pow(a: Dual, n: f64) -> Dual {
    if abs_f64(a.val) < 1.0e-15 && n < 1.0 {
        return Dual { val: 0.0, der: 0.0 }
    }
    var pow_val: f64 = 1.0
    var pow_deriv: f64 = 0.0

    // Compute a^n using exp/log for general n
    if abs_f64(a.val) > 1.0e-15 {
        pow_val = exp(n * log(abs_f64(a.val)))
        if a.val < 0.0 {
            // Handle negative base for odd integer powers
            pow_val = 0.0 - pow_val
        }
        pow_deriv = n * exp((n - 1.0) * log(abs_f64(a.val))) * a.der
        if a.val < 0.0 && n != 2.0 {
            pow_deriv = 0.0 - pow_deriv
        }
    }

    return Dual { val: pow_val, der: pow_deriv }
}

// Sine: sin(a, a') = (sin(a), cos(a) * a')
fn dual_sin(a: Dual) -> Dual {
    return Dual { val: sin(a.val), der: cos(a.val) * a.der }
}

// Cosine: cos(a, a') = (cos(a), -sin(a) * a')
fn dual_cos(a: Dual) -> Dual {
    return Dual { val: cos(a.val), der: 0.0 - sin(a.val) * a.der }
}

// ============================================================================
// SENSITIVITY COEFFICIENT COMPUTATION
// ============================================================================

// Result of sensitivity analysis for a single input
struct SensitivityResult {
    value: f64,         // f(x)
    sensitivity: f64,   // ∂f/∂x
}

// Result of multi-input sensitivity analysis (up to 4 inputs)
struct MultiSensitivity {
    value: f64,
    sens1: f64,  // ∂f/∂x₁
    sens2: f64,  // ∂f/∂x₂
    sens3: f64,  // ∂f/∂x₃
    sens4: f64,  // ∂f/∂x₄
    n_inputs: i32,
}

// ============================================================================
// COMMON FUNCTION PATTERNS WITH AD
// ============================================================================

// f(x) = x² → ∂f/∂x = 2x
fn sensitivity_square(x: f64) -> SensitivityResult {
    let d = dual_var(x)
    let result = dual_mul(d, d)
    return SensitivityResult { value: result.val, sensitivity: result.der }
}

// f(x,y) = x + y → ∂f/∂x = 1, ∂f/∂y = 1
fn sensitivity_add(x: f64, y: f64) -> MultiSensitivity {
    // Copy parameters to locals (workaround for codegen bug with reused params)
    let x_val = x
    let y_val = y

    // Compute ∂f/∂x
    let dx = dual_var(x_val)
    let dy_const = dual_const(y_val)
    let r1 = dual_add(dx, dy_const)

    // Compute ∂f/∂y
    let dx_const = dual_const(x_val)
    let dy = dual_var(y_val)
    let r2 = dual_add(dx_const, dy)

    // Store in intermediates before struct return
    let val = r1.val
    let s1 = r1.der
    let s2 = r2.der

    return MultiSensitivity {
        value: val,
        sens1: s1,
        sens2: s2,
        sens3: 0.0,
        sens4: 0.0,
        n_inputs: 2,
    }
}

// f(x,y) = x * y → ∂f/∂x = y, ∂f/∂y = x
fn sensitivity_mul(x: f64, y: f64) -> MultiSensitivity {
    // Copy parameters to locals (workaround for codegen bug with reused params)
    let x_val = x
    let y_val = y

    let dx = dual_var(x_val)
    let dy_const = dual_const(y_val)
    let r1 = dual_mul(dx, dy_const)

    let dx_const = dual_const(x_val)
    let dy = dual_var(y_val)
    let r2 = dual_mul(dx_const, dy)

    let val = r1.val
    let s1 = r1.der
    let s2 = r2.der

    return MultiSensitivity {
        value: val,
        sens1: s1,
        sens2: s2,
        sens3: 0.0,
        sens4: 0.0,
        n_inputs: 2,
    }
}

// f(x,y) = x / y → ∂f/∂x = 1/y, ∂f/∂y = -x/y²
fn sensitivity_div(x: f64, y: f64) -> MultiSensitivity {
    // Copy parameters to locals (workaround for codegen bug with reused params)
    let x_val = x
    let y_val = y

    let dx = dual_var(x_val)
    let dy_const = dual_const(y_val)
    let r1 = dual_div(dx, dy_const)

    let dx_const = dual_const(x_val)
    let dy = dual_var(y_val)
    let r2 = dual_div(dx_const, dy)

    let val = r1.val
    let s1 = r1.der
    let s2 = r2.der

    return MultiSensitivity {
        value: val,
        sens1: s1,
        sens2: s2,
        sens3: 0.0,
        sens4: 0.0,
        n_inputs: 2,
    }
}

// f(x) = exp(x) → ∂f/∂x = exp(x)
fn sensitivity_exp(x: f64) -> SensitivityResult {
    let d = dual_var(x)
    let result = dual_exp(d)
    return SensitivityResult { value: result.val, sensitivity: result.der }
}

// f(x) = log(x) → ∂f/∂x = 1/x
fn sensitivity_log(x: f64) -> SensitivityResult {
    let d = dual_var(x)
    let result = dual_log(d)
    return SensitivityResult { value: result.val, sensitivity: result.der }
}

// ============================================================================
// GUM UNCERTAINTY PROPAGATION WITH AD
// ============================================================================

// Propagate uncertainty through a function using AD-computed sensitivities
// For y = f(x₁, x₂): u(y)² = (∂f/∂x₁)² u²(x₁) + (∂f/∂x₂)² u²(x₂)
struct ADUncertainty {
    value: f64,
    std_uncert: f64,
    // Budget breakdown
    contrib1: f64,  // (∂f/∂x₁)² u²(x₁)
    contrib2: f64,
    sens1: f64,
    sens2: f64,
}

fn propagate_add_ad(x1: f64, u1: f64, x2: f64, u2: f64) -> ADUncertainty {
    // Copy parameters to locals
    let x1_val = x1
    let u1_val = u1
    let x2_val = x2
    let u2_val = u2

    let sens = sensitivity_add(x1_val, x2_val)
    let c1 = sens.sens1 * sens.sens1 * u1_val * u1_val
    let c2 = sens.sens2 * sens.sens2 * u2_val * u2_val
    let u_combined = sqrt_f64(c1 + c2)

    return ADUncertainty {
        value: sens.value,
        std_uncert: u_combined,
        contrib1: c1,
        contrib2: c2,
        sens1: sens.sens1,
        sens2: sens.sens2,
    }
}

fn propagate_mul_ad(x1: f64, u1: f64, x2: f64, u2: f64) -> ADUncertainty {
    // Copy parameters to locals
    let x1_val = x1
    let u1_val = u1
    let x2_val = x2
    let u2_val = u2

    let sens = sensitivity_mul(x1_val, x2_val)
    let c1 = sens.sens1 * sens.sens1 * u1_val * u1_val
    let c2 = sens.sens2 * sens.sens2 * u2_val * u2_val
    let u_combined = sqrt_f64(c1 + c2)

    return ADUncertainty {
        value: sens.value,
        std_uncert: u_combined,
        contrib1: c1,
        contrib2: c2,
        sens1: sens.sens1,
        sens2: sens.sens2,
    }
}

fn propagate_div_ad(x1: f64, u1: f64, x2: f64, u2: f64) -> ADUncertainty {
    // Copy parameters to locals
    let x1_val = x1
    let u1_val = u1
    let x2_val = x2
    let u2_val = u2

    let sens = sensitivity_div(x1_val, x2_val)
    let c1 = sens.sens1 * sens.sens1 * u1_val * u1_val
    let c2 = sens.sens2 * sens.sens2 * u2_val * u2_val
    let u_combined = sqrt_f64(c1 + c2)

    return ADUncertainty {
        value: sens.value,
        std_uncert: u_combined,
        contrib1: c1,
        contrib2: c2,
        sens1: sens.sens1,
        sens2: sens.sens2,
    }
}

// ============================================================================
// PK EXAMPLE: C(t) = (D/V) * exp(-k*t) WITH AD
// ============================================================================

// Compute concentration with full sensitivity analysis
struct PKSensitivity {
    concentration: f64,
    sens_dose: f64,      // ∂C/∂D
    sens_volume: f64,    // ∂C/∂V
    sens_k_elim: f64,    // ∂C/∂k
    sens_time: f64,      // ∂C/∂t
}

fn pk_concentration_sensitivity(
    dose: f64,
    volume: f64,
    k_elim: f64,
    time: f64
) -> PKSensitivity {
    // Copy parameters to locals (workaround for codegen bug)
    let dose_val = dose
    let volume_val = volume
    let k_val = k_elim
    let time_val = time

    // C(t) = (D/V) * exp(-k*t)

    // ∂C/∂D: differentiate with D as variable
    let d_dose = dual_var(dose_val)
    let d_vol = dual_const(volume_val)
    let d_k = dual_const(k_val)
    let d_t = dual_const(time_val)
    let c_d = dual_mul(
        dual_div(d_dose, d_vol),
        dual_exp(dual_neg(dual_mul(d_k, d_t)))
    )

    // ∂C/∂V: differentiate with V as variable
    let d_dose2 = dual_const(dose_val)
    let d_vol2 = dual_var(volume_val)
    let c_v = dual_mul(
        dual_div(d_dose2, d_vol2),
        dual_exp(dual_neg(dual_mul(d_k, d_t)))
    )

    // ∂C/∂k: differentiate with k as variable
    let d_k2 = dual_var(k_val)
    let c_k = dual_mul(
        dual_div(d_dose2, d_vol),
        dual_exp(dual_neg(dual_mul(d_k2, d_t)))
    )

    // ∂C/∂t: differentiate with t as variable
    let d_t2 = dual_var(time_val)
    let c_t = dual_mul(
        dual_div(d_dose2, d_vol),
        dual_exp(dual_neg(dual_mul(d_k, d_t2)))
    )

    return PKSensitivity {
        concentration: c_d.val,
        sens_dose: c_d.der,
        sens_volume: c_v.der,
        sens_k_elim: c_k.der,
        sens_time: c_t.der,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_dual_add() -> bool {
    let a = dual_var(3.0)
    let b = dual_const(5.0)
    let c = dual_add(a, b)

    // f(x) = x + 5 → f(3) = 8, f'(3) = 1
    if abs_f64(c.val - 8.0) > 0.001 { return false }
    if abs_f64(c.der - 1.0) > 0.001 { return false }
    return true
}

fn test_dual_mul() -> bool {
    let a = dual_var(3.0)
    let b = dual_const(5.0)
    let c = dual_mul(a, b)

    // f(x) = 5x → f(3) = 15, f'(3) = 5
    if abs_f64(c.val - 15.0) > 0.001 { return false }
    if abs_f64(c.der - 5.0) > 0.001 { return false }
    return true
}

fn test_dual_square() -> bool {
    let result = sensitivity_square(4.0)

    // f(x) = x² → f(4) = 16, f'(4) = 8
    if abs_f64(result.value - 16.0) > 0.001 { return false }
    if abs_f64(result.sensitivity - 8.0) > 0.001 { return false }
    return true
}

fn test_dual_exp() -> bool {
    let result = sensitivity_exp(1.0)

    // f(x) = exp(x) → f(1) = e ≈ 2.718, f'(1) = e
    if abs_f64(result.value - 2.718281828) > 0.001 { return false }
    if abs_f64(result.sensitivity - 2.718281828) > 0.001 { return false }
    return true
}

fn test_dual_log() -> bool {
    let result = sensitivity_log(2.0)

    // f(x) = log(x) → f(2) = ln(2) ≈ 0.693, f'(2) = 1/2 = 0.5
    if abs_f64(result.value - 0.693147) > 0.001 { return false }
    if abs_f64(result.sensitivity - 0.5) > 0.001 { return false }
    return true
}

fn test_mul_sensitivity() -> bool {
    let sens = sensitivity_mul(3.0, 4.0)

    // f(x,y) = xy → f(3,4) = 12, ∂f/∂x = 4, ∂f/∂y = 3
    if abs_f64(sens.value - 12.0) > 0.001 { return false }
    if abs_f64(sens.sens1 - 4.0) > 0.001 { return false }
    if abs_f64(sens.sens2 - 3.0) > 0.001 { return false }
    return true
}

fn test_div_sensitivity() -> bool {
    let sens = sensitivity_div(10.0, 2.0)

    // f(x,y) = x/y → f(10,2) = 5, ∂f/∂x = 1/2 = 0.5, ∂f/∂y = -10/4 = -2.5
    if abs_f64(sens.value - 5.0) > 0.001 { return false }
    if abs_f64(sens.sens1 - 0.5) > 0.001 { return false }
    if abs_f64(sens.sens2 - (0.0 - 2.5)) > 0.001 { return false }
    return true
}

fn test_propagate_add() -> bool {
    // x₁ = 10 ± 1, x₂ = 20 ± 2
    let result = propagate_add_ad(10.0, 1.0, 20.0, 2.0)

    // y = 30
    if abs_f64(result.value - 30.0) > 0.001 { return false }

    // u(y) = sqrt(1² + 2²) = sqrt(5) ≈ 2.236
    let expected = sqrt_f64(5.0)
    if abs_f64(result.std_uncert - expected) > 0.01 { return false }

    // Sensitivities should both be 1
    if abs_f64(result.sens1 - 1.0) > 0.001 { return false }
    if abs_f64(result.sens2 - 1.0) > 0.001 { return false }

    return true
}

fn test_propagate_mul() -> bool {
    // x₁ = 10 ± 1 (10% rel), x₂ = 5 ± 0.5 (10% rel)
    let result = propagate_mul_ad(10.0, 1.0, 5.0, 0.5)

    // y = 50
    if abs_f64(result.value - 50.0) > 0.001 { return false }

    // ∂f/∂x₁ = 5, ∂f/∂x₂ = 10
    if abs_f64(result.sens1 - 5.0) > 0.001 { return false }
    if abs_f64(result.sens2 - 10.0) > 0.001 { return false }

    // u(y) = sqrt((5*1)² + (10*0.5)²) = sqrt(25 + 25) = sqrt(50) ≈ 7.07
    let expected = sqrt_f64(50.0)
    if abs_f64(result.std_uncert - expected) > 0.1 { return false }

    return true
}

fn test_pk_sensitivity() -> bool {
    // C(t) = (D/V) * exp(-k*t)
    // D = 300 mg, V = 35 L, k = 0.08 /h, t = 4 h
    let sens = pk_concentration_sensitivity(300.0, 35.0, 0.08, 4.0)

    // C = (300/35) * exp(-0.32) ≈ 8.57 * 0.726 ≈ 6.22
    if abs_f64(sens.concentration - 6.22) > 0.1 { return false }

    // ∂C/∂D = (1/V) * exp(-kt) = (1/35) * 0.726 ≈ 0.0207
    if abs_f64(sens.sens_dose - 0.0207) > 0.005 { return false }

    // ∂C/∂V = -(D/V²) * exp(-kt) ≈ -0.178
    if sens.sens_volume > 0.0 { return false }  // Should be negative

    // ∂C/∂k = -(D/V) * t * exp(-kt) ≈ -24.9
    if sens.sens_k_elim > 0.0 { return false }  // Should be negative

    // ∂C/∂t = -(D/V) * k * exp(-kt) ≈ -0.498
    if sens.sens_time > 0.0 { return false }  // Should be negative

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_dual_add() { return 1 }
    if !test_dual_mul() { return 2 }
    if !test_dual_square() { return 3 }
    if !test_dual_exp() { return 4 }
    if !test_dual_log() { return 5 }
    if !test_mul_sensitivity() { return 6 }
    if !test_div_sensitivity() { return 7 }
    if !test_propagate_add() { return 8 }
    if !test_propagate_mul() { return 9 }
    if !test_pk_sensitivity() { return 10 }

    return 0
}
