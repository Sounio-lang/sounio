//! Automatic Differentiation via Dual Numbers
//!
//! This module implements forward-mode automatic differentiation using dual numbers.
//! A dual number is a pair (value, derivative) where arithmetic operations propagate
//! derivatives according to calculus rules:
//!
//! - dual(a, a') + dual(b, b') = dual(a + b, a' + b')
//! - dual(a, a') - dual(b, b') = dual(a - b, a' - b')
//! - dual(a, a') * dual(b, b') = dual(a * b, a' * b + a * b')  [Product rule]
//! - dual(a, a') / dual(b, b') = dual(a / b, (a' * b - a * b') / b²)  [Quotient rule]
//!
//! For computing gradients:
//! - Set input x = dual(x_value, 1.0) to track derivative with respect to x
//! - The result's derivative component is df/dx at x_value
//!
//! This approach is exact (no numerical errors from finite differences) and efficient
//! for computing derivatives of scalar functions.

#[cfg(feature = "jit")]
use cranelift_codegen::ir::condcodes::FloatCC;
#[cfg(feature = "jit")]
use cranelift_codegen::ir::{types, FuncRef, InstBuilder, Value};
#[cfg(feature = "jit")]
use cranelift_frontend::FunctionBuilder;

/// Dual number operations for automatic differentiation
/// Layout: F64X2 where lane 0 = value, lane 1 = derivative
#[cfg(feature = "jit")]
pub struct DualOps;

#[cfg(feature = "jit")]
impl DualOps {
    /// Create a dual number from value and derivative
    pub fn create(builder: &mut FunctionBuilder, value: Value, derivative: Value) -> Value {
        // Create F64X2 vector with [value, derivative]
        let vec = builder.ins().scalar_to_vector(types::F64X2, value);
        builder.ins().insertlane(vec, derivative, 1)
    }

    /// Create a constant dual number (derivative = 0)
    pub fn constant(builder: &mut FunctionBuilder, value: f64) -> Value {
        let val = builder.ins().f64const(value);
        let zero = builder.ins().f64const(0.0);
        Self::create(builder, val, zero)
    }

    /// Create a variable dual number (derivative = 1, for computing df/dx)
    pub fn variable(builder: &mut FunctionBuilder, value: Value) -> Value {
        let one = builder.ins().f64const(1.0);
        Self::create(builder, value, one)
    }

    /// Extract the value component (lane 0)
    pub fn value(builder: &mut FunctionBuilder, dual: Value) -> Value {
        builder.ins().extractlane(dual, 0u8)
    }

    /// Extract the derivative component (lane 1)
    pub fn derivative(builder: &mut FunctionBuilder, dual: Value) -> Value {
        builder.ins().extractlane(dual, 1u8)
    }

    /// Addition: (a, a') + (b, b') = (a + b, a' + b')
    pub fn add(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fadd(a, b)
    }

    /// Subtraction: (a, a') - (b, b') = (a - b, a' - b')
    pub fn sub(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fsub(a, b)
    }

    /// Multiplication (product rule): (a, a') * (b, b') = (a*b, a'*b + a*b')
    pub fn mul(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);
        let b_val = Self::value(builder, b);
        let b_der = Self::derivative(builder, b);

        // Result value: a * b
        let result_val = builder.ins().fmul(a_val, b_val);

        // Result derivative: a' * b + a * b'
        let term1 = builder.ins().fmul(a_der, b_val);
        let term2 = builder.ins().fmul(a_val, b_der);
        let result_der = builder.ins().fadd(term1, term2);

        Self::create(builder, result_val, result_der)
    }

    /// Division (quotient rule): (a, a') / (b, b') = (a/b, (a'*b - a*b') / b²)
    pub fn div(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);
        let b_val = Self::value(builder, b);
        let b_der = Self::derivative(builder, b);

        // Result value: a / b
        let result_val = builder.ins().fdiv(a_val, b_val);

        // Result derivative: (a' * b - a * b') / b²
        let term1 = builder.ins().fmul(a_der, b_val);
        let term2 = builder.ins().fmul(a_val, b_der);
        let numerator = builder.ins().fsub(term1, term2);
        let b_squared = builder.ins().fmul(b_val, b_val);
        let result_der = builder.ins().fdiv(numerator, b_squared);

        Self::create(builder, result_val, result_der)
    }

    /// Negation: -(a, a') = (-a, -a')
    pub fn neg(builder: &mut FunctionBuilder, a: Value) -> Value {
        builder.ins().fneg(a)
    }

    /// Square root: sqrt(a, a') = (sqrt(a), a' / (2 * sqrt(a)))
    pub fn sqrt(builder: &mut FunctionBuilder, a: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);

        let sqrt_val = builder.ins().sqrt(a_val);

        // Derivative: a' / (2 * sqrt(a))
        let two = builder.ins().f64const(2.0);
        let denom = builder.ins().fmul(two, sqrt_val);
        let result_der = builder.ins().fdiv(a_der, denom);

        Self::create(builder, sqrt_val, result_der)
    }

    /// Power with constant exponent: (a, a')^n = (a^n, n * a^(n-1) * a')
    pub fn pow_const(builder: &mut FunctionBuilder, a: Value, n: f64) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);

        // For integer powers, we could use repeated multiplication
        // For now, use the general formula
        let n_val = builder.ins().f64const(n);
        let n_minus_1 = builder.ins().f64const(n - 1.0);

        // a^n - we need to implement pow, use exp(n * log(a)) for now
        // This is a simplification; a proper implementation would handle special cases
        let log_a = Self::log_value(builder, a_val);
        let n_log_a = builder.ins().fmul(n_val, log_a);
        let result_val = Self::exp_value(builder, n_log_a);

        // n * a^(n-1) * a'
        let log_a_2 = Self::log_value(builder, a_val);
        let nm1_log_a = builder.ins().fmul(n_minus_1, log_a_2);
        let a_nm1 = Self::exp_value(builder, nm1_log_a);
        let term = builder.ins().fmul(n_val, a_nm1);
        let result_der = builder.ins().fmul(term, a_der);

        Self::create(builder, result_val, result_der)
    }

    /// Exponential: exp(a, a') = (exp(a), exp(a) * a')
    pub fn exp(builder: &mut FunctionBuilder, a: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);

        let exp_val = Self::exp_value(builder, a_val);
        let result_der = builder.ins().fmul(exp_val, a_der);

        Self::create(builder, exp_val, result_der)
    }

    /// Natural logarithm: log(a, a') = (log(a), a' / a)
    pub fn log(builder: &mut FunctionBuilder, a: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);

        let log_val = Self::log_value(builder, a_val);
        let result_der = builder.ins().fdiv(a_der, a_val);

        Self::create(builder, log_val, result_der)
    }

    /// Sine: sin(a, a') = (sin(a), cos(a) * a')
    pub fn sin(builder: &mut FunctionBuilder, a: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);

        let sin_val = Self::sin_value(builder, a_val);
        let cos_val = Self::cos_value(builder, a_val);
        let result_der = builder.ins().fmul(cos_val, a_der);

        Self::create(builder, sin_val, result_der)
    }

    /// Cosine: cos(a, a') = (cos(a), -sin(a) * a')
    pub fn cos(builder: &mut FunctionBuilder, a: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);

        let cos_val = Self::cos_value(builder, a_val);
        let sin_val = Self::sin_value(builder, a_val);
        let neg_sin = builder.ins().fneg(sin_val);
        let result_der = builder.ins().fmul(neg_sin, a_der);

        Self::create(builder, cos_val, result_der)
    }

    /// Tangent: tan(a, a') = (tan(a), a' / cos²(a))
    pub fn tan(builder: &mut FunctionBuilder, a: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);

        let sin_val = Self::sin_value(builder, a_val);
        let cos_val = Self::cos_value(builder, a_val);
        let tan_val = builder.ins().fdiv(sin_val, cos_val);

        // a' / cos²(a) = a' * sec²(a)
        let cos_sq = builder.ins().fmul(cos_val, cos_val);
        let result_der = builder.ins().fdiv(a_der, cos_sq);

        Self::create(builder, tan_val, result_der)
    }

    /// Absolute value: abs(a, a') = (|a|, sign(a) * a')
    pub fn abs(builder: &mut FunctionBuilder, a: Value) -> Value {
        let a_val = Self::value(builder, a);
        let a_der = Self::derivative(builder, a);

        let abs_val = builder.ins().fabs(a_val);

        // sign(a) = a / |a| when a != 0
        let sign = builder.ins().fdiv(a_val, abs_val);
        let result_der = builder.ins().fmul(sign, a_der);

        Self::create(builder, abs_val, result_der)
    }

    // ==================== Helper functions for math operations ====================
    // TODO(Forward-Mode AD): These are placeholders for Cranelift JIT codegen.
    // To complete integration:
    // 1. Thread FuncRef parameters through DualOps methods
    // 2. Emit calls to pre-declared runtime_math_* functions (see cranelift.rs)
    // 3. For now, these return identity/constants since DualOps isn't used in production codegen yet
    //
    // For runtime evaluation (interpreter/testing), use the functions below this impl block.

    /// Compute exp(x) - placeholder for Cranelift IR emission
    fn exp_value(_builder: &mut FunctionBuilder, x: Value) -> Value {
        x // TODO: emit call to runtime_math_exp FuncRef
    }

    /// Compute log(x) - placeholder for Cranelift IR emission
    fn log_value(_builder: &mut FunctionBuilder, x: Value) -> Value {
        x // TODO: emit call to runtime_math_log FuncRef
    }

    /// Compute sin(x) - placeholder for Cranelift IR emission
    fn sin_value(_builder: &mut FunctionBuilder, x: Value) -> Value {
        x // TODO: emit call to runtime_math_sin FuncRef
    }

    /// Compute cos(x) - placeholder for Cranelift IR emission
    fn cos_value(builder: &mut FunctionBuilder, _x: Value) -> Value {
        builder.ins().f64const(1.0) // TODO: emit call to runtime_math_cos FuncRef
    }
}

// ==================== Runtime Dual Number Evaluation ====================
//
// These functions provide actual dual number arithmetic for interpreter/testing.
// They use pure-Rust libm for portability.

/// Runtime dual number for forward-mode AD
///
/// A dual number represents a value and its derivative: (x, dx/dt)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// The value component
    pub val: f64,
    /// The derivative component
    pub deriv: f64,
}

impl Dual {
    /// Create a new dual number
    pub fn new(val: f64, deriv: f64) -> Self {
        Self { val, deriv }
    }

    /// Create a constant (deriv = 0)
    pub fn constant(val: f64) -> Self {
        Self { val, deriv: 0.0 }
    }

    /// Create a variable (deriv = 1, for computing df/dx)
    pub fn variable(val: f64) -> Self {
        Self { val, deriv: 1.0 }
    }

    /// Addition: (a, a') + (b, b') = (a + b, a' + b')
    pub fn add(self, other: Self) -> Self {
        Self {
            val: self.val + other.val,
            deriv: self.deriv + other.deriv,
        }
    }

    /// Subtraction: (a, a') - (b, b') = (a - b, a' - b')
    pub fn sub(self, other: Self) -> Self {
        Self {
            val: self.val - other.val,
            deriv: self.deriv - other.deriv,
        }
    }

    /// Multiplication (product rule): (a, a') * (b, b') = (a*b, a'*b + a*b')
    pub fn mul(self, other: Self) -> Self {
        Self {
            val: self.val * other.val,
            deriv: self.deriv * other.val + self.val * other.deriv,
        }
    }

    /// Division (quotient rule): (a, a') / (b, b') = (a/b, (a'*b - a*b') / b²)
    pub fn div(self, other: Self) -> Self {
        Self {
            val: self.val / other.val,
            deriv: (self.deriv * other.val - self.val * other.deriv) / (other.val * other.val),
        }
    }

    /// Negation: -(a, a') = (-a, -a')
    pub fn neg(self) -> Self {
        Self {
            val: -self.val,
            deriv: -self.deriv,
        }
    }

    /// Square root: sqrt(a, a') = (sqrt(a), a' / (2 * sqrt(a)))
    pub fn sqrt(self) -> Self {
        let sqrt_val = libm::sqrt(self.val);
        Self {
            val: sqrt_val,
            deriv: self.deriv / (2.0 * sqrt_val),
        }
    }

    /// Exponential: exp(a, a') = (exp(a), exp(a) * a')
    pub fn exp(self) -> Self {
        let exp_val = libm::exp(self.val);
        Self {
            val: exp_val,
            deriv: exp_val * self.deriv,
        }
    }

    /// Natural logarithm: log(a, a') = (log(a), a' / a)
    pub fn log(self) -> Self {
        Self {
            val: libm::log(self.val),
            deriv: self.deriv / self.val,
        }
    }

    /// Sine: sin(a, a') = (sin(a), cos(a) * a')
    pub fn sin(self) -> Self {
        Self {
            val: libm::sin(self.val),
            deriv: libm::cos(self.val) * self.deriv,
        }
    }

    /// Cosine: cos(a, a') = (cos(a), -sin(a) * a')
    pub fn cos(self) -> Self {
        Self {
            val: libm::cos(self.val),
            deriv: -libm::sin(self.val) * self.deriv,
        }
    }

    /// Tangent: tan(a, a') = (tan(a), a' / cos²(a))
    pub fn tan(self) -> Self {
        let cos_val = libm::cos(self.val);
        Self {
            val: libm::tan(self.val),
            deriv: self.deriv / (cos_val * cos_val),
        }
    }

    /// Hyperbolic sine: sinh(a, a') = (sinh(a), cosh(a) * a')
    pub fn sinh(self) -> Self {
        Self {
            val: libm::sinh(self.val),
            deriv: libm::cosh(self.val) * self.deriv,
        }
    }

    /// Hyperbolic cosine: cosh(a, a') = (cosh(a), sinh(a) * a')
    pub fn cosh(self) -> Self {
        Self {
            val: libm::cosh(self.val),
            deriv: libm::sinh(self.val) * self.deriv,
        }
    }

    /// Hyperbolic tangent: tanh(a, a') = (tanh(a), a' / cosh²(a))
    pub fn tanh(self) -> Self {
        let cosh_val = libm::cosh(self.val);
        Self {
            val: libm::tanh(self.val),
            deriv: self.deriv / (cosh_val * cosh_val),
        }
    }

    /// Power with constant exponent: (a, a')^n = (a^n, n * a^(n-1) * a')
    pub fn pow_const(self, n: f64) -> Self {
        Self {
            val: libm::pow(self.val, n),
            deriv: n * libm::pow(self.val, n - 1.0) * self.deriv,
        }
    }

    /// General power: (a, a')^(b, b') = (a^b, a^b * (b'/a + b*a'/a * log(a)))
    ///
    /// Using the formula: d/dx[f(x)^g(x)] = f(x)^g(x) * [g'(x)*ln(f(x)) + g(x)*f'(x)/f(x)]
    pub fn pow(self, other: Self) -> Self {
        let pow_val = libm::pow(self.val, other.val);
        let log_base = libm::log(self.val);

        // Derivative: a^b * (b' * log(a) + b * a'/a)
        let deriv = pow_val * (other.deriv * log_base + other.val * self.deriv / self.val);

        Self {
            val: pow_val,
            deriv,
        }
    }

    /// Absolute value: |a, a'| = (|a|, sign(a) * a')
    pub fn abs(self) -> Self {
        let sign = if self.val >= 0.0 { 1.0 } else { -1.0 };
        Self {
            val: libm::fabs(self.val),
            deriv: sign * self.deriv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_dual_constant() {
        let x = Dual::constant(5.0);
        assert_eq!(x.val, 5.0);
        assert_eq!(x.deriv, 0.0);
    }

    #[test]
    fn test_dual_variable() {
        let x = Dual::variable(3.0);
        assert_eq!(x.val, 3.0);
        assert_eq!(x.deriv, 1.0); // d(x)/dx = 1
    }

    #[test]
    fn test_dual_add() {
        // f(x) = x + 5, f'(x) = 1
        let x = Dual::variable(3.0);
        let c = Dual::constant(5.0);
        let result = x.add(c);

        assert_eq!(result.val, 8.0);
        assert_eq!(result.deriv, 1.0);
    }

    #[test]
    fn test_dual_sub() {
        // f(x) = x - 2, f'(x) = 1
        let x = Dual::variable(5.0);
        let c = Dual::constant(2.0);
        let result = x.sub(c);

        assert_eq!(result.val, 3.0);
        assert_eq!(result.deriv, 1.0);
    }

    #[test]
    fn test_dual_mul() {
        // f(x) = x * 3, f'(x) = 3
        let x = Dual::variable(4.0);
        let c = Dual::constant(3.0);
        let result = x.mul(c);

        assert_eq!(result.val, 12.0);
        assert_eq!(result.deriv, 3.0);
    }

    #[test]
    fn test_dual_mul_product_rule() {
        // f(x) = x * x = x², f'(x) = 2x
        let x = Dual::variable(5.0);
        let result = x.mul(x);

        assert_eq!(result.val, 25.0);
        assert_eq!(result.deriv, 10.0); // 2 * 5
    }

    #[test]
    fn test_dual_div() {
        // f(x) = x / 2, f'(x) = 1/2
        let x = Dual::variable(10.0);
        let c = Dual::constant(2.0);
        let result = x.div(c);

        assert_eq!(result.val, 5.0);
        assert_eq!(result.deriv, 0.5);
    }

    #[test]
    fn test_dual_div_quotient_rule() {
        // f(x) = x / x = 1, f'(x) = 0
        let x = Dual::variable(7.0);
        let result = x.div(x);

        assert_eq!(result.val, 1.0);
        assert!((result.deriv - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_dual_neg() {
        // f(x) = -x, f'(x) = -1
        let x = Dual::variable(5.0);
        let result = x.neg();

        assert_eq!(result.val, -5.0);
        assert_eq!(result.deriv, -1.0);
    }

    #[test]
    fn test_dual_sqrt() {
        // f(x) = sqrt(x), f'(4) = 1/(2*sqrt(4)) = 1/4
        let x = Dual::variable(4.0);
        let result = x.sqrt();

        assert_eq!(result.val, 2.0);
        assert_eq!(result.deriv, 0.25);
    }

    #[test]
    fn test_dual_exp() {
        // f(x) = exp(x), f'(x) = exp(x)
        // At x=0: exp(0) = 1, f'(0) = 1
        let x = Dual::variable(0.0);
        let result = x.exp();

        assert!((result.val - 1.0).abs() < EPSILON);
        assert!((result.deriv - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dual_log() {
        // f(x) = log(x), f'(e) = 1/e
        let x = Dual::variable(std::f64::consts::E);
        let result = x.log();

        assert!((result.val - 1.0).abs() < EPSILON); // log(e) = 1
        assert!((result.deriv - 1.0 / std::f64::consts::E).abs() < EPSILON);
    }

    #[test]
    fn test_dual_sin() {
        // f(x) = sin(x), f'(0) = cos(0) = 1
        let x = Dual::variable(0.0);
        let result = x.sin();

        assert!((result.val - 0.0).abs() < EPSILON);
        assert!((result.deriv - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dual_cos() {
        // f(x) = cos(x), f'(0) = -sin(0) = 0
        let x = Dual::variable(0.0);
        let result = x.cos();

        assert!((result.val - 1.0).abs() < EPSILON); // cos(0) = 1
        assert!((result.deriv - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_dual_tan() {
        // f(x) = tan(x), f'(0) = 1/cos²(0) = 1
        let x = Dual::variable(0.0);
        let result = x.tan();

        assert!((result.val - 0.0).abs() < EPSILON); // tan(0) = 0
        assert!((result.deriv - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dual_pow_const() {
        // f(x) = x³, f'(2) = 3 * 2² = 12
        let x = Dual::variable(2.0);
        let result = x.pow_const(3.0);

        assert_eq!(result.val, 8.0);
        assert_eq!(result.deriv, 12.0);
    }

    #[test]
    fn test_dual_abs_positive() {
        // f(x) = |x| at x=5, f'(5) = 1
        let x = Dual::variable(5.0);
        let result = x.abs();

        assert_eq!(result.val, 5.0);
        assert_eq!(result.deriv, 1.0);
    }

    #[test]
    fn test_dual_abs_negative() {
        // f(x) = |x| at x=-5, f'(-5) = -1
        let x = Dual::variable(-5.0);
        let result = x.abs();

        assert_eq!(result.val, 5.0);
        assert_eq!(result.deriv, -1.0);
    }

    #[test]
    fn test_composite_function() {
        // f(x) = sin(x²), f'(x) = 2x * cos(x²)
        // At x=0: f(0) = sin(0) = 0, f'(0) = 0
        let x = Dual::variable(0.0);
        let x_squared = x.mul(x); // x²
        let result = x_squared.sin(); // sin(x²)

        assert!((result.val - 0.0).abs() < EPSILON);
        assert!((result.deriv - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_polynomial() {
        // f(x) = 3x² + 2x + 1
        // f'(x) = 6x + 2
        // At x=1: f(1) = 6, f'(1) = 8
        let x = Dual::variable(1.0);

        let three_x_sq = Dual::constant(3.0).mul(x.mul(x));
        let two_x = Dual::constant(2.0).mul(x);
        let one = Dual::constant(1.0);

        let result = three_x_sq.add(two_x).add(one);

        assert_eq!(result.val, 6.0);
        assert_eq!(result.deriv, 8.0);
    }

    #[test]
    fn test_chain_rule() {
        // f(x) = exp(sin(x)), f'(x) = cos(x) * exp(sin(x))
        // At x=0: f(0) = exp(0) = 1, f'(0) = 1 * 1 = 1
        let x = Dual::variable(0.0);
        let result = x.sin().exp();

        assert!((result.val - 1.0).abs() < EPSILON);
        assert!((result.deriv - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dual_sinh() {
        // f(x) = sinh(x), f'(x) = cosh(x)
        // At x=0: sinh(0) = 0, cosh(0) = 1
        let x = Dual::variable(0.0);
        let result = x.sinh();

        assert!(result.val.abs() < EPSILON);
        assert!((result.deriv - 1.0).abs() < EPSILON);

        // At x=1: sinh(1) ≈ 1.175, cosh(1) ≈ 1.543
        let x = Dual::variable(1.0);
        let result = x.sinh();
        assert!((result.val - libm::sinh(1.0)).abs() < EPSILON);
        assert!((result.deriv - libm::cosh(1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_dual_cosh() {
        // f(x) = cosh(x), f'(x) = sinh(x)
        // At x=0: cosh(0) = 1, sinh(0) = 0
        let x = Dual::variable(0.0);
        let result = x.cosh();

        assert!((result.val - 1.0).abs() < EPSILON);
        assert!(result.deriv.abs() < EPSILON);

        // At x=1: cosh(1) ≈ 1.543, sinh(1) ≈ 1.175
        let x = Dual::variable(1.0);
        let result = x.cosh();
        assert!((result.val - libm::cosh(1.0)).abs() < EPSILON);
        assert!((result.deriv - libm::sinh(1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_dual_tanh() {
        // f(x) = tanh(x), f'(x) = 1/cosh²(x) = sech²(x)
        // At x=0: tanh(0) = 0, sech²(0) = 1
        let x = Dual::variable(0.0);
        let result = x.tanh();

        assert!(result.val.abs() < EPSILON);
        assert!((result.deriv - 1.0).abs() < EPSILON);

        // At x=1: tanh(1) ≈ 0.762, sech²(1) = 1/cosh²(1) ≈ 0.420
        let x = Dual::variable(1.0);
        let result = x.tanh();
        let expected_deriv = 1.0 / (libm::cosh(1.0) * libm::cosh(1.0));
        assert!((result.val - libm::tanh(1.0)).abs() < EPSILON);
        assert!((result.deriv - expected_deriv).abs() < EPSILON);
    }

    #[test]
    fn test_dual_pow_general() {
        // f(x,y) = x^y, ∂f/∂x = y*x^(y-1), ∂f/∂y = x^y * ln(x)
        // At (x,y) = (2,3): 2^3 = 8
        // ∂f/∂x = 3*2^2 = 12, ∂f/∂y = 8*ln(2) ≈ 5.545

        // Test derivative w.r.t. base (x)
        let x = Dual::variable(2.0);
        let y = Dual::constant(3.0);
        let result = x.pow(y);

        assert!((result.val - 8.0).abs() < EPSILON);
        assert!((result.deriv - 12.0).abs() < 1e-9);

        // Test derivative w.r.t. exponent (y)
        let x = Dual::constant(2.0);
        let y = Dual::variable(3.0);
        let result = x.pow(y);

        let expected_deriv = 8.0 * libm::log(2.0);
        assert!((result.val - 8.0).abs() < EPSILON);
        assert!((result.deriv - expected_deriv).abs() < 1e-9);
    }

    #[test]
    fn test_jacobian_simple() {
        // f(x, y) = [x + y, x * y]
        // J = [[1, 1], [y, x]]
        // At (2, 3): J = [[1, 1], [3, 2]]
        let f = |inputs: &[Dual]| vec![inputs[0].add(inputs[1]), inputs[0].mul(inputs[1])];

        let jacobian = compute_jacobian(&f, &[2.0, 3.0]);

        assert_eq!(jacobian.len(), 2); // 2 outputs
        assert_eq!(jacobian[0].len(), 2); // 2 inputs

        // Check Jacobian values
        assert!((jacobian[0][0] - 1.0).abs() < EPSILON); // ∂(x+y)/∂x = 1
        assert!((jacobian[0][1] - 1.0).abs() < EPSILON); // ∂(x+y)/∂y = 1
        assert!((jacobian[1][0] - 3.0).abs() < EPSILON); // ∂(x*y)/∂x = y = 3
        assert!((jacobian[1][1] - 2.0).abs() < EPSILON); // ∂(x*y)/∂y = x = 2
    }

    #[test]
    fn test_jacobian_nonlinear() {
        // f(x, y) = [x², sin(y)]
        // J = [[2x, 0], [0, cos(y)]]
        // At (2, π/2): J = [[4, 0], [0, 0]]
        use std::f64::consts::PI;

        let f = |inputs: &[Dual]| vec![inputs[0].pow_const(2.0), inputs[1].sin()];

        let jacobian = compute_jacobian(&f, &[2.0, PI / 2.0]);

        assert_eq!(jacobian.len(), 2);
        assert_eq!(jacobian[0].len(), 2);

        // Check Jacobian values
        assert!((jacobian[0][0] - 4.0).abs() < EPSILON); // ∂(x²)/∂x = 2x = 4
        assert!(jacobian[0][1].abs() < EPSILON); // ∂(x²)/∂y = 0
        assert!(jacobian[1][0].abs() < EPSILON); // ∂(sin(y))/∂x = 0
        assert!(jacobian[1][1].abs() < EPSILON); // ∂(sin(y))/∂y = cos(π/2) = 0
    }
}

// ==================== QUATERNIONIC NEURAL NETWORK AUTODIFF ====================
//
// QNN Gradient Computation via Quaternion Dual Numbers
//
// For QNNs, we track gradients for each quaternion component (w, x, y, z).
// The Hamilton product gradient follows specific rules for noncommutative multiplication.
//
// Given: y = q1 ⊗ q2
// Where ⊗ is the Hamilton product:
//   y.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
//   y.x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
//   y.y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
//   y.z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
//
// Gradient of scalar loss L w.r.t quaternion q:
//   dL/dq = (dL/dy) ⊗ q*  (conjugate of the other operand)
//

#[cfg(feature = "jit")]
pub struct QuatDualOps;

#[cfg(feature = "jit")]
impl QuatDualOps {
    /// Create a quaternion dual number with value and 4-component gradient
    /// Layout: F32X4 lanes [w, x, y, z] for both value and gradient
    /// Stored as struct { value: F32X4, grad: F32X4 }
    pub fn create_quat_dual(
        builder: &mut FunctionBuilder,
        value: Value, // F32X4 quaternion value
        grad: Value,  // F32X4 gradient
    ) -> Value {
        // For now, we track gradients separately per component
        // A full implementation would use a struct with two F32X4 vectors
        grad
    }

    /// Hamilton product with gradient tracking
    /// Given dual quaternions (q1, dq1) and (q2, dq2):
    /// Returns (q1 ⊗ q2, d(q1 ⊗ q2))
    /// where d(q1 ⊗ q2) = dq1 ⊗ q2 + q1 ⊗ dq2
    pub fn hamilton_product_dual(
        builder: &mut FunctionBuilder,
        q1_val: Value,
        q1_grad: Value,
        q2_val: Value,
        q2_grad: Value,
    ) -> Value {
        use crate::codegen::simd::SimdQuat;

        // Forward pass: q = q1 ⊗ q2
        let q_val = SimdQuat::hamilton_product(builder, q1_val, q2_val);

        // Backward pass: dq = dq1 ⊗ q2 + q1 ⊗ dq2
        let term1 = SimdQuat::hamilton_product(builder, q1_grad, q2_val);
        let term2 = SimdQuat::hamilton_product(builder, q1_val, q2_grad);
        let q_grad = builder.ins().fadd(term1, term2);

        q_grad
    }

    /// Quaternion linear layer gradient: y = W ⊗ x + b
    /// For each output quaternion: y_o = Σ_i (W_{o,i} ⊗ x_i) + b_o
    ///
    /// # Gradients
    /// * dW[o,i] = dY[o] ⊗ conj(X[i]) - weight gradient via quaternion product
    /// * dX[i] = Σ_o (conj(W[o,i]) ⊗ dY[o]) - input gradient via transposed weights
    /// * dB[o] = dY[o] - bias gradient (direct)
    ///
    /// # Arguments
    /// * `w_val` - Weight quaternion (single element for demonstration)
    /// * `x_val` - Input quaternion
    /// * `dy_val` - Gradient w.r.t. output
    ///
    /// # Returns
    /// (dW, dX) - Gradients for weights and input
    ///
    /// # Note
    /// This is a simplified single-element implementation showing the core logic.
    /// Full batched implementation would iterate over in_features and out_features.
    pub fn quat_linear_grad(
        builder: &mut FunctionBuilder,
        w_val: Value,  // Single weight quaternion
        x_val: Value,  // Single input quaternion
        dy_val: Value, // Gradient w.r.t. output quaternion
        _batch_size: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> (Value, Value) {
        use crate::codegen::simd::SimdQuat;

        // Gradient w.r.t. weight: dW = dY ⊗ conj(X)
        // This follows from: d(W ⊗ X)/dW = X when taking gradients
        let x_conj = SimdQuat::conjugate(builder, x_val);
        let dw = SimdQuat::hamilton_product(builder, dy_val, x_conj);

        // Gradient w.r.t. input: dX = conj(W) ⊗ dY
        // This follows from: d(W ⊗ X)/dX = W when taking gradients
        let w_conj = SimdQuat::conjugate(builder, w_val);
        let dx = SimdQuat::hamilton_product(builder, w_conj, dy_val);

        (dw, dx)
    }

    /// Quaternion activation gradient (ReLU applied component-wise)
    /// Since ReLU is applied to each real component independently:
    /// d(ReLU(q))/dq = diag(mask) where mask[i] = 1 if q[i] > 0 else 0
    pub fn quat_relu_grad(builder: &mut FunctionBuilder, q_val: Value, dq_val: Value) -> Value {
        use crate::codegen::simd::SimdVec;

        // Extract components from q_val
        let q0 = builder.ins().extractlane(q_val, 0u8);
        let q1 = builder.ins().extractlane(q_val, 1u8);
        let q2 = builder.ins().extractlane(q_val, 2u8);
        let q3 = builder.ins().extractlane(q_val, 3u8);

        // Extract gradient components
        let dq0 = builder.ins().extractlane(dq_val, 0u8);
        let dq1 = builder.ins().extractlane(dq_val, 1u8);
        let dq2 = builder.ins().extractlane(dq_val, 2u8);
        let dq3 = builder.ins().extractlane(dq_val, 3u8);

        // Zero for masking
        let zero = builder.ins().f32const(0.0);

        // ReLU backward: if q[i] > 0, pass gradient; else 0
        // Compare each component with zero
        let mask0 = builder.ins().fcmp(FloatCC::GreaterThan, q0, zero);
        let mask1 = builder.ins().fcmp(FloatCC::GreaterThan, q1, zero);
        let mask2 = builder.ins().fcmp(FloatCC::GreaterThan, q2, zero);
        let mask3 = builder.ins().fcmp(FloatCC::GreaterThan, q3, zero);

        // Select gradient or zero based on mask
        let g0 = builder.ins().select(mask0, dq0, zero);
        let g1 = builder.ins().select(mask1, dq1, zero);
        let g2 = builder.ins().select(mask2, dq2, zero);
        let g3 = builder.ins().select(mask3, dq3, zero);

        // Pack into result quaternion
        SimdVec::splat_f32x4(builder, g0, g1, g2, g3)
    }

    /// Quaternion batch normalization gradient
    /// BN normalizes each quaternion component independently
    ///
    /// # Arguments
    /// * `x_val` - Input value (before normalization)
    /// * `gamma_val` - Scale parameter (learnable)
    /// * `mean_val` - Batch mean (4 components)
    /// * `var_val` - Batch variance (4 components)
    /// * `d_y_val` - Gradient w.r.t. output
    ///
    /// # Returns
    /// (dgamma, dbeta, dx) - Gradients for scale, shift, and input
    pub fn quat_bn_grad(
        builder: &mut FunctionBuilder,
        x_val: Value,
        gamma_val: Value,
        mean_val: Value,
        var_val: Value,
        d_y_val: Value,
    ) -> (Value, Value, Value) {
        use crate::codegen::simd::SimdVec;

        let eps = builder.ins().f32const(1e-5);

        // Process each component independently
        let mut dgamma_components = Vec::with_capacity(4);
        let mut dbeta_components = Vec::with_capacity(4);
        let mut dx_components = Vec::with_capacity(4);

        for lane in 0..4 {
            let x = builder.ins().extractlane(x_val, lane as u8);
            let gamma = builder.ins().extractlane(gamma_val, lane as u8);
            let mean = builder.ins().extractlane(mean_val, lane as u8);
            let var = builder.ins().extractlane(var_val, lane as u8);
            let dy = builder.ins().extractlane(d_y_val, lane as u8);

            // Compute inv_std = 1 / sqrt(var + eps)
            let var_eps = builder.ins().fadd(var, eps);
            let std = builder.ins().sqrt(var_eps);
            let inv_std = builder.ins().fdiv(builder.ins().f32const(1.0), std);

            // x_normalized = (x - mean) / std
            let x_centered = builder.ins().fsub(x, mean);
            let x_norm = builder.ins().fmul(x_centered, inv_std);

            // dgamma = dy * x_normalized
            let dgamma = builder.ins().fmul(dy, x_norm);
            dgamma_components.push(dgamma);

            // dbeta = dy (gradient flows directly through bias)
            dbeta_components.push(dy);

            // dx = dy * gamma * inv_std
            // This is the gradient backpropagated to the input
            let dy_gamma = builder.ins().fmul(dy, gamma);
            let dx = builder.ins().fmul(dy_gamma, inv_std);
            dx_components.push(dx);
        }

        // Pack results into quaternions
        let dgamma = SimdVec::splat_f32x4(
            builder,
            dgamma_components[0],
            dgamma_components[1],
            dgamma_components[2],
            dgamma_components[3],
        );

        let dbeta = SimdVec::splat_f32x4(
            builder,
            dbeta_components[0],
            dbeta_components[1],
            dbeta_components[2],
            dbeta_components[3],
        );

        let dx = SimdVec::splat_f32x4(
            builder,
            dx_components[0],
            dx_components[1],
            dx_components[2],
            dx_components[3],
        );

        (dgamma, dbeta, dx)
    }
}

/// Compute gradient of a scalar function at a point
/// grad(f, x) evaluates f(dual(x, 1.0)) and returns the derivative component
#[cfg(feature = "jit")]
pub fn compute_gradient<F>(f: F, x: f64) -> f64
where
    F: Fn(f64, f64) -> (f64, f64), // Takes (value, deriv), returns (value, deriv)
{
    let (_, derivative) = f(x, 1.0);
    derivative
}

/// Compute Jacobian matrix of a multi-dimensional function
///
/// For f: R^n → R^m, the Jacobian J is an m×n matrix where:
/// J[i][j] = ∂f_i/∂x_j
///
/// Uses forward-mode AD by computing one column of J per call.
/// For each input dimension j, we set the j-th variable to have deriv=1
/// and all others to deriv=0, then evaluate f to get column j of the Jacobian.
///
/// # Arguments
/// * `f` - Function taking a vector of Dual numbers and returning a vector of Dual numbers
/// * `x` - Point at which to evaluate the Jacobian
///
/// # Returns
/// Jacobian matrix J where J[i][j] = ∂f_i/∂x_j
///
/// # Example
/// ```
/// use souc::codegen::autodiff::{Dual, compute_jacobian};
///
/// // f(x, y) = [x*y, x + y²]
/// let f = |inputs: &[Dual]| {
///     vec![
///         inputs[0].mul(inputs[1]),           // x * y
///         inputs[0].add(inputs[1].pow_const(2.0)), // x + y²
///     ]
/// };
///
/// let jacobian = compute_jacobian(&f, &[2.0, 3.0]);
/// // J = [[∂(xy)/∂x, ∂(xy)/∂y  ],  = [[y, x    ],  = [[3, 2 ],
/// //      [∂(x+y²)/∂x, ∂(x+y²)/∂y]]    [1, 2y   ]]    [1, 6 ]]
/// ```
pub fn compute_jacobian<F>(f: &F, x: &[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[Dual]) -> Vec<Dual>,
{
    let n_inputs = x.len();

    // Evaluate f once to determine output dimension
    let test_inputs: Vec<Dual> = x.iter().map(|&xi| Dual::constant(xi)).collect();
    let test_output = f(&test_inputs);
    let n_outputs = test_output.len();

    // Initialize Jacobian matrix (m × n)
    let mut jacobian = vec![vec![0.0; n_inputs]; n_outputs];

    // Compute each column of the Jacobian
    for j in 0..n_inputs {
        // Create input vector with j-th variable as active (deriv=1)
        let inputs: Vec<Dual> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                if i == j {
                    Dual::variable(xi) // deriv = 1
                } else {
                    Dual::constant(xi) // deriv = 0
                }
            })
            .collect();

        // Evaluate f with this input
        let outputs = f(&inputs);

        // Extract derivatives (column j of Jacobian)
        for (i, output) in outputs.iter().enumerate() {
            jacobian[i][j] = output.deriv;
        }
    }

    jacobian
}
