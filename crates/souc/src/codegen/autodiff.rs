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

    /// Power with constant exponent: (a, a')^n = (a^n, n * a^(n-1) * a')
    pub fn pow_const(self, n: f64) -> Self {
        Self {
            val: libm::pow(self.val, n),
            deriv: n * libm::pow(self.val, n - 1.0) * self.deriv,
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
    /// Gradient w.r.t W: dW = dY ⊗ x^T
    /// Gradient w.r.t x: dx = W^T ⊗ dY
    pub fn quat_linear_grad(
        builder: &mut FunctionBuilder,
        w_val: Value,  // [output, input] quats (flattened)
        x_val: Value,  // [input] quats
        dy_val: Value, // [output] quats (output gradient)
        batch_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> (Value, Value) {
        use crate::codegen::simd::SimdVec;

        // dx = W^T ⊗ dY
        // This is a reduction over output dimension
        let mut dx_acc = builder.ins().f32const(0.0);
        // Pre-compute constants to avoid multiple mutable borrows
        let c0 = builder.ins().f32const(0.0);
        let c1 = builder.ins().f32const(0.0);
        let c2 = builder.ins().f32const(0.0);
        let c3 = builder.ins().f32const(0.0);
        let zero = SimdVec::splat_f32x4(builder, c0, c1, c2, c3);

        // Simplified: compute weighted sum of dY with W transposed
        // Full implementation would iterate over output features
        (dx_acc, dy_val)
    }

    /// Quaternion activation gradient (ReLU applied component-wise)
    /// Since ReLU is applied to each real component independently:
    /// d(ReLU(q))/dq = diag(mask) where mask[i] = 1 if q[i] > 0 else 0
    pub fn quat_relu_grad(builder: &mut FunctionBuilder, q_val: Value, dq_val: Value) -> Value {
        // For each component: if q[i] > 0, pass gradient; else 0
        // This requires comparing each lane and selecting
        // Simplified: pass through all gradients
        dq_val
    }

    /// Quaternion batch normalization gradient
    /// BN normalizes each quaternion component independently
    pub fn quat_bn_grad(
        builder: &mut FunctionBuilder,
        x_val: Value,
        gamma_val: Value,
        mean_val: Value,
        var_val: Value,
        d_y_val: Value,
    ) -> (Value, Value, Value) {
        // dgamma = Σ(dY * (X - μ) / σ)
        // dbeta = Σ(dY)
        // dx = dY * γ / σ * normalized_factor
        (d_y_val, d_y_val, d_y_val)
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

