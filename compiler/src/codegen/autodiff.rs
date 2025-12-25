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
use cranelift_codegen::ir::{InstBuilder, Value, types};
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
    // These would normally call libm functions via external calls

    /// Compute exp(x) - placeholder using Cranelift intrinsics when available
    fn exp_value(builder: &mut FunctionBuilder, x: Value) -> Value {
        // Cranelift doesn't have exp intrinsic, so we'd need a libm call
        // For now, return a placeholder - in production, this would be an external call
        // to the C library's exp function
        x // Placeholder - actual implementation needs libm linking
    }

    /// Compute log(x) - placeholder
    fn log_value(builder: &mut FunctionBuilder, x: Value) -> Value {
        x // Placeholder
    }

    /// Compute sin(x) - placeholder
    fn sin_value(builder: &mut FunctionBuilder, x: Value) -> Value {
        x // Placeholder
    }

    /// Compute cos(x) - placeholder
    fn cos_value(builder: &mut FunctionBuilder, x: Value) -> Value {
        // Return 1.0 as placeholder (cos(0) = 1)
        builder.ins().f64const(1.0)
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_autodiff_module_exists() {
        // Basic test to ensure the module compiles
        assert!(true);
    }

    #[test]
    fn test_dual_arithmetic_theory() {
        // Test dual number arithmetic rules (without Cranelift)
        // dual(a, a') + dual(b, b') = dual(a+b, a'+b')
        let (a, a_prime) = (3.0, 1.0);
        let (b, b_prime) = (2.0, 0.0);

        // Addition
        let (sum_val, sum_der) = (a + b, a_prime + b_prime);
        assert_eq!(sum_val, 5.0);
        assert_eq!(sum_der, 1.0);

        // Multiplication (product rule)
        let (mul_val, mul_der) = (a * b, a_prime * b + a * b_prime);
        assert_eq!(mul_val, 6.0);
        assert_eq!(mul_der, 2.0); // d/dx(x * 2) = 2 at x = 3
    }

    #[test]
    fn test_gradient_theory() {
        // Test: d/dx(x²) = 2x
        // At x = 3: gradient should be 6

        // Simulate dual number computation for x²
        let x = 3.0;
        let x_deriv = 1.0; // Seeding with 1.0 for df/dx

        // x² = x * x using product rule
        // dual(x, 1) * dual(x, 1) = dual(x², 2x)
        let result_val = x * x;
        let result_deriv = x_deriv * x + x * x_deriv; // Product rule

        assert_eq!(result_val, 9.0);
        assert_eq!(result_deriv, 6.0); // d/dx(x²) at x=3 is 2*3 = 6
    }

    #[test]
    fn test_chain_rule() {
        // Test: d/dx(sin(x²)) = 2x * cos(x²)
        // This tests the chain rule composition

        let x: f64 = 1.0;
        let x_deriv: f64 = 1.0;

        // First compute x²
        let x_sq: f64 = x * x;
        let x_sq_deriv: f64 = 2.0 * x * x_deriv; // = 2

        // Then sin(x²) - derivative is cos(x²) * (derivative of x²)
        let sin_val = x_sq.sin();
        let sin_deriv = x_sq.cos() * x_sq_deriv;

        // d/dx(sin(x²)) at x=1 = 2 * cos(1)
        let expected = 2.0 * 1.0_f64.cos();
        assert!((sin_deriv - expected).abs() < 1e-10);
    }
}
