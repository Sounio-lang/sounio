//! Uncertain<T> Type with Error Propagation
//!
//! This module provides the `uncertain<T>` type for scientific computing,
//! enabling automatic error propagation through arithmetic operations.
//!
//! # Error Propagation Rules
//!
//! For independent uncertain values:
//! - Addition/Subtraction: σ(a ± b) = √(σ_a² + σ_b²)
//! - Multiplication: σ(a × b) = |a×b| × √((σ_a/a)² + (σ_b/b)²)
//! - Division: σ(a / b) = |a/b| × √((σ_a/a)² + (σ_b/b)²)
//! - Power: σ(a^n) = |n × a^(n-1)| × σ_a
//! - Functions: σ(f(x)) = |f'(x)| × σ_x
//!
//! # Example
//!
//! ```d
//! let mass = uncertain(5.0, 0.1);     // 5.0 ± 0.1 kg
//! let accel = uncertain(9.81, 0.02);  // 9.81 ± 0.02 m/s²
//! let force = mass * accel;            // Automatic error propagation
//! // force = 49.05 ± 0.51 N
//! ```

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Uncertain value with error/standard deviation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uncertain<T = f64> {
    /// Central value (mean)
    pub value: T,
    /// Standard deviation/uncertainty
    pub error: T,
}

impl<T: Default> Default for Uncertain<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            error: T::default(),
        }
    }
}

impl Uncertain<f64> {
    /// Create a new uncertain value
    #[inline]
    pub fn new(value: f64, error: f64) -> Self {
        Self {
            value,
            error: error.abs(),
        }
    }

    /// Create an exact (zero-error) value
    #[inline]
    pub fn exact(value: f64) -> Self {
        Self { value, error: 0.0 }
    }

    /// Get the value
    #[inline]
    pub fn val(&self) -> f64 {
        self.value
    }

    /// Get the error (standard deviation)
    #[inline]
    pub fn err(&self) -> f64 {
        self.error
    }

    /// Get relative error (σ/|value|)
    #[inline]
    pub fn relative_error(&self) -> f64 {
        if self.value.abs() < 1e-15 {
            f64::INFINITY
        } else {
            self.error / self.value.abs()
        }
    }

    /// Check if values are compatible within n standard deviations
    pub fn compatible_within(&self, other: &Self, n_sigma: f64) -> bool {
        let diff = (self.value - other.value).abs();
        let combined_error = (self.error.powi(2) + other.error.powi(2)).sqrt();
        diff <= n_sigma * combined_error
    }

    /// Square root with error propagation
    /// σ(√x) = σ_x / (2√x)
    pub fn sqrt(&self) -> Self {
        let val = self.value.sqrt();
        let err = self.error / (2.0 * val);
        Self::new(val, err)
    }

    /// Power with error propagation
    /// σ(x^n) = |n × x^(n-1)| × σ_x
    pub fn powf(&self, n: f64) -> Self {
        let val = self.value.powf(n);
        let err = (n * self.value.powf(n - 1.0)).abs() * self.error;
        Self::new(val, err)
    }

    /// Integer power
    pub fn powi(&self, n: i32) -> Self {
        self.powf(n as f64)
    }

    /// Exponential with error propagation
    /// σ(e^x) = e^x × σ_x
    pub fn exp(&self) -> Self {
        let val = self.value.exp();
        let err = val * self.error;
        Self::new(val, err)
    }

    /// Natural logarithm with error propagation
    /// σ(ln(x)) = σ_x / |x|
    pub fn ln(&self) -> Self {
        let val = self.value.ln();
        let err = self.error / self.value.abs();
        Self::new(val, err)
    }

    /// Log base 10
    pub fn log10(&self) -> Self {
        let val = self.value.log10();
        let err = self.error / (self.value.abs() * 10.0_f64.ln());
        Self::new(val, err)
    }

    /// Sine with error propagation
    /// σ(sin(x)) = |cos(x)| × σ_x
    pub fn sin(&self) -> Self {
        let val = self.value.sin();
        let err = self.value.cos().abs() * self.error;
        Self::new(val, err)
    }

    /// Cosine with error propagation
    /// σ(cos(x)) = |sin(x)| × σ_x
    pub fn cos(&self) -> Self {
        let val = self.value.cos();
        let err = self.value.sin().abs() * self.error;
        Self::new(val, err)
    }

    /// Tangent with error propagation
    /// σ(tan(x)) = σ_x / cos²(x)
    pub fn tan(&self) -> Self {
        let val = self.value.tan();
        let cos_x = self.value.cos();
        let err = self.error / (cos_x * cos_x);
        Self::new(val, err)
    }

    /// Absolute value
    pub fn abs(&self) -> Self {
        Self::new(self.value.abs(), self.error)
    }

    /// Format with appropriate significant figures
    pub fn to_string_sigfigs(&self) -> String {
        if self.error < 1e-15 {
            format!("{}", self.value)
        } else {
            let err_magnitude = self.error.log10().floor() as i32;
            let precision = (-err_magnitude).max(0) as usize;
            format!(
                "{:.prec$} ± {:.prec$}",
                self.value,
                self.error,
                prec = precision
            )
        }
    }
}

// Arithmetic operations with error propagation

impl Add for Uncertain<f64> {
    type Output = Self;

    /// Addition: σ(a + b) = √(σ_a² + σ_b²)
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.value + rhs.value,
            (self.error.powi(2) + rhs.error.powi(2)).sqrt(),
        )
    }
}

impl Add<f64> for Uncertain<f64> {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        Self::new(self.value + rhs, self.error)
    }
}

impl Sub for Uncertain<f64> {
    type Output = Self;

    /// Subtraction: σ(a - b) = √(σ_a² + σ_b²)
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.value - rhs.value,
            (self.error.powi(2) + rhs.error.powi(2)).sqrt(),
        )
    }
}

impl Sub<f64> for Uncertain<f64> {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        Self::new(self.value - rhs, self.error)
    }
}

impl Mul for Uncertain<f64> {
    type Output = Self;

    /// Multiplication: σ(a × b) = |a×b| × √((σ_a/a)² + (σ_b/b)²)
    fn mul(self, rhs: Self) -> Self::Output {
        let val = self.value * rhs.value;
        let rel_err_sq = self.relative_error().powi(2) + rhs.relative_error().powi(2);
        let err = val.abs() * rel_err_sq.sqrt();
        Self::new(val, err)
    }
}

impl Mul<f64> for Uncertain<f64> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.value * rhs, self.error * rhs.abs())
    }
}

impl Div for Uncertain<f64> {
    type Output = Self;

    /// Division: σ(a / b) = |a/b| × √((σ_a/a)² + (σ_b/b)²)
    fn div(self, rhs: Self) -> Self::Output {
        let val = self.value / rhs.value;
        let rel_err_sq = self.relative_error().powi(2) + rhs.relative_error().powi(2);
        let err = val.abs() * rel_err_sq.sqrt();
        Self::new(val, err)
    }
}

impl Div<f64> for Uncertain<f64> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.value / rhs, self.error / rhs.abs())
    }
}

impl Neg for Uncertain<f64> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.value, self.error)
    }
}

impl fmt::Display for Uncertain<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ± {}", self.value, self.error)
    }
}

/// Operations for uncertain values in codegen
pub struct UncertainOps;

impl UncertainOps {
    /// Create uncertain value from (value, error) pair
    pub fn create(value: f64, error: f64) -> Uncertain<f64> {
        Uncertain::new(value, error)
    }

    /// Extract value component
    pub fn value(u: &Uncertain<f64>) -> f64 {
        u.value
    }

    /// Extract error component
    pub fn error(u: &Uncertain<f64>) -> f64 {
        u.error
    }

    /// Add two uncertain values
    pub fn add(a: &Uncertain<f64>, b: &Uncertain<f64>) -> Uncertain<f64> {
        *a + *b
    }

    /// Subtract uncertain values
    pub fn sub(a: &Uncertain<f64>, b: &Uncertain<f64>) -> Uncertain<f64> {
        *a - *b
    }

    /// Multiply uncertain values
    pub fn mul(a: &Uncertain<f64>, b: &Uncertain<f64>) -> Uncertain<f64> {
        *a * *b
    }

    /// Divide uncertain values
    pub fn div(a: &Uncertain<f64>, b: &Uncertain<f64>) -> Uncertain<f64> {
        *a / *b
    }

    /// Square root with error propagation
    pub fn sqrt(u: &Uncertain<f64>) -> Uncertain<f64> {
        u.sqrt()
    }

    /// Exponential with error propagation
    pub fn exp(u: &Uncertain<f64>) -> Uncertain<f64> {
        u.exp()
    }

    /// Natural logarithm with error propagation
    pub fn ln(u: &Uncertain<f64>) -> Uncertain<f64> {
        u.ln()
    }

    /// Sine with error propagation
    pub fn sin(u: &Uncertain<f64>) -> Uncertain<f64> {
        u.sin()
    }

    /// Cosine with error propagation
    pub fn cos(u: &Uncertain<f64>) -> Uncertain<f64> {
        u.cos()
    }

    /// Power with error propagation
    pub fn pow(u: &Uncertain<f64>, n: f64) -> Uncertain<f64> {
        u.powf(n)
    }
}

/// Weighted mean of uncertain values
/// Result: value = Σ(w_i × x_i) / Σ(w_i), where w_i = 1/σ_i²
/// Error: σ = 1 / √(Σ(w_i))
pub fn weighted_mean(values: &[Uncertain<f64>]) -> Uncertain<f64> {
    if values.is_empty() {
        return Uncertain::new(0.0, f64::INFINITY);
    }

    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;

    for u in values {
        if u.error > 0.0 {
            let w = 1.0 / u.error.powi(2);
            sum_w += w;
            sum_wx += w * u.value;
        }
    }

    if sum_w < 1e-15 {
        // All values have zero error - simple average
        let avg = values.iter().map(|u| u.value).sum::<f64>() / values.len() as f64;
        Uncertain::exact(avg)
    } else {
        Uncertain::new(sum_wx / sum_w, 1.0 / sum_w.sqrt())
    }
}

/// Chi-squared test for consistency of uncertain values
pub fn chi_squared(values: &[Uncertain<f64>], expected: f64) -> f64 {
    values
        .iter()
        .map(|u| ((u.value - expected) / u.error).powi(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn assert_approx(a: f64, b: f64, msg: &str) {
        assert!((a - b).abs() < EPSILON, "{}: {} != {}", msg, a, b);
    }

    #[test]
    fn test_creation() {
        let u = Uncertain::new(5.0, 0.1);
        assert_eq!(u.value, 5.0);
        assert_eq!(u.error, 0.1);

        // Error should be absolute
        let u2 = Uncertain::new(5.0, -0.1);
        assert_eq!(u2.error, 0.1);
    }

    #[test]
    fn test_addition() {
        let a = Uncertain::new(10.0, 0.3);
        let b = Uncertain::new(5.0, 0.4);
        let c = a + b;

        assert_eq!(c.value, 15.0);
        // σ = √(0.3² + 0.4²) = √(0.09 + 0.16) = √0.25 = 0.5
        assert_approx(c.error, 0.5, "addition error");
    }

    #[test]
    fn test_subtraction() {
        let a = Uncertain::new(10.0, 0.3);
        let b = Uncertain::new(5.0, 0.4);
        let c = a - b;

        assert_eq!(c.value, 5.0);
        assert_approx(c.error, 0.5, "subtraction error");
    }

    #[test]
    fn test_multiplication() {
        let a = Uncertain::new(10.0, 0.1); // 1% relative error
        let b = Uncertain::new(5.0, 0.05); // 1% relative error
        let c = a * b;

        assert_eq!(c.value, 50.0);
        // Relative error = √(0.01² + 0.01²) ≈ 0.01414
        // Absolute error = 50.0 * 0.01414 ≈ 0.707
        assert!((c.error - 0.707).abs() < 0.01, "multiplication error");
    }

    #[test]
    fn test_division() {
        let a = Uncertain::new(10.0, 0.1);
        let b = Uncertain::new(5.0, 0.05);
        let c = a / b;

        assert_eq!(c.value, 2.0);
        // Same relative error propagation as multiplication
        assert!(
            (c.relative_error() - 0.01414).abs() < 0.001,
            "division relative error"
        );
    }

    #[test]
    fn test_sqrt() {
        let a = Uncertain::new(4.0, 0.04); // 1% error
        let b = a.sqrt();

        assert_eq!(b.value, 2.0);
        // σ(√x) = σ_x / (2√x) = 0.04 / (2*2) = 0.01
        assert_approx(b.error, 0.01, "sqrt error");
    }

    #[test]
    fn test_exp() {
        let a = Uncertain::new(1.0, 0.01);
        let b = a.exp();

        let e = std::f64::consts::E;
        assert_approx(b.value, e, "exp value");
        // σ(e^x) = e^x × σ_x = e × 0.01
        assert_approx(b.error, e * 0.01, "exp error");
    }

    #[test]
    fn test_ln() {
        let a = Uncertain::new(std::f64::consts::E, 0.01);
        let b = a.ln();

        assert_approx(b.value, 1.0, "ln value");
        // σ(ln(x)) = σ_x / |x| = 0.01 / e
        assert_approx(b.error, 0.01 / std::f64::consts::E, "ln error");
    }

    #[test]
    fn test_sin_cos() {
        use std::f64::consts::PI;

        let a = Uncertain::new(0.0, 0.01);
        let sin_a = a.sin();
        let cos_a = a.cos();

        assert_approx(sin_a.value, 0.0, "sin(0)");
        assert_approx(cos_a.value, 1.0, "cos(0)");

        // At x=0: σ(sin(x)) = |cos(0)| × σ_x = 1 × 0.01
        assert_approx(sin_a.error, 0.01, "sin(0) error");
        // At x=0: σ(cos(x)) = |sin(0)| × σ_x = 0 × 0.01
        assert_approx(cos_a.error, 0.0, "cos(0) error");
    }

    #[test]
    fn test_compatibility() {
        let a = Uncertain::new(10.0, 1.0);
        let b = Uncertain::new(10.5, 0.5);

        // Within 1σ: combined error = √(1 + 0.25) ≈ 1.118
        // Difference = 0.5, so 0.5 / 1.118 ≈ 0.45σ
        assert!(
            a.compatible_within(&b, 1.0),
            "should be compatible within 1σ"
        );

        let c = Uncertain::new(15.0, 0.5);
        // Difference = 5.0, combined error ≈ 1.118
        // 5.0 / 1.118 ≈ 4.5σ
        assert!(
            !a.compatible_within(&c, 3.0),
            "should not be compatible within 3σ"
        );
    }

    #[test]
    fn test_weighted_mean() {
        let values = vec![Uncertain::new(10.0, 1.0), Uncertain::new(11.0, 2.0)];

        let mean = weighted_mean(&values);

        // w1 = 1, w2 = 0.25
        // mean = (10 × 1 + 11 × 0.25) / 1.25 = 12.75 / 1.25 = 10.2
        assert!((mean.value - 10.2).abs() < 0.01, "weighted mean value");

        // σ = 1/√(1 + 0.25) = 1/√1.25 ≈ 0.894
        assert!((mean.error - 0.894).abs() < 0.01, "weighted mean error");
    }

    #[test]
    fn test_display() {
        let u = Uncertain::new(3.14159, 0.001);
        let s = format!("{}", u);
        assert!(s.contains("±"), "should contain ±");
    }

    #[test]
    fn test_physics_example() {
        // Calculate kinetic energy: E = 0.5 × m × v²
        let mass = Uncertain::new(2.0, 0.05); // 2.0 ± 0.05 kg
        let velocity = Uncertain::new(3.0, 0.1); // 3.0 ± 0.1 m/s

        let v_squared = velocity * velocity;
        let energy = mass * v_squared * 0.5;

        // E = 0.5 × 2 × 9 = 9 J
        assert_approx(energy.value, 9.0, "kinetic energy value");

        // Relative errors:
        // - mass: 0.05/2.0 = 2.5%
        // - velocity: 0.1/3.0 = 3.33%
        // - v²: sqrt(3.33² + 3.33²) = sqrt(2) × 3.33% = 4.71%
        // - m×v²: sqrt(2.5² + 4.71²) = 5.33%
        // Absolute error for E = 9 × 0.0533 ≈ 0.48 J
        assert!(
            (energy.error - 0.48).abs() < 0.05,
            "kinetic energy error: got {}",
            energy.error
        );
    }
}
