//! Interval Arithmetic for Mock SMT Solver
//!
//! This module implements interval arithmetic operations used by the mock
//! solver to evaluate constraints without calling Z3. This provides fast
//! (but incomplete) verification for common cases.
//!
//! # Theory
//!
//! Interval arithmetic computes bounds on expressions by tracking the
//! possible range of values. For example:
//! - [1, 3] + [2, 4] = [3, 7]
//! - [1, 3] * [-2, 2] = [-6, 6]
//!
//! This is sound (never misses real bugs) but incomplete (may report
//! false positives when intervals overlap).

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An interval [lo, hi] representing a range of possible values
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Lower bound (inclusive)
    pub lo: f64,
    /// Upper bound (inclusive)
    pub hi: f64,
}

impl Interval {
    /// Create a new interval [lo, hi]
    ///
    /// # Panics
    /// Panics if lo > hi (except for empty interval representation)
    pub fn new(lo: f64, hi: f64) -> Self {
        debug_assert!(
            lo <= hi || (lo.is_nan() && hi.is_nan()),
            "Invalid interval: [{}, {}]",
            lo,
            hi
        );
        Self { lo, hi }
    }

    /// Create a point interval [x, x]
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    /// Create the entire real line (-∞, ∞)
    pub fn all() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// Create an empty interval (represents contradiction)
    pub fn empty() -> Self {
        Self {
            lo: f64::NAN,
            hi: f64::NAN,
        }
    }

    /// Create a non-negative interval [0, ∞)
    pub fn non_negative() -> Self {
        Self {
            lo: 0.0,
            hi: f64::INFINITY,
        }
    }

    /// Create a positive interval (0, ∞)
    pub fn positive() -> Self {
        Self {
            lo: f64::MIN_POSITIVE,
            hi: f64::INFINITY,
        }
    }

    /// Create an epsilon interval [0, bound] for uncertainty
    pub fn epsilon(bound: f64) -> Self {
        Self { lo: 0.0, hi: bound }
    }

    /// Check if this interval is empty
    pub fn is_empty(&self) -> bool {
        self.lo.is_nan() || self.hi.is_nan()
    }

    /// Check if this interval is a single point
    pub fn is_point(&self) -> bool {
        !self.is_empty() && self.lo == self.hi
    }

    /// Check if this interval contains a value
    pub fn contains(&self, x: f64) -> bool {
        !self.is_empty() && self.lo <= x && x <= self.hi
    }

    /// Check if this interval contains zero
    pub fn contains_zero(&self) -> bool {
        self.contains(0.0)
    }

    /// Check if this interval is entirely positive
    pub fn is_positive(&self) -> bool {
        !self.is_empty() && self.lo > 0.0
    }

    /// Check if this interval is entirely negative
    pub fn is_negative(&self) -> bool {
        !self.is_empty() && self.hi < 0.0
    }

    /// Check if this interval is entirely non-negative
    pub fn is_non_negative(&self) -> bool {
        !self.is_empty() && self.lo >= 0.0
    }

    /// Check if this interval is entirely non-positive
    pub fn is_non_positive(&self) -> bool {
        !self.is_empty() && self.hi <= 0.0
    }

    /// Intersect two intervals
    pub fn intersect(self, other: Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            Self::empty()
        } else {
            Self { lo, hi }
        }
    }

    /// Union of two intervals (hull)
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Width of the interval
    pub fn width(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    /// Midpoint of the interval
    pub fn midpoint(&self) -> f64 {
        if self.is_empty() {
            f64::NAN
        } else {
            (self.lo + self.hi) / 2.0
        }
    }

    /// Absolute value of interval
    pub fn abs(self) -> Self {
        if self.is_empty() {
            return self;
        }
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        } else {
            Self {
                lo: 0.0,
                hi: self.lo.abs().max(self.hi.abs()),
            }
        }
    }

    /// Square of interval
    pub fn sqr(self) -> Self {
        if self.is_empty() {
            return self;
        }
        if self.lo >= 0.0 {
            Self {
                lo: self.lo * self.lo,
                hi: self.hi * self.hi,
            }
        } else if self.hi <= 0.0 {
            Self {
                lo: self.hi * self.hi,
                hi: self.lo * self.lo,
            }
        } else {
            Self {
                lo: 0.0,
                hi: (self.lo * self.lo).max(self.hi * self.hi),
            }
        }
    }

    /// Square root of interval (defined for non-negative intervals)
    pub fn sqrt(self) -> Self {
        if self.is_empty() || self.hi < 0.0 {
            return Self::empty();
        }
        Self {
            lo: if self.lo > 0.0 { self.lo.sqrt() } else { 0.0 },
            hi: self.hi.sqrt(),
        }
    }
}

impl Add for Interval {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        Self {
            lo: self.lo + rhs.lo,
            hi: self.hi + rhs.hi,
        }
    }
}

impl Sub for Interval {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        Self {
            lo: self.lo - rhs.hi,
            hi: self.hi - rhs.lo,
        }
    }
}

impl Mul for Interval {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }

        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];

        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self { lo, hi }
    }
}

impl Div for Interval {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }

        // Division by interval containing zero is undefined
        if rhs.contains_zero() {
            return Self::all();
        }

        let quotients = [
            self.lo / rhs.lo,
            self.lo / rhs.hi,
            self.hi / rhs.lo,
            self.hi / rhs.hi,
        ];

        let lo = quotients.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = quotients.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self { lo, hi }
    }
}

impl Neg for Interval {
    type Output = Self;

    fn neg(self) -> Self {
        if self.is_empty() {
            return self;
        }
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "∅")
        } else if self.is_point() {
            write!(f, "[{}]", self.lo)
        } else {
            write!(f, "[{}, {}]", self.lo, self.hi)
        }
    }
}

/// Trait for interval arithmetic on expressions
pub trait IntervalArithmetic {
    /// Evaluate expression and return interval bounds
    fn evaluate(&self, env: &IntervalEnv) -> Interval;
}

/// Environment mapping variables to their interval bounds
#[derive(Debug, Clone, Default)]
pub struct IntervalEnv {
    bindings: std::collections::HashMap<String, Interval>,
}

impl IntervalEnv {
    /// Create a new empty environment
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a variable to an interval
    pub fn bind(&mut self, name: String, interval: Interval) {
        self.bindings.insert(name, interval);
    }

    /// Get the interval for a variable (defaults to all reals)
    pub fn get(&self, name: &str) -> Interval {
        self.bindings
            .get(name)
            .copied()
            .unwrap_or_else(Interval::all)
    }

    /// Check if a variable is bound
    pub fn has(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }
}

/// Epsilon propagation through operations
///
/// Models how uncertainty (epsilon) propagates through arithmetic:
/// - Addition: ε₁ + ε₂ (uncertainties add)
/// - Multiplication: |a|ε₂ + |b|ε₁ + ε₁ε₂ (approximated as ε₁ + ε₂ for small ε)
/// - Division: (ε₁ + |a/b|ε₂) / (1 - ε₂/|b|) (for b far from zero)
#[derive(Debug, Clone, Copy)]
pub struct EpsilonPropagation;

impl EpsilonPropagation {
    /// Propagate epsilon through addition: Knowledge[a, ε₁] + Knowledge[b, ε₂]
    pub fn add(eps1: f64, eps2: f64) -> f64 {
        eps1 + eps2
    }

    /// Propagate epsilon through subtraction
    pub fn sub(eps1: f64, eps2: f64) -> f64 {
        eps1 + eps2
    }

    /// Propagate epsilon through multiplication
    /// For small ε: ε_result ≈ |a|ε₂ + |b|ε₁
    pub fn mul(a: Interval, eps1: f64, b: Interval, eps2: f64) -> f64 {
        let abs_a = a.abs();
        let abs_b = b.abs();
        // Upper bound on propagated epsilon
        abs_a.hi * eps2 + abs_b.hi * eps1 + eps1 * eps2
    }

    /// Propagate epsilon through division
    /// Requires b to not contain zero
    pub fn div(a: Interval, eps1: f64, b: Interval, eps2: f64) -> f64 {
        if b.contains_zero() {
            return f64::INFINITY;
        }

        let abs_a = a.abs();
        let abs_b = b.abs();

        // For |b| bounded away from zero
        let min_abs_b = abs_b.lo.max(f64::MIN_POSITIVE);

        // Linearized propagation
        (eps1 + (abs_a.hi / min_abs_b) * eps2) / (1.0 - eps2 / min_abs_b).max(0.5)
    }

    /// Check if epsilon stays within bound after operation
    pub fn check_bounded(eps_result: f64, bound: f64) -> bool {
        eps_result <= bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_arithmetic() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 4.0);

        // Addition
        let sum = a + b;
        assert_eq!(sum.lo, 3.0);
        assert_eq!(sum.hi, 7.0);

        // Subtraction
        let diff = a - b;
        assert_eq!(diff.lo, -3.0);
        assert_eq!(diff.hi, 1.0);

        // Multiplication
        let prod = a * b;
        assert_eq!(prod.lo, 2.0);
        assert_eq!(prod.hi, 12.0);
    }

    #[test]
    fn test_interval_with_negative() {
        let a = Interval::new(-2.0, 3.0);
        let b = Interval::new(1.0, 2.0);

        let prod = a * b;
        assert_eq!(prod.lo, -4.0);
        assert_eq!(prod.hi, 6.0);
    }

    #[test]
    fn test_interval_division() {
        let a = Interval::new(2.0, 4.0);
        let b = Interval::new(1.0, 2.0);

        let quot = a / b;
        assert_eq!(quot.lo, 1.0);
        assert_eq!(quot.hi, 4.0);
    }

    #[test]
    fn test_interval_contains() {
        let a = Interval::new(1.0, 5.0);

        assert!(a.contains(1.0));
        assert!(a.contains(3.0));
        assert!(a.contains(5.0));
        assert!(!a.contains(0.0));
        assert!(!a.contains(6.0));
    }

    #[test]
    fn test_interval_predicates() {
        let pos = Interval::new(1.0, 5.0);
        let neg = Interval::new(-5.0, -1.0);
        let mixed = Interval::new(-2.0, 3.0);

        assert!(pos.is_positive());
        assert!(neg.is_negative());
        assert!(!mixed.is_positive());
        assert!(!mixed.is_negative());
        assert!(mixed.contains_zero());
    }

    #[test]
    fn test_interval_abs() {
        let neg = Interval::new(-5.0, -1.0);
        let abs_neg = neg.abs();
        assert_eq!(abs_neg.lo, 1.0);
        assert_eq!(abs_neg.hi, 5.0);

        let mixed = Interval::new(-2.0, 3.0);
        let abs_mixed = mixed.abs();
        assert_eq!(abs_mixed.lo, 0.0);
        assert_eq!(abs_mixed.hi, 3.0);
    }

    #[test]
    fn test_interval_intersect() {
        let a = Interval::new(1.0, 5.0);
        let b = Interval::new(3.0, 7.0);

        let inter = a.intersect(b);
        assert_eq!(inter.lo, 3.0);
        assert_eq!(inter.hi, 5.0);

        let c = Interval::new(6.0, 8.0);
        let empty = a.intersect(c);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_epsilon_propagation() {
        // Addition: uncertainties add
        let eps = EpsilonPropagation::add(0.01, 0.02);
        assert!((eps - 0.03).abs() < 1e-10);

        // Multiplication with intervals
        let a = Interval::new(10.0, 20.0);
        let b = Interval::new(2.0, 3.0);
        let eps_mul = EpsilonPropagation::mul(a, 0.01, b, 0.02);
        // Should be bounded
        assert!(eps_mul < 1.0);
    }

    #[test]
    fn test_interval_env() {
        let mut env = IntervalEnv::new();
        env.bind("x".to_string(), Interval::new(0.0, 10.0));
        env.bind("epsilon_x".to_string(), Interval::epsilon(0.01));

        let x = env.get("x");
        assert_eq!(x.lo, 0.0);
        assert_eq!(x.hi, 10.0);

        let eps = env.get("epsilon_x");
        assert_eq!(eps.lo, 0.0);
        assert_eq!(eps.hi, 0.01);

        // Unknown variable returns all reals
        let y = env.get("y");
        assert!(y.lo.is_infinite() && y.lo < 0.0);
        assert!(y.hi.is_infinite() && y.hi > 0.0);
    }
}
