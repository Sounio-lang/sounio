//! Product Domain
//!
//! Combines multiple abstract domains to improve precision through
//! reduced product. Each domain tracks different properties, and
//! reduction propagates information between them.
//!
//! # Theory
//!
//! The reduced product of domains D1 and D2 is:
//! { (d1, d2) | gamma(d1) intersect gamma(d2) != empty }
//!
//! Reduction refines each component based on information from others.
//!
//! # References
//!
//! - Cousot, P., & Cousot, R. (1979). Systematic design of program
//!   analysis frameworks.

use super::congruence::Congruence;
use super::domain::AbstractDomain;
use super::intervals::Interval;
use super::signs::Sign;
use std::fmt;

/// A product of two abstract domains with reduction
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Product<D1: AbstractDomain, D2: AbstractDomain> {
    /// First component
    pub first: D1,
    /// Second component
    pub second: D2,
}

impl<D1: AbstractDomain, D2: AbstractDomain> Product<D1, D2> {
    /// Create a new product
    pub fn new(first: D1, second: D2) -> Self {
        Self { first, second }
    }

    /// Get the first component
    pub fn first(&self) -> &D1 {
        &self.first
    }

    /// Get the second component
    pub fn second(&self) -> &D2 {
        &self.second
    }
}

impl<D1: AbstractDomain, D2: AbstractDomain> AbstractDomain for Product<D1, D2> {
    fn top() -> Self {
        Self::new(D1::top(), D2::top())
    }

    fn bottom() -> Self {
        Self::new(D1::bottom(), D2::bottom())
    }

    fn join(&self, other: &Self) -> Self {
        Self::new(
            self.first.join(&other.first),
            self.second.join(&other.second),
        )
    }

    fn meet(&self, other: &Self) -> Self {
        Self::new(
            self.first.meet(&other.first),
            self.second.meet(&other.second),
        )
    }

    fn widen(&self, other: &Self) -> Self {
        Self::new(
            self.first.widen(&other.first),
            self.second.widen(&other.second),
        )
    }

    fn narrow(&self, other: &Self) -> Self {
        Self::new(
            self.first.narrow(&other.first),
            self.second.narrow(&other.second),
        )
    }

    fn leq(&self, other: &Self) -> bool {
        self.first.leq(&other.first) && self.second.leq(&other.second)
    }

    fn is_bottom(&self) -> bool {
        self.first.is_bottom() || self.second.is_bottom()
    }

    fn is_top(&self) -> bool {
        self.first.is_top() && self.second.is_top()
    }
}

impl<D1: AbstractDomain + fmt::Display, D2: AbstractDomain + fmt::Display> fmt::Display
    for Product<D1, D2>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.first, self.second)
    }
}

/// Reduced product of Interval and Sign domains
///
/// This is a common and useful combination where:
/// - Intervals provide precise numeric bounds
/// - Signs provide quick sign checks
///
/// Reduction propagates information in both directions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalSign {
    pub interval: Interval,
    pub sign: Sign,
}

impl IntervalSign {
    /// Create a new interval-sign product
    pub fn new(interval: Interval, sign: Sign) -> Self {
        let mut result = Self { interval, sign };
        result.reduce();
        result
    }

    /// Create from just an interval
    pub fn from_interval(interval: Interval) -> Self {
        let sign = Self::interval_to_sign(&interval);
        Self { interval, sign }
    }

    /// Create from just a sign
    pub fn from_sign(sign: Sign) -> Self {
        let interval = Self::sign_to_interval(&sign);
        Self { interval, sign }
    }

    /// Create from a constant
    pub fn constant(n: i64) -> Self {
        Self::new(Interval::constant(n), Sign::from_int(n))
    }

    /// Convert interval to sign
    fn interval_to_sign(interval: &Interval) -> Sign {
        match interval {
            Interval::Bottom => Sign::bottom(),
            Interval::Range { lo, hi } => {
                use super::intervals::Bound;
                let neg = *lo < Bound::Finite(0);
                let zero = *lo <= Bound::Finite(0) && Bound::Finite(0) <= *hi;
                let pos = *hi > Bound::Finite(0);
                Sign::new(neg, zero, pos)
            }
        }
    }

    /// Convert sign to interval (loose bounds)
    fn sign_to_interval(sign: &Sign) -> Interval {
        if sign.is_bottom() {
            return Interval::Bottom;
        }

        use super::intervals::Bound;

        let lo = if sign.neg {
            Bound::NegInf
        } else if sign.zero {
            Bound::Finite(0)
        } else {
            Bound::Finite(1)
        };

        let hi = if sign.pos {
            Bound::PosInf
        } else if sign.zero {
            Bound::Finite(0)
        } else {
            Bound::Finite(-1)
        };

        Interval::new(lo, hi)
    }

    /// Reduce the product - propagate information between components
    fn reduce(&mut self) {
        // First check for bottom
        if self.interval.is_bottom() || self.sign.is_bottom() {
            self.interval = Interval::Bottom;
            self.sign = Sign::bottom();
            return;
        }

        // Refine sign from interval
        let interval_sign = Self::interval_to_sign(&self.interval);
        self.sign = self.sign.meet(&interval_sign);

        // Refine interval from sign
        let sign_interval = Self::sign_to_interval(&self.sign);
        self.interval = self.interval.meet(&sign_interval);

        // Check consistency
        if self.interval.is_bottom() {
            self.sign = Sign::bottom();
        } else if self.sign.is_bottom() {
            self.interval = Interval::Bottom;
        }
    }

    /// Abstract addition
    pub fn add(&self, other: &Self) -> Self {
        let interval = self.interval.add(&other.interval);
        let sign = self.sign.add(&other.sign);
        Self::new(interval, sign)
    }

    /// Abstract subtraction
    pub fn sub(&self, other: &Self) -> Self {
        let interval = self.interval.sub(&other.interval);
        let sign = self.sign.sub(&other.sign);
        Self::new(interval, sign)
    }

    /// Abstract multiplication
    pub fn mul(&self, other: &Self) -> Self {
        let interval = self.interval.mul(&other.interval);
        let sign = self.sign.mul(&other.sign);
        Self::new(interval, sign)
    }

    /// Abstract division
    pub fn div(&self, other: &Self) -> Self {
        let interval = self.interval.div(&other.interval);
        let sign = self.sign.div(&other.sign);
        Self::new(interval, sign)
    }

    /// Abstract negation
    pub fn neg(&self) -> Self {
        Self::new(self.interval.neg(), self.sign.neg())
    }

    /// Check if definitely positive
    pub fn is_positive(&self) -> bool {
        self.interval.is_positive() || self.sign.is_definitely_positive()
    }

    /// Check if definitely non-negative
    pub fn is_non_negative(&self) -> bool {
        self.interval.is_non_negative() || self.sign.is_non_negative()
    }

    /// Check if definitely zero
    pub fn is_zero(&self) -> bool {
        self.interval.is_constant() == Some(0) || self.sign.is_definitely_zero()
    }

    /// Check if definitely non-zero
    pub fn is_non_zero(&self) -> bool {
        !self.interval.may_contain_zero() || self.sign.is_non_zero()
    }
}

impl Default for IntervalSign {
    fn default() -> Self {
        Self::top()
    }
}

impl AbstractDomain for IntervalSign {
    fn top() -> Self {
        Self {
            interval: Interval::top(),
            sign: Sign::top(),
        }
    }

    fn bottom() -> Self {
        Self {
            interval: Interval::Bottom,
            sign: Sign::bottom(),
        }
    }

    fn join(&self, other: &Self) -> Self {
        Self::new(
            self.interval.join(&other.interval),
            self.sign.join(&other.sign),
        )
    }

    fn meet(&self, other: &Self) -> Self {
        Self::new(
            self.interval.meet(&other.interval),
            self.sign.meet(&other.sign),
        )
    }

    fn widen(&self, other: &Self) -> Self {
        Self::new(
            self.interval.widen(&other.interval),
            self.sign.join(&other.sign),
        )
    }

    fn narrow(&self, other: &Self) -> Self {
        Self::new(
            self.interval.narrow(&other.interval),
            self.sign.meet(&other.sign),
        )
    }

    fn leq(&self, other: &Self) -> bool {
        self.interval.leq(&other.interval) && self.sign.leq(&other.sign)
    }

    fn is_bottom(&self) -> bool {
        self.interval.is_bottom() || self.sign.is_bottom()
    }

    fn is_top(&self) -> bool {
        self.interval.is_top() && self.sign.is_top()
    }
}

impl fmt::Display for IntervalSign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.interval, self.sign)
    }
}

/// Reduced product of Interval and Congruence domains
///
/// This combination is useful for:
/// - Array index analysis (bounds + alignment)
/// - Loop counter analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalCongruence {
    pub interval: Interval,
    pub congruence: Congruence,
}

impl IntervalCongruence {
    /// Create a new interval-congruence product
    pub fn new(interval: Interval, congruence: Congruence) -> Self {
        let mut result = Self {
            interval,
            congruence,
        };
        result.reduce();
        result
    }

    /// Create from just an interval
    pub fn from_interval(interval: Interval) -> Self {
        let congruence = Congruence::top();
        Self {
            interval,
            congruence,
        }
    }

    /// Create from just a congruence
    pub fn from_congruence(congruence: Congruence) -> Self {
        let interval = Interval::top();
        Self {
            interval,
            congruence,
        }
    }

    /// Create from a constant
    pub fn constant(n: i64) -> Self {
        Self::new(Interval::constant(n), Congruence::constant(n))
    }

    /// Reduce the product
    fn reduce(&mut self) {
        // If interval is a constant, congruence should be constant
        if let Some(c) = self.interval.is_constant() {
            self.congruence = self.congruence.meet(&Congruence::constant(c));
        }

        // If congruence is constant, refine interval
        if let Some(c) = self.congruence.is_constant() {
            self.interval = self.interval.meet(&Interval::constant(c));
        }

        // Check for bottom
        if self.interval.is_bottom() {
            self.congruence = Congruence::Bottom;
        } else if self.congruence.is_bottom() {
            self.interval = Interval::Bottom;
        }
    }

    /// Abstract addition
    pub fn add(&self, other: &Self) -> Self {
        let interval = self.interval.add(&other.interval);
        let congruence = self.congruence.add(&other.congruence);
        Self::new(interval, congruence)
    }

    /// Abstract subtraction
    pub fn sub(&self, other: &Self) -> Self {
        let interval = self.interval.sub(&other.interval);
        let congruence = self.congruence.sub(&other.congruence);
        Self::new(interval, congruence)
    }

    /// Abstract multiplication
    pub fn mul(&self, other: &Self) -> Self {
        let interval = self.interval.mul(&other.interval);
        let congruence = self.congruence.mul(&other.congruence);
        Self::new(interval, congruence)
    }

    /// Check if value is aligned to n bytes
    pub fn is_aligned(&self, n: u64) -> bool {
        self.congruence.is_aligned(n)
    }

    /// Check if value is even
    pub fn is_even(&self) -> bool {
        self.congruence.is_even()
            || self
                .interval
                .is_constant()
                .map(|c| c % 2 == 0)
                .unwrap_or(false)
    }

    /// Check if value is odd
    pub fn is_odd(&self) -> bool {
        self.congruence.is_odd()
            || self
                .interval
                .is_constant()
                .map(|c| c % 2 == 1)
                .unwrap_or(false)
    }
}

impl Default for IntervalCongruence {
    fn default() -> Self {
        Self::top()
    }
}

impl AbstractDomain for IntervalCongruence {
    fn top() -> Self {
        Self {
            interval: Interval::top(),
            congruence: Congruence::top(),
        }
    }

    fn bottom() -> Self {
        Self {
            interval: Interval::Bottom,
            congruence: Congruence::Bottom,
        }
    }

    fn join(&self, other: &Self) -> Self {
        Self::new(
            self.interval.join(&other.interval),
            self.congruence.join(&other.congruence),
        )
    }

    fn meet(&self, other: &Self) -> Self {
        Self::new(
            self.interval.meet(&other.interval),
            self.congruence.meet(&other.congruence),
        )
    }

    fn widen(&self, other: &Self) -> Self {
        Self::new(
            self.interval.widen(&other.interval),
            self.congruence.join(&other.congruence), // Congruence has finite height
        )
    }

    fn leq(&self, other: &Self) -> bool {
        self.interval.leq(&other.interval) && self.congruence.leq(&other.congruence)
    }

    fn is_bottom(&self) -> bool {
        self.interval.is_bottom() || self.congruence.is_bottom()
    }

    fn is_top(&self) -> bool {
        self.interval.is_top() && self.congruence.is_top()
    }
}

impl fmt::Display for IntervalCongruence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.interval, self.congruence)
    }
}

/// Full numeric domain combining Interval, Sign, and Congruence
///
/// Provides comprehensive numeric analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericDomain {
    pub interval: Interval,
    pub sign: Sign,
    pub congruence: Congruence,
}

impl NumericDomain {
    /// Create a new numeric domain
    pub fn new(interval: Interval, sign: Sign, congruence: Congruence) -> Self {
        let mut result = Self {
            interval,
            sign,
            congruence,
        };
        result.reduce();
        result
    }

    /// Create from a constant
    pub fn constant(n: i64) -> Self {
        Self::new(
            Interval::constant(n),
            Sign::from_int(n),
            Congruence::constant(n),
        )
    }

    /// Reduce all components
    fn reduce(&mut self) {
        // Check for any bottom
        if self.interval.is_bottom() || self.sign.is_bottom() || self.congruence.is_bottom() {
            self.interval = Interval::Bottom;
            self.sign = Sign::bottom();
            self.congruence = Congruence::Bottom;
            return;
        }

        // Reduce interval-sign
        let interval_sign = IntervalSign::interval_to_sign(&self.interval);
        self.sign = self.sign.meet(&interval_sign);

        let sign_interval = IntervalSign::sign_to_interval(&self.sign);
        self.interval = self.interval.meet(&sign_interval);

        // Reduce interval-congruence
        if let Some(c) = self.interval.is_constant() {
            self.congruence = self.congruence.meet(&Congruence::constant(c));
        }
        if let Some(c) = self.congruence.is_constant() {
            self.interval = self.interval.meet(&Interval::constant(c));
            self.sign = self.sign.meet(&Sign::from_int(c));
        }
    }

    /// Abstract addition
    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            self.interval.add(&other.interval),
            self.sign.add(&other.sign),
            self.congruence.add(&other.congruence),
        )
    }

    /// Abstract subtraction
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(
            self.interval.sub(&other.interval),
            self.sign.sub(&other.sign),
            self.congruence.sub(&other.congruence),
        )
    }

    /// Abstract multiplication
    pub fn mul(&self, other: &Self) -> Self {
        Self::new(
            self.interval.mul(&other.interval),
            self.sign.mul(&other.sign),
            self.congruence.mul(&other.congruence),
        )
    }

    /// Abstract negation
    pub fn neg(&self) -> Self {
        Self::new(self.interval.neg(), self.sign.neg(), self.congruence.neg())
    }
}

impl Default for NumericDomain {
    fn default() -> Self {
        Self::top()
    }
}

impl AbstractDomain for NumericDomain {
    fn top() -> Self {
        Self {
            interval: Interval::top(),
            sign: Sign::top(),
            congruence: Congruence::top(),
        }
    }

    fn bottom() -> Self {
        Self {
            interval: Interval::Bottom,
            sign: Sign::bottom(),
            congruence: Congruence::Bottom,
        }
    }

    fn join(&self, other: &Self) -> Self {
        Self::new(
            self.interval.join(&other.interval),
            self.sign.join(&other.sign),
            self.congruence.join(&other.congruence),
        )
    }

    fn meet(&self, other: &Self) -> Self {
        Self::new(
            self.interval.meet(&other.interval),
            self.sign.meet(&other.sign),
            self.congruence.meet(&other.congruence),
        )
    }

    fn widen(&self, other: &Self) -> Self {
        Self::new(
            self.interval.widen(&other.interval),
            self.sign.join(&other.sign),
            self.congruence.join(&other.congruence),
        )
    }

    fn leq(&self, other: &Self) -> bool {
        self.interval.leq(&other.interval)
            && self.sign.leq(&other.sign)
            && self.congruence.leq(&other.congruence)
    }

    fn is_bottom(&self) -> bool {
        self.interval.is_bottom() || self.sign.is_bottom() || self.congruence.is_bottom()
    }

    fn is_top(&self) -> bool {
        self.interval.is_top() && self.sign.is_top() && self.congruence.is_top()
    }
}

impl fmt::Display for NumericDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.interval, self.sign, self.congruence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_basic() {
        let prod: Product<Interval, Sign> =
            Product::new(Interval::from_range(1, 10), Sign::positive());

        assert!(!prod.is_bottom());
        assert!(!prod.is_top());
    }

    #[test]
    fn test_interval_sign_reduction() {
        // Interval [1, 10] should reduce sign to positive
        let is = IntervalSign::from_interval(Interval::from_range(1, 10));
        assert!(is.sign.is_definitely_positive());

        // Interval [-5, 5] should have all signs
        let is2 = IntervalSign::from_interval(Interval::from_range(-5, 5));
        assert!(is2.sign.may_be_negative());
        assert!(is2.sign.may_be_zero());
        assert!(is2.sign.may_be_positive());
    }

    #[test]
    fn test_interval_sign_reduction_reverse() {
        // Sign positive should reduce interval to [1, +inf]
        let is = IntervalSign::from_sign(Sign::positive());
        assert!(is.interval.is_positive());
    }

    #[test]
    fn test_interval_sign_ops() {
        let a = IntervalSign::constant(5);
        let b = IntervalSign::constant(3);

        let sum = a.add(&b);
        assert_eq!(sum.interval.is_constant(), Some(8));
        assert!(sum.sign.is_definitely_positive());

        let diff = a.sub(&b);
        assert_eq!(diff.interval.is_constant(), Some(2));
        assert!(diff.sign.is_definitely_positive());
    }

    #[test]
    fn test_interval_congruence_reduction() {
        // Constant interval should make congruence constant
        let ic = IntervalCongruence::new(Interval::constant(42), Congruence::even());
        assert_eq!(ic.congruence.is_constant(), Some(42));
    }

    #[test]
    fn test_interval_congruence_alignment() {
        let ic = IntervalCongruence::new(Interval::from_range(0, 100), Congruence::divisible_by(8));

        assert!(ic.is_aligned(8));
        assert!(ic.is_aligned(4));
        assert!(ic.is_even());
    }

    #[test]
    fn test_numeric_domain() {
        let nd = NumericDomain::constant(42);

        assert_eq!(nd.interval.is_constant(), Some(42));
        assert!(nd.sign.is_definitely_positive());
        assert!(nd.congruence.is_even());
    }

    #[test]
    fn test_numeric_domain_ops() {
        let a = NumericDomain::constant(10);
        let b = NumericDomain::constant(3);

        let sum = a.add(&b);
        assert_eq!(sum.interval.is_constant(), Some(13));
        assert!(sum.sign.is_definitely_positive());
        assert!(sum.congruence.is_odd());
    }

    #[test]
    fn test_product_join() {
        let a = IntervalSign::constant(5);
        let b = IntervalSign::constant(10);

        let joined = a.join(&b);

        assert!(joined.interval.contains(5));
        assert!(joined.interval.contains(10));
        assert!(joined.sign.is_definitely_positive());
    }

    #[test]
    fn test_product_meet() {
        let a = IntervalSign::new(Interval::from_range(0, 10), Sign::top());
        let b = IntervalSign::new(Interval::from_range(5, 15), Sign::positive());

        let met = a.meet(&b);

        // Interval intersection: [5, 10]
        assert!(!met.interval.contains(4));
        assert!(met.interval.contains(5));
        assert!(met.interval.contains(10));
        assert!(!met.interval.contains(11));

        // Sign is refined to positive
        assert!(met.sign.is_definitely_positive());
    }

    #[test]
    fn test_product_bottom() {
        let a = IntervalSign::new(
            Interval::from_range(1, 10),
            Sign::negative(), // Contradicts interval
        );

        // Reduction should detect bottom
        assert!(a.is_bottom());
    }
}
