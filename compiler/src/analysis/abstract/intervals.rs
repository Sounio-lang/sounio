//! Interval Domain
//!
//! The interval domain abstracts integer values as ranges [lo, hi].
//! It is one of the most widely used numerical abstract domains due to
//! its simplicity and efficiency.
//!
//! # Theory
//!
//! An interval [l, u] represents the set of integers {x | l <= x <= u}.
//! Special values +inf and -inf represent unbounded intervals.
//!
//! # References
//!
//! - Cousot, P., & Cousot, R. (1976). Static determination of dynamic
//!   properties of programs.

use super::domain::{AbstractDomain, BoolDomain};
use std::cmp::{max, min};
use std::fmt;

/// Bound of an interval - can be finite or infinite
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bound {
    /// Negative infinity
    NegInf,
    /// Finite bound
    Finite(i64),
    /// Positive infinity
    PosInf,
}

impl Bound {
    /// Check if this is negative infinity
    pub fn is_neg_inf(&self) -> bool {
        matches!(self, Bound::NegInf)
    }

    /// Check if this is positive infinity
    pub fn is_pos_inf(&self) -> bool {
        matches!(self, Bound::PosInf)
    }

    /// Check if this is finite
    pub fn is_finite(&self) -> bool {
        matches!(self, Bound::Finite(_))
    }

    /// Get the finite value, or None if infinite
    pub fn as_finite(&self) -> Option<i64> {
        match self {
            Bound::Finite(n) => Some(*n),
            _ => None,
        }
    }

    /// Minimum of two bounds
    pub fn min(self, other: Self) -> Self {
        match (self, other) {
            (Bound::NegInf, _) | (_, Bound::NegInf) => Bound::NegInf,
            (Bound::PosInf, x) | (x, Bound::PosInf) => x,
            (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(min(a, b)),
        }
    }

    /// Maximum of two bounds
    pub fn max(self, other: Self) -> Self {
        match (self, other) {
            (Bound::PosInf, _) | (_, Bound::PosInf) => Bound::PosInf,
            (Bound::NegInf, x) | (x, Bound::NegInf) => x,
            (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(max(a, b)),
        }
    }

    /// Negate a bound (for subtraction)
    pub fn negate(self) -> Self {
        match self {
            Bound::NegInf => Bound::PosInf,
            Bound::PosInf => Bound::NegInf,
            Bound::Finite(n) => Bound::Finite(-n),
        }
    }

    /// Add two bounds
    pub fn add(self, other: Self) -> Self {
        match (self, other) {
            (Bound::NegInf, Bound::PosInf) | (Bound::PosInf, Bound::NegInf) => {
                // Indeterminate - return top-like bound
                Bound::PosInf
            }
            (Bound::NegInf, _) | (_, Bound::NegInf) => Bound::NegInf,
            (Bound::PosInf, _) | (_, Bound::PosInf) => Bound::PosInf,
            (Bound::Finite(a), Bound::Finite(b)) => {
                // Handle overflow
                match a.checked_add(b) {
                    Some(c) => Bound::Finite(c),
                    None if a > 0 => Bound::PosInf,
                    None => Bound::NegInf,
                }
            }
        }
    }

    /// Subtract two bounds
    pub fn sub(self, other: Self) -> Self {
        self.add(other.negate())
    }

    /// Multiply two bounds (for lower bound computation)
    pub fn mul_lo(self, other: Self) -> Self {
        match (self, other) {
            (Bound::Finite(0), _) | (_, Bound::Finite(0)) => Bound::Finite(0),
            (Bound::NegInf, Bound::PosInf) | (Bound::PosInf, Bound::NegInf) => Bound::NegInf,
            (Bound::NegInf, Bound::Finite(n)) if n > 0 => Bound::NegInf,
            (Bound::Finite(n), Bound::NegInf) if n > 0 => Bound::NegInf,
            (Bound::PosInf, Bound::Finite(n)) if n < 0 => Bound::NegInf,
            (Bound::Finite(n), Bound::PosInf) if n < 0 => Bound::NegInf,
            (Bound::PosInf, Bound::PosInf) | (Bound::NegInf, Bound::NegInf) => Bound::PosInf,
            (Bound::NegInf, Bound::Finite(n)) if n < 0 => Bound::PosInf,
            (Bound::Finite(n), Bound::NegInf) if n < 0 => Bound::PosInf,
            (Bound::PosInf, Bound::Finite(n)) if n > 0 => Bound::PosInf,
            (Bound::Finite(n), Bound::PosInf) if n > 0 => Bound::PosInf,
            (Bound::Finite(a), Bound::Finite(b)) => match a.checked_mul(b) {
                Some(c) => Bound::Finite(c),
                None if (a > 0) == (b > 0) => Bound::PosInf,
                None => Bound::NegInf,
            },
            _ => Bound::NegInf, // Conservative
        }
    }
}

impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bound {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Bound::NegInf, Bound::NegInf) => std::cmp::Ordering::Equal,
            (Bound::NegInf, _) => std::cmp::Ordering::Less,
            (_, Bound::NegInf) => std::cmp::Ordering::Greater,
            (Bound::PosInf, Bound::PosInf) => std::cmp::Ordering::Equal,
            (Bound::PosInf, _) => std::cmp::Ordering::Greater,
            (_, Bound::PosInf) => std::cmp::Ordering::Less,
            (Bound::Finite(a), Bound::Finite(b)) => a.cmp(b),
        }
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bound::NegInf => write!(f, "-inf"),
            Bound::PosInf => write!(f, "+inf"),
            Bound::Finite(n) => write!(f, "{}", n),
        }
    }
}

/// Interval domain for integer abstract interpretation.
///
/// Represents sets of integers as closed intervals [lo, hi].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interval {
    /// Empty interval (no values - unreachable)
    Bottom,
    /// Non-empty interval [lo, hi]
    Range { lo: Bound, hi: Bound },
}

impl Interval {
    /// Create an interval from bounds
    pub fn new(lo: Bound, hi: Bound) -> Self {
        if lo > hi {
            Interval::Bottom
        } else {
            Interval::Range { lo, hi }
        }
    }

    /// Create an interval for a single value
    pub fn constant(n: i64) -> Self {
        Interval::Range {
            lo: Bound::Finite(n),
            hi: Bound::Finite(n),
        }
    }

    /// Create an interval from finite bounds
    pub fn from_range(lo: i64, hi: i64) -> Self {
        Self::new(Bound::Finite(lo), Bound::Finite(hi))
    }

    /// Create an interval [lo, +inf)
    pub fn from_lower(lo: i64) -> Self {
        Interval::Range {
            lo: Bound::Finite(lo),
            hi: Bound::PosInf,
        }
    }

    /// Create an interval (-inf, hi]
    pub fn to_upper(hi: i64) -> Self {
        Interval::Range {
            lo: Bound::NegInf,
            hi: Bound::Finite(hi),
        }
    }

    /// Get the lower bound
    pub fn lo(&self) -> Option<Bound> {
        match self {
            Interval::Bottom => None,
            Interval::Range { lo, .. } => Some(*lo),
        }
    }

    /// Get the upper bound
    pub fn hi(&self) -> Option<Bound> {
        match self {
            Interval::Bottom => None,
            Interval::Range { hi, .. } => Some(*hi),
        }
    }

    /// Check if this interval contains a value
    pub fn contains(&self, n: i64) -> bool {
        match self {
            Interval::Bottom => false,
            Interval::Range { lo, hi } => *lo <= Bound::Finite(n) && Bound::Finite(n) <= *hi,
        }
    }

    /// Check if this interval may contain zero
    pub fn may_contain_zero(&self) -> bool {
        self.contains(0)
    }

    /// Check if this interval is definitely positive
    pub fn is_positive(&self) -> bool {
        match self {
            Interval::Bottom => false,
            Interval::Range { lo, .. } => *lo > Bound::Finite(0),
        }
    }

    /// Check if this interval is definitely non-negative
    pub fn is_non_negative(&self) -> bool {
        match self {
            Interval::Bottom => false,
            Interval::Range { lo, .. } => *lo >= Bound::Finite(0),
        }
    }

    /// Check if this interval is definitely negative
    pub fn is_negative(&self) -> bool {
        match self {
            Interval::Bottom => false,
            Interval::Range { hi, .. } => *hi < Bound::Finite(0),
        }
    }

    /// Check if this interval is definitely non-positive
    pub fn is_non_positive(&self) -> bool {
        match self {
            Interval::Bottom => false,
            Interval::Range { hi, .. } => *hi <= Bound::Finite(0),
        }
    }

    /// Check if this is a singleton interval
    pub fn is_constant(&self) -> Option<i64> {
        match self {
            Interval::Range {
                lo: Bound::Finite(a),
                hi: Bound::Finite(b),
            } if a == b => Some(*a),
            _ => None,
        }
    }

    /// Abstract addition
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => Interval::new(l1.add(*l2), h1.add(*h2)),
        }
    }

    /// Abstract subtraction
    pub fn sub(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => Interval::new(l1.sub(*h2), h1.sub(*l2)),
        }
    }

    /// Abstract multiplication
    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => {
                // Compute all four corners and take min/max
                let corners = [
                    l1.mul_lo(*l2),
                    l1.mul_lo(*h2),
                    h1.mul_lo(*l2),
                    h1.mul_lo(*h2),
                ];
                let lo = corners
                    .iter()
                    .cloned()
                    .min()
                    .unwrap_or(Bound::NegInf);
                let hi = corners
                    .iter()
                    .cloned()
                    .max()
                    .unwrap_or(Bound::PosInf);
                Interval::new(lo, hi)
            }
        }
    }

    /// Abstract division (signed)
    pub fn div(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (_, divisor) if divisor.may_contain_zero() => {
                // Division by zero possible - return top
                Interval::top()
            }
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => {
                // Divisor doesn't contain zero
                let corners = match (l1.as_finite(), h1.as_finite(), l2.as_finite(), h2.as_finite())
                {
                    (Some(a), Some(b), Some(c), Some(d)) => {
                        vec![a / c, a / d, b / c, b / d]
                    }
                    _ => return Interval::top(),
                };
                let lo = corners.iter().cloned().min().unwrap_or(i64::MIN);
                let hi = corners.iter().cloned().max().unwrap_or(i64::MAX);
                Interval::from_range(lo, hi)
            }
        }
    }

    /// Abstract remainder
    pub fn rem(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (_, divisor) if divisor.may_contain_zero() => Interval::top(),
            (Interval::Range { .. }, Interval::Range { lo: l2, hi: h2 }) => {
                // Result is bounded by divisor
                match (l2.as_finite(), h2.as_finite()) {
                    (Some(l), Some(h)) => {
                        let bound = max(l.abs(), h.abs()) - 1;
                        Interval::from_range(-bound, bound)
                    }
                    _ => Interval::top(),
                }
            }
        }
    }

    /// Abstract negation
    pub fn neg(&self) -> Self {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Range { lo, hi } => Interval::new(hi.negate(), lo.negate()),
        }
    }

    /// Abstract less-than comparison
    pub fn lt(&self, other: &Self) -> BoolDomain {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => BoolDomain::Bottom,
            (
                Interval::Range { hi: h1, .. },
                Interval::Range { lo: l2, .. },
            ) => {
                if *h1 < *l2 {
                    BoolDomain::True
                } else if self.lo().unwrap() >= other.hi().unwrap() {
                    BoolDomain::False
                } else {
                    BoolDomain::Top
                }
            }
        }
    }

    /// Abstract less-than-or-equal comparison
    pub fn le(&self, other: &Self) -> BoolDomain {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => BoolDomain::Bottom,
            (
                Interval::Range { hi: h1, .. },
                Interval::Range { lo: l2, .. },
            ) => {
                if *h1 <= *l2 {
                    BoolDomain::True
                } else if self.lo().unwrap() > other.hi().unwrap() {
                    BoolDomain::False
                } else {
                    BoolDomain::Top
                }
            }
        }
    }

    /// Abstract equality comparison
    pub fn eq(&self, other: &Self) -> BoolDomain {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => BoolDomain::Bottom,
            _ => {
                if let (Some(a), Some(b)) = (self.is_constant(), other.is_constant()) {
                    if a == b {
                        BoolDomain::True
                    } else {
                        BoolDomain::False
                    }
                } else if self.meet(other).is_bottom() {
                    BoolDomain::False
                } else {
                    BoolDomain::Top
                }
            }
        }
    }

    /// Abstract not-equal comparison
    pub fn ne(&self, other: &Self) -> BoolDomain {
        self.eq(other).not()
    }

    /// Refine interval assuming condition is true
    /// For x < c, return intersection of self with (-inf, c-1]
    pub fn assume_lt(&self, bound: i64) -> Self {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Range { lo, .. } => {
                Interval::new(*lo, Bound::Finite(bound - 1)).meet(self)
            }
        }
    }

    /// Refine interval assuming x <= c
    pub fn assume_le(&self, bound: i64) -> Self {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Range { lo, .. } => {
                Interval::new(*lo, Bound::Finite(bound)).meet(self)
            }
        }
    }

    /// Refine interval assuming x > c
    pub fn assume_gt(&self, bound: i64) -> Self {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Range { hi, .. } => {
                Interval::new(Bound::Finite(bound + 1), *hi).meet(self)
            }
        }
    }

    /// Refine interval assuming x >= c
    pub fn assume_ge(&self, bound: i64) -> Self {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Range { hi, .. } => {
                Interval::new(Bound::Finite(bound), *hi).meet(self)
            }
        }
    }
}

impl Default for Interval {
    fn default() -> Self {
        Interval::top()
    }
}

impl AbstractDomain for Interval {
    fn top() -> Self {
        Interval::Range {
            lo: Bound::NegInf,
            hi: Bound::PosInf,
        }
    }

    fn bottom() -> Self {
        Interval::Bottom
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, x) | (x, Interval::Bottom) => *x,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => Interval::Range {
                lo: (*l1).min(*l2),
                hi: (*h1).max(*h2),
            },
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => Interval::new((*l1).max(*l2), (*h1).min(*h2)),
        }
    }

    fn widen(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, x) | (x, Interval::Bottom) => *x,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => {
                let new_lo = if *l2 < *l1 { Bound::NegInf } else { *l1 };
                let new_hi = if *h2 > *h1 { Bound::PosInf } else { *h1 };
                Interval::Range {
                    lo: new_lo,
                    hi: new_hi,
                }
            }
        }
    }

    fn narrow(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => {
                let new_lo = if l1.is_neg_inf() { *l2 } else { *l1 };
                let new_hi = if h1.is_pos_inf() { *h2 } else { *h1 };
                Interval::new(new_lo, new_hi)
            }
        }
    }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (Interval::Bottom, _) => true,
            (_, Interval::Bottom) => false,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => *l2 <= *l1 && *h1 <= *h2,
        }
    }

    fn is_bottom(&self) -> bool {
        matches!(self, Interval::Bottom)
    }

    fn is_top(&self) -> bool {
        matches!(
            self,
            Interval::Range {
                lo: Bound::NegInf,
                hi: Bound::PosInf
            }
        )
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Interval::Bottom => write!(f, "bottom"),
            Interval::Range { lo, hi } => write!(f, "[{}, {}]", lo, hi),
        }
    }
}

/// Widening with thresholds for more precise iteration
#[derive(Debug, Clone)]
pub struct ThresholdWidening {
    /// Sorted thresholds for widening
    thresholds: Vec<i64>,
}

impl ThresholdWidening {
    /// Create threshold widening with given thresholds
    pub fn new(mut thresholds: Vec<i64>) -> Self {
        thresholds.sort();
        thresholds.dedup();
        Self { thresholds }
    }

    /// Default thresholds: powers of 2, common loop bounds
    pub fn default_thresholds() -> Self {
        Self::new(vec![
            -1024, -256, -128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256,
            512, 1024, i32::MAX as i64,
        ])
    }

    /// Widen interval with thresholds
    pub fn widen(&self, old: &Interval, new: &Interval) -> Interval {
        match (old, new) {
            (Interval::Bottom, x) | (x, Interval::Bottom) => *x,
            (
                Interval::Range { lo: l1, hi: h1 },
                Interval::Range { lo: l2, hi: h2 },
            ) => {
                let new_lo = if *l2 < *l1 {
                    // Find largest threshold <= l2
                    self.find_threshold_below(*l2)
                } else {
                    *l1
                };
                let new_hi = if *h2 > *h1 {
                    // Find smallest threshold >= h2
                    self.find_threshold_above(*h2)
                } else {
                    *h1
                };
                Interval::new(new_lo, new_hi)
            }
        }
    }

    fn find_threshold_below(&self, bound: Bound) -> Bound {
        match bound {
            Bound::NegInf => Bound::NegInf,
            Bound::PosInf => {
                if let Some(&t) = self.thresholds.last() {
                    Bound::Finite(t)
                } else {
                    Bound::NegInf
                }
            }
            Bound::Finite(n) => {
                for &t in self.thresholds.iter().rev() {
                    if t <= n {
                        return Bound::Finite(t);
                    }
                }
                Bound::NegInf
            }
        }
    }

    fn find_threshold_above(&self, bound: Bound) -> Bound {
        match bound {
            Bound::PosInf => Bound::PosInf,
            Bound::NegInf => {
                if let Some(&t) = self.thresholds.first() {
                    Bound::Finite(t)
                } else {
                    Bound::PosInf
                }
            }
            Bound::Finite(n) => {
                for &t in &self.thresholds {
                    if t >= n {
                        return Bound::Finite(t);
                    }
                }
                Bound::PosInf
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_creation() {
        let i = Interval::from_range(1, 10);
        assert!(i.contains(5));
        assert!(!i.contains(0));
        assert!(!i.contains(11));
    }

    #[test]
    fn test_interval_constant() {
        let c = Interval::constant(42);
        assert_eq!(c.is_constant(), Some(42));
        assert!(c.contains(42));
        assert!(!c.contains(43));
    }

    #[test]
    fn test_interval_join() {
        let a = Interval::from_range(0, 5);
        let b = Interval::from_range(3, 10);
        let joined = a.join(&b);
        assert!(joined.contains(0));
        assert!(joined.contains(10));
    }

    #[test]
    fn test_interval_meet() {
        let a = Interval::from_range(0, 5);
        let b = Interval::from_range(3, 10);
        let met = a.meet(&b);
        assert!(!met.contains(0));
        assert!(met.contains(3));
        assert!(met.contains(5));
        assert!(!met.contains(6));
    }

    #[test]
    fn test_interval_add() {
        let a = Interval::from_range(1, 5);
        let b = Interval::from_range(10, 20);
        let sum = a.add(&b);
        assert!(sum.contains(11)); // 1 + 10
        assert!(sum.contains(25)); // 5 + 20
        assert!(!sum.contains(10));
    }

    #[test]
    fn test_interval_sub() {
        let a = Interval::from_range(10, 20);
        let b = Interval::from_range(1, 5);
        let diff = a.sub(&b);
        assert!(diff.contains(5)); // 10 - 5
        assert!(diff.contains(19)); // 20 - 1
    }

    #[test]
    fn test_interval_mul() {
        let a = Interval::from_range(2, 3);
        let b = Interval::from_range(4, 5);
        let prod = a.mul(&b);
        assert!(prod.contains(8)); // 2 * 4
        assert!(prod.contains(15)); // 3 * 5
    }

    #[test]
    fn test_interval_neg() {
        let a = Interval::from_range(1, 5);
        let neg = a.neg();
        assert!(neg.contains(-1));
        assert!(neg.contains(-5));
        assert!(!neg.contains(1));
    }

    #[test]
    fn test_interval_widen() {
        let a = Interval::from_range(0, 5);
        let b = Interval::from_range(0, 10);
        let widened = a.widen(&b);
        assert!(widened.hi().unwrap().is_pos_inf());
    }

    #[test]
    fn test_interval_narrow() {
        let wide = Interval::new(Bound::Finite(0), Bound::PosInf);
        let precise = Interval::from_range(0, 100);
        let narrowed = wide.narrow(&precise);
        assert_eq!(narrowed.hi(), Some(Bound::Finite(100)));
    }

    #[test]
    fn test_interval_lt() {
        let a = Interval::from_range(0, 5);
        let b = Interval::from_range(10, 20);
        assert_eq!(a.lt(&b), BoolDomain::True);

        let c = Interval::from_range(3, 7);
        assert_eq!(a.lt(&c), BoolDomain::Top);
    }

    #[test]
    fn test_interval_eq() {
        let a = Interval::constant(5);
        let b = Interval::constant(5);
        let c = Interval::constant(10);
        assert_eq!(a.eq(&b), BoolDomain::True);
        assert_eq!(a.eq(&c), BoolDomain::False);

        let d = Interval::from_range(0, 10);
        assert_eq!(a.eq(&d), BoolDomain::Top);
    }

    #[test]
    fn test_interval_assume() {
        let i = Interval::from_range(0, 100);

        let lt50 = i.assume_lt(50);
        assert!(lt50.contains(49));
        assert!(!lt50.contains(50));

        let gt50 = i.assume_gt(50);
        assert!(gt50.contains(51));
        assert!(!gt50.contains(50));
    }

    #[test]
    fn test_threshold_widening() {
        let tw = ThresholdWidening::default_thresholds();

        let a = Interval::from_range(0, 5);
        let b = Interval::from_range(0, 10);
        let widened = tw.widen(&a, &b);

        // Should widen to threshold, not infinity
        assert!(!widened.hi().unwrap().is_pos_inf());
        assert!(widened.contains(16)); // Next threshold
    }

    #[test]
    fn test_interval_is_positive() {
        assert!(Interval::from_range(1, 10).is_positive());
        assert!(!Interval::from_range(0, 10).is_positive());
        assert!(!Interval::from_range(-1, 10).is_positive());
    }

    #[test]
    fn test_interval_is_non_negative() {
        assert!(Interval::from_range(0, 10).is_non_negative());
        assert!(Interval::from_range(1, 10).is_non_negative());
        assert!(!Interval::from_range(-1, 10).is_non_negative());
    }

    #[test]
    fn test_interval_lattice_properties() {
        let a = Interval::from_range(0, 5);
        let b = Interval::from_range(3, 10);

        // Join is upper bound
        assert!(a.leq(&a.join(&b)));
        assert!(b.leq(&a.join(&b)));

        // Meet is lower bound
        assert!(a.meet(&b).leq(&a));
        assert!(a.meet(&b).leq(&b));

        // Bottom is least
        assert!(Interval::bottom().leq(&a));

        // Top is greatest
        assert!(a.leq(&Interval::top()));
    }
}
