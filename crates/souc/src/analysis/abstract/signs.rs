//! Sign Domain
//!
//! A simple abstract domain that tracks the sign of numeric values.
//! While less precise than intervals, it is very efficient and
//! useful for quick checks.
//!
//! # Lattice Structure
//!
//! ```text
//!            Top (-, 0, +)
//!          /   |   \
//!      (-,0) (0,+) (-,+)
//!       /  \   |   /  \
//!      -    0     +
//!       \   |   /
//!         Bottom
//! ```

use super::domain::{AbstractDomain, BoolDomain};
use std::fmt;

/// Sign domain - tracks possible signs of a value
///
/// Represents which signs (negative, zero, positive) a value may have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sign {
    /// May be negative (< 0)
    pub neg: bool,
    /// May be zero (= 0)
    pub zero: bool,
    /// May be positive (> 0)
    pub pos: bool,
}

impl Sign {
    /// Create a sign domain with specific flags
    pub fn new(neg: bool, zero: bool, pos: bool) -> Self {
        Self { neg, zero, pos }
    }

    /// Definitely negative
    pub fn negative() -> Self {
        Self::new(true, false, false)
    }

    /// Definitely zero
    pub fn zero_val() -> Self {
        Self::new(false, true, false)
    }

    /// Definitely positive
    pub fn positive() -> Self {
        Self::new(false, false, true)
    }

    /// Non-negative (>= 0)
    pub fn non_negative() -> Self {
        Self::new(false, true, true)
    }

    /// Non-positive (<= 0)
    pub fn non_positive() -> Self {
        Self::new(true, true, false)
    }

    /// Non-zero (!= 0)
    pub fn non_zero() -> Self {
        Self::new(true, false, true)
    }

    /// Create from a concrete integer
    pub fn from_int(n: i64) -> Self {
        if n < 0 {
            Self::negative()
        } else if n == 0 {
            Self::zero_val()
        } else {
            Self::positive()
        }
    }

    /// Check if this is definitely negative
    pub fn is_definitely_negative(&self) -> bool {
        self.neg && !self.zero && !self.pos
    }

    /// Check if this is definitely zero
    pub fn is_definitely_zero(&self) -> bool {
        !self.neg && self.zero && !self.pos
    }

    /// Check if this is definitely positive
    pub fn is_definitely_positive(&self) -> bool {
        !self.neg && !self.zero && self.pos
    }

    /// Check if this may be negative
    pub fn may_be_negative(&self) -> bool {
        self.neg
    }

    /// Check if this may be zero
    pub fn may_be_zero(&self) -> bool {
        self.zero
    }

    /// Check if this may be positive
    pub fn may_be_positive(&self) -> bool {
        self.pos
    }

    /// Check if this is definitely non-negative
    pub fn is_non_negative(&self) -> bool {
        !self.neg
    }

    /// Check if this is definitely non-positive
    pub fn is_non_positive(&self) -> bool {
        !self.pos
    }

    /// Check if this is definitely non-zero
    pub fn is_non_zero(&self) -> bool {
        !self.zero
    }

    /// Abstract addition
    pub fn add(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }

        let mut result = Self::bottom();

        // neg + neg = neg
        if self.neg && other.neg {
            result.neg = true;
        }
        // neg + zero = neg
        if self.neg && other.zero {
            result.neg = true;
        }
        // zero + neg = neg
        if self.zero && other.neg {
            result.neg = true;
        }
        // neg + pos = any
        if self.neg && other.pos {
            result = Self::top();
        }
        // pos + neg = any
        if self.pos && other.neg {
            result = Self::top();
        }
        // zero + zero = zero
        if self.zero && other.zero {
            result.zero = true;
        }
        // zero + pos = pos
        if self.zero && other.pos {
            result.pos = true;
        }
        // pos + zero = pos
        if self.pos && other.zero {
            result.pos = true;
        }
        // pos + pos = pos
        if self.pos && other.pos {
            result.pos = true;
        }

        result
    }

    /// Abstract subtraction
    pub fn sub(&self, other: &Self) -> Self {
        // a - b = a + (-b)
        self.add(&other.neg())
    }

    /// Abstract multiplication
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }

        // Zero times anything is zero
        if self.is_definitely_zero() || other.is_definitely_zero() {
            return Self::zero_val();
        }

        let mut result = Self::bottom();

        // neg * neg = pos
        if self.neg && other.neg {
            result.pos = true;
        }
        // neg * zero = zero
        if self.neg && other.zero {
            result.zero = true;
        }
        // neg * pos = neg
        if self.neg && other.pos {
            result.neg = true;
        }
        // zero * anything = zero
        if self.zero {
            result.zero = true;
        }
        if other.zero {
            result.zero = true;
        }
        // pos * neg = neg
        if self.pos && other.neg {
            result.neg = true;
        }
        // pos * pos = pos
        if self.pos && other.pos {
            result.pos = true;
        }

        result
    }

    /// Abstract division
    pub fn div(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }

        // Division by zero check
        if other.is_definitely_zero() {
            return Self::bottom(); // Error - undefined
        }

        // Division might include zero
        if other.may_be_zero() {
            // Partial undefined behavior
            return Self::top();
        }

        // Zero divided by non-zero is zero
        if self.is_definitely_zero() {
            return Self::zero_val();
        }

        let mut result = Self::bottom();

        // neg / neg = pos (integer division might be 0)
        if self.neg && other.neg {
            result.pos = true;
            result.zero = true; // -1 / -2 = 0
        }
        // neg / pos = neg or zero
        if self.neg && other.pos {
            result.neg = true;
            result.zero = true;
        }
        // pos / neg = neg or zero
        if self.pos && other.neg {
            result.neg = true;
            result.zero = true;
        }
        // pos / pos = pos or zero
        if self.pos && other.pos {
            result.pos = true;
            result.zero = true;
        }
        // zero / non-zero = zero
        if self.zero && other.is_non_zero() {
            result.zero = true;
        }

        result
    }

    /// Abstract remainder
    pub fn rem(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }

        if other.may_be_zero() {
            return Self::top(); // Division by zero possible
        }

        // Remainder has same sign as dividend (in Rust/C semantics)
        let mut result = Self::bottom();

        if self.neg {
            result.neg = true;
            result.zero = true;
        }
        if self.zero {
            result.zero = true;
        }
        if self.pos {
            result.pos = true;
            result.zero = true;
        }

        result
    }

    /// Abstract negation
    pub fn neg(&self) -> Self {
        Self {
            neg: self.pos,
            zero: self.zero,
            pos: self.neg,
        }
    }

    /// Abstract absolute value
    pub fn abs(&self) -> Self {
        if self.is_bottom() {
            return Self::bottom();
        }
        Self {
            neg: false,
            zero: self.zero,
            pos: self.neg || self.pos,
        }
    }

    /// Abstract less-than comparison
    pub fn lt(&self, other: &Self) -> BoolDomain {
        if self.is_bottom() || other.is_bottom() {
            return BoolDomain::Bottom;
        }

        // neg < pos is always true
        if self.is_definitely_negative() && other.is_definitely_positive() {
            return BoolDomain::True;
        }

        // neg < zero can be true
        if self.is_definitely_negative() && other.is_definitely_zero() {
            return BoolDomain::True;
        }

        // zero < pos can be true
        if self.is_definitely_zero() && other.is_definitely_positive() {
            return BoolDomain::True;
        }

        // pos < neg is always false
        if self.is_definitely_positive() && other.is_definitely_negative() {
            return BoolDomain::False;
        }

        // zero < neg is always false
        if self.is_definitely_zero() && other.is_definitely_negative() {
            return BoolDomain::False;
        }

        // pos < zero is always false
        if self.is_definitely_positive() && other.is_definitely_zero() {
            return BoolDomain::False;
        }

        // zero < zero is always false
        if self.is_definitely_zero() && other.is_definitely_zero() {
            return BoolDomain::False;
        }

        BoolDomain::Top
    }

    /// Abstract less-than-or-equal comparison
    pub fn le(&self, other: &Self) -> BoolDomain {
        if self.is_bottom() || other.is_bottom() {
            return BoolDomain::Bottom;
        }

        // neg <= pos is always true
        if self.is_definitely_negative()
            && (other.is_definitely_positive() || other.is_definitely_zero())
        {
            return BoolDomain::True;
        }

        // zero <= zero is true
        if self.is_definitely_zero() && other.may_be_zero() {
            if other.is_definitely_zero() {
                return BoolDomain::True;
            }
        }

        // zero <= pos is true
        if self.is_definitely_zero() && other.is_definitely_positive() {
            return BoolDomain::True;
        }

        // pos <= neg is always false
        if self.is_definitely_positive() && other.is_definitely_negative() {
            return BoolDomain::False;
        }

        BoolDomain::Top
    }

    /// Abstract equality comparison
    pub fn eq_cmp(&self, other: &Self) -> BoolDomain {
        if self.is_bottom() || other.is_bottom() {
            return BoolDomain::Bottom;
        }

        // Zero can only equal zero
        if self.is_definitely_zero() && other.is_definitely_zero() {
            return BoolDomain::True;
        }

        // If no common signs, definitely not equal
        if !self.neg && other.is_definitely_negative() {
            return BoolDomain::False;
        }
        if !self.pos && other.is_definitely_positive() {
            return BoolDomain::False;
        }
        if !self.zero && other.is_definitely_zero() {
            return BoolDomain::False;
        }
        if self.is_definitely_negative() && !other.neg {
            return BoolDomain::False;
        }
        if self.is_definitely_positive() && !other.pos {
            return BoolDomain::False;
        }
        if self.is_definitely_zero() && !other.zero {
            return BoolDomain::False;
        }

        BoolDomain::Top
    }

    /// Refine assuming positive
    pub fn assume_positive(&self) -> Self {
        Self {
            neg: false,
            zero: false,
            pos: self.pos,
        }
    }

    /// Refine assuming non-negative
    pub fn assume_non_negative(&self) -> Self {
        Self {
            neg: false,
            zero: self.zero,
            pos: self.pos,
        }
    }

    /// Refine assuming negative
    pub fn assume_negative(&self) -> Self {
        Self {
            neg: self.neg,
            zero: false,
            pos: false,
        }
    }

    /// Refine assuming non-positive
    pub fn assume_non_positive(&self) -> Self {
        Self {
            neg: self.neg,
            zero: self.zero,
            pos: false,
        }
    }

    /// Refine assuming zero
    pub fn assume_zero(&self) -> Self {
        Self {
            neg: false,
            zero: self.zero,
            pos: false,
        }
    }

    /// Refine assuming non-zero
    pub fn assume_non_zero(&self) -> Self {
        Self {
            neg: self.neg,
            zero: false,
            pos: self.pos,
        }
    }
}

impl Default for Sign {
    fn default() -> Self {
        Self::top()
    }
}

impl AbstractDomain for Sign {
    fn top() -> Self {
        Self {
            neg: true,
            zero: true,
            pos: true,
        }
    }

    fn bottom() -> Self {
        Self {
            neg: false,
            zero: false,
            pos: false,
        }
    }

    fn join(&self, other: &Self) -> Self {
        Self {
            neg: self.neg || other.neg,
            zero: self.zero || other.zero,
            pos: self.pos || other.pos,
        }
    }

    fn meet(&self, other: &Self) -> Self {
        Self {
            neg: self.neg && other.neg,
            zero: self.zero && other.zero,
            pos: self.pos && other.pos,
        }
    }

    fn leq(&self, other: &Self) -> bool {
        (!self.neg || other.neg) && (!self.zero || other.zero) && (!self.pos || other.pos)
    }

    fn is_bottom(&self) -> bool {
        !self.neg && !self.zero && !self.pos
    }

    fn is_top(&self) -> bool {
        self.neg && self.zero && self.pos
    }
}

impl fmt::Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bottom() {
            write!(f, "bottom")
        } else if self.is_top() {
            write!(f, "top")
        } else {
            let mut parts = Vec::new();
            if self.neg {
                parts.push("-");
            }
            if self.zero {
                parts.push("0");
            }
            if self.pos {
                parts.push("+");
            }
            write!(f, "{{{}}}", parts.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_creation() {
        let neg = Sign::negative();
        assert!(neg.is_definitely_negative());
        assert!(!neg.may_be_zero());
        assert!(!neg.may_be_positive());

        let pos = Sign::positive();
        assert!(pos.is_definitely_positive());
        assert!(!pos.may_be_zero());
        assert!(!pos.may_be_negative());

        let zero = Sign::zero_val();
        assert!(zero.is_definitely_zero());
    }

    #[test]
    fn test_sign_from_int() {
        assert!(Sign::from_int(-5).is_definitely_negative());
        assert!(Sign::from_int(0).is_definitely_zero());
        assert!(Sign::from_int(5).is_definitely_positive());
    }

    #[test]
    fn test_sign_add() {
        let neg = Sign::negative();
        let pos = Sign::positive();
        let zero = Sign::zero_val();

        // neg + neg = neg
        assert!(neg.add(&neg).is_definitely_negative());

        // pos + pos = pos
        assert!(pos.add(&pos).is_definitely_positive());

        // zero + zero = zero
        assert!(zero.add(&zero).is_definitely_zero());

        // neg + pos = any
        assert!(neg.add(&pos).is_top());

        // zero + pos = pos
        assert!(zero.add(&pos).is_definitely_positive());
    }

    #[test]
    fn test_sign_mul() {
        let neg = Sign::negative();
        let pos = Sign::positive();
        let zero = Sign::zero_val();

        // neg * neg = pos
        assert!(neg.mul(&neg).is_definitely_positive());

        // neg * pos = neg
        assert!(neg.mul(&pos).is_definitely_negative());

        // pos * pos = pos
        assert!(pos.mul(&pos).is_definitely_positive());

        // anything * zero = zero
        assert!(neg.mul(&zero).is_definitely_zero());
        assert!(pos.mul(&zero).is_definitely_zero());
    }

    #[test]
    fn test_sign_neg() {
        let neg = Sign::negative();
        let pos = Sign::positive();
        let zero = Sign::zero_val();

        assert!(neg.neg().is_definitely_positive());
        assert!(pos.neg().is_definitely_negative());
        assert!(zero.neg().is_definitely_zero());
    }

    #[test]
    fn test_sign_abs() {
        let neg = Sign::negative();
        let pos = Sign::positive();
        let any = Sign::top();

        assert!(neg.abs().is_definitely_positive());
        assert!(pos.abs().is_definitely_positive());
        assert!(any.abs().is_non_negative());
    }

    #[test]
    fn test_sign_lt() {
        let neg = Sign::negative();
        let pos = Sign::positive();
        let zero = Sign::zero_val();

        assert_eq!(neg.lt(&pos), BoolDomain::True);
        assert_eq!(neg.lt(&zero), BoolDomain::True);
        assert_eq!(zero.lt(&pos), BoolDomain::True);
        assert_eq!(pos.lt(&neg), BoolDomain::False);
        assert_eq!(zero.lt(&zero), BoolDomain::False);
    }

    #[test]
    fn test_sign_join() {
        let neg = Sign::negative();
        let pos = Sign::positive();

        let joined = neg.join(&pos);
        assert!(joined.may_be_negative());
        assert!(!joined.may_be_zero());
        assert!(joined.may_be_positive());
    }

    #[test]
    fn test_sign_meet() {
        let neg_zero = Sign::non_positive();
        let zero_pos = Sign::non_negative();

        let met = neg_zero.meet(&zero_pos);
        assert!(met.is_definitely_zero());
    }

    #[test]
    fn test_sign_assume() {
        let any = Sign::top();

        let pos = any.assume_positive();
        assert!(pos.is_definitely_positive());

        let neg = any.assume_negative();
        assert!(neg.is_definitely_negative());

        let non_neg = any.assume_non_negative();
        assert!(non_neg.is_non_negative());
    }

    #[test]
    fn test_sign_div() {
        let neg = Sign::negative();
        let pos = Sign::positive();
        let zero = Sign::zero_val();

        // Division by zero
        let div_by_zero = pos.div(&zero);
        assert!(div_by_zero.is_bottom());

        // pos / pos might be 0 or pos (integer division)
        let pp = pos.div(&pos);
        assert!(pp.may_be_positive());
        assert!(pp.may_be_zero());

        // neg / neg might be 0 or pos
        let nn = neg.div(&neg);
        assert!(nn.may_be_positive());
        assert!(nn.may_be_zero());
    }

    #[test]
    fn test_sign_lattice() {
        let neg = Sign::negative();
        let pos = Sign::positive();

        // Bottom is least
        assert!(Sign::bottom().leq(&neg));
        assert!(Sign::bottom().leq(&pos));

        // Top is greatest
        assert!(neg.leq(&Sign::top()));
        assert!(pos.leq(&Sign::top()));

        // Specific values are less than joined
        assert!(neg.leq(&neg.join(&pos)));
        assert!(pos.leq(&neg.join(&pos)));
    }
}
