//! Congruence Domain
//!
//! Tracks values modulo some modulus: value = remainder (mod modulus).
//! Useful for array index analysis and alignment checking.
//!
//! # Theory
//!
//! A congruence (a, m) represents the set {a + k*m | k in Z}.
//! Special cases:
//! - (a, 1) represents all integers (top)
//! - (a, 0) represents exactly {a} (constant)
//!
//! # References
//!
//! - Granger, P. (1989). Static analysis of linear congruence equalities.

use super::domain::AbstractDomain;
use std::fmt;

/// Greatest common divisor using Euclidean algorithm
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x, y) = extended_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

/// Congruence domain: value = remainder (mod modulus)
///
/// Represents sets of integers with the same remainder modulo some value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Congruence {
    /// No values (unreachable)
    Bottom,
    /// value = remainder (mod modulus)
    /// - modulus = 0 means exact value
    /// - modulus = 1 means all values (top)
    Cong { modulus: u64, remainder: u64 },
}

impl Congruence {
    /// Create a congruence, normalizing the representation
    pub fn new(modulus: u64, remainder: u64) -> Self {
        if modulus == 0 {
            // Exact constant
            Congruence::Cong { modulus: 0, remainder }
        } else if modulus == 1 {
            // All values
            Congruence::Cong { modulus: 1, remainder: 0 }
        } else {
            // Normalize remainder to be less than modulus
            Congruence::Cong {
                modulus,
                remainder: remainder % modulus,
            }
        }
    }

    /// Create a congruence for a constant value
    pub fn constant(value: i64) -> Self {
        Congruence::Cong {
            modulus: 0,
            remainder: value as u64,
        }
    }

    /// Create a congruence for values divisible by n
    pub fn divisible_by(n: u64) -> Self {
        if n == 0 {
            // Only 0 is divisible by 0
            Self::constant(0)
        } else {
            Self::new(n, 0)
        }
    }

    /// Create a congruence for even numbers
    pub fn even() -> Self {
        Self::divisible_by(2)
    }

    /// Create a congruence for odd numbers
    pub fn odd() -> Self {
        Self::new(2, 1)
    }

    /// Get the modulus
    pub fn modulus(&self) -> Option<u64> {
        match self {
            Congruence::Bottom => None,
            Congruence::Cong { modulus, .. } => Some(*modulus),
        }
    }

    /// Get the remainder
    pub fn remainder(&self) -> Option<u64> {
        match self {
            Congruence::Bottom => None,
            Congruence::Cong { remainder, .. } => Some(*remainder),
        }
    }

    /// Check if this is a constant
    pub fn is_constant(&self) -> Option<i64> {
        match self {
            Congruence::Cong { modulus: 0, remainder } => Some(*remainder as i64),
            _ => None,
        }
    }

    /// Check if the value may be divisible by n
    pub fn may_be_divisible_by(&self, n: u64) -> bool {
        match self {
            Congruence::Bottom => false,
            Congruence::Cong { modulus: 1, .. } => true, // Top
            Congruence::Cong { modulus: 0, remainder } => *remainder % n == 0,
            Congruence::Cong { modulus, remainder } => {
                // Check if remainder mod gcd(modulus, n) == 0
                let g = gcd(*modulus, n);
                *remainder % g == 0
            }
        }
    }

    /// Check if the value is definitely divisible by n
    pub fn is_divisible_by(&self, n: u64) -> bool {
        match self {
            Congruence::Bottom => false,
            Congruence::Cong { modulus: 1, .. } => false, // Top - can be anything
            Congruence::Cong { modulus: 0, remainder } => *remainder % n == 0,
            Congruence::Cong { modulus, remainder } => {
                // Check if n divides both modulus and remainder
                *modulus % n == 0 && *remainder % n == 0
            }
        }
    }

    /// Check if definitely aligned to n bytes (for pointer alignment)
    pub fn is_aligned(&self, alignment: u64) -> bool {
        self.is_divisible_by(alignment)
    }

    /// Check if may be aligned to n bytes
    pub fn may_be_aligned(&self, alignment: u64) -> bool {
        self.may_be_divisible_by(alignment)
    }

    /// Check if definitely even
    pub fn is_even(&self) -> bool {
        self.is_divisible_by(2)
    }

    /// Check if definitely odd
    pub fn is_odd(&self) -> bool {
        match self {
            Congruence::Cong { modulus, remainder } if *modulus % 2 == 0 => {
                *remainder % 2 == 1
            }
            Congruence::Cong { modulus: 0, remainder } => *remainder % 2 == 1,
            _ => false,
        }
    }

    /// Abstract addition
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Congruence::Bottom, _) | (_, Congruence::Bottom) => Congruence::Bottom,
            (
                Congruence::Cong { modulus: m1, remainder: r1 },
                Congruence::Cong { modulus: m2, remainder: r2 },
            ) => {
                let new_modulus = gcd(*m1, *m2);
                let new_remainder = r1.wrapping_add(*r2);
                Congruence::new(new_modulus, new_remainder)
            }
        }
    }

    /// Abstract subtraction
    pub fn sub(&self, other: &Self) -> Self {
        match (self, other) {
            (Congruence::Bottom, _) | (_, Congruence::Bottom) => Congruence::Bottom,
            (
                Congruence::Cong { modulus: m1, remainder: r1 },
                Congruence::Cong { modulus: m2, remainder: r2 },
            ) => {
                let new_modulus = gcd(*m1, *m2);
                let new_remainder = r1.wrapping_sub(*r2);
                Congruence::new(new_modulus, new_remainder)
            }
        }
    }

    /// Abstract multiplication
    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Congruence::Bottom, _) | (_, Congruence::Bottom) => Congruence::Bottom,
            (
                Congruence::Cong { modulus: 0, remainder: r1 },
                Congruence::Cong { modulus: 0, remainder: r2 },
            ) => {
                // Constant * constant
                Congruence::constant((*r1 as i64) * (*r2 as i64))
            }
            (
                Congruence::Cong { modulus: 0, remainder: c },
                Congruence::Cong { modulus: m, remainder: r },
            )
            | (
                Congruence::Cong { modulus: m, remainder: r },
                Congruence::Cong { modulus: 0, remainder: c },
            ) => {
                // Constant * congruence
                if *c == 0 {
                    Congruence::constant(0)
                } else {
                    Congruence::new(m * c, r * c)
                }
            }
            (
                Congruence::Cong { modulus: m1, remainder: r1 },
                Congruence::Cong { modulus: m2, remainder: r2 },
            ) => {
                // General case: (a + k1*m1) * (b + k2*m2)
                // = a*b + a*k2*m2 + b*k1*m1 + k1*k2*m1*m2
                // = a*b (mod gcd(a*m2, b*m1, m1*m2))
                let g1 = r1 * m2;
                let g2 = r2 * m1;
                let g3 = m1 * m2;
                let new_modulus = gcd(gcd(g1, g2), g3);
                let new_remainder = r1 * r2;
                Congruence::new(new_modulus, new_remainder)
            }
        }
    }

    /// Abstract division (loses precision)
    pub fn div(&self, other: &Self) -> Self {
        match (self, other) {
            (Congruence::Bottom, _) | (_, Congruence::Bottom) => Congruence::Bottom,
            (
                Congruence::Cong { modulus: 0, remainder: r1 },
                Congruence::Cong { modulus: 0, remainder: r2 },
            ) if *r2 != 0 => {
                // Constant / constant
                Congruence::constant((*r1 as i64) / (*r2 as i64))
            }
            (_, Congruence::Cong { modulus: 0, remainder: 0 }) => {
                // Division by zero
                Congruence::Bottom
            }
            _ => {
                // Lose precision for non-constant division
                Congruence::top()
            }
        }
    }

    /// Abstract remainder
    pub fn rem(&self, other: &Self) -> Self {
        match (self, other) {
            (Congruence::Bottom, _) | (_, Congruence::Bottom) => Congruence::Bottom,
            (
                Congruence::Cong { modulus: 0, remainder: r1 },
                Congruence::Cong { modulus: 0, remainder: r2 },
            ) if *r2 != 0 => {
                // Constant % constant
                Congruence::constant((*r1 as i64) % (*r2 as i64))
            }
            (_, Congruence::Cong { modulus: 0, remainder: 0 }) => {
                // Remainder by zero
                Congruence::Bottom
            }
            (cong, Congruence::Cong { modulus: 0, remainder: c }) => {
                // x % c has same congruence as x (mod gcd(m, c))
                match cong {
                    Congruence::Cong { modulus: m, remainder: r } => {
                        let new_modulus = gcd(*m, *c);
                        Congruence::new(new_modulus, *r)
                    }
                    Congruence::Bottom => Congruence::Bottom,
                }
            }
            _ => Congruence::top(),
        }
    }

    /// Abstract negation
    pub fn neg(&self) -> Self {
        match self {
            Congruence::Bottom => Congruence::Bottom,
            Congruence::Cong { modulus, remainder } => {
                if *modulus == 0 {
                    Congruence::constant(-(*remainder as i64))
                } else {
                    Congruence::new(*modulus, modulus - (*remainder % modulus))
                }
            }
        }
    }

    /// Abstract left shift
    pub fn shl(&self, bits: u32) -> Self {
        match self {
            Congruence::Bottom => Congruence::Bottom,
            Congruence::Cong { modulus, remainder } => {
                let shift = 1u64 << bits;
                Congruence::new(modulus.saturating_mul(shift), remainder.saturating_mul(shift))
            }
        }
    }

    /// Abstract right shift
    pub fn shr(&self, bits: u32) -> Self {
        match self {
            Congruence::Bottom => Congruence::Bottom,
            Congruence::Cong { modulus: 0, remainder } => {
                Congruence::constant((*remainder as i64) >> bits)
            }
            Congruence::Cong { modulus, remainder } => {
                let shift = 1u64 << bits;
                // Loses precision
                Congruence::new(modulus / shift, remainder / shift)
            }
        }
    }

    /// Abstract bitwise AND with constant
    pub fn and_const(&self, mask: u64) -> Self {
        match self {
            Congruence::Bottom => Congruence::Bottom,
            Congruence::Cong { modulus: 0, remainder } => {
                Congruence::constant((*remainder & mask) as i64)
            }
            _ => {
                // Bitwise AND loses most congruence information
                // But if mask = 2^n - 1, we can preserve mod 2^n info
                if mask.count_ones() == mask.trailing_ones() {
                    // Mask is of form 2^n - 1
                    let new_mod = mask + 1;
                    match self {
                        Congruence::Cong { modulus, remainder } => {
                            let g = gcd(*modulus, new_mod);
                            Congruence::new(g, *remainder % new_mod)
                        }
                        _ => Congruence::top(),
                    }
                } else {
                    Congruence::top()
                }
            }
        }
    }
}

impl Default for Congruence {
    fn default() -> Self {
        Congruence::top()
    }
}

impl AbstractDomain for Congruence {
    fn top() -> Self {
        Congruence::Cong {
            modulus: 1,
            remainder: 0,
        }
    }

    fn bottom() -> Self {
        Congruence::Bottom
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Congruence::Bottom, x) | (x, Congruence::Bottom) => *x,
            (
                Congruence::Cong { modulus: m1, remainder: r1 },
                Congruence::Cong { modulus: m2, remainder: r2 },
            ) => {
                // Join finds a congruence containing both
                // new_modulus = gcd(m1, m2, |r1 - r2|)
                let diff = if *r1 >= *r2 { r1 - r2 } else { r2 - r1 };
                let g1 = gcd(*m1, *m2);
                let new_modulus = gcd(g1, diff);
                // Both remainders should be equivalent mod new_modulus
                Congruence::new(new_modulus, *r1)
            }
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Congruence::Bottom, _) | (_, Congruence::Bottom) => Congruence::Bottom,
            (
                Congruence::Cong { modulus: m1, remainder: r1 },
                Congruence::Cong { modulus: m2, remainder: r2 },
            ) => {
                // Meet finds the intersection
                // Using Chinese Remainder Theorem if compatible
                if *m1 == 0 && *m2 == 0 {
                    // Both constants - must be equal
                    if r1 == r2 {
                        Congruence::constant(*r1 as i64)
                    } else {
                        Congruence::Bottom
                    }
                } else if *m1 == 0 {
                    // r1 must satisfy r1 = r2 (mod m2)
                    if *r1 % m2 == *r2 % m2 {
                        Congruence::constant(*r1 as i64)
                    } else {
                        Congruence::Bottom
                    }
                } else if *m2 == 0 {
                    if *r2 % m1 == *r1 % m1 {
                        Congruence::constant(*r2 as i64)
                    } else {
                        Congruence::Bottom
                    }
                } else {
                    // General case using CRT
                    let g = gcd(*m1, *m2);
                    let diff = if *r1 >= *r2 { r1 - r2 } else { r2 - r1 };
                    if diff % g != 0 {
                        // No solution
                        Congruence::Bottom
                    } else {
                        // lcm = m1 * m2 / gcd
                        let lcm = (m1 / g).saturating_mul(*m2);
                        // Find combined remainder using extended GCD
                        let (_g, x, _y) = extended_gcd(*m1 as i64, *m2 as i64);
                        let r = (*r1 as i64
                            + (*m1 as i64)
                                * x
                                * ((*r2 as i64 - *r1 as i64) / (g as i64)))
                            .rem_euclid(lcm as i64);
                        Congruence::new(lcm, r as u64)
                    }
                }
            }
        }
    }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (Congruence::Bottom, _) => true,
            (_, Congruence::Bottom) => false,
            (
                Congruence::Cong { modulus: m1, remainder: r1 },
                Congruence::Cong { modulus: m2, remainder: r2 },
            ) => {
                // self <= other iff gamma(self) subset gamma(other)
                // This happens when m2 divides m1 and r1 = r2 (mod m2)
                if *m2 == 1 {
                    true // other is top
                } else if *m2 == 0 {
                    *m1 == 0 && r1 == r2 // other is constant
                } else if *m1 == 0 {
                    *r1 % m2 == *r2 // self is constant, must satisfy other's congruence
                } else {
                    *m1 % m2 == 0 && *r1 % m2 == *r2
                }
            }
        }
    }

    fn is_bottom(&self) -> bool {
        matches!(self, Congruence::Bottom)
    }

    fn is_top(&self) -> bool {
        matches!(self, Congruence::Cong { modulus: 1, .. })
    }
}

impl fmt::Display for Congruence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Congruence::Bottom => write!(f, "bottom"),
            Congruence::Cong { modulus: 0, remainder } => write!(f, "{}", remainder),
            Congruence::Cong { modulus: 1, .. } => write!(f, "top"),
            Congruence::Cong { modulus, remainder } => {
                write!(f, "{} (mod {})", remainder, modulus)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_congruence_constant() {
        let c = Congruence::constant(42);
        assert_eq!(c.is_constant(), Some(42));
        assert!(c.is_divisible_by(2));
        assert!(c.is_divisible_by(7));
        assert!(!c.is_divisible_by(5));
    }

    #[test]
    fn test_congruence_even_odd() {
        let even = Congruence::even();
        let odd = Congruence::odd();

        assert!(even.is_even());
        assert!(!even.is_odd());
        assert!(!odd.is_even());
        assert!(odd.is_odd());
    }

    #[test]
    fn test_congruence_add() {
        let even = Congruence::even();
        let odd = Congruence::odd();

        // even + even = even
        let ee = even.add(&even);
        assert!(ee.is_even());

        // even + odd = odd
        let eo = even.add(&odd);
        assert!(eo.is_odd());

        // odd + odd = even
        let oo = odd.add(&odd);
        assert!(oo.is_even());
    }

    #[test]
    fn test_congruence_mul() {
        let c2 = Congruence::constant(2);
        let c3 = Congruence::constant(3);

        let prod = c2.mul(&c3);
        assert_eq!(prod.is_constant(), Some(6));

        // anything * even = even
        let any = Congruence::top();
        let even = Congruence::even();
        let ae = any.mul(&even);
        assert!(ae.is_even());
    }

    #[test]
    fn test_congruence_join() {
        let c4 = Congruence::constant(4);
        let c8 = Congruence::constant(8);

        let joined = c4.join(&c8);
        // Should be 0 (mod 4) since both are divisible by 4
        assert!(joined.is_divisible_by(4));
        assert!(!joined.is_constant().is_some());
    }

    #[test]
    fn test_congruence_meet() {
        let mod4 = Congruence::new(4, 0); // 0 mod 4
        let mod6 = Congruence::new(6, 0); // 0 mod 6

        let met = mod4.meet(&mod6);
        // CRT: should be 0 mod 12
        assert!(met.is_divisible_by(12));
    }

    #[test]
    fn test_congruence_meet_incompatible() {
        let c5 = Congruence::constant(5);
        let c7 = Congruence::constant(7);

        let met = c5.meet(&c7);
        assert!(met.is_bottom());
    }

    #[test]
    fn test_congruence_alignment() {
        let aligned8 = Congruence::divisible_by(8);
        assert!(aligned8.is_aligned(8));
        assert!(aligned8.is_aligned(4));
        assert!(aligned8.is_aligned(2));
        assert!(!aligned8.is_aligned(16));
        assert!(aligned8.may_be_aligned(16));
    }

    #[test]
    fn test_congruence_shift() {
        let c3 = Congruence::constant(3);

        let shl2 = c3.shl(2);
        assert_eq!(shl2.is_constant(), Some(12));

        let shr1 = c3.shr(1);
        assert_eq!(shr1.is_constant(), Some(1));
    }

    #[test]
    fn test_congruence_lattice() {
        let c4 = Congruence::constant(4);
        let mod4 = Congruence::new(4, 0);
        let mod2 = Congruence::new(2, 0);

        // Constant is more precise than congruence
        assert!(c4.leq(&mod4));
        assert!(c4.leq(&mod2));

        // Smaller modulus is less precise
        assert!(mod4.leq(&mod2));

        // Top is least precise
        assert!(c4.leq(&Congruence::top()));
        assert!(mod4.leq(&Congruence::top()));
    }

    #[test]
    fn test_congruence_neg() {
        let c5 = Congruence::constant(5);
        let neg = c5.neg();
        assert_eq!(neg.is_constant(), Some(-5));

        let odd = Congruence::odd();
        let neg_odd = odd.neg();
        assert!(neg_odd.is_odd());
    }

    #[test]
    fn test_congruence_and_const() {
        let any = Congruence::top();

        // x & 0xFF gives 0 mod 256
        let masked = any.and_const(0xFF);
        // Result should know value is bounded by mask
        assert!(masked.modulus().unwrap() <= 256);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(15, 10), 5);
        assert_eq!(gcd(7, 5), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
    }
}
