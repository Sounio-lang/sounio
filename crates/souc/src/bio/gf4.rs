//! GF(4) Galois Field Arithmetic
//!
//! Implements the 4-element Galois field for quaternary DNA encoding.
//! GF(4) = {0, 1, α, α²} where α² + α + 1 = 0.
//!
//! # Encoding
//!
//! | Base | GF(4) |
//! |------|-------|
//! | A    | 0     |
//! | C    | 1     |
//! | G    | α     |
//! | T    | α²    |
//!
//! # Operations
//!
//! Addition (XOR-like):
//! ```text
//! + | 0  1  α  α²
//! --+-------------
//! 0 | 0  1  α  α²
//! 1 | 1  0  α² α
//! α | α  α² 0  1
//! α²| α² α  1  0
//! ```
//!
//! Multiplication:
//! ```text
//! × | 0  1  α  α²
//! --+-------------
//! 0 | 0  0  0  0
//! 1 | 0  1  α  α²
//! α | 0  α  α² 1
//! α²| 0  α² 1  α
//! ```

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Element of GF(4)
///
/// Stored as a 2-bit value: 0, 1, 2 (α), 3 (α²)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GF4Element(u8);

impl GF4Element {
    /// Zero element
    pub const ZERO: Self = Self(0);
    /// One element
    pub const ONE: Self = Self(1);
    /// α element (primitive root)
    pub const ALPHA: Self = Self(2);
    /// α² element
    pub const ALPHA_SQ: Self = Self(3);

    /// Create from raw value (mod 4)
    pub const fn new(val: u8) -> Self {
        Self(val & 0x03)
    }

    /// Get raw value
    pub const fn value(&self) -> u8 {
        self.0
    }

    /// Check if zero
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Check if one
    pub const fn is_one(&self) -> bool {
        self.0 == 1
    }

    /// Additive inverse (same as self in GF(4))
    pub const fn additive_inverse(&self) -> Self {
        // In GF(4), -x = x (characteristic 2)
        *self
    }

    /// Multiplicative inverse (for non-zero elements)
    pub const fn multiplicative_inverse(&self) -> Option<Self> {
        match self.0 {
            0 => None, // Zero has no inverse
            1 => Some(Self::ONE),
            2 => Some(Self::ALPHA_SQ), // α⁻¹ = α²
            3 => Some(Self::ALPHA),    // (α²)⁻¹ = α
            _ => None,
        }
    }

    /// Convert to DNA base
    pub fn to_base(&self) -> super::dna::Base {
        super::dna::Base::from_gf4(self.0)
    }

    /// Create from DNA base
    pub fn from_base(base: super::dna::Base) -> Self {
        Self(base.to_gf4())
    }
}

/// Addition in GF(4)
///
/// Defined by the table:
/// 0+0=0, 0+1=1, 0+α=α, 0+α²=α²
/// 1+1=0, 1+α=α², 1+α²=α
/// α+α=0, α+α²=1
/// α²+α²=0
impl Add for GF4Element {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // GF(4) addition table encoded
        const ADD_TABLE: [[u8; 4]; 4] = [
            [0, 1, 2, 3], // 0 + x
            [1, 0, 3, 2], // 1 + x
            [2, 3, 0, 1], // α + x
            [3, 2, 1, 0], // α² + x
        ];
        Self(ADD_TABLE[self.0 as usize][other.0 as usize])
    }
}

/// Subtraction in GF(4) (same as addition, characteristic 2)
impl Sub for GF4Element {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + other // In characteristic 2, a - b = a + b
    }
}

/// Negation in GF(4) (identity, characteristic 2)
impl Neg for GF4Element {
    type Output = Self;

    fn neg(self) -> Self {
        self // -a = a in characteristic 2
    }
}

/// Multiplication in GF(4)
///
/// Defined by α² + α + 1 = 0, so α² = α + 1
impl Mul for GF4Element {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // GF(4) multiplication table
        const MUL_TABLE: [[u8; 4]; 4] = [
            [0, 0, 0, 0], // 0 × x
            [0, 1, 2, 3], // 1 × x
            [0, 2, 3, 1], // α × x (α×α=α², α×α²=1)
            [0, 3, 1, 2], // α² × x
        ];
        Self(MUL_TABLE[self.0 as usize][other.0 as usize])
    }
}

impl fmt::Display for GF4Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            0 => write!(f, "0"),
            1 => write!(f, "1"),
            2 => write!(f, "α"),
            3 => write!(f, "α²"),
            _ => write!(f, "?"),
        }
    }
}

/// GF(4) vector (for DNA sequences)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GF4 {
    /// Elements
    pub elements: Vec<GF4Element>,
}

impl GF4 {
    /// Create from elements
    pub fn new(elements: Vec<GF4Element>) -> Self {
        Self { elements }
    }

    /// Create from DNA string
    pub fn from_dna(dna: &super::dna::DNAString) -> Self {
        Self {
            elements: dna
                .bases
                .iter()
                .map(|b| GF4Element::from_base(*b))
                .collect(),
        }
    }

    /// Convert to DNA string
    pub fn to_dna(&self) -> super::dna::DNAString {
        super::dna::DNAString {
            bases: self.elements.iter().map(|e| e.to_base()).collect(),
        }
    }

    /// Length
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Add two vectors element-wise
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.len() != other.len() {
            return None;
        }
        Some(Self {
            elements: self
                .elements
                .iter()
                .zip(other.elements.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
        })
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: GF4Element) -> Self {
        Self {
            elements: self.elements.iter().map(|e| *e * scalar).collect(),
        }
    }

    /// Dot product
    pub fn dot(&self, other: &Self) -> Option<GF4Element> {
        if self.len() != other.len() {
            return None;
        }
        Some(
            self.elements
                .iter()
                .zip(other.elements.iter())
                .map(|(a, b)| *a * *b)
                .fold(GF4Element::ZERO, |acc, x| acc + x),
        )
    }

    /// Hamming weight (number of non-zero elements)
    pub fn weight(&self) -> usize {
        self.elements.iter().filter(|e| !e.is_zero()).count()
    }
}

impl fmt::Display for GF4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, e) in self.elements.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", e)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf4_addition() {
        let zero = GF4Element::ZERO;
        let one = GF4Element::ONE;
        let alpha = GF4Element::ALPHA;
        let alpha_sq = GF4Element::ALPHA_SQ;

        // 0 + x = x
        assert_eq!(zero + zero, zero);
        assert_eq!(zero + one, one);
        assert_eq!(zero + alpha, alpha);
        assert_eq!(zero + alpha_sq, alpha_sq);

        // x + x = 0 (characteristic 2)
        assert_eq!(one + one, zero);
        assert_eq!(alpha + alpha, zero);
        assert_eq!(alpha_sq + alpha_sq, zero);

        // 1 + α = α²
        assert_eq!(one + alpha, alpha_sq);
        // α + α² = 1
        assert_eq!(alpha + alpha_sq, one);
    }

    #[test]
    fn test_gf4_multiplication() {
        let zero = GF4Element::ZERO;
        let one = GF4Element::ONE;
        let alpha = GF4Element::ALPHA;
        let alpha_sq = GF4Element::ALPHA_SQ;

        // 0 × x = 0
        assert_eq!(zero * one, zero);
        assert_eq!(zero * alpha, zero);

        // 1 × x = x
        assert_eq!(one * alpha, alpha);
        assert_eq!(one * alpha_sq, alpha_sq);

        // α × α = α²
        assert_eq!(alpha * alpha, alpha_sq);
        // α × α² = 1
        assert_eq!(alpha * alpha_sq, one);
        // α² × α² = α
        assert_eq!(alpha_sq * alpha_sq, alpha);
    }

    #[test]
    fn test_multiplicative_inverse() {
        let one = GF4Element::ONE;
        let alpha = GF4Element::ALPHA;
        let alpha_sq = GF4Element::ALPHA_SQ;

        // 1⁻¹ = 1
        assert_eq!(one.multiplicative_inverse(), Some(one));
        // α⁻¹ = α²
        assert_eq!(alpha.multiplicative_inverse(), Some(alpha_sq));
        // (α²)⁻¹ = α
        assert_eq!(alpha_sq.multiplicative_inverse(), Some(alpha));

        // Verify: x × x⁻¹ = 1
        assert_eq!(alpha * alpha.multiplicative_inverse().unwrap(), one);
        assert_eq!(alpha_sq * alpha_sq.multiplicative_inverse().unwrap(), one);
    }

    #[test]
    fn test_gf4_vector_operations() {
        let v1 = GF4::new(vec![
            GF4Element::ZERO,
            GF4Element::ONE,
            GF4Element::ALPHA,
            GF4Element::ALPHA_SQ,
        ]);
        let v2 = GF4::new(vec![
            GF4Element::ONE,
            GF4Element::ONE,
            GF4Element::ALPHA_SQ,
            GF4Element::ALPHA,
        ]);

        // Addition
        let sum = v1.add(&v2).unwrap();
        assert_eq!(sum.elements[0], GF4Element::ONE); // 0 + 1 = 1
        assert_eq!(sum.elements[1], GF4Element::ZERO); // 1 + 1 = 0
        assert_eq!(sum.elements[2], GF4Element::ONE); // α + α² = 1
        assert_eq!(sum.elements[3], GF4Element::ONE); // α² + α = 1

        // Weight
        assert_eq!(v1.weight(), 3); // 1, α, α² are non-zero
    }

    #[test]
    fn test_dna_to_gf4_roundtrip() {
        use super::super::dna::{Base, DNAString};

        let dna = DNAString::new(&[Base::A, Base::C, Base::G, Base::T]);
        let gf4 = GF4::from_dna(&dna);
        let back = gf4.to_dna();

        assert_eq!(dna, back);
    }
}
