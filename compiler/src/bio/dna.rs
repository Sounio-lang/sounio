//! DNA Sequence Types and Operators
//!
//! Implements DNA transformations from the preprint's operator catalog (Appendix A):
//! - Shift (σ): Cyclic permutation
//! - Reverse (ρ): Sequence reversal
//! - Complement (κ): Watson-Crick base pairing
//! - Edit: Point mutations (breaks bijectivity)
//!
//! # Dihedral Group Structure
//!
//! The bijective operators form a dihedral group D_n:
//! ```text
//! σ^n = id              (n-fold shift returns to start)
//! ρ^2 = id              (reverse is involution)
//! σ ∘ ρ = ρ ∘ σ^(-1)    (anticommutation relation)
//! ```
//!
//! Complement extends this to a larger group:
//! ```text
//! κ^2 = id              (complement is involution)
//! κ ∘ σ = σ ∘ κ         (commutes with shift)
//! κ ∘ ρ = ρ ∘ κ         (commutes with reverse)
//! ```

use std::fmt;

/// DNA nucleotide base
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Base {
    /// Adenine
    A = 0,
    /// Cytosine
    C = 1,
    /// Guanine
    G = 2,
    /// Thymine (or Uracil in RNA)
    T = 3,
}

impl Base {
    /// Get Watson-Crick complement
    pub fn complement(self) -> Self {
        match self {
            Base::A => Base::T,
            Base::T => Base::A,
            Base::C => Base::G,
            Base::G => Base::C,
        }
    }

    /// Convert to GF(4) element (0, 1, 2, 3)
    pub fn to_gf4(self) -> u8 {
        self as u8
    }

    /// Create from GF(4) element
    pub fn from_gf4(val: u8) -> Self {
        match val & 0x03 {
            0 => Base::A,
            1 => Base::C,
            2 => Base::G,
            _ => Base::T,
        }
    }

    /// Parse from character
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'A' => Some(Base::A),
            'C' => Some(Base::C),
            'G' => Some(Base::G),
            'T' | 'U' => Some(Base::T),
            _ => None,
        }
    }

    /// Convert to character
    pub fn to_char(self) -> char {
        match self {
            Base::A => 'A',
            Base::C => 'C',
            Base::G => 'G',
            Base::T => 'T',
        }
    }
}

impl fmt::Display for Base {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// DNA sequence with operator support
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DNAString {
    /// Sequence of bases
    pub bases: Vec<Base>,
}

impl DNAString {
    /// Create new DNA string from bases
    pub fn new(bases: &[Base]) -> Self {
        Self {
            bases: bases.to_vec(),
        }
    }

    /// Parse from string (e.g., "ACGT")
    pub fn from_str(s: &str) -> Option<Self> {
        let bases: Option<Vec<Base>> = s.chars().map(Base::from_char).collect();
        bases.map(|b| Self { bases: b })
    }

    /// Get sequence length
    pub fn len(&self) -> usize {
        self.bases.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    // ========== Operators from the Preprint ==========

    /// Cyclic shift operator (σ): moves last element to front
    ///
    /// σ(ACGT) = TACG
    pub fn shift(&self) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        let mut new_bases = Vec::with_capacity(self.len());
        new_bases.push(self.bases[self.len() - 1]);
        new_bases.extend_from_slice(&self.bases[..self.len() - 1]);
        Self { bases: new_bases }
    }

    /// Shift by k positions
    pub fn shift_by(&self, k: usize) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        let k = k % self.len();
        let split = self.len() - k;
        let mut new_bases = Vec::with_capacity(self.len());
        new_bases.extend_from_slice(&self.bases[split..]);
        new_bases.extend_from_slice(&self.bases[..split]);
        Self { bases: new_bases }
    }

    /// Reverse operator (ρ): reverses sequence
    ///
    /// ρ(ACGT) = TGCA
    pub fn reverse(&self) -> Self {
        Self {
            bases: self.bases.iter().rev().copied().collect(),
        }
    }

    /// Complement operator (κ): Watson-Crick pairing
    ///
    /// κ(ACGT) = TGCA
    pub fn complement(&self) -> Self {
        Self {
            bases: self.bases.iter().map(|b| b.complement()).collect(),
        }
    }

    /// Reverse complement (ρ ∘ κ = κ ∘ ρ)
    ///
    /// This is the standard "antisense" strand operation
    pub fn reverse_complement(&self) -> Self {
        Self {
            bases: self.bases.iter().rev().map(|b| b.complement()).collect(),
        }
    }

    /// Point edit (mutation) - NOT bijective!
    ///
    /// This breaks the group structure as it's not invertible
    pub fn edit(&self, pos: usize, new_base: Base) -> Self {
        let mut new_bases = self.bases.clone();
        if pos < new_bases.len() {
            new_bases[pos] = new_base;
        }
        Self { bases: new_bases }
    }

    /// Insert base at position - NOT bijective!
    pub fn insert(&self, pos: usize, base: Base) -> Self {
        let mut new_bases = self.bases.clone();
        if pos <= new_bases.len() {
            new_bases.insert(pos, base);
        }
        Self { bases: new_bases }
    }

    /// Delete base at position - NOT bijective!
    pub fn delete(&self, pos: usize) -> Self {
        let mut new_bases = self.bases.clone();
        if pos < new_bases.len() {
            new_bases.remove(pos);
        }
        Self { bases: new_bases }
    }

    // ========== Metrics ==========

    /// Calculate GC content (fraction of G+C bases)
    pub fn gc_content(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let gc_count = self
            .bases
            .iter()
            .filter(|b| matches!(b, Base::G | Base::C))
            .count();
        gc_count as f64 / self.len() as f64
    }

    /// Hamming distance to another sequence
    pub fn hamming_distance(&self, other: &Self) -> Option<usize> {
        if self.len() != other.len() {
            return None;
        }
        Some(
            self.bases
                .iter()
                .zip(other.bases.iter())
                .filter(|(a, b)| a != b)
                .count(),
        )
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.bases.iter().map(|b| b.to_char()).collect()
    }
}

impl fmt::Display for DNAString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for base in &self.bases {
            write!(f, "{}", base)?;
        }
        Ok(())
    }
}

/// DNA operator type (for composition)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DNAOperator {
    /// Identity
    Id,
    /// Shift by k
    Shift(usize),
    /// Reverse
    Reverse,
    /// Complement
    Complement,
    /// Reverse then Complement
    ReverseComplement,
}

impl DNAOperator {
    /// Apply operator to DNA string
    pub fn apply(&self, dna: &DNAString) -> DNAString {
        match self {
            DNAOperator::Id => dna.clone(),
            DNAOperator::Shift(k) => dna.shift_by(*k),
            DNAOperator::Reverse => dna.reverse(),
            DNAOperator::Complement => dna.complement(),
            DNAOperator::ReverseComplement => dna.reverse_complement(),
        }
    }

    /// Compose two operators (self ∘ other)
    pub fn compose(&self, other: &DNAOperator) -> DNAOperator {
        // Simplified composition rules
        match (self, other) {
            (DNAOperator::Id, op) => *op,
            (op, DNAOperator::Id) => *op,
            (DNAOperator::Reverse, DNAOperator::Reverse) => DNAOperator::Id,
            (DNAOperator::Complement, DNAOperator::Complement) => DNAOperator::Id,
            (DNAOperator::Reverse, DNAOperator::Complement) => DNAOperator::ReverseComplement,
            (DNAOperator::Complement, DNAOperator::Reverse) => DNAOperator::ReverseComplement,
            (DNAOperator::ReverseComplement, DNAOperator::ReverseComplement) => DNAOperator::Id,
            _ => {
                // For complex compositions, just return self (simplified)
                *self
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_complement() {
        assert_eq!(Base::A.complement(), Base::T);
        assert_eq!(Base::T.complement(), Base::A);
        assert_eq!(Base::C.complement(), Base::G);
        assert_eq!(Base::G.complement(), Base::C);
    }

    #[test]
    fn test_dna_shift() {
        let dna = DNAString::from_str("ACGT").unwrap();
        let shifted = dna.shift();
        assert_eq!(shifted.to_string(), "TACG");
    }

    #[test]
    fn test_dna_reverse() {
        let dna = DNAString::from_str("ACGT").unwrap();
        let reversed = dna.reverse();
        assert_eq!(reversed.to_string(), "TGCA");
    }

    #[test]
    fn test_dna_complement() {
        let dna = DNAString::from_str("ACGT").unwrap();
        let comp = dna.complement();
        assert_eq!(comp.to_string(), "TGCA");
    }

    #[test]
    fn test_dna_reverse_complement() {
        let dna = DNAString::from_str("ACGT").unwrap();
        let rc = dna.reverse_complement();
        assert_eq!(rc.to_string(), "ACGT"); // ACGT is its own reverse complement!
    }

    #[test]
    fn test_complement_involution() {
        let dna = DNAString::from_str("ACGTACGT").unwrap();
        let double_comp = dna.complement().complement();
        assert_eq!(dna, double_comp);
    }

    #[test]
    fn test_reverse_involution() {
        let dna = DNAString::from_str("ACGTACGT").unwrap();
        let double_rev = dna.reverse().reverse();
        assert_eq!(dna, double_rev);
    }

    #[test]
    fn test_gc_content() {
        let dna = DNAString::from_str("ACGT").unwrap();
        assert!((dna.gc_content() - 0.5).abs() < 1e-10);

        let gc_rich = DNAString::from_str("GGCC").unwrap();
        assert!((gc_rich.gc_content() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_distance() {
        let dna1 = DNAString::from_str("ACGT").unwrap();
        let dna2 = DNAString::from_str("ACGG").unwrap();
        assert_eq!(dna1.hamming_distance(&dna2), Some(1));

        let dna3 = DNAString::from_str("TGCA").unwrap();
        assert_eq!(dna1.hamming_distance(&dna3), Some(4));
    }

    #[test]
    fn test_operator_composition() {
        let comp = DNAOperator::Complement;
        let rev = DNAOperator::Reverse;
        let composed = rev.compose(&comp);
        assert_eq!(composed, DNAOperator::ReverseComplement);
    }
}
