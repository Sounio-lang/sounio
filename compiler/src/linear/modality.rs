//! Modality types for linear epistemic types
//!
//! The four modalities form a lattice representing resource usage constraints:
//!
//! ```text
//!        Unrestricted (ω)
//!           /    \
//!       Affine  Relevant
//!         (≤1)    (≥1)
//!           \    /
//!          Linear (1)
//! ```
//!
//! - Linear: Use exactly once
//! - Affine: Use at most once (can discard)
//! - Relevant: Use at least once (must use)
//! - Unrestricted: Use any number of times

use std::fmt;

/// The four modalities of resource usage
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Modality {
    /// Use exactly once (most restrictive)
    /// Neither weakening nor contraction allowed
    #[default]
    Linear,

    /// Use at most once (can discard)
    /// Weakening allowed, contraction not allowed
    Affine,

    /// Use at least once (must use)
    /// Contraction allowed, weakening not allowed
    Relevant,

    /// Use any number of times (least restrictive)
    /// Both weakening and contraction allowed
    Unrestricted,
}

impl Modality {
    /// Combine modalities (meet in the lattice)
    ///
    /// The result is the most restrictive modality that satisfies both constraints.
    ///
    /// ```text
    /// combine(Linear, _) = Linear
    /// combine(_, Linear) = Linear
    /// combine(Affine, Relevant) = Linear
    /// combine(Relevant, Affine) = Linear
    /// combine(Affine, Affine) = Affine
    /// combine(Relevant, Relevant) = Relevant
    /// combine(Unrestricted, x) = x
    /// combine(x, Unrestricted) = x
    /// ```
    pub fn combine(self, other: Modality) -> Modality {
        use Modality::*;
        match (self, other) {
            // Linear is bottom of lattice
            (Linear, _) | (_, Linear) => Linear,

            // Affine + Relevant = Linear (neither can satisfy both)
            (Affine, Relevant) | (Relevant, Affine) => Linear,

            // Same modality
            (Affine, Affine) => Affine,
            (Relevant, Relevant) => Relevant,

            // Unrestricted is top of lattice
            (Unrestricted, x) | (x, Unrestricted) => x,
        }
    }

    /// Join modalities (join in the lattice)
    ///
    /// The result is the least restrictive modality that both can satisfy.
    pub fn join(self, other: Modality) -> Modality {
        use Modality::*;
        match (self, other) {
            // Unrestricted is top
            (Unrestricted, _) | (_, Unrestricted) => Unrestricted,

            // Affine + Relevant = Unrestricted (need both weakening and contraction)
            (Affine, Relevant) | (Relevant, Affine) => Unrestricted,

            // Same modality
            (Linear, Linear) => Linear,
            (Affine, Affine) => Affine,
            (Relevant, Relevant) => Relevant,

            // Linear + Affine = Affine
            (Linear, Affine) | (Affine, Linear) => Affine,

            // Linear + Relevant = Relevant
            (Linear, Relevant) | (Relevant, Linear) => Relevant,
        }
    }

    /// Check if this modality is a subtype of another
    ///
    /// A more restrictive modality is a subtype of a less restrictive one.
    /// Linear <: Affine, Linear <: Relevant
    /// Affine <: Unrestricted, Relevant <: Unrestricted
    pub fn is_subtype_of(self, other: Modality) -> bool {
        use Modality::*;
        match (self, other) {
            // Reflexivity
            (x, y) if x == y => true,

            // Linear is subtype of all
            (Linear, _) => true,

            // All are subtypes of Unrestricted
            (_, Unrestricted) => true,

            // Otherwise not a subtype
            _ => false,
        }
    }

    /// Check if weakening is allowed
    ///
    /// Weakening: Γ ⊢ A ⟹ Γ, B ⊢ A (can ignore resources)
    pub fn allows_weakening(self) -> bool {
        matches!(self, Modality::Affine | Modality::Unrestricted)
    }

    /// Check if contraction is allowed
    ///
    /// Contraction: Γ, A, A ⊢ B ⟹ Γ, A ⊢ B (can duplicate resources)
    pub fn allows_contraction(self) -> bool {
        matches!(self, Modality::Relevant | Modality::Unrestricted)
    }

    /// Check if the resource must be used
    pub fn must_use(self) -> bool {
        matches!(self, Modality::Linear | Modality::Relevant)
    }

    /// Check if the resource can be used multiple times
    pub fn can_reuse(self) -> bool {
        matches!(self, Modality::Relevant | Modality::Unrestricted)
    }

    /// Get the symbol for this modality
    pub fn symbol(self) -> &'static str {
        match self {
            Modality::Linear => "₁",
            Modality::Affine => "≤₁",
            Modality::Relevant => "≥₁",
            Modality::Unrestricted => "ω",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "linear" | "1" | "₁" => Some(Modality::Linear),
            "affine" | "?" | "≤1" | "≤₁" => Some(Modality::Affine),
            "relevant" | "!" | "≥1" | "≥₁" => Some(Modality::Relevant),
            "unrestricted" | "ω" | "*" => Some(Modality::Unrestricted),
            _ => None,
        }
    }
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modality::Linear => write!(f, "linear"),
            Modality::Affine => write!(f, "affine"),
            Modality::Relevant => write!(f, "relevant"),
            Modality::Unrestricted => write!(f, "unrestricted"),
        }
    }
}

impl PartialOrd for Modality {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        if self == other {
            return Some(Ordering::Equal);
        }

        // Linear is minimum
        if *self == Modality::Linear {
            return Some(Ordering::Less);
        }
        if *other == Modality::Linear {
            return Some(Ordering::Greater);
        }

        // Unrestricted is maximum
        if *self == Modality::Unrestricted {
            return Some(Ordering::Greater);
        }
        if *other == Modality::Unrestricted {
            return Some(Ordering::Less);
        }

        // Affine and Relevant are incomparable
        None
    }
}

/// Usage annotation for variables
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModalityAnnotation {
    /// The modality
    pub modality: Modality,
    /// Whether this was inferred or explicit
    pub inferred: bool,
}

impl ModalityAnnotation {
    pub fn new(modality: Modality) -> Self {
        Self {
            modality,
            inferred: false,
        }
    }

    pub fn inferred(modality: Modality) -> Self {
        Self {
            modality,
            inferred: true,
        }
    }
}

/// Grade for graded modal types (future extension)
///
/// A grade is an element of a semiring that tracks resource usage.
/// This is a generalization of modalities.
#[derive(Clone, Debug, PartialEq)]
pub enum Grade {
    /// Exact count
    Exact(usize),
    /// At most n
    AtMost(usize),
    /// At least n
    AtLeast(usize),
    /// Interval [min, max]
    Interval(usize, usize),
    /// Unbounded
    Omega,
}

impl Grade {
    /// Zero grade (unused)
    pub fn zero() -> Self {
        Grade::Exact(0)
    }

    /// One grade (used once)
    pub fn one() -> Self {
        Grade::Exact(1)
    }

    /// Omega grade (unbounded)
    pub fn omega() -> Self {
        Grade::Omega
    }

    /// Addition of grades
    pub fn add(self, other: Grade) -> Grade {
        use Grade::*;
        match (self, other) {
            (Exact(a), Exact(b)) => Exact(a + b),
            (Exact(a), AtMost(b)) => AtMost(a + b),
            (AtMost(a), Exact(b)) => AtMost(a + b),
            (AtMost(a), AtMost(b)) => AtMost(a + b),
            (Exact(a), AtLeast(b)) => AtLeast(a + b),
            (AtLeast(a), Exact(b)) => AtLeast(a + b),
            (AtLeast(a), AtLeast(b)) => AtLeast(a + b),
            (Interval(a1, a2), Interval(b1, b2)) => Interval(a1 + b1, a2 + b2),
            (Omega, _) | (_, Omega) => Omega,
            _ => Omega, // Simplified
        }
    }

    /// Multiplication of grades
    pub fn mul(self, other: Grade) -> Grade {
        use Grade::*;
        match (self, other) {
            (Exact(0), _) | (_, Exact(0)) => Exact(0),
            (Exact(a), Exact(b)) => Exact(a * b),
            (Omega, Omega) => Omega,
            _ => Omega, // Simplified
        }
    }

    /// Convert to modality (approximation)
    pub fn to_modality(&self) -> Modality {
        match self {
            Grade::Exact(0) => Modality::Affine,         // Can be unused
            Grade::Exact(1) => Modality::Linear,         // Exactly once
            Grade::AtMost(1) => Modality::Affine,        // At most once
            Grade::AtLeast(1) => Modality::Relevant,     // At least once
            Grade::Omega => Modality::Unrestricted,      // Unbounded
            Grade::Interval(0, 1) => Modality::Affine,   // 0 or 1
            Grade::Interval(1, _) => Modality::Relevant, // At least 1
            _ => Modality::Unrestricted,
        }
    }

    /// Convert from modality
    pub fn from_modality(m: Modality) -> Self {
        match m {
            Modality::Linear => Grade::Exact(1),
            Modality::Affine => Grade::AtMost(1),
            Modality::Relevant => Grade::AtLeast(1),
            Modality::Unrestricted => Grade::Omega,
        }
    }
}

impl fmt::Display for Grade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Grade::Exact(n) => write!(f, "{}", n),
            Grade::AtMost(n) => write!(f, "≤{}", n),
            Grade::AtLeast(n) => write!(f, "≥{}", n),
            Grade::Interval(a, b) => write!(f, "[{},{}]", a, b),
            Grade::Omega => write!(f, "ω"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_linear_dominates() {
        assert_eq!(Modality::Linear.combine(Modality::Affine), Modality::Linear);
        assert_eq!(
            Modality::Linear.combine(Modality::Relevant),
            Modality::Linear
        );
        assert_eq!(
            Modality::Linear.combine(Modality::Unrestricted),
            Modality::Linear
        );
    }

    #[test]
    fn test_combine_affine_relevant_is_linear() {
        assert_eq!(
            Modality::Affine.combine(Modality::Relevant),
            Modality::Linear
        );
        assert_eq!(
            Modality::Relevant.combine(Modality::Affine),
            Modality::Linear
        );
    }

    #[test]
    fn test_combine_unrestricted_identity() {
        assert_eq!(
            Modality::Unrestricted.combine(Modality::Affine),
            Modality::Affine
        );
        assert_eq!(
            Modality::Unrestricted.combine(Modality::Relevant),
            Modality::Relevant
        );
    }

    #[test]
    fn test_subtype_reflexive() {
        assert!(Modality::Linear.is_subtype_of(Modality::Linear));
        assert!(Modality::Affine.is_subtype_of(Modality::Affine));
        assert!(Modality::Relevant.is_subtype_of(Modality::Relevant));
        assert!(Modality::Unrestricted.is_subtype_of(Modality::Unrestricted));
    }

    #[test]
    fn test_subtype_linear_is_bottom() {
        assert!(Modality::Linear.is_subtype_of(Modality::Affine));
        assert!(Modality::Linear.is_subtype_of(Modality::Relevant));
        assert!(Modality::Linear.is_subtype_of(Modality::Unrestricted));
    }

    #[test]
    fn test_subtype_unrestricted_is_top() {
        assert!(Modality::Affine.is_subtype_of(Modality::Unrestricted));
        assert!(Modality::Relevant.is_subtype_of(Modality::Unrestricted));
    }

    #[test]
    fn test_subtype_affine_relevant_incomparable() {
        assert!(!Modality::Affine.is_subtype_of(Modality::Relevant));
        assert!(!Modality::Relevant.is_subtype_of(Modality::Affine));
    }

    #[test]
    fn test_allows_weakening() {
        assert!(!Modality::Linear.allows_weakening());
        assert!(Modality::Affine.allows_weakening());
        assert!(!Modality::Relevant.allows_weakening());
        assert!(Modality::Unrestricted.allows_weakening());
    }

    #[test]
    fn test_allows_contraction() {
        assert!(!Modality::Linear.allows_contraction());
        assert!(!Modality::Affine.allows_contraction());
        assert!(Modality::Relevant.allows_contraction());
        assert!(Modality::Unrestricted.allows_contraction());
    }

    #[test]
    fn test_must_use() {
        assert!(Modality::Linear.must_use());
        assert!(!Modality::Affine.must_use());
        assert!(Modality::Relevant.must_use());
        assert!(!Modality::Unrestricted.must_use());
    }

    #[test]
    fn test_grade_to_modality() {
        assert_eq!(Grade::Exact(1).to_modality(), Modality::Linear);
        assert_eq!(Grade::AtMost(1).to_modality(), Modality::Affine);
        assert_eq!(Grade::AtLeast(1).to_modality(), Modality::Relevant);
        assert_eq!(Grade::Omega.to_modality(), Modality::Unrestricted);
    }

    #[test]
    fn test_partial_ord() {
        assert!(Modality::Linear < Modality::Affine);
        assert!(Modality::Linear < Modality::Relevant);
        assert!(Modality::Affine < Modality::Unrestricted);
        assert!(Modality::Relevant < Modality::Unrestricted);

        // Affine and Relevant are incomparable
        assert!(Modality::Affine.partial_cmp(&Modality::Relevant).is_none());
    }
}
