//! Linearity Kinds for the Sounio Type System
//!
//! Day 44 of the Sounio compiler implements linearity as a kind system,
//! enabling substructural types that track resource usage at compile time.
//!
//! # The Problem: Resource Safety
//!
//! Many resources must be used exactly once:
//! - File handles must be closed
//! - Memory must be freed
//! - Database connections must be released
//! - Channels must be consumed
//!
//! Traditional type systems allow:
//! - Forgetting resources (memory leak)
//! - Using resources multiple times (use-after-free)
//!
//! # The Solution: Linearity Kinds
//!
//! ```text
//! Linearity ::= Unrestricted  -- Use any number of times (ω)
//!             | Affine        -- Use at most once (1?)
//!             | Linear        -- Use exactly once (1)
//! ```
//!
//! # Subkinding Lattice
//!
//! ```text
//!        Unrestricted (ω)
//!              |
//!           Affine (1?)
//!              |
//!          Linear (1)
//! ```
//!
//! Linear <: Affine <: Unrestricted
//!
//! This means:
//! - A linear value can be used where affine is expected (stricter satisfies laxer)
//! - An affine value can be used where unrestricted is expected
//!
//! # Integration with Sounio Type System
//!
//! Every type in Sounio carries a linearity annotation:
//!
//! ```text
//! τ ::= τ @ ℓ    where ℓ ∈ {Linear, Affine, Unrestricted}
//!
//! linear struct FileHandle { ... }     -- Must close exactly once
//! affine struct TempAlloc { ... }      -- Can free or let drop
//! struct Data { ... }                  -- Unrestricted by default
//! ```

use std::fmt;

/// Linearity kind for substructural typing
///
/// Represents the usage constraints on a type:
/// - `Unrestricted`: Can be used any number of times (weakening + contraction allowed)
/// - `Affine`: Can be used at most once (weakening allowed, no contraction)
/// - `Linear`: Must be used exactly once (no weakening, no contraction)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Linearity {
    /// Linear (1): Must be used exactly once
    ///
    /// Neither weakening nor contraction allowed.
    /// Example: File handles, unique pointers, channels
    Linear,

    /// Affine (1?): Can be used at most once
    ///
    /// Weakening allowed (can drop), no contraction.
    /// Example: Optional cleanup resources, temp allocations
    Affine,

    /// Unrestricted (ω): Can be used any number of times
    ///
    /// Both weakening and contraction allowed.
    /// Example: Integers, strings, immutable data
    #[default]
    Unrestricted,
}

impl Linearity {
    /// Check if this linearity is a subkind of another
    ///
    /// Linear <: Affine <: Unrestricted
    ///
    /// A more restrictive linearity is a subkind of a less restrictive one.
    /// This enables covariant use: a linear value can satisfy an affine requirement.
    pub fn is_subkind_of(self, other: Linearity) -> bool {
        match (self, other) {
            // Reflexivity
            (Linearity::Linear, Linearity::Linear)
            | (Linearity::Affine, Linearity::Affine)
            | (Linearity::Unrestricted, Linearity::Unrestricted) => true,

            // Linear is bottom of lattice
            (Linearity::Linear, _) => true,

            // Unrestricted is top
            (_, Linearity::Unrestricted) => true,

            // Affine <: Unrestricted (already covered above)
            // Linear <: Affine (covered by Linear being bottom)
            _ => false,
        }
    }

    /// Meet of two linearities (greatest lower bound)
    ///
    /// Returns the most restrictive linearity that both can satisfy.
    ///
    /// ```text
    /// meet(Linear, _) = Linear
    /// meet(_, Linear) = Linear
    /// meet(Affine, Affine) = Affine
    /// meet(Unrestricted, x) = x
    /// ```
    pub fn meet(self, other: Linearity) -> Linearity {
        match (self, other) {
            (Linearity::Linear, _) | (_, Linearity::Linear) => Linearity::Linear,
            (Linearity::Affine, _) | (_, Linearity::Affine) => Linearity::Affine,
            (Linearity::Unrestricted, Linearity::Unrestricted) => Linearity::Unrestricted,
        }
    }

    /// Join of two linearities (least upper bound)
    ///
    /// Returns the least restrictive linearity that subsumes both.
    ///
    /// ```text
    /// join(Unrestricted, _) = Unrestricted
    /// join(_, Unrestricted) = Unrestricted
    /// join(Affine, Affine) = Affine
    /// join(Linear, x) = x
    /// ```
    pub fn join(self, other: Linearity) -> Linearity {
        match (self, other) {
            (Linearity::Unrestricted, _) | (_, Linearity::Unrestricted) => Linearity::Unrestricted,
            (Linearity::Affine, _) | (_, Linearity::Affine) => Linearity::Affine,
            (Linearity::Linear, Linearity::Linear) => Linearity::Linear,
        }
    }

    /// Check if weakening is allowed
    ///
    /// Weakening: Γ ⊢ A ⟹ Γ, B ⊢ A (can ignore/drop resources)
    pub fn allows_weakening(self) -> bool {
        matches!(self, Linearity::Affine | Linearity::Unrestricted)
    }

    /// Check if contraction is allowed
    ///
    /// Contraction: Γ, A, A ⊢ B ⟹ Γ, A ⊢ B (can duplicate/copy resources)
    pub fn allows_contraction(self) -> bool {
        matches!(self, Linearity::Unrestricted)
    }

    /// Check if the value must be used
    ///
    /// Returns true for Linear (exactly once required)
    pub fn must_use(self) -> bool {
        matches!(self, Linearity::Linear)
    }

    /// Check if the value can be discarded without use
    pub fn can_discard(self) -> bool {
        self.allows_weakening()
    }

    /// Check if the value can be copied
    pub fn can_copy(self) -> bool {
        self.allows_contraction()
    }

    /// Get the symbol for this linearity
    pub fn symbol(self) -> &'static str {
        match self {
            Linearity::Linear => "1",
            Linearity::Affine => "1?",
            Linearity::Unrestricted => "ω",
        }
    }

    /// Get the keyword for this linearity (for D syntax)
    pub fn keyword(self) -> &'static str {
        match self {
            Linearity::Linear => "linear",
            Linearity::Affine => "affine",
            Linearity::Unrestricted => "",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "linear" | "1" => Some(Linearity::Linear),
            "affine" | "1?" | "?" => Some(Linearity::Affine),
            "unrestricted" | "ω" | "omega" | "*" | "" => Some(Linearity::Unrestricted),
            _ => None,
        }
    }

    /// Convert from the existing Modality type
    pub fn from_modality(modality: super::modality::Modality) -> Self {
        match modality {
            super::modality::Modality::Linear => Linearity::Linear,
            super::modality::Modality::Affine => Linearity::Affine,
            super::modality::Modality::Relevant => Linearity::Linear, // Relevant maps to Linear (must use)
            super::modality::Modality::Unrestricted => Linearity::Unrestricted,
        }
    }

    /// Convert to the existing Modality type
    pub fn to_modality(self) -> super::modality::Modality {
        match self {
            Linearity::Linear => super::modality::Modality::Linear,
            Linearity::Affine => super::modality::Modality::Affine,
            Linearity::Unrestricted => super::modality::Modality::Unrestricted,
        }
    }
}

impl fmt::Display for Linearity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Linearity::Linear => write!(f, "linear"),
            Linearity::Affine => write!(f, "affine"),
            Linearity::Unrestricted => write!(f, "unrestricted"),
        }
    }
}

impl PartialOrd for Linearity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        if self == other {
            return Some(Ordering::Equal);
        }

        // Linear is bottom
        if *self == Linearity::Linear {
            return Some(Ordering::Less);
        }
        if *other == Linearity::Linear {
            return Some(Ordering::Greater);
        }

        // Unrestricted is top
        if *self == Linearity::Unrestricted {
            return Some(Ordering::Greater);
        }
        if *other == Linearity::Unrestricted {
            return Some(Ordering::Less);
        }

        // Affine is in the middle, but we've covered all cases
        Some(Ordering::Equal)
    }
}

impl Ord for Linearity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// A type annotated with linearity
///
/// This wraps any type with a linearity kind, enabling the type system
/// to track resource usage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KindedType<T> {
    /// The underlying type
    pub inner: T,
    /// The linearity kind
    pub linearity: Linearity,
}

impl<T> KindedType<T> {
    /// Create a new kinded type
    pub fn new(inner: T, linearity: Linearity) -> Self {
        Self { inner, linearity }
    }

    /// Create a linear type
    pub fn linear(inner: T) -> Self {
        Self::new(inner, Linearity::Linear)
    }

    /// Create an affine type
    pub fn affine(inner: T) -> Self {
        Self::new(inner, Linearity::Affine)
    }

    /// Create an unrestricted type
    pub fn unrestricted(inner: T) -> Self {
        Self::new(inner, Linearity::Unrestricted)
    }

    /// Map over the inner type
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> KindedType<U> {
        KindedType {
            inner: f(self.inner),
            linearity: self.linearity,
        }
    }

    /// Get a reference to the inner type
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Consume and return the inner type
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Check if this type can be weakened (discarded)
    pub fn can_discard(&self) -> bool {
        self.linearity.can_discard()
    }

    /// Check if this type can be contracted (copied)
    pub fn can_copy(&self) -> bool {
        self.linearity.can_copy()
    }

    /// Check if this type must be used
    pub fn must_use(&self) -> bool {
        self.linearity.must_use()
    }
}

impl<T: fmt::Display> fmt::Display for KindedType<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.linearity == Linearity::Unrestricted {
            write!(f, "{}", self.inner)
        } else {
            write!(f, "{} @ {}", self.inner, self.linearity)
        }
    }
}

/// Linearity constraint for type parameters
///
/// Used in generic contexts to specify linearity bounds:
///
/// ```text
/// fn consume<T: Linear>(x: T) { ... }      -- T must be linear
/// fn maybe_use<T: Affine>(x: T) { ... }    -- T can be linear or affine
/// fn any<T>(x: T) { ... }                  -- T can be anything (unrestricted bound)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinearityBound {
    /// The upper bound (most restrictive linearity accepted)
    pub bound: Linearity,
}

impl LinearityBound {
    /// Create a bound requiring linear types
    pub fn linear() -> Self {
        Self {
            bound: Linearity::Linear,
        }
    }

    /// Create a bound allowing affine and linear types
    pub fn affine() -> Self {
        Self {
            bound: Linearity::Affine,
        }
    }

    /// Create a bound allowing any linearity
    pub fn any() -> Self {
        Self {
            bound: Linearity::Unrestricted,
        }
    }

    /// Check if a linearity satisfies this bound
    pub fn satisfied_by(self, linearity: Linearity) -> bool {
        linearity.is_subkind_of(self.bound)
    }
}

impl fmt::Display for LinearityBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.bound {
            Linearity::Linear => write!(f, ": Linear"),
            Linearity::Affine => write!(f, ": Affine"),
            Linearity::Unrestricted => Ok(()),
        }
    }
}

/// Error for linearity kind violations
#[derive(Clone, Debug, thiserror::Error)]
pub enum LinearityError {
    #[error("Cannot discard linear value of type {typ} (must be used exactly once)")]
    CannotDiscard { typ: String },

    #[error("Cannot copy value of type {typ} (linearity: {linearity})")]
    CannotCopy { typ: String, linearity: Linearity },

    #[error("Linearity mismatch: expected {expected}, found {found}")]
    Mismatch {
        expected: Linearity,
        found: Linearity,
    },

    #[error("Linear value {name} was not used")]
    Unused { name: String },

    #[error("Value {name} used multiple times but has linearity {linearity}")]
    MultipleUse { name: String, linearity: Linearity },

    #[error("Bound not satisfied: {linearity} does not satisfy bound {bound}")]
    BoundNotSatisfied {
        linearity: Linearity,
        bound: LinearityBound,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subkinding_reflexive() {
        assert!(Linearity::Linear.is_subkind_of(Linearity::Linear));
        assert!(Linearity::Affine.is_subkind_of(Linearity::Affine));
        assert!(Linearity::Unrestricted.is_subkind_of(Linearity::Unrestricted));
    }

    #[test]
    fn test_subkinding_linear_is_bottom() {
        assert!(Linearity::Linear.is_subkind_of(Linearity::Affine));
        assert!(Linearity::Linear.is_subkind_of(Linearity::Unrestricted));
    }

    #[test]
    fn test_subkinding_unrestricted_is_top() {
        assert!(Linearity::Affine.is_subkind_of(Linearity::Unrestricted));
        assert!(Linearity::Linear.is_subkind_of(Linearity::Unrestricted));
    }

    #[test]
    fn test_subkinding_affine_not_subkind_of_linear() {
        assert!(!Linearity::Affine.is_subkind_of(Linearity::Linear));
        assert!(!Linearity::Unrestricted.is_subkind_of(Linearity::Linear));
        assert!(!Linearity::Unrestricted.is_subkind_of(Linearity::Affine));
    }

    #[test]
    fn test_meet_operation() {
        assert_eq!(Linearity::Linear.meet(Linearity::Affine), Linearity::Linear);
        assert_eq!(
            Linearity::Affine.meet(Linearity::Unrestricted),
            Linearity::Affine
        );
        assert_eq!(
            Linearity::Unrestricted.meet(Linearity::Unrestricted),
            Linearity::Unrestricted
        );
    }

    #[test]
    fn test_join_operation() {
        assert_eq!(Linearity::Linear.join(Linearity::Affine), Linearity::Affine);
        assert_eq!(
            Linearity::Linear.join(Linearity::Unrestricted),
            Linearity::Unrestricted
        );
        assert_eq!(Linearity::Linear.join(Linearity::Linear), Linearity::Linear);
    }

    #[test]
    fn test_structural_rules() {
        assert!(!Linearity::Linear.allows_weakening());
        assert!(!Linearity::Linear.allows_contraction());

        assert!(Linearity::Affine.allows_weakening());
        assert!(!Linearity::Affine.allows_contraction());

        assert!(Linearity::Unrestricted.allows_weakening());
        assert!(Linearity::Unrestricted.allows_contraction());
    }

    #[test]
    fn test_parse() {
        assert_eq!(Linearity::parse("linear"), Some(Linearity::Linear));
        assert_eq!(Linearity::parse("1"), Some(Linearity::Linear));
        assert_eq!(Linearity::parse("affine"), Some(Linearity::Affine));
        assert_eq!(Linearity::parse("1?"), Some(Linearity::Affine));
        assert_eq!(
            Linearity::parse("unrestricted"),
            Some(Linearity::Unrestricted)
        );
        assert_eq!(Linearity::parse("ω"), Some(Linearity::Unrestricted));
    }

    #[test]
    fn test_kinded_type() {
        let linear_int: KindedType<&str> = KindedType::linear("int");
        assert!(linear_int.must_use());
        assert!(!linear_int.can_discard());
        assert!(!linear_int.can_copy());

        let affine_int: KindedType<&str> = KindedType::affine("int");
        assert!(!affine_int.must_use());
        assert!(affine_int.can_discard());
        assert!(!affine_int.can_copy());

        let unrestricted_int: KindedType<&str> = KindedType::unrestricted("int");
        assert!(!unrestricted_int.must_use());
        assert!(unrestricted_int.can_discard());
        assert!(unrestricted_int.can_copy());
    }

    #[test]
    fn test_linearity_bound() {
        let linear_bound = LinearityBound::linear();
        assert!(linear_bound.satisfied_by(Linearity::Linear));
        assert!(!linear_bound.satisfied_by(Linearity::Affine));
        assert!(!linear_bound.satisfied_by(Linearity::Unrestricted));

        let affine_bound = LinearityBound::affine();
        assert!(affine_bound.satisfied_by(Linearity::Linear));
        assert!(affine_bound.satisfied_by(Linearity::Affine));
        assert!(!affine_bound.satisfied_by(Linearity::Unrestricted));

        let any_bound = LinearityBound::any();
        assert!(any_bound.satisfied_by(Linearity::Linear));
        assert!(any_bound.satisfied_by(Linearity::Affine));
        assert!(any_bound.satisfied_by(Linearity::Unrestricted));
    }

    #[test]
    fn test_ordering() {
        assert!(Linearity::Linear < Linearity::Affine);
        assert!(Linearity::Affine < Linearity::Unrestricted);
        assert!(Linearity::Linear < Linearity::Unrestricted);
    }
}
