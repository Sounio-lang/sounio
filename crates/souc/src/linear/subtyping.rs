//! Linear subtyping rules
//!
//! Subtyping for linear types follows the modality lattice:
//! - Linear <: Affine (can weaken to "at most once")
//! - Linear <: Relevant (can strengthen to "at least once")
//! - Affine <: Unrestricted
//! - Relevant <: Unrestricted
//!
//! For structural types:
//! - Tensor is covariant in both components
//! - Lollipop is contravariant in domain, covariant in codomain
//! - Bang preserves subtyping
//! - Quest preserves subtyping

use std::fmt;

use super::linear_types::LinearType;
use super::modality::Modality;

/// Subtyping error
#[derive(Clone, Debug, thiserror::Error)]
pub enum LinearSubtypeError {
    #[error("Not a subtype: {sub} is not a subtype of {sup}")]
    NotSubtype { sub: String, sup: String },

    #[error("Modality mismatch: {sub:?} cannot be used where {sup:?} is expected")]
    ModalityMismatch { sub: Modality, sup: Modality },

    #[error("Structural mismatch: expected {expected}, found {found}")]
    StructuralMismatch { expected: String, found: String },

    #[error("Incompatible knowledge types")]
    IncompatibleKnowledge,
}

/// Result type for subtyping
pub type LinearSubtypeResult = Result<(), LinearSubtypeError>;

/// Variance annotation for type parameters
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variance {
    /// Covariant: A <: B implies F[A] <: F[B]
    Covariant,
    /// Contravariant: A <: B implies F[B] <: F[A]
    Contravariant,
    /// Invariant: A = B required for F[A] = F[B]
    Invariant,
    /// Bivariant: Any relationship allowed
    Bivariant,
}

impl Variance {
    /// Compose two variances
    pub fn compose(self, other: Variance) -> Variance {
        use Variance::*;
        match (self, other) {
            (Invariant, _) | (_, Invariant) => Invariant,
            (Bivariant, x) | (x, Bivariant) => x,
            (Covariant, Covariant) => Covariant,
            (Contravariant, Contravariant) => Covariant,
            (Covariant, Contravariant) | (Contravariant, Covariant) => Contravariant,
        }
    }

    /// Flip variance (for contravariant positions)
    pub fn flip(self) -> Variance {
        match self {
            Variance::Covariant => Variance::Contravariant,
            Variance::Contravariant => Variance::Covariant,
            Variance::Invariant => Variance::Invariant,
            Variance::Bivariant => Variance::Bivariant,
        }
    }
}

/// Linear subtype checker
pub struct LinearSubtypeChecker {
    /// Whether to allow gradual typing (? matches anything)
    pub gradual: bool,
}

impl LinearSubtypeChecker {
    /// Create a new subtype checker
    pub fn new() -> Self {
        Self { gradual: false }
    }

    /// Create with gradual typing enabled
    pub fn with_gradual(gradual: bool) -> Self {
        Self { gradual }
    }

    /// Check if sub <: sup
    pub fn is_subtype(&self, sub: &LinearType, sup: &LinearType) -> LinearSubtypeResult {
        self.check_subtype(sub, sup, Variance::Covariant)
    }

    /// Check subtype with variance context
    fn check_subtype(
        &self,
        sub: &LinearType,
        sup: &LinearType,
        variance: Variance,
    ) -> LinearSubtypeResult {
        // Handle gradual typing
        if self.gradual
            && (matches!(sub, LinearType::Unknown) || matches!(sup, LinearType::Unknown))
        {
            return Ok(());
        }

        // Handle variance flipping
        let (sub, sup) = match variance {
            Variance::Contravariant => (sup, sub),
            _ => (sub, sup),
        };

        match (sub, sup) {
            // Reflexivity
            _ if sub.definitionally_equal(sup) => Ok(()),

            // Knowledge subtyping: modality must be compatible
            (
                LinearType::Knowledge {
                    inner: i1,
                    confidence: c1,
                    ontology: o1,
                    modality: m1,
                    ..
                },
                LinearType::Knowledge {
                    inner: i2,
                    confidence: c2,
                    ontology: o2,
                    modality: m2,
                    ..
                },
            ) => {
                // Modality: sub must be at least as restrictive
                if !m1.is_subtype_of(*m2) {
                    return Err(LinearSubtypeError::ModalityMismatch { sub: *m1, sup: *m2 });
                }

                // Inner type must match (invariant for now)
                if i1 != i2 {
                    return Err(LinearSubtypeError::IncompatibleKnowledge);
                }

                // Confidence: higher is subtype of lower (covariant)
                // This uses Day 35's confidence comparison
                if !self.confidence_subtype(c1, c2) {
                    return Err(LinearSubtypeError::NotSubtype {
                        sub: format!("{}", c1),
                        sup: format!("{}", c2),
                    });
                }

                // Ontology: more specific is subtype of less specific
                if !self.ontology_subtype(o1, o2) {
                    return Err(LinearSubtypeError::NotSubtype {
                        sub: format!("{}", o1),
                        sup: format!("{}", o2),
                    });
                }

                Ok(())
            }

            // Tensor: covariant in both components
            (LinearType::Tensor(a1, b1), LinearType::Tensor(a2, b2)) => {
                self.check_subtype(a1, a2, Variance::Covariant)?;
                self.check_subtype(b1, b2, Variance::Covariant)?;
                Ok(())
            }

            // With: covariant in both components
            (LinearType::With(a1, b1), LinearType::With(a2, b2)) => {
                self.check_subtype(a1, a2, Variance::Covariant)?;
                self.check_subtype(b1, b2, Variance::Covariant)?;
                Ok(())
            }

            // Plus: covariant in both components
            (LinearType::Plus(a1, b1), LinearType::Plus(a2, b2)) => {
                self.check_subtype(a1, a2, Variance::Covariant)?;
                self.check_subtype(b1, b2, Variance::Covariant)?;
                Ok(())
            }

            // Lollipop: contravariant in domain, covariant in codomain
            (LinearType::Lollipop(a1, b1), LinearType::Lollipop(a2, b2)) => {
                self.check_subtype(a2, a1, Variance::Covariant)?; // Contravariant
                self.check_subtype(b1, b2, Variance::Covariant)?;
                Ok(())
            }

            // Par: covariant in both components
            (LinearType::Par(a1, b1), LinearType::Par(a2, b2)) => {
                self.check_subtype(a1, a2, Variance::Covariant)?;
                self.check_subtype(b1, b2, Variance::Covariant)?;
                Ok(())
            }

            // Bang: covariant
            (LinearType::Bang(a1), LinearType::Bang(a2)) => {
                self.check_subtype(a1, a2, Variance::Covariant)
            }

            // Quest: covariant
            (LinearType::Quest(a1), LinearType::Quest(a2)) => {
                self.check_subtype(a1, a2, Variance::Covariant)
            }

            // Bang subsumes everything (! is top for usage)
            // !A <: A (dereliction allows this coercion)
            (LinearType::Bang(inner), other) => {
                self.check_subtype(inner, other, Variance::Covariant)
            }

            // Quest is subtype (affine subsumes linear for discarding)
            // A <: ?A
            (other, LinearType::Quest(inner)) => {
                self.check_subtype(other, inner, Variance::Covariant)
            }

            // Units
            (LinearType::One, LinearType::One) => Ok(()),
            (LinearType::Bottom, LinearType::Bottom) => Ok(()),
            (LinearType::Top, LinearType::Top) => Ok(()),
            (LinearType::Zero, LinearType::Zero) => Ok(()),

            // Top is supertype of everything (for &)
            (_, LinearType::Top) => Ok(()),

            // Zero is subtype of everything (for ⊕)
            (LinearType::Zero, _) => Ok(()),

            // Variables
            (LinearType::Var(v1), LinearType::Var(v2)) if v1 == v2 => Ok(()),

            // No other subtyping relationships
            _ => Err(LinearSubtypeError::NotSubtype {
                sub: format!("{}", sub),
                sup: format!("{}", sup),
            }),
        }
    }

    /// Check confidence subtyping (higher confidence is more specific)
    fn confidence_subtype(
        &self,
        sub: &crate::dependent::types::ConfidenceType,
        sup: &crate::dependent::types::ConfidenceType,
    ) -> bool {
        use crate::dependent::types::ConfidenceType;

        match (sub, sup) {
            (ConfidenceType::Literal(v1), ConfidenceType::Literal(v2)) => v1 >= v2,
            (ConfidenceType::Unknown, _) | (_, ConfidenceType::Unknown) => self.gradual,
            _ => sub.definitionally_equal(sup),
        }
    }

    /// Check ontology subtyping (more specific is subtype)
    fn ontology_subtype(
        &self,
        sub: &crate::dependent::types::OntologyType,
        sup: &crate::dependent::types::OntologyType,
    ) -> bool {
        use crate::dependent::types::OntologyType;

        match (sub, sup) {
            // Any is supertype of everything
            (_, OntologyType::Any) => true,
            // None is subtype of everything
            (OntologyType::None, _) => true,
            // Unknown with gradual
            (OntologyType::Unknown, _) | (_, OntologyType::Unknown) => self.gradual,
            // Concrete: must be equal or sub must contain sup
            _ => sub.contains(sup),
        }
    }

    /// Check if a modality coercion is valid
    pub fn modality_coercion(&self, from: Modality, to: Modality) -> bool {
        from.is_subtype_of(to)
    }
}

impl Default for LinearSubtypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Coercion between linear types
#[derive(Clone, Debug)]
pub enum LinearCoercion {
    /// Identity (no coercion needed)
    Identity,
    /// Modality weakening (Linear -> Affine, etc.)
    ModalityWeaken(Modality, Modality),
    /// Dereliction (!A -> A)
    Dereliction,
    /// Quest introduction (A -> ?A)
    QuestIntro,
    /// Structural coercion (for nested types)
    Structural(Box<LinearCoercion>, Box<LinearCoercion>),
    /// Composition of coercions
    Compose(Box<LinearCoercion>, Box<LinearCoercion>),
}

impl LinearCoercion {
    /// Compute the coercion needed to go from sub to sup
    pub fn compute(sub: &LinearType, sup: &LinearType) -> Option<LinearCoercion> {
        if sub.definitionally_equal(sup) {
            return Some(LinearCoercion::Identity);
        }

        match (sub, sup) {
            // Modality weakening for Knowledge types
            (
                LinearType::Knowledge { modality: m1, .. },
                LinearType::Knowledge { modality: m2, .. },
            ) if m1.is_subtype_of(*m2) => Some(LinearCoercion::ModalityWeaken(*m1, *m2)),

            // Dereliction: !A -> A
            (LinearType::Bang(inner), other) if inner.definitionally_equal(other) => {
                Some(LinearCoercion::Dereliction)
            }

            // Quest intro: A -> ?A
            (other, LinearType::Quest(inner)) if other.definitionally_equal(inner) => {
                Some(LinearCoercion::QuestIntro)
            }

            // Structural for tensor
            (LinearType::Tensor(a1, b1), LinearType::Tensor(a2, b2)) => {
                let ca = Self::compute(a1, a2)?;
                let cb = Self::compute(b1, b2)?;
                Some(LinearCoercion::Structural(Box::new(ca), Box::new(cb)))
            }

            // Structural for lollipop
            (LinearType::Lollipop(a1, b1), LinearType::Lollipop(a2, b2)) => {
                // Contravariant in domain
                let ca = Self::compute(a2, a1)?;
                let cb = Self::compute(b1, b2)?;
                Some(LinearCoercion::Structural(Box::new(ca), Box::new(cb)))
            }

            _ => None,
        }
    }

    /// Check if this is the identity coercion
    pub fn is_identity(&self) -> bool {
        matches!(self, LinearCoercion::Identity)
    }
}

impl fmt::Display for LinearCoercion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearCoercion::Identity => write!(f, "id"),
            LinearCoercion::ModalityWeaken(from, to) => write!(f, "weaken({} → {})", from, to),
            LinearCoercion::Dereliction => write!(f, "derelict"),
            LinearCoercion::QuestIntro => write!(f, "?-intro"),
            LinearCoercion::Structural(a, b) => write!(f, "struct({}, {})", a, b),
            LinearCoercion::Compose(a, b) => write!(f, "{} ∘ {}", a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dependent::types::OntologyType;
    use crate::types::Type;

    #[test]
    fn test_reflexivity() {
        let checker = LinearSubtypeChecker::new();
        let typ = LinearType::One;
        assert!(checker.is_subtype(&typ, &typ).is_ok());
    }

    #[test]
    fn test_modality_subtyping() {
        let checker = LinearSubtypeChecker::new();

        let linear = LinearType::linear_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let affine = LinearType::affine_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let relevant = LinearType::relevant_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let unrestricted = LinearType::unrestricted_knowledge(Type::Bool, 0.9, OntologyType::Any);

        // Linear <: Affine
        assert!(checker.is_subtype(&linear, &affine).is_ok());

        // Linear <: Relevant
        assert!(checker.is_subtype(&linear, &relevant).is_ok());

        // Affine <: Unrestricted
        assert!(checker.is_subtype(&affine, &unrestricted).is_ok());

        // Relevant <: Unrestricted
        assert!(checker.is_subtype(&relevant, &unrestricted).is_ok());

        // NOT: Affine <: Relevant
        assert!(checker.is_subtype(&affine, &relevant).is_err());
    }

    #[test]
    fn test_confidence_subtyping() {
        let checker = LinearSubtypeChecker::new();

        let high = LinearType::linear_knowledge(Type::Bool, 0.95, OntologyType::Any);
        let low = LinearType::linear_knowledge(Type::Bool, 0.80, OntologyType::Any);

        // High confidence <: Low confidence (more specific is subtype)
        assert!(checker.is_subtype(&high, &low).is_ok());

        // NOT: Low <: High
        assert!(checker.is_subtype(&low, &high).is_err());
    }

    #[test]
    fn test_tensor_covariant() {
        let checker = LinearSubtypeChecker::new();

        let a1 = LinearType::linear_knowledge(Type::Bool, 0.95, OntologyType::Any);
        let a2 = LinearType::linear_knowledge(Type::Bool, 0.80, OntologyType::Any);

        let tensor1 = LinearType::tensor(a1.clone(), LinearType::One);
        let tensor2 = LinearType::tensor(a2.clone(), LinearType::One);

        // Covariant: a1 <: a2 implies tensor(a1, 1) <: tensor(a2, 1)
        assert!(checker.is_subtype(&tensor1, &tensor2).is_ok());
    }

    #[test]
    fn test_lollipop_contravariant() {
        let checker = LinearSubtypeChecker::new();

        let high = LinearType::linear_knowledge(Type::Bool, 0.95, OntologyType::Any);
        let low = LinearType::linear_knowledge(Type::Bool, 0.80, OntologyType::Any);

        // high -> 1 should be subtype of low -> 1
        // (contravariant in domain: accepts wider input)
        let f1 = LinearType::lollipop(low.clone(), LinearType::One);
        let f2 = LinearType::lollipop(high.clone(), LinearType::One);

        assert!(checker.is_subtype(&f1, &f2).is_ok());
    }

    #[test]
    fn test_bang_dereliction() {
        let checker = LinearSubtypeChecker::new();

        let inner = LinearType::One;
        let bang = LinearType::bang(inner.clone());

        // !A <: A (dereliction)
        assert!(checker.is_subtype(&bang, &inner).is_ok());
    }

    #[test]
    fn test_quest_introduction() {
        let checker = LinearSubtypeChecker::new();

        let inner = LinearType::One;
        let quest = LinearType::quest(inner.clone());

        // A <: ?A
        assert!(checker.is_subtype(&inner, &quest).is_ok());
    }

    #[test]
    fn test_top_is_supertype() {
        let checker = LinearSubtypeChecker::new();

        assert!(
            checker
                .is_subtype(&LinearType::One, &LinearType::Top)
                .is_ok()
        );
        assert!(
            checker
                .is_subtype(&LinearType::Zero, &LinearType::Top)
                .is_ok()
        );
    }

    #[test]
    fn test_zero_is_subtype() {
        let checker = LinearSubtypeChecker::new();

        assert!(
            checker
                .is_subtype(&LinearType::Zero, &LinearType::One)
                .is_ok()
        );
        assert!(
            checker
                .is_subtype(&LinearType::Zero, &LinearType::Top)
                .is_ok()
        );
    }

    #[test]
    fn test_gradual_subtyping() {
        let checker = LinearSubtypeChecker::with_gradual(true);

        let unknown = LinearType::Unknown;
        let concrete = LinearType::One;

        // With gradual, Unknown matches anything
        assert!(checker.is_subtype(&unknown, &concrete).is_ok());
        assert!(checker.is_subtype(&concrete, &unknown).is_ok());
    }

    #[test]
    fn test_coercion_identity() {
        let typ = LinearType::One;
        let coercion = LinearCoercion::compute(&typ, &typ).unwrap();
        assert!(coercion.is_identity());
    }

    #[test]
    fn test_coercion_dereliction() {
        let inner = LinearType::One;
        let bang = LinearType::bang(inner.clone());

        let coercion = LinearCoercion::compute(&bang, &inner).unwrap();
        assert!(matches!(coercion, LinearCoercion::Dereliction));
    }

    #[test]
    fn test_variance_compose() {
        assert_eq!(
            Variance::Covariant.compose(Variance::Covariant),
            Variance::Covariant
        );
        assert_eq!(
            Variance::Contravariant.compose(Variance::Contravariant),
            Variance::Covariant
        );
        assert_eq!(
            Variance::Covariant.compose(Variance::Contravariant),
            Variance::Contravariant
        );
    }
}
