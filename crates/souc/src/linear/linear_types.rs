//! Linear type definitions
//!
//! This module defines the core linear types based on Girard's Linear Logic:
//!
//! - `Tensor` (⊗): Multiplicative conjunction - both must be used
//! - `With` (&): Additive conjunction - internal choice
//! - `Plus` (⊕): Additive disjunction - external choice
//! - `Lollipop` (⊸): Linear implication - consuming A produces B
//! - `Bang` (!): Of course - unlimited use
//! - `Quest` (?): Why not - can discard

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use super::modality::Modality;
use crate::dependent::types::{ConfidenceType, OntologyType, TemporalType};
use crate::types::Type;

/// Linear epistemic type
#[derive(Clone, Debug)]
pub enum LinearType {
    /// Knowledge with modality
    Knowledge {
        inner: Arc<Type>,
        confidence: ConfidenceType,
        ontology: OntologyType,
        temporal: TemporalType,
        modality: Modality,
    },

    /// Tensor product (⊗) - both must be used
    Tensor(Arc<LinearType>, Arc<LinearType>),

    /// With (&) - internal choice (same context, choose one to provide)
    With(Arc<LinearType>, Arc<LinearType>),

    /// Plus (⊕) - external choice (environment chooses)
    Plus(Arc<LinearType>, Arc<LinearType>),

    /// Linear function (⊸) - consuming A produces B
    Lollipop(Arc<LinearType>, Arc<LinearType>),

    /// Of course (!) - promotes to unrestricted
    Bang(Arc<LinearType>),

    /// Why not (?) - demotes to affine
    Quest(Arc<LinearType>),

    /// Par (⅋) - parallel disjunction (dual of tensor)
    Par(Arc<LinearType>, Arc<LinearType>),

    /// Multiplicative unit (1) - unit for tensor
    One,

    /// Multiplicative false (⊥) - unit for par
    Bottom,

    /// Additive unit (⊤) - unit for with
    Top,

    /// Additive false (0) - unit for plus
    Zero,

    /// Type variable
    Var(String),

    /// Unknown (for gradual typing)
    Unknown,
}

impl LinearType {
    // ========================================================================
    // Constructors for Knowledge types
    // ========================================================================

    /// Create linear knowledge (use exactly once)
    pub fn linear_knowledge(inner: Type, confidence: f64, ontology: OntologyType) -> Self {
        LinearType::Knowledge {
            inner: Arc::new(inner),
            confidence: ConfidenceType::Literal(confidence),
            ontology,
            temporal: TemporalType::Instant(None),
            modality: Modality::Linear,
        }
    }

    /// Create affine knowledge (use at most once)
    pub fn affine_knowledge(inner: Type, confidence: f64, ontology: OntologyType) -> Self {
        LinearType::Knowledge {
            inner: Arc::new(inner),
            confidence: ConfidenceType::Literal(confidence),
            ontology,
            temporal: TemporalType::Instant(None),
            modality: Modality::Affine,
        }
    }

    /// Create relevant knowledge (use at least once)
    pub fn relevant_knowledge(inner: Type, confidence: f64, ontology: OntologyType) -> Self {
        LinearType::Knowledge {
            inner: Arc::new(inner),
            confidence: ConfidenceType::Literal(confidence),
            ontology,
            temporal: TemporalType::Instant(None),
            modality: Modality::Relevant,
        }
    }

    /// Create unrestricted knowledge (use freely)
    pub fn unrestricted_knowledge(inner: Type, confidence: f64, ontology: OntologyType) -> Self {
        LinearType::Knowledge {
            inner: Arc::new(inner),
            confidence: ConfidenceType::Literal(confidence),
            ontology,
            temporal: TemporalType::Instant(None),
            modality: Modality::Unrestricted,
        }
    }

    // ========================================================================
    // Constructors for Linear Connectives
    // ========================================================================

    /// Create tensor product (⊗)
    pub fn tensor(left: LinearType, right: LinearType) -> Self {
        LinearType::Tensor(Arc::new(left), Arc::new(right))
    }

    /// Create with (&)
    pub fn with(left: LinearType, right: LinearType) -> Self {
        LinearType::With(Arc::new(left), Arc::new(right))
    }

    /// Create plus (⊕)
    pub fn plus(left: LinearType, right: LinearType) -> Self {
        LinearType::Plus(Arc::new(left), Arc::new(right))
    }

    /// Create lollipop (⊸)
    pub fn lollipop(from: LinearType, to: LinearType) -> Self {
        LinearType::Lollipop(Arc::new(from), Arc::new(to))
    }

    /// Create par (⅋)
    pub fn par(left: LinearType, right: LinearType) -> Self {
        LinearType::Par(Arc::new(left), Arc::new(right))
    }

    /// Create bang (!)
    pub fn bang(inner: LinearType) -> Self {
        LinearType::Bang(Arc::new(inner))
    }

    /// Create quest (?)
    pub fn quest(inner: LinearType) -> Self {
        LinearType::Quest(Arc::new(inner))
    }

    // ========================================================================
    // Modality Operations
    // ========================================================================

    /// Get the modality of this type
    pub fn modality(&self) -> Modality {
        match self {
            LinearType::Knowledge { modality, .. } => *modality,
            LinearType::Tensor(a, b) => a.modality().combine(b.modality()),
            LinearType::With(a, b) => a.modality().combine(b.modality()),
            LinearType::Plus(a, b) => a.modality().combine(b.modality()),
            LinearType::Lollipop(_, b) => b.modality(),
            LinearType::Par(a, b) => a.modality().combine(b.modality()),
            LinearType::Bang(_) => Modality::Unrestricted,
            LinearType::Quest(_) => Modality::Affine,
            LinearType::One | LinearType::Top => Modality::Unrestricted,
            LinearType::Bottom | LinearType::Zero => Modality::Linear,
            LinearType::Var(_) | LinearType::Unknown => Modality::Linear,
        }
    }

    /// Check if this type is linear (use exactly once)
    pub fn is_linear(&self) -> bool {
        self.modality() == Modality::Linear
    }

    /// Check if this type is affine (use at most once)
    pub fn is_affine(&self) -> bool {
        matches!(self.modality(), Modality::Linear | Modality::Affine)
    }

    /// Check if this type is relevant (use at least once)
    pub fn is_relevant(&self) -> bool {
        matches!(self.modality(), Modality::Linear | Modality::Relevant)
    }

    /// Check if this type is unrestricted
    pub fn is_unrestricted(&self) -> bool {
        self.modality() == Modality::Unrestricted
    }

    // ========================================================================
    // Linear Logic Operations
    // ========================================================================

    /// Compute the linear negation (dual)
    ///
    /// ```text
    /// (A ⊗ B)⊥ = A⊥ ⅋ B⊥
    /// (A ⅋ B)⊥ = A⊥ ⊗ B⊥
    /// (A ⊸ B)⊥ = A ⊗ B⊥
    /// (A & B)⊥ = A⊥ ⊕ B⊥
    /// (A ⊕ B)⊥ = A⊥ & B⊥
    /// (!A)⊥ = ?(A⊥)
    /// (?A)⊥ = !(A⊥)
    /// 1⊥ = ⊥
    /// ⊥⊥ = 1
    /// ⊤⊥ = 0
    /// 0⊥ = ⊤
    /// ```
    pub fn dual(&self) -> LinearType {
        match self {
            LinearType::Tensor(a, b) => LinearType::par(a.dual(), b.dual()),
            LinearType::Par(a, b) => LinearType::tensor(a.dual(), b.dual()),
            LinearType::Lollipop(a, b) => {
                // (A ⊸ B)⊥ = A ⊗ B⊥
                LinearType::tensor((**a).clone(), b.dual())
            }
            LinearType::With(a, b) => LinearType::plus(a.dual(), b.dual()),
            LinearType::Plus(a, b) => LinearType::with(a.dual(), b.dual()),
            LinearType::Bang(a) => LinearType::quest(a.dual()),
            LinearType::Quest(a) => LinearType::bang(a.dual()),
            LinearType::One => LinearType::Bottom,
            LinearType::Bottom => LinearType::One,
            LinearType::Top => LinearType::Zero,
            LinearType::Zero => LinearType::Top,
            LinearType::Knowledge { .. } => self.clone(), // Knowledge is self-dual
            LinearType::Var(v) => LinearType::Var(format!("{}⊥", v)),
            LinearType::Unknown => LinearType::Unknown,
        }
    }

    /// Check if two types are definitionally equal
    pub fn definitionally_equal(&self, other: &LinearType) -> bool {
        match (self, other) {
            (
                LinearType::Knowledge {
                    inner: i1,
                    confidence: c1,
                    ontology: o1,
                    temporal: t1,
                    modality: m1,
                },
                LinearType::Knowledge {
                    inner: i2,
                    confidence: c2,
                    ontology: o2,
                    temporal: t2,
                    modality: m2,
                },
            ) => i1 == i2 && c1 == c2 && o1 == o2 && t1 == t2 && m1 == m2,

            (LinearType::Tensor(a1, b1), LinearType::Tensor(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (LinearType::With(a1, b1), LinearType::With(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (LinearType::Plus(a1, b1), LinearType::Plus(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (LinearType::Lollipop(a1, b1), LinearType::Lollipop(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (LinearType::Par(a1, b1), LinearType::Par(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (LinearType::Bang(a1), LinearType::Bang(a2)) => a1.definitionally_equal(a2),
            (LinearType::Quest(a1), LinearType::Quest(a2)) => a1.definitionally_equal(a2),
            (LinearType::One, LinearType::One) => true,
            (LinearType::Bottom, LinearType::Bottom) => true,
            (LinearType::Top, LinearType::Top) => true,
            (LinearType::Zero, LinearType::Zero) => true,
            (LinearType::Var(v1), LinearType::Var(v2)) => v1 == v2,
            (LinearType::Unknown, LinearType::Unknown) => true,
            _ => false,
        }
    }

    /// Get free type variables
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, vars: &mut HashSet<String>) {
        match self {
            LinearType::Var(v) => {
                vars.insert(v.clone());
            }
            LinearType::Tensor(a, b)
            | LinearType::With(a, b)
            | LinearType::Plus(a, b)
            | LinearType::Lollipop(a, b)
            | LinearType::Par(a, b) => {
                a.collect_free_vars(vars);
                b.collect_free_vars(vars);
            }
            LinearType::Bang(a) | LinearType::Quest(a) => {
                a.collect_free_vars(vars);
            }
            _ => {}
        }
    }

    /// Substitute a type variable
    pub fn substitute(&self, var: &str, replacement: &LinearType) -> LinearType {
        match self {
            LinearType::Var(v) if v == var => replacement.clone(),
            LinearType::Tensor(a, b) => LinearType::tensor(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            LinearType::With(a, b) => LinearType::with(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            LinearType::Plus(a, b) => LinearType::plus(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            LinearType::Lollipop(a, b) => LinearType::lollipop(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            LinearType::Par(a, b) => LinearType::par(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            LinearType::Bang(a) => LinearType::bang(a.substitute(var, replacement)),
            LinearType::Quest(a) => LinearType::quest(a.substitute(var, replacement)),
            _ => self.clone(),
        }
    }
}

impl fmt::Display for LinearType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearType::Knowledge {
                inner,
                confidence,
                modality,
                ..
            } => {
                write!(
                    f,
                    "{}Knowledge[{:?}, {}]",
                    modality.symbol(),
                    inner,
                    confidence
                )
            }
            LinearType::Tensor(a, b) => write!(f, "({} ⊗ {})", a, b),
            LinearType::With(a, b) => write!(f, "({} & {})", a, b),
            LinearType::Plus(a, b) => write!(f, "({} ⊕ {})", a, b),
            LinearType::Lollipop(a, b) => write!(f, "({} ⊸ {})", a, b),
            LinearType::Par(a, b) => write!(f, "({} ⅋ {})", a, b),
            LinearType::Bang(a) => write!(f, "!{}", a),
            LinearType::Quest(a) => write!(f, "?{}", a),
            LinearType::One => write!(f, "1"),
            LinearType::Bottom => write!(f, "⊥"),
            LinearType::Top => write!(f, "⊤"),
            LinearType::Zero => write!(f, "0"),
            LinearType::Var(v) => write!(f, "{}", v),
            LinearType::Unknown => write!(f, "?"),
        }
    }
}

// ============================================================================
// Type Aliases for Convenience
// ============================================================================

/// Tensor type alias
pub type TensorType = (Arc<LinearType>, Arc<LinearType>);

/// With type alias
pub type WithType = (Arc<LinearType>, Arc<LinearType>);

/// Plus type alias
pub type PlusType = (Arc<LinearType>, Arc<LinearType>);

/// Lollipop type alias
pub type LollipopType = (Arc<LinearType>, Arc<LinearType>);

// ============================================================================
// Specialized Types
// ============================================================================

/// Intervention type (linear by nature)
pub fn intervention_type(target: Type, value: Type) -> LinearType {
    LinearType::linear_knowledge(
        Type::Named {
            name: format!("Intervention[{:?}, {:?}]", target, value),
            args: vec![],
        },
        1.0,
        OntologyType::Concrete {
            ontology: "CAUSAL".to_string(),
            term: None,
        },
    )
}

/// Observation type (can be linear or affine)
pub fn observation_type(observed: Type, linear: bool) -> LinearType {
    let ont = OntologyType::Concrete {
        ontology: "OBSERVATION".to_string(),
        term: None,
    };
    if linear {
        LinearType::linear_knowledge(observed, 0.95, ont)
    } else {
        LinearType::affine_knowledge(observed, 0.95, ont)
    }
}

/// Credential type (affine - can expire unused)
pub fn credential_type(scope: &str) -> LinearType {
    LinearType::affine_knowledge(
        Type::Named {
            name: format!("Credential[{}]", scope),
            args: vec![],
        },
        1.0,
        OntologyType::Concrete {
            ontology: "AUTH".to_string(),
            term: None,
        },
    )
}

/// Published knowledge type (unrestricted)
pub fn published_type(inner: Type, confidence: f64) -> LinearType {
    LinearType::bang(LinearType::unrestricted_knowledge(
        inner,
        confidence,
        OntologyType::Concrete {
            ontology: "PUBLISHED".to_string(),
            term: None,
        },
    ))
}

/// Mandatory evidence type (relevant - must use)
pub fn mandatory_evidence_type(inner: Type, confidence: f64) -> LinearType {
    LinearType::relevant_knowledge(
        inner,
        confidence,
        OntologyType::Concrete {
            ontology: "REGULATORY".to_string(),
            term: None,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_modality() {
        let a = LinearType::linear_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let b = LinearType::linear_knowledge(Type::Bool, 0.8, OntologyType::Any);
        let tensor = LinearType::tensor(a, b);
        assert_eq!(tensor.modality(), Modality::Linear);
    }

    #[test]
    fn test_bang_is_unrestricted() {
        let a = LinearType::linear_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let bang = LinearType::bang(a);
        assert_eq!(bang.modality(), Modality::Unrestricted);
    }

    #[test]
    fn test_quest_is_affine() {
        let a = LinearType::linear_knowledge(Type::Bool, 0.9, OntologyType::Any);
        let quest = LinearType::quest(a);
        assert_eq!(quest.modality(), Modality::Affine);
    }

    #[test]
    fn test_dual_tensor_is_par() {
        let a = LinearType::One;
        let b = LinearType::One;
        let tensor = LinearType::tensor(a, b);
        let dual = tensor.dual();

        match dual {
            LinearType::Par(_, _) => {}
            _ => panic!("Expected Par, got {:?}", dual),
        }
    }

    #[test]
    fn test_dual_involution() {
        let typ = LinearType::tensor(LinearType::One, LinearType::Bang(Arc::new(LinearType::Top)));

        let dual1 = typ.dual();
        let dual2 = dual1.dual();

        // dual(dual(A)) = A (up to structural equality)
        assert!(typ.definitionally_equal(&dual2));
    }

    #[test]
    fn test_dual_units() {
        assert!(matches!(LinearType::One.dual(), LinearType::Bottom));
        assert!(matches!(LinearType::Bottom.dual(), LinearType::One));
        assert!(matches!(LinearType::Top.dual(), LinearType::Zero));
        assert!(matches!(LinearType::Zero.dual(), LinearType::Top));
    }

    #[test]
    fn test_free_vars() {
        let typ = LinearType::lollipop(
            LinearType::Var("A".to_string()),
            LinearType::Var("B".to_string()),
        );

        let vars = typ.free_vars();
        assert!(vars.contains("A"));
        assert!(vars.contains("B"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_substitute() {
        let typ = LinearType::Var("A".to_string());
        let replacement = LinearType::One;
        let result = typ.substitute("A", &replacement);

        assert!(matches!(result, LinearType::One));
    }

    #[test]
    fn test_intervention_type() {
        let interv = intervention_type(Type::Bool, Type::I32);
        assert!(interv.is_linear());
    }

    #[test]
    fn test_credential_type() {
        let cred = credential_type("admin");
        assert!(cred.is_affine());
    }

    #[test]
    fn test_published_type() {
        let pub_t = published_type(Type::Bool, 0.95);
        assert!(pub_t.is_unrestricted());
    }
}
