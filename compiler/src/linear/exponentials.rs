//! Exponential modalities (! and ?)
//!
//! The exponentials provide controlled access to structural rules:
//!
//! - `!A` (of course): A can be used any number of times (including zero)
//! - `?A` (why not): A can be discarded but not necessarily duplicated
//!
//! In categorical terms:
//! - `!` is a comonad with dereliction and digging
//! - `?` is a monad (dual of !)
//!
//! Key rules:
//! ```text
//! !A → A              (dereliction)
//! !A → !!A            (digging)
//! !A → !A ⊗ !A        (contraction)
//! !A → 1              (weakening)
//! !A ⊗ !B ≅ !(A & B)  (Seely isomorphism)
//! ```

use std::fmt;
use std::sync::Arc;

use super::linear_types::LinearType;
use super::modality::Modality;

/// Bang type (!A) - the "of course" modality
///
/// Represents a resource that can be used any number of times.
/// In epistemic terms, this is published or established knowledge.
#[derive(Clone, Debug)]
pub struct BangType {
    /// The underlying type
    pub inner: Arc<LinearType>,
}

impl BangType {
    /// Create a new bang type
    pub fn new(inner: LinearType) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Dereliction: !A → A
    ///
    /// Use the bang-typed resource once, "forgetting" that it's unlimited.
    pub fn dereliction(&self) -> LinearType {
        (*self.inner).clone()
    }

    /// Digging: !A → !!A
    ///
    /// Wrap in another bang layer.
    pub fn digging(&self) -> BangType {
        BangType::new(LinearType::Bang(self.inner.clone()))
    }

    /// Contraction: !A → !A ⊗ !A
    ///
    /// Duplicate the resource.
    pub fn contraction(&self) -> LinearType {
        let bang = LinearType::Bang(self.inner.clone());
        LinearType::tensor(bang.clone(), bang)
    }

    /// Weakening: !A → 1
    ///
    /// Discard the resource.
    pub fn weakening(&self) -> LinearType {
        LinearType::One
    }

    /// Check if inner type satisfies condition for promotion
    ///
    /// To form !A, all free variables in A must already be !-typed.
    pub fn can_promote(inner: &LinearType) -> bool {
        match inner {
            LinearType::Knowledge { modality, .. } => *modality == Modality::Unrestricted,
            LinearType::Tensor(a, b) => Self::can_promote(a) && Self::can_promote(b),
            LinearType::With(a, b) => Self::can_promote(a) && Self::can_promote(b),
            LinearType::Plus(a, b) => Self::can_promote(a) && Self::can_promote(b),
            LinearType::Lollipop(a, b) => Self::can_promote(a) && Self::can_promote(b),
            LinearType::Par(a, b) => Self::can_promote(a) && Self::can_promote(b),
            LinearType::Bang(_) => true,
            LinearType::Quest(a) => Self::can_promote(a),
            LinearType::One | LinearType::Top => true,
            LinearType::Bottom | LinearType::Zero => true,
            LinearType::Var(_) => false, // Variables might not be unrestricted
            LinearType::Unknown => true, // Gradual typing allows promotion
        }
    }
}

impl fmt::Display for BangType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "!{}", self.inner)
    }
}

/// Quest type (?A) - the "why not" modality
///
/// Represents a resource that can be discarded (affine).
/// Dual of ! in linear logic.
#[derive(Clone, Debug)]
pub struct QuestType {
    /// The underlying type
    pub inner: Arc<LinearType>,
}

impl QuestType {
    /// Create a new quest type
    pub fn new(inner: LinearType) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Unit: A → ?A
    ///
    /// Any type can be weakened to a quest type.
    pub fn unit(inner: LinearType) -> Self {
        Self::new(inner)
    }

    /// Multiplication: ??A → ?A
    ///
    /// Nested quest collapses.
    pub fn multiplication(nested: QuestType) -> QuestType {
        // ??A → ?A: unwrap one layer if inner is also ?
        match &*nested.inner {
            LinearType::Quest(inner) => QuestType::new((**inner).clone()),
            _ => nested,
        }
    }
}

impl fmt::Display for QuestType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}", self.inner)
    }
}

/// Seely isomorphism: !A ⊗ !B ≅ !(A & B)
///
/// This isomorphism is fundamental to the categorical semantics of linear logic.
pub fn seely_forward(a: &BangType, b: &BangType) -> BangType {
    // !A ⊗ !B → !(A & B)
    let with = LinearType::with((*a.inner).clone(), (*b.inner).clone());
    BangType::new(with)
}

/// Inverse of Seely isomorphism: !(A & B) → !A ⊗ !B
pub fn seely_backward(ab: &BangType) -> Option<(BangType, BangType)> {
    match &*ab.inner {
        LinearType::With(a, b) => {
            Some((BangType::new((**a).clone()), BangType::new((**b).clone())))
        }
        _ => None,
    }
}

/// Unit isomorphism: 1 ≅ !⊤
pub fn unit_iso_forward() -> BangType {
    BangType::new(LinearType::Top)
}

pub fn unit_iso_backward(bang: &BangType) -> Option<LinearType> {
    match &*bang.inner {
        LinearType::Top => Some(LinearType::One),
        _ => None,
    }
}

/// Comonoid structure on !A
///
/// Every !A forms a cocommutative comonoid with:
/// - counit: !A → 1 (weakening)
/// - comult: !A → !A ⊗ !A (contraction)
pub struct BangComonoid {
    pub typ: BangType,
}

impl BangComonoid {
    pub fn new(typ: BangType) -> Self {
        Self { typ }
    }

    /// Counit (weakening)
    pub fn counit(&self) -> LinearType {
        self.typ.weakening()
    }

    /// Comultiplication (contraction)
    pub fn comult(&self) -> LinearType {
        self.typ.contraction()
    }

    /// n-ary contraction: !A → !A ⊗ ... ⊗ !A (n times)
    pub fn contract_n(&self, n: usize) -> LinearType {
        if n == 0 {
            return LinearType::One;
        }

        let bang = LinearType::Bang(self.typ.inner.clone());
        let mut result = bang.clone();

        for _ in 1..n {
            result = LinearType::tensor(result, bang.clone());
        }

        result
    }
}

/// Promotion rule checker
///
/// To promote Γ ⊢ A to !Γ ⊢ !A, all variables in Γ must be !-typed.
pub fn check_promotion_context(context_types: &[(&str, &LinearType)]) -> Result<(), String> {
    for (name, typ) in context_types {
        if !BangType::can_promote(typ) {
            return Err(format!(
                "Cannot promote context: variable '{}' has non-unrestricted type {}",
                name, typ
            ));
        }
    }
    Ok(())
}

/// Exponential coercion rules
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExponentialCoercion {
    /// !A → A (dereliction)
    Dereliction,
    /// !A → !!A (digging)
    Digging,
    /// !A → !A ⊗ !A (contraction)
    Contraction,
    /// !A → 1 (weakening)
    Weakening,
    /// A → ?A (? introduction)
    QuestIntro,
    /// ??A → ?A (? multiplication)
    QuestMult,
}

impl ExponentialCoercion {
    /// Apply the coercion to a type
    pub fn apply(&self, typ: &LinearType) -> Option<LinearType> {
        match self {
            ExponentialCoercion::Dereliction => {
                if let LinearType::Bang(inner) = typ {
                    Some((**inner).clone())
                } else {
                    None
                }
            }
            ExponentialCoercion::Digging => {
                if let LinearType::Bang(_) = typ {
                    Some(LinearType::bang(typ.clone()))
                } else {
                    None
                }
            }
            ExponentialCoercion::Contraction => {
                if let LinearType::Bang(_) = typ {
                    Some(LinearType::tensor(typ.clone(), typ.clone()))
                } else {
                    None
                }
            }
            ExponentialCoercion::Weakening => {
                if let LinearType::Bang(_) = typ {
                    Some(LinearType::One)
                } else {
                    None
                }
            }
            ExponentialCoercion::QuestIntro => Some(LinearType::quest(typ.clone())),
            ExponentialCoercion::QuestMult => {
                if let LinearType::Quest(inner) = typ {
                    if let LinearType::Quest(inner2) = &**inner {
                        Some(LinearType::quest((**inner2).clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bang_dereliction() {
        let inner = LinearType::One;
        let bang = BangType::new(inner.clone());
        let derel = bang.dereliction();
        assert!(matches!(derel, LinearType::One));
    }

    #[test]
    fn test_bang_contraction() {
        let inner = LinearType::One;
        let bang = BangType::new(inner);
        let contracted = bang.contraction();

        match contracted {
            LinearType::Tensor(a, b) => {
                assert!(matches!(&*a, LinearType::Bang(_)));
                assert!(matches!(&*b, LinearType::Bang(_)));
            }
            _ => panic!("Expected tensor"),
        }
    }

    #[test]
    fn test_bang_weakening() {
        let inner = LinearType::One;
        let bang = BangType::new(inner);
        let weak = bang.weakening();
        assert!(matches!(weak, LinearType::One));
    }

    #[test]
    fn test_seely_isomorphism() {
        let a = BangType::new(LinearType::One);
        let b = BangType::new(LinearType::Top);

        let combined = seely_forward(&a, &b);

        // Should be !(One & Top)
        match &*combined.inner {
            LinearType::With(_, _) => {}
            _ => panic!("Expected With type"),
        }

        // Backward should recover original
        let (a2, b2) = seely_backward(&combined).unwrap();
        assert!(matches!(&*a2.inner, LinearType::One));
        assert!(matches!(&*b2.inner, LinearType::Top));
    }

    #[test]
    fn test_comonoid_contract_n() {
        let bang = BangType::new(LinearType::One);
        let comonoid = BangComonoid::new(bang);

        let zero = comonoid.contract_n(0);
        assert!(matches!(zero, LinearType::One));

        let one = comonoid.contract_n(1);
        assert!(matches!(one, LinearType::Bang(_)));

        let two = comonoid.contract_n(2);
        assert!(matches!(two, LinearType::Tensor(_, _)));
    }

    #[test]
    fn test_can_promote_unrestricted() {
        let unrest = LinearType::unrestricted_knowledge(
            crate::types::Type::Bool,
            0.9,
            crate::dependent::types::OntologyType::Any,
        );
        assert!(BangType::can_promote(&unrest));
    }

    #[test]
    fn test_cannot_promote_linear() {
        let linear = LinearType::linear_knowledge(
            crate::types::Type::Bool,
            0.9,
            crate::dependent::types::OntologyType::Any,
        );
        assert!(!BangType::can_promote(&linear));
    }

    #[test]
    fn test_exponential_coercion_dereliction() {
        let bang = LinearType::bang(LinearType::One);
        let result = ExponentialCoercion::Dereliction.apply(&bang);
        assert!(matches!(result, Some(LinearType::One)));
    }

    #[test]
    fn test_exponential_coercion_quest_intro() {
        let typ = LinearType::One;
        let result = ExponentialCoercion::QuestIntro.apply(&typ);
        assert!(matches!(result, Some(LinearType::Quest(_))));
    }
}
