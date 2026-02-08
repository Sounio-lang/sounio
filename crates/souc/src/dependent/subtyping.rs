//! Subtyping rules for dependent epistemic types
//!
//! This module implements the subtyping relation for epistemic types,
//! following these key principles:
//!
//! # Subtyping Laws
//!
//! ```text
//! (S-REFL)   τ <: τ
//! (S-TRANS)  τ₁ <: τ₂ ∧ τ₂ <: τ₃ ⟹ τ₁ <: τ₃
//!
//! (K-COV)    ε₁ ≥ ε₂ ∧ δ₁ ⊇ δ₂ ⟹ Knowledge[τ,ε₁,δ₁] <: Knowledge[τ,ε₂,δ₂]
//!
//! (H-SK)     StructuralKnowledge <: CausalKnowledge
//! (H-CK)     CausalKnowledge <: Knowledge
//!
//! (R-WEAK)   {x:τ|P} <: τ
//! (R-IMPL)   P₁ ⟹ P₂ ⟹ {x:τ|P₁} <: {x:τ|P₂}
//!
//! (Π-SUB)    τ₁' <: τ₁ ∧ (∀x. τ₂ <: τ₂') ⟹ Π(x:τ₁).τ₂ <: Π(x:τ₁').τ₂'
//! ```
//!
//! Key insight: Higher confidence and more specific ontology is a SUBTYPE
//! (can be used where less is required).

use super::TypeContext;
use super::predicates::{ConfidencePredicate, Predicate};
use super::proofs::Proof;
use super::types::{ConfidenceType, EpistemicType, OntologyType};

/// Result of a subtyping check
#[derive(Debug, Clone)]
pub enum SubtypeResult {
    /// Types are in subtype relation with proof
    Subtype {
        /// The proof that sub <: sup
        proof: Proof,
    },

    /// Types are definitely not in subtype relation
    NotSubtype {
        /// Reason for failure
        reason: String,
    },

    /// Cannot determine (e.g., unbound variables)
    Unknown {
        /// What couldn't be determined
        reason: String,
    },
}

impl SubtypeResult {
    /// Check if this is a successful subtype result
    pub fn is_subtype(&self) -> bool {
        matches!(self, Self::Subtype { .. })
    }

    /// Check if this is a definite failure
    pub fn is_not_subtype(&self) -> bool {
        matches!(self, Self::NotSubtype { .. })
    }

    /// Check if this is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }

    /// Get the proof if successful
    pub fn proof(&self) -> Option<&Proof> {
        match self {
            Self::Subtype { proof } => Some(proof),
            _ => None,
        }
    }
}

/// Error during subtype checking
#[derive(Debug, Clone, thiserror::Error)]
pub enum SubtypeError {
    #[error("Types are incompatible: {0}")]
    Incompatible(String),

    #[error("Cannot prove constraint: {0}")]
    ProofFailed(String),

    #[error("Unbound variable: {0}")]
    UnboundVariable(String),

    #[error("Variance mismatch: expected {expected}, found {found}")]
    VarianceMismatch { expected: Variance, found: Variance },
}

/// Variance annotation for type parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variance {
    /// Covariant: preserves subtyping direction
    Covariant,
    /// Contravariant: reverses subtyping direction
    Contravariant,
    /// Invariant: requires equality
    Invariant,
    /// Bivariant: both directions allowed (rare)
    Bivariant,
}

impl std::fmt::Display for Variance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Covariant => write!(f, "covariant"),
            Self::Contravariant => write!(f, "contravariant"),
            Self::Invariant => write!(f, "invariant"),
            Self::Bivariant => write!(f, "bivariant"),
        }
    }
}

/// Subtype checker for epistemic types
pub struct SubtypeChecker<'a> {
    /// The type context
    ctx: &'a TypeContext,
    /// Whether to use gradual typing for unknowns
    gradual: bool,
}

impl<'a> SubtypeChecker<'a> {
    /// Create a new subtype checker
    pub fn new(ctx: &'a TypeContext) -> Self {
        Self {
            ctx,
            gradual: false,
        }
    }

    /// Enable gradual typing mode
    pub fn with_gradual(mut self, gradual: bool) -> Self {
        self.gradual = gradual;
        self
    }

    /// Check if sub is a subtype of sup
    pub fn check(&self, sub: &EpistemicType, sup: &EpistemicType) -> SubtypeResult {
        // S-REFL: identical types
        if self.definitionally_equal(sub, sup) {
            return SubtypeResult::Subtype {
                proof: Proof::refl(ConfidenceType::Unknown),
            };
        }

        match (sub, sup) {
            // Knowledge hierarchy: StructuralKnowledge <: CausalKnowledge <: Knowledge
            (EpistemicType::StructuralKnowledge { .. }, EpistemicType::CausalKnowledge { .. }) => {
                self.check_structural_to_causal(sub, sup)
            }

            (EpistemicType::StructuralKnowledge { .. }, EpistemicType::Knowledge { .. }) => {
                self.check_structural_to_knowledge(sub, sup)
            }

            (EpistemicType::CausalKnowledge { .. }, EpistemicType::Knowledge { .. }) => {
                self.check_causal_to_knowledge(sub, sup)
            }

            // Same-level knowledge subtyping (confidence and ontology covariance)
            (
                EpistemicType::Knowledge {
                    inner: inner1,
                    confidence: conf1,
                    ontology: ont1,
                    ..
                },
                EpistemicType::Knowledge {
                    inner: inner2,
                    confidence: conf2,
                    ontology: ont2,
                    ..
                },
            ) => self.check_knowledge_subtype(inner1, conf1, ont1, inner2, conf2, ont2),

            (
                EpistemicType::CausalKnowledge {
                    inner: inner1,
                    confidence: conf1,
                    ontology: ont1,
                    graph: graph1,
                    ..
                },
                EpistemicType::CausalKnowledge {
                    inner: inner2,
                    confidence: conf2,
                    ontology: ont2,
                    graph: graph2,
                    ..
                },
            ) => self.check_causal_knowledge_subtype(
                inner1, conf1, ont1, graph1, inner2, conf2, ont2, graph2,
            ),

            // Refinement subtyping
            (
                EpistemicType::Refinement {
                    base: base1,
                    predicate: pred1,
                },
                EpistemicType::Refinement {
                    base: base2,
                    predicate: pred2,
                },
            ) => self.check_refinement_subtype(base1, pred1, base2, pred2),

            // Refinement weakening: {x:τ|P} <: τ
            (EpistemicType::Refinement { base, .. }, sup) => self.check(base, sup),

            // Π-type subtyping (contravariant domain, covariant codomain)
            (
                EpistemicType::Pi {
                    param_name: name1,
                    param_type: ty1,
                    body: body1,
                },
                EpistemicType::Pi {
                    param_name: name2,
                    param_type: ty2,
                    body: body2,
                },
            ) => self.check_pi_subtype(name1, ty1, body1, name2, ty2, body2),

            // Σ-type subtyping (covariant both components)
            (
                EpistemicType::Sigma {
                    fst_name: name1,
                    fst_type: ty1,
                    snd_type: snd1,
                },
                EpistemicType::Sigma {
                    fst_name: name2,
                    fst_type: ty2,
                    snd_type: snd2,
                },
            ) => self.check_sigma_subtype(name1, ty1, snd1, name2, ty2, snd2),

            // Proof types are subtypes if predicates imply
            (EpistemicType::Proof(p1), EpistemicType::Proof(p2)) => {
                self.check_predicate_implication(p1, p2)
            }

            // Unknown (gradual typing)
            (EpistemicType::Unknown, _) | (_, EpistemicType::Unknown) => {
                if self.gradual {
                    SubtypeResult::Subtype {
                        proof: Proof::runtime_check(Predicate::true_()),
                    }
                } else {
                    SubtypeResult::Unknown {
                        reason: "Unknown type involved".to_string(),
                    }
                }
            }

            // Type variables
            (EpistemicType::Var(v1), EpistemicType::Var(v2)) if v1 == v2 => {
                SubtypeResult::Subtype {
                    proof: Proof::refl(ConfidenceType::Unknown),
                }
            }

            // Incompatible types
            _ => SubtypeResult::NotSubtype {
                reason: format!("Incompatible type constructors: {} vs {}", sub, sup),
            },
        }
    }

    /// Check knowledge subtyping with confidence and ontology covariance
    fn check_knowledge_subtype(
        &self,
        inner1: &crate::types::Type,
        conf1: &ConfidenceType,
        ont1: &OntologyType,
        inner2: &crate::types::Type,
        conf2: &ConfidenceType,
        ont2: &OntologyType,
    ) -> SubtypeResult {
        // Inner types must be equal (or subtype in structural type system)
        if inner1 != inner2 {
            return SubtypeResult::NotSubtype {
                reason: format!("Inner types differ: {:?} vs {:?}", inner1, inner2),
            };
        }

        // Confidence: conf1 ≥ conf2 (covariant - higher is subtype)
        let conf_result = self.check_confidence_geq(conf1, conf2);

        // Ontology: ont1 ⊇ ont2 (covariant - more specific is subtype)
        let ont_result = self.check_ontology_superset(ont1, ont2);

        match (conf_result, ont_result) {
            (SubtypeResult::Subtype { proof: p1 }, SubtypeResult::Subtype { proof: p2 }) => {
                SubtypeResult::Subtype {
                    proof: Proof::and_intro(p1, p2),
                }
            }
            (SubtypeResult::NotSubtype { reason }, _) => SubtypeResult::NotSubtype {
                reason: format!("Confidence check failed: {}", reason),
            },
            (_, SubtypeResult::NotSubtype { reason }) => SubtypeResult::NotSubtype {
                reason: format!("Ontology check failed: {}", reason),
            },
            (SubtypeResult::Unknown { reason: r1 }, _) => SubtypeResult::Unknown {
                reason: format!("Confidence unknown: {}", r1),
            },
            (_, SubtypeResult::Unknown { reason: r2 }) => SubtypeResult::Unknown {
                reason: format!("Ontology unknown: {}", r2),
            },
        }
    }

    /// Check causal knowledge subtyping
    #[allow(clippy::too_many_arguments)]
    fn check_causal_knowledge_subtype(
        &self,
        inner1: &crate::types::Type,
        conf1: &ConfidenceType,
        ont1: &OntologyType,
        graph1: &super::types::CausalGraphType,
        inner2: &crate::types::Type,
        conf2: &ConfidenceType,
        ont2: &OntologyType,
        graph2: &super::types::CausalGraphType,
    ) -> SubtypeResult {
        // First check basic knowledge subtyping
        let base_result = self.check_knowledge_subtype(inner1, conf1, ont1, inner2, conf2, ont2);

        if !base_result.is_subtype() {
            return base_result;
        }

        // Graphs must be compatible (sub must have at least as much structure)
        if !self.graph_is_subgraph(graph1, graph2) {
            return SubtypeResult::NotSubtype {
                reason: "Causal graph structure incompatible".to_string(),
            };
        }

        base_result
    }

    /// Check if graph1 is a subgraph of graph2 (has all edges of graph2)
    fn graph_is_subgraph(
        &self,
        graph1: &super::types::CausalGraphType,
        graph2: &super::types::CausalGraphType,
    ) -> bool {
        // graph1 should contain all nodes and edges of graph2
        graph2.nodes.is_subset(&graph1.nodes) && graph2.edges.is_subset(&graph1.edges)
    }

    /// Check StructuralKnowledge <: CausalKnowledge
    fn check_structural_to_causal(
        &self,
        sub: &EpistemicType,
        sup: &EpistemicType,
    ) -> SubtypeResult {
        if let (
            EpistemicType::StructuralKnowledge {
                inner: inner1,
                confidence: conf1,
                ontology: ont1,
                graph: graph1,
                ..
            },
            EpistemicType::CausalKnowledge {
                inner: inner2,
                confidence: conf2,
                ontology: ont2,
                graph: graph2,
                ..
            },
        ) = (sub, sup)
        {
            self.check_causal_knowledge_subtype(
                inner1, conf1, ont1, graph1, inner2, conf2, ont2, graph2,
            )
        } else {
            SubtypeResult::NotSubtype {
                reason: "Expected StructuralKnowledge and CausalKnowledge".to_string(),
            }
        }
    }

    /// Check StructuralKnowledge <: Knowledge
    fn check_structural_to_knowledge(
        &self,
        sub: &EpistemicType,
        sup: &EpistemicType,
    ) -> SubtypeResult {
        if let (
            EpistemicType::StructuralKnowledge {
                inner: inner1,
                confidence: conf1,
                ontology: ont1,
                ..
            },
            EpistemicType::Knowledge {
                inner: inner2,
                confidence: conf2,
                ontology: ont2,
                ..
            },
        ) = (sub, sup)
        {
            self.check_knowledge_subtype(inner1, conf1, ont1, inner2, conf2, ont2)
        } else {
            SubtypeResult::NotSubtype {
                reason: "Expected StructuralKnowledge and Knowledge".to_string(),
            }
        }
    }

    /// Check CausalKnowledge <: Knowledge
    fn check_causal_to_knowledge(&self, sub: &EpistemicType, sup: &EpistemicType) -> SubtypeResult {
        if let (
            EpistemicType::CausalKnowledge {
                inner: inner1,
                confidence: conf1,
                ontology: ont1,
                ..
            },
            EpistemicType::Knowledge {
                inner: inner2,
                confidence: conf2,
                ontology: ont2,
                ..
            },
        ) = (sub, sup)
        {
            self.check_knowledge_subtype(inner1, conf1, ont1, inner2, conf2, ont2)
        } else {
            SubtypeResult::NotSubtype {
                reason: "Expected CausalKnowledge and Knowledge".to_string(),
            }
        }
    }

    /// Check refinement subtyping: {x:τ|P₁} <: {x:τ|P₂} when P₁ ⟹ P₂
    fn check_refinement_subtype(
        &self,
        base1: &EpistemicType,
        pred1: &Predicate,
        base2: &EpistemicType,
        pred2: &Predicate,
    ) -> SubtypeResult {
        // First, bases must be subtypes
        let base_result = self.check(base1, base2);
        if !base_result.is_subtype() {
            return base_result;
        }

        // Then, pred1 must imply pred2
        self.check_predicate_implication(pred1, pred2)
    }

    /// Check Π-type subtyping: Π(x:A).B <: Π(x:A').B' when A' <: A and B <: B'
    fn check_pi_subtype(
        &self,
        _name1: &str,
        ty1: &crate::types::Type,
        body1: &EpistemicType,
        _name2: &str,
        ty2: &crate::types::Type,
        body2: &EpistemicType,
    ) -> SubtypeResult {
        // Contravariant in domain: ty2 <: ty1
        if ty1 != ty2 {
            // For now, require exact equality on parameter types
            // Full implementation would check structural subtyping
            return SubtypeResult::NotSubtype {
                reason: "Parameter types differ".to_string(),
            };
        }

        // Covariant in codomain: body1 <: body2
        self.check(body1, body2)
    }

    /// Check Σ-type subtyping: Σ(x:A).B <: Σ(x:A').B' when A <: A' and B <: B'
    fn check_sigma_subtype(
        &self,
        _name1: &str,
        ty1: &crate::types::Type,
        snd1: &EpistemicType,
        _name2: &str,
        ty2: &crate::types::Type,
        snd2: &EpistemicType,
    ) -> SubtypeResult {
        // Covariant in first component
        if ty1 != ty2 {
            return SubtypeResult::NotSubtype {
                reason: "First component types differ".to_string(),
            };
        }

        // Covariant in second component
        self.check(snd1, snd2)
    }

    /// Check conf1 ≥ conf2
    fn check_confidence_geq(
        &self,
        conf1: &ConfidenceType,
        conf2: &ConfidenceType,
    ) -> SubtypeResult {
        // Try to evaluate both
        if let (Some(v1), Some(v2)) = (conf1.evaluate(self.ctx), conf2.evaluate(self.ctx)) {
            if v1 >= v2 {
                return SubtypeResult::Subtype {
                    proof: Proof::literal_cmp(v1, v2).unwrap(),
                };
            } else {
                return SubtypeResult::NotSubtype {
                    reason: format!("{} < {}", v1, v2),
                };
            }
        }

        // Try using lower bounds
        if let (Some(lb1), Some(v2)) = (conf1.lower_bound(self.ctx), conf2.evaluate(self.ctx))
            && lb1 >= v2
        {
            return SubtypeResult::Subtype {
                proof: Proof::arith(
                    super::proofs::ArithDerivation::lower_bound(lb1, v2),
                    Predicate::confidence_geq(conf1.clone(), conf2.clone()),
                ),
            };
        }

        // Definitional equality
        if conf1.definitionally_equal(conf2) {
            return SubtypeResult::Subtype {
                proof: Proof::refl(conf1.clone()),
            };
        }

        // If gradual, allow with runtime check
        if self.gradual {
            SubtypeResult::Subtype {
                proof: Proof::runtime_check(Predicate::confidence_geq(
                    conf1.clone(),
                    conf2.clone(),
                )),
            }
        } else {
            SubtypeResult::Unknown {
                reason: format!("Cannot prove {} ≥ {}", conf1, conf2),
            }
        }
    }

    /// Check ont1 ⊇ ont2
    fn check_ontology_superset(&self, ont1: &OntologyType, ont2: &OntologyType) -> SubtypeResult {
        if ont1.contains(ont2) {
            SubtypeResult::Subtype {
                proof: Proof::trusted(
                    "ontology containment",
                    Predicate::ontology_superset(ont1.clone(), ont2.clone()),
                ),
            }
        } else if ont1.definitionally_equal(ont2) {
            SubtypeResult::Subtype {
                proof: Proof::refl(ConfidenceType::Unknown),
            }
        } else if self.gradual {
            SubtypeResult::Subtype {
                proof: Proof::runtime_check(Predicate::ontology_superset(
                    ont1.clone(),
                    ont2.clone(),
                )),
            }
        } else {
            SubtypeResult::NotSubtype {
                reason: format!("{} does not contain {}", ont1, ont2),
            }
        }
    }

    /// Check if pred1 implies pred2
    fn check_predicate_implication(&self, pred1: &Predicate, pred2: &Predicate) -> SubtypeResult {
        // Trivial cases
        if pred1 == pred2 {
            return SubtypeResult::Subtype {
                proof: Proof::refl(ConfidenceType::Unknown),
            };
        }

        if pred2.is_trivially_true() {
            return SubtypeResult::Subtype {
                proof: Proof::trusted("trivially true", pred2.clone()),
            };
        }

        if pred1.is_trivially_false() {
            // False implies anything
            return SubtypeResult::Subtype {
                proof: Proof::trusted("ex falso", Predicate::implies(pred1.clone(), pred2.clone())),
            };
        }

        // Try to evaluate confidence predicates
        if let (
            super::predicates::PredicateKind::Confidence(ConfidencePredicate::Geq(a1, b1)),
            super::predicates::PredicateKind::Confidence(ConfidencePredicate::Geq(a2, b2)),
        ) = (&pred1.kind, &pred2.kind)
        {
            // If a1 ≥ b1 and we need a2 ≥ b2
            // This holds if a1 ≥ a2 and b1 ≤ b2 (strengthening antecedent, weakening consequent)
            // Simplified: check if the predicates are compatible
            if a1.definitionally_equal(a2) && b1.definitionally_equal(b2) {
                return SubtypeResult::Subtype {
                    proof: Proof::refl(ConfidenceType::Unknown),
                };
            }
        }

        // If gradual, allow with runtime check
        if self.gradual {
            SubtypeResult::Subtype {
                proof: Proof::runtime_check(Predicate::implies(pred1.clone(), pred2.clone())),
            }
        } else {
            SubtypeResult::Unknown {
                reason: format!("Cannot prove {} ⟹ {}", pred1, pred2),
            }
        }
    }

    /// Check definitional equality of epistemic types
    fn definitionally_equal(&self, t1: &EpistemicType, t2: &EpistemicType) -> bool {
        match (t1, t2) {
            (
                EpistemicType::Knowledge {
                    inner: i1,
                    confidence: c1,
                    ontology: o1,
                    ..
                },
                EpistemicType::Knowledge {
                    inner: i2,
                    confidence: c2,
                    ontology: o2,
                    ..
                },
            ) => i1 == i2 && c1.definitionally_equal(c2) && o1.definitionally_equal(o2),

            (EpistemicType::Var(v1), EpistemicType::Var(v2)) => v1 == v2,

            (EpistemicType::Unknown, EpistemicType::Unknown) => true,

            (EpistemicType::Proof(p1), EpistemicType::Proof(p2)) => p1 == p2,

            _ => false,
        }
    }
}

/// Check if a type is well-formed in a context
pub fn check_well_formed(ctx: &TypeContext, ty: &EpistemicType) -> Result<(), SubtypeError> {
    match ty {
        EpistemicType::Knowledge { confidence, .. } => {
            // Check that all confidence variables are bound
            for var in confidence.free_vars() {
                if ctx.lookup_confidence(&var).is_none() {
                    return Err(SubtypeError::UnboundVariable(var));
                }
            }
            Ok(())
        }
        EpistemicType::Refinement { base, predicate } => {
            check_well_formed(ctx, base)?;
            // Check predicate variables are bound
            for var in predicate.free_vars() {
                if ctx.lookup_confidence(&var).is_none() {
                    return Err(SubtypeError::UnboundVariable(var));
                }
            }
            Ok(())
        }
        EpistemicType::Pi { body, .. } => check_well_formed(ctx, body),
        EpistemicType::Sigma { snd_type, .. } => check_well_formed(ctx, snd_type),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Type;

    #[test]
    fn test_reflexivity() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let ty = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let result = checker.check(&ty, &ty);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_confidence_covariance() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let high = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let low = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.90),
            OntologyType::concrete("PKPD"),
        );

        // High confidence is subtype of low (can be used where low is required)
        let result = checker.check(&high, &low);
        assert!(result.is_subtype());

        // Low confidence is NOT subtype of high
        let result2 = checker.check(&low, &high);
        assert!(result2.is_not_subtype());
    }

    #[test]
    fn test_ontology_covariance() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let pkpd_chebi = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::union(
                OntologyType::concrete("PKPD"),
                OntologyType::concrete("ChEBI"),
            ),
        );

        let pkpd_only = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        // More specific (union) is subtype of less specific
        let result = checker.check(&pkpd_chebi, &pkpd_only);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_knowledge_hierarchy() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let mut graph = super::super::types::CausalGraphType::new();
        graph.add_edge("X", "Y");

        let causal = EpistemicType::causal_knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
            graph,
        );

        let knowledge = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.90),
            OntologyType::concrete("PKPD"),
        );

        // CausalKnowledge <: Knowledge
        let result = checker.check(&causal, &knowledge);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_refinement_weakening() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx);

        let base = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        let refined = EpistemicType::refinement(
            base.clone(),
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.99)),
        );

        // Refined type is subtype of base (weakening)
        let result = checker.check(&refined, &base);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_gradual_unknown() {
        let ctx = TypeContext::new();
        let checker = SubtypeChecker::new(&ctx).with_gradual(true);

        let known = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        // Unknown is compatible with anything in gradual mode
        let result = checker.check(&EpistemicType::Unknown, &known);
        assert!(result.is_subtype());

        let result2 = checker.check(&known, &EpistemicType::Unknown);
        assert!(result2.is_subtype());
    }

    #[test]
    fn test_confidence_variable_bounds() {
        let mut ctx = TypeContext::new();
        ctx.bind_confidence("ε", ConfidenceType::literal(0.97));

        let checker = SubtypeChecker::new(&ctx);

        let with_var = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::var("ε"),
            OntologyType::concrete("PKPD"),
        );

        let with_literal = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        // ε = 0.97 ≥ 0.95, so this should succeed
        let result = checker.check(&with_var, &with_literal);
        assert!(result.is_subtype());
    }

    #[test]
    fn test_well_formed_check() {
        let mut ctx = TypeContext::new();
        let ty = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::var("ε"),
            OntologyType::concrete("PKPD"),
        );

        // Not well-formed: ε is unbound
        let result = check_well_formed(&ctx, &ty);
        assert!(result.is_err());

        // Now bind ε
        ctx.bind_confidence("ε", ConfidenceType::literal(0.95));
        let result2 = check_well_formed(&ctx, &ty);
        assert!(result2.is_ok());
    }
}
