//! Epistemic Composition Algebra - L0 Primitives for Knowledge Combination
//!
//! This module provides formal operators for combining epistemic knowledge:
//!
//! - **TENSOR (⊗)**: Combines independent knowledge about different aspects
//! - **JOIN (⊔)**: Fuses knowledge about the same phenomenon with conflict resolution
//! - **CONDITION (|)**: Bayesian/Jeffrey update with evidence
//! - **LIFT/EXTRACT**: Boundaries between pure values and epistemic domain
//!
//! # Theoretical Foundation
//!
//! The algebra is grounded in:
//! - Bayesian probability theory for conditioning
//! - Dempster-Shafer theory for combining uncertain evidence
//! - Jeffrey conditioning for uncertain evidence
//! - Category theory for compositional semantics
//!
//! # Algebraic Laws
//!
//! ## Tensor Laws
//! - (T1) Associativity: (K₁ ⊗ K₂) ⊗ K₃ ≅ K₁ ⊗ (K₂ ⊗ K₃)
//! - (T2) Commutativity: K₁ ⊗ K₂ ≅ K₂ ⊗ K₁
//! - (T3) Identity: K ⊗ I = K
//! - (T4) Monotonicity: ε(K₁) ≤ ε(K₂) ⟹ ε(K₁ ⊗ K) ≤ ε(K₂ ⊗ K)
//!
//! ## Join Laws
//! - (J1) Commutativity: K₁ ⊔ K₂ = K₂ ⊔ K₁
//! - (J2) Idempotence: K ⊔ K = K' where ε(K') ≥ ε(K)
//! - (J3) Concordance: conflict(K₁,K₂) = 0 ⟹ ε(K₁ ⊔ K₂) > max(ε(K₁),ε(K₂))
//! - (J4) Conflict: conflict(K₁,K₂) ≥ θ ⟹ K₁ ⊔ K₂ = Irreconcilable
//!
//! ## Condition Laws
//! - (C1) Neutral: K | E with (λ_ _. 0.5) = K
//! - (C2) Certainty: K | E with (λ_ _. 1.0) → ε = 1
//! - (C3) Composition: (K | E₁) | E₂ = K | (E₁, E₂)
//!
//! # Example
//!
//! ```rust,ignore
//! use sounio::epistemic::composition::*;
//!
//! // Combine independent measurements
//! let absorption = EpistemicValue::new(1.2, 0.90);
//! let elimination = EpistemicValue::new(0.3, 0.85);
//! let pk_model = absorption.tensor(elimination);
//!
//! // Fuse conflicting estimates
//! let estimate1 = EpistemicValue::new(5.2, 0.80);
//! let estimate2 = EpistemicValue::new(5.1, 0.78);
//! let fused = estimate1.join(estimate2, 0.3);
//!
//! // Update with evidence
//! let prior = EpistemicValue::new(true, 0.30);
//! let posterior = prior.condition(&evidence, |h, e| 0.95, 0.10);
//! ```

pub mod category;
pub mod condition;
pub mod confidence;
pub mod join;
pub mod knowledge;
pub mod provenance;
pub mod tensor;

// Re-export core types
pub use category::{EpistemicCategory, EpistemicFunctor, EpistemicMonad};
pub use condition::{ConditioningStrategy, Evidence, EvidenceStrength};
pub use confidence::{
    CombinationStrategy, ConfidenceError, ConfidenceValue, combine_confidence, select_strategy,
};
pub use join::{ConflictLevel, Fusible, JoinResult};
pub use knowledge::{EpistemicValue, ExtractError, OntologyRef as CompositionOntologyRef};
pub use provenance::{DerivationChain, ProvenanceNode, SourceInfo};
pub use tensor::ontology_correlation;
