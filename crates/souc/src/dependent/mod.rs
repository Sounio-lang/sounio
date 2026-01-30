//! Dependent Epistemic Types - The Capstone of the Type System
//!
//! Day 35 of the Sounio compiler implements dependent types for epistemic
//! knowledge, enabling compile-time guarantees about confidence levels,
//! ontology bindings, and causal identifiability.
//!
//! # The Problem Solved
//!
//! Before Day 35, runtime failures were possible:
//! ```ignore
//! fn require_high_confidence<T>(k: Knowledge<T>) -> T {
//!     k.extract(0.95).unwrap()  // Could panic if ε < 0.95
//! }
//! ```
//!
//! With dependent types, extraction is statically safe:
//! ```ignore
//! fn require_high_confidence<T>(k: Knowledge<T, ε ≥ 0.95>) -> T {
//!     k.extract()  // Cannot fail - guaranteed by TYPE
//! }
//! ```
//!
//! # Theoretical Foundation
//!
//! Based on Martin-Löf Type Theory:
//!
//! - **Π-Types** (Dependent Products): Functions where return type depends on argument
//! - **Σ-Types** (Dependent Sums): Pairs with existential quantification
//! - **Refinement Types**: Types with predicates `{x : τ | P(x)}`
//! - **Proof Terms**: Evidence for type-level claims
//!
//! # Key Components
//!
//! - [`ConfidenceType`]: Type-level confidence expressions
//! - [`OntologyType`]: Type-level ontology specifications
//! - [`Predicate`]: Refinement predicates
//! - [`Proof`]: Proof terms for type-level claims
//! - [`EpistemicType`]: The full dependent epistemic type
//!
//! # Integration with Days 32-34
//!
//! ```text
//! Day 32 (Composition)  → Type-level ε₁ * ε₂ * γ
//! Day 33 (Temporal)     → Type-level decay(ε, λ, t)
//! Day 34 (Causal)       → Type-level identifiable(G, X, Y)
//! Day 35 (Dependent)    → Proofs that these hold
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! // Safe extraction with compile-time proof
//! fn safe_extract<T>(k: Knowledge<T, ε: ε ≥ 0.95>) -> T {
//!     k.extract()  // Proven safe at compile time
//! }
//!
//! // Causal effect with identifiability proof
//! fn causal_effect<G: CausalGraph>(
//!     k: CausalKnowledge<Effect, ε, δ, G>,
//!     intervention: Intervention,
//! ) -> InterventionResult<Effect>
//! where
//!     Proof<identifiable(G, intervention.target, k.outcome)>
//! {
//!     k.do_intervention(intervention)
//! }
//! ```

pub mod gradual;
pub mod inference;
pub mod predicates;
pub mod proof_search;
pub mod proofs;
pub mod subtyping;
pub mod types;

// Re-exports
pub use gradual::{
    GradualAnnotation, GradualConfig, GradualDiagnostics, GradualMode, GradualWarning,
    RuntimeCheck, RuntimeCheckKind, SourceLocation, confidence_check,
};
pub use inference::{
    Constraint, ConstraintKind, ConstraintSolver, InferenceContext, InferenceError,
    InferenceResult, TypeVar, TypeVarKind,
};
pub use predicates::{
    CausalPredicate, ConfidencePredicate, OntologyPredicate, Predicate, PredicateKind,
    TemporalPredicate,
};
pub use proof_search::{ProofResult, ProofSearchConfig, ProofSearcher, SearchStrategy};
pub use proofs::{ArithDerivation, CausalProof, Proof, ProofKind, ProofTerm};
pub use subtyping::{SubtypeChecker, SubtypeError, SubtypeResult, Variance};
pub use types::{
    CausalGraphType, ConfidenceType, EpistemicType, OntologyType, ProvenanceType, TemporalType,
};

/// Type context for dependent type checking
#[derive(Debug, Clone, Default)]
pub struct TypeContext {
    /// Type variable bindings
    pub bindings: std::collections::HashMap<String, ConfidenceType>,
    /// Known predicates (assumptions in scope)
    pub assumptions: Vec<Predicate>,
    /// Ontology type bindings
    pub ontology_bindings: std::collections::HashMap<String, OntologyType>,
    /// Causal graph type bindings
    pub graph_bindings: std::collections::HashMap<String, CausalGraphType>,
    /// Whether gradual typing is enabled
    pub gradual_mode: GradualMode,
}

impl TypeContext {
    /// Create a new empty type context
    pub fn new() -> Self {
        Self::default()
    }

    /// Create context with gradual typing enabled
    pub fn with_gradual(mode: GradualMode) -> Self {
        Self {
            gradual_mode: mode,
            ..Default::default()
        }
    }

    /// Add a confidence binding
    pub fn bind_confidence(&mut self, name: impl Into<String>, conf: ConfidenceType) {
        self.bindings.insert(name.into(), conf);
    }

    /// Add an assumption
    pub fn assume(&mut self, pred: Predicate) {
        self.assumptions.push(pred);
    }

    /// Look up a confidence binding
    pub fn lookup_confidence(&self, name: &str) -> Option<&ConfidenceType> {
        self.bindings.get(name)
    }

    /// Check if a predicate is assumed
    pub fn is_assumed(&self, pred: &Predicate) -> bool {
        self.assumptions.iter().any(|a| a == pred)
    }

    /// Extend context with new bindings (for entering scopes)
    pub fn extend(&self) -> Self {
        self.clone()
    }

    /// Add ontology binding
    pub fn bind_ontology(&mut self, name: impl Into<String>, ont: OntologyType) {
        self.ontology_bindings.insert(name.into(), ont);
    }

    /// Look up an ontology binding
    pub fn lookup_ontology(&self, name: &str) -> Option<&OntologyType> {
        self.ontology_bindings.get(name)
    }

    /// Add causal graph binding
    pub fn bind_graph(&mut self, name: impl Into<String>, graph: CausalGraphType) {
        self.graph_bindings.insert(name.into(), graph);
    }

    /// Look up a causal graph binding
    pub fn lookup_graph(&self, name: &str) -> Option<&CausalGraphType> {
        self.graph_bindings.get(name)
    }
}

/// Errors that can occur during dependent type checking
#[derive(Debug, Clone, thiserror::Error)]
pub enum DependentTypeError {
    #[error("Cannot prove predicate: {predicate}")]
    ProofFailed {
        predicate: String,
        reason: Option<String>,
    },

    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("Confidence constraint violated: {0}")]
    ConfidenceViolation(String),

    #[error("Ontology constraint violated: {0}")]
    OntologyViolation(String),

    #[error("Causal identifiability failed: {0}")]
    CausalIdentifiabilityFailed(String),

    #[error("Unbound type variable: {0}")]
    UnboundVariable(String),

    #[error("Subtyping failed: {sub} is not a subtype of {sup}")]
    SubtypingFailed { sub: String, sup: String },

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Gradual type error at runtime: {0}")]
    GradualRuntimeError(String),
}

/// Result type for dependent type operations
pub type DependentResult<T> = Result<T, DependentTypeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_context_creation() {
        let ctx = TypeContext::new();
        assert!(ctx.bindings.is_empty());
        assert!(ctx.assumptions.is_empty());
    }

    #[test]
    fn test_confidence_binding() {
        let mut ctx = TypeContext::new();
        ctx.bind_confidence("ε", ConfidenceType::Literal(0.95));
        assert!(ctx.lookup_confidence("ε").is_some());
    }

    #[test]
    fn test_context_extension() {
        let mut ctx = TypeContext::new();
        ctx.bind_confidence("ε", ConfidenceType::Literal(0.95));

        let extended = ctx.extend();
        assert!(extended.lookup_confidence("ε").is_some());
    }
}
