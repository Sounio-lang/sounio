//! Linear Epistemic Types - Knowledge as Computational Resource
//!
//! Day 36 of the Sounio compiler implements Girard's Linear Logic (1987)
//! for epistemic knowledge, modeling knowledge as a computational resource
//! with precise usage semantics.
//!
//! # The Problem with Classical Logic
//!
//! Classical epistemic logic assumes structural rules that are epistemically false:
//!
//! ```text
//! WEAKENING:    Γ ⊢ A ⟹ Γ, B ⊢ A     "Can ignore knowledge"
//! CONTRACTION:  Γ, A, A ⊢ B ⟹ Γ, A ⊢ B "Can duplicate knowledge"
//! ```
//!
//! These fail in scientific contexts:
//! - In clinical trials, you CANNOT ignore collected data
//! - In quantum measurements, you CANNOT measure twice without perturbation
//! - In Bayesian updating, order of evidence may matter
//!
//! # The Solution: Linear Types
//!
//! Knowledge is a RESOURCE with usage constraints:
//!
//! ```text
//! LINEAR (use exactly once):
//!   - Destructive measurements
//!   - One-time observations
//!   - Zero-knowledge proofs
//!
//! AFFINE (use at most once):
//!   - Access credentials
//!   - Authorization tokens
//!   - Expiring opportunities
//!
//! RELEVANT (use at least once):
//!   - Mandatory evidence
//!   - Regulatory data
//!   - Audit requirements
//!
//! UNRESTRICTED (use freely):
//!   - Published knowledge
//!   - Scientific laws
//!   - Reference ontologies
//! ```
//!
//! # Linear Connectives
//!
//! ```text
//! A ⊗ B    Tensor (both must be used)
//! A & B    With (internal choice)
//! A ⊕ B    Plus (external choice)
//! A ⊸ B    Linear implication (lollipop)
//! !A       Of course (unlimited use)
//! ?A       Why not (can discard)
//! ```
//!
//! # Session Types
//!
//! Protocols for epistemic communication:
//!
//! ```text
//! Send[A].S    Send A, continue with S
//! Recv[A].S    Receive A, continue with S
//! Choice[S,T]  Offer choice
//! Select[S,T]  Make choice
//! End          Session complete
//! ```
//!
//! # Integration with Days 32-35
//!
//! ```text
//! Day 32 (Composition) → Tensor is linear
//! Day 33 (Temporal)    → Decay consumes freshness
//! Day 34 (Causal)      → Intervention consumes observability
//! Day 35 (Dependent)   → Proofs as resources
//! Day 36 (Linear)      → Resource semantics for all
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! // Quantum measurement is linear
//! fn measure(state: LinearK<QuantumState>) -> Observation {
//!     // state is consumed - cannot measure again
//!     state.observe()
//! }
//!
//! // Credential is affine (can discard without using)
//! fn maybe_access(cred: AffineK<Credential>) -> Option<Session> {
//!     if confirmed() {
//!         Some(use_credential(cred))  // Consumed
//!     } else {
//!         None  // Discarded (affine allows)
//!     }
//! }
//!
//! // Published knowledge is unrestricted
//! fn cite(paper: !Knowledge<τ>) -> (τ, !Knowledge<τ>) {
//!     let value = extract(paper);  // Can extract multiple times
//!     (value, paper)               // Paper still available
//! }
//! ```

pub mod consumption;
pub mod context;
pub mod exponentials;
pub mod integrate;
pub mod kind;
pub mod linear_types;
pub mod modality;
pub mod resource;
pub mod session_types;
pub mod subtyping;
pub mod typed;

// Re-exports
pub use consumption::{ConsumptionError, ConsumptionTracker, ResourceState};
pub use context::{LinearBinding, LinearContext, UsageCount};
pub use exponentials::{BangType, QuestType};
pub use kind::{KindedType, Linearity, LinearityBound, LinearityError};
pub use linear_types::{LinearType, LollipopType, PlusType, TensorType, WithType};
pub use modality::Modality;
pub use resource::{Capability, LinearResource, ResourceHandle, ResourceKind, ResourceType};
pub use session_types::{ProtocolError, SessionType};
pub use subtyping::{LinearSubtypeChecker, LinearSubtypeError};
pub use typed::{Affine, Linear, LinearChoice, LinearPair, LinearRef, Unrestricted};

// Integration types
pub use integrate::{
    Credential, Intervention, LinearKnowledge, LinearQuantity, Observation, PublishedKnowledge,
};

use std::collections::HashMap;

/// Linear type checking context
#[derive(Debug, Clone, Default)]
pub struct LinearTypeContext {
    /// Variable bindings with modality
    pub bindings: HashMap<String, (LinearType, Modality)>,
    /// Usage tracking
    pub usage: HashMap<String, UsageCount>,
    /// Current session states
    pub sessions: HashMap<String, SessionType>,
    /// Whether to allow gradual typing
    pub gradual_mode: bool,
}

impl LinearTypeContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Create context with gradual typing enabled
    pub fn with_gradual(gradual: bool) -> Self {
        Self {
            gradual_mode: gradual,
            ..Default::default()
        }
    }

    /// Add a linear binding
    pub fn bind_linear(&mut self, name: impl Into<String>, typ: LinearType) {
        let name = name.into();
        self.bindings.insert(name.clone(), (typ, Modality::Linear));
        self.usage.insert(name, UsageCount::Zero);
    }

    /// Add an affine binding
    pub fn bind_affine(&mut self, name: impl Into<String>, typ: LinearType) {
        let name = name.into();
        self.bindings.insert(name.clone(), (typ, Modality::Affine));
        self.usage.insert(name, UsageCount::Zero);
    }

    /// Add a relevant binding
    pub fn bind_relevant(&mut self, name: impl Into<String>, typ: LinearType) {
        let name = name.into();
        self.bindings
            .insert(name.clone(), (typ, Modality::Relevant));
        self.usage.insert(name, UsageCount::Zero);
    }

    /// Add an unrestricted binding
    pub fn bind_unrestricted(&mut self, name: impl Into<String>, typ: LinearType) {
        let name = name.into();
        self.bindings
            .insert(name.clone(), (typ, Modality::Unrestricted));
        self.usage.insert(name, UsageCount::Zero);
    }

    /// Use a variable (increment usage count)
    pub fn use_var(&mut self, name: &str) -> Result<&LinearType, LinearError> {
        if let Some(count) = self.usage.get_mut(name) {
            *count = count.increment();
        } else {
            return Err(LinearError::UnboundVariable(name.to_string()));
        }

        self.bindings
            .get(name)
            .map(|(typ, _)| typ)
            .ok_or_else(|| LinearError::UnboundVariable(name.to_string()))
    }

    /// Look up a binding without using it
    pub fn lookup(&self, name: &str) -> Option<(&LinearType, &Modality)> {
        self.bindings.get(name).map(|(t, m)| (t, m))
    }

    /// Get usage count for a variable
    pub fn get_usage(&self, name: &str) -> Option<UsageCount> {
        self.usage.get(name).copied()
    }

    /// Check that all linear and relevant bindings are properly used
    pub fn check_exhausted(&self) -> Result<(), LinearError> {
        for (name, (_, modality)) in &self.bindings {
            let usage = self.usage.get(name).copied().unwrap_or(UsageCount::Zero);

            match (modality, usage) {
                (Modality::Linear, UsageCount::Zero) => {
                    return Err(LinearError::UnusedLinear(name.clone()));
                }
                (Modality::Linear, UsageCount::Many) => {
                    return Err(LinearError::OverusedLinear(name.clone()));
                }
                (Modality::Relevant, UsageCount::Zero) => {
                    return Err(LinearError::UnusedRelevant(name.clone()));
                }
                (Modality::Affine, UsageCount::Many) => {
                    return Err(LinearError::OverusedAffine(name.clone()));
                }
                _ => {} // OK
            }
        }
        Ok(())
    }

    /// Register a session
    pub fn register_session(&mut self, name: impl Into<String>, session: SessionType) {
        self.sessions.insert(name.into(), session);
    }

    /// Get session state
    pub fn get_session(&self, name: &str) -> Option<&SessionType> {
        self.sessions.get(name)
    }

    /// Advance session state
    pub fn advance_session(
        &mut self,
        name: &str,
        new_state: SessionType,
    ) -> Result<(), LinearError> {
        if self.sessions.contains_key(name) {
            self.sessions.insert(name.to_string(), new_state);
            Ok(())
        } else {
            Err(LinearError::UnknownSession(name.to_string()))
        }
    }
}

/// Errors that can occur during linear type checking
#[derive(Debug, Clone, thiserror::Error)]
pub enum LinearError {
    #[error("Unbound variable: {0}")]
    UnboundVariable(String),

    #[error("Linear variable '{0}' was not used (must use exactly once)")]
    UnusedLinear(String),

    #[error("Linear variable '{0}' was used more than once")]
    OverusedLinear(String),

    #[error("Relevant variable '{0}' was not used (must use at least once)")]
    UnusedRelevant(String),

    #[error("Affine variable '{0}' was used more than once (can use at most once)")]
    OverusedAffine(String),

    #[error("Expected tensor type, found: {0}")]
    ExpectedTensor(String),

    #[error("Expected linear function, found: {0}")]
    ExpectedFunction(String),

    #[error("Expected bang type (!), found: {0}")]
    ExpectedBang(String),

    #[error("Expected linear knowledge, found: {0}")]
    ExpectedLinearKnowledge(String),

    #[error("Cannot promote to ! because context contains non-unrestricted binding: {0}")]
    CannotPromote(String),

    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("Modality mismatch: expected {expected:?}, found {found:?}")]
    ModalityMismatch { expected: Modality, found: Modality },

    #[error("Confidence proof failed")]
    ConfidenceProofFailed,

    #[error("Unknown session: {0}")]
    UnknownSession(String),

    #[error("Session protocol violation: {0}")]
    SessionViolation(String),

    #[error("Session not complete (expected End)")]
    SessionNotComplete,

    #[error("Context split failed: {0}")]
    ContextSplitFailed(String),

    #[error("Consumption error: {0}")]
    ConsumptionError(String),

    #[error("Subtyping failed: {0}")]
    SubtypingFailed(String),
}

/// Result type for linear operations
pub type LinearResult<T> = Result<T, LinearError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = LinearTypeContext::new();
        assert!(ctx.bindings.is_empty());
    }

    #[test]
    fn test_linear_binding() {
        let mut ctx = LinearTypeContext::new();
        ctx.bind_linear("x", LinearType::One);
        assert!(ctx.lookup("x").is_some());
        assert_eq!(ctx.get_usage("x"), Some(UsageCount::Zero));
    }

    #[test]
    fn test_use_var() {
        let mut ctx = LinearTypeContext::new();
        ctx.bind_linear("x", LinearType::One);

        let _ = ctx.use_var("x").unwrap();
        assert_eq!(ctx.get_usage("x"), Some(UsageCount::One));
    }

    #[test]
    fn test_linear_exhaustion_unused() {
        let mut ctx = LinearTypeContext::new();
        ctx.bind_linear("x", LinearType::One);

        // Not used - should fail
        let result = ctx.check_exhausted();
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_exhaustion_used_once() {
        let mut ctx = LinearTypeContext::new();
        ctx.bind_linear("x", LinearType::One);

        let _ = ctx.use_var("x").unwrap();

        // Used exactly once - should succeed
        let result = ctx.check_exhausted();
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_can_be_unused() {
        let mut ctx = LinearTypeContext::new();
        ctx.bind_affine("x", LinearType::One);

        // Not used - should be OK for affine
        let result = ctx.check_exhausted();
        assert!(result.is_ok());
    }

    #[test]
    fn test_relevant_must_be_used() {
        let mut ctx = LinearTypeContext::new();
        ctx.bind_relevant("x", LinearType::One);

        // Not used - should fail for relevant
        let result = ctx.check_exhausted();
        assert!(result.is_err());
    }

    #[test]
    fn test_unrestricted_any_usage() {
        let mut ctx = LinearTypeContext::new();
        ctx.bind_unrestricted("x", LinearType::One);

        // Use multiple times
        let _ = ctx.use_var("x").unwrap();
        let _ = ctx.use_var("x").unwrap();
        let _ = ctx.use_var("x").unwrap();

        // Should be OK
        let result = ctx.check_exhausted();
        assert!(result.is_ok());
    }
}
