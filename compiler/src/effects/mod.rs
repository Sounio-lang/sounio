//! Effect system implementation
//!
//! This module re-exports the effect types from the types module and provides
//! additional runtime support for effect handling.
//!
//! # Continuation Support
//!
//! The `continuation` submodule provides the infrastructure for capturing and
//! resuming continuations, which is essential for implementing algebraic effect
//! handlers. See the module documentation for details on the theoretical
//! background and implementation approach.
//!
//! # Concrete Handlers
//!
//! The `handlers` submodule contains concrete implementations of effect handlers
//! using the `HandlerCapability` trait. These enable both Track A (CPS-based)
//! and Track B (epistemic) effect handling.

pub mod continuation;
pub mod epistemic_effects;
pub mod handler_capability;
pub mod handlers;
pub mod inference;

pub use crate::types::effects::*;
pub use continuation::{
    CapturedContinuation, ContinuationError, ContinuationId, ContinuationStore, ResumePoint,
};
pub use epistemic_effects::{
    apply_epistemic_tracking, ConfidenceModifier, EpistemicEvent, EpistemicImpactRegistry,
    EpistemicTracker,
};
pub use handler_capability::{
    ConfidenceModifier as HandlerConfidenceModifier, Continuation as HandlerContinuation,
    ContinuationId as HandlerContinuationId, EpistemicImpact, HandlerCapability, HandlerError,
    HandlerResult, HandlerState, OperationSpec, SuspensionId,
};
pub use handlers::{AllocHandler, AsyncHandler, IOHandler, MutHandler, PanicHandler, ProbHandler};
pub use inference::{EffectChecker, EffectError, EffectErrorKind};

/// Runtime effect handler trait
pub trait Handler<E> {
    type Output;

    fn handle(&self, effect: E) -> Self::Output;
}

/// Effect continuation
pub struct Continuation<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> Continuation<T> {
    /// Resume the continuation with a value
    pub fn resume(self, _value: T) {
        // Placeholder for continuation implementation
    }
}
