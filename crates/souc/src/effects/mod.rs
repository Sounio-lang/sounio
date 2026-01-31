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
//!
//! # SIMD Parallel Dispatch
//!
//! The `simd_dispatch` submodule provides parallel effect dispatch for performing
//! multiple effect operations simultaneously:
//!
//! ```text
//! // Instead of sequential:
//! perform(msg1, print); perform(msg2, print); ... perform(msg8, print);
//!
//! // Use parallel simd_perform:
//! simd_perform([msg1..msg8], print);  // 8 parallel IOs
//! ```
//!
//! See [`SimdEffectDispatcher`] for the main API.

pub mod continuation;
pub mod continuation_context;
pub mod epistemic_effects;
pub mod handler_capability;
pub mod handlers;
pub mod inference;
pub mod jit_resume;
pub mod linearity;
pub mod resilience;
pub mod resilient_dispatch;
pub mod simd_dispatch;

pub use crate::types::effects::*;
pub use continuation::{
    CapturedContinuation, ContinuationError, ContinuationId, ContinuationStore, ResumePoint,
};
pub use continuation_context::ContinuationContext;
pub use epistemic_effects::{
    apply_epistemic_tracking, ConfidenceModifier, EpistemicEvent, EpistemicImpactRegistry,
    EpistemicTracker,
};
pub use handler_capability::{
    ConfidenceModifier as HandlerConfidenceModifier, Continuation as HandlerContinuation,
    ContinuationId as HandlerContinuationId, EpistemicImpact, HandlerCapability, HandlerError,
    HandlerResult, HandlerState, OperationSpec, SuspensionId,
};
pub use handlers::{
    AllocHandler, AsyncHandler, DefaultHandlerFactory, DivHandler, EpistemicHandler, ExnHandler,
    GpuHandler, HandlerFactory, HandlerRegistry, HandlerRegistryBuilder, IOHandler, MutHandler,
    NetworkHandler, PanicHandler, ProbHandler, SensorHandler,
};
pub use inference::{EffectChecker, EffectError, EffectErrorKind, EffectSource, TypeInfo};
pub use jit_resume::{
    clear_jit_resume_state, f64_to_value, has_pending_jit_resume, resume_jit_continuation,
    schedule_jit_resume, store_jit_context, take_pending_jit_resume, JitResumeContext,
    JitResumeStore,
};
pub use linearity::{EffectOpInfo, Linearity};
pub use resilience::{with_retry, CircuitBreaker, CircuitState, RetryConfig, RetryResult};
pub use resilient_dispatch::{
    DispatchConfig, DispatchOutcome, ResilientBuilder, ResilientDispatcher,
};
pub use simd_dispatch::{
    simd_perform, simd_perform_batch, BatchEffectResult, DispatchError, DispatchHandle, EffectOp,
    ParallelEffectResult, ParallelTiming, SimdContinuation, SimdDispatchConfig,
    SimdEffectDispatcher,
};

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
