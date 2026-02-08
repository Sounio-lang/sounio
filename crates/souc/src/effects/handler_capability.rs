//! Handler Capability Interface for Algebraic Effects
//!
//! This module defines the contract for effect handlers, enabling:
//! - Parallel development of handler implementations (Track A: foundational, Track B: epistemic)
//! - Both foundational (CPS/continuation-based) and epistemic effect handlers
//! - Future extension to row-polymorphic effects
//!
//! # Architecture
//!
//! The `HandlerCapability` trait is the core interface that all effect handlers must implement.
//! This enables:
//! - Track A work: foundational handlers with real continuation capture/restore (CPS transform)
//! - Track B work: epistemic handlers with confidence tracking and provenance
//!
//! # Example
//!
//! ```ignore
//! use sounio::effects::handler_capability::*;
//!
//! #[derive(Debug)]
//! struct MyProbHandler;
//!
//! impl HandlerCapability for MyProbHandler {
//!     fn effect_name(&self) -> &str { "Prob" }
//!     fn handler_name(&self) -> &str { "MyProbHandler" }
//!     fn operations(&self) -> &[OperationSpec] { &[] }
//!     fn handle(&self, op: &str, args: &[Value], cont: Continuation, state: &mut HandlerState)
//!         -> HandlerResult {
//!         HandlerResult::Resume(Value::Unit)
//!     }
//! }
//! ```

use crate::effects::epistemic_effects::{
    ConfidenceModifier as EpistemicConfidenceModifier, EpistemicImpactRegistry, EpistemicTracker,
};
use crate::interp::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, instrument, warn};

/// Global counter for generating unique continuation IDs
static CONTINUATION_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique continuation ID
fn next_continuation_id() -> ContinuationId {
    ContinuationId(CONTINUATION_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
}

/// Unique identifier for a continuation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContinuationId(pub u64);

/// Specification for an effect operation
#[derive(Debug, Clone)]
pub struct OperationSpec {
    /// Operation name (e.g., "sample", "read_sensor")
    pub name: String,
    /// Parameter type names (for documentation/checking)
    pub param_types: Vec<String>,
    /// Return type name
    pub return_type: String,
    /// Whether this operation uses the continuation
    pub uses_continuation: bool,
    /// Epistemic impact: how this operation affects confidence
    pub confidence_factor: Option<f64>,
}

impl OperationSpec {
    pub fn new(name: impl Into<String>, return_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_types: Vec::new(),
            return_type: return_type.into(),
            uses_continuation: false,
            confidence_factor: None,
        }
    }

    pub fn with_params(mut self, params: Vec<&str>) -> Self {
        self.param_types = params.into_iter().map(String::from).collect();
        self
    }

    pub fn with_continuation(mut self) -> Self {
        self.uses_continuation = true;
        self
    }

    pub fn with_confidence_factor(mut self, factor: f64) -> Self {
        self.confidence_factor = Some(factor);
        self
    }
}

/// Result of handling an effect operation
#[derive(Debug)]
pub enum HandlerResult {
    /// Return a value immediately (operation complete)
    Return(Value),
    /// Resume the continuation with a value
    Resume(Value),
    /// Suspend execution (for async or multi-shot)
    Suspend(SuspensionId),
    /// Abort with an error
    Abort(HandlerError),
}

/// Unique ID for suspended computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SuspensionId(pub u64);

/// Error from handler execution
///
/// Extended for D.4 production hardening with:
/// - Source error chain for debugging
/// - Context map for request correlation
/// - Timestamps for error timing analysis
#[derive(Debug, Clone)]
pub struct HandlerError {
    /// Human-readable error message
    pub message: String,
    /// Effect that produced the error
    pub effect: String,
    /// Operation that failed
    pub operation: String,
    /// Source error message (for error chaining)
    pub source: Option<String>,
    /// Contextual information (request_id, handler_name, etc.)
    pub context: HashMap<String, String>,
    /// When the error occurred (monotonic, for duration calculations)
    pub timestamp: Option<Instant>,
}

impl HandlerError {
    pub fn new(
        effect: impl Into<String>,
        operation: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let effect_str = effect.into();
        let operation_str = operation.into();
        let message_str = message.into();

        // Log the error at warn level for observability
        warn!(
            effect = %effect_str,
            operation = %operation_str,
            "Handler error: {}",
            message_str
        );

        Self {
            effect: effect_str,
            operation: operation_str,
            message: message_str,
            source: None,
            context: HashMap::new(),
            timestamp: Some(Instant::now()),
        }
    }

    /// Create an error with a source cause
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add context to the error
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Create from another error type
    pub fn from_error<E: std::error::Error>(
        effect: impl Into<String>,
        operation: impl Into<String>,
        error: E,
    ) -> Self {
        Self::new(effect, operation, error.to_string()).with_source(format!("{:?}", error))
    }

    /// Get a formatted error chain for debugging
    pub fn error_chain(&self) -> String {
        let mut chain = self.message.clone();
        if let Some(ref source) = self.source {
            chain.push_str("\n  caused by: ");
            chain.push_str(source);
        }
        chain
    }
}

/// Epistemic metadata for handler operations
///
/// This is the primary epistemic impact type used internally.
#[derive(Debug, Clone, Default)]
pub struct EpistemicImpact {
    /// Multiplicative factor on confidence (1.0 = no change)
    pub confidence_factor: f64,
    /// Provenance tag to add
    pub provenance_tag: Option<String>,
    /// Whether this operation crosses a firewall boundary
    pub crosses_firewall: bool,
}

impl EpistemicImpact {
    pub fn none() -> Self {
        Self {
            confidence_factor: 1.0,
            provenance_tag: None,
            crosses_firewall: false,
        }
    }

    pub fn with_confidence(factor: f64) -> Self {
        Self {
            confidence_factor: factor,
            provenance_tag: None,
            crosses_firewall: false,
        }
    }

    pub fn with_provenance(mut self, tag: impl Into<String>) -> Self {
        self.provenance_tag = Some(tag.into());
        self
    }
}

/// Confidence modifier for Track B epistemic effects
///
/// This struct defines how an effect operation modifies the epistemic
/// confidence of values flowing through it. It enables:
/// - Confidence degradation/amplification through factor multiplication
/// - Minimum confidence floors (e.g., for unreliable sources)
/// - Provenance tracking for audit trails
///
/// # Example
///
/// ```ignore
/// // A sensor reading that degrades confidence by 10% and floors at 0.5
/// let modifier = ConfidenceModifier {
///     factor: 0.9,
///     min_confidence: 0.5,
///     provenance_tag: "temperature_sensor_v2".to_string(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ConfidenceModifier {
    /// Multiplicative factor for confidence (e.g., 0.9 = 10% degradation)
    pub factor: f64,
    /// Floor confidence at this minimum value
    pub min_confidence: f64,
    /// Provenance tag to add to the chain
    pub provenance_tag: String,
}

impl ConfidenceModifier {
    /// Create a new confidence modifier
    pub fn new(factor: f64, min_confidence: f64, provenance_tag: impl Into<String>) -> Self {
        Self {
            factor,
            min_confidence,
            provenance_tag: provenance_tag.into(),
        }
    }

    /// Create a modifier that only adds a provenance tag (no confidence change)
    pub fn provenance_only(tag: impl Into<String>) -> Self {
        Self {
            factor: 1.0,
            min_confidence: 0.0,
            provenance_tag: tag.into(),
        }
    }

    /// Create a modifier that degrades confidence by a percentage
    pub fn degrade(percentage: f64, tag: impl Into<String>) -> Self {
        Self {
            factor: 1.0 - (percentage / 100.0),
            min_confidence: 0.0,
            provenance_tag: tag.into(),
        }
    }

    /// Apply this modifier to a confidence value
    pub fn apply(&self, confidence: f64) -> f64 {
        (confidence * self.factor).max(self.min_confidence)
    }
}

impl Default for ConfidenceModifier {
    fn default() -> Self {
        Self {
            factor: 1.0,
            min_confidence: 0.0,
            provenance_tag: String::new(),
        }
    }
}

impl From<ConfidenceModifier> for EpistemicImpact {
    fn from(modifier: ConfidenceModifier) -> Self {
        EpistemicImpact {
            confidence_factor: modifier.factor,
            provenance_tag: if modifier.provenance_tag.is_empty() {
                None
            } else {
                Some(modifier.provenance_tag)
            },
            crosses_firewall: false,
        }
    }
}

/// Handler state passed to operations
pub struct HandlerState {
    /// Random number generator (for Prob)
    pub rng_seed: u64,
    /// Named state storage
    pub named_state: std::collections::HashMap<String, Value>,
    /// Accumulated epistemic impact (legacy field, prefer epistemic_tracker)
    pub epistemic_impact: EpistemicImpact,
    /// Epistemic tracker for detailed confidence tracking through effect operations
    pub epistemic_tracker: EpistemicTracker,
    /// Registry for looking up epistemic impacts of effect operations
    pub impact_registry: EpistemicImpactRegistry,
    /// Handler-specific data
    pub custom_data: Box<dyn std::any::Any + Send>,
}

impl Default for HandlerState {
    fn default() -> Self {
        Self {
            rng_seed: 42,
            named_state: std::collections::HashMap::new(),
            epistemic_impact: EpistemicImpact::none(),
            epistemic_tracker: EpistemicTracker::new(),
            impact_registry: EpistemicImpactRegistry::new(),
            custom_data: Box::new(()),
        }
    }
}

impl HandlerState {
    /// Create a new handler state with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a handler state without history recording (for performance)
    pub fn without_history() -> Self {
        Self {
            epistemic_tracker: EpistemicTracker::without_history(),
            ..Self::default()
        }
    }

    /// Record an effect operation's epistemic impact using the registry
    pub fn record_effect(&mut self, effect: &str, operation: &str) {
        let modifier = self.impact_registry.get_impact(effect, operation);
        self.epistemic_tracker.record(effect, operation, modifier);
    }

    /// Record an effect operation with a custom modifier
    pub fn record_effect_with_modifier(
        &mut self,
        effect: &str,
        operation: &str,
        modifier: EpistemicConfidenceModifier,
    ) {
        self.epistemic_tracker.record(effect, operation, modifier);
    }

    /// Get the final confidence after all recorded operations
    pub fn final_confidence(&self, initial: f64) -> f64 {
        self.epistemic_tracker.final_confidence(initial)
    }

    /// Check if any operations have degraded confidence
    pub fn has_degradation(&self) -> bool {
        self.epistemic_tracker.has_degradation()
    }

    /// Reset the epistemic tracker (for reuse)
    pub fn reset_epistemic(&mut self) {
        self.epistemic_tracker.reset();
    }
}

/// Continuation for resuming computation
///
/// This is the core abstraction for delimited continuations in the effect system.
///
/// # Track A (Foundational Handlers)
///
/// Track A implements actual continuation capture/restore using CPS transformation.
/// The continuation captures the rest of the computation after an effect operation,
/// allowing handlers to:
/// - Resume execution with a value
/// - Discard the continuation (abort)
/// - Resume multiple times (multi-shot, for effects like `Amb`)
///
/// # Track B (Epistemic Effects)
///
/// Track B can wrap continuations to add confidence tracking, ensuring that
/// epistemic metadata flows correctly through effect handlers.
///
/// # Placeholder Status
///
/// Currently, the Continuation type is a placeholder. Track A will implement
/// the actual CPS-based continuation capture mechanism.
pub struct Continuation {
    /// Unique ID for this continuation
    id: ContinuationId,
    /// Whether this continuation can be cloned (multi-shot)
    is_multi_shot: bool,
    /// Resume function (will be implemented by Track A)
    /// Using Arc to enable try_clone for multi-shot continuations
    resume_fn: Option<std::sync::Arc<dyn Fn(Value) -> Value + Send + Sync>>,
}

impl Continuation {
    /// Create a placeholder continuation (for initial development)
    pub fn placeholder(id: u64) -> Self {
        Self {
            id: ContinuationId(id),
            is_multi_shot: false,
            resume_fn: None,
        }
    }

    /// Create a new continuation with auto-generated ID
    pub fn new() -> Self {
        Self {
            id: next_continuation_id(),
            is_multi_shot: false,
            resume_fn: None,
        }
    }

    /// Create with actual resume function (single-shot)
    pub fn with_resume<F>(id: u64, f: F) -> Self
    where
        F: Fn(Value) -> Value + Send + Sync + 'static,
    {
        Self {
            id: ContinuationId(id),
            is_multi_shot: false,
            resume_fn: Some(std::sync::Arc::new(f)),
        }
    }

    /// Create a multi-shot continuation
    pub fn multi_shot<F>(id: u64, f: F) -> Self
    where
        F: Fn(Value) -> Value + Send + Sync + 'static,
    {
        Self {
            id: ContinuationId(id),
            is_multi_shot: true,
            resume_fn: Some(std::sync::Arc::new(f)),
        }
    }

    /// Get the continuation's unique ID
    pub fn id(&self) -> ContinuationId {
        self.id
    }

    /// Check if this is a multi-shot continuation
    pub fn is_multi_shot(&self) -> bool {
        self.is_multi_shot
    }

    /// Resume the continuation with a value
    ///
    /// This consumes the continuation (for single-shot) or uses a clone (for multi-shot).
    pub fn resume(self, value: Value) -> Value {
        if let Some(f) = self.resume_fn {
            f(value)
        } else {
            // Placeholder: just return the value
            value
        }
    }

    /// Try to clone this continuation (for multi-shot effects)
    ///
    /// Returns `None` if this is a single-shot continuation.
    /// Multi-shot continuations (used for effects like `Amb` or `Choice`)
    /// can be resumed multiple times.
    pub fn try_clone(&self) -> Option<Continuation> {
        if self.is_multi_shot {
            Some(Continuation {
                id: self.id,
                is_multi_shot: true,
                resume_fn: self.resume_fn.clone(),
            })
        } else {
            None
        }
    }
}

impl Default for Continuation {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for Continuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Continuation")
            .field("id", &self.id)
            .field("is_multi_shot", &self.is_multi_shot)
            .field("has_resume_fn", &self.resume_fn.is_some())
            .finish()
    }
}

/// The core handler capability trait
///
/// Implementations:
/// - Track A: Foundational handlers with real continuation support
/// - Track B: Epistemic handlers with confidence tracking
pub trait HandlerCapability: Debug {
    /// Name of the effect this handler handles
    fn effect_name(&self) -> &str;

    /// Human-readable handler name
    fn handler_name(&self) -> &str;

    /// Operations this handler provides
    fn operations(&self) -> &[OperationSpec];

    /// Handle an effect operation
    fn handle(
        &self,
        operation: &str,
        args: &[Value],
        continuation: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult;

    /// Get the epistemic impact of an operation (Track B extension point)
    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        // Default: look up from operation spec
        self.operations()
            .iter()
            .find(|op| op.name == operation)
            .and_then(|op| op.confidence_factor)
            .map(EpistemicImpact::with_confidence)
            .unwrap_or_else(EpistemicImpact::none)
    }

    /// Whether this handler supports multi-shot continuations
    fn supports_multi_shot(&self) -> bool {
        false
    }

    /// Get the linearity constraint for an operation
    ///
    /// Returns the linearity for a specific operation:
    /// - `ExactlyOnce`: Linear - continuation must be resumed exactly once (default)
    /// - `OnceOrNever`: Affine - continuation may be resumed zero or one times
    /// - `MultiShot`: Unrestricted - continuation may be resumed multiple times
    ///
    /// Default is `ExactlyOnce` for safety.
    fn operation_linearity(&self, _operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        Linearity::ExactlyOnce
    }
}

/// Instrumented handler dispatch with tracing and timing
///
/// Wraps a handler call with:
/// - Tracing span for the operation
/// - Timing measurement
/// - Debug logging for entry/exit
/// - Warning logging for errors
///
/// Use this for production-hardened handler dispatch.
#[instrument(
    skip(handler, args, continuation, state),
    fields(
        effect = %handler.effect_name(),
        handler = %handler.handler_name(),
    )
)]
pub fn instrumented_dispatch<H: HandlerCapability + ?Sized>(
    handler: &H,
    operation: &str,
    args: &[Value],
    continuation: Continuation,
    state: &mut HandlerState,
) -> HandlerResult {
    let start = Instant::now();
    debug!(
        operation = %operation,
        num_args = args.len(),
        "Dispatching effect operation"
    );

    let result = handler.handle(operation, args, continuation, state);
    let elapsed = start.elapsed();

    match &result {
        HandlerResult::Return(val) => {
            debug!(
                operation = %operation,
                elapsed_us = elapsed.as_micros(),
                "Handler returned: {:?}",
                val
            );
        }
        HandlerResult::Resume(val) => {
            debug!(
                operation = %operation,
                elapsed_us = elapsed.as_micros(),
                "Handler resumed with: {:?}",
                val
            );
        }
        HandlerResult::Suspend(id) => {
            debug!(
                operation = %operation,
                elapsed_us = elapsed.as_micros(),
                suspension_id = ?id,
                "Handler suspended"
            );
        }
        HandlerResult::Abort(err) => {
            warn!(
                operation = %operation,
                elapsed_us = elapsed.as_micros(),
                error = %err.message,
                source = ?err.source,
                "Handler aborted"
            );
        }
    }

    result
}

/// Panic-safe handler dispatch
///
/// Wraps handler execution in `catch_unwind` to convert panics to errors.
/// This prevents handler panics from crashing the entire runtime.
///
/// # Safety
///
/// This function catches panics and converts them to `HandlerResult::Abort`.
/// The handler must be `UnwindSafe` - most handlers satisfy this by not
/// holding mutable references across the call boundary.
///
/// # Example
///
/// ```ignore
/// use sounio::effects::handler_capability::safe_dispatch;
///
/// let result = safe_dispatch(&handler, "print", &args, cont, &mut state);
/// // Even if handler panics, we get HandlerResult::Abort instead of unwinding
/// ```
pub fn safe_dispatch<H: HandlerCapability + std::panic::RefUnwindSafe + ?Sized>(
    handler: &H,
    operation: &str,
    args: &[Value],
    continuation: Continuation,
    state: &mut HandlerState,
) -> HandlerResult {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let effect_name = handler.effect_name().to_string();
    let handler_name = handler.handler_name().to_string();
    let operation = operation.to_string();

    // We need to wrap state in AssertUnwindSafe since it contains non-UnwindSafe types
    // This is safe because we're not resuming from the panic - we're converting it to an error
    let result = catch_unwind(AssertUnwindSafe(|| {
        handler.handle(&operation, args, continuation, state)
    }));

    match result {
        Ok(handler_result) => handler_result,
        Err(panic_payload) => {
            // Extract panic message
            let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };

            warn!(
                effect = %effect_name,
                handler = %handler_name,
                operation = %operation,
                "Handler panicked: {}",
                panic_msg
            );

            HandlerResult::Abort(
                HandlerError::new(
                    &effect_name,
                    &operation,
                    format!("Handler panicked: {}", panic_msg),
                )
                .with_context("handler", handler_name)
                .with_context("panic_recovered", "true"),
            )
        }
    }
}

/// Helper macro for defining simple handlers
#[macro_export]
macro_rules! define_handler {
    ($name:ident for $effect:expr => {
        $($op:ident($($arg:ident),*) => $body:expr),* $(,)?
    }) => {
        #[derive(Debug)]
        pub struct $name;

        impl $crate::effects::handler_capability::HandlerCapability for $name {
            fn effect_name(&self) -> &str { $effect }
            fn handler_name(&self) -> &str { stringify!($name) }

            fn operations(&self) -> &[$crate::effects::handler_capability::OperationSpec] {
                // Note: This returns a slice from a vec, which requires static storage
                // For now, we return an empty slice - real implementations should
                // store operations in a static or use a different pattern
                &[]
            }

            fn handle(
                &self,
                operation: &str,
                args: &[$crate::interp::Value],
                continuation: $crate::effects::handler_capability::Continuation,
                state: &mut $crate::effects::handler_capability::HandlerState,
            ) -> $crate::effects::handler_capability::HandlerResult {
                match operation {
                    $(stringify!($op) => {
                        let result = $body;
                        $crate::effects::handler_capability::HandlerResult::Resume(result)
                    })*
                    _ => $crate::effects::handler_capability::HandlerResult::Abort(
                        $crate::effects::handler_capability::HandlerError::new(
                            self.effect_name(),
                            operation,
                            "Unknown operation"
                        )
                    )
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestHandler;

    impl HandlerCapability for TestHandler {
        fn effect_name(&self) -> &str {
            "Test"
        }
        fn handler_name(&self) -> &str {
            "TestHandler"
        }

        fn operations(&self) -> &[OperationSpec] {
            // Use a static slice for test purposes
            static OPS: &[OperationSpec] = &[];
            OPS
        }

        fn handle(
            &self,
            operation: &str,
            _args: &[Value],
            _continuation: Continuation,
            _state: &mut HandlerState,
        ) -> HandlerResult {
            match operation {
                "get_value" => HandlerResult::Resume(Value::Int(42)),
                "set_value" => HandlerResult::Resume(Value::Unit),
                _ => HandlerResult::Abort(HandlerError::new("Test", operation, "Unknown")),
            }
        }
    }

    #[test]
    fn test_handler_capability() {
        let handler = TestHandler;
        assert_eq!(handler.effect_name(), "Test");
        assert_eq!(handler.handler_name(), "TestHandler");
    }

    #[test]
    fn test_handler_handle() {
        let handler = TestHandler;
        let mut state = HandlerState::default();
        let continuation = Continuation::placeholder(1);

        match handler.handle("get_value", &[], continuation, &mut state) {
            HandlerResult::Resume(Value::Int(42)) => {}
            _ => panic!("Expected Resume(Int(42))"),
        }
    }

    #[test]
    fn test_epistemic_impact() {
        let impact = EpistemicImpact::with_confidence(0.95).with_provenance("sensor_reading");
        assert_eq!(impact.confidence_factor, 0.95);
        assert_eq!(impact.provenance_tag, Some("sensor_reading".to_string()));
    }

    #[test]
    fn test_operation_spec_builder() {
        let spec = OperationSpec::new("sample", "f64")
            .with_params(vec!["Distribution"])
            .with_continuation()
            .with_confidence_factor(0.9);

        assert_eq!(spec.name, "sample");
        assert_eq!(spec.return_type, "f64");
        assert_eq!(spec.param_types, vec!["Distribution"]);
        assert!(spec.uses_continuation);
        assert_eq!(spec.confidence_factor, Some(0.9));
    }

    #[test]
    fn test_handler_error() {
        let err = HandlerError::new("Prob", "sample", "Invalid distribution");
        assert_eq!(err.effect, "Prob");
        assert_eq!(err.operation, "sample");
        assert_eq!(err.message, "Invalid distribution");
    }

    #[test]
    fn test_suspension_id() {
        let id1 = SuspensionId(1);
        let id2 = SuspensionId(1);
        let id3 = SuspensionId(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_continuation_placeholder() {
        let cont = Continuation::placeholder(42);
        assert_eq!(cont.id(), ContinuationId(42));
        assert!(!cont.is_multi_shot());

        // Placeholder continuation just returns the value
        let result = cont.resume(Value::Int(100));
        match result {
            Value::Int(100) => {}
            _ => panic!("Expected Int(100)"),
        }
    }

    #[test]
    fn test_continuation_with_resume() {
        let cont = Continuation::with_resume(1, |v| {
            if let Value::Int(n) = v {
                Value::Int(n * 2)
            } else {
                v
            }
        });

        let result = cont.resume(Value::Int(21));
        match result {
            Value::Int(42) => {}
            _ => panic!("Expected Int(42)"),
        }
    }

    #[test]
    fn test_handler_state_default() {
        let state = HandlerState::default();
        assert_eq!(state.rng_seed, 42);
        assert!(state.named_state.is_empty());
        assert_eq!(state.epistemic_impact.confidence_factor, 1.0);
    }

    #[test]
    fn test_handler_state_epistemic_tracking() {
        let mut state = HandlerState::new();

        // Initially no degradation
        assert!(!state.has_degradation());
        assert!((state.final_confidence(1.0) - 1.0).abs() < 0.001);

        // Record an IO operation
        state.record_effect("IO", "read_file");

        // Should now have degradation (0.95 factor)
        assert!(state.has_degradation());
        assert!((state.final_confidence(1.0) - 0.95).abs() < 0.001);

        // Record a Prob operation
        state.record_effect("Prob", "sample");

        // Confidence should now be 0.95 * 0.9 = 0.855
        assert!((state.final_confidence(1.0) - 0.855).abs() < 0.001);
    }

    #[test]
    fn test_handler_state_reset_epistemic() {
        let mut state = HandlerState::new();

        state.record_effect("IO", "read_file");
        assert!(state.has_degradation());

        state.reset_epistemic();
        assert!(!state.has_degradation());
        assert!((state.final_confidence(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_handler_state_custom_modifier() {
        use crate::effects::epistemic_effects::ConfidenceModifier;

        let mut state = HandlerState::new();
        let modifier = ConfidenceModifier::with_factor_and_provenance(0.8, "custom_sensor");

        state.record_effect_with_modifier("Sensor", "read", modifier);

        assert!(state.has_degradation());
        assert!((state.final_confidence(1.0) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_handler_state_without_history() {
        let mut state = HandlerState::without_history();

        state.record_effect("IO", "read_file");

        // Degradation is still tracked
        assert!(state.has_degradation());
        assert!((state.final_confidence(1.0) - 0.95).abs() < 0.001);

        // But history is not recorded (for performance)
        assert_eq!(state.epistemic_tracker.event_count(), 0);
    }

    /// Handler that panics for testing panic safety
    #[derive(Debug)]
    struct PanickingHandler;

    impl std::panic::RefUnwindSafe for PanickingHandler {}

    impl HandlerCapability for PanickingHandler {
        fn effect_name(&self) -> &str {
            "Panic"
        }
        fn handler_name(&self) -> &str {
            "PanickingHandler"
        }
        fn operations(&self) -> &[OperationSpec] {
            &[]
        }
        fn handle(
            &self,
            operation: &str,
            _args: &[Value],
            _continuation: Continuation,
            _state: &mut HandlerState,
        ) -> HandlerResult {
            if operation == "panic" {
                panic!("intentional test panic");
            }
            HandlerResult::Resume(Value::Unit)
        }
    }

    #[test]
    fn test_safe_dispatch_catches_panic() {
        let handler = PanickingHandler;
        let mut state = HandlerState::default();
        let continuation = Continuation::placeholder(1);

        // Normal operation should work
        let result = safe_dispatch(
            &handler,
            "normal",
            &[],
            Continuation::placeholder(2),
            &mut state,
        );
        assert!(matches!(result, HandlerResult::Resume(Value::Unit)));

        // Panicking operation should be caught and converted to Abort
        let result = safe_dispatch(&handler, "panic", &[], continuation, &mut state);
        match result {
            HandlerResult::Abort(err) => {
                assert!(err.message.contains("panicked"));
                assert_eq!(
                    err.context.get("panic_recovered"),
                    Some(&"true".to_string())
                );
            }
            _ => panic!("Expected Abort for panicking handler"),
        }
    }

    #[test]
    fn test_handler_error_with_context() {
        let err = HandlerError::new("Test", "op", "test error")
            .with_source("underlying cause")
            .with_context("request_id", "123")
            .with_context("user_id", "456");

        assert_eq!(err.source, Some("underlying cause".to_string()));
        assert_eq!(err.context.get("request_id"), Some(&"123".to_string()));
        assert_eq!(err.context.get("user_id"), Some(&"456".to_string()));

        let chain = err.error_chain();
        assert!(chain.contains("test error"));
        assert!(chain.contains("underlying cause"));
    }
}
