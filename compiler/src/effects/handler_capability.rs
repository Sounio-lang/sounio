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
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};

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
#[derive(Debug, Clone)]
pub struct HandlerError {
    pub message: String,
    pub effect: String,
    pub operation: String,
}

impl HandlerError {
    pub fn new(
        effect: impl Into<String>,
        operation: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            effect: effect.into(),
            operation: operation.into(),
            message: message.into(),
        }
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
pub trait HandlerCapability: Debug + Send + Sync {
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
}
