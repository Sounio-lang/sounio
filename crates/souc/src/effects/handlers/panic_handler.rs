//! Panic Effect Handler
//!
//! This module implements the `Panic` effect handler for recoverable failure
//! through algebraic effects. The Panic effect enables structured error handling
//! with explicit effect tracking.
//!
//! # Effect Operations
//!
//! | Operation     | Arguments              | Return Type | Description                    |
//! |--------------|------------------------|-------------|--------------------------------|
//! | `panic`      | `(message: String)`    | `Never`     | Raise a panic with message     |
//! | `throw`      | `(message: String)`    | `Never`     | Alias for panic                |
//! | `catch`      | `()`                   | `Value`     | Catch point for handler        |
//! | `unreachable`| `()`                   | `Never`     | Mark code as unreachable       |
//! | `assert`     | `(condition: Bool, message: String)` | `Unit` | Assert condition or panic |
//! | `unwrap`     | `(option: Option)`     | `Value`     | Unwrap or panic if None        |
//! | `expect`     | `(option: Option, message: String)` | `Value` | Unwrap or panic with message |
//!
//! # Epistemic Impact
//!
//! Panic operations do not degrade epistemic confidence. Panics are exceptional
//! control flow that either succeeds (no panic) or fails completely (panic).
//! There is no uncertainty about whether a panic occurred.
//!
//! # Handler Semantics
//!
//! Unlike other effects, Panic has special semantics:
//! - `panic`/`throw` abort the current computation and return to the handler
//! - Handlers can choose to recover, re-throw, or transform the panic
//! - Unhandled panics propagate up the handler stack
//!
//! # Example
//!
//! ```text
//! // Sounio code using Panic effect:
//! fn safe_divide(a: i64, b: i64) -> i64 with Panic {
//!     if b == 0 {
//!         panic("division by zero")
//!     }
//!     a / b
//! }
//!
//! fn main() {
//!     handle Panic {
//!         let result = safe_divide(10, 0)
//!         println(result)
//!     } with {
//!         catch(msg) => {
//!             println("caught: " ++ msg)
//!             0
//!         }
//!     }
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use std::sync::OnceLock;

/// Key for storing the current panic message in HandlerState
const PANIC_MESSAGE_KEY: &str = "__panic_message";

/// Key for storing whether we're in a panic state
const PANIC_ACTIVE_KEY: &str = "__panic_active";

/// Static operation specifications for PanicHandler
static PANIC_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_panic_operations() -> &'static [OperationSpec] {
    PANIC_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("panic", "Never").with_params(vec!["String"]),
            OperationSpec::new("throw", "Never").with_params(vec!["String"]),
            OperationSpec::new("catch", "Value").with_params(vec![]),
            OperationSpec::new("unreachable", "Never").with_params(vec![]),
            OperationSpec::new("assert", "Unit").with_params(vec!["Bool", "String"]),
            OperationSpec::new("unwrap", "Value").with_params(vec!["Option"]),
            OperationSpec::new("expect", "Value").with_params(vec!["Option", "String"]),
        ]
    })
}

/// Handler for the Panic (recoverable failure) effect
///
/// PanicHandler provides structured error handling through algebraic effects.
/// Panics can be caught and handled by effect handlers, enabling recovery
/// from exceptional conditions.
///
/// # Panic State
///
/// When a panic occurs, the handler:
/// 1. Stores the panic message in HandlerState
/// 2. Returns `HandlerResult::Return` to abort the current computation
/// 3. The effect handler can then decide how to proceed
///
/// # Recovery
///
/// Handlers can recover from panics by:
/// - Catching the panic and returning a default value
/// - Logging and re-throwing
/// - Transforming the panic into a different error type
#[derive(Debug)]
pub struct PanicHandler {
    _private: (),
}

impl Default for PanicHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl PanicHandler {
    /// Create a new PanicHandler
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Handle panic operation - raise a panic with a message
    fn handle_panic(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        match self.extract_message(args, "panic") {
            Ok(message) => self.raise_panic(message, state),
            Err(result) => result,
        }
    }

    /// Handle throw operation - alias for panic
    fn handle_throw(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        match self.extract_message(args, "throw") {
            Ok(message) => self.raise_panic(message, state),
            Err(result) => result,
        }
    }

    /// Handle unreachable operation
    fn handle_unreachable(&self, state: &mut HandlerState) -> HandlerResult {
        self.raise_panic("entered unreachable code".to_string(), state)
    }

    /// Handle assert operation
    fn handle_assert(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Panic",
                "assert",
                "assert requires a condition argument",
            ));
        }

        let condition = match &args[0] {
            Value::Bool(b) => *b,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Panic",
                    "assert",
                    format!("expected Bool for condition, got {:?}", args[0].type_name()),
                ));
            }
        };

        if condition {
            HandlerResult::Resume(Value::Unit)
        } else {
            let message = if args.len() > 1 {
                match &args[1] {
                    Value::String(s) => s.clone(),
                    _ => "assertion failed".to_string(),
                }
            } else {
                "assertion failed".to_string()
            };
            self.raise_panic(message, state)
        }
    }

    /// Handle unwrap operation
    fn handle_unwrap(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Panic",
                "unwrap",
                "unwrap requires an option argument",
            ));
        }

        match &args[0] {
            Value::Some(inner) => HandlerResult::Resume((**inner).clone()),
            Value::None => self.raise_panic("called unwrap on None".to_string(), state),
            other => {
                // For non-Option types, just return the value
                // This allows unwrap to work as identity for non-optional values
                HandlerResult::Resume(other.clone())
            }
        }
    }

    /// Handle expect operation
    fn handle_expect(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Panic",
                "expect",
                "expect requires an option argument",
            ));
        }

        let message = if args.len() > 1 {
            match &args[1] {
                Value::String(s) => s.clone(),
                _ => "expect failed".to_string(),
            }
        } else {
            "expect failed".to_string()
        };

        match &args[0] {
            Value::Some(inner) => HandlerResult::Resume((**inner).clone()),
            Value::None => self.raise_panic(message, state),
            other => {
                // For non-Option types, just return the value
                HandlerResult::Resume(other.clone())
            }
        }
    }

    /// Handle catch operation - returns the current panic message if in panic state
    fn handle_catch(&self, state: &mut HandlerState) -> HandlerResult {
        let is_active = matches!(
            state.named_state.get(PANIC_ACTIVE_KEY),
            Some(Value::Bool(true))
        );

        if is_active {
            let message = state
                .named_state
                .get(PANIC_MESSAGE_KEY)
                .cloned()
                .unwrap_or(Value::String("unknown panic".to_string()));

            // Clear panic state
            state.named_state.remove(PANIC_ACTIVE_KEY);
            state.named_state.remove(PANIC_MESSAGE_KEY);

            HandlerResult::Resume(message)
        } else {
            // No active panic - return None
            HandlerResult::Resume(Value::None)
        }
    }

    /// Extract message from arguments
    fn extract_message(&self, args: &[Value], op: &str) -> Result<String, HandlerResult> {
        if args.is_empty() {
            return Err(HandlerResult::Abort(HandlerError::new(
                "Panic",
                op,
                format!("{} requires a message argument", op),
            )));
        }

        match &args[0] {
            Value::String(s) => Ok(s.clone()),
            other => {
                // Convert other types to string representation
                Ok(format!("{:?}", other))
            }
        }
    }

    /// Raise a panic with the given message
    fn raise_panic(&self, message: String, state: &mut HandlerState) -> HandlerResult {
        // Store panic state
        state
            .named_state
            .insert(PANIC_ACTIVE_KEY.to_string(), Value::Bool(true));
        state.named_state.insert(
            PANIC_MESSAGE_KEY.to_string(),
            Value::String(message.clone()),
        );

        // Return the panic - this aborts the current computation
        // The handler will catch this and can decide to recover or propagate
        HandlerResult::Return(Value::String(message))
    }

    /// Check if a panic is currently active
    pub fn is_panic_active(state: &HandlerState) -> bool {
        matches!(
            state.named_state.get(PANIC_ACTIVE_KEY),
            Some(Value::Bool(true))
        )
    }

    /// Get the current panic message if active
    pub fn get_panic_message(state: &HandlerState) -> Option<String> {
        if Self::is_panic_active(state) {
            state.named_state.get(PANIC_MESSAGE_KEY).and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
        } else {
            None
        }
    }

    /// Clear the panic state (used after catching)
    pub fn clear_panic(state: &mut HandlerState) {
        state.named_state.remove(PANIC_ACTIVE_KEY);
        state.named_state.remove(PANIC_MESSAGE_KEY);
    }
}

impl HandlerCapability for PanicHandler {
    fn effect_name(&self) -> &str {
        "Panic"
    }

    fn handler_name(&self) -> &str {
        "PanicHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_panic_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "panic" => self.handle_panic(args, state),
            "throw" => self.handle_throw(args, state),
            "catch" => self.handle_catch(state),
            "unreachable" => self.handle_unreachable(state),
            "assert" => self.handle_assert(args, state),
            "unwrap" => self.handle_unwrap(args, state),
            "expect" => self.handle_expect(args, state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Panic",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        false
    }

    fn epistemic_impact(&self, _operation: &str) -> EpistemicImpact {
        // Panic operations do not degrade confidence
        // Either the operation succeeds or it panics - no uncertainty
        EpistemicImpact::none()
    }

    fn operation_linearity(&self, operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        // Panic operations abort execution - must be exactly once
        // (though in practice they never resume)
        match operation {
            "panic" | "throw" | "unreachable" | "assert" | "unwrap" | "expect" => {
                Linearity::ExactlyOnce
            }
            // Catch is a read operation - safe for multi-shot
            "catch" => Linearity::MultiShot,
            _ => Linearity::ExactlyOnce,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_state() -> HandlerState {
        HandlerState::new()
    }

    fn cont() -> Continuation {
        Continuation::new()
    }

    #[test]
    fn test_handler_identity() {
        let handler = PanicHandler::new();
        assert_eq!(handler.effect_name(), "Panic");
        assert_eq!(handler.handler_name(), "PanicHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = PanicHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"panic"));
        assert!(names.contains(&"throw"));
        assert!(names.contains(&"catch"));
        assert!(names.contains(&"unreachable"));
        assert!(names.contains(&"assert"));
        assert!(names.contains(&"unwrap"));
        assert!(names.contains(&"expect"));
    }

    #[test]
    fn test_does_not_support_multi_shot() {
        let handler = PanicHandler::new();
        assert!(!handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact_is_identity() {
        let handler = PanicHandler::new();

        let panic_impact = handler.epistemic_impact("panic");
        assert!((panic_impact.confidence_factor - 1.0).abs() < 0.001);
        assert!(!panic_impact.crosses_firewall);

        let throw_impact = handler.epistemic_impact("throw");
        assert!((throw_impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_panic_basic() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "panic",
            &[Value::String("test error".into())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Return(Value::String(msg)) => {
                assert_eq!(msg, "test error");
            }
            other => panic!("expected Return(String), got {:?}", other),
        }

        // Check panic state is set
        assert!(PanicHandler::is_panic_active(&state));
        assert_eq!(
            PanicHandler::get_panic_message(&state),
            Some("test error".to_string())
        );
    }

    #[test]
    fn test_panic_requires_message() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle("panic", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("requires a message"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_throw_is_alias_for_panic() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "throw",
            &[Value::String("thrown error".into())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Return(Value::String(msg)) => {
                assert_eq!(msg, "thrown error");
            }
            other => panic!("expected Return(String), got {:?}", other),
        }

        assert!(PanicHandler::is_panic_active(&state));
    }

    #[test]
    fn test_unreachable() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle("unreachable", &[], cont(), &mut state);

        match result {
            HandlerResult::Return(Value::String(msg)) => {
                assert!(msg.contains("unreachable"));
            }
            other => panic!("expected Return(String), got {:?}", other),
        }
    }

    #[test]
    fn test_assert_success() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle("assert", &[Value::Bool(true)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        assert!(!PanicHandler::is_panic_active(&state));
    }

    #[test]
    fn test_assert_failure() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle("assert", &[Value::Bool(false)], cont(), &mut state);

        match result {
            HandlerResult::Return(Value::String(msg)) => {
                assert!(msg.contains("assertion failed"));
            }
            other => panic!("expected Return(String), got {:?}", other),
        }
    }

    #[test]
    fn test_assert_with_message() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "assert",
            &[Value::Bool(false), Value::String("custom message".into())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Return(Value::String(msg)) => {
                assert_eq!(msg, "custom message");
            }
            other => panic!("expected Return(String), got {:?}", other),
        }
    }

    #[test]
    fn test_assert_requires_condition() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle("assert", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("requires a condition"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_unwrap_some() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "unwrap",
            &[Value::Some(Box::new(Value::Int(42)))],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Int(42)) => {}
            other => panic!("expected Resume(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_unwrap_none() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle("unwrap", &[Value::None], cont(), &mut state);

        match result {
            HandlerResult::Return(Value::String(msg)) => {
                assert!(msg.contains("unwrap on None"));
            }
            other => panic!("expected Return(String), got {:?}", other),
        }
    }

    #[test]
    fn test_unwrap_non_option() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        // unwrap on non-Option returns the value as-is
        let result = handler.handle("unwrap", &[Value::Int(42)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(42)) => {}
            other => panic!("expected Resume(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_expect_some() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "expect",
            &[
                Value::Some(Box::new(Value::Int(42))),
                Value::String("should have value".into()),
            ],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Int(42)) => {}
            other => panic!("expected Resume(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_expect_none() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "expect",
            &[Value::None, Value::String("expected a value".into())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Return(Value::String(msg)) => {
                assert_eq!(msg, "expected a value");
            }
            other => panic!("expected Return(String), got {:?}", other),
        }
    }

    #[test]
    fn test_catch_with_active_panic() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        // First, trigger a panic
        handler.handle(
            "panic",
            &[Value::String("caught this".into())],
            cont(),
            &mut state,
        );

        assert!(PanicHandler::is_panic_active(&state));

        // Now catch it
        let result = handler.handle("catch", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::String(msg)) => {
                assert_eq!(msg, "caught this");
            }
            other => panic!("expected Resume(String), got {:?}", other),
        }

        // Panic should be cleared
        assert!(!PanicHandler::is_panic_active(&state));
    }

    #[test]
    fn test_catch_without_panic() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle("catch", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::None) => {}
            other => panic!("expected Resume(None), got {:?}", other),
        }
    }

    #[test]
    fn test_clear_panic() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        // Trigger a panic
        handler.handle(
            "panic",
            &[Value::String("error".into())],
            cont(),
            &mut state,
        );

        assert!(PanicHandler::is_panic_active(&state));

        // Clear it manually
        PanicHandler::clear_panic(&mut state);

        assert!(!PanicHandler::is_panic_active(&state));
        assert!(PanicHandler::get_panic_message(&state).is_none());
    }

    #[test]
    fn test_unknown_operation() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        let result = handler.handle("unknown_op", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("unknown operation"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_default_impl() {
        let handler = PanicHandler::default();
        assert_eq!(handler.effect_name(), "Panic");
    }

    #[test]
    fn test_panic_with_non_string_converts() {
        let handler = PanicHandler::new();
        let mut state = new_state();

        // Non-string values should be converted to string representation
        let result = handler.handle("panic", &[Value::Int(42)], cont(), &mut state);

        match result {
            HandlerResult::Return(Value::String(msg)) => {
                assert!(msg.contains("42"));
            }
            other => panic!("expected Return(String), got {:?}", other),
        }
    }
}
