//! Exception Effect Handler
//!
//! This module implements the `Exn` (Exception) effect handler for typed exception
//! handling through algebraic effects. The Exn effect enables structured error
//! handling with type-safe exceptions that can carry payload data.
//!
//! # Effect Operations
//!
//! | Operation    | Arguments                      | Return Type | Confidence |
//! |-------------|--------------------------------|-------------|------------|
//! | `throw`     | `(message: String)`            | `Never`     | 1.0        |
//! | `throw_with`| `(tag: String, payload: Value)`| `Never`     | 1.0        |
//! | `try`       | `(expr: Fn)`                   | `Result`    | 1.0        |
//! | `catch`     | `(tag: String, handler: Fn)`   | `Unit`      | 1.0        |
//! | `finally`   | `(cleanup: Fn)`                | `Unit`      | 1.0        |
//! | `rethrow`   | `()`                           | `Never`     | 1.0        |
//! | `get_exception` | `()`                       | `Option<Exception>` | 1.0 |
//! | `is_exception`  | `(tag: String)`            | `Bool`      | 1.0        |
//!
//! # Difference from Panic Effect
//!
//! While both Exn and Panic handle error conditions, they serve different purposes:
//!
//! - **Exn (Exceptions)**: For recoverable, expected error conditions that are
//!   part of normal program flow. Examples: file not found, invalid input,
//!   network timeout. Exceptions are typed and can carry structured payloads.
//!
//! - **Panic**: For unrecoverable errors, assertion failures, and programming
//!   bugs. Panics indicate something went seriously wrong.
//!
//! # Epistemic Impact
//!
//! Exception operations do not degrade epistemic confidence because they
//! represent structured control flow, not uncertainty in data. The exception
//! either happened or it didn't - there's no epistemic uncertainty.
//!
//! # Exception Tags
//!
//! Exceptions are identified by string tags, allowing typed exception handling:
//! - `"IOError"`: I/O operation failures
//! - `"ValueError"`: Invalid value or argument
//! - `"TypeError"`: Type-related errors
//! - `"NetworkError"`: Network operation failures
//! - `"TimeoutError"`: Operation timed out
//! - `"NotFoundError"`: Resource not found
//! - Custom tags for domain-specific exceptions
//!
//! # Example
//!
//! ```text
//! // Sounio code using Exn effect:
//! fn parse_number(s: string) -> int with Exn {
//!     if !is_numeric(s) {
//!         throw_with("ValueError", "not a valid number")
//!     }
//!     parse_int(s)
//! }
//!
//! fn safe_parse(s: string) -> Option<int> with Exn {
//!     try {
//!         Some(parse_number(s))
//!     } catch "ValueError" {
//!         None
//!     }
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::effects::linearity::Linearity;
use crate::interp::Value;
use std::sync::OnceLock;

/// Key for storing current exception
const CURRENT_EXCEPTION_KEY: &str = "__exn_current";

/// Key for storing exception tag
const EXCEPTION_TAG_KEY: &str = "__exn_tag";

/// Key for storing exception payload
const EXCEPTION_PAYLOAD_KEY: &str = "__exn_payload";

/// Key for storing exception stack
const EXCEPTION_STACK_KEY: &str = "__exn_stack";

/// Key for storing registered handlers
const EXCEPTION_HANDLERS_KEY: &str = "__exn_handlers";

/// Key for storing exception count
const EXCEPTION_COUNT_KEY: &str = "__exn_count";

/// Static operation specifications for ExnHandler
static EXN_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_exn_operations() -> &'static [OperationSpec] {
    EXN_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("throw", "Never").with_params(vec!["String"]),
            OperationSpec::new("throw_with", "Never").with_params(vec!["String", "Value"]),
            OperationSpec::new("try_catch", "Result").with_params(vec!["Value", "String", "Value"]),
            OperationSpec::new("get_exception", "Option").with_params(vec![]),
            OperationSpec::new("get_exception_tag", "Option").with_params(vec![]),
            OperationSpec::new("get_exception_payload", "Option").with_params(vec![]),
            OperationSpec::new("is_exception", "Bool").with_params(vec!["String"]),
            OperationSpec::new("clear_exception", "Unit").with_params(vec![]),
            OperationSpec::new("rethrow", "Never").with_params(vec![]),
            OperationSpec::new("exception_count", "Int").with_params(vec![]),
        ]
    })
}

/// Exception data structure
#[derive(Debug, Clone)]
pub struct Exception {
    /// Exception tag (type identifier)
    pub tag: String,
    /// Human-readable message
    pub message: String,
    /// Optional payload data
    pub payload: Option<Value>,
}

impl Exception {
    /// Create a new exception with just a message
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            tag: "Exception".to_string(),
            message: message.into(),
            payload: None,
        }
    }

    /// Create a new exception with a tag and message
    pub fn with_tag(tag: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            message: message.into(),
            payload: None,
        }
    }

    /// Create a new exception with tag, message, and payload
    pub fn with_payload(
        tag: impl Into<String>,
        message: impl Into<String>,
        payload: Value,
    ) -> Self {
        Self {
            tag: tag.into(),
            message: message.into(),
            payload: Some(payload),
        }
    }
}

/// Handler for the Exn (Exception) effect
///
/// ExnHandler provides typed exception handling through algebraic effects.
/// It enables structured error handling with exception tags and payloads.
///
/// # State Management
///
/// The handler tracks:
/// - Current exception (if any)
/// - Exception tag and payload
/// - Exception count (for diagnostics)
///
/// # Exception Flow
///
/// When an exception is thrown:
/// 1. Exception data is stored in handler state
/// 2. Control returns via HandlerResult::Abort
/// 3. Catch handlers can intercept specific exception tags
/// 4. Uncaught exceptions propagate up the handler stack
#[derive(Debug)]
pub struct ExnHandler {
    _private: (),
}

impl Default for ExnHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ExnHandler {
    /// Create a new ExnHandler
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Check if an exception is currently active
    fn has_exception(state: &HandlerState) -> bool {
        state.named_state.contains_key(CURRENT_EXCEPTION_KEY)
    }

    /// Get current exception tag
    fn get_exception_tag_value(state: &HandlerState) -> Option<String> {
        match state.named_state.get(EXCEPTION_TAG_KEY) {
            Some(Value::String(tag)) => Some(tag.clone()),
            _ => None,
        }
    }

    /// Get current exception message
    fn get_exception_message(state: &HandlerState) -> Option<String> {
        match state.named_state.get(CURRENT_EXCEPTION_KEY) {
            Some(Value::String(msg)) => Some(msg.clone()),
            _ => None,
        }
    }

    /// Get current exception payload
    fn get_exception_payload_value(state: &HandlerState) -> Option<Value> {
        state.named_state.get(EXCEPTION_PAYLOAD_KEY).cloned()
    }

    /// Set current exception
    fn set_exception(state: &mut HandlerState, tag: &str, message: &str, payload: Option<Value>) {
        state.named_state.insert(
            CURRENT_EXCEPTION_KEY.to_string(),
            Value::String(message.to_string()),
        );
        state.named_state.insert(
            EXCEPTION_TAG_KEY.to_string(),
            Value::String(tag.to_string()),
        );
        if let Some(p) = payload {
            state
                .named_state
                .insert(EXCEPTION_PAYLOAD_KEY.to_string(), p);
        }
        Self::increment_exception_count(state);
    }

    /// Clear current exception
    fn clear_exception_state(state: &mut HandlerState) {
        state.named_state.remove(CURRENT_EXCEPTION_KEY);
        state.named_state.remove(EXCEPTION_TAG_KEY);
        state.named_state.remove(EXCEPTION_PAYLOAD_KEY);
    }

    /// Increment exception count
    fn increment_exception_count(state: &mut HandlerState) {
        let count = match state.named_state.get(EXCEPTION_COUNT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        state
            .named_state
            .insert(EXCEPTION_COUNT_KEY.to_string(), Value::Int(count + 1));
    }

    /// Get exception count
    fn get_exception_count(state: &HandlerState) -> i64 {
        match state.named_state.get(EXCEPTION_COUNT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        }
    }

    /// Extract a string from a Value
    fn extract_string(value: &Value, op: &str, param: &str) -> Result<String, HandlerResult> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Exn",
                op,
                format!("expected String for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Handle throw operation
    fn handle_throw(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Exn",
                "throw",
                "throw requires a message argument",
            ));
        }

        let message = match Self::extract_string(&args[0], "throw", "message") {
            Ok(m) => m,
            Err(result) => return result,
        };

        Self::set_exception(state, "Exception", &message, None);

        HandlerResult::Abort(HandlerError::new(
            "Exn",
            "throw",
            format!("Exception: {}", message),
        ))
    }

    /// Handle throw_with operation
    fn handle_throw_with(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Exn",
                "throw_with",
                "throw_with requires tag and payload arguments",
            ));
        }

        let tag = match Self::extract_string(&args[0], "throw_with", "tag") {
            Ok(t) => t,
            Err(result) => return result,
        };

        let payload = args[1].clone();
        let message = match &payload {
            Value::String(s) => s.clone(),
            _ => format!("{:?}", payload),
        };

        Self::set_exception(state, &tag, &message, Some(payload));

        HandlerResult::Abort(HandlerError::new(
            "Exn",
            "throw_with",
            format!("{}: {}", tag, message),
        ))
    }

    /// Handle try_catch operation
    /// This is a simplified try-catch that takes a value and checks for exceptions
    fn handle_try_catch(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 3 {
            return HandlerResult::Abort(HandlerError::new(
                "Exn",
                "try_catch",
                "try_catch requires value, exception_tag, and default_value arguments",
            ));
        }

        // If there's a current exception matching the tag, return default
        let tag_to_catch = match Self::extract_string(&args[1], "try_catch", "exception_tag") {
            Ok(t) => t,
            Err(result) => return result,
        };

        if Self::has_exception(state) {
            if let Some(current_tag) = Self::get_exception_tag_value(state) {
                if current_tag == tag_to_catch || tag_to_catch == "*" {
                    Self::clear_exception_state(state);
                    return HandlerResult::Resume(args[2].clone());
                }
            }
        }

        // No exception or tag doesn't match - return the value
        HandlerResult::Resume(args[0].clone())
    }

    /// Handle get_exception operation
    fn handle_get_exception(&self, state: &HandlerState) -> HandlerResult {
        if Self::has_exception(state) {
            let tag = Self::get_exception_tag_value(state).unwrap_or_default();
            let message = Self::get_exception_message(state).unwrap_or_default();
            let payload = Self::get_exception_payload_value(state);

            // Return as tuple (tag, message, payload)
            let result = Value::Tuple(vec![
                Value::String(tag),
                Value::String(message),
                payload.unwrap_or(Value::Unit),
            ]);
            HandlerResult::Resume(Value::Some(Box::new(result)))
        } else {
            HandlerResult::Resume(Value::None)
        }
    }

    /// Handle get_exception_tag operation
    fn handle_get_exception_tag(&self, state: &HandlerState) -> HandlerResult {
        match Self::get_exception_tag_value(state) {
            Some(tag) => HandlerResult::Resume(Value::Some(Box::new(Value::String(tag)))),
            None => HandlerResult::Resume(Value::None),
        }
    }

    /// Handle get_exception_payload operation
    fn handle_get_exception_payload(&self, state: &HandlerState) -> HandlerResult {
        match Self::get_exception_payload_value(state) {
            Some(payload) => HandlerResult::Resume(Value::Some(Box::new(payload))),
            None => HandlerResult::Resume(Value::None),
        }
    }

    /// Handle is_exception operation
    fn handle_is_exception(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Exn",
                "is_exception",
                "is_exception requires a tag argument",
            ));
        }

        let tag = match Self::extract_string(&args[0], "is_exception", "tag") {
            Ok(t) => t,
            Err(result) => return result,
        };

        let is_match = if Self::has_exception(state) {
            if let Some(current_tag) = Self::get_exception_tag_value(state) {
                current_tag == tag || tag == "*"
            } else {
                false
            }
        } else {
            false
        };

        HandlerResult::Resume(Value::Bool(is_match))
    }

    /// Handle clear_exception operation
    fn handle_clear_exception(&self, state: &mut HandlerState) -> HandlerResult {
        Self::clear_exception_state(state);
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle rethrow operation
    fn handle_rethrow(&self, state: &HandlerState) -> HandlerResult {
        if Self::has_exception(state) {
            let tag =
                Self::get_exception_tag_value(state).unwrap_or_else(|| "Exception".to_string());
            let message =
                Self::get_exception_message(state).unwrap_or_else(|| "unknown error".to_string());

            HandlerResult::Abort(HandlerError::new(
                "Exn",
                "rethrow",
                format!("{}: {}", tag, message),
            ))
        } else {
            HandlerResult::Abort(HandlerError::new(
                "Exn",
                "rethrow",
                "no exception to rethrow",
            ))
        }
    }

    /// Handle exception_count operation
    fn handle_exception_count(&self, state: &HandlerState) -> HandlerResult {
        let count = Self::get_exception_count(state);
        HandlerResult::Resume(Value::Int(count))
    }

    /// Check if an exception is active (public accessor for testing)
    pub fn has_active_exception(state: &HandlerState) -> bool {
        Self::has_exception(state)
    }

    /// Get exception count (public accessor for testing)
    pub fn total_exception_count(state: &HandlerState) -> i64 {
        Self::get_exception_count(state)
    }

    /// Get current exception tag (public accessor for testing)
    pub fn current_exception_tag(state: &HandlerState) -> Option<String> {
        Self::get_exception_tag_value(state)
    }
}

impl HandlerCapability for ExnHandler {
    fn effect_name(&self) -> &str {
        "Exn"
    }

    fn handler_name(&self) -> &str {
        "ExnHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_exn_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "throw" => self.handle_throw(args, state),
            "throw_with" => self.handle_throw_with(args, state),
            "try_catch" => self.handle_try_catch(args, state),
            "get_exception" => self.handle_get_exception(state),
            "get_exception_tag" => self.handle_get_exception_tag(state),
            "get_exception_payload" => self.handle_get_exception_payload(state),
            "is_exception" => self.handle_is_exception(args, state),
            "clear_exception" => self.handle_clear_exception(state),
            "rethrow" => self.handle_rethrow(state),
            "exception_count" => self.handle_exception_count(state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Exn",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        // Exception handling is deterministic
        true
    }

    fn operation_linearity(&self, operation: &str) -> Linearity {
        // Exception handling is deterministic - can retry error handling
        match operation {
            "throw" | "try" | "catch" | "finally" | "raise" => Linearity::MultiShot,
            // Recovery operations might have side effects
            _ => Linearity::ExactlyOnce,
        }
    }

    fn epistemic_impact(&self, _operation: &str) -> EpistemicImpact {
        // Exceptions don't affect epistemic confidence
        EpistemicImpact::none()
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
        let handler = ExnHandler::new();
        assert_eq!(handler.effect_name(), "Exn");
        assert_eq!(handler.handler_name(), "ExnHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = ExnHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"throw"));
        assert!(names.contains(&"throw_with"));
        assert!(names.contains(&"try_catch"));
        assert!(names.contains(&"get_exception"));
        assert!(names.contains(&"is_exception"));
        assert!(names.contains(&"clear_exception"));
        assert!(names.contains(&"rethrow"));
    }

    #[test]
    fn test_supports_multi_shot() {
        let handler = ExnHandler::new();
        assert!(handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact() {
        let handler = ExnHandler::new();
        let impact = handler.epistemic_impact("throw");
        assert!((impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_throw() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "throw",
            &[Value::String("something went wrong".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("something went wrong"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }

        assert!(ExnHandler::has_active_exception(&state));
        assert_eq!(ExnHandler::total_exception_count(&state), 1);
    }

    #[test]
    fn test_throw_with() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "throw_with",
            &[
                Value::String("ValueError".to_string()),
                Value::String("invalid input".to_string()),
            ],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("ValueError"));
                assert!(e.message.contains("invalid input"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }

        assert_eq!(
            ExnHandler::current_exception_tag(&state),
            Some("ValueError".to_string())
        );
    }

    #[test]
    fn test_throw_with_payload() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw_with",
            &[Value::String("DataError".to_string()), Value::Int(42)],
            cont(),
            &mut state,
        );

        // Check payload
        let result = handler.handle("get_exception_payload", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Some(boxed)) => match *boxed {
                Value::Int(42) => {}
                other => panic!("expected Int(42), got {:?}", other),
            },
            other => panic!("expected Resume(Some), got {:?}", other),
        }
    }

    #[test]
    fn test_get_exception_none() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        let result = handler.handle("get_exception", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::None) => {}
            other => panic!("expected Resume(None), got {:?}", other),
        }
    }

    #[test]
    fn test_get_exception_some() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw",
            &[Value::String("test error".to_string())],
            cont(),
            &mut state,
        );

        let result = handler.handle("get_exception", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Some(boxed)) => match *boxed {
                Value::Tuple(elems) => {
                    assert_eq!(elems.len(), 3);
                }
                other => panic!("expected Tuple, got {:?}", other),
            },
            other => panic!("expected Resume(Some), got {:?}", other),
        }
    }

    #[test]
    fn test_get_exception_tag() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw_with",
            &[
                Value::String("CustomError".to_string()),
                Value::String("details".to_string()),
            ],
            cont(),
            &mut state,
        );

        let result = handler.handle("get_exception_tag", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Some(boxed)) => match *boxed {
                Value::String(tag) => assert_eq!(tag, "CustomError"),
                other => panic!("expected String, got {:?}", other),
            },
            other => panic!("expected Resume(Some), got {:?}", other),
        }
    }

    #[test]
    fn test_is_exception_match() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw_with",
            &[
                Value::String("IOError".to_string()),
                Value::String("file not found".to_string()),
            ],
            cont(),
            &mut state,
        );

        let result = handler.handle(
            "is_exception",
            &[Value::String("IOError".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Bool(true)) => {}
            other => panic!("expected Resume(Bool(true)), got {:?}", other),
        }
    }

    #[test]
    fn test_is_exception_no_match() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw_with",
            &[
                Value::String("IOError".to_string()),
                Value::String("file not found".to_string()),
            ],
            cont(),
            &mut state,
        );

        let result = handler.handle(
            "is_exception",
            &[Value::String("ValueError".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Bool(false)) => {}
            other => panic!("expected Resume(Bool(false)), got {:?}", other),
        }
    }

    #[test]
    fn test_is_exception_wildcard() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw_with",
            &[
                Value::String("AnyError".to_string()),
                Value::String("details".to_string()),
            ],
            cont(),
            &mut state,
        );

        let result = handler.handle(
            "is_exception",
            &[Value::String("*".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Bool(true)) => {}
            other => panic!("expected Resume(Bool(true)), got {:?}", other),
        }
    }

    #[test]
    fn test_clear_exception() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw",
            &[Value::String("error".to_string())],
            cont(),
            &mut state,
        );

        assert!(ExnHandler::has_active_exception(&state));

        handler.handle("clear_exception", &[], cont(), &mut state);

        assert!(!ExnHandler::has_active_exception(&state));
    }

    #[test]
    fn test_rethrow() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw_with",
            &[
                Value::String("OriginalError".to_string()),
                Value::String("original message".to_string()),
            ],
            cont(),
            &mut state,
        );

        let result = handler.handle("rethrow", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("OriginalError"));
                assert!(e.message.contains("original message"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_rethrow_no_exception() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        let result = handler.handle("rethrow", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("no exception to rethrow"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_try_catch_no_exception() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "try_catch",
            &[
                Value::Int(42),
                Value::String("Exception".to_string()),
                Value::Int(0),
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
    fn test_try_catch_with_exception() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        // Throw an exception first
        handler.handle(
            "throw_with",
            &[
                Value::String("TestError".to_string()),
                Value::String("test".to_string()),
            ],
            cont(),
            &mut state,
        );

        // Try to catch it
        let result = handler.handle(
            "try_catch",
            &[
                Value::Int(42),
                Value::String("TestError".to_string()),
                Value::Int(-1),
            ],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Int(-1)) => {}
            other => panic!("expected Resume(Int(-1)), got {:?}", other),
        }

        // Exception should be cleared
        assert!(!ExnHandler::has_active_exception(&state));
    }

    #[test]
    fn test_exception_count() {
        let handler = ExnHandler::new();
        let mut state = new_state();

        handler.handle(
            "throw",
            &[Value::String("error 1".to_string())],
            cont(),
            &mut state,
        );
        handler.handle("clear_exception", &[], cont(), &mut state);

        handler.handle(
            "throw",
            &[Value::String("error 2".to_string())],
            cont(),
            &mut state,
        );

        let result = handler.handle("exception_count", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(2)) => {}
            other => panic!("expected Resume(Int(2)), got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = ExnHandler::new();
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
        let handler = ExnHandler::default();
        assert_eq!(handler.effect_name(), "Exn");
    }

    #[test]
    fn test_exception_struct() {
        let exn = Exception::new("test error");
        assert_eq!(exn.tag, "Exception");
        assert_eq!(exn.message, "test error");
        assert!(exn.payload.is_none());

        let exn = Exception::with_tag("IOError", "file not found");
        assert_eq!(exn.tag, "IOError");

        let exn = Exception::with_payload("DataError", "invalid", Value::Int(42));
        assert!(exn.payload.is_some());
    }
}
