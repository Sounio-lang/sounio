//! Mutable State Effect Handler
//!
//! This module implements the `Mut` effect handler for managing mutable state
//! through algebraic effects. The Mut effect is foundational for stateful
//! computations while maintaining referential transparency at the effect level.
//!
//! # Effect Operations
//!
//! - `get(name: String) -> Value`: Retrieve a named state value
//! - `set(name: String, value: Value) -> Unit`: Store a value under a name
//! - `modify(name: String, f: Closure) -> Value`: Apply a function to state
//! - `with_state(name: String, init: Value, f: Closure) -> Value`: Run with scoped state
//!
//! # Epistemic Impact
//!
//! Mutable state operations do not degrade epistemic confidence. Pure state
//! manipulation is deterministic and does not introduce uncertainty.
//!
//! # Example
//!
//! ```text
//! // Sounio code using Mut effect:
//! fn counter() with Mut {
//!     set("count", 0)
//!     let current = get("count")
//!     set("count", current + 1)
//!     get("count")  // returns 1
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use std::cell::RefCell;
use std::rc::Rc;

/// Handler for the Mut (mutable state) effect
///
/// MutHandler provides stateful operations through algebraic effects.
/// State is stored in `HandlerState.named_state` and supports:
/// - Named state slots for multiple independent state values
/// - Scoped state with automatic cleanup
/// - Functional modification of state
///
/// # Thread Safety
///
/// MutHandler implements `Send + Sync` as required by `HandlerCapability`.
/// The actual state is stored in `HandlerState.named_state` which is passed
/// by mutable reference, ensuring exclusive access during operations.
///
/// # Scoped State
///
/// For scoped state operations (`with_state`), the handler returns a tuple
/// containing the scope information. The interpreter is responsible for
/// managing the scope stack and calling `restore_scope` when the scope exits.
/// This design avoids interior mutability in the handler itself.
#[derive(Debug)]
pub struct MutHandler {
    /// Static operation specifications
    operations: Vec<OperationSpec>,
}

impl MutHandler {
    /// Create a new MutHandler
    pub fn new() -> Self {
        let operations = vec![
            OperationSpec::new("get", "Value").with_params(vec!["String"]),
            OperationSpec::new("set", "Unit").with_params(vec!["String", "Value"]),
            OperationSpec::new("modify", "Value")
                .with_params(vec!["String", "Closure"])
                .with_continuation(),
            OperationSpec::new("with_state", "Value")
                .with_params(vec!["String", "Value", "Closure"])
                .with_continuation(),
            OperationSpec::new("delete", "Unit").with_params(vec!["String"]),
            OperationSpec::new("exists", "Bool").with_params(vec!["String"]),
            OperationSpec::new("keys", "Array").with_params(vec![]),
            OperationSpec::new("clear", "Unit").with_params(vec![]),
            OperationSpec::new("enter_scope", "Value").with_params(vec!["String", "Value"]),
            OperationSpec::new("exit_scope", "Value").with_params(vec!["String", "Value"]),
        ];

        Self { operations }
    }

    /// Get a value from state, returning None if not found
    fn handle_get(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Mut",
                "get",
                "get requires a name argument",
            ));
        }

        let name = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Mut",
                    "get",
                    format!("expected String for name, got {:?}", args[0].type_name()),
                ));
            }
        };

        let value = state.named_state.get(&name).cloned().unwrap_or(Value::None);

        HandlerResult::Resume(value)
    }

    /// Set a value in state
    fn handle_set(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Mut",
                "set",
                format!("set requires 2 arguments (name, value), got {}", args.len()),
            ));
        }

        let name = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Mut",
                    "set",
                    format!("expected String for name, got {:?}", args[0].type_name()),
                ));
            }
        };

        let value = args[1].clone();
        state.named_state.insert(name, value);

        HandlerResult::Resume(Value::Unit)
    }

    /// Modify state by applying a function
    ///
    /// For functions, returns a tuple for the interpreter to process.
    /// For direct values, applies the value immediately.
    fn handle_modify(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Mut",
                "modify",
                format!(
                    "modify requires 2 arguments (name, function), got {}",
                    args.len()
                ),
            ));
        }

        let name = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Mut",
                    "modify",
                    format!("expected String for name, got {:?}", args[0].type_name()),
                ));
            }
        };

        let modifier = &args[1];

        // Get current value (or None if not exists)
        let current = state.named_state.get(&name).cloned().unwrap_or(Value::None);

        // Check if modifier is a function or direct value
        match modifier {
            Value::Function { .. } | Value::Builtin(_) => {
                // Return tuple for interpreter to apply the function
                // Format: ("modify", state_name, current_value, function)
                HandlerResult::Resume(Value::Tuple(vec![
                    Value::String("modify".to_string()),
                    Value::String(name),
                    current,
                    modifier.clone(),
                ]))
            }
            _ => {
                // Direct value: set it immediately
                state.named_state.insert(name, modifier.clone());
                HandlerResult::Resume(modifier.clone())
            }
        }
    }

    /// Run a computation with scoped state
    ///
    /// Returns a tuple containing scope information for the interpreter.
    /// The interpreter should:
    /// 1. Save the previous value (if any)
    /// 2. Set the initial value
    /// 3. Execute the body
    /// 4. Restore the previous value
    fn handle_with_state(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 3 {
            return HandlerResult::Abort(HandlerError::new(
                "Mut",
                "with_state",
                format!(
                    "with_state requires 3 arguments (name, initial, function), got {}",
                    args.len()
                ),
            ));
        }

        let name = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Mut",
                    "with_state",
                    format!("expected String for name, got {:?}", args[0].type_name()),
                ));
            }
        };

        let initial = args[1].clone();
        let body = &args[2];

        // Save previous value for the interpreter to restore later
        let previous = state.named_state.get(&name).cloned().unwrap_or(Value::None);

        // Set initial value for the scope
        state.named_state.insert(name.clone(), initial);

        // Return tuple for interpreter to handle scope execution
        // Format: ("with_state", name, previous_value, body_function)
        HandlerResult::Resume(Value::Tuple(vec![
            Value::String("with_state".to_string()),
            Value::String(name),
            previous,
            body.clone(),
        ]))
    }

    /// Enter a scope by saving the current value and setting a new one
    ///
    /// This is a lower-level operation that can be called directly.
    /// Returns the previous value (or None) for later restoration.
    fn handle_enter_scope(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Mut",
                "enter_scope",
                format!(
                    "enter_scope requires 2 arguments (name, value), got {}",
                    args.len()
                ),
            ));
        }

        let name = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Mut",
                    "enter_scope",
                    format!("expected String for name, got {:?}", args[0].type_name()),
                ));
            }
        };

        let new_value = args[1].clone();

        // Get previous value before overwriting
        let previous = state.named_state.get(&name).cloned().unwrap_or(Value::None);

        // Set new value
        state.named_state.insert(name, new_value);

        // Return previous value for caller to restore later
        HandlerResult::Resume(previous)
    }

    /// Exit a scope by restoring a previous value
    ///
    /// If previous_value is None, removes the key from state.
    fn handle_exit_scope(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Mut",
                "exit_scope",
                format!(
                    "exit_scope requires 2 arguments (name, previous_value), got {}",
                    args.len()
                ),
            ));
        }

        let name = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Mut",
                    "exit_scope",
                    format!("expected String for name, got {:?}", args[0].type_name()),
                ));
            }
        };

        let previous = &args[1];

        match previous {
            Value::None => {
                // No previous value existed, remove the key
                state.named_state.remove(&name);
            }
            _ => {
                // Restore previous value
                state.named_state.insert(name, previous.clone());
            }
        }

        HandlerResult::Resume(Value::Unit)
    }

    /// Delete a state entry
    fn handle_delete(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Mut",
                "delete",
                "delete requires a name argument",
            ));
        }

        let name = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Mut",
                    "delete",
                    format!("expected String for name, got {:?}", args[0].type_name()),
                ));
            }
        };

        state.named_state.remove(&name);
        HandlerResult::Resume(Value::Unit)
    }

    /// Check if a state entry exists
    fn handle_exists(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Mut",
                "exists",
                "exists requires a name argument",
            ));
        }

        let name = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Mut",
                    "exists",
                    format!("expected String for name, got {:?}", args[0].type_name()),
                ));
            }
        };

        let exists = state.named_state.contains_key(&name);
        HandlerResult::Resume(Value::Bool(exists))
    }

    /// Get all state keys
    fn handle_keys(&self, state: &HandlerState) -> HandlerResult {
        let keys: Vec<Value> = state
            .named_state
            .keys()
            .map(|k| Value::String(k.clone()))
            .collect();

        HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(keys))))
    }

    /// Clear all state
    fn handle_clear(&self, state: &mut HandlerState) -> HandlerResult {
        state.named_state.clear();
        HandlerResult::Resume(Value::Unit)
    }
}

impl Default for MutHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl HandlerCapability for MutHandler {
    fn effect_name(&self) -> &str {
        "Mut"
    }

    fn handler_name(&self) -> &str {
        "MutHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        &self.operations
    }

    fn handle(
        &self,
        operation: &str,
        args: &[Value],
        _continuation: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match operation {
            "get" => self.handle_get(args, state),
            "set" => self.handle_set(args, state),
            "modify" => self.handle_modify(args, state),
            "with_state" => self.handle_with_state(args, state),
            "enter_scope" => self.handle_enter_scope(args, state),
            "exit_scope" => self.handle_exit_scope(args, state),
            "delete" => self.handle_delete(args, state),
            "exists" => self.handle_exists(args, state),
            "keys" => self.handle_keys(state),
            "clear" => self.handle_clear(state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Mut",
                operation,
                format!("unknown operation: {}", operation),
            )),
        }
    }

    fn epistemic_impact(&self, _operation: &str) -> EpistemicImpact {
        // Mutable state operations do not degrade confidence
        // State is deterministic and introduces no uncertainty
        EpistemicImpact::none()
    }

    fn supports_multi_shot(&self) -> bool {
        // State operations don't support multi-shot by default
        // Backtracking would require explicit snapshot/restore
        false
    }

    fn operation_linearity(&self, operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        // Mutation operations modify state - must be one-shot to prevent
        // unintended replays of state modifications
        match operation {
            "set" | "modify" | "delete" | "clear" | "exit_scope" => Linearity::ExactlyOnce,
            // Read-only operations are safe for multi-shot
            "get" | "exists" | "keys" => Linearity::MultiShot,
            // Scope entry creates state - one-shot
            "enter_scope" | "with_state" => Linearity::ExactlyOnce,
            _ => Linearity::ExactlyOnce,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state() -> HandlerState {
        HandlerState::new()
    }

    fn create_continuation() -> Continuation {
        Continuation::new()
    }

    #[test]
    fn test_handler_identity() {
        let handler = MutHandler::new();
        assert_eq!(handler.effect_name(), "Mut");
        assert_eq!(handler.handler_name(), "MutHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = MutHandler::new();
        let ops = handler.operations();

        // Check we have the expected operations
        let op_names: Vec<&str> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(op_names.contains(&"get"));
        assert!(op_names.contains(&"set"));
        assert!(op_names.contains(&"modify"));
        assert!(op_names.contains(&"with_state"));
        assert!(op_names.contains(&"delete"));
        assert!(op_names.contains(&"exists"));
        assert!(op_names.contains(&"keys"));
        assert!(op_names.contains(&"clear"));
        assert!(op_names.contains(&"enter_scope"));
        assert!(op_names.contains(&"exit_scope"));
    }

    #[test]
    fn test_set_and_get() {
        let handler = MutHandler::new();
        let mut state = create_test_state();
        let cont = create_continuation();

        // Set a value
        let set_result = handler.handle(
            "set",
            &[Value::String("counter".to_string()), Value::Int(42)],
            cont,
            &mut state,
        );
        assert!(matches!(set_result, HandlerResult::Resume(Value::Unit)));

        // Get the value back
        let cont = create_continuation();
        let get_result = handler.handle(
            "get",
            &[Value::String("counter".to_string())],
            cont,
            &mut state,
        );
        assert!(matches!(get_result, HandlerResult::Resume(Value::Int(42))));
    }

    #[test]
    fn test_get_missing_returns_none() {
        let handler = MutHandler::new();
        let mut state = create_test_state();
        let cont = create_continuation();

        let result = handler.handle(
            "get",
            &[Value::String("nonexistent".to_string())],
            cont,
            &mut state,
        );
        assert!(matches!(result, HandlerResult::Resume(Value::None)));
    }

    #[test]
    fn test_get_requires_argument() {
        let handler = MutHandler::new();
        let mut state = create_test_state();
        let cont = create_continuation();

        let result = handler.handle("get", &[], cont, &mut state);
        assert!(matches!(result, HandlerResult::Abort(_)));
    }

    #[test]
    fn test_set_requires_two_arguments() {
        let handler = MutHandler::new();
        let mut state = create_test_state();
        let cont = create_continuation();

        let result = handler.handle("set", &[Value::String("x".to_string())], cont, &mut state);
        assert!(matches!(result, HandlerResult::Abort(_)));
    }

    #[test]
    fn test_delete() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // First set a value
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("key".to_string()), Value::Int(100)],
            cont,
            &mut state,
        );

        // Verify it exists
        let cont = create_continuation();
        let exists_result = handler.handle(
            "exists",
            &[Value::String("key".to_string())],
            cont,
            &mut state,
        );
        assert!(matches!(
            exists_result,
            HandlerResult::Resume(Value::Bool(true))
        ));

        // Delete it
        let cont = create_continuation();
        let delete_result = handler.handle(
            "delete",
            &[Value::String("key".to_string())],
            cont,
            &mut state,
        );
        assert!(matches!(delete_result, HandlerResult::Resume(Value::Unit)));

        // Verify it's gone
        let cont = create_continuation();
        let exists_result = handler.handle(
            "exists",
            &[Value::String("key".to_string())],
            cont,
            &mut state,
        );
        assert!(matches!(
            exists_result,
            HandlerResult::Resume(Value::Bool(false))
        ));
    }

    #[test]
    fn test_exists() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Check non-existent key
        let cont = create_continuation();
        let result = handler.handle(
            "exists",
            &[Value::String("missing".to_string())],
            cont,
            &mut state,
        );
        assert!(matches!(result, HandlerResult::Resume(Value::Bool(false))));

        // Add key
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("present".to_string()), Value::Int(1)],
            cont,
            &mut state,
        );

        // Check existing key
        let cont = create_continuation();
        let result = handler.handle(
            "exists",
            &[Value::String("present".to_string())],
            cont,
            &mut state,
        );
        assert!(matches!(result, HandlerResult::Resume(Value::Bool(true))));
    }

    #[test]
    fn test_keys() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Add some keys
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("a".to_string()), Value::Int(1)],
            cont,
            &mut state,
        );
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("b".to_string()), Value::Int(2)],
            cont,
            &mut state,
        );

        // Get keys
        let cont = create_continuation();
        let result = handler.handle("keys", &[], cont, &mut state);

        if let HandlerResult::Resume(Value::Array(arr)) = result {
            let keys: Vec<String> = arr
                .borrow()
                .iter()
                .filter_map(|v| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect();
            assert!(keys.contains(&"a".to_string()));
            assert!(keys.contains(&"b".to_string()));
            assert_eq!(keys.len(), 2);
        } else {
            panic!("Expected Array result from keys operation");
        }
    }

    #[test]
    fn test_clear() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Add some values
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("x".to_string()), Value::Int(1)],
            cont,
            &mut state,
        );
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("y".to_string()), Value::Int(2)],
            cont,
            &mut state,
        );

        assert!(!state.named_state.is_empty());

        // Clear
        let cont = create_continuation();
        let result = handler.handle("clear", &[], cont, &mut state);
        assert!(matches!(result, HandlerResult::Resume(Value::Unit)));
        assert!(state.named_state.is_empty());
    }

    #[test]
    fn test_modify_with_direct_value() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Set initial value
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("n".to_string()), Value::Int(10)],
            cont,
            &mut state,
        );

        // Modify with a direct value (not a function)
        let cont = create_continuation();
        let result = handler.handle(
            "modify",
            &[Value::String("n".to_string()), Value::Int(20)],
            cont,
            &mut state,
        );

        // When given a direct value, it should set that value
        assert!(matches!(result, HandlerResult::Resume(Value::Int(20))));

        // Verify state was updated
        let cont = create_continuation();
        let get_result = handler.handle("get", &[Value::String("n".to_string())], cont, &mut state);
        assert!(matches!(get_result, HandlerResult::Resume(Value::Int(20))));
    }

    #[test]
    fn test_epistemic_impact_is_none() {
        let handler = MutHandler::new();

        // All Mut operations should have no epistemic impact
        let get_impact = handler.epistemic_impact("get");
        assert!((get_impact.confidence_factor - 1.0).abs() < 0.001);
        assert!(get_impact.provenance_tag.is_none());
        assert!(!get_impact.crosses_firewall);

        let set_impact = handler.epistemic_impact("set");
        assert!((set_impact.confidence_factor - 1.0).abs() < 0.001);

        let modify_impact = handler.epistemic_impact("modify");
        assert!((modify_impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_unknown_operation() {
        let handler = MutHandler::new();
        let mut state = create_test_state();
        let cont = create_continuation();

        let result = handler.handle("unknown_op", &[], cont, &mut state);
        assert!(matches!(result, HandlerResult::Abort(_)));
    }

    #[test]
    fn test_type_error_on_non_string_name() {
        let handler = MutHandler::new();
        let mut state = create_test_state();
        let cont = create_continuation();

        // Try to get with an integer instead of string
        let result = handler.handle("get", &[Value::Int(123)], cont, &mut state);
        assert!(matches!(result, HandlerResult::Abort(_)));
    }

    #[test]
    fn test_multiple_state_values() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Set multiple values
        let values = vec![
            ("counter", Value::Int(0)),
            ("name", Value::String("test".to_string())),
            ("flag", Value::Bool(true)),
            ("data", Value::Float(3.14)),
        ];

        for (key, value) in &values {
            let cont = create_continuation();
            handler.handle(
                "set",
                &[Value::String(key.to_string()), value.clone()],
                cont,
                &mut state,
            );
        }

        // Verify all values
        for (key, expected) in &values {
            let cont = create_continuation();
            let result = handler.handle("get", &[Value::String(key.to_string())], cont, &mut state);
            if let HandlerResult::Resume(got) = result {
                assert_eq!(&got, expected);
            } else {
                panic!("Expected Resume result for key {}", key);
            }
        }
    }

    #[test]
    fn test_overwrite_value() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Set initial value
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("x".to_string()), Value::Int(1)],
            cont,
            &mut state,
        );

        // Overwrite
        let cont = create_continuation();
        handler.handle(
            "set",
            &[Value::String("x".to_string()), Value::Int(2)],
            cont,
            &mut state,
        );

        // Verify overwritten value
        let cont = create_continuation();
        let result = handler.handle("get", &[Value::String("x".to_string())], cont, &mut state);
        assert!(matches!(result, HandlerResult::Resume(Value::Int(2))));
    }

    #[test]
    fn test_enter_and_exit_scope() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Set an initial value
        state.named_state.insert("x".to_string(), Value::Int(10));

        // Enter scope with new value
        let cont = create_continuation();
        let enter_result = handler.handle(
            "enter_scope",
            &[Value::String("x".to_string()), Value::Int(99)],
            cont,
            &mut state,
        );

        // Should return the previous value
        assert!(matches!(
            enter_result,
            HandlerResult::Resume(Value::Int(10))
        ));

        // Verify new value is set
        assert_eq!(state.named_state.get("x"), Some(&Value::Int(99)));

        // Exit scope, restoring previous value
        let cont = create_continuation();
        let exit_result = handler.handle(
            "exit_scope",
            &[Value::String("x".to_string()), Value::Int(10)],
            cont,
            &mut state,
        );
        assert!(matches!(exit_result, HandlerResult::Resume(Value::Unit)));

        // Verify original value is restored
        assert_eq!(state.named_state.get("x"), Some(&Value::Int(10)));
    }

    #[test]
    fn test_exit_scope_with_none_removes_key() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Set a value
        state.named_state.insert("temp".to_string(), Value::Int(1));

        // Exit scope with None as previous value (simulating key didn't exist before)
        let cont = create_continuation();
        handler.handle(
            "exit_scope",
            &[Value::String("temp".to_string()), Value::None],
            cont,
            &mut state,
        );

        // Key should be removed
        assert!(!state.named_state.contains_key("temp"));
    }

    #[test]
    fn test_nested_scopes_manual() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Set initial value
        state.named_state.insert("x".to_string(), Value::Int(1));

        // Enter first scope
        let cont = create_continuation();
        let prev1 = handler.handle(
            "enter_scope",
            &[Value::String("x".to_string()), Value::Int(2)],
            cont,
            &mut state,
        );
        let prev1_val = if let HandlerResult::Resume(v) = prev1 {
            v
        } else {
            panic!("Expected Resume");
        };

        // Enter second scope
        let cont = create_continuation();
        let prev2 = handler.handle(
            "enter_scope",
            &[Value::String("x".to_string()), Value::Int(3)],
            cont,
            &mut state,
        );
        let prev2_val = if let HandlerResult::Resume(v) = prev2 {
            v
        } else {
            panic!("Expected Resume");
        };

        // Verify innermost value
        assert_eq!(state.named_state.get("x"), Some(&Value::Int(3)));

        // Exit second scope
        let cont = create_continuation();
        handler.handle(
            "exit_scope",
            &[Value::String("x".to_string()), prev2_val],
            cont,
            &mut state,
        );
        assert_eq!(state.named_state.get("x"), Some(&Value::Int(2)));

        // Exit first scope
        let cont = create_continuation();
        handler.handle(
            "exit_scope",
            &[Value::String("x".to_string()), prev1_val],
            cont,
            &mut state,
        );
        assert_eq!(state.named_state.get("x"), Some(&Value::Int(1)));
    }

    #[test]
    fn test_does_not_support_multi_shot() {
        let handler = MutHandler::new();
        assert!(!handler.supports_multi_shot());
    }

    #[test]
    fn test_with_state_returns_tuple() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Set an initial value
        state
            .named_state
            .insert("counter".to_string(), Value::Int(0));

        // Call with_state
        let cont = create_continuation();
        let result = handler.handle(
            "with_state",
            &[
                Value::String("counter".to_string()),
                Value::Int(100),
                Value::Builtin("identity".to_string()),
            ],
            cont,
            &mut state,
        );

        // Should return a tuple with scope information
        if let HandlerResult::Resume(Value::Tuple(tuple)) = result {
            assert_eq!(tuple.len(), 4);
            assert!(matches!(&tuple[0], Value::String(s) if s == "with_state"));
            assert!(matches!(&tuple[1], Value::String(s) if s == "counter"));
            assert!(matches!(&tuple[2], Value::Int(0))); // Previous value
            assert!(matches!(&tuple[3], Value::Builtin(_))); // Body function
        } else {
            panic!("Expected Tuple result from with_state");
        }

        // State should be updated to initial value
        assert_eq!(state.named_state.get("counter"), Some(&Value::Int(100)));
    }

    #[test]
    fn test_modify_with_function_returns_tuple() {
        let handler = MutHandler::new();
        let mut state = create_test_state();

        // Set initial value
        state.named_state.insert("n".to_string(), Value::Int(5));

        // Modify with a builtin function
        let cont = create_continuation();
        let result = handler.handle(
            "modify",
            &[
                Value::String("n".to_string()),
                Value::Builtin("increment".to_string()),
            ],
            cont,
            &mut state,
        );

        // Should return a tuple for the interpreter to process
        if let HandlerResult::Resume(Value::Tuple(tuple)) = result {
            assert_eq!(tuple.len(), 4);
            assert!(matches!(&tuple[0], Value::String(s) if s == "modify"));
            assert!(matches!(&tuple[1], Value::String(s) if s == "n"));
            assert!(matches!(&tuple[2], Value::Int(5))); // Current value
            assert!(matches!(&tuple[3], Value::Builtin(_))); // Function
        } else {
            panic!("Expected Tuple result from modify with function");
        }
    }
}
