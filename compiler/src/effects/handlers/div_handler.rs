//! Division Effect Handler
//!
//! This module implements the `Div` effect handler for safe division operations
//! through algebraic effects. The Div effect enables checked division that
//! explicitly handles division by zero.
//!
//! # Effect Operations
//!
//! | Operation      | Arguments              | Return Type | Confidence |
//! |---------------|------------------------|-------------|------------|
//! | `div`         | `(a: Float, b: Float)` | `Float`     | 1.0        |
//! | `checked_div` | `(a: Float, b: Float)` | `Option<Float>` | 1.0    |
//! | `safe_div`    | `(a: Float, b: Float, default: Float)` | `Float` | 1.0 |
//! | `int_div`     | `(a: Int, b: Int)`     | `Int`       | 1.0        |
//! | `rem`         | `(a: Int, b: Int)`     | `Int`       | 1.0        |
//!
//! # Epistemic Impact
//!
//! Division operations do not inherently degrade epistemic confidence.
//! The uncertainty comes from potential failure (division by zero), not
//! from the operation itself.
//!
//! # Example
//!
//! ```text
//! // Sounio code using Div effect:
//! fn safe_average(sum: f64, count: f64) -> f64 with Div {
//!     div(sum, count)  // Raises effect if count is 0
//! }
//!
//! fn guarded_average(sum: f64, count: f64) -> f64 with Div {
//!     safe_div(sum, count, 0.0)  // Returns 0.0 if count is 0
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use std::sync::OnceLock;

/// Static operation specifications for DivHandler
static DIV_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_div_operations() -> &'static [OperationSpec] {
    DIV_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("div", "Float")
                .with_params(vec!["Float", "Float"]),
            OperationSpec::new("checked_div", "Option")
                .with_params(vec!["Float", "Float"]),
            OperationSpec::new("safe_div", "Float")
                .with_params(vec!["Float", "Float", "Float"]),
            OperationSpec::new("int_div", "Int")
                .with_params(vec!["Int", "Int"]),
            OperationSpec::new("rem", "Int")
                .with_params(vec!["Int", "Int"]),
            OperationSpec::new("mod", "Float")
                .with_params(vec!["Float", "Float"]),
        ]
    })
}

/// Handler for the Div (division) effect
///
/// DivHandler provides safe division operations that handle division by zero
/// through algebraic effects. This allows callers to install custom handlers
/// for how to deal with division by zero, rather than panicking.
///
/// # Division Modes
///
/// - `div`: Standard division that aborts on divide-by-zero
/// - `checked_div`: Returns None on divide-by-zero
/// - `safe_div`: Returns a default value on divide-by-zero
/// - `int_div`: Integer division
/// - `rem`: Integer remainder (modulo)
#[derive(Debug)]
pub struct DivHandler {
    _private: (),
}

impl Default for DivHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl DivHandler {
    /// Create a new DivHandler
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Extract a float from a Value
    fn extract_float(value: &Value, param: &str) -> Result<f64, HandlerResult> {
        match value {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Div",
                "div",
                format!("expected Float for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Extract an integer from a Value
    fn extract_int(value: &Value, op: &str, param: &str) -> Result<i64, HandlerResult> {
        match value {
            Value::Int(i) => Ok(*i),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Div",
                op,
                format!("expected Int for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Handle div operation
    fn handle_div(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "div",
                "div requires two arguments: dividend and divisor",
            ));
        }

        let a = match Self::extract_float(&args[0], "dividend") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let b = match Self::extract_float(&args[1], "divisor") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if b == 0.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "div",
                "division by zero",
            ));
        }

        HandlerResult::Resume(Value::Float(a / b))
    }

    /// Handle checked_div operation
    fn handle_checked_div(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "checked_div",
                "checked_div requires two arguments: dividend and divisor",
            ));
        }

        let a = match Self::extract_float(&args[0], "dividend") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let b = match Self::extract_float(&args[1], "divisor") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if b == 0.0 {
            HandlerResult::Resume(Value::None)
        } else {
            HandlerResult::Resume(Value::Some(Box::new(Value::Float(a / b))))
        }
    }

    /// Handle safe_div operation
    fn handle_safe_div(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 3 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "safe_div",
                "safe_div requires three arguments: dividend, divisor, and default",
            ));
        }

        let a = match Self::extract_float(&args[0], "dividend") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let b = match Self::extract_float(&args[1], "divisor") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let default = match Self::extract_float(&args[2], "default") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if b == 0.0 {
            HandlerResult::Resume(Value::Float(default))
        } else {
            HandlerResult::Resume(Value::Float(a / b))
        }
    }

    /// Handle int_div operation
    fn handle_int_div(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "int_div",
                "int_div requires two arguments: dividend and divisor",
            ));
        }

        let a = match Self::extract_int(&args[0], "int_div", "dividend") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let b = match Self::extract_int(&args[1], "int_div", "divisor") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if b == 0 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "int_div",
                "integer division by zero",
            ));
        }

        HandlerResult::Resume(Value::Int(a / b))
    }

    /// Handle rem (remainder) operation
    fn handle_rem(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "rem",
                "rem requires two arguments: dividend and divisor",
            ));
        }

        let a = match Self::extract_int(&args[0], "rem", "dividend") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let b = match Self::extract_int(&args[1], "rem", "divisor") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if b == 0 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "rem",
                "remainder division by zero",
            ));
        }

        HandlerResult::Resume(Value::Int(a % b))
    }

    /// Handle mod (floating-point modulo) operation
    fn handle_mod(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "mod",
                "mod requires two arguments: dividend and divisor",
            ));
        }

        let a = match Self::extract_float(&args[0], "dividend") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let b = match Self::extract_float(&args[1], "divisor") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if b == 0.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Div",
                "mod",
                "modulo by zero",
            ));
        }

        HandlerResult::Resume(Value::Float(a % b))
    }
}

impl HandlerCapability for DivHandler {
    fn effect_name(&self) -> &str {
        "Div"
    }

    fn handler_name(&self) -> &str {
        "DivHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_div_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        _state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "div" => self.handle_div(args),
            "checked_div" => self.handle_checked_div(args),
            "safe_div" => self.handle_safe_div(args),
            "int_div" => self.handle_int_div(args),
            "rem" => self.handle_rem(args),
            "mod" => self.handle_mod(args),
            _ => HandlerResult::Abort(HandlerError::new(
                "Div",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        // Division is deterministic, multi-shot is safe
        true
    }

    fn epistemic_impact(&self, _operation: &str) -> EpistemicImpact {
        // Division doesn't degrade confidence
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
        let handler = DivHandler::new();
        assert_eq!(handler.effect_name(), "Div");
        assert_eq!(handler.handler_name(), "DivHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = DivHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"div"));
        assert!(names.contains(&"checked_div"));
        assert!(names.contains(&"safe_div"));
        assert!(names.contains(&"int_div"));
        assert!(names.contains(&"rem"));
        assert!(names.contains(&"mod"));
    }

    #[test]
    fn test_supports_multi_shot() {
        let handler = DivHandler::new();
        assert!(handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact() {
        let handler = DivHandler::new();
        let impact = handler.epistemic_impact("div");
        assert!((impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_div_basic() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "div",
            &[Value::Float(10.0), Value::Float(2.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!((v - 5.0).abs() < 0.001);
            }
            other => panic!("expected Resume(Float(5.0)), got {:?}", other),
        }
    }

    #[test]
    fn test_div_by_zero() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "div",
            &[Value::Float(10.0), Value::Float(0.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("division by zero"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_div_with_ints() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "div",
            &[Value::Int(10), Value::Int(4)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!((v - 2.5).abs() < 0.001);
            }
            other => panic!("expected Resume(Float(2.5)), got {:?}", other),
        }
    }

    #[test]
    fn test_checked_div_success() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "checked_div",
            &[Value::Float(10.0), Value::Float(2.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Some(boxed)) => {
                match *boxed {
                    Value::Float(v) => assert!((v - 5.0).abs() < 0.001),
                    other => panic!("expected Float, got {:?}", other),
                }
            }
            other => panic!("expected Resume(Some), got {:?}", other),
        }
    }

    #[test]
    fn test_checked_div_by_zero() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "checked_div",
            &[Value::Float(10.0), Value::Float(0.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::None) => {}
            other => panic!("expected Resume(None), got {:?}", other),
        }
    }

    #[test]
    fn test_safe_div_success() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "safe_div",
            &[Value::Float(10.0), Value::Float(2.0), Value::Float(0.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!((v - 5.0).abs() < 0.001);
            }
            other => panic!("expected Resume(Float(5.0)), got {:?}", other),
        }
    }

    #[test]
    fn test_safe_div_by_zero() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "safe_div",
            &[Value::Float(10.0), Value::Float(0.0), Value::Float(-1.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!((v - (-1.0)).abs() < 0.001);
            }
            other => panic!("expected Resume(Float(-1.0)), got {:?}", other),
        }
    }

    #[test]
    fn test_int_div_basic() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "int_div",
            &[Value::Int(10), Value::Int(3)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Int(v)) => {
                assert_eq!(v, 3);
            }
            other => panic!("expected Resume(Int(3)), got {:?}", other),
        }
    }

    #[test]
    fn test_int_div_by_zero() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "int_div",
            &[Value::Int(10), Value::Int(0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("division by zero"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_int_div_negative() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "int_div",
            &[Value::Int(-10), Value::Int(3)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Int(v)) => {
                assert_eq!(v, -3);
            }
            other => panic!("expected Resume(Int(-3)), got {:?}", other),
        }
    }

    #[test]
    fn test_rem_basic() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "rem",
            &[Value::Int(10), Value::Int(3)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Int(v)) => {
                assert_eq!(v, 1);
            }
            other => panic!("expected Resume(Int(1)), got {:?}", other),
        }
    }

    #[test]
    fn test_rem_by_zero() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "rem",
            &[Value::Int(10), Value::Int(0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("division by zero"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_mod_basic() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "mod",
            &[Value::Float(10.5), Value::Float(3.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!((v - 1.5).abs() < 0.001);
            }
            other => panic!("expected Resume(Float(1.5)), got {:?}", other),
        }
    }

    #[test]
    fn test_mod_by_zero() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "mod",
            &[Value::Float(10.0), Value::Float(0.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("modulo by zero"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = DivHandler::new();
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
        let handler = DivHandler::default();
        assert_eq!(handler.effect_name(), "Div");
    }

    #[test]
    fn test_div_missing_args() {
        let handler = DivHandler::new();
        let mut state = new_state();

        let result = handler.handle("div", &[Value::Float(10.0)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("requires"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }
}
