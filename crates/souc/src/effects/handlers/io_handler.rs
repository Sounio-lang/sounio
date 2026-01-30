//! IO Effect Handler using HandlerCapability trait
//!
//! This module provides a concrete implementation of `HandlerCapability` for
//! the IO effect. It handles all standard input/output operations with proper
//! epistemic impact tracking for read operations.
//!
//! # Epistemic Impact
//!
//! IO operations that read external data introduce uncertainty because the
//! external world is inherently less certain than computed values:
//!
//! - `print`, `println`: No confidence degradation (output only)
//! - `read_line`: 5% confidence degradation (user input uncertainty)
//! - `read_file`: 5% confidence degradation (file content uncertainty)
//! - `write_file`: No confidence degradation (output only)
//!
//! # Operations
//!
//! | Operation    | Arguments                | Return Type | Confidence Factor |
//! |-------------|--------------------------|-------------|-------------------|
//! | `print`     | `(value: any)`          | `unit`      | 1.0               |
//! | `println`   | `(value: any)`          | `unit`      | 1.0               |
//! | `read_line` | `()`                    | `string`    | 0.95              |
//! | `read_file` | `(path: string)`        | `string`    | 0.95              |
//! | `write_file`| `(path: string, content: string)` | `unit` | 1.0          |
//!
//! # Example
//!
//! ```ignore
//! use sounio::effects::handlers::IOHandler;
//! use sounio::effects::handler_capability::{HandlerCapability, Continuation, HandlerState};
//! use sounio::interp::Value;
//!
//! let handler = IOHandler::new();
//! let mut state = HandlerState::new();
//! let cont = Continuation::new();
//!
//! // Print a string
//! let result = handler.handle("println", &[Value::String("Hello!".into())], cont, &mut state);
//! ```

use std::fs;
use std::io::{self, BufRead, Write as IoWrite};
use std::sync::OnceLock;

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;

/// Confidence factor for read operations (5% degradation)
///
/// Reading from external sources introduces epistemic uncertainty because:
/// - File contents may have changed since last observation
/// - User input is inherently uncertain and may contain errors
/// - External data cannot be verified without additional context
const READ_CONFIDENCE_FACTOR: f64 = 0.95;

/// Confidence factor for write operations (no degradation)
///
/// Writing does not degrade confidence because it does not introduce
/// new information into the computation.
const WRITE_CONFIDENCE_FACTOR: f64 = 1.0;

/// Static operation specifications for IOHandler
///
/// Using OnceLock to create static operation specs that can be referenced
/// without allocation on each call to `operations()`.
static IO_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_io_operations() -> &'static [OperationSpec] {
    IO_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("print", "unit")
                .with_params(vec!["any"])
                .with_confidence_factor(WRITE_CONFIDENCE_FACTOR),
            OperationSpec::new("println", "unit")
                .with_params(vec!["any"])
                .with_confidence_factor(WRITE_CONFIDENCE_FACTOR),
            OperationSpec::new("read_line", "string")
                .with_params(vec![])
                .with_confidence_factor(READ_CONFIDENCE_FACTOR),
            OperationSpec::new("read_file", "string")
                .with_params(vec!["string"])
                .with_confidence_factor(READ_CONFIDENCE_FACTOR),
            OperationSpec::new("write_file", "unit")
                .with_params(vec!["string", "string"])
                .with_confidence_factor(WRITE_CONFIDENCE_FACTOR),
        ]
    })
}

/// IO effect handler implementing HandlerCapability
///
/// This handler provides standard input/output operations with epistemic
/// impact tracking. Read operations degrade confidence slightly to reflect
/// the uncertainty introduced by external data.
///
/// # Thread Safety
///
/// This handler is `Send + Sync` and can be shared across threads. Each
/// invocation uses thread-local or synchronized IO operations.
///
/// # Custom Output
///
/// For testing or custom scenarios, use `IOHandler::with_output` to redirect
/// output to a custom writer.
#[derive(Debug)]
pub struct IOHandler {
    /// Optional custom output writer for testing
    /// Note: For production use, stdout is used directly
    _custom_output: Option<std::sync::Arc<std::sync::Mutex<Vec<u8>>>>,
}

impl IOHandler {
    /// Create a new IO handler using standard input/output
    pub fn new() -> Self {
        Self {
            _custom_output: None,
        }
    }

    /// Create an IO handler with captured output for testing
    ///
    /// Returns both the handler and a reference to the captured output buffer.
    pub fn with_captured_output() -> (Self, std::sync::Arc<std::sync::Mutex<Vec<u8>>>) {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let handler = Self {
            _custom_output: Some(buffer.clone()),
        };
        (handler, buffer)
    }

    /// Handle the print operation
    fn handle_print(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "IO",
                "print",
                "print requires at least one argument",
            ));
        }

        let output = format_value(&args[0]);

        // Use custom output if available, otherwise use stdout
        if let Some(ref buffer) = self._custom_output {
            if let Ok(mut buf) = buffer.lock() {
                let _ = buf.write_all(output.as_bytes());
            }
        } else {
            print!("{}", output);
            let _ = io::stdout().flush();
        }

        // Record epistemic impact (print doesn't degrade confidence)
        state.record_effect("IO", "print");

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle the println operation
    fn handle_println(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        let output = if args.is_empty() {
            String::new()
        } else {
            format_value(&args[0])
        };

        // Use custom output if available, otherwise use stdout
        if let Some(ref buffer) = self._custom_output {
            if let Ok(mut buf) = buffer.lock() {
                let _ = writeln!(buf, "{}", output);
            }
        } else {
            println!("{}", output);
        }

        // Record epistemic impact (println doesn't degrade confidence)
        state.record_effect("IO", "print");

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle the read_line operation
    fn handle_read_line(&self, _args: &[Value], state: &mut HandlerState) -> HandlerResult {
        let stdin = io::stdin();
        let mut line = String::new();

        match stdin.lock().read_line(&mut line) {
            Ok(_) => {
                // Remove trailing newline
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }

                // Record epistemic impact (read operations degrade confidence)
                state.record_effect("IO", "read_line");

                HandlerResult::Resume(Value::String(line))
            }
            Err(e) => HandlerResult::Abort(HandlerError::new(
                "IO",
                "read_line",
                format!("Failed to read from stdin: {}", e),
            )),
        }
    }

    /// Handle the read_file operation
    fn handle_read_file(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "IO",
                "read_file",
                "read_file requires a path argument",
            ));
        }

        let path = match &args[0] {
            Value::String(s) => s.clone(),
            other => {
                return HandlerResult::Abort(HandlerError::new(
                    "IO",
                    "read_file",
                    format!("read_file path must be a string, got {}", other.type_name()),
                ));
            }
        };

        match fs::read_to_string(&path) {
            Ok(content) => {
                // Record epistemic impact (read operations degrade confidence)
                state.record_effect("IO", "read_file");

                HandlerResult::Resume(Value::String(content))
            }
            Err(e) => HandlerResult::Abort(HandlerError::new(
                "IO",
                "read_file",
                format!("Failed to read file '{}': {}", path, e),
            )),
        }
    }

    /// Handle the write_file operation
    fn handle_write_file(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "IO",
                "write_file",
                "write_file requires path and content arguments",
            ));
        }

        let path = match &args[0] {
            Value::String(s) => s.clone(),
            other => {
                return HandlerResult::Abort(HandlerError::new(
                    "IO",
                    "write_file",
                    format!(
                        "write_file path must be a string, got {}",
                        other.type_name()
                    ),
                ));
            }
        };

        let content = match &args[1] {
            Value::String(s) => s.clone(),
            other => {
                return HandlerResult::Abort(HandlerError::new(
                    "IO",
                    "write_file",
                    format!(
                        "write_file content must be a string, got {}",
                        other.type_name()
                    ),
                ));
            }
        };

        match fs::write(&path, &content) {
            Ok(()) => {
                // Record epistemic impact (write operations don't degrade confidence)
                state.record_effect("IO", "write_file");

                HandlerResult::Resume(Value::Unit)
            }
            Err(e) => HandlerResult::Abort(HandlerError::new(
                "IO",
                "write_file",
                format!("Failed to write file '{}': {}", path, e),
            )),
        }
    }
}

impl Default for IOHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl HandlerCapability for IOHandler {
    fn effect_name(&self) -> &str {
        "IO"
    }

    fn handler_name(&self) -> &str {
        "IOHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_io_operations()
    }

    fn handle(
        &self,
        operation: &str,
        args: &[Value],
        _continuation: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match operation {
            "print" => self.handle_print(args, state),
            "println" => self.handle_println(args, state),
            "read_line" => self.handle_read_line(args, state),
            "read_file" => self.handle_read_file(args, state),
            "write_file" => self.handle_write_file(args, state),
            _ => HandlerResult::Abort(HandlerError::new(
                "IO",
                operation,
                format!("Unknown IO operation: {}", operation),
            )),
        }
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        match operation {
            "read_line" | "read_file" => EpistemicImpact::with_confidence(READ_CONFIDENCE_FACTOR)
                .with_provenance(format!("io_{}", operation)),
            "print" | "println" | "write_file" => EpistemicImpact::none(),
            _ => EpistemicImpact::none(),
        }
    }

    fn operation_linearity(&self, operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        // IO operations are deterministic and idempotent - safe for multi-shot
        // Note: file reads may see different content if the file changes, but that's
        // an external state change, not a continuation linearity issue
        match operation {
            "print" | "println" | "read_line" | "read_file" | "write_file" => Linearity::MultiShot,
            _ => Linearity::ExactlyOnce,
        }
    }
}

/// Format a Value for printing
///
/// This handles the conversion of various Value types to their string
/// representation for output operations.
fn format_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Int(n) => n.to_string(),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => b.to_string(),
        Value::Unit => "()".to_string(),
        Value::None => "None".to_string(),
        Value::Some(v) => format!("Some({})", format_value(v)),
        Value::Ok(v) => format!("Ok({})", format_value(v)),
        Value::Err(v) => format!("Err({})", format_value(v)),
        Value::Tuple(vals) => {
            let parts: Vec<String> = vals.iter().map(format_value).collect();
            format!("({})", parts.join(", "))
        }
        Value::Array(arr) => {
            let arr = arr.borrow();
            let parts: Vec<String> = arr.iter().map(format_value).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Struct { name, fields } => {
            let parts: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("{}: {}", k, format_value(v)))
                .collect();
            format!("{} {{ {} }}", name, parts.join(", "))
        }
        Value::Variant {
            variant_name,
            fields,
            ..
        } => {
            if fields.is_empty() {
                variant_name.clone()
            } else {
                let parts: Vec<String> = fields.iter().map(format_value).collect();
                format!("{}({})", variant_name, parts.join(", "))
            }
        }
        // For complex types, use Debug formatting
        other => format!("{:?}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_io_handler_effect_name() {
        let handler = IOHandler::new();
        assert_eq!(handler.effect_name(), "IO");
        assert_eq!(handler.handler_name(), "IOHandler");
    }

    #[test]
    fn test_io_handler_operations() {
        let handler = IOHandler::new();
        let ops = handler.operations();

        assert_eq!(ops.len(), 5);

        let op_names: Vec<&str> = ops.iter().map(|op| op.name.as_str()).collect();
        assert!(op_names.contains(&"print"));
        assert!(op_names.contains(&"println"));
        assert!(op_names.contains(&"read_line"));
        assert!(op_names.contains(&"read_file"));
        assert!(op_names.contains(&"write_file"));
    }

    #[test]
    fn test_println_with_captured_output() {
        let (handler, buffer) = IOHandler::with_captured_output();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle(
            "println",
            &[Value::String("Hello, World!".to_string())],
            cont,
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            _ => panic!("Expected Resume(Unit), got {:?}", result),
        }

        let output = buffer.lock().unwrap();
        assert_eq!(String::from_utf8_lossy(&output), "Hello, World!\n");
    }

    #[test]
    fn test_print_with_captured_output() {
        let (handler, buffer) = IOHandler::with_captured_output();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle(
            "print",
            &[Value::String("test".to_string())],
            cont,
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            _ => panic!("Expected Resume(Unit), got {:?}", result),
        }

        let output = buffer.lock().unwrap();
        assert_eq!(String::from_utf8_lossy(&output), "test");
    }

    #[test]
    fn test_println_empty() {
        let (handler, buffer) = IOHandler::with_captured_output();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle("println", &[], cont, &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            _ => panic!("Expected Resume(Unit), got {:?}", result),
        }

        let output = buffer.lock().unwrap();
        assert_eq!(String::from_utf8_lossy(&output), "\n");
    }

    #[test]
    fn test_print_requires_argument() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle("print", &[], cont, &mut state);

        match result {
            HandlerResult::Abort(err) => {
                assert_eq!(err.effect, "IO");
                assert_eq!(err.operation, "print");
                assert!(err.message.contains("requires"));
            }
            _ => panic!("Expected Abort, got {:?}", result),
        }
    }

    #[test]
    fn test_read_file_success() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        // Create a temp file with content
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test content").unwrap();
        let path = temp_file.path().to_string_lossy().to_string();

        let result = handler.handle("read_file", &[Value::String(path)], cont, &mut state);

        match result {
            HandlerResult::Resume(Value::String(content)) => {
                assert!(content.contains("test content"));
            }
            _ => panic!("Expected Resume(String), got {:?}", result),
        }
    }

    #[test]
    fn test_read_file_not_found() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle(
            "read_file",
            &[Value::String("/nonexistent/path/to/file.txt".to_string())],
            cont,
            &mut state,
        );

        match result {
            HandlerResult::Abort(err) => {
                assert_eq!(err.effect, "IO");
                assert_eq!(err.operation, "read_file");
                assert!(err.message.contains("Failed to read"));
            }
            _ => panic!("Expected Abort, got {:?}", result),
        }
    }

    #[test]
    fn test_read_file_requires_path() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle("read_file", &[], cont, &mut state);

        match result {
            HandlerResult::Abort(err) => {
                assert_eq!(err.effect, "IO");
                assert_eq!(err.operation, "read_file");
                assert!(err.message.contains("requires"));
            }
            _ => panic!("Expected Abort, got {:?}", result),
        }
    }

    #[test]
    fn test_read_file_type_mismatch() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle("read_file", &[Value::Int(42)], cont, &mut state);

        match result {
            HandlerResult::Abort(err) => {
                assert_eq!(err.effect, "IO");
                assert_eq!(err.operation, "read_file");
                assert!(err.message.contains("string"));
            }
            _ => panic!("Expected Abort, got {:?}", result),
        }
    }

    #[test]
    fn test_write_file_success() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_string_lossy().to_string();

        let result = handler.handle(
            "write_file",
            &[
                Value::String(path.clone()),
                Value::String("written content".to_string()),
            ],
            cont,
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            _ => panic!("Expected Resume(Unit), got {:?}", result),
        }

        // Verify content was written
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "written content");
    }

    #[test]
    fn test_write_file_requires_two_args() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle(
            "write_file",
            &[Value::String("/tmp/test.txt".to_string())],
            cont,
            &mut state,
        );

        match result {
            HandlerResult::Abort(err) => {
                assert_eq!(err.effect, "IO");
                assert_eq!(err.operation, "write_file");
                assert!(err.message.contains("requires"));
            }
            _ => panic!("Expected Abort, got {:?}", result),
        }
    }

    #[test]
    fn test_write_file_type_mismatch_path() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle(
            "write_file",
            &[Value::Int(42), Value::String("content".to_string())],
            cont,
            &mut state,
        );

        match result {
            HandlerResult::Abort(err) => {
                assert!(err.message.contains("path"));
                assert!(err.message.contains("string"));
            }
            _ => panic!("Expected Abort, got {:?}", result),
        }
    }

    #[test]
    fn test_write_file_type_mismatch_content() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle(
            "write_file",
            &[Value::String("/tmp/test.txt".to_string()), Value::Int(42)],
            cont,
            &mut state,
        );

        match result {
            HandlerResult::Abort(err) => {
                assert!(err.message.contains("content"));
                assert!(err.message.contains("string"));
            }
            _ => panic!("Expected Abort, got {:?}", result),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle("unknown_op", &[], cont, &mut state);

        match result {
            HandlerResult::Abort(err) => {
                assert_eq!(err.effect, "IO");
                assert_eq!(err.operation, "unknown_op");
                assert!(err.message.contains("Unknown"));
            }
            _ => panic!("Expected Abort, got {:?}", result),
        }
    }

    #[test]
    fn test_epistemic_impact_read_operations() {
        let handler = IOHandler::new();

        let read_line_impact = handler.epistemic_impact("read_line");
        assert!((read_line_impact.confidence_factor - 0.95).abs() < 0.001);
        assert_eq!(
            read_line_impact.provenance_tag,
            Some("io_read_line".to_string())
        );

        let read_file_impact = handler.epistemic_impact("read_file");
        assert!((read_file_impact.confidence_factor - 0.95).abs() < 0.001);
        assert_eq!(
            read_file_impact.provenance_tag,
            Some("io_read_file".to_string())
        );
    }

    #[test]
    fn test_epistemic_impact_write_operations() {
        let handler = IOHandler::new();

        let print_impact = handler.epistemic_impact("print");
        assert!((print_impact.confidence_factor - 1.0).abs() < 0.001);

        let println_impact = handler.epistemic_impact("println");
        assert!((println_impact.confidence_factor - 1.0).abs() < 0.001);

        let write_file_impact = handler.epistemic_impact("write_file");
        assert!((write_file_impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_tracking_through_handler_state() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();

        // Initial confidence should be 1.0
        assert!((state.final_confidence(1.0) - 1.0).abs() < 0.001);
        assert!(!state.has_degradation());

        // Perform a read_file operation
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_string_lossy().to_string();
        fs::write(&path, "test").unwrap();

        let cont = Continuation::new();
        let _ = handler.handle("read_file", &[Value::String(path)], cont, &mut state);

        // After read operation, confidence should be degraded
        assert!(state.has_degradation());
        assert!((state.final_confidence(1.0) - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_multiple_read_operations_compound() {
        let handler = IOHandler::new();
        let mut state = HandlerState::new();

        // Create two temp files
        let temp_file1 = NamedTempFile::new().unwrap();
        let path1 = temp_file1.path().to_string_lossy().to_string();
        fs::write(&path1, "test1").unwrap();

        let temp_file2 = NamedTempFile::new().unwrap();
        let path2 = temp_file2.path().to_string_lossy().to_string();
        fs::write(&path2, "test2").unwrap();

        // First read
        let cont1 = Continuation::new();
        let _ = handler.handle("read_file", &[Value::String(path1)], cont1, &mut state);

        // Second read
        let cont2 = Continuation::new();
        let _ = handler.handle("read_file", &[Value::String(path2)], cont2, &mut state);

        // Confidence should be 0.95 * 0.95 = 0.9025
        assert!((state.final_confidence(1.0) - 0.9025).abs() < 0.001);
    }

    #[test]
    fn test_format_value_string() {
        assert_eq!(format_value(&Value::String("hello".to_string())), "hello");
    }

    #[test]
    fn test_format_value_int() {
        assert_eq!(format_value(&Value::Int(42)), "42");
    }

    #[test]
    fn test_format_value_float() {
        let formatted = format_value(&Value::Float(3.14));
        assert!(formatted.starts_with("3.14"));
    }

    #[test]
    fn test_format_value_bool() {
        assert_eq!(format_value(&Value::Bool(true)), "true");
        assert_eq!(format_value(&Value::Bool(false)), "false");
    }

    #[test]
    fn test_format_value_unit() {
        assert_eq!(format_value(&Value::Unit), "()");
    }

    #[test]
    fn test_format_value_none() {
        assert_eq!(format_value(&Value::None), "None");
    }

    #[test]
    fn test_format_value_some() {
        assert_eq!(
            format_value(&Value::Some(Box::new(Value::Int(42)))),
            "Some(42)"
        );
    }

    #[test]
    fn test_format_value_tuple() {
        let tuple = Value::Tuple(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert_eq!(format_value(&tuple), "(1, 2, 3)");
    }

    #[test]
    fn test_operation_specs_have_correct_confidence() {
        let ops = get_io_operations();

        for op in ops {
            match op.name.as_str() {
                "read_line" | "read_file" => {
                    assert_eq!(
                        op.confidence_factor,
                        Some(READ_CONFIDENCE_FACTOR),
                        "Operation {} should degrade confidence",
                        op.name
                    );
                }
                "print" | "println" | "write_file" => {
                    assert_eq!(
                        op.confidence_factor,
                        Some(WRITE_CONFIDENCE_FACTOR),
                        "Operation {} should not degrade confidence",
                        op.name
                    );
                }
                _ => panic!("Unexpected operation: {}", op.name),
            }
        }
    }

    #[test]
    fn test_default_impl() {
        let handler = IOHandler::default();
        assert_eq!(handler.effect_name(), "IO");
    }

    #[test]
    fn test_supports_multi_shot_is_false() {
        let handler = IOHandler::new();
        assert!(!handler.supports_multi_shot());
    }
}
