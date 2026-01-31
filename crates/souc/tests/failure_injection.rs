//! Phase D.5: Production Testing - Failure Injection Tests
//!
//! Tests for panic recovery, error propagation, and context preservation.

use sounio::effects::handler_capability::{
    safe_dispatch, Continuation, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use sounio::interp::Value;
use std::panic::RefUnwindSafe;

/// Handler that panics on specific operations
#[derive(Debug)]
struct PanickingHandler {
    panic_on: String,
}

impl RefUnwindSafe for PanickingHandler {}

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
        if operation == self.panic_on {
            panic!("intentional panic on operation: {}", operation);
        }
        HandlerResult::Resume(Value::Unit)
    }
}

/// Test that handler panic is converted to Abort
#[test]
fn test_panic_converted_to_abort() {
    let handler = PanickingHandler {
        panic_on: "boom".to_string(),
    };
    let mut state = HandlerState::new();

    // Normal operation should work
    let result = safe_dispatch(&handler, "normal", &[], Continuation::new(), &mut state);
    assert!(matches!(result, HandlerResult::Resume(Value::Unit)));

    // Panicking operation should be caught
    let result = safe_dispatch(&handler, "boom", &[], Continuation::new(), &mut state);

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("panicked"));
        }
        other => panic!("Expected Abort, got {:?}", other),
    }
}

/// Test that panic error includes handler context
#[test]
fn test_panic_context_preserved() {
    let handler = PanickingHandler {
        panic_on: "explode".to_string(),
    };
    let mut state = HandlerState::new();

    let result = safe_dispatch(&handler, "explode", &[], Continuation::new(), &mut state);

    match result {
        HandlerResult::Abort(err) => {
            // Should include handler name
            assert_eq!(
                err.context.get("handler"),
                Some(&"PanickingHandler".to_string())
            );
            // Should indicate panic was recovered
            assert_eq!(
                err.context.get("panic_recovered"),
                Some(&"true".to_string())
            );
        }
        other => panic!("Expected Abort, got {:?}", other),
    }
}

/// Test error chain formatting
#[test]
fn test_error_chain_formatting() {
    let err = HandlerError::new("Network", "connect", "Connection refused")
        .with_source("TCP handshake failed: timeout after 30s");

    let chain = err.error_chain();

    assert!(chain.contains("Connection refused"));
    assert!(chain.contains("caused by:"));
    assert!(chain.contains("TCP handshake failed"));
}

/// Test HandlerError includes timestamp
#[test]
fn test_error_timestamp_present() {
    let before = std::time::Instant::now();
    let err = HandlerError::new("IO", "read", "File not found");
    let after = std::time::Instant::now();

    assert!(err.timestamp.is_some());
    let timestamp = err.timestamp.unwrap();

    // Timestamp should be between before and after
    assert!(timestamp >= before);
    assert!(timestamp <= after);
}

/// Test handler state is not corrupted after abort
#[test]
fn test_abort_does_not_corrupt_state() {
    let handler = PanickingHandler {
        panic_on: "panic".to_string(),
    };
    let mut state = HandlerState::new();

    // Set some state
    state.named_state.insert("key".to_string(), Value::Int(42));

    // Trigger panic
    let _ = safe_dispatch(&handler, "panic", &[], Continuation::new(), &mut state);

    // State should still be accessible
    assert_eq!(state.named_state.get("key"), Some(&Value::Int(42)));

    // Should be able to continue using the state
    state
        .named_state
        .insert("key2".to_string(), Value::String("test".to_string()));
    assert!(state.named_state.contains_key("key2"));
}

/// Test nested handler panic isolation
#[test]
fn test_nested_handler_panic_isolation() {
    /// Outer handler that delegates to inner
    #[derive(Debug)]
    struct OuterHandler {
        inner_panic_on: String,
    }

    impl RefUnwindSafe for OuterHandler {}

    impl HandlerCapability for OuterHandler {
        fn effect_name(&self) -> &str {
            "Outer"
        }
        fn handler_name(&self) -> &str {
            "OuterHandler"
        }
        fn operations(&self) -> &[OperationSpec] {
            &[]
        }
        fn handle(
            &self,
            operation: &str,
            _args: &[Value],
            _continuation: Continuation,
            state: &mut HandlerState,
        ) -> HandlerResult {
            if operation == "delegate" {
                let inner = PanickingHandler {
                    panic_on: self.inner_panic_on.clone(),
                };
                // Delegate to inner handler with safe_dispatch
                return safe_dispatch(
                    &inner,
                    &self.inner_panic_on,
                    &[],
                    Continuation::new(),
                    state,
                );
            }
            HandlerResult::Resume(Value::Int(1))
        }
    }

    let outer = OuterHandler {
        inner_panic_on: "inner_boom".to_string(),
    };
    let mut state = HandlerState::new();

    // Inner panic should be caught and not propagate to outer
    let result = safe_dispatch(&outer, "delegate", &[], Continuation::new(), &mut state);

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("panicked"));
            // Should be from inner handler
            assert_eq!(
                err.context.get("handler"),
                Some(&"PanickingHandler".to_string())
            );
        }
        other => panic!("Expected Abort from inner panic, got {:?}", other),
    }

    // Outer handler should still work for other operations
    let result = outer.handle("other", &[], Continuation::new(), &mut state);
    assert!(matches!(result, HandlerResult::Resume(Value::Int(1))));
}

/// Test error context accumulation
#[test]
fn test_error_context_accumulation() {
    let err = HandlerError::new("Network", "http_get", "Request failed")
        .with_context("request_id", "abc-123")
        .with_context("user_id", "user-456")
        .with_context("retry_count", "3")
        .with_source("DNS resolution failed");

    assert_eq!(err.context.get("request_id"), Some(&"abc-123".to_string()));
    assert_eq!(err.context.get("user_id"), Some(&"user-456".to_string()));
    assert_eq!(err.context.get("retry_count"), Some(&"3".to_string()));
    assert_eq!(err.source, Some("DNS resolution failed".to_string()));

    let chain = err.error_chain();
    assert!(chain.contains("Request failed"));
    assert!(chain.contains("DNS resolution failed"));
}

/// Test HandlerError::from_error
#[test]
fn test_handler_error_from_std_error() {
    use std::io;

    let io_err = io::Error::new(io::ErrorKind::NotFound, "file.txt not found");
    let handler_err = HandlerError::from_error("IO", "read_file", io_err);

    assert_eq!(handler_err.effect, "IO");
    assert_eq!(handler_err.operation, "read_file");
    assert!(handler_err.message.contains("not found"));
    assert!(handler_err.source.is_some());
}

/// Test multiple consecutive panics are all caught
#[test]
fn test_multiple_panics_caught() {
    let handler = PanickingHandler {
        panic_on: "crash".to_string(),
    };
    let mut state = HandlerState::new();

    for i in 0..10 {
        let result = safe_dispatch(&handler, "crash", &[], Continuation::new(), &mut state);

        match result {
            HandlerResult::Abort(err) => {
                assert!(err.message.contains("panicked"));
            }
            other => panic!("Iteration {}: Expected Abort, got {:?}", i, other),
        }
    }
}

/// Test panic with various message types
#[test]
fn test_panic_message_extraction() {
    // String panic
    #[derive(Debug)]
    struct StringPanicHandler;
    impl RefUnwindSafe for StringPanicHandler {}
    impl HandlerCapability for StringPanicHandler {
        fn effect_name(&self) -> &str {
            "Test"
        }
        fn handler_name(&self) -> &str {
            "StringPanicHandler"
        }
        fn operations(&self) -> &[OperationSpec] {
            &[]
        }
        fn handle(
            &self,
            _: &str,
            _: &[Value],
            _: Continuation,
            _: &mut HandlerState,
        ) -> HandlerResult {
            panic!("string panic message");
        }
    }

    let mut state = HandlerState::new();
    let result = safe_dispatch(
        &StringPanicHandler,
        "op",
        &[],
        Continuation::new(),
        &mut state,
    );

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("string panic message"));
        }
        other => panic!("Expected Abort, got {:?}", other),
    }

    // Static str panic
    #[derive(Debug)]
    struct StaticStrPanicHandler;
    impl RefUnwindSafe for StaticStrPanicHandler {}
    impl HandlerCapability for StaticStrPanicHandler {
        fn effect_name(&self) -> &str {
            "Test"
        }
        fn handler_name(&self) -> &str {
            "StaticStrPanicHandler"
        }
        fn operations(&self) -> &[OperationSpec] {
            &[]
        }
        fn handle(
            &self,
            _: &str,
            _: &[Value],
            _: Continuation,
            _: &mut HandlerState,
        ) -> HandlerResult {
            panic!("static str panic");
        }
    }

    let result = safe_dispatch(
        &StaticStrPanicHandler,
        "op",
        &[],
        Continuation::new(),
        &mut state,
    );

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("static str panic"));
        }
        other => panic!("Expected Abort, got {:?}", other),
    }
}

/// Test safe_dispatch preserves handler result for non-panic cases
#[test]
fn test_safe_dispatch_preserves_results() {
    #[derive(Debug)]
    struct ResultHandler;
    impl RefUnwindSafe for ResultHandler {}
    impl HandlerCapability for ResultHandler {
        fn effect_name(&self) -> &str {
            "Test"
        }
        fn handler_name(&self) -> &str {
            "ResultHandler"
        }
        fn operations(&self) -> &[OperationSpec] {
            &[]
        }
        fn handle(
            &self,
            op: &str,
            _: &[Value],
            _: Continuation,
            _: &mut HandlerState,
        ) -> HandlerResult {
            match op {
                "return" => HandlerResult::Return(Value::Int(42)),
                "resume" => HandlerResult::Resume(Value::String("hello".to_string())),
                "abort" => HandlerResult::Abort(HandlerError::new("Test", "abort", "intentional")),
                _ => HandlerResult::Resume(Value::Unit),
            }
        }
    }

    let handler = ResultHandler;
    let mut state = HandlerState::new();

    // Return is preserved
    match safe_dispatch(&handler, "return", &[], Continuation::new(), &mut state) {
        HandlerResult::Return(Value::Int(42)) => {}
        other => panic!("Expected Return(Int(42)), got {:?}", other),
    }

    // Resume is preserved
    match safe_dispatch(&handler, "resume", &[], Continuation::new(), &mut state) {
        HandlerResult::Resume(Value::String(s)) if s == "hello" => {}
        other => panic!("Expected Resume(String('hello')), got {:?}", other),
    }

    // Abort is preserved
    match safe_dispatch(&handler, "abort", &[], Continuation::new(), &mut state) {
        HandlerResult::Abort(err) => {
            assert_eq!(err.message, "intentional");
        }
        other => panic!("Expected Abort, got {:?}", other),
    }
}
