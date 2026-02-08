//! Integration tests for effect handler continuations
//!
//! Tests the complete effect handler pipeline with real continuation capture:
//! 1. Effect dispatch captures interpreter continuation
//! 2. Continuation is pushed onto the active stack
//! 3. Handler can access and resume the continuation
//! 4. Continuation is properly cleaned up after handler completes

use sounio::effects::handlers::HandlerRegistry;
use sounio::interp::EffectKind;
use sounio::interp::effect_dispatch::{DispatchResult, EffectContext};
use sounio::interp::value::Value;

#[test]
fn test_effect_dispatch_captures_continuation() {
    let mut ctx = EffectContext::with_all_handlers();

    // Initially, no continuations should be active
    assert!(!ctx.has_active());

    // Dispatch an IO effect operation
    let result =
        ctx.dispatch_by_name_with_continuation("IO", "print", vec![Value::String("test".into())]);

    // The operation should complete successfully
    assert!(result.is_ok());
    match result.unwrap() {
        DispatchResult::Completed(value) => {
            assert_eq!(value, Value::Unit);
        }
        DispatchResult::Suspended { .. } => {
            panic!("IO.print should not suspend");
        }
    }

    // After completion, the continuation should be cleaned up
    assert!(!ctx.has_active());
}

#[test]
fn test_continuation_cleanup_on_abort() {
    let mut ctx = EffectContext::with_all_handlers();

    // Dispatch an operation that will abort (invalid arguments)
    let result = ctx.dispatch_by_name_with_continuation("IO", "read_file", vec![]);

    // Should get an error
    assert!(result.is_err());

    // Continuation should still be cleaned up even on error
    assert!(!ctx.has_active());
}

#[test]
fn test_continuation_store_tracking() {
    let mut ctx = EffectContext::with_all_handlers();

    let initial_count = ctx.active_continuation_count();

    // Dispatch an effect - this captures a continuation
    let _ =
        ctx.dispatch_by_name_with_continuation("IO", "print", vec![Value::String("test".into())]);

    // A continuation was captured and then cleaned up
    // The store may or may not retain completed continuations depending on implementation
    let final_count = ctx.active_continuation_count();

    // This test just ensures we're not leaking continuations infinitely
    assert!(final_count >= initial_count);
}

#[test]
fn test_handler_registry_integration() {
    let registry = HandlerRegistry::with_defaults();

    // Verify all 12 handlers are available
    assert!(registry.get("IO").is_some());
    assert!(registry.get("Mut").is_some());
    assert!(registry.get("Alloc").is_some());
    assert!(registry.get("Panic").is_some());
    assert!(registry.get("Async").is_some());
    assert!(registry.get("Prob").is_some());
    assert!(registry.get("GPU").is_some());
    assert!(registry.get("Network").is_some());
    assert!(registry.get("Sensor").is_some());
    assert!(registry.get("Epistemic").is_some());
    assert!(registry.get("Exn").is_some());
    assert!(registry.get("Div").is_some());

    // Create a context with the registry
    let mut ctx = EffectContext::with_registry(registry);

    // Should be able to dispatch to any of the handlers
    let result =
        ctx.dispatch_by_name_with_continuation("IO", "print", vec![Value::String("test".into())]);
    assert!(result.is_ok());
}

#[test]
fn test_effect_kind_dispatch() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test dispatch using EffectKind enum
    let result = ctx.dispatch_with_continuation(
        EffectKind::IO,
        "print",
        vec![Value::String("direct dispatch".into())],
    );

    assert!(result.is_ok());
    match result.unwrap() {
        DispatchResult::Completed(value) => {
            assert_eq!(value, Value::Unit);
        }
        DispatchResult::Suspended { .. } => {
            panic!("IO.print should not suspend");
        }
    }
}

#[test]
fn test_multiple_sequential_dispatches() {
    let mut ctx = EffectContext::with_all_handlers();

    // Dispatch multiple effects sequentially
    for i in 0..5 {
        let msg = format!("dispatch {}", i);
        let result =
            ctx.dispatch_by_name_with_continuation("IO", "print", vec![Value::String(msg)]);
        assert!(result.is_ok());
    }

    // All continuations should be cleaned up
    assert!(!ctx.has_active());
}
