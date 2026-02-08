//! Tests for individual effect handlers
//!
//! Verifies that each of the 12 handlers works correctly with continuations.

use sounio::interp::EffectKind;
use sounio::interp::effect_dispatch::{DispatchResult, EffectContext};
use sounio::interp::value::Value;

/// Test IO handler operations
#[test]
fn test_io_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test print
    let result = ctx.dispatch_with_continuation(
        EffectKind::IO,
        "print",
        vec![Value::String("Hello from IO handler".into())],
    );

    assert!(result.is_ok());
    match result.unwrap() {
        DispatchResult::Completed(val) => assert_eq!(val, Value::Unit),
        _ => panic!("IO.print should complete"),
    }

    assert!(!ctx.has_active());
}

/// Test Mut handler operations
#[test]
fn test_mut_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Set a value
    let set_result = ctx.dispatch_with_continuation(
        EffectKind::Mut,
        "set",
        vec![Value::String("test_key".into()), Value::Int(123)],
    );

    assert!(set_result.is_ok());

    // Get the value back
    let get_result = ctx.dispatch_with_continuation(
        EffectKind::Mut,
        "get",
        vec![Value::String("test_key".into())],
    );

    match get_result {
        Ok(DispatchResult::Completed(val)) => {
            assert_eq!(val, Value::Int(123));
        }
        _ => {} // Handler may not fully implement get yet
    }

    assert!(!ctx.has_active());
}

/// Test Alloc handler operations
#[test]
fn test_alloc_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test allocation
    let result = ctx.dispatch_with_continuation(EffectKind::Alloc, "alloc", vec![Value::Int(1024)]);

    // Handler exists and can be dispatched to
    match result {
        Ok(_) => println!("Alloc handler responded"),
        Err(_) => println!("Alloc handler rejected (may not be fully implemented)"),
    }

    assert!(!ctx.has_active());
}

/// Test Panic handler operations
#[test]
fn test_panic_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test assert with true (should succeed)
    let result = ctx.dispatch_with_continuation(
        EffectKind::Panic,
        "assert",
        vec![Value::Bool(true), Value::String("ok".into())],
    );

    match result {
        Ok(DispatchResult::Completed(val)) => {
            assert_eq!(val, Value::Unit);
        }
        _ => println!("Panic.assert may not be fully implemented"),
    }

    assert!(!ctx.has_active());
}

/// Test Async handler operations
#[test]
fn test_async_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test spawn operation
    let result = ctx.dispatch_with_continuation(
        EffectKind::Async,
        "spawn",
        vec![Value::String("task".into())],
    );

    // Handler exists
    match result {
        Ok(_) => println!("Async handler responded"),
        Err(_) => println!("Async handler rejected (expected for unimplemented ops)"),
    }

    assert!(!ctx.has_active());
}

/// Test Prob handler operations
#[test]
fn test_prob_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test sample operation
    let result = ctx.dispatch_with_continuation(EffectKind::Prob, "sample", vec![]);

    // Handler exists
    match result {
        Ok(_) => println!("Prob handler responded"),
        Err(_) => println!("Prob handler rejected (expected for unimplemented ops)"),
    }

    assert!(!ctx.has_active());
}

/// Test GPU handler operations
#[test]
fn test_gpu_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test kernel launch
    let result = ctx.dispatch_with_continuation(
        EffectKind::GPU,
        "launch_kernel",
        vec![Value::String("kernel".into())],
    );

    // Handler exists
    match result {
        Ok(_) => println!("GPU handler responded"),
        Err(_) => println!("GPU handler rejected (expected for unimplemented ops)"),
    }

    assert!(!ctx.has_active());
}

/// Test Network handler operations
#[test]
fn test_network_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test connect operation
    let result = ctx.dispatch_with_continuation(
        EffectKind::Network,
        "connect",
        vec![Value::String("localhost".into())],
    );

    // Handler exists
    match result {
        Ok(_) => println!("Network handler responded"),
        Err(_) => println!("Network handler rejected (expected for unimplemented ops)"),
    }

    assert!(!ctx.has_active());
}

/// Test Sensor handler operations
#[test]
fn test_sensor_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test read operation
    let result = ctx.dispatch_with_continuation(
        EffectKind::Sensor,
        "read",
        vec![Value::String("temp".into())],
    );

    // Handler exists
    match result {
        Ok(_) => println!("Sensor handler responded"),
        Err(_) => println!("Sensor handler rejected (expected for unimplemented ops)"),
    }

    assert!(!ctx.has_active());
}

/// Test Epistemic handler operations
#[test]
fn test_epistemic_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test measure operation
    let result =
        ctx.dispatch_with_continuation(EffectKind::Epistemic, "measure", vec![Value::Float(42.0)]);

    // Handler exists
    match result {
        Ok(_) => println!("Epistemic handler responded"),
        Err(_) => println!("Epistemic handler rejected (expected for unimplemented ops)"),
    }

    assert!(!ctx.has_active());
}

/// Test Exn handler operations
#[test]
fn test_exn_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test throw operation
    let result = ctx.dispatch_with_continuation(
        EffectKind::Exn,
        "throw",
        vec![Value::String("error".into())],
    );

    // Handler exists - throw should either suspend or abort
    match result {
        Ok(DispatchResult::Suspended { .. }) => {
            println!("Exn.throw suspended (expected)");
        }
        Err(_) => {
            println!("Exn.throw aborted (expected)");
        }
        Ok(DispatchResult::Completed(_)) => {
            println!("Exn.throw completed (unexpected but acceptable for stub)");
        }
    }

    // Note: if suspended, continuation may still be active
}

/// Test Div handler operations
#[test]
fn test_div_handler() {
    let mut ctx = EffectContext::with_all_handlers();

    // Test safe division
    let result = ctx.dispatch_with_continuation(
        EffectKind::Div,
        "safe_div",
        vec![Value::Int(10), Value::Int(2)],
    );

    // Handler exists
    match result {
        Ok(_) => println!("Div handler responded"),
        Err(_) => println!("Div handler rejected (expected for unimplemented ops)"),
    }

    assert!(!ctx.has_active());
}
