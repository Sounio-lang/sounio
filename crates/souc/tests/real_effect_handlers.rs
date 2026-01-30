//! Integration tests for real effect handlers
//!
//! Tests the production-ready effect handlers:
//! - RealAsyncHandler (Tokio-based async/await)
//! - RealGpuHandler (WGPU-based GPU compute)
//! - RealNetworkHandler (Tokio-based network I/O)
//!
//! These tests verify that:
//! 1. Handlers correctly implement the HandlerCapability trait
//! 2. Operations execute successfully with valid inputs
//! 3. Error handling works for invalid inputs
//! 4. Epistemic impact tracking is correct
//! 5. Linearity enforcement works
//! 6. Feature gating provides helpful errors when features disabled

use sounio::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerResult, HandlerState,
};
use sounio::effects::handlers::async_handler_real::RealAsyncHandler;
use sounio::effects::linearity::Linearity;
use sounio::interp::Value;

// ============================================================================
// RealAsyncHandler Tests
// ============================================================================

#[test]
fn test_async_handler_creation() {
    let handler = RealAsyncHandler::new();
    assert_eq!(handler.effect_name(), "Async");
    assert_eq!(handler.handler_name(), "RealAsyncHandler");
}

#[test]
fn test_async_handler_operations_list() {
    let handler = RealAsyncHandler::new();
    let ops = handler.operations();

    // Should have 7 operations
    assert_eq!(ops.len(), 7);

    let op_names: Vec<&str> = ops.iter().map(|o| o.name.as_str()).collect();
    assert!(op_names.contains(&"spawn"));
    assert!(op_names.contains(&"await"));
    assert!(op_names.contains(&"yield"));
    assert!(op_names.contains(&"sleep"));
    assert!(op_names.contains(&"join"));
    assert!(op_names.contains(&"cancel"));
    assert!(op_names.contains(&"timeout"));
}

#[test]
fn test_async_yield_operation() {
    let handler = RealAsyncHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    let result = handler.handle("yield", &[], cont, &mut state);

    match result {
        HandlerResult::Resume(Value::Unit) => {}
        other => panic!("Expected Resume(Unit), got {:?}", other),
    }
}

#[test]
fn test_async_sleep_operation() {
    let handler = RealAsyncHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Test with 10ms sleep
    let start = std::time::Instant::now();
    let result = handler.handle("sleep", &[Value::Int(10)], cont, &mut state);
    let elapsed = start.elapsed();

    match result {
        HandlerResult::Resume(Value::Unit) => {}
        other => panic!("Expected Resume(Unit), got {:?}", other),
    }

    // Should have actually slept
    assert!(elapsed.as_millis() >= 10);
}

#[test]
fn test_async_sleep_negative_duration() {
    let handler = RealAsyncHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    let result = handler.handle("sleep", &[Value::Int(-10)], cont, &mut state);

    match result {
        HandlerResult::Abort(_) => {}
        other => panic!("Expected Abort for negative duration, got {:?}", other),
    }
}

#[test]
fn test_async_sleep_missing_argument() {
    let handler = RealAsyncHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    let result = handler.handle("sleep", &[], cont, &mut state);

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("duration"));
        }
        other => panic!("Expected Abort for missing argument, got {:?}", other),
    }
}

#[test]
fn test_async_unknown_operation() {
    let handler = RealAsyncHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    let result = handler.handle("unknown_op", &[], cont, &mut state);

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("unknown operation"));
        }
        other => panic!("Expected Abort for unknown operation, got {:?}", other),
    }
}

#[test]
fn test_async_handler_linearity() {
    let handler = RealAsyncHandler::new();

    // All async operations should be exactly-once
    assert_eq!(handler.operation_linearity("spawn"), Linearity::ExactlyOnce);
    assert_eq!(handler.operation_linearity("await"), Linearity::ExactlyOnce);
    assert_eq!(handler.operation_linearity("yield"), Linearity::ExactlyOnce);
    assert_eq!(handler.operation_linearity("sleep"), Linearity::ExactlyOnce);
}

#[test]
fn test_async_handler_epistemic_impact() {
    let handler = RealAsyncHandler::new();

    // Spawn/await/yield introduce timing uncertainty
    let spawn_impact = handler.epistemic_impact("spawn");
    assert!(spawn_impact.confidence_factor < 1.0);
    assert!(spawn_impact.confidence_factor >= 0.9);

    // Sleep is deterministic
    let sleep_impact = handler.epistemic_impact("sleep");
    assert_eq!(sleep_impact.confidence_factor, 1.0);
}

#[test]
fn test_async_handler_not_multishot() {
    let handler = RealAsyncHandler::new();
    assert!(!handler.supports_multi_shot());
}

// ============================================================================
// RealGpuHandler Tests (feature-gated)
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_handler_creation() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();
    assert_eq!(handler.effect_name(), "GPU");
    assert_eq!(handler.handler_name(), "RealGpuHandler");
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_handler_operations_list() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();
    let ops = handler.operations();

    // Should have 8 operations
    assert_eq!(ops.len(), 8);

    let op_names: Vec<&str> = ops.iter().map(|o| o.name.as_str()).collect();
    assert!(op_names.contains(&"compile_kernel"));
    assert!(op_names.contains(&"launch"));
    assert!(op_names.contains(&"allocate"));
    assert!(op_names.contains(&"copy_to_device"));
    assert!(op_names.contains(&"copy_to_host"));
    assert!(op_names.contains(&"free"));
    assert!(op_names.contains(&"sync"));
    assert!(op_names.contains(&"get_device_info"));
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_handler_linearity() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();

    // GPU operations should be exactly-once
    assert_eq!(
        handler.operation_linearity("launch"),
        Linearity::ExactlyOnce
    );
    assert_eq!(
        handler.operation_linearity("allocate"),
        Linearity::ExactlyOnce
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_handler_epistemic_impact() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();

    // GPU compute has high confidence (0.99) due to precision differences
    let launch_impact = handler.epistemic_impact("launch");
    assert!(launch_impact.confidence_factor >= 0.99);
    assert!(launch_impact.confidence_factor < 1.0);
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_compile_kernel_invalid_shader() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Invalid WGSL shader
    let invalid_shader = "this is not valid WGSL";

    let result = handler.handle(
        "compile_kernel",
        &[Value::String(invalid_shader.to_string())],
        cont,
        &mut state,
    );

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("shader") || err.message.contains("compile"));
        }
        other => panic!("Expected Abort for invalid shader, got {:?}", other),
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_compile_kernel_valid_shader() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Valid minimal WGSL shader
    let shader = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            data[global_id.x] = f32(global_id.x);
        }
    "#;

    let result = handler.handle(
        "compile_kernel",
        &[Value::String(shader.to_string())],
        cont,
        &mut state,
    );

    match result {
        HandlerResult::Resume(Value::Int(_kernel_id)) => {}
        other => panic!("Expected Resume with kernel ID, got {:?}", other),
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_allocate_buffer() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Allocate 1024 bytes
    let result = handler.handle("allocate", &[Value::Int(1024)], cont, &mut state);

    match result {
        HandlerResult::Resume(Value::Int(buffer_id)) => {
            assert!(buffer_id > 0);
        }
        other => panic!("Expected Resume with buffer ID, got {:?}", other),
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_allocate_invalid_size() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Negative size
    let result = handler.handle("allocate", &[Value::Int(-100)], cont, &mut state);

    match result {
        HandlerResult::Abort(_) => {}
        other => panic!("Expected Abort for negative size, got {:?}", other),
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_get_device_info() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    let result = handler.handle("get_device_info", &[], cont, &mut state);

    match result {
        HandlerResult::Resume(Value::String(info)) => {
            // Should contain some device information
            assert!(!info.is_empty());
        }
        other => panic!("Expected Resume with device info, got {:?}", other),
    }
}

// ============================================================================
// RealNetworkHandler Tests (feature-gated)
// ============================================================================

#[cfg(feature = "tokio")]
#[test]
fn test_network_handler_creation() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();
    assert_eq!(handler.effect_name(), "Network");
    assert_eq!(handler.handler_name(), "RealNetworkHandler");
}

#[cfg(feature = "tokio")]
#[test]
fn test_network_handler_operations_list() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();
    let ops = handler.operations();

    // Should have 10 operations
    assert_eq!(ops.len(), 10);

    let op_names: Vec<&str> = ops.iter().map(|o| o.name.as_str()).collect();
    assert!(op_names.contains(&"connect"));
    assert!(op_names.contains(&"listen"));
    assert!(op_names.contains(&"send"));
    assert!(op_names.contains(&"recv"));
    assert!(op_names.contains(&"close"));
    assert!(op_names.contains(&"http_get"));
    assert!(op_names.contains(&"http_post"));
    assert!(op_names.contains(&"udp_send"));
    assert!(op_names.contains(&"udp_bind"));
    assert!(op_names.contains(&"udp_recv"));
}

#[cfg(feature = "tokio")]
#[test]
fn test_network_handler_linearity() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();

    // Network operations should be exactly-once
    assert_eq!(handler.operation_linearity("send"), Linearity::ExactlyOnce);
    assert_eq!(handler.operation_linearity("recv"), Linearity::ExactlyOnce);
}

#[cfg(feature = "tokio")]
#[test]
fn test_network_handler_epistemic_impact() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();

    // Receive operations have reduced confidence (external data)
    let recv_impact = handler.epistemic_impact("recv");
    assert!(recv_impact.confidence_factor < 1.0);
    assert!(recv_impact.confidence_factor >= 0.9);

    // Send operations don't reduce confidence
    let send_impact = handler.epistemic_impact("send");
    assert_eq!(send_impact.confidence_factor, 1.0);
}

#[cfg(feature = "tokio")]
#[test]
fn test_network_connect_invalid_host() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Invalid host
    let result = handler.handle(
        "connect",
        &[
            Value::String("invalid.host.that.does.not.exist.xyz".to_string()),
            Value::Int(80),
        ],
        cont,
        &mut state,
    );

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("connect") || err.message.contains("host"));
        }
        other => panic!("Expected Abort for invalid host, got {:?}", other),
    }
}

#[cfg(feature = "tokio")]
#[test]
fn test_network_connect_missing_arguments() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Missing port
    let result = handler.handle(
        "connect",
        &[Value::String("localhost".to_string())],
        cont,
        &mut state,
    );

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("argument") || err.message.contains("port"));
        }
        other => panic!("Expected Abort for missing argument, got {:?}", other),
    }
}

#[cfg(all(feature = "tokio", feature = "reqwest"))]
#[test]
fn test_network_http_get_invalid_url() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Invalid URL scheme
    let result = handler.handle(
        "http_get",
        &[Value::String("ftp://example.com".to_string())],
        cont,
        &mut state,
    );

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("http") || err.message.contains("scheme"));
        }
        other => panic!("Expected Abort for invalid URL scheme, got {:?}", other),
    }
}

#[cfg(all(feature = "tokio", feature = "reqwest"))]
#[test]
#[ignore] // Requires network access
fn test_network_http_get_success() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Use a reliable test endpoint
    let result = handler.handle(
        "http_get",
        &[Value::String("https://httpbin.org/get".to_string())],
        cont,
        &mut state,
    );

    match result {
        HandlerResult::Resume(Value::String(body)) => {
            assert!(!body.is_empty());
            assert!(body.contains("httpbin"));
        }
        other => panic!("Expected Resume with response body, got {:?}", other),
    }
}

// ============================================================================
// Feature Gate Tests
// ============================================================================

#[test]
fn test_async_handler_works_without_tokio_feature() {
    // RealAsyncHandler should always compile, but provide fallback behavior
    // when tokio is not available
    let handler = RealAsyncHandler::new();
    assert_eq!(handler.effect_name(), "Async");
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn test_gpu_handler_fallback_without_wgpu() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Should return helpful error when WGPU not available
    let result = handler.handle("allocate", &[Value::Int(1024)], cont, &mut state);

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("wgpu") || err.message.contains("feature"));
        }
        other => panic!("Expected Abort with feature error, got {:?}", other),
    }
}

#[cfg(not(feature = "tokio"))]
#[test]
fn test_network_handler_fallback_without_tokio() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    let handler = RealNetworkHandler::new();
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    // Should return helpful error when tokio not available
    let result = handler.handle(
        "connect",
        &[Value::String("localhost".to_string()), Value::Int(8080)],
        cont,
        &mut state,
    );

    match result {
        HandlerResult::Abort(err) => {
            assert!(err.message.contains("tokio") || err.message.contains("feature"));
        }
        other => panic!("Expected Abort with feature error, got {:?}", other),
    }
}

// ============================================================================
// Cross-Handler Integration Tests
// ============================================================================

#[cfg(all(feature = "tokio", feature = "reqwest"))]
#[test]
#[ignore] // Requires network access
fn test_async_with_network() {
    use sounio::effects::handlers::network_handler_real::RealNetworkHandler;

    // Test that async and network handlers can work together
    let async_handler = RealAsyncHandler::new();
    let network_handler = RealNetworkHandler::new();

    let mut async_state = HandlerState::new();
    let mut network_state = HandlerState::new();

    // Sleep before network request
    let sleep_result = async_handler.handle(
        "sleep",
        &[Value::Int(10)],
        Continuation::new(),
        &mut async_state,
    );
    assert!(matches!(sleep_result, HandlerResult::Resume(_)));

    // Make network request
    let http_result = network_handler.handle(
        "http_get",
        &[Value::String("https://httpbin.org/get".to_string())],
        Continuation::new(),
        &mut network_state,
    );
    assert!(matches!(http_result, HandlerResult::Resume(_)));
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gpu_with_epistemic_tracking() {
    use sounio::effects::handlers::gpu_handler_real::RealGpuHandler;

    let handler = RealGpuHandler::new();

    // Verify epistemic impact is properly configured
    let allocate_impact = handler.epistemic_impact("allocate");
    let launch_impact = handler.epistemic_impact("launch");

    // Both should have slight confidence reduction
    assert!(allocate_impact.confidence_factor >= 0.99);
    assert!(launch_impact.confidence_factor >= 0.99);
}
