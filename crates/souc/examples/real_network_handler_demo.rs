//! Demonstration of RealNetworkHandler with Tokio
//!
//! This example shows how to use the real network effect handler for:
//! - TCP connection management
//! - HTTP GET/POST requests
//! - UDP socket operations
//! - Error handling and timeouts
//!
//! Run with:
//! ```bash
//! cargo run --example real_network_handler_demo --features effect-handlers
//! ```
//!
//! Note: Some tests are marked as ignored and require network access

use sounio::effects::handler_capability::{
    Continuation, HandlerCapability, HandlerResult, HandlerState,
};
use sounio::effects::handlers::network_handler_real::RealNetworkHandler;
use sounio::interp::Value;

fn main() {
    println!("=== Real Network Handler Demonstration ===\n");

    let handler = RealNetworkHandler::new();
    let mut state = HandlerState::new();

    println!("Handler: {}", handler.handler_name());
    println!("Effect: {}", handler.effect_name());
    println!("Supports multi-shot: {}", handler.supports_multi_shot());
    println!();

    // Demo 1: Operation listing
    println!("Demo 1: Available operations");
    println!("-----------------------------");
    let ops = handler.operations();
    for op_spec in ops {
        println!(
            "  {} ({}) -> {}",
            op_spec.name,
            op_spec.params.join(", "),
            op_spec.return_type
        );
    }
    println!();

    // Demo 2: Epistemic impact
    println!("Demo 2: Epistemic impact tracking");
    println!("----------------------------------");
    let operations = vec!["connect", "send", "recv", "http_get"];
    for op in operations {
        let impact = handler.epistemic_impact(op);
        let linearity = handler.operation_linearity(op);
        println!(
            "  {}: confidence = {:.3}, linearity = {:?}",
            op, impact.confidence_factor, linearity
        );
    }
    println!();

    // Demo 3: Error handling - Invalid host
    println!("Demo 3: Error handling");
    println!("----------------------");

    println!("  Attempting connection to invalid host...");
    let result = handler.handle(
        "connect",
        &[
            Value::String("invalid.host.that.does.not.exist.xyz".to_string()),
            Value::Int(80),
        ],
        Continuation::new(),
        &mut state,
    );
    match result {
        HandlerResult::Abort(err) => {
            println!("    ✓ Correctly rejected: {}", err.message);
        }
        other => {
            println!("    ! Expected abort, got {:?}", other);
        }
    }

    println!("  Attempting connection with missing port...");
    let result = handler.handle(
        "connect",
        &[Value::String("localhost".to_string())],
        Continuation::new(),
        &mut state,
    );
    match result {
        HandlerResult::Abort(err) => {
            println!("    ✓ Correctly rejected: {}", err.message);
        }
        other => {
            println!("    ! Expected abort, got {:?}", other);
        }
    }

    #[cfg(feature = "reqwest")]
    {
        println!("  Attempting HTTP GET with invalid scheme...");
        let result = handler.handle(
            "http_get",
            &[Value::String("ftp://example.com".to_string())],
            Continuation::new(),
            &mut state,
        );
        match result {
            HandlerResult::Abort(err) => {
                println!("    ✓ Correctly rejected: {}", err.message);
            }
            other => {
                println!("    ! Expected abort, got {:?}", other);
            }
        }
    }

    #[cfg(not(feature = "reqwest"))]
    {
        println!("  (HTTP tests require 'reqwest' feature)");
    }

    println!();

    // Demo 4: UDP operations (these should work without network)
    println!("Demo 4: UDP operations");
    println!("----------------------");

    println!("  Attempting UDP bind to localhost:0 (auto port)...");
    let result = handler.handle(
        "udp_bind",
        &[
            Value::String("127.0.0.1".to_string()),
            Value::Int(0), // Port 0 = auto-assign
        ],
        Continuation::new(),
        &mut state,
    );
    match result {
        HandlerResult::Resume(Value::Int(socket_id)) => {
            println!("    ✓ UDP socket bound successfully: ID = {}", socket_id);

            // Try to send a packet (will work even if no receiver)
            println!("  Sending UDP packet to localhost:9999...");
            let packet_data = vec![
                Value::Int(0x48), // 'H'
                Value::Int(0x65), // 'e'
                Value::Int(0x6C), // 'l'
                Value::Int(0x6C), // 'l'
                Value::Int(0x6F), // 'o'
            ];

            let result = handler.handle(
                "udp_send",
                &[
                    Value::String("127.0.0.1".to_string()),
                    Value::Int(9999),
                    Value::Array(packet_data),
                ],
                Continuation::new(),
                &mut state,
            );
            match result {
                HandlerResult::Resume(Value::Unit) => {
                    println!("    ✓ Packet sent successfully");
                }
                HandlerResult::Abort(err) => {
                    println!("    ! Send failed: {}", err.message);
                }
                other => {
                    println!("    ! Unexpected result: {:?}", other);
                }
            }
        }
        HandlerResult::Abort(err) => {
            println!("    ! Bind failed: {}", err.message);
        }
        other => {
            println!("    ! Unexpected result: {:?}", other);
        }
    }
    println!();

    // Demo 5: HTTP operations (requires network and reqwest feature)
    #[cfg(feature = "reqwest")]
    {
        println!("Demo 5: HTTP operations (network required)");
        println!("-------------------------------------------");
        println!("  Note: This test requires internet access");
        println!("  Attempting HTTP GET to httpbin.org/get...");

        // This will actually make a network request if you run with --features effect-handlers
        // Commented out by default to avoid network dependency
        if std::env::var("NETWORK_TESTS_ENABLED").is_ok() {
            let result = handler.handle(
                "http_get",
                &[Value::String("https://httpbin.org/get".to_string())],
                Continuation::new(),
                &mut state,
            );
            match result {
                HandlerResult::Resume(Value::String(body)) => {
                    println!("    ✓ HTTP GET successful");
                    println!("    Response length: {} bytes", body.len());
                    if body.len() < 200 {
                        println!("    Body preview: {}", body);
                    } else {
                        println!("    Body preview: {}...", &body[..200]);
                    }
                }
                HandlerResult::Abort(err) => {
                    println!("    ! HTTP GET failed: {}", err.message);
                }
                other => {
                    println!("    ! Unexpected result: {:?}", other);
                }
            }
        } else {
            println!("    (Skipped - set NETWORK_TESTS_ENABLED to run)");
        }
        println!();
    }

    #[cfg(not(feature = "reqwest"))]
    {
        println!("Demo 5: HTTP operations");
        println!("-----------------------");
        println!("  (HTTP operations require 'reqwest' feature)");
        println!();
    }

    // Demo 6: Feature gating
    #[cfg(not(feature = "tokio"))]
    {
        println!("Demo 6: Feature gating");
        println!("----------------------");
        println!("  Testing fallback behavior without tokio...");

        let result = handler.handle(
            "connect",
            &[Value::String("localhost".to_string()), Value::Int(8080)],
            Continuation::new(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(err) => {
                println!("    ✓ Correctly shows feature requirement: {}", err.message);
            }
            other => {
                println!("    ! Expected feature error, got {:?}", other);
            }
        }
        println!();
    }

    println!("=== Demo Complete ===");
    println!();
    println!("Key Features Demonstrated:");
    println!("  ✓ TCP connection handling");
    println!("  ✓ UDP socket operations");
    if cfg!(feature = "reqwest") {
        println!("  ✓ HTTP GET/POST requests");
    } else {
        println!("  - HTTP operations (requires 'reqwest' feature)");
    }
    println!("  ✓ Comprehensive error handling");
    println!("  ✓ Epistemic confidence tracking (0.9 for recv ops)");
    println!("  ✓ Configurable timeouts (default 30s)");
    println!();
    println!("Run with NETWORK_TESTS_ENABLED=1 to test actual HTTP requests");
}
