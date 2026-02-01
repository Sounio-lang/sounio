//! Demonstration of RealAsyncHandler with Tokio
//!
//! This example shows how to use the real async effect handler for:
//! - Cooperative task yielding
//! - Sleep/delay operations
//! - Task spawning (framework in place)
//! - Async/await patterns
//!
//! Run with:
//! ```bash
//! cargo run --example real_async_handler_demo --features effect-handlers
//! ```

use sounio::effects::handler_capability::{
    Continuation, HandlerCapability, HandlerResult, HandlerState,
};
use sounio::effects::handlers::RealAsyncHandler;
use sounio::interp::Value;

fn main() {
    println!("=== Real Async Handler Demonstration ===\n");

    let handler = RealAsyncHandler::new();
    let mut state = HandlerState::new();

    println!("Handler: {}", handler.handler_name());
    println!("Effect: {}", handler.effect_name());
    println!("Supports multi-shot: {}", handler.supports_multi_shot());
    println!();

    // Demo 1: Yield operation
    println!("Demo 1: Cooperative yielding");
    println!("-----------------------------");
    for i in 1..=3 {
        let result = handler.handle("yield", &[], Continuation::new(), &mut state);
        match result {
            HandlerResult::Resume(Value::Unit) => {
                println!("  Yield {} completed", i);
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }
    println!();

    // Demo 2: Sleep operations with timing
    println!("Demo 2: Sleep operations");
    println!("------------------------");

    let sleep_durations = vec![10, 25, 50];
    for duration in sleep_durations {
        println!("  Sleeping for {}ms...", duration);
        let start = std::time::Instant::now();

        let result = handler.handle(
            "sleep",
            &[Value::Int(duration)],
            Continuation::new(),
            &mut state,
        );

        let elapsed = start.elapsed();

        match result {
            HandlerResult::Resume(Value::Unit) => {
                println!(
                    "    Completed in {:.2}ms (expected {}ms)",
                    elapsed.as_secs_f64() * 1000.0,
                    duration
                );
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }
    println!();

    // Demo 3: Error handling
    println!("Demo 3: Error handling");
    println!("----------------------");

    // Negative duration
    println!("  Attempting sleep with negative duration...");
    let result = handler.handle("sleep", &[Value::Int(-10)], Continuation::new(), &mut state);
    match result {
        HandlerResult::Abort(err) => {
            println!("    ✓ Correctly rejected: {}", err.message);
        }
        other => panic!("Expected abort, got {:?}", other),
    }

    // Missing argument
    println!("  Attempting sleep without duration...");
    let result = handler.handle("sleep", &[], Continuation::new(), &mut state);
    match result {
        HandlerResult::Abort(err) => {
            println!("    ✓ Correctly rejected: {}", err.message);
        }
        other => panic!("Expected abort, got {:?}", other),
    }

    // Unknown operation
    println!("  Attempting unknown operation...");
    let result = handler.handle("nonexistent", &[], Continuation::new(), &mut state);
    match result {
        HandlerResult::Abort(err) => {
            println!("    ✓ Correctly rejected: {}", err.message);
        }
        other => panic!("Expected abort, got {:?}", other),
    }
    println!();

    // Demo 4: Epistemic impact
    println!("Demo 4: Epistemic impact tracking");
    println!("----------------------------------");

    let operations = vec!["spawn", "await", "yield", "sleep"];
    for op in operations {
        let impact = handler.epistemic_impact(op);
        println!("  {}: confidence = {:.3}", op, impact.confidence_factor);
    }
    println!();

    // Demo 5: Operation specifications
    println!("Demo 5: Available operations");
    println!("----------------------------");

    let ops = handler.operations();
    for op_spec in ops {
        println!(
            "  {} ({}) -> {}",
            op_spec.name,
            op_spec.param_types.join(", "),
            op_spec.return_type
        );
        if let Some(factor) = op_spec.confidence_factor {
            println!("    Confidence factor: {:.3}", factor);
        }
    }
    println!();

    // Demo 6: Concurrent sleep demonstration
    println!("Demo 6: Sequential sleep demonstration");
    println!("---------------------------------------");
    println!("  Sleeping 3 times (20ms each) sequentially...");

    let total_start = std::time::Instant::now();
    for i in 1..=3 {
        let result = handler.handle("sleep", &[Value::Int(20)], Continuation::new(), &mut state);
        match result {
            HandlerResult::Resume(Value::Unit) => {
                println!("    Task {} completed", i);
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }
    let total_elapsed = total_start.elapsed();
    println!(
        "  Total time: {:.2}ms (expected ~60ms)",
        total_elapsed.as_secs_f64() * 1000.0
    );
    println!();

    println!("=== Demo Complete ===");
    println!();
    println!("Key Features Demonstrated:");
    println!("  ✓ Cooperative yielding with tokio runtime");
    println!("  ✓ Real sleep/delay operations");
    println!("  ✓ Robust error handling");
    println!("  ✓ Epistemic confidence tracking");
    println!("  ✓ Operation introspection");
    println!();
    println!("Next Steps:");
    println!("  - Task spawning (framework in place, needs closure support)");
    println!("  - Async/await patterns");
    println!("  - Task joining and cancellation");
    println!("  - Timeout handling");
}
