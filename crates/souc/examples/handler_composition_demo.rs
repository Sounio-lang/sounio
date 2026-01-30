//! Handler Composition Demo
//!
//! Demonstrates the new handler composition feature (Task 3.1 from Q1 literature integration).
//! Based on "Soundly Handling Linearity" (POPL 2024).
//!
//! This example shows:
//! - Composing multiple effect handlers (IO, Mut, Epistemic)
//! - Different composition orders (LeftToRight, RightToLeft, Priority)
//! - Handler dispatch semantics
//! - Linear handler stack for one-time consumption

use sounio::effects::handler_capability::Continuation;
use sounio::effects::handlers::{
    ComposedHandler, CompositionOrder, EpistemicHandler, HandlerComposer, HandlerStack, IOHandler,
    MutHandler, PanicHandler,
};
use sounio::effects::{HandlerCapability, HandlerResult, HandlerState};
use sounio::interp::Value;
use std::sync::Arc;

fn main() {
    println!("=== Handler Composition Demo ===\n");

    // Example 1: Basic composition with left-to-right order
    demo_basic_composition();

    // Example 2: Priority-based dispatch
    demo_priority_composition();

    // Example 3: Linear handler stack
    demo_handler_stack();
}

fn demo_basic_composition() {
    println!("--- Example 1: Basic Composition (Left-to-Right) ---");

    let composed = HandlerComposer::new()
        .with(Arc::new(IOHandler::new()))
        .with(Arc::new(MutHandler::new()))
        .with(Arc::new(EpistemicHandler::new()))
        .order(CompositionOrder::LeftToRight)
        .build()
        .expect("Valid composition");

    println!("Composed handler: {}", composed.effect_name());
    println!("Effect names: {:?}", composed.effect_names());
    println!("Handler count: {}", composed.handler_count());

    // Dispatch to IO handler
    let mut state = HandlerState::new();
    let cont = Continuation::new();

    let result = composed.handle(
        "print",
        &[Value::String("Hello from composed handler!".into())],
        cont,
        &mut state,
    );

    match result {
        HandlerResult::Resume(_) => {
            println!("✓ Successfully dispatched 'print' to IO handler\n");
        }
        _ => println!("✗ Unexpected result\n"),
    }
}

fn demo_priority_composition() {
    println!("--- Example 2: Priority-Based Dispatch ---");

    // Priority: Epistemic > IO > Mut
    let priority = vec!["Epistemic".to_string(), "IO".to_string(), "Mut".to_string()];

    let composed = HandlerComposer::new()
        .with(Arc::new(IOHandler::new()))
        .with(Arc::new(MutHandler::new()))
        .with(Arc::new(EpistemicHandler::new()))
        .order(CompositionOrder::Priority(priority))
        .build()
        .expect("Valid composition");

    println!("Priority order: Epistemic > IO > Mut");
    println!("Composed: {}\n", composed.effect_name());

    // Test that we can find specific handlers
    if let Some(handler) = composed.find_handler("Epistemic") {
        println!("✓ Found handler: {}", handler.effect_name());
    }

    println!();
}

fn demo_handler_stack() {
    println!("--- Example 3: Linear Handler Stack ---");

    let mut stack = HandlerStack::new();

    // Push handlers onto stack
    stack.push_handler(Arc::new(IOHandler::new()));
    stack.push_handler(Arc::new(MutHandler::new()));
    stack.push_handler(Arc::new(PanicHandler::new()));

    println!("Stack depth: {}", stack.len());
    println!("Effect names (top to bottom): {:?}", stack.effect_names());

    // Pop handlers (linear consumption)
    while let Some(handler) = stack.pop_handler() {
        println!("Consumed: {} (marked as consumed)", handler.effect_name());
    }

    // Verify consumed status
    println!("\nConsumed checks:");
    println!("  Panic consumed? {}", stack.is_consumed("Panic"));
    println!("  Mut consumed? {}", stack.is_consumed("Mut"));
    println!("  IO consumed? {}", stack.is_consumed("IO"));
    println!("\nStack is now empty: {}", stack.is_empty());
}
