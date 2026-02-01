//! Example demonstrating the real GPU handler with WGPU
//!
//! This example shows how to use the RealGpuHandler to compile and execute
//! GPU compute shaders using WGPU.
//!
//! # Running
//!
//! With WGPU feature enabled:
//! ```bash
//! cargo run --example real_gpu_handler_demo --features wgpu
//! ```
//!
//! Without WGPU feature (will show fallback behavior):
//! ```bash
//! cargo run --example real_gpu_handler_demo
//! ```

use sounio::effects::handler_capability::{Continuation, HandlerCapability, HandlerState};
use sounio::effects::handlers::RealGpuHandler;
use sounio::interp::Value;
use std::cell::RefCell;
use std::rc::Rc;

/// Create a fresh continuation for each handler call
fn fresh_cont() -> Continuation {
    Continuation::new()
}

fn main() {
    println!("=== Real GPU Handler Demo ===\n");

    let handler = RealGpuHandler::new();
    let mut state = HandlerState::new();

    println!("Handler: {}", handler.handler_name());
    println!("Effect: {}\n", handler.effect_name());

    // List available operations
    println!("Available operations:");
    for op in handler.operations() {
        println!(
            "  - {}({}) -> {}",
            op.name,
            op.param_types.join(", "),
            op.return_type
        );
    }
    println!();

    // Example 1: Get device info
    println!("Example 1: Get Device Info");
    println!("----------------------------");
    let result = handler.handle("get_device_info", &[], fresh_cont(), &mut state);
    match result {
        sounio::effects::handler_capability::HandlerResult::Resume(value) => {
            println!("Device info: {:?}", value);
        }
        sounio::effects::handler_capability::HandlerResult::Abort(err) => {
            println!("Error: {}", err.message);
        }
        _ => println!("Unexpected result"),
    }
    println!();

    // Example 2: Compile a simple WGSL shader
    println!("Example 2: Compile WGSL Shader");
    println!("-------------------------------");

    let wgsl_source = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = input[idx] * 2.0;
}
"#;

    let source = Value::String(wgsl_source.to_string());
    let entry = Value::String("main".to_string());

    let result = handler.handle("compile_kernel", &[source, entry], fresh_cont(), &mut state);

    let kernel_id = match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Int(id)) => {
            println!("Shader compiled successfully! Kernel ID: {}", id);
            id
        }
        sounio::effects::handler_capability::HandlerResult::Abort(err) => {
            println!("Compilation failed: {}", err.message);
            return;
        }
        _ => {
            println!("Unexpected result from compilation");
            return;
        }
    };
    println!();

    // Example 3: Allocate GPU buffers
    println!("Example 3: Allocate GPU Buffers");
    println!("--------------------------------");

    // Allocate input buffer (1024 * 4 bytes = 4 KB)
    let size = Value::Int(4096);
    let result = handler.handle("allocate", &[size.clone()], fresh_cont(), &mut state);
    let input_buffer_id = match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Int(id)) => {
            println!("Input buffer allocated: ID = {}", id);
            id
        }
        sounio::effects::handler_capability::HandlerResult::Abort(err) => {
            println!("Allocation failed: {}", err.message);
            return;
        }
        _ => {
            println!("Unexpected result from allocation");
            return;
        }
    };

    // Allocate output buffer
    let result = handler.handle("allocate", &[size], fresh_cont(), &mut state);
    let output_buffer_id = match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Int(id)) => {
            println!("Output buffer allocated: ID = {}", id);
            id
        }
        _ => {
            println!("Output buffer allocation failed");
            return;
        }
    };

    println!(
        "Total allocated: {} bytes\n",
        RealGpuHandler::total_allocated_memory(&state)
    );

    // Example 4: Copy data to device
    println!("Example 4: Copy Data to Device");
    println!("-------------------------------");

    // Create input data (1024 floats)
    let input_data: Vec<Value> = (0..1024).map(|i| Value::Float(i as f64)).collect();
    let input_array = Value::Array(Rc::new(RefCell::new(input_data)));

    let result = handler.handle(
        "copy_to_device",
        &[input_array, Value::Int(input_buffer_id)],
        fresh_cont(),
        &mut state,
    );

    match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Unit) => {
            println!("Data copied to device successfully!");
        }
        sounio::effects::handler_capability::HandlerResult::Abort(err) => {
            println!("Copy to device failed: {}", err.message);
            return;
        }
        _ => {
            println!("Unexpected result from copy_to_device");
            return;
        }
    }
    println!();

    // Example 5: Launch kernel
    println!("Example 5: Launch Kernel");
    println!("------------------------");

    // Grid: (4, 1, 1) workgroups
    let grid_dims = Value::Tuple(vec![Value::Int(4), Value::Int(1), Value::Int(1)]);
    // Block: (256, 1, 1) threads per workgroup
    let block_dims = Value::Tuple(vec![Value::Int(256), Value::Int(1), Value::Int(1)]);

    let result = handler.handle(
        "launch",
        &[Value::Int(kernel_id), grid_dims, block_dims],
        fresh_cont(),
        &mut state,
    );

    match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Unit) => {
            println!("Kernel launched successfully!");
            println!("  Grid dimensions: (4, 1, 1)");
            println!("  Workgroup size: (256, 1, 1)");
            println!("  Total threads: 1024");
        }
        sounio::effects::handler_capability::HandlerResult::Abort(err) => {
            println!("Kernel launch failed: {}", err.message);
            return;
        }
        _ => {
            println!("Unexpected result from launch");
            return;
        }
    }
    println!();

    // Example 6: Synchronize
    println!("Example 6: Synchronize GPU");
    println!("--------------------------");

    let result = handler.handle("sync", &[], fresh_cont(), &mut state);
    match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Unit) => {
            println!("GPU synchronized successfully!");
        }
        sounio::effects::handler_capability::HandlerResult::Abort(err) => {
            println!("Sync failed: {}", err.message);
            return;
        }
        _ => {
            println!("Unexpected result from sync");
            return;
        }
    }
    println!();

    // Example 7: Copy results back to host
    println!("Example 7: Copy Results to Host");
    println!("--------------------------------");

    let result = handler.handle(
        "copy_to_host",
        &[Value::Int(output_buffer_id), Value::Int(1024)],
        fresh_cont(),
        &mut state,
    );

    match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Array(arr)) => {
            let data = arr.borrow();
            println!("Copied {} elements from device", data.len());
            if !data.is_empty() {
                println!("First 5 elements: {:?}", &data[..5.min(data.len())]);
            }
        }
        sounio::effects::handler_capability::HandlerResult::Abort(err) => {
            println!("Copy to host failed: {}", err.message);
            return;
        }
        _ => {
            println!("Unexpected result from copy_to_host");
            return;
        }
    }
    println!();

    // Example 8: Free GPU buffers
    println!("Example 8: Free GPU Buffers");
    println!("---------------------------");

    let result = handler.handle(
        "free",
        &[Value::Int(input_buffer_id)],
        fresh_cont(),
        &mut state,
    );
    match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Unit) => {
            println!("Input buffer freed");
        }
        _ => println!("Failed to free input buffer"),
    }

    let result = handler.handle(
        "free",
        &[Value::Int(output_buffer_id)],
        fresh_cont(),
        &mut state,
    );
    match result {
        sounio::effects::handler_capability::HandlerResult::Resume(Value::Unit) => {
            println!("Output buffer freed");
        }
        _ => println!("Failed to free output buffer"),
    }

    println!(
        "Total allocated after cleanup: {} bytes\n",
        RealGpuHandler::total_allocated_memory(&state)
    );

    // Summary
    println!("=== Summary ===");
    println!("✓ Compiled WGSL shader");
    println!("✓ Allocated GPU buffers");
    println!("✓ Copied data to device");
    println!("✓ Launched compute kernel");
    println!("✓ Synchronized GPU");
    println!("✓ Copied results to host");
    println!("✓ Freed GPU resources");
    println!("\nAll GPU operations completed successfully!");

    // Show epistemic impact
    println!("\n=== Epistemic Impact ===");
    let launch_impact = handler.epistemic_impact("launch");
    println!(
        "Launch operation confidence: {:.2}%",
        launch_impact.confidence_factor * 100.0
    );
    let copy_impact = handler.epistemic_impact("copy_to_host");
    println!(
        "Copy to host confidence: {:.2}%",
        copy_impact.confidence_factor * 100.0
    );
}
