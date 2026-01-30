//! Simple MIR Pipeline Example
//!
//! This is a minimal example showing how to use the MIR optimization pipeline.

use sounio::mir::builder::ModuleBuilder;
use sounio::mir::optimization::{OptimizationLevel, PassManager};
use sounio::mir::types::MirType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simple MIR Pipeline Example");
    println!("============================");

    // Create a simple MIR module with one function
    let mut module_builder = ModuleBuilder::new("example");

    let mut func = module_builder.create_function("add_numbers".to_string(), MirType::I64);

    // Build a simple function that returns a constant
    let result = func.build_i64(42);
    func.build_return(Some(result));

    module_builder.add_function(func.build());
    let module = module_builder.build();

    println!("Created module with {} function(s)", module.functions.len());

    // Create pass manager with O2 optimization level
    let mut pass_manager = PassManager::new_with_level(OptimizationLevel::O2);

    // Run optimization passes on each function
    for func in &mut module.functions.clone() {
        let result = pass_manager.run_function_passes(&mut func.clone())?;
        println!("Passes applied: {}", result.passes_applied.len());
        println!("Instructions reduced: {}", result.instructions_reduced);
    }

    println!("\nOptimization complete!");
    Ok(())
}
