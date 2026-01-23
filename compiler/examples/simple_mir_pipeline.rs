//! Simple MIR Pipeline Example
//!
//! This is a minimal example showing how to use the MIR optimization pipeline.

use sounio_compiler::mir::{MirModule, MirFunction};
use sounio_compiler::mir::builder::ModuleBuilder;
use sounio_compiler::mir::types::MirType;
use sounio_compiler::mir::instructions::{MirInstruction, MirConstant, MirBinaryOp};
use sounio_compiler::mir::optimization::{OptimizationLevel, create_default_pass_manager};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Simple MIR Pipeline Example");
    println!("============================");
    
    // Create a simple MIR function
    let mut module = create_simple_function()?;
    println!("📦 Created simple function:");
    print_simple_stats(&module);
    
    // Apply O2 optimization
    println!("\n⚡ Applying O2 optimization...");
    let mut pass_manager = create_default_pass_manager(OptimizationLevel::O2);
    
    for func in &mut module.functions {
        let result = pass_manager.run_function_passes(func)?;
        println!("  📊 Passes applied: {}", result.passes_applied.len());
        println!("  🗑️ Instructions reduced: {}", result.instructions_reduced);
    }
    
    // Print optimized result
    println!("\n📊 After optimization:");
    print_simple_stats(&module);
    
    println!("\n✅ Simple pipeline demo completed!");
    Ok(())
}

fn create_simple_function() -> Result<MirModule, String> {
    let mut module_builder = ModuleBuilder::new("simple_demo");
    
    // Create: fn add(a: i64, b: i64) -> i64 { return a + b; }
    let mut func = module_builder.create_function("add".to_string(), MirType::I64);
    
    // Add parameters
    func.add_param("a".to_string(), MirType::I64);
    func.add_param("b".to_string(), MirType::I64);
    
    // Build: return a + b
    let result = func.fresh_value();
    func.build_add(result, func.get_param(0), func.get_param(1), MirType::I64);
    func.build_return(Some(result));
    
    module_builder.add_function(func.build());
    
    Ok(module_builder.build())
}

fn print_simple_stats(module: &MirModule) {
    for func in &module.functions {
        let total_instructions: usize = func.blocks.iter()
            .map(|block| block.instructions.len())
            .sum();
        
        println!("  Function: {}", func.name);
        println!("  Parameters: {}", func.params.len());
        println!("  Instructions: {}", total_instructions);
        println!("  Blocks: {}", func.blocks.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_pipeline() {
        let module = create_simple_function().unwrap();
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, "add");
    }
}
