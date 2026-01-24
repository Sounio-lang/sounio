//! Complete MIR Optimization Pipeline Example
//!
//! This example demonstrates the MIR optimization pipeline with SSA validation.

use sounio::mir::builder::ModuleBuilder;
use sounio::mir::types::MirType;
use sounio::mir::analysis::ssa_validator::SSAValidator;
use sounio::mir::optimization::{OptimizationLevel, PassManager};

fn main() -> Result<(), String> {
    println!("Complete MIR Optimization Pipeline Demo");
    println!("=======================================");

    // 1. Create test module with epistemic features
    let module = create_test_module()?;
    println!("Created test module with {} function(s)", module.functions.len());

    // 2. Run SSA validation
    let mut validator = SSAValidator::new();
    for func in &module.functions {
        let result = validator.validate_function(func);
        if result.is_valid {
            println!("SSA validation passed for: {}", func.name);
        } else {
            println!("SSA validation errors for {}: {:?}", func.name, result.errors);
        }
    }

    // 3. Run optimizations
    let mut pass_manager = PassManager::new_with_level(OptimizationLevel::O2);
    for func in &module.functions {
        let result = pass_manager.run_function_passes(&mut func.clone())
            .map_err(|e| format!("{:?}", e))?;
        println!(
            "Optimized {}: {} passes, {} instructions reduced",
            func.name,
            result.passes_applied.len(),
            result.instructions_reduced
        );
    }

    println!("\nPipeline complete!");
    Ok(())
}

fn create_test_module() -> Result<sounio::mir::MirModule, String> {
    let mut module_builder = ModuleBuilder::new("epistemic_example");

    // Create function with epistemic operations
    let mut func = module_builder.create_function("knowledge_computation".to_string(), MirType::I64);

    // Create knowledge construction with confidence
    let knowledge_const = func.build_i64(42);
    func.build_return(Some(knowledge_const));

    module_builder.add_function(func.build());

    // Create function with uncertainty handling
    let mut uncertain_func = module_builder.create_function("uncertain_computation".to_string(), MirType::F64);

    let result = uncertain_func.build_f64(3.14159);
    uncertain_func.build_return(Some(result));

    module_builder.add_function(uncertain_func.build());

    Ok(module_builder.build())
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_pipeline_integration() {
        let module = create_test_module().unwrap();
        assert!(!module.functions.is_empty());
    }

    #[test]
    fn test_ssa_validation() {
        let module = create_test_module().unwrap();
        let mut validator = SSAValidator::new();
        for func in &module.functions {
            // Validation should not panic
            let _ = validator.validate_function(func);
        }
    }
}
