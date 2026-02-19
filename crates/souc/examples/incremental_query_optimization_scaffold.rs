//! Incremental query-based optimization scaffold demo.
//!
//! Run with:
//! `cargo run -p souc --example incremental_query_optimization_scaffold`

use sounio::mir::builder::ModuleBuilder;
use sounio::mir::optimization::{
    IncrementalFunctionOptimizer, OptimizationLevel, PassManager,
};
use sounio::mir::types::MirType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut module_builder = ModuleBuilder::new("incremental_demo");
    let mut function_builder = module_builder.create_function("stable_fn".to_string(), MirType::I64);
    let value = function_builder.build_i64(42);
    function_builder.build_return(Some(value));
    module_builder.add_function(function_builder.build());
    let module = module_builder.build();
    let function = module
        .functions
        .first()
        .cloned()
        .ok_or("demo function missing")?;

    let mut incremental = IncrementalFunctionOptimizer::new();

    let mut pass_manager_first = PassManager::new_with_level(OptimizationLevel::O2);
    let first = incremental.optimize_function(&function, OptimizationLevel::O2, |candidate| {
        pass_manager_first.run_function_passes(candidate)
    })?;

    let mut pass_manager_second = PassManager::new_with_level(OptimizationLevel::O2);
    let second = incremental.optimize_function(&function, OptimizationLevel::O2, |candidate| {
        pass_manager_second.run_function_passes(candidate)
    })?;

    println!("Incremental Query Optimization Scaffold");
    println!("  first run cache_hit: {}", first.cache_hit);
    println!("  second run cache_hit: {}", second.cache_hit);
    println!(
        "  query hit rate: {:.1}%",
        incremental.hit_rate() * 100.0
    );
    println!(
        "  optimized function: {}",
        second.optimized_function.name
    );

    Ok(())
}
