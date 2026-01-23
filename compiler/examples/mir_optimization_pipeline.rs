//! MIR Optimization Pipeline Example
//!
//! This example demonstrates the complete MIR optimization pipeline
//! from source code to optimized executable code.

use sounio_compiler::mir::{MirModule, MirFunction};
use sounio_compiler::mir::builder::ModuleBuilder;
use sounio_compiler::mir::types::MirType;
use sounio_compiler::mir::instructions::{MirInstruction, MirConstant, MirBinaryOp};
use sounio_compiler::mir::optimization::{
    OptimizationLevel, create_default_pass_manager,
    ConstantPropagation, DeadCodeElimination, 
    CommonSubexpressionElimination, StrengthReduction,
    FunctionInlining, LoopInvariantCodeMotion
};
use sounio_compiler::codegen::mir_cranelift::MirAwareCraneliftJit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 MIR Optimization Pipeline Demo");
    println!("================================");
    
    // Step 1: Create a MIR module with optimizations potential
    let mut module = create_optimization_example_module()?;
    
    println!("📦 Created MIR module:");
    print_module_info(&module);
    
    // Step 2: Apply different optimization levels
    demonstrate_optimization_levels(&mut module)?;
    
    // Step 3: Compile to native code with optimizations
    compile_with_optimizations(&module)?;
    
    println!("✅ Pipeline demo completed successfully!");
    Ok(())
}

/// Create a MIR module that demonstrates optimization opportunities
fn create_optimization_example_module() -> Result<MirModule, String> {
    let mut module_builder = ModuleBuilder::new("optimization_demo");
    
    // Create a function with multiple optimization opportunities
    let mut func = module_builder.create_function("compute_heavy".to_string(), MirType::I64);
    
    // Function parameters
    func.add_param("x".to_string(), MirType::I64);
    func.add_param("y".to_string(), MirType::I64);
    
    // Opportunity 1: Constant propagation
    // Many constant operations that can be folded
    let const_10 = func.build_i64(10);
    let const_20 = func.build_i64(20);
    let const_30 = func.build_i64(30);
    
    // Complex expression that can be constant-folded
    let sum1 = func.fresh_value();
    func.build_add(sum1, const_10, const_20, MirType::I64);
    
    let sum2 = func.fresh_value();
    func.build_add(sum2, sum1, const_30, MirType::I64);
    
    // Opportunity 2: Common subexpression elimination
    // Repeated computations that can be eliminated
    let compute1 = func.fresh_value();
    func.build_mul(compute1, sum2, const_10, MirType::I64);
    
    let compute2 = func.fresh_value();
    func.build_mul(compute2, sum2, const_10, MirType::I64); // Same as compute1!
    
    // Opportunity 3: Strength reduction
    // Division by constant can be optimized
    let divisor = func.fresh_value();
    func.build_const(divisor, MirConstant::Int(2), MirType::I64);
    
    let result1 = func.fresh_value();
    func.build_div(result1, compute1, divisor, MirType::I64);
    
    let result2 = func.fresh_value();
    func.build_div(result2, compute2, divisor, MirType::I64);
    
    // Final computation
    let final_result = func.fresh_value();
    func.build_add(final_result, result1, result2, MirType::I64);
    
    func.build_return(Some(final_result));
    module_builder.add_function(func.build());
    
    // Create a small helper function for inlining demonstration
    let mut helper = module_builder.create_function("helper".to_string(), MirType::I64);
    helper.add_param("a".to_string(), MirType::I64);
    
    let helper_result = helper.fresh_value();
    helper.build_add(helper_result, helper.get_param(0), helper.get_param(0), MirType::I64);
    helper.build_return(Some(helper_result));
    module_builder.add_function(helper.build());
    
    Ok(module_builder.build())
}

/// Demonstrate different optimization levels
fn demonstrate_optimization_levels(module: &mut MirModule) -> Result<(), String> {
    println!("\n🔧 Demonstrating Optimization Levels:");
    println!("=====================================");
    
    let test_func = module.functions[0].clone();
    
    // O0: No optimization
    println!("\n📊 O0 (No optimization):");
    print_function_stats(&test_func);
    
    // O1: Basic optimizations
    println!("\n⚡ O1 (Basic optimizations):");
    let mut o1_func = test_func.clone();
    apply_optimization_level(&mut o1_func, OptimizationLevel::O1)?;
    print_function_stats(&o1_func);
    
    // O2: Aggressive optimizations
    println!("\n🚀 O2 (Aggressive optimizations):");
    let mut o2_func = test_func.clone();
    apply_optimization_level(&mut o2_func, OptimizationLevel::O2)?;
    print_function_stats(&o2_func);
    
    // O3: Maximum optimization
    println!("\n🔥 O3 (Maximum optimization):");
    let mut o3_func = test_func.clone();
    apply_optimization_level(&mut o3_func, OptimizationLevel::O3)?;
    print_function_stats(&o3_func);
    
    Ok(())
}

/// Apply optimization level to a function
fn apply_optimization_level(
    func: &mut MirFunction, 
    level: OptimizationLevel
) -> Result<(), String> {
    use std::time::Instant;
    
    let start = Instant::now();
    let mut pass_manager = create_default_pass_manager(level);
    
    let result = pass_manager.run_function_passes(func)?;
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Optimization time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  📝 Passes applied: {}", result.passes_applied.len());
    println!("  🗑️  Instructions reduced: {}", result.instructions_reduced);
    
    for pass_result in &result.passes_applied {
        println!("    - {}: {:.2}ms", pass_result.name, pass_result.time.as_secs_f64() * 1000.0);
    }
    
    Ok(())
}

/// Compile module with optimizations using Cranelift
fn compile_with_optimizations(module: &MirModule) -> Result<(), String> {
    println!("\n🎯 Compiling with Cranelift JIT:");
    println!("===============================");
    
    // Create JIT compiler with maximum optimization
    let jit = MirAwareCraneliftJit::new()
        .with_optimization()
        .with_mir_optimization(OptimizationLevel::O3);
    
    println!("  📦 Compiling MIR to native code...");
    let start = std::time::Instant::now();
    
    let compiled = jit.compile_mir(module)
        .map_err(|e| format!("Compilation failed: {}", e))?;
    
    let compile_time = start.elapsed();
    println!("  ✅ Compilation successful!");
    println!("  ⏱️  Compile time: {:.2}ms", compile_time.as_secs_f64() * 1000.0);
    
    // Try to execute the compiled function
    println!("  🚀 Executing compiled function...");
    
    // For this demo, we assume the function can be called
    // In a real scenario, you'd use the compiled function pointer
    
    println!("  💡 Function compiled and ready for execution");
    
    Ok(())
}

/// Print information about a MIR module
fn print_module_info(module: &MirModule) {
    println!("  📦 Module: {}", module.name);
    println!("  📋 Functions: {}", module.functions.len());
    
    for (i, func) in module.functions.iter().enumerate() {
        println!("    {}. {}: {} parameters, {} blocks", 
                 i + 1, func.name, func.params.len(), func.blocks.len());
        
        let total_instructions: usize = func.blocks.iter()
            .map(|block| block.instructions.len())
            .sum();
        println!("       Total instructions: {}", total_instructions);
    }
}

/// Print statistics for a function
fn print_function_stats(func: &MirFunction) {
    let total_instructions: usize = func.blocks.iter()
        .map(|block| block.instructions.len())
        .sum();
    
    let total_blocks = func.blocks.len();
    
    println!("  📊 Instructions: {}", total_instructions);
    println!("  📊 Blocks: {}", total_blocks);
    
    // Analyze instruction types
    let mut instruction_counts = std::collections::HashMap::new();
    
    for block in &func.blocks {
        for instruction in &block.instructions {
            let kind = match instruction {
                MirInstruction::Const { .. } => "Const",
                MirInstruction::Binary { .. } => "Binary",
                MirInstruction::Unary { .. } => "Unary",
                MirInstruction::Compare { .. } => "Compare",
                MirInstruction::Call { .. } => "Call",
                _ => "Other",
            };
            
            *instruction_counts.entry(kind).or_insert(0) += 1;
        }
    }
    
    println!("  📋 Instruction breakdown:");
    for (kind, count) in instruction_counts {
        println!("    - {}: {}", kind, count);
    }
}

/// Example of advanced optimization pipeline
fn demonstrate_advanced_optimizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Advanced Optimization Pipeline:");
    println!("===================================");
    
    // Create a more complex example with loops and function calls
    let mut module = create_advanced_example_module()?;
    
    // Create custom optimization pipeline
    let mut custom_manager = create_default_pass_manager(OptimizationLevel::O3);
    
    // Add additional passes
    custom_manager.add_pass(Box::new(LoopInvariantCodeMotion::new()));
    
    println!("  🎯 Running custom optimization pipeline...");
    
    for func in &mut module.functions {
        let result = custom_manager.run_function_passes(func)?;
        println!("  📊 {} optimizations applied", result.passes_applied.len());
    }
    
    print_module_info(&module);
    Ok(())
}

/// Create a more advanced example with loops and function calls
fn create_advanced_example_module() -> Result<MirModule, String> {
    let mut module_builder = ModuleBuilder::new("advanced_demo");
    
    // Function with loop (for optimization opportunities)
    let mut loop_func = module_builder.create_function("matrix_multiply".to_string(), MirType::Unit);
    
    // Create loop structure: for i in 0..n { for j in 0..n { sum += a[i] * b[j]; } }
    
    // Entry block
    let entry_block = loop_func.create_block();
    let i_loop = loop_func.create_block();
    let j_loop = loop_func.create_block();
    let body = loop_func.create_block();
    let exit = loop_func.create_block();
    
    // Set up blocks
    loop_func.set_block(entry_block);
    let const_n = loop_func.build_i64(100);
    let i_var = loop_func.fresh_value();
    loop_func.build_const(i_var, MirConstant::Int(0), MirType::I64);
    loop_func.build_branch(i_loop);
    
    loop_func.set_block(i_loop);
    let i_cmp = loop_func.fresh_value();
    loop_func.build_compare(i_cmp, i_var, const_n, MirType::I64);
    loop_func.build_cond_branch(i_cmp, j_loop, exit);
    
    loop_func.set_block(j_loop);
    let j_var = loop_func.fresh_value();
    loop_func.build_const(j_var, MirConstant::Int(0), MirType::I64);
    let j_cmp = loop_func.fresh_value();
    loop_func.build_compare(j_cmp, j_var, const_n, MirType::I64);
    loop_func.build_cond_branch(j_cmp, body, i_loop);
    
    loop_func.set_block(body);
    // Matrix computation (simplified)
    let temp = loop_func.fresh_value();
    loop_func.build_add(temp, i_var, j_var, MirType::I64);
    let j_inc = loop_func.fresh_value();
    let const_1 = loop_func.build_i64(1);
    loop_func.build_add(j_inc, j_var, const_1, MirType::I64);
    loop_func.build_branch(j_loop);
    
    loop_func.set_block(exit);
    loop_func.build_return(None);
    
    module_builder.add_function(loop_func.build());
    
    // Function with multiple calls (for inlining)
    let mut caller = module_builder.create_function("caller".to_string(), MirType::I64);
    
    // Multiple calls to helper function
    let mut accum = caller.build_i64(0);
    
    for i in 0..5 {
        let arg = caller.build_i64(i);
        let call_result = caller.fresh_value();
        caller.build_call(Some(call_result), "helper".to_string(), vec![arg], MirType::I64);
        
        accum = caller.fresh_value();
        caller.build_add(accum, accum, call_result, MirType::I64);
    }
    
    caller.build_return(Some(accum));
    module_builder.add_function(caller.build());
    
    Ok(module_builder.build())
}

/// Benchmark optimization performance
fn benchmark_optimizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Optimization Performance Benchmark:");
    println!("==================================");
    
    let iterations = 10;
    let mut total_time = std::time::Duration::ZERO;
    
    for i in 0..iterations {
        println!("  🔄 Iteration {}/{}", i + 1, iterations);
        
        let mut module = create_optimization_example_module()?;
        
        let start = std::time::Instant::now();
        let mut pass_manager = create_default_pass_manager(OptimizationLevel::O3);
        
        for func in &mut module.functions {
            let _ = pass_manager.run_function_passes(func)?;
        }
        
        let elapsed = start.elapsed();
        total_time += elapsed;
        
        println!("    ⏱️  {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    }
    
    let average = total_time / iterations as u32;
    println!("  📊 Average optimization time: {:.2}ms", average.as_secs_f64() * 1000.0);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimization_example_creation() {
        let module = create_optimization_example_module().unwrap();
        assert_eq!(module.functions.len(), 2);
        assert_eq!(module.functions[0].name, "compute_heavy");
    }
    
    #[test]
    fn test_advanced_example_creation() {
        let module = create_advanced_example_module().unwrap();
        assert_eq!(module.functions.len(), 2);
        assert_eq!(module.functions[0].name, "matrix_multiply");
    }
    
    #[test]
    fn test_optimization_levels() {
        let mut module = create_optimization_example_module().unwrap();
        let func = &mut module.functions[0];
        
        // Test O1 optimization
        let result = apply_optimization_level(func, OptimizationLevel::O1);
        assert!(result.is_ok());
    }
}

#[cfg(feature = "demo")]
fn run_interactive_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎮 Interactive Optimization Demo");
    println!("==============================");
    
    loop {
        println!("\nChoose optimization level:");
        println!("1. O0 (No optimization)");
        println!("2. O1 (Basic optimizations)");
        println!("3. O2 (Aggressive optimizations)");
        println!("4. O3 (Maximum optimization)");
        println!("5. Advanced optimizations");
        println!("6. Performance benchmark");
        println!("0. Exit");
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        let choice = input.trim().parse::<u32>()?;
        
        match choice {
            0 => break,
            1..=4 => {
                let level = match choice {
                    1 => OptimizationLevel::O0,
                    2 => OptimizationLevel::O1,
                    3 => OptimizationLevel::O2,
                    4 => OptimizationLevel::O3,
                    _ => unreachable!(),
                };
                
                let mut module = create_optimization_example_module()?;
                let func = &mut module.functions[0];
                apply_optimization_level(func, level)?;
            }
            5 => demonstrate_advanced_optimizations()?,
            6 => benchmark_optimizations()?,
            _ => println!("Invalid choice"),
        }
    }
    
    Ok(())
}
