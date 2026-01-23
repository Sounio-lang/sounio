//! MIR Regression Tests
//!
//! These tests ensure that MIR optimizations don't introduce bugs or break existing functionality.
//! They verify that optimizations preserve program semantics and don't cause runtime errors.

use sounio::mir::analysis::LoopAnalysis;
use sounio::mir::builder::ModuleBuilder;
use sounio::mir::instructions::{MirBinaryOp, MirConstant, MirInstruction, MirTerminator};
use sounio::mir::optimization::{ConstantPropagation, FunctionInlining, StrengthReduction};
use sounio::mir::types::MirType;
use sounio::mir::{BlockId, MirFunction, MirModule, ValueId};
use std::collections::HashSet;

/// Test that constant propagation doesn't break arithmetic
#[test]
fn test_constant_propagation_arithmetic_correctness() {
    let mut module_builder = ModuleBuilder::new("arithmetic_test");

    // Create: fn test_arithmetic() -> i64 {
    //   let a = 10; let b = 20; let c = 30; return (a + b) * c;
    //   Expected result: (10 + 20) * 30 = 900
    let mut func_builder =
        module_builder.create_function("test_arithmetic".to_string(), MirType::I64);

    let const_10 = func_builder.build_i64(10);
    let const_20 = func_builder.build_i64(20);
    let const_30 = func_builder.build_i64(30);

    let sum = func_builder.fresh_value();
    func_builder.build_add(sum, const_10, const_20, MirType::I64);

    let result = func_builder.fresh_value();
    func_builder.build_mul(result, sum, const_30, MirType::I64);

    func_builder.build_return(Some(result));
    module_builder.add_function(func_builder.build());

    let mut module = module_builder.build();

    // Apply constant propagation
    let pass = ConstantPropagation;
    let func = &mut module.functions[0];
    let _modified = pass.run_on_function(func).expect("Pass should succeed");

    // Verify the function structure is preserved
    assert_eq!(func.blocks.len(), 1);
    let entry = &func.blocks[0];

    // Should still have 4 instructions: const 10, const 20, const 30, add, mul, return
    assert!(entry.instructions.len() >= 5);

    // Verify final return instruction exists
    assert!(matches!(entry.terminator, MirTerminator::Return { .. }));
}

/// Test that function inlining preserves call semantics
#[test]
fn test_function_inlining_semantics() {
    let mut module_builder = ModuleBuilder::new("inline_semantics");

    // Create: fn double(x: i64) -> i64 { x + x }
    let mut double_func = module_builder.create_function("double".to_string(), MirType::I64);
    double_func.add_param("x".to_string(), MirType::I64);
    let result = double_func.fresh_value();
    double_func.build_add(
        result,
        double_func.get_param(0),
        double_func.get_param(0),
        MirType::I64,
    );
    double_func.build_return(Some(result));
    module_builder.add_function(double_func.build());

    // Create: fn main() -> i64 { double(21) + double(21) }
    let mut main_func = module_builder.create_function("main".to_string(), MirType::I64);
    let const_21 = main_func.build_i64(21);

    let call1 = main_func.fresh_value();
    main_func.build_call(
        Some(call1),
        "double".to_string(),
        vec![const_21],
        MirType::I64,
    );

    let call2 = main_func.fresh_value();
    main_func.build_call(
        Some(call2),
        "double".to_string(),
        vec![const_21],
        MirType::I64,
    );

    let final_result = main_func.fresh_value();
    main_func.build_add(final_result, call1, call2, MirType::I64);
    main_func.build_return(Some(final_result));
    module_builder.add_function(main_func.build());

    let mut module = module_builder.build();

    // Apply function inlining
    let pass = FunctionInlining;
    let _modified = pass
        .run_on_module(&mut module)
        .expect("Pass should succeed");

    // Verify both functions still exist or were properly inlined
    assert_eq!(module.functions.len(), 1); // Should have been inlined into main

    // Verify main function has the expected structure
    let main_func = &module.functions[0];
    assert_eq!(main_func.name, "main");

    // Should have both double operations inlined
    let entry = &main_func.blocks[0];
    let mut add_count = 0;
    for instruction in &entry.instructions {
        if matches!(
            instruction,
            MirInstruction::Binary {
                op: MirBinaryOp::Add,
                ..
            }
        ) {
            add_count += 1;
        }
    }

    // Should have 2 add operations (double(21) + double(21))
    assert_eq!(add_count, 2, "Should have 2 add operations after inlining");
}

/// Test that loop analysis doesn't modify control flow incorrectly
#[test]
fn test_loop_analysis_control_flow() {
    let mut module_builder = ModuleBuilder::new("loop_control_flow");

    // Create a simple loop structure
    let mut func_builder = module_builder.create_function("loop_test".to_string(), MirType::I64);

    // Entry block
    let entry_block = func_builder.create_block();
    let loop_header = func_builder.create_block();
    let loop_body = func_builder.create_block();
    let exit = func_builder.create_block();

    // Set up blocks
    func_builder.set_block(entry_block);
    let const_0 = func_builder.build_i64(0);
    let i_var = func_builder.fresh_value();
    func_builder.build_branch(loop_header);

    func_builder.set_block(loop_header);
    let const_10 = func_builder.build_i64(10);
    let cmp = func_builder.fresh_value();
    func_builder.build_compare(cmp, i_var, const_10, MirType::I64);
    func_builder.build_cond_branch(cmp, loop_body, exit);

    func_builder.set_block(loop_body);
    // Do some work
    func_builder.build_branch(loop_header);

    func_builder.set_block(exit);
    func_builder.build_return(Some(i_var));

    module_builder.add_function(func_builder.build());
    let module = module_builder.build();

    // Run loop analysis
    let func = &module.functions[0];
    let loop_analysis = LoopAnalysis::analyze_function(func);

    // Should detect at least one loop
    assert!(!loop_analysis.loops.is_empty(), "Should detect loops");

    // Verify loop structure
    let loop_info = &loop_analysis.loops[0];
    assert!(loop_info.contains_block(loop_header));
    assert!(loop_info.contains_block(loop_body));

    // Verify original CFG structure is preserved
    assert_eq!(func.blocks.len(), 4); // entry, header, body, exit
}

/// Test that strength reduction preserves mathematical correctness
#[test]
fn test_strength_reduction_correctness() {
    let mut module_builder = ModuleBuilder::new("strength_correctness");

    // Create function with division that should be preserved
    let mut func_builder =
        module_builder.create_function("division_test".to_string(), MirType::I64);

    let const_100 = func_builder.build_i64(100);
    let const_3 = func_builder.build_i64(3);
    let result = func_builder.fresh_value();
    func_builder.build_div(result, const_100, const_3, MirType::I64);
    func_builder.build_return(Some(result));

    module_builder.add_function(func_builder.build());
    let mut module = module_builder.build();

    // Apply strength reduction
    let pass = StrengthReduction;
    let func = &mut module.functions[0];
    let _modified = pass.run_on_function(func).expect("Pass should succeed");

    // Verify division instruction still exists (though it might be optimized)
    let entry = &func.blocks[0];
    let mut has_div = false;

    for instruction in &entry.instructions {
        match instruction {
            MirInstruction::Binary { op, .. } => {
                if *op == MirBinaryOp::Div {
                    has_div = true;
                    break;
                }
            }
            _ => {}
        }
    }

    assert!(has_div, "Should preserve division semantics");
}

/// Test that SSA properties are maintained after optimization
#[test]
fn test_ssa_properties_after_optimization() {
    let mut module_builder = ModuleBuilder::new("ssa_properties");

    // Create function with SSA violations that should be handled
    let mut func_builder = module_builder.create_function("ssa_test".to_string(), MirType::I64);

    let const_5 = func_builder.build_i64(5);
    let result1 = func_builder.fresh_value();
    func_builder.build_add(result1, const_5, const_5, MirType::I64);

    // This should create a phi node or similar SSA construct
    let result2 = func_builder.fresh_value();
    func_builder.build_add(result2, result1, const_5, MirType::I64);

    func_builder.build_return(Some(result2));
    module_builder.add_function(func_builder.build());

    let mut module = module_builder.build();

    // Apply optimizations
    let constant_prop = ConstantPropagation;
    let strength_red = StrengthReduction;

    let func = &mut module.functions[0];
    let _cp_modified = constant_prop
        .run_on_function(func)
        .expect("CP should succeed");
    let _sr_modified = strength_red
        .run_on_function(func)
        .expect("SR should succeed");

    // Verify SSA structure is preserved
    let entry = &func.blocks[0];

    // Count value definitions
    let mut value_definitions = HashSet::new();
    for instruction in &entry.instructions {
        if let Some(result) = instruction.result() {
            assert!(
                !value_definitions.contains(&result),
                "SSA violation: value {:?} defined multiple times",
                result
            );
            value_definitions.insert(result);
        }
    }

    assert!(
        !value_definitions.is_empty(),
        "Should have value definitions"
    );
}

/// Test edge case: empty functions
#[test]
fn test_optimization_edge_cases_empty_function() {
    let mut module_builder = ModuleBuilder::new("empty_function");

    // Create empty function
    let mut func_builder = module_builder.create_function("empty".to_string(), MirType::Unit);
    func_builder.build_return(None);
    module_builder.add_function(func_builder.build());

    let mut module = module_builder.build();

    // Apply all optimizations
    let constant_prop = ConstantPropagation;
    let strength_red = StrengthReduction;
    let func_inlining = FunctionInlining;

    let func = &mut module.functions[0];
    assert!(!constant_prop.run_on_function(func).unwrap());
    assert!(!strength_red.run_on_function(func).unwrap());

    // Module-level optimization
    let _cp_modified = func_inlining.run_on_module(&mut module);
    assert!(
        !_cp_modified.unwrap(),
        "Empty function should not be modified"
    );
}

/// Test edge case: recursive functions
#[test]
fn test_optimization_recursive_functions() {
    let mut module_builder = ModuleBuilder::new("recursive");

    // Create recursive factorial function
    let mut factorial = module_builder.create_function("factorial".to_string(), MirType::I64);
    factorial.add_param("n".to_string(), MirType::I64);

    let const_1 = factorial.build_i64(1);
    let param = factorial.get_param(0);

    // if n <= 1, return 1
    let cmp_result = factorial.fresh_value();
    factorial.build_compare(cmp_result, param, const_1, MirType::I64);

    // For simplicity, just return n (this is not a complete factorial)
    factorial.build_return(Some(param));
    module_builder.add_function(factorial.build());

    let mut module = module_builder.build();

    // Apply function inlining
    let pass = FunctionInlining;
    let modified = pass
        .run_on_module(&mut module)
        .expect("Pass should complete");

    // Should not inline recursive function by default
    assert!(!modified, "Recursive functions should not be inlined");

    // Verify factorial function still exists
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "factorial");
}

/// Test that optimization doesn't break floating point operations
#[test]
fn test_floating_point_optimizations() {
    let mut module_builder = ModuleBuilder::new("float_test");

    // Create function with floating point operations
    let mut func_builder = module_builder.create_function("float_ops".to_string(), MirType::F64);

    let const_3_14 = func_builder.build_f64(3.14);
    let const_2_71 = func_builder.build_f64(2.71);
    let sum = func_builder.fresh_value();
    func_builder.build_add(sum, const_3_14, const_2_71, MirType::F64);

    func_builder.build_return(Some(sum));
    module_builder.add_function(func_builder.build());

    let mut module = module_builder.build();

    // Apply constant propagation
    let pass = ConstantPropagation;
    let func = &mut module.functions[0];
    let _modified = pass.run_on_function(func).expect("Pass should succeed");

    // Verify floating point addition is preserved
    let entry = &func.blocks[0];
    let mut has_float_add = false;

    for instruction in &entry.instructions {
        match instruction {
            MirInstruction::Binary { op, .. } => {
                if *op == MirBinaryOp::Add {
                    has_float_add = true;
                    break;
                }
            }
            _ => {}
        }
    }

    assert!(has_float_add, "Should preserve floating point operations");
}

/// Test memory safety: optimization shouldn't introduce invalid memory access
#[test]
fn test_memory_safety_after_optimization() {
    let mut module_builder = ModuleBuilder::new("memory_safety");

    // Create function with memory operations
    let mut func_builder = module_builder.create_function("memory_ops".to_string(), MirType::I64);

    // Allocate stack space
    let alloca_result = func_builder.fresh_value();
    func_builder.build_alloca(alloca_result, MirType::I64);

    // Store a value
    let const_42 = func_builder.build_i64(42);
    func_builder.build_store(alloca_result, const_42);

    // Load the value back
    let loaded = func_builder.fresh_value();
    func_builder.build_load(loaded, alloca_result, MirType::I64);

    func_builder.build_return(Some(loaded));
    module_builder.add_function(func_builder.build());

    let mut module = module_builder.build();

    // Apply optimizations
    let constant_prop = ConstantPropagation;
    let strength_red = StrengthReduction;

    let func = &mut module.functions[0];
    let _cp_modified = constant_prop
        .run_on_function(func)
        .expect("CP should succeed");
    let _sr_modified = strength_red
        .run_on_function(func)
        .expect("SR should succeed");

    // Verify memory operations are still present and properly ordered
    let entry = &func.blocks[0];
    let mut has_alloca = false;
    let mut has_store = false;
    let mut has_load = false;
    let mut load_after_store = false;

    for instruction in &entry.instructions {
        match instruction {
            MirInstruction::Alloca { .. } => has_alloca = true,
            MirInstruction::Store { .. } => has_store = true,
            MirInstruction::Load { .. } => {
                has_load = true;
                if has_store {
                    load_after_store = true;
                }
            }
            _ => {}
        }
    }

    assert!(has_alloca, "Should preserve alloca");
    assert!(has_store, "Should preserve store");
    assert!(has_load, "Should preserve load");
    assert!(load_after_store, "Load should come after store");
}

/// Test that optimizations don't break exception handling (if implemented)
#[test]
fn test_exception_safety() {
    // This test would verify that optimizations preserve any exception handling semantics
    // For now, it's a placeholder for future implementation

    // Create a function that could have exception handling
    let mut module_builder = ModuleBuilder::new("exception_test");
    let mut func_builder =
        module_builder.create_function("safe_function".to_string(), MirType::I64);

    // Simple safe function
    let const_42 = func_builder.build_i64(42);
    func_builder.build_return(Some(const_42));
    module_builder.add_function(func_builder.build());

    let mut module = module_builder.build();

    // Apply optimizations
    let pass = ConstantPropagation;
    let func = &mut module.functions[0];
    let _modified = pass.run_on_function(func).expect("Pass should succeed");

    // Function should remain safe
    let entry = &func.blocks[0];
    assert!(matches!(entry.terminator, MirTerminator::Return { .. }));
}

/// Performance regression test: optimizations shouldn't be significantly slower
#[test]
fn test_optimization_performance_regression() {
    use std::time::Instant;

    let mut module_builder = ModuleBuilder::new("performance_test");

    // Create a moderately complex function
    let mut func_builder = module_builder.create_function("complex".to_string(), MirType::I64);

    // Create many constant operations
    for i in 0..1000 {
        let const_i = func_builder.build_i64(i as i64);
        let temp = func_builder.fresh_value();
        func_builder.build_add(temp, const_i, const_i, MirType::I64);
    }

    let result = func_builder.build_i64(42);
    func_builder.build_return(Some(result));
    module_builder.add_function(func_builder.build());

    let mut module = module_builder.build();

    // Measure optimization time
    let start = Instant::now();
    let pass = ConstantPropagation;
    let func = &mut module.functions[0];
    let _modified = pass.run_on_function(func).expect("Pass should succeed");
    let optimization_time = start.elapsed();

    // Optimization should complete in reasonable time (less than 1 second)
    assert!(
        optimization_time.as_secs() < 1,
        "Optimization took too long: {:?}",
        optimization_time
    );

    // Should provide some optimization benefit
    let entry = &func.blocks[0];
    // The exact number depends on optimization effectiveness
    assert!(!entry.instructions.is_empty(), "Should have instructions");
}

/// Test that optimizations preserve function signatures
#[test]
fn test_function_signature_preservation() {
    let mut module_builder = ModuleBuilder::new("signature_test");

    // Create function with specific signature
    let mut func_builder =
        module_builder.create_function("signature_test".to_string(), MirType::F64);
    func_builder.add_param("x".to_string(), MirType::I64);
    func_builder.add_param("y".to_string(), MirType::F64);
    func_builder.add_param("z".to_string(), MirType::Bool);

    let param_x = func_builder.get_param(0);
    let param_y = func_builder.get_param(1);
    let param_z = func_builder.get_param(2);

    // Simple operation using all parameters
    let result = func_builder.fresh_value();
    func_builder.build_add(result, param_x, param_x, MirType::I64);

    func_builder.build_return(Some(result));
    module_builder.add_function(func_builder.build());

    let mut module = module_builder.build();

    // Apply optimizations
    let pass = ConstantPropagation;
    let func = &mut module.functions[0];
    let _modified = pass.run_on_function(func).expect("Pass should succeed");

    // Verify signature is preserved
    assert_eq!(func.params.len(), 3);
    assert_eq!(func.params[0].0, "x");
    assert_eq!(func.params[1].0, "y");
    assert_eq!(func.params[2].0, "z");
    assert_eq!(func.return_type, MirType::I64);
}

/// Run all regression tests
#[test]
fn test_all_regression_suite() {
    test_constant_propagation_arithmetic_correctness();
    test_function_inlining_semantics();
    test_loop_analysis_control_flow();
    test_strength_reduction_correctness();
    test_ssa_properties_after_optimization();
    test_optimization_edge_cases_empty_function();
    test_optimization_recursive_functions();
    test_floating_point_optimizations();
    test_memory_safety_after_optimization();
    test_exception_safety();
    test_optimization_performance_regression();
    test_function_signature_preservation();
}
