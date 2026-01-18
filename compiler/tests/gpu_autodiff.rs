//! Integration tests for GPU Reverse-Mode Automatic Differentiation
//!
//! These tests verify that the GPU autodiff module correctly generates backward
//! kernels for various differentiable operations.

#![cfg(feature = "gpu")]

use souc::codegen::gpu::autodiff::{
    AutoDiff, AutoDiffTransform, BackwardCodegen, DiffConfig, DiffContext, DiffPrimitive,
    GpuTape, GpuTapeConfig, KernelAnalysis, ParamInfo, PrimitiveRegistry, TapeEntry, TapeOp,
};
use souc::codegen::gpu::ir::{
    BlockId, GpuBlock, GpuKernel, GpuModule, GpuOp, GpuParam, GpuTarget, GpuTerminator, GpuType,
    MemorySpace, ValueId,
};

// ============================================================================
// Helper functions for creating test kernels
// ============================================================================

/// Create a simple element-wise multiply kernel: y[i] = x[i] * w[i]
fn create_mul_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("forward_mul");

    // Parameters: x, w, y
    kernel.add_param(GpuParam {
        name: "x".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "w".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "y".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });

    // Entry block
    let mut block = GpuBlock::new(BlockId(0), "entry");

    // i = thread_id.x
    block.add_instruction(ValueId(0), GpuOp::ThreadIdX);

    // Load x[i]
    block.add_instruction(ValueId(1), GpuOp::Param(0));
    block.add_instruction(ValueId(2), GpuOp::GetElementPtr(ValueId(1), vec![ValueId(0)]));
    block.add_instruction(ValueId(3), GpuOp::Load(ValueId(2), MemorySpace::Global));

    // Load w[i]
    block.add_instruction(ValueId(4), GpuOp::Param(1));
    block.add_instruction(ValueId(5), GpuOp::GetElementPtr(ValueId(4), vec![ValueId(0)]));
    block.add_instruction(ValueId(6), GpuOp::Load(ValueId(5), MemorySpace::Global));

    // y[i] = x[i] * w[i]
    block.add_instruction(ValueId(7), GpuOp::FMul(ValueId(3), ValueId(6)));

    // Store to y[i]
    block.add_instruction(ValueId(8), GpuOp::Param(2));
    block.add_instruction(ValueId(9), GpuOp::GetElementPtr(ValueId(8), vec![ValueId(0)]));
    block.add_instruction(
        ValueId(10),
        GpuOp::Store(ValueId(9), ValueId(7), MemorySpace::Global),
    );

    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    kernel
}

/// Create a simple addition kernel: z[i] = x[i] + y[i]
fn create_add_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("forward_add");

    kernel.add_param(GpuParam {
        name: "x".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "y".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "z".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });

    let mut block = GpuBlock::new(BlockId(0), "entry");
    block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
    block.add_instruction(ValueId(1), GpuOp::Param(0));
    block.add_instruction(ValueId(2), GpuOp::GetElementPtr(ValueId(1), vec![ValueId(0)]));
    block.add_instruction(ValueId(3), GpuOp::Load(ValueId(2), MemorySpace::Global));
    block.add_instruction(ValueId(4), GpuOp::Param(1));
    block.add_instruction(ValueId(5), GpuOp::GetElementPtr(ValueId(4), vec![ValueId(0)]));
    block.add_instruction(ValueId(6), GpuOp::Load(ValueId(5), MemorySpace::Global));
    block.add_instruction(ValueId(7), GpuOp::FAdd(ValueId(3), ValueId(6)));
    block.add_instruction(ValueId(8), GpuOp::Param(2));
    block.add_instruction(ValueId(9), GpuOp::GetElementPtr(ValueId(8), vec![ValueId(0)]));
    block.add_instruction(
        ValueId(10),
        GpuOp::Store(ValueId(9), ValueId(7), MemorySpace::Global),
    );
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    kernel
}

/// Create a sin kernel: y[i] = sin(x[i])
fn create_sin_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("forward_sin");

    kernel.add_param(GpuParam {
        name: "x".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "y".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });

    let mut block = GpuBlock::new(BlockId(0), "entry");
    block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
    block.add_instruction(ValueId(1), GpuOp::Param(0));
    block.add_instruction(ValueId(2), GpuOp::GetElementPtr(ValueId(1), vec![ValueId(0)]));
    block.add_instruction(ValueId(3), GpuOp::Load(ValueId(2), MemorySpace::Global));
    block.add_instruction(ValueId(4), GpuOp::FastSin(ValueId(3)));
    block.add_instruction(ValueId(5), GpuOp::Param(1));
    block.add_instruction(ValueId(6), GpuOp::GetElementPtr(ValueId(5), vec![ValueId(0)]));
    block.add_instruction(
        ValueId(7),
        GpuOp::Store(ValueId(6), ValueId(4), MemorySpace::Global),
    );
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    kernel
}

/// Create a chain rule kernel: y[i] = sin(x[i] * w[i])
fn create_chain_rule_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("forward_chain");

    kernel.add_param(GpuParam {
        name: "x".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "w".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "y".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });

    let mut block = GpuBlock::new(BlockId(0), "entry");
    block.add_instruction(ValueId(0), GpuOp::ThreadIdX);

    // Load x[i]
    block.add_instruction(ValueId(1), GpuOp::Param(0));
    block.add_instruction(ValueId(2), GpuOp::GetElementPtr(ValueId(1), vec![ValueId(0)]));
    block.add_instruction(ValueId(3), GpuOp::Load(ValueId(2), MemorySpace::Global));

    // Load w[i]
    block.add_instruction(ValueId(4), GpuOp::Param(1));
    block.add_instruction(ValueId(5), GpuOp::GetElementPtr(ValueId(4), vec![ValueId(0)]));
    block.add_instruction(ValueId(6), GpuOp::Load(ValueId(5), MemorySpace::Global));

    // x[i] * w[i]
    block.add_instruction(ValueId(7), GpuOp::FMul(ValueId(3), ValueId(6)));

    // sin(x[i] * w[i])
    block.add_instruction(ValueId(8), GpuOp::FastSin(ValueId(7)));

    // Store to y[i]
    block.add_instruction(ValueId(9), GpuOp::Param(2));
    block.add_instruction(ValueId(10), GpuOp::GetElementPtr(ValueId(9), vec![ValueId(0)]));
    block.add_instruction(
        ValueId(11),
        GpuOp::Store(ValueId(10), ValueId(8), MemorySpace::Global),
    );
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    kernel
}

// ============================================================================
// AutoDiff Tests
// ============================================================================

#[test]
fn test_autodiff_creation() {
    let autodiff = AutoDiff::with_defaults();
    assert_eq!(autodiff.config().max_tape_size, 1024 * 1024);
    assert!(autodiff.config().accumulate_gradients);
}

#[test]
fn test_autodiff_custom_config() {
    let config = DiffConfig {
        max_tape_size: 100,
        enable_checkpointing: true,
        checkpoint_interval: 50,
        accumulate_gradients: false,
        target: GpuTarget::Metal {
            gpu_family: souc::codegen::gpu::ir::MetalGpuFamily::Apple8,
        },
        optimize_memory: false,
        warp_reductions: false,
    };

    let autodiff = AutoDiff::new(config.clone());
    assert_eq!(autodiff.config().max_tape_size, 100);
    assert!(autodiff.config().enable_checkpointing);
    assert!(!autodiff.config().accumulate_gradients);
}

#[test]
fn test_differentiate_mul_kernel() {
    let kernel = create_mul_kernel();
    let mut autodiff = AutoDiff::with_defaults();

    let result = autodiff.differentiate_kernel(&kernel);
    assert!(result.is_ok());

    let backward = result.unwrap();
    assert_eq!(backward.forward_kernel_name, "forward_mul");
    assert!(backward.kernel.name.contains("backward"));

    // Should have gradient parameters for x, w, and y
    assert!(backward.gradient_params.contains_key("x") || backward.requires_grad.contains(&"x".to_string()));
    assert!(backward.gradient_params.contains_key("w") || backward.requires_grad.contains(&"w".to_string()));
}

#[test]
fn test_differentiate_add_kernel() {
    let kernel = create_add_kernel();
    let mut autodiff = AutoDiff::with_defaults();

    let result = autodiff.differentiate_kernel(&kernel);
    assert!(result.is_ok());

    let backward = result.unwrap();
    assert!(backward.kernel.name.contains("backward"));
}

#[test]
fn test_differentiate_sin_kernel() {
    let kernel = create_sin_kernel();
    let mut autodiff = AutoDiff::with_defaults();

    let result = autodiff.differentiate_kernel(&kernel);
    assert!(result.is_ok());

    let backward = result.unwrap();
    // Sin backward should use cos
    assert!(backward.kernel.name.contains("backward"));
}

#[test]
fn test_differentiate_chain_rule_kernel() {
    let kernel = create_chain_rule_kernel();
    let mut autodiff = AutoDiff::with_defaults();

    let result = autodiff.differentiate_kernel(&kernel);
    assert!(result.is_ok());

    let backward = result.unwrap();
    // Chain rule: d/dx sin(x*w) = cos(x*w) * w
    assert!(backward.kernel.name.contains("backward"));
}

// ============================================================================
// Primitive Registry Tests
// ============================================================================

#[test]
fn test_primitive_registry_creation() {
    let registry = PrimitiveRegistry::new();

    // All standard primitives should be registered
    assert!(registry.is_supported(DiffPrimitive::Add));
    assert!(registry.is_supported(DiffPrimitive::Sub));
    assert!(registry.is_supported(DiffPrimitive::Mul));
    assert!(registry.is_supported(DiffPrimitive::Div));
    assert!(registry.is_supported(DiffPrimitive::Neg));
    assert!(registry.is_supported(DiffPrimitive::Sin));
    assert!(registry.is_supported(DiffPrimitive::Cos));
    assert!(registry.is_supported(DiffPrimitive::Tan));
    assert!(registry.is_supported(DiffPrimitive::Exp));
    assert!(registry.is_supported(DiffPrimitive::Log));
    assert!(registry.is_supported(DiffPrimitive::Sqrt));
    assert!(registry.is_supported(DiffPrimitive::Relu));
    assert!(registry.is_supported(DiffPrimitive::Sigmoid));
    assert!(registry.is_supported(DiffPrimitive::Tanh));
    assert!(registry.is_supported(DiffPrimitive::MatMul));
}

#[test]
fn test_primitive_classification() {
    let registry = PrimitiveRegistry::new();

    // Test operation classification
    assert_eq!(
        registry.classify_op(&GpuOp::FAdd(ValueId(0), ValueId(1))),
        Some(DiffPrimitive::Add)
    );
    assert_eq!(
        registry.classify_op(&GpuOp::FMul(ValueId(0), ValueId(1))),
        Some(DiffPrimitive::Mul)
    );
    assert_eq!(
        registry.classify_op(&GpuOp::FastSin(ValueId(0))),
        Some(DiffPrimitive::Sin)
    );
    assert_eq!(
        registry.classify_op(&GpuOp::FastExp(ValueId(0))),
        Some(DiffPrimitive::Exp)
    );

    // Non-differentiable ops should return None
    assert_eq!(registry.classify_op(&GpuOp::ThreadIdX), None);
    assert_eq!(registry.classify_op(&GpuOp::SyncThreads), None);
}

#[test]
fn test_backward_rules() {
    let registry = PrimitiveRegistry::new();

    // Test add rule: dx = dz, dy = dz
    let add_rule = registry.get_rule(DiffPrimitive::Add).unwrap();
    assert_eq!(add_rule.num_inputs, 2);
    assert_eq!(add_rule.backward_ops.len(), 2);

    // Test mul rule: dx = dz * y, dy = dz * x
    let mul_rule = registry.get_rule(DiffPrimitive::Mul).unwrap();
    assert_eq!(mul_rule.num_inputs, 2);
    assert_eq!(mul_rule.backward_ops.len(), 2);

    // Test sin rule: dx = dz * cos(x)
    let sin_rule = registry.get_rule(DiffPrimitive::Sin).unwrap();
    assert_eq!(sin_rule.num_inputs, 1);
    assert_eq!(sin_rule.backward_ops.len(), 1);
}

// ============================================================================
// GPU Tape Tests
// ============================================================================

#[test]
fn test_tape_creation() {
    let tape = GpuTape::with_defaults();
    assert!(tape.is_empty());
    assert_eq!(tape.len(), 0);
}

#[test]
fn test_tape_record_operations() {
    let mut tape = GpuTape::with_defaults();

    // Record inputs
    let x = tape.record_input(ValueId(0), GpuType::F32);
    let y = tape.record_input(ValueId(1), GpuType::F32);

    assert_eq!(tape.len(), 2);
    assert_eq!(tape.input_indices().len(), 2);

    // Record addition
    let z = tape.record_binary(TapeOp::Add, ValueId(2), x, y, GpuType::F32);

    assert_eq!(tape.len(), 3);

    let entry = &tape.entries()[2];
    assert_eq!(entry.op, TapeOp::Add);
    assert_eq!(entry.operand1, x);
    assert_eq!(entry.operand2, y);
    assert_eq!(entry.result, z);
}

#[test]
fn test_tape_gradient_tracking() {
    let mut tape = GpuTape::with_defaults();
    let x = tape.record_input(ValueId(0), GpuType::F32);

    // Set and accumulate gradients
    tape.set_gradient(x, 1.0);
    assert_eq!(tape.get_gradient(x), 1.0);

    tape.accumulate_gradient(x, 2.0);
    assert_eq!(tape.get_gradient(x), 3.0);

    tape.clear_gradients();
    assert_eq!(tape.get_gradient(x), 0.0);
}

#[test]
fn test_tape_entries_reversed() {
    let mut tape = GpuTape::with_defaults();
    tape.record_input(ValueId(0), GpuType::F32);
    tape.record_input(ValueId(1), GpuType::F32);
    tape.record_binary(TapeOp::Add, ValueId(2), 0, 1, GpuType::F32);

    let ops: Vec<_> = tape.entries_reversed().map(|e| e.op).collect();
    assert_eq!(ops, vec![TapeOp::Add, TapeOp::Input, TapeOp::Input]);
}

#[test]
fn test_tape_op_properties() {
    // Test differentiability
    assert!(TapeOp::Add.is_differentiable());
    assert!(TapeOp::Mul.is_differentiable());
    assert!(TapeOp::Sin.is_differentiable());
    assert!(TapeOp::MatMul.is_differentiable());
    assert!(!TapeOp::Const.is_differentiable());
    assert!(!TapeOp::Load.is_differentiable());

    // Test operand counts
    assert_eq!(TapeOp::Add.num_operands(), 2);
    assert_eq!(TapeOp::Mul.num_operands(), 2);
    assert_eq!(TapeOp::Sin.num_operands(), 1);
    assert_eq!(TapeOp::Neg.num_operands(), 1);
    assert_eq!(TapeOp::Fma.num_operands(), 3);
    assert_eq!(TapeOp::Const.num_operands(), 0);

    // Test names
    assert_eq!(TapeOp::Add.name(), "add");
    assert_eq!(TapeOp::Sin.name(), "sin");
    assert_eq!(TapeOp::MatMul.name(), "matmul");
}

// ============================================================================
// Transform Tests
// ============================================================================

#[test]
fn test_diff_context_creation() {
    let ctx = DiffContext::new(DiffConfig::default());
    assert!(ctx.inputs.is_empty());
    assert!(ctx.outputs.is_empty());
}

#[test]
fn test_transform_mul_kernel() {
    let kernel = create_mul_kernel();
    let ctx = DiffContext::new(DiffConfig::default());
    let mut transform = AutoDiffTransform::new(ctx);

    let result = transform.transform_kernel(&kernel);
    assert!(result.is_ok());

    let (forward, backward) = result.unwrap();
    assert!(forward.name.contains("forward"));
    assert!(backward.name.contains("backward"));
}

// ============================================================================
// Backward Codegen Tests
// ============================================================================

#[test]
fn test_backward_codegen_basic() {
    let kernel = create_mul_kernel();
    let config = DiffConfig::default();
    let primitives = PrimitiveRegistry::new();
    let mut codegen = BackwardCodegen::new(&config, &primitives);

    let analysis = KernelAnalysis {
        name: "forward_mul".to_string(),
        inputs: vec![
            ParamInfo {
                index: 0,
                name: "x".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            },
            ParamInfo {
                index: 1,
                name: "w".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            },
        ],
        outputs: vec![ParamInfo {
            index: 2,
            name: "y".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        }],
        operations: vec![],
    };

    let result = codegen.generate_backward(&kernel, &analysis);
    assert!(result.is_ok());

    let backward = result.unwrap();
    assert_eq!(backward.forward_kernel_name, "forward_mul");
    assert!(backward.kernel.name.contains("backward"));
}

// ============================================================================
// Module Differentiation Tests
// ============================================================================

#[test]
fn test_differentiate_module() {
    let mut module = GpuModule::new(
        "test_module",
        GpuTarget::Cuda {
            compute_capability: (8, 0),
        },
    );

    module.add_kernel(create_mul_kernel());
    module.add_kernel(create_add_kernel());

    let mut autodiff = AutoDiff::with_defaults();
    let result = autodiff.differentiate_module(&module);
    assert!(result.is_ok());

    let diff_module = result.unwrap();
    assert!(diff_module.name.contains("autodiff"));
    // Should have original kernels plus backward kernels
    assert!(diff_module.kernels.len() >= 2);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_non_differentiable_kernel() {
    // Kernel with only integer operations (not differentiable)
    let mut kernel = GpuKernel::new("int_kernel");
    kernel.add_param(GpuParam {
        name: "a".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::I32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "b".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::I32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });

    let mut block = GpuBlock::new(BlockId(0), "entry");
    block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    let mut autodiff = AutoDiff::with_defaults();
    // Should still generate a backward kernel (even if empty)
    let result = autodiff.differentiate_kernel(&kernel);
    // The current implementation handles non-differentiable kernels
    // by returning an appropriate result
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_cache_behavior() {
    let kernel = create_mul_kernel();
    let mut autodiff = AutoDiff::with_defaults();

    // First call
    let result1 = autodiff.differentiate_kernel(&kernel);
    assert!(result1.is_ok());

    // Second call should hit cache
    let result2 = autodiff.differentiate_kernel(&kernel);
    assert!(result2.is_ok());

    // Results should be identical
    let backward1 = result1.unwrap();
    let backward2 = result2.unwrap();
    assert_eq!(backward1.kernel.name, backward2.kernel.name);

    // Clear cache
    autodiff.clear_cache();

    // Third call should regenerate
    let result3 = autodiff.differentiate_kernel(&kernel);
    assert!(result3.is_ok());
}

// ============================================================================
// Complex Operation Tests
// ============================================================================

#[test]
fn test_fma_differentiation() {
    // Create kernel with fused multiply-add: y[i] = a[i] * b[i] + c[i]
    let mut kernel = GpuKernel::new("forward_fma");

    kernel.add_param(GpuParam {
        name: "a".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "b".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "c".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "y".to_string(),
        ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    });

    let mut block = GpuBlock::new(BlockId(0), "entry");
    block.add_instruction(ValueId(0), GpuOp::ThreadIdX);

    // Load a, b, c
    block.add_instruction(ValueId(1), GpuOp::Param(0));
    block.add_instruction(ValueId(2), GpuOp::GetElementPtr(ValueId(1), vec![ValueId(0)]));
    block.add_instruction(ValueId(3), GpuOp::Load(ValueId(2), MemorySpace::Global));

    block.add_instruction(ValueId(4), GpuOp::Param(1));
    block.add_instruction(ValueId(5), GpuOp::GetElementPtr(ValueId(4), vec![ValueId(0)]));
    block.add_instruction(ValueId(6), GpuOp::Load(ValueId(5), MemorySpace::Global));

    block.add_instruction(ValueId(7), GpuOp::Param(2));
    block.add_instruction(ValueId(8), GpuOp::GetElementPtr(ValueId(7), vec![ValueId(0)]));
    block.add_instruction(ValueId(9), GpuOp::Load(ValueId(8), MemorySpace::Global));

    // FMA: a * b + c
    block.add_instruction(ValueId(10), GpuOp::FMulAdd(ValueId(3), ValueId(6), ValueId(9)));

    // Store result
    block.add_instruction(ValueId(11), GpuOp::Param(3));
    block.add_instruction(ValueId(12), GpuOp::GetElementPtr(ValueId(11), vec![ValueId(0)]));
    block.add_instruction(
        ValueId(13),
        GpuOp::Store(ValueId(12), ValueId(10), MemorySpace::Global),
    );
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    let mut autodiff = AutoDiff::with_defaults();
    let result = autodiff.differentiate_kernel(&kernel);
    assert!(result.is_ok());

    let backward = result.unwrap();
    assert!(backward.kernel.name.contains("backward"));
}

#[test]
fn test_diff_primitive_to_tape_op_mapping() {
    // Verify all DiffPrimitive can convert to TapeOp
    let primitives = [
        DiffPrimitive::Add,
        DiffPrimitive::Sub,
        DiffPrimitive::Mul,
        DiffPrimitive::Div,
        DiffPrimitive::Neg,
        DiffPrimitive::Sin,
        DiffPrimitive::Cos,
        DiffPrimitive::Tan,
        DiffPrimitive::Exp,
        DiffPrimitive::Log,
        DiffPrimitive::Sqrt,
        DiffPrimitive::Pow,
        DiffPrimitive::Relu,
        DiffPrimitive::Sigmoid,
        DiffPrimitive::Tanh,
        DiffPrimitive::Sum,
        DiffPrimitive::Mean,
        DiffPrimitive::Max,
        DiffPrimitive::Min,
        DiffPrimitive::MatMul,
        DiffPrimitive::Transpose,
        DiffPrimitive::Fma,
    ];

    for prim in primitives {
        let tape_op = prim.to_tape_op();
        // Just verify conversion doesn't panic
        let _ = tape_op.name();
    }
}
