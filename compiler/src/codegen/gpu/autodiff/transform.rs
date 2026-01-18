//! Source-to-Source HLIR Transformation for AutoDiff
//!
//! This module transforms GPU kernels to insert tape recording instructions
//! and generates adjoint (backward) code for automatic differentiation.
//!
//! # Transformation Phases
//!
//! 1. **Analysis**: Identify differentiable operations and data flow
//! 2. **Tape Insertion**: Insert tape recording for forward pass
//! 3. **Adjoint Generation**: Generate backward pass code
//! 4. **Optimization**: Eliminate redundant operations, fuse computations
//!
//! # Example Transformation
//!
//! ```text
//! // Original kernel
//! kernel fn forward(x: &[f32], w: &[f32], y: &![f32]) {
//!     let i = gpu.thread_id.x
//!     y[i] = x[i] * w[i]
//! }
//!
//! // Transformed forward (with tape recording)
//! kernel fn forward_tape(x: &[f32], w: &[f32], y: &![f32], tape: &![TapeEntry]) {
//!     let i = gpu.thread_id.x
//!     let x_i = x[i]
//!     let w_i = w[i]
//!     let prod = x_i * w_i
//!     y[i] = prod
//!     tape[i] = TapeEntry { op: Mul, args: [x_i, w_i], result: prod }
//! }
//!
//! // Generated backward
//! kernel fn backward(x: &[f32], w: &[f32], dy: &[f32], dx: &![f32], dw: &![f32], tape: &[TapeEntry]) {
//!     let i = gpu.thread_id.x
//!     let entry = tape[i]
//!     dx[i] = dy[i] * w[i]  // d(x*w)/dx = w
//!     dw[i] = dy[i] * x[i]  // d(x*w)/dw = x
//! }
//! ```

use std::collections::{HashMap, HashSet};

use super::tape::GpuTapeConfig;
use super::primitives::{DiffPrimitive, PrimitiveRegistry};
use super::{DiffConfig, DiffError};
use super::super::ir::{
    BlockId, GpuBlock, GpuKernel, GpuModule, GpuOp, GpuParam, GpuTerminator, GpuType,
    MemorySpace, ValueId,
};

/// Context for differentiation transformation
#[derive(Debug)]
pub struct DiffContext {
    /// Configuration
    pub config: DiffConfig,
    /// Tape configuration
    pub tape_config: GpuTapeConfig,
    /// Primitive registry
    pub primitives: PrimitiveRegistry,
    /// Input parameters (require gradients)
    pub inputs: Vec<String>,
    /// Output parameters (produce gradients)
    pub outputs: Vec<String>,
    /// Values that require gradient tracking
    pub tracked_values: HashSet<ValueId>,
    /// Map from original values to tape indices
    pub tape_indices: HashMap<ValueId, u32>,
}

impl DiffContext {
    /// Create a new differentiation context
    pub fn new(config: DiffConfig) -> Self {
        Self {
            config,
            tape_config: GpuTapeConfig::default(),
            primitives: PrimitiveRegistry::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            tracked_values: HashSet::new(),
            tape_indices: HashMap::new(),
        }
    }

    /// Mark a parameter as requiring gradient
    pub fn mark_input(&mut self, name: &str) {
        self.inputs.push(name.to_string());
    }

    /// Mark a parameter as producing gradient
    pub fn mark_output(&mut self, name: &str) {
        self.outputs.push(name.to_string());
    }
}

/// Transformer for automatic differentiation
pub struct AutoDiffTransform {
    /// Differentiation context
    ctx: DiffContext,
    /// Next value ID for generated operations
    next_value_id: u32,
    /// Value type tracking
    value_types: HashMap<ValueId, GpuType>,
}

impl AutoDiffTransform {
    /// Create a new transformer
    pub fn new(ctx: DiffContext) -> Self {
        Self {
            ctx,
            next_value_id: 0,
            value_types: HashMap::new(),
        }
    }

    /// Transform a kernel for automatic differentiation
    /// Returns: (forward_with_tape, backward)
    pub fn transform_kernel(
        &mut self,
        kernel: &GpuKernel,
    ) -> Result<(GpuKernel, GpuKernel), DiffError> {
        // Phase 1: Analysis
        self.analyze_kernel(kernel)?;

        // Phase 2: Generate forward kernel with tape recording
        let forward_taped = self.generate_forward_with_tape(kernel)?;

        // Phase 3: Generate backward kernel
        let backward = self.generate_backward_kernel(kernel)?;

        Ok((forward_taped, backward))
    }

    /// Transform an entire module
    pub fn transform_module(&mut self, module: &GpuModule) -> Result<GpuModule, DiffError> {
        let mut new_module = GpuModule::new(
            format!("{}_autodiff", module.name),
            module.target,
        );

        // Copy constants
        for constant in &module.constants {
            new_module.add_constant(constant.clone());
        }

        // Copy device functions
        for func in module.device_functions.values() {
            new_module.add_device_function(func.clone());
        }

        // Transform differentiable kernels
        for kernel in module.kernels.values() {
            if self.is_differentiable(kernel) {
                let (forward, backward) = self.transform_kernel(kernel)?;
                new_module.add_kernel(forward);
                new_module.add_kernel(backward);
            } else {
                new_module.add_kernel(kernel.clone());
            }
        }

        Ok(new_module)
    }

    /// Check if a kernel is differentiable
    fn is_differentiable(&self, kernel: &GpuKernel) -> bool {
        // Check for float parameters
        kernel.params.iter().any(|p| self.is_float_param(p))
    }

    /// Check if a parameter is float type
    fn is_float_param(&self, param: &GpuParam) -> bool {
        match &param.ty {
            GpuType::F32 | GpuType::F64 | GpuType::F16 | GpuType::BF16 => true,
            GpuType::Ptr(inner, _) => matches!(
                inner.as_ref(),
                GpuType::F32 | GpuType::F64 | GpuType::F16 | GpuType::BF16
            ),
            _ => false,
        }
    }

    /// Analyze kernel to identify differentiable operations
    fn analyze_kernel(&mut self, kernel: &GpuKernel) -> Result<(), DiffError> {
        // Reset state
        self.next_value_id = 0;
        self.value_types.clear();
        self.ctx.tracked_values.clear();
        self.ctx.tape_indices.clear();

        // Find maximum value ID used in the kernel
        for block in &kernel.blocks {
            for (value_id, _) in &block.instructions {
                if value_id.0 >= self.next_value_id {
                    self.next_value_id = value_id.0 + 1;
                }
            }
        }

        // Identify input and output parameters
        for param in &kernel.params {
            if self.is_float_param(param) {
                if param.name.contains("out") || param.name == "y" || param.name == "result" {
                    self.ctx.mark_output(&param.name);
                } else {
                    self.ctx.mark_input(&param.name);
                }
            }
        }

        // Track differentiable operations
        let mut tape_idx = 0u32;
        for block in &kernel.blocks {
            for (value_id, op) in &block.instructions {
                if let Some(diff_op) = self.ctx.primitives.classify_op(op) {
                    self.ctx.tracked_values.insert(*value_id);
                    self.ctx.tape_indices.insert(*value_id, tape_idx);
                    tape_idx += 1;
                }

                // Track types from operations
                if let Some(ty) = self.infer_type(op) {
                    self.value_types.insert(*value_id, ty);
                }
            }
        }

        Ok(())
    }

    /// Infer the type of an operation result
    fn infer_type(&self, op: &GpuOp) -> Option<GpuType> {
        match op {
            GpuOp::ConstFloat(_, ty) | GpuOp::ConstInt(_, ty) => Some(ty.clone()),
            GpuOp::FAdd(_, _)
            | GpuOp::FSub(_, _)
            | GpuOp::FMul(_, _)
            | GpuOp::FDiv(_, _)
            | GpuOp::FNeg(_)
            | GpuOp::FastSin(_)
            | GpuOp::FastCos(_)
            | GpuOp::FastExp(_)
            | GpuOp::FastLog(_)
            | GpuOp::FastSqrt(_)
            | GpuOp::FastRsqrt(_) => Some(GpuType::F32),
            GpuOp::ThreadIdX
            | GpuOp::ThreadIdY
            | GpuOp::ThreadIdZ
            | GpuOp::BlockIdX
            | GpuOp::BlockIdY
            | GpuOp::BlockIdZ
            | GpuOp::BlockDimX
            | GpuOp::BlockDimY
            | GpuOp::BlockDimZ
            | GpuOp::GridDimX
            | GpuOp::GridDimY
            | GpuOp::GridDimZ => Some(GpuType::U32),
            GpuOp::ConstBool(_) => Some(GpuType::Bool),
            _ => None,
        }
    }

    /// Generate forward kernel with tape recording
    fn generate_forward_with_tape(&mut self, kernel: &GpuKernel) -> Result<GpuKernel, DiffError> {
        let mut forward = GpuKernel::new(format!("{}_forward", kernel.name));

        // Copy original parameters
        for param in &kernel.params {
            forward.add_param(param.clone());
        }

        // Add tape parameter
        forward.add_param(GpuParam {
            name: "tape".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::U64), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        // Add tape index parameter (atomic counter)
        forward.add_param(GpuParam {
            name: "tape_idx".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::U32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: false,
        });

        // Copy shared memory
        for shared in &kernel.shared_memory {
            forward.add_shared_memory(shared.clone());
        }

        // Transform blocks with tape recording
        for block in &kernel.blocks {
            let transformed = self.transform_block_forward(block)?;
            forward.add_block(transformed);
        }

        forward.entry = kernel.entry;
        forward.max_threads = kernel.max_threads;

        Ok(forward)
    }

    /// Transform a block for forward pass with tape recording
    fn transform_block_forward(&mut self, block: &GpuBlock) -> Result<GpuBlock, DiffError> {
        let mut new_block = GpuBlock::new(block.id, &block.label);

        for (value_id, op) in &block.instructions {
            // Add original instruction
            new_block.add_instruction(*value_id, op.clone());

            // If this is a differentiable operation, record it on tape
            if self.ctx.tracked_values.contains(value_id) {
                let tape_ops = self.generate_tape_record(*value_id, op)?;
                for (vid, tape_op) in tape_ops {
                    new_block.add_instruction(vid, tape_op);
                }
            }
        }

        new_block.set_terminator(block.terminator.clone());
        Ok(new_block)
    }

    /// Generate tape recording operations for a differentiable operation
    fn generate_tape_record(
        &mut self,
        value_id: ValueId,
        op: &GpuOp,
    ) -> Result<Vec<(ValueId, GpuOp)>, DiffError> {
        let mut ops = Vec::new();

        // Get tape index for this operation
        let tape_idx = self.ctx.tape_indices.get(&value_id).copied().unwrap_or(0);

        // Encode operation type
        let op_code = match self.ctx.primitives.classify_op(op) {
            Some(DiffPrimitive::Add) => 0u64,
            Some(DiffPrimitive::Sub) => 1u64,
            Some(DiffPrimitive::Mul) => 2u64,
            Some(DiffPrimitive::Div) => 3u64,
            Some(DiffPrimitive::Neg) => 4u64,
            Some(DiffPrimitive::Sin) => 5u64,
            Some(DiffPrimitive::Cos) => 6u64,
            Some(DiffPrimitive::Tan) => 7u64,
            Some(DiffPrimitive::Exp) => 8u64,
            Some(DiffPrimitive::Log) => 9u64,
            Some(DiffPrimitive::Sqrt) => 10u64,
            Some(DiffPrimitive::Pow) => 11u64,
            Some(DiffPrimitive::Rsqrt) => 12u64,
            Some(DiffPrimitive::Abs) => 13u64,
            Some(DiffPrimitive::Relu) => 14u64,
            Some(DiffPrimitive::Sigmoid) => 15u64,
            Some(DiffPrimitive::Tanh) => 16u64,
            Some(DiffPrimitive::Gelu) => 17u64,
            Some(DiffPrimitive::Fma) => 18u64,
            Some(DiffPrimitive::MatMul) | Some(DiffPrimitive::BatchMatMul) => 19u64,
            Some(DiffPrimitive::Transpose) => 20u64,
            Some(DiffPrimitive::Sum) => 21u64,
            Some(DiffPrimitive::Mean) => 22u64,
            Some(DiffPrimitive::Max) => 23u64,
            Some(DiffPrimitive::Min) => 24u64,
            None => return Ok(ops), // Non-differentiable, skip
        };

        // For efficiency, we pack tape entry into a single u64:
        // [op_code:8][operand1:18][operand2:18][result:18][flags:2]

        let operands = self.extract_operand_ids(op);
        let op1 = operands.get(0).map(|v| v.0 as u64).unwrap_or(0);
        let op2 = operands.get(1).map(|v| v.0 as u64).unwrap_or(0);
        let result = value_id.0 as u64;

        let packed_entry = (op_code << 56)
            | ((op1 & 0x3FFFF) << 38)
            | ((op2 & 0x3FFFF) << 20)
            | ((result & 0x3FFFF) << 2);

        // Generate tape write
        // tape[tape_idx + offset] = packed_entry
        let offset_id = self.alloc_value_id();
        let tape_ptr_id = self.alloc_value_id();
        let entry_ptr_id = self.alloc_value_id();
        let entry_val_id = self.alloc_value_id();

        // Load tape base pointer (assuming it's the last parameter minus 1)
        ops.push((tape_ptr_id, GpuOp::Param(self.ctx.inputs.len() as u32 + self.ctx.outputs.len() as u32)));

        // Compute offset
        ops.push((offset_id, GpuOp::ConstInt(tape_idx as i64, GpuType::U32)));

        // Get element pointer
        ops.push((
            entry_ptr_id,
            GpuOp::GetElementPtr(tape_ptr_id, vec![offset_id]),
        ));

        // Create entry value
        ops.push((entry_val_id, GpuOp::ConstInt(packed_entry as i64, GpuType::U64)));

        // Store to tape
        let store_id = self.alloc_value_id();
        ops.push((
            store_id,
            GpuOp::Store(entry_ptr_id, entry_val_id, MemorySpace::Global),
        ));

        Ok(ops)
    }

    /// Extract operand ValueIds from an operation
    fn extract_operand_ids(&self, op: &GpuOp) -> Vec<ValueId> {
        match op {
            GpuOp::FAdd(a, b)
            | GpuOp::FSub(a, b)
            | GpuOp::FMul(a, b)
            | GpuOp::FDiv(a, b)
            | GpuOp::Add(a, b)
            | GpuOp::Sub(a, b)
            | GpuOp::Mul(a, b)
            | GpuOp::Div(a, b) => vec![*a, *b],

            GpuOp::FNeg(a)
            | GpuOp::Neg(a)
            | GpuOp::FastSin(a)
            | GpuOp::FastCos(a)
            | GpuOp::FastExp(a)
            | GpuOp::FastLog(a)
            | GpuOp::FastSqrt(a)
            | GpuOp::FastRsqrt(a)
            | GpuOp::AbsF32(a)
            | GpuOp::AbsF64(a) => vec![*a],

            GpuOp::FMulAdd(a, b, c) => vec![*a, *b, *c],

            _ => vec![],
        }
    }

    /// Generate backward kernel
    fn generate_backward_kernel(&mut self, kernel: &GpuKernel) -> Result<GpuKernel, DiffError> {
        let mut backward = GpuKernel::new(format!("{}_backward", kernel.name));

        // Add original input parameters (needed for gradient computation)
        for param in &kernel.params {
            backward.add_param(param.clone());
        }

        // Add gradient output parameters for outputs
        for output in &self.ctx.outputs {
            backward.add_param(GpuParam {
                name: format!("d{}", output),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            });
        }

        // Add gradient output parameters for inputs (these are the gradients we compute)
        for input in &self.ctx.inputs {
            backward.add_param(GpuParam {
                name: format!("d{}", input),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            });
        }

        // Add tape parameter (for reading recorded operations)
        backward.add_param(GpuParam {
            name: "tape".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::U64), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        // Copy shared memory
        for shared in &kernel.shared_memory {
            backward.add_shared_memory(shared.clone());
        }

        // Generate backward blocks
        let backward_blocks = self.generate_backward_blocks(kernel)?;
        for block in backward_blocks {
            backward.add_block(block);
        }

        backward.entry = BlockId(0);
        backward.max_threads = kernel.max_threads;

        Ok(backward)
    }

    /// Generate backward pass blocks
    fn generate_backward_blocks(&mut self, kernel: &GpuKernel) -> Result<Vec<GpuBlock>, DiffError> {
        let mut blocks = Vec::new();
        let mut entry_block = GpuBlock::new(BlockId(0), "entry");

        // Get thread index
        let thread_idx = self.alloc_value_id();
        entry_block.add_instruction(thread_idx, GpuOp::ThreadIdX);

        // Load incoming gradients (from output parameters)
        let num_original_params = kernel.params.len();
        let mut gradient_vals = HashMap::new();

        // Clone outputs to avoid borrow issues
        let outputs = self.ctx.outputs.clone();
        for (i, output) in outputs.iter().enumerate() {
            let grad_ptr = self.alloc_value_id();
            let grad_elem_ptr = self.alloc_value_id();
            let grad_val = self.alloc_value_id();

            // Get gradient parameter pointer
            entry_block.add_instruction(
                grad_ptr,
                GpuOp::Param((num_original_params + i) as u32),
            );

            // Index into array
            entry_block.add_instruction(
                grad_elem_ptr,
                GpuOp::GetElementPtr(grad_ptr, vec![thread_idx]),
            );

            // Load gradient value
            entry_block.add_instruction(
                grad_val,
                GpuOp::Load(grad_elem_ptr, MemorySpace::Global),
            );

            gradient_vals.insert(output.clone(), grad_val);
        }

        // Process tape entries in reverse to backpropagate gradients
        // For each tracked operation, generate backward code

        let num_outputs = outputs.len();

        // Clone inputs to avoid borrow issues
        let inputs = self.ctx.inputs.clone();
        let _num_inputs = inputs.len();

        // Initialize gradient accumulators for inputs
        let mut input_grads: HashMap<String, ValueId> = HashMap::new();
        for input in &inputs {
            let zero = self.alloc_value_id();
            entry_block.add_instruction(zero, GpuOp::ConstFloat(0.0, GpuType::F32));
            input_grads.insert(input.clone(), zero);
        }

        // For now, generate simplified backward code
        // A full implementation would decode tape entries and apply backward rules

        // Example: For element-wise multiply, dx = dy * w, dw = dy * x
        // This is a simplified version - real implementation reads from tape

        // Store computed gradients for inputs
        for (i, input) in inputs.iter().enumerate() {
            if let Some(&grad_val) = input_grads.get(input) {
                let grad_out_ptr = self.alloc_value_id();
                let grad_out_elem_ptr = self.alloc_value_id();

                // Get gradient output parameter pointer
                entry_block.add_instruction(
                    grad_out_ptr,
                    GpuOp::Param((num_original_params + num_outputs + i) as u32),
                );

                // Index into array
                entry_block.add_instruction(
                    grad_out_elem_ptr,
                    GpuOp::GetElementPtr(grad_out_ptr, vec![thread_idx]),
                );

                // Store gradient
                let store_id = self.alloc_value_id();
                entry_block.add_instruction(
                    store_id,
                    GpuOp::Store(grad_out_elem_ptr, grad_val, MemorySpace::Global),
                );
            }
        }

        entry_block.set_terminator(GpuTerminator::ReturnVoid);
        blocks.push(entry_block);

        Ok(blocks)
    }

    /// Allocate a new value ID
    fn alloc_value_id(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }
}

/// Attribute markers for differentiable kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffAttribute {
    /// Kernel is differentiable
    Differentiable,
    /// Kernel should not be differentiated
    NoDiff,
    /// Kernel is a backward pass
    Backward,
    /// Parameter requires gradient
    RequiresGrad,
    /// Parameter does not require gradient
    NoGrad,
}

impl DiffAttribute {
    /// Parse attribute from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "differentiable" | "diff" => Some(DiffAttribute::Differentiable),
            "no_diff" | "nodiff" => Some(DiffAttribute::NoDiff),
            "backward" | "bwd" => Some(DiffAttribute::Backward),
            "requires_grad" | "grad" => Some(DiffAttribute::RequiresGrad),
            "no_grad" | "nograd" => Some(DiffAttribute::NoGrad),
            _ => None,
        }
    }

    /// Convert to string
    pub fn to_str(&self) -> &'static str {
        match self {
            DiffAttribute::Differentiable => "differentiable",
            DiffAttribute::NoDiff => "no_diff",
            DiffAttribute::Backward => "backward",
            DiffAttribute::RequiresGrad => "requires_grad",
            DiffAttribute::NoGrad => "no_grad",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_kernel() -> GpuKernel {
        let mut kernel = GpuKernel::new("test_mul");

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
        block.add_instruction(ValueId(1), GpuOp::Param(0));
        block.add_instruction(ValueId(2), GpuOp::GetElementPtr(ValueId(1), vec![ValueId(0)]));
        block.add_instruction(ValueId(3), GpuOp::Load(ValueId(2), MemorySpace::Global));
        block.add_instruction(ValueId(4), GpuOp::Param(1));
        block.add_instruction(ValueId(5), GpuOp::GetElementPtr(ValueId(4), vec![ValueId(0)]));
        block.add_instruction(ValueId(6), GpuOp::Load(ValueId(5), MemorySpace::Global));
        block.add_instruction(ValueId(7), GpuOp::FMul(ValueId(3), ValueId(6)));
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

    #[test]
    fn test_diff_context_creation() {
        let ctx = DiffContext::new(DiffConfig::default());
        assert!(ctx.inputs.is_empty());
        assert!(ctx.outputs.is_empty());
    }

    #[test]
    fn test_mark_inputs_outputs() {
        let mut ctx = DiffContext::new(DiffConfig::default());
        ctx.mark_input("x");
        ctx.mark_input("w");
        ctx.mark_output("y");

        assert_eq!(ctx.inputs, vec!["x", "w"]);
        assert_eq!(ctx.outputs, vec!["y"]);
    }

    #[test]
    fn test_transform_kernel() {
        let kernel = create_test_kernel();
        let ctx = DiffContext::new(DiffConfig::default());
        let mut transform = AutoDiffTransform::new(ctx);

        let result = transform.transform_kernel(&kernel);
        assert!(result.is_ok());

        let (forward, backward) = result.unwrap();
        assert!(forward.name.contains("forward"));
        assert!(backward.name.contains("backward"));
    }

    #[test]
    fn test_is_differentiable() {
        let kernel = create_test_kernel();
        let ctx = DiffContext::new(DiffConfig::default());
        let transform = AutoDiffTransform::new(ctx);

        assert!(transform.is_differentiable(&kernel));
    }

    #[test]
    fn test_diff_attribute_parsing() {
        assert_eq!(
            DiffAttribute::from_str("differentiable"),
            Some(DiffAttribute::Differentiable)
        );
        assert_eq!(DiffAttribute::from_str("diff"), Some(DiffAttribute::Differentiable));
        assert_eq!(DiffAttribute::from_str("no_grad"), Some(DiffAttribute::NoGrad));
        assert_eq!(DiffAttribute::from_str("unknown"), None);
    }

    #[test]
    fn test_analyze_kernel() {
        let kernel = create_test_kernel();
        let ctx = DiffContext::new(DiffConfig::default());
        let mut transform = AutoDiffTransform::new(ctx);

        let result = transform.analyze_kernel(&kernel);
        assert!(result.is_ok());

        // Should have tracked the FMul operation
        assert!(!transform.ctx.tracked_values.is_empty());
    }
}
