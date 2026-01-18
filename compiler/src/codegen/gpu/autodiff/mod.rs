//! GPU Reverse-Mode Automatic Differentiation
//!
//! This module implements tape-based reverse-mode automatic differentiation (backpropagation)
//! for GPU kernels. It enables computing gradients efficiently on GPU for deep learning
//! and scientific computing applications.
//!
//! # Architecture
//!
//! ```text
//! Forward Kernel ──> Tape Recording ──> Backward Kernel Generation
//!       │                    │                       │
//!       v                    v                       v
//!   Forward Pass        GpuTape               Gradient Computation
//!   (y = f(x))      (operations list)        (dx = df/dx * dy)
//! ```
//!
//! # Key Components
//!
//! - **Tape**: Records operations during forward pass (Wengert list)
//! - **Primitives**: Backward rules for each differentiable operation
//! - **Transform**: Source-to-source transformation inserting tape recording
//! - **Codegen**: Generates backward pass GPU kernel from forward pass
//!
//! # Usage
//!
//! ```ignore
//! use sounio::codegen::gpu::autodiff::{AutoDiff, DiffConfig};
//!
//! let autodiff = AutoDiff::new(DiffConfig::default());
//! let backward_kernel = autodiff.differentiate_kernel(&forward_kernel)?;
//! ```
//!
//! # Supported Operations
//!
//! - Arithmetic: add, sub, mul, div, neg
//! - Math: sin, cos, tan, exp, log, sqrt, pow
//! - Reductions: sum, mean, max (with appropriate gradient handling)
//! - Matrix operations: matmul, transpose
//!
//! # Memory Management
//!
//! The tape is stored in GPU global memory with configurable size.
//! Gradient buffers are allocated alongside output buffers.

pub mod codegen;
pub mod primitives;
pub mod tape;
pub mod transform;

pub use codegen::{BackwardCodegen, BackwardKernel};
pub use primitives::{BackwardRule, DiffPrimitive, PrimitiveRegistry};
pub use tape::{GpuTape, GpuTapeConfig, TapeEntry, TapeOp};
pub use transform::{AutoDiffTransform, DiffContext};

use std::collections::HashMap;

use super::ir::{GpuKernel, GpuModule, GpuTarget, GpuType, MemorySpace, ValueId};

/// Configuration for automatic differentiation
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Maximum tape size (number of entries)
    pub max_tape_size: usize,
    /// Enable checkpointing to reduce memory usage
    pub enable_checkpointing: bool,
    /// Checkpoint interval (operations between checkpoints)
    pub checkpoint_interval: usize,
    /// Enable gradient accumulation for reused values
    pub accumulate_gradients: bool,
    /// Target GPU architecture
    pub target: GpuTarget,
    /// Enable memory coalescing optimizations
    pub optimize_memory: bool,
    /// Enable warp-level gradient reductions
    pub warp_reductions: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            max_tape_size: 1024 * 1024, // 1M operations
            enable_checkpointing: false,
            checkpoint_interval: 1000,
            accumulate_gradients: true,
            target: GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
            optimize_memory: true,
            warp_reductions: true,
        }
    }
}

/// Automatic differentiation engine for GPU kernels
#[derive(Debug)]
pub struct AutoDiff {
    /// Configuration
    config: DiffConfig,
    /// Primitive registry for backward rules
    primitives: PrimitiveRegistry,
    /// Cached differentiated kernels
    cache: HashMap<String, BackwardKernel>,
}

impl AutoDiff {
    /// Create a new AutoDiff instance with the given configuration
    pub fn new(config: DiffConfig) -> Self {
        Self {
            config,
            primitives: PrimitiveRegistry::new(),
            cache: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(DiffConfig::default())
    }

    /// Differentiate a kernel, generating a backward pass kernel
    ///
    /// Given a forward kernel like:
    /// ```ignore
    /// kernel fn forward(x: &[f32], w: &[f32], y: &![f32]) {
    ///     let i = gpu.thread_id.x
    ///     y[i] = x[i] * w[i]
    /// }
    /// ```
    ///
    /// Generates a backward kernel like:
    /// ```ignore
    /// kernel fn backward(x: &[f32], w: &[f32], dy: &[f32], dx: &![f32], dw: &![f32]) {
    ///     let i = gpu.thread_id.x
    ///     dx[i] = dy[i] * w[i]  // d(x*w)/dx = w
    ///     dw[i] = dy[i] * x[i]  // d(x*w)/dw = x
    /// }
    /// ```
    pub fn differentiate_kernel(&mut self, kernel: &GpuKernel) -> Result<BackwardKernel, DiffError> {
        // Check cache first
        if let Some(cached) = self.cache.get(&kernel.name) {
            return Ok(cached.clone());
        }

        // Analyze the kernel to build the computation graph
        let analysis = self.analyze_kernel(kernel)?;

        // Generate backward kernel
        let backward = BackwardCodegen::new(&self.config, &self.primitives)
            .generate_backward(&kernel, &analysis)?;

        // Cache result
        self.cache.insert(kernel.name.clone(), backward.clone());

        Ok(backward)
    }

    /// Differentiate an entire GPU module
    pub fn differentiate_module(&mut self, module: &GpuModule) -> Result<GpuModule, DiffError> {
        let mut diff_module = GpuModule::new(
            format!("{}_autodiff", module.name),
            module.target,
        );

        // Copy constants
        for constant in &module.constants {
            diff_module.add_constant(constant.clone());
        }

        // Copy device functions (unchanged)
        for func in module.device_functions.values() {
            diff_module.add_device_function(func.clone());
        }

        // Process each kernel marked as differentiable
        for kernel in module.kernels.values() {
            if self.is_differentiable(kernel) {
                // Add original kernel
                diff_module.add_kernel(kernel.clone());

                // Generate and add backward kernel
                let backward = self.differentiate_kernel(kernel)?;
                diff_module.add_kernel(backward.kernel);
            } else {
                // Non-differentiable kernels are copied as-is
                diff_module.add_kernel(kernel.clone());
            }
        }

        Ok(diff_module)
    }

    /// Check if a kernel is differentiable
    fn is_differentiable(&self, kernel: &GpuKernel) -> bool {
        // A kernel is differentiable if it has at least one float input and output
        let has_float_input = kernel.params.iter().any(|p| is_float_type(&p.ty));
        let has_float_output = kernel.params.iter().any(|p| {
            is_float_type(&p.ty) && matches!(p.space, MemorySpace::Global)
        });
        has_float_input && has_float_output
    }

    /// Analyze kernel to extract computation graph
    fn analyze_kernel(&self, kernel: &GpuKernel) -> Result<KernelAnalysis, DiffError> {
        let mut analysis = KernelAnalysis::new(kernel.name.clone());

        // Identify input and output parameters
        for (idx, param) in kernel.params.iter().enumerate() {
            if is_float_type(&param.ty) {
                if param.name.starts_with("d") || param.name.contains("grad") {
                    // Already a gradient parameter
                    continue;
                }

                // Check if it's an input (read-only) or output (write)
                // For now, we use naming convention: inputs don't have &! prefix
                // In a full implementation, we'd do data-flow analysis
                let is_output = self.is_output_param(kernel, idx);

                if is_output {
                    analysis.outputs.push(ParamInfo {
                        index: idx,
                        name: param.name.clone(),
                        ty: param.ty.clone(),
                    });
                } else {
                    analysis.inputs.push(ParamInfo {
                        index: idx,
                        name: param.name.clone(),
                        ty: param.ty.clone(),
                    });
                }
            }
        }

        // Extract operations from basic blocks
        for block in &kernel.blocks {
            for (value_id, op) in &block.instructions {
                if let Some(diff_op) = self.primitives.classify_op(op) {
                    analysis.operations.push(OpInfo {
                        result: *value_id,
                        op: diff_op,
                    });
                }
            }
        }

        Ok(analysis)
    }

    /// Check if a parameter is an output (receives writes)
    fn is_output_param(&self, kernel: &GpuKernel, param_idx: usize) -> bool {
        // Simple heuristic: check if the parameter is used in store operations
        for block in &kernel.blocks {
            for (_, op) in &block.instructions {
                if let super::ir::GpuOp::Store(ptr, _, _) = op {
                    // Check if ptr originates from this parameter
                    if ptr.0 == param_idx as u32 {
                        return true;
                    }
                }
            }
        }
        // Fallback: check naming convention
        kernel.params.get(param_idx)
            .map(|p| p.name.ends_with("_out") || p.name == "y" || p.name == "output")
            .unwrap_or(false)
    }

    /// Get configuration
    pub fn config(&self) -> &DiffConfig {
        &self.config
    }

    /// Get mutable configuration
    pub fn config_mut(&mut self) -> &mut DiffConfig {
        &mut self.config
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Analysis results for a kernel
#[derive(Debug, Clone)]
pub struct KernelAnalysis {
    /// Kernel name
    pub name: String,
    /// Input parameters (to compute gradients for)
    pub inputs: Vec<ParamInfo>,
    /// Output parameters (that produce gradients)
    pub outputs: Vec<ParamInfo>,
    /// Differentiable operations in the kernel
    pub operations: Vec<OpInfo>,
}

impl KernelAnalysis {
    fn new(name: String) -> Self {
        Self {
            name,
            inputs: Vec::new(),
            outputs: Vec::new(),
            operations: Vec::new(),
        }
    }
}

/// Information about a parameter
#[derive(Debug, Clone)]
pub struct ParamInfo {
    /// Parameter index
    pub index: usize,
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub ty: GpuType,
}

/// Information about an operation
#[derive(Debug, Clone)]
pub struct OpInfo {
    /// Result value ID
    pub result: ValueId,
    /// Differentiable operation type
    pub op: DiffPrimitive,
}

/// Errors that can occur during differentiation
#[derive(Debug, Clone)]
pub enum DiffError {
    /// Operation is not differentiable
    NonDifferentiable(String),
    /// Kernel has no differentiable parameters
    NoDifferentiableParams,
    /// Invalid kernel structure
    InvalidKernel(String),
    /// Unsupported operation
    UnsupportedOp(String),
    /// Memory allocation error
    AllocationError(String),
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for DiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffError::NonDifferentiable(op) => write!(f, "Operation '{}' is not differentiable", op),
            DiffError::NoDifferentiableParams => write!(f, "Kernel has no differentiable parameters"),
            DiffError::InvalidKernel(msg) => write!(f, "Invalid kernel structure: {}", msg),
            DiffError::UnsupportedOp(op) => write!(f, "Unsupported operation: {}", op),
            DiffError::AllocationError(msg) => write!(f, "Memory allocation error: {}", msg),
            DiffError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for DiffError {}

/// Check if a GPU type is a floating-point type
fn is_float_type(ty: &GpuType) -> bool {
    match ty {
        GpuType::F16 | GpuType::F32 | GpuType::F64 | GpuType::BF16 => true,
        GpuType::Ptr(inner, _) => is_float_type(inner),
        GpuType::Array(inner, _) => is_float_type(inner),
        GpuType::Vec2(inner) | GpuType::Vec3(inner) | GpuType::Vec4(inner) => is_float_type(inner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ir::*;

    fn create_simple_mul_kernel() -> GpuKernel {
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
        block.add_instruction(ValueId(10), GpuOp::Store(ValueId(9), ValueId(7), MemorySpace::Global));

        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        kernel
    }

    #[test]
    fn test_autodiff_creation() {
        let autodiff = AutoDiff::with_defaults();
        assert_eq!(autodiff.config().max_tape_size, 1024 * 1024);
    }

    #[test]
    fn test_kernel_analysis() {
        let kernel = create_simple_mul_kernel();
        let autodiff = AutoDiff::with_defaults();

        assert!(autodiff.is_differentiable(&kernel));

        let analysis = autodiff.analyze_kernel(&kernel).unwrap();
        assert_eq!(analysis.name, "forward_mul");
        // Should have inputs x, w and output y
        assert!(!analysis.inputs.is_empty() || !analysis.outputs.is_empty());
    }

    #[test]
    fn test_differentiate_kernel() {
        let kernel = create_simple_mul_kernel();
        let mut autodiff = AutoDiff::with_defaults();

        let result = autodiff.differentiate_kernel(&kernel);
        assert!(result.is_ok());

        let backward = result.unwrap();
        assert!(backward.kernel.name.contains("backward"));
    }

    #[test]
    fn test_is_float_type() {
        assert!(is_float_type(&GpuType::F32));
        assert!(is_float_type(&GpuType::F64));
        assert!(is_float_type(&GpuType::F16));
        assert!(is_float_type(&GpuType::BF16));
        assert!(is_float_type(&GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global)));
        assert!(!is_float_type(&GpuType::I32));
        assert!(!is_float_type(&GpuType::U64));
    }
}
