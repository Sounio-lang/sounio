//! HLIR to GPU IR Lowering
//!
//! This module provides the critical bridge between the high-level SSA IR (HLIR)
//! and the GPU-specific IR that can be compiled to PTX or SPIR-V.
//!
//! # Architecture
//!
//! ```text
//! HLIR (SSA) ──┬──> GPU IR ──> PTX (NVIDIA)
//!              │           └──> SPIR-V (Vulkan/OpenCL)
//!              │
//!              └──> Epistemic Extension
//!                   • Shadow registers for ε (epsilon/uncertainty)
//!                   • Validity predicates
//!                   • Provenance tracking (u64 bitmask)
//! ```
//!
//! # Epistemic State as Hardware Resources
//!
//! Every Knowledge value in Sounio is lowered to a tuple of GPU values:
//!
//! ```text
//! Knowledge[T, ε, δ, Φ] ──> {
//!     value: T,           // The actual data
//!     epsilon: f32,       // Uncertainty bound (shadow register)
//!     validity: pred,     // Predicate register for validity
//!     provenance: u64,    // Bit-packed provenance mask
//! }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use sounio::codegen::gpu::{hlir_to_gpu, GpuTarget, PtxCodegen};
//! use sounio::hlir::HlirModule;
//!
//! let hlir: HlirModule = /* ... */;
//! let gpu_module = hlir_to_gpu::lower(&hlir, GpuTarget::Cuda { compute_capability: (8, 0) });
//! let ptx = PtxCodegen::new((8, 0)).generate(&gpu_module);
//! ```

use rustc_hash::FxHashMap;
use std::collections::HashMap;

use super::async_pipeline::{
    AsyncPipeline, PipelineSchedule, create_pipeline_from_tile, schedule_pipeline,
};
use super::autotune::{AutoTuneConfig, AutoTuner, TunedConfig};
use super::fusion::{FusionConfig, FusionError, analyze_and_fuse_kernels};
use super::graph::build_graph_from_module;
use super::ir::*;
use crate::hlir::{
    BinaryOp, BlockId as HlirBlockId, HlirBlock, HlirConstant, HlirFunction, HlirInstr, HlirModule,
    HlirTerminator, HlirType, Op, UnaryOp, ValueId as HlirValueId,
};

/// Configuration for HLIR to GPU lowering
#[derive(Debug, Clone)]
pub struct LoweringConfig {
    /// Target GPU architecture
    pub target: GpuTarget,
    /// Enable epistemic state tracking (shadow registers)
    pub epistemic_enabled: bool,
    /// Enable counterfactual execution support
    pub counterfactual_enabled: bool,
    /// Maximum threads per block hint
    pub max_threads_per_block: Option<u32>,
    /// Shared memory size hint (bytes)
    pub shared_memory_hint: u32,
    /// Enable fast math approximations
    pub fast_math: bool,
    /// Debug info generation
    pub debug_info: bool,

    // === Optimization Passes (Phase 6) ===
    /// Enable auto-tuning for kernel launch configuration
    pub auto_tune: bool,
    /// Auto-tuning configuration (uses default if None)
    pub tune_config: Option<AutoTuneConfig>,
    /// Enable kernel fusion optimization
    pub enable_fusion: bool,
    /// Fusion configuration (uses default if None)
    pub fusion_config: Option<FusionConfig>,
    /// Enable async memory pipelining
    pub enable_pipelining: bool,
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self {
            target: GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
            epistemic_enabled: true,
            counterfactual_enabled: false,
            max_threads_per_block: Some(256),
            shared_memory_hint: 48 * 1024, // 48KB default
            fast_math: true,
            debug_info: false,
            // Optimization passes disabled by default for backward compatibility
            auto_tune: false,
            tune_config: None,
            enable_fusion: false,
            fusion_config: None,
            enable_pipelining: false,
        }
    }
}

/// Lower HLIR module to GPU IR with default configuration
pub fn lower(hlir: &HlirModule, target: GpuTarget) -> GpuModule {
    let config = LoweringConfig {
        target,
        ..Default::default()
    };
    lower_with_config(hlir, &config)
}

/// Lower HLIR module to GPU IR with custom configuration
pub fn lower_with_config(hlir: &HlirModule, config: &LoweringConfig) -> GpuModule {
    let mut lowering = HlirToGpuLowering::new(config.clone());
    lowering.lower_module(hlir)
}

// ============================================================================
// Optimized Lowering (Phase 6 Integration)
// ============================================================================

/// Result of optimized GPU lowering with metadata
#[derive(Debug, Clone)]
pub struct OptimizedGpuModule {
    /// The lowered and optimized GPU module
    pub module: GpuModule,
    /// Tuned configurations per kernel (kernel name -> TunedConfig)
    pub tuned_configs: HashMap<String, TunedConfig>,
    /// Pipeline schedules per kernel (kernel name -> (pipeline, schedule))
    pub pipelines: HashMap<String, (AsyncPipeline, PipelineSchedule)>,
    /// Whether fusion was applied
    pub fusion_applied: bool,
    /// Number of kernels before fusion
    pub original_kernel_count: usize,
}

/// Lower HLIR to GPU with all optimizations applied
///
/// This is the recommended entry point for production compilation.
/// It chains: lowering → auto-tuning → fusion → pipelining
pub fn lower_and_optimize(hlir: &HlirModule, config: &LoweringConfig) -> OptimizedGpuModule {
    // Step 1: Basic lowering
    let mut module = lower_with_config(hlir, config);
    let original_kernel_count = module.kernels.len();

    // Step 2: Auto-tuning (per-kernel)
    let tuned_configs = if config.auto_tune {
        apply_auto_tuning(&mut module, config)
    } else {
        HashMap::new()
    };

    // Step 3: Kernel fusion (module-wide)
    let fusion_applied = if config.enable_fusion {
        match apply_fusion(&module, config) {
            Ok(fused_module) => {
                module = fused_module;
                true
            }
            Err(_) => false, // Fusion failed, continue with unfused module
        }
    } else {
        false
    };

    // Step 4: Async pipelining (per-kernel with tile config)
    let pipelines = if config.enable_pipelining {
        apply_pipelining(&mut module, &tuned_configs, config)
    } else {
        HashMap::new()
    };

    OptimizedGpuModule {
        module,
        tuned_configs,
        pipelines,
        fusion_applied,
        original_kernel_count,
    }
}

/// Apply auto-tuning to all kernels in the module
fn apply_auto_tuning(
    module: &mut GpuModule,
    config: &LoweringConfig,
) -> HashMap<String, TunedConfig> {
    let tune_config = config
        .tune_config
        .clone()
        .unwrap_or_else(|| AutoTuneConfig {
            target: config.target,
            ..Default::default()
        });

    let tuner = AutoTuner::new(tune_config);
    let mut results = HashMap::new();

    for (name, kernel) in &mut module.kernels {
        let tuned = tuner.tune_kernel(kernel);

        // Apply tuned config to kernel
        kernel.max_threads = Some(tuned.block_shape.total_threads());

        // Update shared memory if tuner recommends more
        if tuned.shared_mem_bytes > kernel.shared_mem_size {
            kernel.shared_mem_size = tuned.shared_mem_bytes;
        }

        results.insert(name.clone(), tuned);
    }

    results
}

/// Apply kernel fusion optimization to the module
fn apply_fusion(module: &GpuModule, config: &LoweringConfig) -> Result<GpuModule, FusionError> {
    // Build dependency graph from module
    let graph = build_graph_from_module(module);

    // Apply fusion with config
    let fusion_config = config.fusion_config.clone();
    analyze_and_fuse_kernels(module, &graph, fusion_config)
}

/// Apply async pipelining to kernels with tile configurations
fn apply_pipelining(
    module: &mut GpuModule,
    tuned_configs: &HashMap<String, TunedConfig>,
    config: &LoweringConfig,
) -> HashMap<String, (AsyncPipeline, PipelineSchedule)> {
    let arch = match config.target {
        GpuTarget::Cuda { compute_capability } => {
            CudaArch::from_compute_capability(compute_capability).unwrap_or(CudaArch::Ampere)
        }
        _ => CudaArch::Ampere, // Default for non-CUDA targets
    };

    let mut pipelines = HashMap::new();

    for (name, kernel) in &mut module.kernels {
        // Check if kernel has tile config with pipelining
        let tile_config = tuned_configs.get(name).and_then(|t| t.tile_config.as_ref());

        if let Some(tile) = tile_config
            && tile.pipeline_stages > 1
        {
            let pipeline = create_pipeline_from_tile(tile, arch);
            let schedule = schedule_pipeline(&pipeline);

            // Update kernel shared memory for pipeline buffers
            kernel.shared_mem_size += pipeline.total_shared_memory();

            pipelines.insert(name.clone(), (pipeline, schedule));
        }
    }

    pipelines
}

/// HLIR to GPU lowering context
struct HlirToGpuLowering {
    /// Configuration
    config: LoweringConfig,
    /// Output GPU module
    module: GpuModule,
    /// Value mapping: HLIR ValueId -> GPU ValueId
    value_map: FxHashMap<HlirValueId, ValueId>,
    /// Block mapping: HLIR BlockId -> GPU BlockId
    block_map: FxHashMap<HlirBlockId, BlockId>,
    /// Next GPU value ID
    next_value_id: u32,
    /// Next GPU block ID
    next_block_id: u32,
    /// Epistemic shadow values: original value -> (epsilon, validity, provenance)
    epistemic_shadows: FxHashMap<ValueId, EpistemicShadow>,
    /// Type cache for values
    value_types: FxHashMap<ValueId, GpuType>,
    /// Current function's local variables (stack slots)
    locals: FxHashMap<String, ValueId>,
    /// Parameter mapping for current function
    param_values: Vec<ValueId>,
}

/// Epistemic shadow state for a value
#[derive(Debug, Clone)]
struct EpistemicShadow {
    /// Epsilon (uncertainty) shadow value - f32
    pub epsilon: ValueId,
    /// Validity predicate - bool/pred
    pub validity: ValueId,
    /// Provenance bitmask - u64
    pub provenance: ValueId,
}

impl HlirToGpuLowering {
    fn new(config: LoweringConfig) -> Self {
        let module = GpuModule::new("sounio_module", config.target);
        Self {
            config,
            module,
            value_map: FxHashMap::default(),
            block_map: FxHashMap::default(),
            next_value_id: 0,
            next_block_id: 0,
            epistemic_shadows: FxHashMap::default(),
            value_types: FxHashMap::default(),
            locals: FxHashMap::default(),
            param_values: Vec::new(),
        }
    }

    /// Allocate a new GPU value ID
    fn alloc_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Allocate a new GPU block ID
    fn alloc_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Lower the entire HLIR module
    fn lower_module(&mut self, hlir: &HlirModule) -> GpuModule {
        // Lower all functions
        for func in &hlir.functions {
            if func.is_kernel {
                let kernel = self.lower_kernel(func);
                self.module.add_kernel(kernel);
            } else {
                let device_func = self.lower_device_function(func);
                self.module.add_device_function(device_func);
            }
        }

        // Lower global constants
        for global in &hlir.globals {
            if global.is_const
                && let Some(init) = &global.init
            {
                let constant = GpuConstant {
                    name: global.name.clone(),
                    ty: self.lower_type(&global.ty),
                    value: self.lower_const_value(init),
                };
                self.module.add_constant(constant);
            }
        }

        std::mem::replace(
            &mut self.module,
            GpuModule::new("empty", self.config.target),
        )
    }

    /// Lower an HLIR kernel function to GPU kernel
    fn lower_kernel(&mut self, func: &HlirFunction) -> GpuKernel {
        // Reset per-function state
        self.value_map.clear();
        self.block_map.clear();
        self.epistemic_shadows.clear();
        self.locals.clear();
        self.param_values.clear();
        self.next_value_id = 0;
        self.next_block_id = 0;

        let mut kernel = GpuKernel::new(&func.name);

        // Set hints
        kernel.max_threads = self.config.max_threads_per_block;

        // Lower parameters
        for (idx, param) in func.params.iter().enumerate() {
            let gpu_type = self.lower_type(&param.ty);
            let gpu_param = GpuParam {
                name: param.name.clone(),
                ty: gpu_type.clone(),
                space: MemorySpace::Global,
                restrict: true,
            };
            kernel.add_param(gpu_param);

            // Map parameter to value
            let value_id = self.alloc_value();
            self.value_map.insert(param.value, value_id);
            self.param_values.push(value_id);
            self.value_types.insert(value_id, gpu_type);

            // Create epistemic shadows if this is a Knowledge type
            if self.config.epistemic_enabled && self.is_epistemic_param(&param.ty) {
                self.create_epistemic_shadow(value_id);
            }
        }

        // Pre-allocate block IDs
        for block in &func.blocks {
            let gpu_block_id = self.alloc_block();
            self.block_map.insert(block.id, gpu_block_id);
        }

        // Lower blocks
        for hlir_block in &func.blocks {
            let gpu_block = self.lower_block(hlir_block);
            kernel.add_block(gpu_block);
        }

        kernel.entry = BlockId(0);
        kernel
    }

    /// Lower an HLIR device function
    fn lower_device_function(&mut self, func: &HlirFunction) -> GpuFunction {
        // Reset per-function state
        self.value_map.clear();
        self.block_map.clear();
        self.epistemic_shadows.clear();
        self.locals.clear();
        self.param_values.clear();
        self.next_value_id = 0;
        self.next_block_id = 0;

        let return_type = self.lower_type(&func.return_type);
        let mut device_func = GpuFunction::new(&func.name, return_type);

        // Lower parameters
        for param in &func.params {
            let gpu_type = self.lower_type(&param.ty);
            let gpu_param = GpuParam {
                name: param.name.clone(),
                ty: gpu_type.clone(),
                space: MemorySpace::Generic,
                restrict: false,
            };
            device_func.add_param(gpu_param);

            let value_id = self.alloc_value();
            self.value_map.insert(param.value, value_id);
            self.param_values.push(value_id);
            self.value_types.insert(value_id, gpu_type);
        }

        // Pre-allocate block IDs
        for block in &func.blocks {
            let gpu_block_id = self.alloc_block();
            self.block_map.insert(block.id, gpu_block_id);
        }

        // Lower blocks
        for hlir_block in &func.blocks {
            let gpu_block = self.lower_block(hlir_block);
            device_func.add_block(gpu_block);
        }

        device_func.entry = BlockId(0);
        device_func
    }

    /// Lower an HLIR basic block
    fn lower_block(&mut self, hlir_block: &HlirBlock) -> GpuBlock {
        let gpu_block_id = self.block_map[&hlir_block.id];
        let mut gpu_block = GpuBlock::new(gpu_block_id, &hlir_block.label);

        // Lower block parameters (phi predecessors)
        for (value_id, ty) in &hlir_block.params {
            let gpu_value = self.alloc_value();
            self.value_map.insert(*value_id, gpu_value);
            self.value_types.insert(gpu_value, self.lower_type(ty));
        }

        // Lower instructions
        for instr in &hlir_block.instructions {
            self.lower_instruction(&mut gpu_block, instr);
        }

        // Lower terminator
        let terminator = self.lower_terminator(&hlir_block.terminator);
        gpu_block.set_terminator(terminator);

        gpu_block
    }

    /// Lower an HLIR instruction
    fn lower_instruction(&mut self, block: &mut GpuBlock, instr: &HlirInstr) {
        let result_type = self.lower_type(&instr.ty);

        match &instr.op {
            // Constants
            Op::Const(constant) => {
                if let Some(result) = instr.result {
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    let gpu_op = self.lower_constant(constant, &result_type);
                    block.add_instruction(gpu_value, gpu_op);

                    // Create epistemic shadow with zero uncertainty for constants
                    if self.config.epistemic_enabled {
                        self.create_constant_epistemic_shadow(block, gpu_value);
                    }
                }
            }

            // Copy/move
            Op::Copy(src) => {
                if let Some(result) = instr.result {
                    let src_gpu = self.get_value(*src);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    // For GPU, copy is just a move
                    let gpu_op = match &result_type {
                        GpuType::F32 | GpuType::F64 => {
                            GpuOp::FAdd(src_gpu, self.emit_zero(block, &result_type))
                        }
                        _ => GpuOp::Add(src_gpu, self.emit_zero_int(block, &result_type)),
                    };
                    block.add_instruction(gpu_value, gpu_op);

                    // Propagate epistemic shadow
                    if self.config.epistemic_enabled {
                        self.propagate_epistemic_shadow(src_gpu, gpu_value);
                    }
                }
            }

            // Binary operations
            Op::Binary { op, left, right } => {
                if let Some(result) = instr.result {
                    let left_gpu = self.get_value(*left);
                    let right_gpu = self.get_value(*right);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    let gpu_op = self.lower_binary_op(*op, left_gpu, right_gpu, &result_type);
                    block.add_instruction(gpu_value, gpu_op);

                    // Propagate epistemic uncertainty
                    if self.config.epistemic_enabled {
                        self.emit_epistemic_binary(block, gpu_value, *op, left_gpu, right_gpu);
                    }
                }
            }

            // Unary operations
            Op::Unary { op, operand } => {
                if let Some(result) = instr.result {
                    let operand_gpu = self.get_value(*operand);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    let gpu_op = self.lower_unary_op(*op, operand_gpu, &result_type);
                    block.add_instruction(gpu_value, gpu_op);

                    // Propagate epistemic shadow
                    if self.config.epistemic_enabled {
                        self.emit_epistemic_unary(block, gpu_value, *op, operand_gpu);
                    }
                }
            }

            // Function calls
            Op::Call { func, args } => {
                let func_name = format!("func_{}", func.0);
                let gpu_args: Vec<ValueId> = args.iter().map(|a| self.get_value(*a)).collect();

                if let Some(result) = instr.result {
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());
                    block.add_instruction(gpu_value, GpuOp::Call(func_name, gpu_args));
                } else {
                    let gpu_value = self.alloc_value();
                    block.add_instruction(gpu_value, GpuOp::Call(func_name, gpu_args));
                }
            }

            Op::CallDirect { name, args } => {
                let func_name = name.clone();

                let gpu_args: Vec<ValueId> = args.iter().map(|a| self.get_value(*a)).collect();

                if let Some(result) = instr.result {
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());
                    block.add_instruction(gpu_value, GpuOp::Call(func_name, gpu_args));
                } else {
                    let gpu_value = self.alloc_value();
                    block.add_instruction(gpu_value, GpuOp::Call(func_name, gpu_args));
                }
            }

            // Memory operations
            Op::Load { ptr } => {
                if let Some(result) = instr.result {
                    let ptr_gpu = self.get_value(*ptr);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());
                    block.add_instruction(gpu_value, GpuOp::Load(ptr_gpu, MemorySpace::Global));

                    if self.config.epistemic_enabled {
                        // Load epsilon shadow from adjacent memory
                        self.emit_epistemic_load(block, gpu_value, ptr_gpu);
                    }
                }
            }

            Op::Store { ptr, value } => {
                let ptr_gpu = self.get_value(*ptr);
                let value_gpu = self.get_value(*value);
                let store_id = self.alloc_value();
                block.add_instruction(
                    store_id,
                    GpuOp::Store(ptr_gpu, value_gpu, MemorySpace::Global),
                );

                if self.config.epistemic_enabled {
                    // Store epsilon shadow to adjacent memory
                    self.emit_epistemic_store(block, ptr_gpu, value_gpu);
                }
            }

            // Stack allocation
            Op::Alloca { ty } => {
                if let Some(result) = instr.result {
                    let gpu_type = self.lower_type(ty);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    // Alloca returns a pointer in local memory
                    let ptr_type = GpuType::Ptr(Box::new(gpu_type), MemorySpace::Local);
                    self.value_types.insert(gpu_value, ptr_type);
                    // On GPU, we emit a placeholder - actual allocation handled by register allocator
                    block.add_instruction(gpu_value, GpuOp::ConstInt(0, GpuType::U64));
                }
            }

            // GEP
            Op::GetFieldPtr { base, field } => {
                if let Some(result) = instr.result {
                    let base_gpu = self.get_value(*base);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    // Calculate field offset
                    let offset_value = self.alloc_value();
                    block.add_instruction(
                        offset_value,
                        GpuOp::ConstInt(*field as i64 * 8, GpuType::U64),
                    );
                    block.add_instruction(
                        gpu_value,
                        GpuOp::GetElementPtr(base_gpu, vec![offset_value]),
                    );
                }
            }

            Op::GetElementPtr { base, index } => {
                if let Some(result) = instr.result {
                    let base_gpu = self.get_value(*base);
                    let index_gpu = self.get_value(*index);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());
                    block.add_instruction(
                        gpu_value,
                        GpuOp::GetElementPtr(base_gpu, vec![index_gpu]),
                    );
                }
            }

            // Type cast
            Op::Cast { value, target, .. } => {
                if let Some(result) = instr.result {
                    let value_gpu = self.get_value(*value);
                    let gpu_value = self.alloc_value();
                    let target_gpu_type = self.lower_type(target);
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, target_gpu_type.clone());

                    let gpu_op = self.emit_cast(value_gpu, &target_gpu_type);
                    block.add_instruction(gpu_value, gpu_op);

                    // Preserve epistemic shadow through cast
                    if self.config.epistemic_enabled {
                        self.propagate_epistemic_shadow(value_gpu, gpu_value);
                    }
                }
            }

            // Phi node
            Op::Phi { incoming } => {
                if let Some(result) = instr.result {
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    let gpu_incoming: Vec<(BlockId, ValueId)> = incoming
                        .iter()
                        .map(|(block_id, value_id)| {
                            (self.block_map[block_id], self.get_value(*value_id))
                        })
                        .collect();
                    block.add_instruction(gpu_value, GpuOp::Phi(gpu_incoming));
                }
            }

            // Aggregate operations
            Op::ExtractValue { base, index } => {
                if let Some(result) = instr.result {
                    let base_gpu = self.get_value(*base);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    // Extract via pointer arithmetic
                    let offset = self.alloc_value();
                    block.add_instruction(offset, GpuOp::ConstInt(*index as i64 * 8, GpuType::U64));
                    let ptr = self.alloc_value();
                    block.add_instruction(ptr, GpuOp::GetElementPtr(base_gpu, vec![offset]));
                    block.add_instruction(gpu_value, GpuOp::Load(ptr, MemorySpace::Local));
                }
            }

            Op::InsertValue { base, value, index } => {
                if let Some(result) = instr.result {
                    let base_gpu = self.get_value(*base);
                    let value_gpu = self.get_value(*value);
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    // Copy base, then store new value
                    // This is simplified - real impl would handle properly
                    let offset = self.alloc_value();
                    block.add_instruction(offset, GpuOp::ConstInt(*index as i64 * 8, GpuType::U64));
                    let ptr = self.alloc_value();
                    block.add_instruction(ptr, GpuOp::GetElementPtr(base_gpu, vec![offset]));
                    block.add_instruction(
                        gpu_value,
                        GpuOp::Store(ptr, value_gpu, MemorySpace::Local),
                    );
                }
            }

            // Tuple/Array/Struct construction
            Op::Tuple(elements) | Op::Array(elements) => {
                // For GPU, we allocate local memory and store elements
                if let Some(result) = instr.result {
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    // Store elements sequentially
                    for (i, elem) in elements.iter().enumerate() {
                        let elem_gpu = self.get_value(*elem);
                        let offset = self.alloc_value();
                        block.add_instruction(offset, GpuOp::ConstInt(i as i64 * 8, GpuType::U64));
                        let ptr = self.alloc_value();
                        block.add_instruction(ptr, GpuOp::GetElementPtr(gpu_value, vec![offset]));
                        let store_id = self.alloc_value();
                        block.add_instruction(
                            store_id,
                            GpuOp::Store(ptr, elem_gpu, MemorySpace::Local),
                        );
                    }
                }
            }

            Op::Struct { name, fields } => {
                if let Some(result) = instr.result {
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());

                    // Store fields sequentially
                    for (i, (_, field_value)) in fields.iter().enumerate() {
                        let field_gpu = self.get_value(*field_value);
                        let offset = self.alloc_value();
                        block.add_instruction(offset, GpuOp::ConstInt(i as i64 * 8, GpuType::U64));
                        let ptr = self.alloc_value();
                        block.add_instruction(ptr, GpuOp::GetElementPtr(gpu_value, vec![offset]));
                        let store_id = self.alloc_value();
                        block.add_instruction(
                            store_id,
                            GpuOp::Store(ptr, field_gpu, MemorySpace::Local),
                        );
                    }
                }
            }

            // Effect operations - emit as calls to runtime
            Op::PerformEffect { effect, op, args } => {
                let gpu_args: Vec<ValueId> = args.iter().map(|a| self.get_value(*a)).collect();
                let effect_func = format!("__sounio_effect_{}_{}", effect, op);

                if let Some(result) = instr.result {
                    let gpu_value = self.alloc_value();
                    self.value_map.insert(result, gpu_value);
                    self.value_types.insert(gpu_value, result_type.clone());
                    block.add_instruction(gpu_value, GpuOp::Call(effect_func, gpu_args));
                } else {
                    let gpu_value = self.alloc_value();
                    block.add_instruction(gpu_value, GpuOp::Call(effect_func, gpu_args));
                }
            }
        }
    }

    /// Lower a terminator
    fn lower_terminator(&self, term: &HlirTerminator) -> GpuTerminator {
        match term {
            HlirTerminator::Return(None) => GpuTerminator::ReturnVoid,
            HlirTerminator::Return(Some(value)) => GpuTerminator::Return(self.get_value(*value)),
            HlirTerminator::Branch(target) => GpuTerminator::Br(self.block_map[target]),
            HlirTerminator::CondBranch {
                condition,
                then_block,
                else_block,
            } => {
                let cond_gpu = self.get_value(*condition);
                GpuTerminator::CondBr(
                    cond_gpu,
                    self.block_map[then_block],
                    self.block_map[else_block],
                )
            }
            HlirTerminator::Switch {
                value,
                default,
                cases,
            } => {
                // Lower switch to series of conditional branches
                // For simplicity, we just branch to default
                // Real implementation would emit proper switch
                GpuTerminator::Br(self.block_map[default])
            }
            HlirTerminator::Unreachable => GpuTerminator::Unreachable,
        }
    }

    /// Lower HLIR type to GPU type
    fn lower_type(&self, ty: &HlirType) -> GpuType {
        match ty {
            HlirType::Void => GpuType::Void,
            HlirType::Bool => GpuType::Bool,
            HlirType::I8 => GpuType::I8,
            HlirType::I16 => GpuType::I16,
            HlirType::I32 => GpuType::I32,
            HlirType::I64 => GpuType::I64,
            HlirType::I128 => GpuType::I64, // Downgrade to 64-bit on GPU
            HlirType::U8 => GpuType::U8,
            HlirType::U16 => GpuType::U16,
            HlirType::U32 => GpuType::U32,
            HlirType::U64 => GpuType::U64,
            HlirType::U128 => GpuType::U64, // Downgrade
            HlirType::F32 => GpuType::F32,
            HlirType::F64 => GpuType::F64,
            HlirType::Ptr(inner) => {
                GpuType::Ptr(Box::new(self.lower_type(inner)), MemorySpace::Global)
            }
            HlirType::Array(elem, size) => {
                GpuType::Array(Box::new(self.lower_type(elem)), *size as u32)
            }
            HlirType::Struct(name) => GpuType::Struct(name.clone(), vec![]),
            HlirType::Tuple(elems) => {
                let gpu_elems: Vec<(String, GpuType)> = elems
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (format!("_{}", i), self.lower_type(t)))
                    .collect();
                GpuType::Struct("tuple".to_string(), gpu_elems)
            }
            HlirType::Function { .. } => GpuType::U64, // Function pointer
            // Linear algebra primitives - map to GPU vector types (f32 components)
            HlirType::Vec2 => GpuType::Vec2(Box::new(GpuType::F32)),
            HlirType::Vec3 => GpuType::Vec3(Box::new(GpuType::F32)),
            HlirType::Vec4 => GpuType::Vec4(Box::new(GpuType::F32)),
            // Matrices stored as arrays of column vectors
            HlirType::Mat2 => GpuType::Array(Box::new(GpuType::Vec2(Box::new(GpuType::F32))), 2),
            HlirType::Mat3 => GpuType::Array(Box::new(GpuType::Vec3(Box::new(GpuType::F32))), 3),
            HlirType::Mat4 => GpuType::Array(Box::new(GpuType::Vec4(Box::new(GpuType::F32))), 4),
            // Quaternion as vec4 (x, y, z, w)
            HlirType::Quat => GpuType::Vec4(Box::new(GpuType::F32)),
            // Dual number as vec2 of f64 (value, derivative)
            HlirType::Dual => GpuType::Vec2(Box::new(GpuType::F64)),
        }
    }

    /// Lower constant to GPU operation
    fn lower_constant(&self, constant: &HlirConstant, ty: &GpuType) -> GpuOp {
        match constant {
            HlirConstant::Unit => GpuOp::ConstInt(0, GpuType::U32),
            HlirConstant::Bool(b) => GpuOp::ConstBool(*b),
            HlirConstant::Int(n, _) => GpuOp::ConstInt(*n, ty.clone()),
            HlirConstant::Float(f, _) => GpuOp::ConstFloat(*f, ty.clone()),
            HlirConstant::String(_) => GpuOp::ConstInt(0, GpuType::U64), // Pointer to string
            HlirConstant::Null(_) => GpuOp::ConstInt(0, GpuType::U64),
            HlirConstant::Undef(_) => GpuOp::ConstInt(0, ty.clone()),
            HlirConstant::FunctionRef(_) => GpuOp::ConstInt(0, GpuType::U64),
            HlirConstant::GlobalRef(_) => GpuOp::ConstInt(0, GpuType::U64),
            HlirConstant::Array(_) => GpuOp::ConstInt(0, GpuType::U64), // Array pointer
            HlirConstant::Struct(_) => GpuOp::ConstInt(0, GpuType::U64), // Struct pointer
        }
    }

    /// Lower constant value for global
    fn lower_const_value(&self, constant: &HlirConstant) -> GpuConstValue {
        match constant {
            HlirConstant::Unit => GpuConstValue::Int(0),
            HlirConstant::Bool(b) => GpuConstValue::Bool(*b),
            HlirConstant::Int(n, _) => GpuConstValue::Int(*n),
            HlirConstant::Float(f, _) => GpuConstValue::Float(*f),
            HlirConstant::Array(elems) => {
                GpuConstValue::Array(elems.iter().map(|e| self.lower_const_value(e)).collect())
            }
            HlirConstant::Struct(fields) => {
                GpuConstValue::Struct(fields.iter().map(|f| self.lower_const_value(f)).collect())
            }
            _ => GpuConstValue::Int(0),
        }
    }

    /// Lower binary operation
    fn lower_binary_op(&self, op: BinaryOp, left: ValueId, right: ValueId, ty: &GpuType) -> GpuOp {
        match op {
            // Integer arithmetic
            BinaryOp::Add => GpuOp::Add(left, right),
            BinaryOp::Sub => GpuOp::Sub(left, right),
            BinaryOp::Mul => GpuOp::Mul(left, right),
            BinaryOp::SDiv | BinaryOp::UDiv => GpuOp::Div(left, right),
            BinaryOp::SRem | BinaryOp::URem => GpuOp::Rem(left, right),

            // Float arithmetic
            BinaryOp::FAdd => GpuOp::FAdd(left, right),
            BinaryOp::FSub => GpuOp::FSub(left, right),
            BinaryOp::FMul => GpuOp::FMul(left, right),
            BinaryOp::FDiv => GpuOp::FDiv(left, right),
            BinaryOp::FRem => GpuOp::FDiv(left, right), // Approximate

            // Bitwise
            BinaryOp::And => GpuOp::BitAnd(left, right),
            BinaryOp::Or => GpuOp::BitOr(left, right),
            BinaryOp::Xor => GpuOp::BitXor(left, right),
            BinaryOp::Shl => GpuOp::Shl(left, right),
            BinaryOp::AShr => GpuOp::Shr(left, right),
            BinaryOp::LShr => GpuOp::LShr(left, right),

            // Integer comparisons
            BinaryOp::Eq => GpuOp::Eq(left, right),
            BinaryOp::Ne => GpuOp::Ne(left, right),
            BinaryOp::SLt | BinaryOp::ULt => GpuOp::Lt(left, right),
            BinaryOp::SLe | BinaryOp::ULe => GpuOp::Le(left, right),
            BinaryOp::SGt | BinaryOp::UGt => GpuOp::Gt(left, right),
            BinaryOp::SGe | BinaryOp::UGe => GpuOp::Ge(left, right),

            // Float comparisons
            BinaryOp::FOEq => GpuOp::FEq(left, right),
            BinaryOp::FONe => GpuOp::FNe(left, right),
            BinaryOp::FOLt => GpuOp::FLt(left, right),
            BinaryOp::FOLe => GpuOp::FLe(left, right),
            BinaryOp::FOGt => GpuOp::FGt(left, right),
            BinaryOp::FOGe => GpuOp::FGe(left, right),

            // Array concatenation (not supported on GPU, fallback to add)
            BinaryOp::Concat => GpuOp::Add(left, right),
        }
    }

    /// Lower unary operation
    fn lower_unary_op(&self, op: UnaryOp, operand: ValueId, ty: &GpuType) -> GpuOp {
        match op {
            UnaryOp::Neg => {
                if ty.is_float() {
                    GpuOp::FNeg(operand)
                } else {
                    GpuOp::Neg(operand)
                }
            }
            UnaryOp::FNeg => GpuOp::FNeg(operand),
            UnaryOp::Not => GpuOp::Not(operand),
        }
    }

    /// Emit a cast operation
    fn emit_cast(&self, value: ValueId, target: &GpuType) -> GpuOp {
        // Simplified - real implementation would check source type
        match target {
            GpuType::F32 => GpuOp::SiToFp(value, GpuType::F32),
            GpuType::F64 => GpuOp::SiToFp(value, GpuType::F64),
            GpuType::I32 => GpuOp::FpToSi(value, GpuType::I32),
            GpuType::I64 => GpuOp::FpToSi(value, GpuType::I64),
            GpuType::U32 => GpuOp::FpToUi(value, GpuType::U32),
            GpuType::U64 => GpuOp::FpToUi(value, GpuType::U64),
            _ => GpuOp::Bitcast(value, target.clone()),
        }
    }

    /// Get GPU value for HLIR value
    fn get_value(&self, hlir_value: HlirValueId) -> ValueId {
        *self.value_map.get(&hlir_value).unwrap_or(&ValueId(0))
    }

    /// Check if type is epistemic (Knowledge type)
    fn is_epistemic_param(&self, ty: &HlirType) -> bool {
        // Knowledge types are lowered to structs with epistemic prefix
        matches!(ty, HlirType::Struct(name) if name.starts_with("Knowledge") || name.starts_with("Epistemic"))
    }

    /// Emit zero constant
    fn emit_zero(&mut self, block: &mut GpuBlock, ty: &GpuType) -> ValueId {
        let value = self.alloc_value();
        let op = match ty {
            GpuType::F32 => GpuOp::ConstFloat(0.0, GpuType::F32),
            GpuType::F64 => GpuOp::ConstFloat(0.0, GpuType::F64),
            _ => GpuOp::ConstInt(0, ty.clone()),
        };
        block.add_instruction(value, op);
        self.value_types.insert(value, ty.clone());
        value
    }

    /// Emit zero integer constant
    fn emit_zero_int(&mut self, block: &mut GpuBlock, ty: &GpuType) -> ValueId {
        let value = self.alloc_value();
        block.add_instruction(value, GpuOp::ConstInt(0, ty.clone()));
        self.value_types.insert(value, ty.clone());
        value
    }

    // ========================================================================
    // EPISTEMIC STATE TRACKING - Shadow Registers
    // ========================================================================

    /// Create epistemic shadow for a value (ε, validity, provenance)
    fn create_epistemic_shadow(&mut self, value: ValueId) {
        let epsilon = self.alloc_value();
        let validity = self.alloc_value();
        let provenance = self.alloc_value();

        self.value_types.insert(epsilon, GpuType::F32);
        self.value_types.insert(validity, GpuType::Bool);
        self.value_types.insert(provenance, GpuType::U64);

        self.epistemic_shadows.insert(
            value,
            EpistemicShadow {
                epsilon,
                validity,
                provenance,
            },
        );
    }

    /// Create epistemic shadow for constant (zero uncertainty)
    fn create_constant_epistemic_shadow(&mut self, block: &mut GpuBlock, value: ValueId) {
        let epsilon = self.alloc_value();
        let validity = self.alloc_value();
        let provenance = self.alloc_value();

        // Constants have zero uncertainty
        block.add_instruction(epsilon, GpuOp::ConstFloat(0.0, GpuType::F32));
        // Constants are always valid
        block.add_instruction(validity, GpuOp::ConstBool(true));
        // Constants have no provenance (all zeros)
        block.add_instruction(provenance, GpuOp::ConstInt(0, GpuType::U64));

        self.value_types.insert(epsilon, GpuType::F32);
        self.value_types.insert(validity, GpuType::Bool);
        self.value_types.insert(provenance, GpuType::U64);

        self.epistemic_shadows.insert(
            value,
            EpistemicShadow {
                epsilon,
                validity,
                provenance,
            },
        );
    }

    /// Propagate epistemic shadow from source to destination
    fn propagate_epistemic_shadow(&mut self, src: ValueId, dst: ValueId) {
        if let Some(src_shadow) = self.epistemic_shadows.get(&src).cloned() {
            self.epistemic_shadows.insert(dst, src_shadow);
        }
    }

    /// Emit epistemic propagation for binary operations
    fn emit_epistemic_binary(
        &mut self,
        block: &mut GpuBlock,
        result: ValueId,
        op: BinaryOp,
        left: ValueId,
        right: ValueId,
    ) {
        let left_shadow = self.epistemic_shadows.get(&left).cloned();
        let right_shadow = self.epistemic_shadows.get(&right).cloned();

        if left_shadow.is_none() && right_shadow.is_none() {
            return;
        }

        // Allocate shadow values for result
        let result_epsilon = self.alloc_value();
        let result_validity = self.alloc_value();
        let result_provenance = self.alloc_value();

        self.value_types.insert(result_epsilon, GpuType::F32);
        self.value_types.insert(result_validity, GpuType::Bool);
        self.value_types.insert(result_provenance, GpuType::U64);

        // Get or create default shadows
        let left_eps = left_shadow.as_ref().map(|s| s.epsilon);
        let right_eps = right_shadow.as_ref().map(|s| s.epsilon);
        let left_valid = left_shadow.as_ref().map(|s| s.validity);
        let right_valid = right_shadow.as_ref().map(|s| s.validity);
        let left_prov = left_shadow.as_ref().map(|s| s.provenance);
        let right_prov = right_shadow.as_ref().map(|s| s.provenance);

        // Epsilon propagation depends on operation
        match op {
            // Additive: ε_result = ε_left + ε_right (quadrature would be sqrt(ε_l² + ε_r²))
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::FAdd | BinaryOp::FSub => {
                if let (Some(l_eps), Some(r_eps)) = (left_eps, right_eps) {
                    // Quadrature: sqrt(ε_l² + ε_r²)
                    let l_sq = self.alloc_value();
                    let r_sq = self.alloc_value();
                    let sum_sq = self.alloc_value();

                    block.add_instruction(l_sq, GpuOp::FMul(l_eps, l_eps));
                    block.add_instruction(r_sq, GpuOp::FMul(r_eps, r_eps));
                    block.add_instruction(sum_sq, GpuOp::FAdd(l_sq, r_sq));
                    block.add_instruction(result_epsilon, GpuOp::FastSqrt(sum_sq));
                } else if let Some(l_eps) = left_eps {
                    // Just copy left epsilon - emit zero first to avoid borrow issue
                    let zero = self.alloc_value();
                    block.add_instruction(zero, GpuOp::ConstFloat(0.0, GpuType::F32));
                    block.add_instruction(result_epsilon, GpuOp::FAdd(l_eps, zero));
                } else if let Some(r_eps) = right_eps {
                    let zero = self.alloc_value();
                    block.add_instruction(zero, GpuOp::ConstFloat(0.0, GpuType::F32));
                    block.add_instruction(result_epsilon, GpuOp::FAdd(r_eps, zero));
                } else {
                    block.add_instruction(result_epsilon, GpuOp::ConstFloat(0.0, GpuType::F32));
                }
            }

            // Multiplicative: ε_result ≈ |a|·ε_b + |b|·ε_a (first-order)
            BinaryOp::Mul | BinaryOp::FMul => {
                if let (Some(l_eps), Some(r_eps)) = (left_eps, right_eps) {
                    // |left| * ε_right
                    let abs_left = self.alloc_value();
                    let term1 = self.alloc_value();
                    // For simplicity, we use the values directly (should use abs)
                    block.add_instruction(term1, GpuOp::FMul(left, r_eps));

                    // |right| * ε_left
                    let term2 = self.alloc_value();
                    block.add_instruction(term2, GpuOp::FMul(right, l_eps));

                    // Sum (conservative)
                    block.add_instruction(result_epsilon, GpuOp::FAdd(term1, term2));
                } else {
                    block.add_instruction(result_epsilon, GpuOp::ConstFloat(0.0, GpuType::F32));
                }
            }

            // Division: more complex, ε widens near zero
            BinaryOp::SDiv | BinaryOp::UDiv | BinaryOp::FDiv => {
                // Simplified: ε_result = (|a|·ε_b + |b|·ε_a) / b²
                if let (Some(l_eps), Some(r_eps)) = (left_eps, right_eps) {
                    let term1 = self.alloc_value();
                    block.add_instruction(term1, GpuOp::FMul(left, r_eps));
                    let term2 = self.alloc_value();
                    block.add_instruction(term2, GpuOp::FMul(right, l_eps));
                    let numer = self.alloc_value();
                    block.add_instruction(numer, GpuOp::FAdd(term1, term2));
                    let denom_sq = self.alloc_value();
                    block.add_instruction(denom_sq, GpuOp::FMul(right, right));
                    block.add_instruction(result_epsilon, GpuOp::FDiv(numer, denom_sq));
                } else {
                    block.add_instruction(result_epsilon, GpuOp::ConstFloat(0.0, GpuType::F32));
                }
            }

            // Comparisons: epsilon doesn't directly propagate, but we track validity
            _ => {
                block.add_instruction(result_epsilon, GpuOp::ConstFloat(0.0, GpuType::F32));
            }
        }

        // Validity: AND of both operands
        if let (Some(l_valid), Some(r_valid)) = (left_valid, right_valid) {
            block.add_instruction(result_validity, GpuOp::And(l_valid, r_valid));
        } else if let Some(l_valid) = left_valid {
            block.add_instruction(result_validity, GpuOp::And(l_valid, l_valid)); // Copy
        } else if let Some(r_valid) = right_valid {
            block.add_instruction(result_validity, GpuOp::And(r_valid, r_valid));
        } else {
            block.add_instruction(result_validity, GpuOp::ConstBool(true));
        }

        // Provenance: XOR merge
        if let (Some(l_prov), Some(r_prov)) = (left_prov, right_prov) {
            block.add_instruction(result_provenance, GpuOp::BitXor(l_prov, r_prov));
        } else if let Some(l_prov) = left_prov {
            block.add_instruction(result_provenance, GpuOp::BitOr(l_prov, l_prov));
        } else if let Some(r_prov) = right_prov {
            block.add_instruction(result_provenance, GpuOp::BitOr(r_prov, r_prov));
        } else {
            block.add_instruction(result_provenance, GpuOp::ConstInt(0, GpuType::U64));
        }

        self.epistemic_shadows.insert(
            result,
            EpistemicShadow {
                epsilon: result_epsilon,
                validity: result_validity,
                provenance: result_provenance,
            },
        );
    }

    /// Emit epistemic propagation for unary operations
    fn emit_epistemic_unary(
        &mut self,
        block: &mut GpuBlock,
        result: ValueId,
        op: UnaryOp,
        operand: ValueId,
    ) {
        if let Some(shadow) = self.epistemic_shadows.get(&operand).cloned() {
            // Unary ops preserve epsilon (negation doesn't change uncertainty magnitude)
            self.epistemic_shadows.insert(result, shadow);
        }
    }

    /// Emit epistemic load (load shadow values from adjacent memory)
    fn emit_epistemic_load(&mut self, block: &mut GpuBlock, value: ValueId, ptr: ValueId) {
        // Shadow layout in memory:
        // [value: T][epsilon: f32][validity: u8][padding][provenance: u64]

        let epsilon = self.alloc_value();
        let validity = self.alloc_value();
        let provenance = self.alloc_value();

        // Calculate shadow addresses
        let eps_offset = self.alloc_value();
        block.add_instruction(eps_offset, GpuOp::ConstInt(8, GpuType::U64)); // After value
        let eps_ptr = self.alloc_value();
        block.add_instruction(eps_ptr, GpuOp::GetElementPtr(ptr, vec![eps_offset]));
        block.add_instruction(epsilon, GpuOp::Load(eps_ptr, MemorySpace::Global));

        let valid_offset = self.alloc_value();
        block.add_instruction(valid_offset, GpuOp::ConstInt(12, GpuType::U64));
        let valid_ptr = self.alloc_value();
        block.add_instruction(valid_ptr, GpuOp::GetElementPtr(ptr, vec![valid_offset]));
        block.add_instruction(validity, GpuOp::Load(valid_ptr, MemorySpace::Global));

        let prov_offset = self.alloc_value();
        block.add_instruction(prov_offset, GpuOp::ConstInt(16, GpuType::U64));
        let prov_ptr = self.alloc_value();
        block.add_instruction(prov_ptr, GpuOp::GetElementPtr(ptr, vec![prov_offset]));
        block.add_instruction(provenance, GpuOp::Load(prov_ptr, MemorySpace::Global));

        self.value_types.insert(epsilon, GpuType::F32);
        self.value_types.insert(validity, GpuType::Bool);
        self.value_types.insert(provenance, GpuType::U64);

        self.epistemic_shadows.insert(
            value,
            EpistemicShadow {
                epsilon,
                validity,
                provenance,
            },
        );
    }

    /// Emit epistemic store (store shadow values to adjacent memory)
    fn emit_epistemic_store(&mut self, block: &mut GpuBlock, ptr: ValueId, value: ValueId) {
        if let Some(shadow) = self.epistemic_shadows.get(&value).cloned() {
            // Store epsilon
            let eps_offset = self.alloc_value();
            block.add_instruction(eps_offset, GpuOp::ConstInt(8, GpuType::U64));
            let eps_ptr = self.alloc_value();
            block.add_instruction(eps_ptr, GpuOp::GetElementPtr(ptr, vec![eps_offset]));
            let store_eps = self.alloc_value();
            block.add_instruction(
                store_eps,
                GpuOp::Store(eps_ptr, shadow.epsilon, MemorySpace::Global),
            );

            // Store validity
            let valid_offset = self.alloc_value();
            block.add_instruction(valid_offset, GpuOp::ConstInt(12, GpuType::U64));
            let valid_ptr = self.alloc_value();
            block.add_instruction(valid_ptr, GpuOp::GetElementPtr(ptr, vec![valid_offset]));
            let store_valid = self.alloc_value();
            block.add_instruction(
                store_valid,
                GpuOp::Store(valid_ptr, shadow.validity, MemorySpace::Global),
            );

            // Store provenance
            let prov_offset = self.alloc_value();
            block.add_instruction(prov_offset, GpuOp::ConstInt(16, GpuType::U64));
            let prov_ptr = self.alloc_value();
            block.add_instruction(prov_ptr, GpuOp::GetElementPtr(ptr, vec![prov_offset]));
            let store_prov = self.alloc_value();
            block.add_instruction(
                store_prov,
                GpuOp::Store(prov_ptr, shadow.provenance, MemorySpace::Global),
            );
        }
    }
}

// ============================================================================
// Public API Functions
// ============================================================================

/// Compile HLIR directly to PTX string
pub fn compile_to_ptx(hlir: &HlirModule, sm_version: (u32, u32)) -> String {
    let target = GpuTarget::Cuda {
        compute_capability: sm_version,
    };
    let gpu_module = lower(hlir, target);
    let mut codegen = super::ptx::PtxCodegen::new(sm_version);
    codegen.generate(&gpu_module)
}

/// Compile HLIR directly to PTX with epistemic tracking
pub fn compile_to_ptx_epistemic(
    hlir: &HlirModule,
    sm_version: (u32, u32),
    epistemic: bool,
) -> String {
    let config = LoweringConfig {
        target: GpuTarget::Cuda {
            compute_capability: sm_version,
        },
        epistemic_enabled: epistemic,
        ..Default::default()
    };
    let gpu_module = lower_with_config(hlir, &config);
    let mut codegen = super::ptx::PtxCodegen::new(sm_version);
    codegen.generate(&gpu_module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hlir::{FunctionId, HlirBlock, HlirFunction, HlirInstr, HlirParam, HlirTerminator};
    use std::collections::HashMap;

    fn create_test_module() -> HlirModule {
        let mut module = HlirModule::new("test");

        // Create a simple kernel function
        let func = HlirFunction {
            id: FunctionId(0),
            name: "test_kernel".to_string(),
            link_name: None,
            params: vec![HlirParam {
                value: HlirValueId(0),
                name: "data".to_string(),
                ty: HlirType::Ptr(Box::new(HlirType::F32)),
            }],
            return_type: HlirType::Void,
            effects: vec![],
            blocks: vec![HlirBlock {
                id: HlirBlockId(0),
                label: "entry".to_string(),
                params: vec![],
                instructions: vec![
                    // Load parameter
                    HlirInstr {
                        result: Some(HlirValueId(1)),
                        op: Op::Const(HlirConstant::Int(0, HlirType::I32)),
                        ty: HlirType::I32,
                    },
                ],
                terminator: HlirTerminator::Return(None),
            }],
            is_kernel: true,
            locals: HashMap::new(),
            is_variadic: false,
            abi: crate::ast::Abi::Rust,
            is_exported: false,
        };

        module.functions.push(func);
        module
    }

    #[test]
    fn test_basic_lowering() {
        let hlir = create_test_module();
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let gpu_module = lower(&hlir, target);

        assert_eq!(gpu_module.kernels.len(), 1);
        assert!(gpu_module.kernels.contains_key("test_kernel"));
    }

    #[test]
    fn test_ptx_compilation() {
        let hlir = create_test_module();
        let ptx = compile_to_ptx(&hlir, (7, 5));

        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".target sm_75"));
        assert!(ptx.contains(".entry test_kernel"));
    }

    #[test]
    fn test_epistemic_lowering() {
        let hlir = create_test_module();
        let config = LoweringConfig {
            target: GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
            epistemic_enabled: true,
            ..Default::default()
        };
        let gpu_module = lower_with_config(&hlir, &config);

        // Should have the kernel
        assert!(gpu_module.kernels.contains_key("test_kernel"));
    }

    #[test]
    fn test_type_lowering() {
        let lowering = HlirToGpuLowering::new(LoweringConfig::default());

        assert_eq!(lowering.lower_type(&HlirType::F32), GpuType::F32);
        assert_eq!(lowering.lower_type(&HlirType::I64), GpuType::I64);
        assert_eq!(lowering.lower_type(&HlirType::Bool), GpuType::Bool);

        // I128 downgrades to I64 on GPU
        assert_eq!(lowering.lower_type(&HlirType::I128), GpuType::I64);
    }
}
