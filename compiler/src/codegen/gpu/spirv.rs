//! SPIR-V Code Generator
//!
//! Generates SPIR-V binary from GPU IR for Vulkan and OpenCL.
//!
//! References:
//! - SPIR-V Specification: https://www.khronos.org/registry/SPIR-V/
//! - rspirv: https://docs.rs/rspirv/

use std::collections::HashMap;

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use spirv::Word;
use thiserror::Error;

use super::ir::*;

/// Error type for SPIR-V code generation.
#[derive(Error, Debug)]
pub enum SpirvError {
    /// Error from rspirv builder operations.
    #[error("Builder error: {0}")]
    BuilderError(String),

    /// When a type is not found in the type cache.
    #[error("Missing type: {0}")]
    MissingType(String),

    /// When a value ID is not in the values vector.
    #[error("Missing value ID: {0}")]
    MissingValue(u32),

    /// When a block ID is not found.
    #[error("Missing block ID: {0}")]
    MissingBlock(u32),

    /// Generic error for invalid operations.
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Wrap a builder result, converting to SpirvError
fn wrap_err<T>(result: Result<T, rspirv::dr::Error>) -> Result<T, SpirvError> {
    result.map_err(|e| SpirvError::BuilderError(format!("{:?}", e)))
}

/// SPIR-V code generator
pub struct SpirvCodegen {
    /// SPIR-V builder
    builder: Builder,

    /// Type cache: type key -> SPIR-V ID
    types: HashMap<String, Word>,

    /// Constant cache
    constants: HashMap<String, Word>,

    /// Variable cache
    variables: HashMap<String, Word>,

    /// Function cache
    functions: HashMap<String, Word>,

    /// Value to ID mapping
    values: Vec<Word>,

    /// Block to ID mapping
    blocks: HashMap<BlockId, Word>,

    /// Execution model
    execution_model: spirv::ExecutionModel,

    /// Target environment
    target_env: SpirvTarget,

    /// GLSL.std.450 extended instruction set ID (cached)
    glsl_ext: Option<Word>,
}

/// SPIR-V target environment
#[derive(Debug, Clone, Copy)]
pub enum SpirvTarget {
    Vulkan1_0,
    Vulkan1_1,
    Vulkan1_2,
    OpenCL1_2,
    OpenCL2_0,
}

impl Default for SpirvTarget {
    fn default() -> Self {
        SpirvTarget::Vulkan1_2
    }
}

impl SpirvCodegen {
    pub fn new(execution_model: spirv::ExecutionModel) -> Self {
        let mut builder = Builder::new();

        // Set up capabilities
        builder.capability(spirv::Capability::Shader);
        builder.capability(spirv::Capability::Int64);
        builder.capability(spirv::Capability::Float64);

        // Memory model
        builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

        Self {
            builder,
            types: HashMap::new(),
            constants: HashMap::new(),
            variables: HashMap::new(),
            functions: HashMap::new(),
            values: Vec::new(),
            blocks: HashMap::new(),
            execution_model,
            target_env: SpirvTarget::default(),
            glsl_ext: None,
        }
    }

    /// Create a new generator with a specific target
    pub fn with_target(execution_model: spirv::ExecutionModel, target: SpirvTarget) -> Self {
        let mut codegen = Self::new(execution_model);
        codegen.target_env = target;
        codegen
    }

    /// Get a type ID from the cache, returning an error if not found
    fn get_type(&self, key: &str) -> Result<Word, SpirvError> {
        self.types
            .get(key)
            .copied()
            .ok_or_else(|| SpirvError::MissingType(key.to_string()))
    }

    /// Get a value ID from the values vector, returning an error if out of bounds
    fn get_value(&self, idx: u32) -> Result<Word, SpirvError> {
        self.values
            .get(idx as usize)
            .copied()
            .ok_or_else(|| SpirvError::MissingValue(idx))
    }

    /// Get a block ID from the blocks map, returning an error if not found
    fn get_block(&self, id: &BlockId) -> Result<Word, SpirvError> {
        self.blocks
            .get(id)
            .copied()
            .ok_or_else(|| SpirvError::MissingBlock(id.0))
    }

    /// Get a variable ID from the variables map, returning an error if not found
    fn get_variable(&self, name: &str) -> Result<Word, SpirvError> {
        self.variables
            .get(name)
            .copied()
            .ok_or_else(|| SpirvError::InvalidOperation(format!("Missing variable: {}", name)))
    }

    /// Generate SPIR-V module from GPU module
    pub fn generate(mut self, module: &GpuModule) -> Result<Vec<u8>, SpirvError> {
        // Generate type definitions
        self.define_types();

        // Generate constants
        for constant in &module.constants {
            self.define_constant(constant)?;
        }

        // Generate kernels/functions
        for (_, kernel) in &module.kernels {
            self.generate_kernel(kernel)?;
        }

        // Build module (consumes the builder)
        let spirv_module = self.builder.module();
        let words = spirv_module.assemble();
        // Convert Vec<u32> to Vec<u8>
        Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
    }

    fn define_types(&mut self) {
        // Void
        let void_ty = self.builder.type_void();
        self.types.insert("void".to_string(), void_ty);

        // Bool
        let bool_ty = self.builder.type_bool();
        self.types.insert("bool".to_string(), bool_ty);

        // Integers
        let i8_ty = self.builder.type_int(8, 1);
        self.types.insert("i8".to_string(), i8_ty);

        let u8_ty = self.builder.type_int(8, 0);
        self.types.insert("u8".to_string(), u8_ty);

        let i16_ty = self.builder.type_int(16, 1);
        self.types.insert("i16".to_string(), i16_ty);

        let u16_ty = self.builder.type_int(16, 0);
        self.types.insert("u16".to_string(), u16_ty);

        let i32_ty = self.builder.type_int(32, 1);
        self.types.insert("i32".to_string(), i32_ty);

        let u32_ty = self.builder.type_int(32, 0);
        self.types.insert("u32".to_string(), u32_ty);

        let i64_ty = self.builder.type_int(64, 1);
        self.types.insert("i64".to_string(), i64_ty);

        let u64_ty = self.builder.type_int(64, 0);
        self.types.insert("u64".to_string(), u64_ty);

        // Floats
        let f32_ty = self.builder.type_float(32);
        self.types.insert("f32".to_string(), f32_ty);

        let f64_ty = self.builder.type_float(64);
        self.types.insert("f64".to_string(), f64_ty);

        // Pointer types for storage buffer
        let ptr_f32 = self
            .builder
            .type_pointer(None, spirv::StorageClass::StorageBuffer, f32_ty);
        self.types.insert("ptr_f32_storage".to_string(), ptr_f32);

        let ptr_i32 = self
            .builder
            .type_pointer(None, spirv::StorageClass::StorageBuffer, i32_ty);
        self.types.insert("ptr_i32_storage".to_string(), ptr_i32);

        let ptr_u32 = self
            .builder
            .type_pointer(None, spirv::StorageClass::StorageBuffer, u32_ty);
        self.types.insert("ptr_u32_storage".to_string(), ptr_u32);

        // Vector types
        let vec3_u32 = self.builder.type_vector(u32_ty, 3);
        self.types.insert("vec3_u32".to_string(), vec3_u32);

        let vec4_f32 = self.builder.type_vector(f32_ty, 4);
        self.types.insert("vec4_f32".to_string(), vec4_f32);

        let ptr_vec3_input = self
            .builder
            .type_pointer(None, spirv::StorageClass::Input, vec3_u32);
        self.types
            .insert("ptr_vec3_input".to_string(), ptr_vec3_input);

        // Function type (void -> void)
        let fn_void = self.builder.type_function(void_ty, vec![]);
        self.types.insert("fn_void".to_string(), fn_void);

        // Workgroup pointer for shared memory
        let ptr_f32_workgroup =
            self.builder
                .type_pointer(None, spirv::StorageClass::Workgroup, f32_ty);
        self.types
            .insert("ptr_f32_workgroup".to_string(), ptr_f32_workgroup);
    }

    fn define_constant(&mut self, constant: &GpuConstant) -> Result<(), SpirvError> {
        let id = match &constant.value {
            GpuConstValue::Int(n) => {
                let ty = self.gpu_type_to_spirv(&constant.ty)?;
                if constant.ty.size_bytes() <= 4 {
                    self.builder.constant_bit32(ty, *n as u32)
                } else {
                    self.builder.constant_bit64(ty, *n as u64)
                }
            }
            GpuConstValue::Float(n) => {
                let ty = self.gpu_type_to_spirv(&constant.ty)?;
                if matches!(constant.ty, GpuType::F32) {
                    self.builder.constant_bit32(ty, (*n as f32).to_bits())
                } else {
                    self.builder.constant_bit64(ty, n.to_bits())
                }
            }
            GpuConstValue::Bool(b) => {
                let ty = self.get_type("bool")?;
                if *b {
                    self.builder.constant_true(ty)
                } else {
                    self.builder.constant_false(ty)
                }
            }
            _ => {
                // Complex constants handled separately
                self.get_type("i64")? // Placeholder
            }
        };

        self.constants.insert(constant.name.clone(), id);
        Ok(())
    }

    fn generate_kernel(&mut self, kernel: &GpuKernel) -> Result<(), SpirvError> {
        // Reset per-kernel state
        self.values.clear();
        self.blocks.clear();

        // Create function type
        let void_ty = self.get_type("void")?;
        let fn_ty = self.get_type("fn_void")?;

        // Begin function
        let fn_id = self
            .builder
            .begin_function(void_ty, None, spirv::FunctionControl::NONE, fn_ty)
            .map_err(|e| SpirvError::BuilderError(format!("{:?}", e)))?;

        self.functions.insert(kernel.name.clone(), fn_id);

        // Define built-in variables for compute shader
        let interface_vars = self.define_builtin_variables()?;

        // Create entry block
        let entry_label = self
            .builder
            .begin_block(None)
            .map_err(|e| SpirvError::BuilderError(format!("{:?}", e)))?;

        // Generate instructions for each block
        for block in &kernel.blocks {
            if block.id.0 == 0 {
                // First block uses entry label
                self.blocks.insert(block.id, entry_label);
            } else {
                // Create new block
                let block_label = self
                    .builder
                    .begin_block(None)
                    .map_err(|e| SpirvError::BuilderError(format!("{:?}", e)))?;
                self.blocks.insert(block.id, block_label);
            }
        }

        // Generate instructions
        for block in &kernel.blocks {
            let _block_id = self.get_block(&block.id)?;
            // Select block would go here if we had multi-block support

            for (_value_id, op) in &block.instructions {
                let id = self.generate_op(op)?;
                self.values.push(id);
            }

            self.generate_terminator(&block.terminator)?;
        }

        self.builder
            .end_function()
            .map_err(|e| SpirvError::BuilderError(format!("{:?}", e)))?;

        // Add entry point
        self.builder
            .entry_point(self.execution_model, fn_id, &kernel.name, interface_vars);

        // Add execution mode for compute shaders
        if self.execution_model == spirv::ExecutionModel::GLCompute {
            let local_size = kernel.max_threads.unwrap_or(256);
            self.builder
                .execution_mode(fn_id, spirv::ExecutionMode::LocalSize, [local_size, 1, 1]);
        }

        Ok(())
    }

    fn define_builtin_variables(&mut self) -> Result<Vec<Word>, SpirvError> {
        let vec3_u32 = self.get_type("vec3_u32")?;
        let ptr_vec3 = self.get_type("ptr_vec3_input")?;

        let mut interface = Vec::new();

        // GlobalInvocationId
        let global_id = self
            .builder
            .variable(ptr_vec3, None, spirv::StorageClass::Input, None);
        self.builder.decorate(
            global_id,
            spirv::Decoration::BuiltIn,
            vec![Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)],
        );
        self.variables
            .insert("GlobalInvocationId".to_string(), global_id);
        interface.push(global_id);

        // LocalInvocationId
        let local_id = self
            .builder
            .variable(ptr_vec3, None, spirv::StorageClass::Input, None);
        self.builder.decorate(
            local_id,
            spirv::Decoration::BuiltIn,
            vec![Operand::BuiltIn(spirv::BuiltIn::LocalInvocationId)],
        );
        self.variables
            .insert("LocalInvocationId".to_string(), local_id);
        interface.push(local_id);

        // WorkgroupId
        let wg_id = self
            .builder
            .variable(ptr_vec3, None, spirv::StorageClass::Input, None);
        self.builder.decorate(
            wg_id,
            spirv::Decoration::BuiltIn,
            vec![Operand::BuiltIn(spirv::BuiltIn::WorkgroupId)],
        );
        self.variables.insert("WorkgroupId".to_string(), wg_id);
        interface.push(wg_id);

        // NumWorkgroups
        let num_wg = self
            .builder
            .variable(ptr_vec3, None, spirv::StorageClass::Input, None);
        self.builder.decorate(
            num_wg,
            spirv::Decoration::BuiltIn,
            vec![Operand::BuiltIn(spirv::BuiltIn::NumWorkgroups)],
        );
        self.variables.insert("NumWorkgroups".to_string(), num_wg);
        interface.push(num_wg);

        // WorkgroupSize
        let wg_size = self
            .builder
            .variable(ptr_vec3, None, spirv::StorageClass::Input, None);
        self.builder.decorate(
            wg_size,
            spirv::Decoration::BuiltIn,
            vec![Operand::BuiltIn(spirv::BuiltIn::WorkgroupSize)],
        );
        self.variables.insert("WorkgroupSize".to_string(), wg_size);
        interface.push(wg_size);

        Ok(interface)
    }

    /// Get or import the GLSL.std.450 extended instruction set
    fn get_or_import_glsl_ext(&mut self) -> Word {
        if let Some(ext) = self.glsl_ext {
            ext
        } else {
            let ext = self.builder.ext_inst_import("GLSL.std.450");
            self.glsl_ext = Some(ext);
            ext
        }
    }

    fn generate_op(&mut self, op: &GpuOp) -> Result<Word, SpirvError> {
        match op {
            GpuOp::ConstInt(n, ty) => {
                let spirv_ty = self.gpu_type_to_spirv(ty)?;
                Ok(if ty.size_bytes() <= 4 {
                    self.builder.constant_bit32(spirv_ty, *n as u32)
                } else {
                    self.builder.constant_bit64(spirv_ty, *n as u64)
                })
            }

            GpuOp::ConstFloat(n, ty) => {
                let spirv_ty = self.gpu_type_to_spirv(ty)?;
                Ok(if matches!(ty, GpuType::F32) {
                    self.builder.constant_bit32(spirv_ty, (*n as f32).to_bits())
                } else {
                    self.builder.constant_bit64(spirv_ty, n.to_bits())
                })
            }

            GpuOp::ConstBool(b) => {
                let ty = self.get_type("bool")?;
                Ok(if *b {
                    self.builder.constant_true(ty)
                } else {
                    self.builder.constant_false(ty)
                })
            }

            GpuOp::Add(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.i_add(ty, None, l, r))
            }

            GpuOp::Sub(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.i_sub(ty, None, l, r))
            }

            GpuOp::Mul(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.i_mul(ty, None, l, r))
            }

            GpuOp::Div(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.s_div(ty, None, l, r))
            }

            GpuOp::Rem(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.s_rem(ty, None, l, r))
            }

            GpuOp::Neg(val) => {
                let v = self.get_value(val.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.s_negate(ty, None, v))
            }

            GpuOp::FAdd(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("f32")?;
                wrap_err(self.builder.f_add(ty, None, l, r))
            }

            GpuOp::FSub(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("f32")?;
                wrap_err(self.builder.f_sub(ty, None, l, r))
            }

            GpuOp::FMul(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("f32")?;
                wrap_err(self.builder.f_mul(ty, None, l, r))
            }

            GpuOp::FDiv(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("f32")?;
                wrap_err(self.builder.f_div(ty, None, l, r))
            }

            GpuOp::FNeg(val) => {
                let v = self.get_value(val.0)?;
                let ty = self.get_type("f32")?;
                wrap_err(self.builder.f_negate(ty, None, v))
            }

            GpuOp::Load(ptr, _) => {
                let p = self.get_value(ptr.0)?;
                let ty = self.get_type("f32")?;
                wrap_err(self.builder.load(ty, None, p, None, vec![]))
            }

            GpuOp::Store(ptr, val, _) => {
                let p = self.get_value(ptr.0)?;
                let v = self.get_value(val.0)?;
                wrap_err(self.builder.store(p, v, None, vec![]))?;
                Ok(0) // Void
            }

            GpuOp::ThreadIdX => {
                let var = self.get_variable("LocalInvocationId")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![0]))
            }

            GpuOp::ThreadIdY => {
                let var = self.get_variable("LocalInvocationId")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![1]))
            }

            GpuOp::ThreadIdZ => {
                let var = self.get_variable("LocalInvocationId")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![2]))
            }

            GpuOp::BlockIdX => {
                let var = self.get_variable("WorkgroupId")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![0]))
            }

            GpuOp::BlockIdY => {
                let var = self.get_variable("WorkgroupId")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![1]))
            }

            GpuOp::BlockIdZ => {
                let var = self.get_variable("WorkgroupId")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![2]))
            }

            GpuOp::BlockDimX => {
                let var = self.get_variable("WorkgroupSize")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![0]))
            }

            GpuOp::BlockDimY => {
                let var = self.get_variable("WorkgroupSize")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![1]))
            }

            GpuOp::BlockDimZ => {
                let var = self.get_variable("WorkgroupSize")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![2]))
            }

            GpuOp::GridDimX => {
                let var = self.get_variable("NumWorkgroups")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![0]))
            }

            GpuOp::GridDimY => {
                let var = self.get_variable("NumWorkgroups")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![1]))
            }

            GpuOp::GridDimZ => {
                let var = self.get_variable("NumWorkgroups")?;
                let vec3_ty = self.get_type("vec3_u32")?;
                let u32_ty = self.get_type("u32")?;

                let vec = wrap_err(self.builder.load(vec3_ty, None, var, None, vec![]))?;
                wrap_err(self.builder.composite_extract(u32_ty, None, vec, vec![2]))
            }

            GpuOp::SyncThreads => {
                wrap_err(self.builder.control_barrier(
                    spirv::Scope::Workgroup as u32,
                    spirv::Scope::Workgroup as u32,
                    (spirv::MemorySemantics::WORKGROUP_MEMORY
                        | spirv::MemorySemantics::ACQUIRE_RELEASE)
                        .bits(),
                ))?;
                Ok(0) // Void
            }

            GpuOp::MemoryFence(_) => {
                wrap_err(self.builder.memory_barrier(
                    spirv::Scope::Workgroup as u32,
                    (spirv::MemorySemantics::WORKGROUP_MEMORY
                        | spirv::MemorySemantics::ACQUIRE_RELEASE)
                        .bits(),
                ))?;
                Ok(0) // Void
            }

            GpuOp::Lt(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.s_less_than(ty, None, l, r))
            }

            GpuOp::Le(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.s_less_than_equal(ty, None, l, r))
            }

            GpuOp::Gt(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.s_greater_than(ty, None, l, r))
            }

            GpuOp::Ge(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.s_greater_than_equal(ty, None, l, r))
            }

            GpuOp::Eq(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.i_equal(ty, None, l, r))
            }

            GpuOp::Ne(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.i_not_equal(ty, None, l, r))
            }

            GpuOp::FLt(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.f_ord_less_than(ty, None, l, r))
            }

            GpuOp::FLe(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.f_ord_less_than_equal(ty, None, l, r))
            }

            GpuOp::FGt(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.f_ord_greater_than(ty, None, l, r))
            }

            GpuOp::FGe(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.f_ord_greater_than_equal(ty, None, l, r))
            }

            GpuOp::FEq(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.f_ord_equal(ty, None, l, r))
            }

            GpuOp::FNe(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.f_ord_not_equal(ty, None, l, r))
            }

            GpuOp::And(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.logical_and(ty, None, l, r))
            }

            GpuOp::Or(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.logical_or(ty, None, l, r))
            }

            GpuOp::Not(val) => {
                let v = self.get_value(val.0)?;
                let ty = self.get_type("bool")?;
                wrap_err(self.builder.logical_not(ty, None, v))
            }

            GpuOp::BitAnd(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.bitwise_and(ty, None, l, r))
            }

            GpuOp::BitOr(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.bitwise_or(ty, None, l, r))
            }

            GpuOp::BitXor(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.bitwise_xor(ty, None, l, r))
            }

            GpuOp::BitNot(val) => {
                let v = self.get_value(val.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.not(ty, None, v))
            }

            GpuOp::Shl(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.shift_left_logical(ty, None, l, r))
            }

            GpuOp::Shr(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.shift_right_arithmetic(ty, None, l, r))
            }

            GpuOp::LShr(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("u32")?;
                wrap_err(self.builder.shift_right_logical(ty, None, l, r))
            }

            GpuOp::Select(cond, t, f) => {
                let c = self.get_value(cond.0)?;
                let tv = self.get_value(t.0)?;
                let fv = self.get_value(f.0)?;
                let ty = self.get_type("i32")?;
                wrap_err(self.builder.select(ty, None, c, tv, fv))
            }

            // GLSL.std.450 extended math operations
            GpuOp::AbsF32(val) => {
                let v = self.get_value(val.0)?;
                let ty = self.get_type("f32")?;
                let ext = self.get_or_import_glsl_ext();
                wrap_err(self.builder.ext_inst(ty, None, ext, 4, vec![Operand::IdRef(v)]))
            }

            GpuOp::AbsF64(val) => {
                let v = self.get_value(val.0)?;
                let ty = self.get_type("f64")?;
                let ext = self.get_or_import_glsl_ext();
                wrap_err(self.builder.ext_inst(ty, None, ext, 4, vec![Operand::IdRef(v)]))
            }

            GpuOp::AbsI32(val) => {
                let v = self.get_value(val.0)?;
                let ty = self.get_type("i32")?;
                let ext = self.get_or_import_glsl_ext();
                wrap_err(self.builder.ext_inst(ty, None, ext, 5, vec![Operand::IdRef(v)]))
            }

            GpuOp::MinF32(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("f32")?;
                let ext = self.get_or_import_glsl_ext();
                wrap_err(self.builder.ext_inst(ty, None, ext, 37, vec![Operand::IdRef(l), Operand::IdRef(r)]))
            }

            GpuOp::MaxF32(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("f32")?;
                let ext = self.get_or_import_glsl_ext();
                wrap_err(self.builder.ext_inst(ty, None, ext, 40, vec![Operand::IdRef(l), Operand::IdRef(r)]))
            }

            GpuOp::MinI32(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                let ext = self.get_or_import_glsl_ext();
                wrap_err(self.builder.ext_inst(ty, None, ext, 39, vec![Operand::IdRef(l), Operand::IdRef(r)]))
            }

            GpuOp::MaxI32(lhs, rhs) => {
                let l = self.get_value(lhs.0)?;
                let r = self.get_value(rhs.0)?;
                let ty = self.get_type("i32")?;
                let ext = self.get_or_import_glsl_ext();
                wrap_err(self.builder.ext_inst(ty, None, ext, 42, vec![Operand::IdRef(l), Operand::IdRef(r)]))
            }

            GpuOp::FMulAdd(a, b, c) => {
                let va = self.get_value(a.0)?;
                let vb = self.get_value(b.0)?;
                let vc = self.get_value(c.0)?;
                let ty = self.get_type("f32")?;
                let ext = self.get_or_import_glsl_ext();
                wrap_err(self.builder.ext_inst(ty, None, ext, 50, vec![Operand::IdRef(va), Operand::IdRef(vb), Operand::IdRef(vc)]))
            }

            _ => {
                // Placeholder for unimplemented ops
                let ty = self.get_type("i32")?;
                Ok(self.builder.constant_bit32(ty, 0))
            }
        }
    }

    fn generate_terminator(&mut self, term: &GpuTerminator) -> Result<(), SpirvError> {
        match term {
            GpuTerminator::Br(target) => {
                let block = self.get_block(target)?;
                wrap_err(self.builder.branch(block))?;
            }

            GpuTerminator::CondBr(cond, then_block, else_block) => {
                let c = self.get_value(cond.0)?;
                let then_b = self.get_block(then_block)?;
                let else_b = self.get_block(else_block)?;
                wrap_err(self.builder.branch_conditional(c, then_b, else_b, vec![]))?;
            }

            GpuTerminator::ReturnVoid => {
                wrap_err(self.builder.ret())?;
            }

            GpuTerminator::Return(val) => {
                let v = self.get_value(val.0)?;
                wrap_err(self.builder.ret_value(v))?;
            }

            GpuTerminator::Unreachable => {
                wrap_err(self.builder.unreachable())?;
            }
        }
        Ok(())
    }

    fn gpu_type_to_spirv(&self, ty: &GpuType) -> Result<Word, SpirvError> {
        match ty {
            GpuType::Void => self.get_type("void"),
            GpuType::Bool => self.get_type("bool"),
            GpuType::I8 => self.get_type("i8"),
            GpuType::U8 => self.get_type("u8"),
            GpuType::I16 => self.get_type("i16"),
            GpuType::U16 => self.get_type("u16"),
            GpuType::I32 => self.get_type("i32"),
            GpuType::U32 => self.get_type("u32"),
            GpuType::I64 => self.get_type("i64"),
            GpuType::U64 => self.get_type("u64"),
            GpuType::F32 => self.get_type("f32"),
            GpuType::F64 => self.get_type("f64"),
            _ => self.get_type("i32"), // Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spirv_generation() {
        let mut module = GpuModule::new("test", GpuTarget::Vulkan { version: (1, 2) });

        let mut kernel = GpuKernel::new("compute");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let codegen = SpirvCodegen::new(spirv::ExecutionModel::GLCompute);
        let spirv_bytes = codegen.generate(&module).unwrap();

        // SPIR-V magic number: 0x07230203
        assert!(spirv_bytes.len() >= 4);
        assert_eq!(spirv_bytes[0], 0x03);
        assert_eq!(spirv_bytes[1], 0x02);
        assert_eq!(spirv_bytes[2], 0x23);
        assert_eq!(spirv_bytes[3], 0x07);
    }

    #[test]
    fn test_spirv_arithmetic() {
        let mut module = GpuModule::new("test", GpuTarget::Vulkan { version: (1, 2) });

        let mut kernel = GpuKernel::new("math");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstInt(10, GpuType::I32));
        block.add_instruction(ValueId(1), GpuOp::ConstInt(20, GpuType::I32));
        block.add_instruction(ValueId(2), GpuOp::Add(ValueId(0), ValueId(1)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let codegen = SpirvCodegen::new(spirv::ExecutionModel::GLCompute);
        let spirv_bytes = codegen.generate(&module).unwrap();

        // Should produce valid SPIR-V
        assert!(!spirv_bytes.is_empty());
    }
}
