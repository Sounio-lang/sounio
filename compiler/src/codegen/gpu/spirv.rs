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

use super::ir::*;

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
        }
    }

    /// Create a new generator with a specific target
    pub fn with_target(execution_model: spirv::ExecutionModel, target: SpirvTarget) -> Self {
        let mut codegen = Self::new(execution_model);
        codegen.target_env = target;
        codegen
    }

    /// Generate SPIR-V module from GPU module
    pub fn generate(mut self, module: &GpuModule) -> Vec<u8> {
        // Generate type definitions
        self.define_types();

        // Generate constants
        for constant in &module.constants {
            self.define_constant(constant);
        }

        // Generate kernels/functions
        for (_, kernel) in &module.kernels {
            self.generate_kernel(kernel);
        }

        // Build module (consumes the builder)
        let spirv_module = self.builder.module();
        let words = spirv_module.assemble();
        // Convert Vec<u32> to Vec<u8>
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
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

    fn define_constant(&mut self, constant: &GpuConstant) {
        let id = match &constant.value {
            GpuConstValue::Int(n) => {
                let ty = self.gpu_type_to_spirv(&constant.ty);
                if constant.ty.size_bytes() <= 4 {
                    self.builder.constant_bit32(ty, *n as u32)
                } else {
                    self.builder.constant_bit64(ty, *n as u64)
                }
            }
            GpuConstValue::Float(n) => {
                let ty = self.gpu_type_to_spirv(&constant.ty);
                if matches!(constant.ty, GpuType::F32) {
                    self.builder.constant_bit32(ty, (*n as f32).to_bits())
                } else {
                    self.builder.constant_bit64(ty, n.to_bits())
                }
            }
            GpuConstValue::Bool(b) => {
                let ty = self.types["bool"];
                if *b {
                    self.builder.constant_true(ty)
                } else {
                    self.builder.constant_false(ty)
                }
            }
            _ => {
                // Complex constants handled separately
                self.types["i64"] // Placeholder
            }
        };

        self.constants.insert(constant.name.clone(), id);
    }

    fn generate_kernel(&mut self, kernel: &GpuKernel) {
        // Reset per-kernel state
        self.values.clear();
        self.blocks.clear();

        // Create function type
        let void_ty = self.types["void"];
        let fn_ty = self.types["fn_void"];

        // Begin function
        let fn_id = self
            .builder
            .begin_function(void_ty, None, spirv::FunctionControl::NONE, fn_ty)
            .unwrap();

        self.functions.insert(kernel.name.clone(), fn_id);

        // Define built-in variables for compute shader
        let interface_vars = self.define_builtin_variables();

        // Create entry block
        let entry_label = self.builder.begin_block(None).unwrap();

        // Generate instructions for each block
        for block in &kernel.blocks {
            if block.id.0 == 0 {
                // First block uses entry label
                self.blocks.insert(block.id, entry_label);
            } else {
                // Create new block
                let block_label = self.builder.begin_block(None).unwrap();
                self.blocks.insert(block.id, block_label);
            }
        }

        // Generate instructions
        for block in &kernel.blocks {
            let _block_id = self.blocks[&block.id];
            // Select block would go here if we had multi-block support

            for (_value_id, op) in &block.instructions {
                let id = self.generate_op(op);
                self.values.push(id);
            }

            self.generate_terminator(&block.terminator);
        }

        self.builder.end_function().unwrap();

        // Add entry point
        self.builder
            .entry_point(self.execution_model, fn_id, &kernel.name, interface_vars);

        // Add execution mode for compute shaders
        if self.execution_model == spirv::ExecutionModel::GLCompute {
            let local_size = kernel.max_threads.unwrap_or(256);
            self.builder
                .execution_mode(fn_id, spirv::ExecutionMode::LocalSize, [local_size, 1, 1]);
        }
    }

    fn define_builtin_variables(&mut self) -> Vec<Word> {
        let vec3_u32 = self.types["vec3_u32"];
        let ptr_vec3 = self.types["ptr_vec3_input"];

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

        interface
    }

    fn generate_op(&mut self, op: &GpuOp) -> Word {
        match op {
            GpuOp::ConstInt(n, ty) => {
                let spirv_ty = self.gpu_type_to_spirv(ty);
                if ty.size_bytes() <= 4 {
                    self.builder.constant_bit32(spirv_ty, *n as u32)
                } else {
                    self.builder.constant_bit64(spirv_ty, *n as u64)
                }
            }

            GpuOp::ConstFloat(n, ty) => {
                let spirv_ty = self.gpu_type_to_spirv(ty);
                if matches!(ty, GpuType::F32) {
                    self.builder.constant_bit32(spirv_ty, (*n as f32).to_bits())
                } else {
                    self.builder.constant_bit64(spirv_ty, n.to_bits())
                }
            }

            GpuOp::ConstBool(b) => {
                let ty = self.types["bool"];
                if *b {
                    self.builder.constant_true(ty)
                } else {
                    self.builder.constant_false(ty)
                }
            }

            GpuOp::Add(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.i_add(ty, None, l, r).unwrap()
            }

            GpuOp::Sub(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.i_sub(ty, None, l, r).unwrap()
            }

            GpuOp::Mul(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.i_mul(ty, None, l, r).unwrap()
            }

            GpuOp::Div(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.s_div(ty, None, l, r).unwrap()
            }

            GpuOp::Rem(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.s_rem(ty, None, l, r).unwrap()
            }

            GpuOp::Neg(val) => {
                let v = self.values[val.0 as usize];
                let ty = self.types["i32"];
                self.builder.s_negate(ty, None, v).unwrap()
            }

            GpuOp::FAdd(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["f32"];
                self.builder.f_add(ty, None, l, r).unwrap()
            }

            GpuOp::FSub(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["f32"];
                self.builder.f_sub(ty, None, l, r).unwrap()
            }

            GpuOp::FMul(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["f32"];
                self.builder.f_mul(ty, None, l, r).unwrap()
            }

            GpuOp::FDiv(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["f32"];
                self.builder.f_div(ty, None, l, r).unwrap()
            }

            GpuOp::FNeg(val) => {
                let v = self.values[val.0 as usize];
                let ty = self.types["f32"];
                self.builder.f_negate(ty, None, v).unwrap()
            }

            GpuOp::Load(ptr, _) => {
                let p = self.values[ptr.0 as usize];
                let ty = self.types["f32"];
                self.builder.load(ty, None, p, None, vec![]).unwrap()
            }

            GpuOp::Store(ptr, val, _) => {
                let p = self.values[ptr.0 as usize];
                let v = self.values[val.0 as usize];
                self.builder.store(p, v, None, vec![]).unwrap();
                0 // Void
            }

            GpuOp::ThreadIdX => {
                let var = self.variables["LocalInvocationId"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![0])
                    .unwrap()
            }

            GpuOp::ThreadIdY => {
                let var = self.variables["LocalInvocationId"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![1])
                    .unwrap()
            }

            GpuOp::ThreadIdZ => {
                let var = self.variables["LocalInvocationId"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![2])
                    .unwrap()
            }

            GpuOp::BlockIdX => {
                let var = self.variables["WorkgroupId"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![0])
                    .unwrap()
            }

            GpuOp::BlockIdY => {
                let var = self.variables["WorkgroupId"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![1])
                    .unwrap()
            }

            GpuOp::BlockIdZ => {
                let var = self.variables["WorkgroupId"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![2])
                    .unwrap()
            }

            GpuOp::BlockDimX => {
                let var = self.variables["WorkgroupSize"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![0])
                    .unwrap()
            }

            GpuOp::BlockDimY => {
                let var = self.variables["WorkgroupSize"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![1])
                    .unwrap()
            }

            GpuOp::BlockDimZ => {
                let var = self.variables["WorkgroupSize"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![2])
                    .unwrap()
            }

            GpuOp::GridDimX => {
                let var = self.variables["NumWorkgroups"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![0])
                    .unwrap()
            }

            GpuOp::GridDimY => {
                let var = self.variables["NumWorkgroups"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![1])
                    .unwrap()
            }

            GpuOp::GridDimZ => {
                let var = self.variables["NumWorkgroups"];
                let vec3_ty = self.types["vec3_u32"];
                let u32_ty = self.types["u32"];

                let vec = self.builder.load(vec3_ty, None, var, None, vec![]).unwrap();
                self.builder
                    .composite_extract(u32_ty, None, vec, vec![2])
                    .unwrap()
            }

            GpuOp::SyncThreads => {
                self.builder
                    .control_barrier(
                        spirv::Scope::Workgroup as u32,
                        spirv::Scope::Workgroup as u32,
                        (spirv::MemorySemantics::WORKGROUP_MEMORY
                            | spirv::MemorySemantics::ACQUIRE_RELEASE)
                            .bits(),
                    )
                    .unwrap();
                0 // Void
            }

            GpuOp::MemoryFence(_) => {
                self.builder
                    .memory_barrier(
                        spirv::Scope::Workgroup as u32,
                        (spirv::MemorySemantics::WORKGROUP_MEMORY
                            | spirv::MemorySemantics::ACQUIRE_RELEASE)
                            .bits(),
                    )
                    .unwrap();
                0 // Void
            }

            GpuOp::Lt(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.s_less_than(ty, None, l, r).unwrap()
            }

            GpuOp::Le(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.s_less_than_equal(ty, None, l, r).unwrap()
            }

            GpuOp::Gt(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.s_greater_than(ty, None, l, r).unwrap()
            }

            GpuOp::Ge(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.s_greater_than_equal(ty, None, l, r).unwrap()
            }

            GpuOp::Eq(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.i_equal(ty, None, l, r).unwrap()
            }

            GpuOp::Ne(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.i_not_equal(ty, None, l, r).unwrap()
            }

            GpuOp::FLt(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.f_ord_less_than(ty, None, l, r).unwrap()
            }

            GpuOp::FLe(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.f_ord_less_than_equal(ty, None, l, r).unwrap()
            }

            GpuOp::FGt(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.f_ord_greater_than(ty, None, l, r).unwrap()
            }

            GpuOp::FGe(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder
                    .f_ord_greater_than_equal(ty, None, l, r)
                    .unwrap()
            }

            GpuOp::FEq(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.f_ord_equal(ty, None, l, r).unwrap()
            }

            GpuOp::FNe(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.f_ord_not_equal(ty, None, l, r).unwrap()
            }

            GpuOp::And(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.logical_and(ty, None, l, r).unwrap()
            }

            GpuOp::Or(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["bool"];
                self.builder.logical_or(ty, None, l, r).unwrap()
            }

            GpuOp::Not(val) => {
                let v = self.values[val.0 as usize];
                let ty = self.types["bool"];
                self.builder.logical_not(ty, None, v).unwrap()
            }

            GpuOp::BitAnd(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.bitwise_and(ty, None, l, r).unwrap()
            }

            GpuOp::BitOr(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.bitwise_or(ty, None, l, r).unwrap()
            }

            GpuOp::BitXor(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.bitwise_xor(ty, None, l, r).unwrap()
            }

            GpuOp::BitNot(val) => {
                let v = self.values[val.0 as usize];
                let ty = self.types["i32"];
                self.builder.not(ty, None, v).unwrap()
            }

            GpuOp::Shl(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.shift_left_logical(ty, None, l, r).unwrap()
            }

            GpuOp::Shr(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["i32"];
                self.builder.shift_right_arithmetic(ty, None, l, r).unwrap()
            }

            GpuOp::LShr(lhs, rhs) => {
                let l = self.values[lhs.0 as usize];
                let r = self.values[rhs.0 as usize];
                let ty = self.types["u32"];
                self.builder.shift_right_logical(ty, None, l, r).unwrap()
            }

            GpuOp::Select(cond, t, f) => {
                let c = self.values[cond.0 as usize];
                let tv = self.values[t.0 as usize];
                let fv = self.values[f.0 as usize];
                let ty = self.types["i32"];
                self.builder.select(ty, None, c, tv, fv).unwrap()
            }

            _ => {
                // Placeholder for unimplemented ops
                let ty = self.types["i32"];
                self.builder.constant_bit32(ty, 0)
            }
        }
    }

    fn generate_terminator(&mut self, term: &GpuTerminator) {
        match term {
            GpuTerminator::Br(target) => {
                let block = self.blocks[target];
                self.builder.branch(block).unwrap();
            }

            GpuTerminator::CondBr(cond, then_block, else_block) => {
                let c = self.values[cond.0 as usize];
                let then_b = self.blocks[then_block];
                let else_b = self.blocks[else_block];
                self.builder
                    .branch_conditional(c, then_b, else_b, vec![])
                    .unwrap();
            }

            GpuTerminator::ReturnVoid => {
                self.builder.ret().unwrap();
            }

            GpuTerminator::Return(val) => {
                let v = self.values[val.0 as usize];
                self.builder.ret_value(v).unwrap();
            }

            GpuTerminator::Unreachable => {
                self.builder.unreachable().unwrap();
            }
        }
    }

    fn gpu_type_to_spirv(&self, ty: &GpuType) -> Word {
        match ty {
            GpuType::Void => self.types["void"],
            GpuType::Bool => self.types["bool"],
            GpuType::I8 => self.types["i8"],
            GpuType::U8 => self.types["u8"],
            GpuType::I16 => self.types["i16"],
            GpuType::U16 => self.types["u16"],
            GpuType::I32 => self.types["i32"],
            GpuType::U32 => self.types["u32"],
            GpuType::I64 => self.types["i64"],
            GpuType::U64 => self.types["u64"],
            GpuType::F32 => self.types["f32"],
            GpuType::F64 => self.types["f64"],
            _ => self.types["i32"], // Default
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

        let mut codegen = SpirvCodegen::new(spirv::ExecutionModel::GLCompute);
        let spirv_bytes = codegen.generate(&module);

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

        let mut codegen = SpirvCodegen::new(spirv::ExecutionModel::GLCompute);
        let spirv_bytes = codegen.generate(&module);

        // Should produce valid SPIR-V
        assert!(!spirv_bytes.is_empty());
    }
}
