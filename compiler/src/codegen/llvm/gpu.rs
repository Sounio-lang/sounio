//! GPU-specific LLVM Code Generation
//!
//! Translates GPU IR (GpuModule) to LLVM IR for NVPTX and AMDGPU targets.
//!
//! # Architecture
//!
//! ```text
//! GpuModule ──> LlvmGpuCodegen ──> LLVM IR ──> PTX/GCN Assembly
//!                    │
//!                    ├── NVPTX intrinsics (thread_id, sync, etc.)
//!                    ├── Address space mapping
//!                    └── Kernel metadata
//! ```
//!
//! # NVPTX Address Spaces
//!
//! | D Memory Space | LLVM Address Space |
//! |----------------|-------------------|
//! | Generic        | 0                 |
//! | Global         | 1                 |
//! | Shared         | 3                 |
//! | Constant       | 4                 |
//! | Local          | 5                 |
//!
//! # References
//!
//! - NVPTX Backend: <https://llvm.org/docs/NVPTXUsage.html>
//! - AMDGPU Backend: <https://llvm.org/docs/AMDGPUUsage.html>

use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FloatType, IntType};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue,
};
use inkwell::{FloatPredicate, IntPredicate};

use std::collections::HashMap;

use super::codegen::OptLevel;
use super::target::GpuTargetConfig;

use crate::codegen::gpu::ir::{
    BlockId, GpuBlock, GpuConstValue, GpuKernel, GpuModule, GpuOp, GpuParam, GpuTarget,
    GpuTerminator, GpuType, MemorySpace, SharedMemDecl, ValueId, WarpReduceOp, WarpVoteOp,
};

/// LLVM-based GPU code generator
pub struct LlvmGpuCodegen<'ctx> {
    /// LLVM context
    context: &'ctx Context,

    /// LLVM module being built
    module: Module<'ctx>,

    /// IR builder
    builder: Builder<'ctx>,

    /// GPU target
    target: GpuTarget,

    /// Value map: GPU ValueId → LLVM Value
    values: HashMap<ValueId, BasicValueEnum<'ctx>>,

    /// Block map: GPU BlockId → LLVM BasicBlock
    blocks: HashMap<BlockId, BasicBlock<'ctx>>,

    /// Shared memory declarations
    shared_memory: HashMap<String, PointerValue<'ctx>>,

    /// Current function being compiled
    current_function: Option<FunctionValue<'ctx>>,

    /// Optimization level
    opt_level: OptLevel,
}

impl<'ctx> LlvmGpuCodegen<'ctx> {
    /// Helper to create array type from BasicTypeEnum (LLVM 15+ compatibility)
    fn make_array_type(elem_ty: BasicTypeEnum<'ctx>, size: u32) -> inkwell::types::ArrayType<'ctx> {
        match elem_ty {
            BasicTypeEnum::IntType(t) => t.array_type(size),
            BasicTypeEnum::FloatType(t) => t.array_type(size),
            BasicTypeEnum::PointerType(t) => t.array_type(size),
            BasicTypeEnum::ArrayType(t) => t.array_type(size),
            BasicTypeEnum::StructType(t) => t.array_type(size),
            BasicTypeEnum::VectorType(t) => t.array_type(size),
        }
    }

    /// Create a new GPU code generator
    pub fn new(context: &'ctx Context, module_name: &str, target: GpuTarget) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        // Set target triple based on GPU target
        let triple = match target {
            GpuTarget::Cuda { .. } => "nvptx64-nvidia-cuda",
            GpuTarget::Rocm => "amdgcn-amd-amdhsa",
            _ => "nvptx64-nvidia-cuda", // Default to CUDA
        };
        module.set_triple(&inkwell::targets::TargetTriple::create(triple));

        // Set data layout for NVPTX
        if matches!(target, GpuTarget::Cuda { .. }) {
            module.set_data_layout(&inkwell::targets::TargetData::create(
                "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64",
            ).get_data_layout());
        }

        Self {
            context,
            module,
            builder,
            target,
            values: HashMap::new(),
            blocks: HashMap::new(),
            shared_memory: HashMap::new(),
            current_function: None,
            opt_level: OptLevel::O2,
        }
    }

    /// Set optimization level
    pub fn with_opt_level(mut self, level: OptLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Compile a GPU module to LLVM IR
    pub fn compile(&mut self, gpu_module: &GpuModule) -> &Module<'ctx> {
        // Add global constants
        for constant in &gpu_module.constants {
            self.add_constant(constant);
        }

        // Compile kernels
        for (_, kernel) in &gpu_module.kernels {
            self.compile_kernel(kernel);
        }

        // Compile device functions
        for (_, func) in &gpu_module.device_functions {
            self.compile_device_function(func);
        }

        &self.module
    }

    /// Add a global constant
    fn add_constant(&mut self, constant: &crate::codegen::gpu::ir::GpuConstant) {
        let llvm_type = self.convert_type(&constant.ty);
        let init_value = self.compile_const_value(&constant.value, &constant.ty);

        if let Some(init) = init_value {
            let global = self.module.add_global(
                llvm_type,
                Some(AddressSpace::from(4u16)), // Constant memory
                &constant.name,
            );
            global.set_initializer(&init);
            global.set_constant(true);
        }
    }

    /// Compile a kernel function
    fn compile_kernel(&mut self, kernel: &GpuKernel) {
        // Clear per-function state
        self.values.clear();
        self.blocks.clear();
        self.shared_memory.clear();

        // Build function type
        let param_types: Vec<BasicMetadataTypeEnum> = kernel
            .params
            .iter()
            .map(|p| self.convert_type(&p.ty).into())
            .collect();

        let fn_type = self.context.void_type().fn_type(&param_types, false);
        let fn_val = self.module.add_function(&kernel.name, fn_type, None);

        // Mark as kernel
        self.add_kernel_metadata(&fn_val);

        // Set parameter names
        for (i, param) in kernel.params.iter().enumerate() {
            if let Some(param_val) = fn_val.get_nth_param(i as u32) {
                param_val.set_name(&param.name);
            }
        }

        self.current_function = Some(fn_val);

        // Create basic blocks
        for block in &kernel.blocks {
            let bb = self.context.append_basic_block(fn_val, &block.label);
            self.blocks.insert(block.id, bb);
        }

        // Map parameters to values using index as ValueId
        for (i, _param) in kernel.params.iter().enumerate() {
            if let Some(param_val) = fn_val.get_nth_param(i as u32) {
                // Use parameter index as ValueId
                self.values.insert(ValueId(i as u32), param_val);
            }
        }

        // Allocate shared memory
        for shared in &kernel.shared_memory {
            self.allocate_shared_memory(shared, fn_val);
        }

        // Compile blocks
        for block in &kernel.blocks {
            self.compile_block(block);
        }

        self.current_function = None;
    }

    /// Compile a device function (callable from kernels)
    fn compile_device_function(&mut self, func: &crate::codegen::gpu::ir::GpuFunction) {
        // Clear per-function state
        self.values.clear();
        self.blocks.clear();

        // Build function type
        let param_types: Vec<BasicMetadataTypeEnum> = func
            .params
            .iter()
            .map(|p| self.convert_type(&p.ty).into())
            .collect();

        let ret_type = self.convert_type(&func.return_type);
        let fn_type = ret_type.fn_type(&param_types, false);
        let fn_val = self
            .module
            .add_function(&func.name, fn_type, Some(Linkage::Internal));

        // Mark as device function
        fn_val.add_attribute(
            inkwell::attributes::AttributeLoc::Function,
            self.context.create_string_attribute("noinline", ""),
        );

        self.current_function = Some(fn_val);

        // Create basic blocks
        for block in &func.blocks {
            let bb = self.context.append_basic_block(fn_val, &block.label);
            self.blocks.insert(block.id, bb);
        }

        // Map parameters to values using index as ValueId
        for (i, _param) in func.params.iter().enumerate() {
            if let Some(param_val) = fn_val.get_nth_param(i as u32) {
                // Use parameter index as ValueId
                self.values.insert(ValueId(i as u32), param_val);
            }
        }

        // Compile blocks
        for block in &func.blocks {
            self.compile_block(block);
        }

        self.current_function = None;
    }

    /// Add kernel metadata (NVPTX specific)
    fn add_kernel_metadata(&self, fn_val: &FunctionValue<'ctx>) {
        // Add nvvm.annotations for kernel
        let kernel_md = self.context.metadata_node(&[
            fn_val.as_global_value().as_pointer_value().into(),
            self.context.metadata_string("kernel").into(),
            self.context.i32_type().const_int(1, false).into(),
        ]);

        // In inkwell 0.5+, use add_global_metadata with key and metadata value
        let existing = self.module.get_global_metadata("nvvm.annotations");
        if existing.is_empty() {
            let _ = self
                .module
                .add_global_metadata("nvvm.annotations", &kernel_md);
        }
    }

    /// Allocate shared memory
    fn allocate_shared_memory(&mut self, shared: &SharedMemDecl, _fn_val: FunctionValue<'ctx>) {
        let elem_type = self.convert_type(&shared.elem_type);
        let array_type = Self::make_array_type(elem_type, shared.size);

        // Shared memory address space (3 for NVPTX)
        let global =
            self.module
                .add_global(array_type, Some(AddressSpace::from(3u16)), &shared.name);

        global.set_linkage(Linkage::Internal);
        global.set_alignment(shared.align);

        let ptr = global.as_pointer_value();
        self.shared_memory.insert(shared.name.clone(), ptr);
    }

    /// Compile a basic block
    fn compile_block(&mut self, block: &GpuBlock) {
        let bb = match self.blocks.get(&block.id) {
            Some(b) => *b,
            None => return,
        };

        self.builder.position_at_end(bb);

        // Compile instructions
        for (value_id, op) in &block.instructions {
            if let Some(val) = self.compile_op(op) {
                self.values.insert(*value_id, val);
            }
        }

        // Compile terminator
        self.compile_terminator(&block.terminator);
    }

    /// Compile a GPU operation to LLVM IR
    fn compile_op(&mut self, op: &GpuOp) -> Option<BasicValueEnum<'ctx>> {
        match op {
            // === Constants ===
            GpuOp::ConstInt(val, ty) => {
                let int_ty = self.convert_int_type(ty);
                Some(int_ty.const_int(*val as u64, ty.is_signed()).into())
            }
            GpuOp::ConstFloat(val, ty) => {
                let float_ty = self.convert_float_type(ty);
                Some(float_ty.const_float(*val).into())
            }
            GpuOp::ConstBool(val) => Some(
                self.context
                    .bool_type()
                    .const_int(*val as u64, false)
                    .into(),
            ),

            // === Arithmetic ===
            GpuOp::Add(a, b) => {
                let lhs = self.get_value(*a)?.into_int_value();
                let rhs = self.get_value(*b)?.into_int_value();
                self.builder
                    .build_int_add(lhs, rhs, "add")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::Sub(a, b) => {
                let lhs = self.get_value(*a)?.into_int_value();
                let rhs = self.get_value(*b)?.into_int_value();
                self.builder
                    .build_int_sub(lhs, rhs, "sub")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::Mul(a, b) => {
                let lhs = self.get_value(*a)?.into_int_value();
                let rhs = self.get_value(*b)?.into_int_value();
                self.builder
                    .build_int_mul(lhs, rhs, "mul")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::Div(a, b) => {
                let lhs = self.get_value(*a)?.into_int_value();
                let rhs = self.get_value(*b)?.into_int_value();
                self.builder
                    .build_int_signed_div(lhs, rhs, "div")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::Neg(a) => {
                let val = self.get_value(*a)?.into_int_value();
                self.builder
                    .build_int_neg(val, "neg")
                    .ok()
                    .map(|v| v.into())
            }

            // === Floating-point ===
            GpuOp::FAdd(a, b) => {
                let lhs = self.get_value(*a)?.into_float_value();
                let rhs = self.get_value(*b)?.into_float_value();
                self.builder
                    .build_float_add(lhs, rhs, "fadd")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::FSub(a, b) => {
                let lhs = self.get_value(*a)?.into_float_value();
                let rhs = self.get_value(*b)?.into_float_value();
                self.builder
                    .build_float_sub(lhs, rhs, "fsub")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::FMul(a, b) => {
                let lhs = self.get_value(*a)?.into_float_value();
                let rhs = self.get_value(*b)?.into_float_value();
                self.builder
                    .build_float_mul(lhs, rhs, "fmul")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::FDiv(a, b) => {
                let lhs = self.get_value(*a)?.into_float_value();
                let rhs = self.get_value(*b)?.into_float_value();
                self.builder
                    .build_float_div(lhs, rhs, "fdiv")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::FNeg(a) => {
                let val = self.get_value(*a)?.into_float_value();
                self.builder
                    .build_float_neg(val, "fneg")
                    .ok()
                    .map(|v| v.into())
            }

            // === GPU Intrinsics ===
            GpuOp::ThreadIdX => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.tid.x"),
            GpuOp::ThreadIdY => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.tid.y"),
            GpuOp::ThreadIdZ => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.tid.z"),
            GpuOp::BlockIdX => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.ctaid.x"),
            GpuOp::BlockIdY => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.ctaid.y"),
            GpuOp::BlockIdZ => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.ctaid.z"),
            GpuOp::BlockDimX => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.ntid.x"),
            GpuOp::BlockDimY => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.ntid.y"),
            GpuOp::BlockDimZ => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.ntid.z"),
            GpuOp::GridDimX => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.nctaid.x"),
            GpuOp::GridDimY => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.nctaid.y"),
            GpuOp::GridDimZ => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.nctaid.z"),
            GpuOp::WarpId => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.warpid"),
            GpuOp::LaneId => self.call_nvptx_intrinsic("llvm.nvvm.read.ptx.sreg.laneid"),
            GpuOp::WarpSize => {
                // Warp size is always 32 on NVIDIA GPUs
                Some(self.context.i32_type().const_int(32, false).into())
            }

            // === Synchronization ===
            GpuOp::SyncThreads => {
                self.call_nvptx_barrier0();
                None
            }
            GpuOp::SyncWarp(mask) => {
                self.call_nvptx_syncwarp(*mask);
                None
            }
            GpuOp::MemoryFence(space) => {
                self.call_nvptx_membar(space);
                None
            }

            // === Memory ===
            GpuOp::Load(ptr, space) => {
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                let pointee_type = self.context.f32_type(); // TODO: Infer from context
                self.builder.build_load(pointee_type, ptr_val, "load").ok()
            }
            GpuOp::Store(ptr, val, space) => {
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                let value = self.get_value(*val)?;
                self.builder.build_store(ptr_val, value).ok()?;
                None
            }

            // === Atomic operations ===
            GpuOp::AtomicAdd(ptr, val) => {
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                let value = self.get_value(*val)?;
                self.builder
                    .build_atomicrmw(
                        inkwell::AtomicRMWBinOp::Add,
                        ptr_val,
                        value.into_int_value(),
                        inkwell::AtomicOrdering::SequentiallyConsistent,
                    )
                    .ok()
                    .map(|v| v.into())
            }

            // === Comparisons ===
            GpuOp::Eq(a, b) => {
                let lhs = self.get_value(*a)?.into_int_value();
                let rhs = self.get_value(*b)?.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::EQ, lhs, rhs, "eq")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::Lt(a, b) => {
                let lhs = self.get_value(*a)?.into_int_value();
                let rhs = self.get_value(*b)?.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::SLT, lhs, rhs, "lt")
                    .ok()
                    .map(|v| v.into())
            }
            GpuOp::FLt(a, b) => {
                let lhs = self.get_value(*a)?.into_float_value();
                let rhs = self.get_value(*b)?.into_float_value();
                self.builder
                    .build_float_compare(FloatPredicate::OLT, lhs, rhs, "flt")
                    .ok()
                    .map(|v| v.into())
            }

            // === Warp operations ===
            GpuOp::WarpShuffle(val, lane) => {
                let value = self.get_value(*val)?;
                let lane_id = self.get_value(*lane)?.into_int_value();
                self.call_nvptx_shfl_sync(value, lane_id)
            }

            // Default: not yet implemented
            _ => {
                // TODO: Implement remaining operations
                None
            }
        }
    }

    /// Compile terminator
    fn compile_terminator(&mut self, term: &GpuTerminator) {
        match term {
            GpuTerminator::ReturnVoid => {
                let _ = self.builder.build_return(None);
            }
            GpuTerminator::Return(val_id) => {
                if let Some(ret_val) = self.get_value(*val_id) {
                    let _ = self.builder.build_return(Some(&ret_val));
                } else {
                    let _ = self.builder.build_return(None);
                }
            }
            GpuTerminator::Br(target) => {
                if let Some(bb) = self.blocks.get(target) {
                    let _ = self.builder.build_unconditional_branch(*bb);
                }
            }
            GpuTerminator::CondBr(condition, then_block, else_block) => {
                if let (Some(cond), Some(then_bb), Some(else_bb)) = (
                    self.get_value(*condition),
                    self.blocks.get(then_block),
                    self.blocks.get(else_block),
                ) {
                    let cond_int = cond.into_int_value();
                    let _ = self
                        .builder
                        .build_conditional_branch(cond_int, *then_bb, *else_bb);
                }
            }
            GpuTerminator::Unreachable => {
                let _ = self.builder.build_unreachable();
            }
        }
    }

    // === Helper methods ===

    /// Get a value from the map
    fn get_value(&self, id: ValueId) -> Option<BasicValueEnum<'ctx>> {
        self.values.get(&id).copied()
    }

    /// Convert GPU type to LLVM type
    fn convert_type(&self, ty: &GpuType) -> BasicTypeEnum<'ctx> {
        match ty {
            GpuType::Void => self.context.struct_type(&[], false).into(),
            GpuType::Bool => self.context.bool_type().into(),
            GpuType::I8 => self.context.i8_type().into(),
            GpuType::I16 => self.context.i16_type().into(),
            GpuType::I32 => self.context.i32_type().into(),
            GpuType::I64 => self.context.i64_type().into(),
            GpuType::U8 => self.context.i8_type().into(),
            GpuType::U16 => self.context.i16_type().into(),
            GpuType::U32 => self.context.i32_type().into(),
            GpuType::U64 => self.context.i64_type().into(),
            GpuType::F16 => self.context.f16_type().into(),
            GpuType::BF16 => self.context.i16_type().into(), // BF16 stored as i16
            GpuType::F8E4M3 => self.context.i8_type().into(), // FP8 stored as i8
            GpuType::F8E5M2 => self.context.i8_type().into(), // FP8 stored as i8
            GpuType::F4 => self.context.i8_type().into(),     // FP4 stored as i8 (packed)
            GpuType::F32 => self.context.f32_type().into(),
            GpuType::F64 => self.context.f64_type().into(),
            GpuType::Vec2(elem) => {
                let elem_ty = self.convert_type(elem);
                Self::make_array_type(elem_ty, 2).into()
            }
            GpuType::Vec3(elem) => {
                let elem_ty = self.convert_type(elem);
                Self::make_array_type(elem_ty, 3).into()
            }
            GpuType::Vec4(elem) => {
                let elem_ty = self.convert_type(elem);
                Self::make_array_type(elem_ty, 4).into()
            }
            GpuType::Ptr(inner, space) => {
                let addr_space = self.convert_memory_space(space);
                // LLVM 15+ uses opaque pointers with address spaces
                let _inner_ty = self.convert_type(inner);
                self.context.ptr_type(addr_space).into()
            }
            GpuType::Array(elem, size) => {
                let elem_ty = self.convert_type(elem);
                Self::make_array_type(elem_ty, *size).into()
            }
            GpuType::Struct(_, fields) => {
                // fields is Vec<(String, GpuType)> - extract just the types
                let field_types: Vec<_> =
                    fields.iter().map(|(_, ty)| self.convert_type(ty)).collect();
                self.context.struct_type(&field_types, false).into()
            }
        }
    }

    /// Convert GPU int type to LLVM int type
    fn convert_int_type(&self, ty: &GpuType) -> IntType<'ctx> {
        match ty {
            GpuType::Bool => self.context.bool_type(),
            GpuType::I8 | GpuType::U8 => self.context.i8_type(),
            GpuType::I16 | GpuType::U16 => self.context.i16_type(),
            GpuType::I32 | GpuType::U32 => self.context.i32_type(),
            GpuType::I64 | GpuType::U64 => self.context.i64_type(),
            _ => self.context.i32_type(), // Default
        }
    }

    /// Convert GPU float type to LLVM float type
    fn convert_float_type(&self, ty: &GpuType) -> FloatType<'ctx> {
        match ty {
            GpuType::F16 => self.context.f16_type(),
            GpuType::F32 => self.context.f32_type(),
            GpuType::F64 => self.context.f64_type(),
            _ => self.context.f32_type(), // Default
        }
    }

    /// Convert memory space to LLVM address space (NVPTX)
    fn convert_memory_space(&self, space: &MemorySpace) -> AddressSpace {
        match space {
            MemorySpace::Generic => AddressSpace::from(0u16),
            MemorySpace::Global => AddressSpace::from(1u16),
            MemorySpace::Shared => AddressSpace::from(3u16),
            MemorySpace::Constant => AddressSpace::from(4u16),
            MemorySpace::Local => AddressSpace::from(5u16),
            MemorySpace::Texture => AddressSpace::from(1u16), // Same as global
        }
    }

    /// Compile a constant value
    fn compile_const_value(
        &self,
        value: &GpuConstValue,
        ty: &GpuType,
    ) -> Option<BasicValueEnum<'ctx>> {
        match value {
            GpuConstValue::Int(n) => {
                let int_ty = self.convert_int_type(ty);
                Some(int_ty.const_int(*n as u64, ty.is_signed()).into())
            }
            GpuConstValue::Float(f) => {
                let float_ty = self.convert_float_type(ty);
                Some(float_ty.const_float(*f).into())
            }
            GpuConstValue::Bool(b) => {
                Some(self.context.bool_type().const_int(*b as u64, false).into())
            }
            _ => None, // TODO: Arrays and structs
        }
    }

    // === NVPTX Intrinsics ===

    /// Call an NVPTX intrinsic that returns i32
    fn call_nvptx_intrinsic(&self, name: &str) -> Option<BasicValueEnum<'ctx>> {
        let i32_ty = self.context.i32_type();
        let fn_type = i32_ty.fn_type(&[], false);

        let fn_val = self
            .module
            .add_function(name, fn_type, Some(Linkage::External));

        self.builder
            .build_call(fn_val, &[], "intrinsic")
            .ok()?
            .try_as_basic_value()
            .left()
    }

    /// Call barrier0 (block sync)
    fn call_nvptx_barrier0(&self) {
        let void_ty = self.context.void_type();
        let fn_type = void_ty.fn_type(&[], false);

        let fn_val =
            self.module
                .add_function("llvm.nvvm.barrier0", fn_type, Some(Linkage::External));

        let _ = self.builder.build_call(fn_val, &[], "");
    }

    /// Call syncwarp
    fn call_nvptx_syncwarp(&self, mask: u32) {
        let void_ty = self.context.void_type();
        let i32_ty = self.context.i32_type();
        let fn_type = void_ty.fn_type(&[i32_ty.into()], false);

        let fn_val =
            self.module
                .add_function("llvm.nvvm.bar.warp.sync", fn_type, Some(Linkage::External));

        let mask_val = i32_ty.const_int(mask as u64, false);
        let _ = self.builder.build_call(fn_val, &[mask_val.into()], "");
    }

    /// Call memory barrier
    fn call_nvptx_membar(&self, space: &MemorySpace) {
        let void_ty = self.context.void_type();
        let fn_type = void_ty.fn_type(&[], false);

        let name = match space {
            MemorySpace::Shared => "llvm.nvvm.membar.cta",
            MemorySpace::Global => "llvm.nvvm.membar.gl",
            _ => "llvm.nvvm.membar.sys",
        };

        let fn_val = self
            .module
            .add_function(name, fn_type, Some(Linkage::External));
        let _ = self.builder.build_call(fn_val, &[], "");
    }

    /// Call warp shuffle
    fn call_nvptx_shfl_sync(
        &self,
        value: BasicValueEnum<'ctx>,
        lane: IntValue<'ctx>,
    ) -> Option<BasicValueEnum<'ctx>> {
        let i32_ty = self.context.i32_type();
        let fn_type = i32_ty.fn_type(
            &[i32_ty.into(), i32_ty.into(), i32_ty.into(), i32_ty.into()],
            false,
        );

        let fn_val = self.module.add_function(
            "llvm.nvvm.shfl.sync.idx.i32",
            fn_type,
            Some(Linkage::External),
        );

        let mask = i32_ty.const_int(0xFFFFFFFF, false); // Full warp mask
        let width = i32_ty.const_int(31, false); // Lane mask

        self.builder
            .build_call(
                fn_val,
                &[
                    mask.into(),
                    value.into_int_value().into(),
                    lane.into(),
                    width.into(),
                ],
                "shfl",
            )
            .ok()?
            .try_as_basic_value()
            .left()
    }

    /// Get the compiled module
    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Print LLVM IR to string
    pub fn print_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }

    /// Verify the module
    pub fn verify(&self) -> Result<(), String> {
        self.module.verify().map_err(|e| e.to_string())
    }

    /// Compile to PTX assembly
    pub fn compile_to_ptx(&self, config: &GpuTargetConfig) -> Result<String, String> {
        use inkwell::targets::FileType;

        let target_machine = config.create_target_machine(self.opt_level)?;

        let buffer = target_machine
            .write_to_memory_buffer(&self.module, FileType::Assembly)
            .map_err(|e| format!("Failed to compile to PTX: {}", e))?;

        String::from_utf8(buffer.as_slice().to_vec())
            .map_err(|e| format!("Invalid UTF-8 in PTX: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_gpu_codegen() {
        let context = Context::create();
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let codegen = LlvmGpuCodegen::new(&context, "test_kernel", target);
        let ir = codegen.print_ir();
        assert!(ir.contains("test_kernel"));
    }

    #[test]
    fn test_memory_space_conversion() {
        let context = Context::create();
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let codegen = LlvmGpuCodegen::new(&context, "test", target);

        assert_eq!(
            codegen.convert_memory_space(&MemorySpace::Global),
            AddressSpace::from(1u16)
        );
        assert_eq!(
            codegen.convert_memory_space(&MemorySpace::Shared),
            AddressSpace::from(3u16)
        );
        assert_eq!(
            codegen.convert_memory_space(&MemorySpace::Constant),
            AddressSpace::from(4u16)
        );
    }

    #[test]
    fn test_type_conversion() {
        let context = Context::create();
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let codegen = LlvmGpuCodegen::new(&context, "test", target);

        let f32_type = codegen.convert_type(&GpuType::F32);
        assert!(f32_type.is_float_type());

        let i32_type = codegen.convert_type(&GpuType::I32);
        assert!(i32_type.is_int_type());

        let ptr_type =
            codegen.convert_type(&GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global));
        assert!(ptr_type.is_pointer_type());
    }
}
