//! Main LLVM code generator
//!
//! This module provides the core code generation logic that translates
//! HLIR to LLVM IR.

use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue,
};
use inkwell::{FloatPredicate, IntPredicate};

use std::collections::HashMap;

use super::types::TypeConverter;
use crate::ast::Abi;
use crate::hlir::{
    BinaryOp, BlockId, HlirBlock, HlirConstant, HlirFunction, HlirInstr, HlirModule,
    HlirTerminator, HlirType, Op, UnaryOp, ValueId,
};

/// Optimization level for LLVM compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// No optimization
    O0,
    /// Basic optimizations
    O1,
    /// Standard optimizations (default)
    #[default]
    O2,
    /// Aggressive optimizations
    O3,
    /// Size optimization
    Os,
    /// Aggressive size optimization
    Oz,
}

impl OptLevel {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "0" | "O0" => Some(OptLevel::O0),
            "1" | "O1" => Some(OptLevel::O1),
            "2" | "O2" => Some(OptLevel::O2),
            "3" | "O3" => Some(OptLevel::O3),
            "s" | "Os" => Some(OptLevel::Os),
            "z" | "Oz" => Some(OptLevel::Oz),
            _ => None,
        }
    }

    /// Convert to inkwell optimization level
    pub fn to_inkwell(&self) -> inkwell::OptimizationLevel {
        match self {
            OptLevel::O0 => inkwell::OptimizationLevel::None,
            OptLevel::O1 => inkwell::OptimizationLevel::Less,
            OptLevel::O2 | OptLevel::Os => inkwell::OptimizationLevel::Default,
            OptLevel::O3 | OptLevel::Oz => inkwell::OptimizationLevel::Aggressive,
        }
    }
}

/// LLVM Code Generator
pub struct LLVMCodegen<'ctx> {
    /// LLVM context
    context: &'ctx Context,

    /// LLVM module being built
    module: Module<'ctx>,

    /// IR builder
    builder: Builder<'ctx>,

    /// Type converter
    types: TypeConverter<'ctx>,

    /// Current function being compiled
    current_function: Option<FunctionValue<'ctx>>,

    /// Value map: HLIR ValueId → LLVM Value
    values: HashMap<ValueId, BasicValueEnum<'ctx>>,

    /// Type map: HLIR ValueId → HlirType (for GEP element type calculation)
    value_types: HashMap<ValueId, HlirType>,

    /// Block map: HLIR BlockId → LLVM BasicBlock
    blocks: HashMap<BlockId, BasicBlock<'ctx>>,

    /// Function map: name → LLVM Function
    functions: HashMap<String, FunctionValue<'ctx>>,

    /// Global strings
    strings: HashMap<String, PointerValue<'ctx>>,

    /// Optimization level
    opt_level: OptLevel,

    /// Generate debug info
    debug: bool,
}

impl<'ctx> LLVMCodegen<'ctx> {
    /// Create a new code generator
    pub fn new(
        context: &'ctx Context,
        module_name: &str,
        opt_level: OptLevel,
        debug: bool,
    ) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        let types = TypeConverter::new(context);

        Self {
            context,
            module,
            builder,
            types,
            current_function: None,
            values: HashMap::new(),
            value_types: HashMap::new(),
            blocks: HashMap::new(),
            functions: HashMap::new(),
            strings: HashMap::new(),
            opt_level,
            debug,
        }
    }

    /// Compile an HLIR module to LLVM IR
    pub fn compile(&mut self, hlir: &HlirModule) -> &Module<'ctx> {
        // Declare all functions first (for forward references)
        for func in &hlir.functions {
            self.declare_function(func);
        }

        // Compile function bodies
        for func in &hlir.functions {
            self.compile_function(func);
        }

        // Create main wrapper if there's a main function
        if self.functions.contains_key("main") {
            self.create_main_wrapper();
        }

        &self.module
    }

    /// Declare a function (without body)
    fn declare_function(&mut self, func: &HlirFunction) {
        let param_types: Vec<HlirType> = func.params.iter().map(|p| p.ty.clone()).collect();

        let fn_type =
            self.types
                .function_type_variadic(&param_types, &func.return_type, func.is_variadic);

        // Determine linkage based on ABI and export status
        // extern "C" functions or exported functions get External linkage
        let linkage = match (&func.abi, func.is_exported) {
            (Abi::C | Abi::CUnwind | Abi::System | Abi::SystemUnwind, _) => Some(Linkage::External),
            (_, true) => Some(Linkage::External),
            _ => None, // Default (internal) linkage
        };

        let symbol_name = func.link_name.as_deref().unwrap_or(&func.name);
        let fn_val = self.module.add_function(symbol_name, fn_type, linkage);

        // Set calling convention for C ABI
        match &func.abi {
            Abi::C | Abi::CUnwind => {
                fn_val.set_call_conventions(0); // C calling convention
            }
            _ => {}
        }

        // Set parameter names
        for (i, param) in func.params.iter().enumerate() {
            if let Some(param_val) = fn_val.get_nth_param(i as u32) {
                param_val.set_name(&param.name);
            }
        }

        self.functions.insert(func.name.clone(), fn_val);
    }

    /// Compile a function body
    fn compile_function(&mut self, func: &HlirFunction) {
        let fn_val = match self.functions.get(&func.name) {
            Some(f) => *f,
            None => return,
        };

        self.current_function = Some(fn_val);

        // Clear per-function state
        self.values.clear();
        self.value_types.clear();
        self.blocks.clear();

        // Create basic blocks
        for block in &func.blocks {
            let bb = self.context.append_basic_block(fn_val, &block.label);
            self.blocks.insert(block.id, bb);
        }

        // Map parameters to values and types
        for (i, param) in func.params.iter().enumerate() {
            if let Some(param_val) = fn_val.get_nth_param(i as u32) {
                self.values.insert(param.value, param_val);
                self.value_types.insert(param.value, param.ty.clone());
            }
        }

        // Compile each block
        for block in &func.blocks {
            self.compile_block(block);
        }

        self.current_function = None;
    }

    /// Compile a basic block
    fn compile_block(&mut self, block: &HlirBlock) {
        let bb = match self.blocks.get(&block.id) {
            Some(b) => *b,
            None => return,
        };

        self.builder.position_at_end(bb);

        // Compile instructions
        for instr in &block.instructions {
            if let Some(val) = self.compile_instruction(instr) {
                if let Some(result_id) = instr.result {
                    self.values.insert(result_id, val);
                    self.value_types.insert(result_id, instr.ty.clone());
                }
            }
        }

        // Compile terminator
        self.compile_terminator(&block.terminator);
    }

    /// Compile an instruction
    fn compile_instruction(&mut self, instr: &HlirInstr) -> Option<BasicValueEnum<'ctx>> {
        match &instr.op {
            Op::Const(constant) => self.compile_constant(constant),

            Op::Copy(val_id) => self.get_value(*val_id),

            Op::Binary { op, left, right } => {
                let lhs = self.get_value(*left)?;
                let rhs = self.get_value(*right)?;
                self.compile_binary_op(*op, lhs, rhs, &instr.ty)
            }

            Op::Unary { op, operand } => {
                let val = self.get_value(*operand)?;
                self.compile_unary_op(*op, val, &instr.ty)
            }

            Op::Call { func, args } => {
                let fn_ptr = self.get_value(*func)?;
                let arg_vals: Vec<BasicMetadataValueEnum> = args
                    .iter()
                    .filter_map(|a| self.get_value(*a))
                    .map(|v| v.into())
                    .collect();

                // For indirect calls via function pointer
                if let BasicValueEnum::PointerValue(_ptr) = fn_ptr {
                    // Would need CallableValue, skipping for now
                    None
                } else {
                    None
                }
            }

            Op::CallDirect { name, args } => {
                let fn_val = self.functions.get(name)?;
                let arg_vals: Vec<_> = args
                    .iter()
                    .filter_map(|a| self.get_value(*a))
                    .map(|v| v.into())
                    .collect();

                let call = self.builder.build_call(*fn_val, &arg_vals, "call").ok()?;

                call.try_as_basic_value().left()
            }

            Op::Load { ptr } => {
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                let load_ty = self.types.convert(&instr.ty);
                self.builder.build_load(load_ty, ptr_val, "load").ok()
            }

            Op::Store { ptr, value } => {
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                let val = self.get_value(*value)?;
                self.builder.build_store(ptr_val, val).ok()?;
                // Return unit value
                Some(self.context.struct_type(&[], false).const_zero().into())
            }

            Op::GetFieldPtr { base, field } => {
                let ptr_val = self.get_value(*base)?.into_pointer_value();
                // Need struct type for GEP
                let zero = self.context.i32_type().const_zero();
                let field_idx = self.context.i32_type().const_int(*field as u64, false);

                // Fix: Get the struct type from the base pointer's type (Ptr(Struct) -> Struct)
                let struct_ty = match self.value_types.get(base) {
                    Some(HlirType::Ptr(inner)) => self.types.convert(inner),
                    _ => {
                        // Fallback: try to infer from instr.ty (less reliable)
                        self.types.convert(&instr.ty)
                    }
                };
                unsafe {
                    self.builder
                        .build_gep(struct_ty, ptr_val, &[zero, field_idx], "field_ptr")
                        .ok()
                        .map(|v| v.into())
                }
            }

            Op::GetElementPtr { base, index } => {
                let ptr_val = self.get_value(*base)?.into_pointer_value();
                let idx = self.get_value(*index)?.into_int_value();

                // Fix: instr.ty is Ptr(element_type), we need element_type for GEP
                let elem_ty = match &instr.ty {
                    HlirType::Ptr(inner) => self.types.convert(inner),
                    _ => self.types.convert(&instr.ty), // Fallback (shouldn't happen)
                };
                unsafe {
                    self.builder
                        .build_gep(elem_ty, ptr_val, &[idx], "elem_ptr")
                        .ok()
                        .map(|v| v.into())
                }
            }

            Op::Alloca { ty } => {
                let alloc_ty = self.types.convert(ty);
                self.builder
                    .build_alloca(alloc_ty, "alloca")
                    .ok()
                    .map(|v| v.into())
            }

            Op::Cast {
                value,
                source,
                target,
            } => {
                let val = self.get_value(*value)?;
                self.compile_cast(val, source, target)
            }

            Op::Phi { incoming } => {
                let phi_ty = self.types.convert(&instr.ty);
                let phi = self.builder.build_phi(phi_ty, "phi").ok()?;

                for (block_id, val_id) in incoming {
                    if let (Some(bb), Some(val)) =
                        (self.blocks.get(block_id), self.get_value(*val_id))
                    {
                        phi.add_incoming(&[(&val, *bb)]);
                    }
                }

                Some(phi.as_basic_value())
            }

            Op::ExtractValue { base, index } => {
                let agg = self.get_value(*base)?;
                match agg {
                    BasicValueEnum::StructValue(sv) => self
                        .builder
                        .build_extract_value(sv, *index as u32, "extract")
                        .ok(),
                    BasicValueEnum::ArrayValue(av) => self
                        .builder
                        .build_extract_value(av, *index as u32, "extract")
                        .ok(),
                    _ => None,
                }
            }

            Op::InsertValue { base, value, index } => {
                let agg = self.get_value(*base)?;
                let val = self.get_value(*value)?;

                match agg {
                    BasicValueEnum::StructValue(sv) => self
                        .builder
                        .build_insert_value(sv, val, *index as u32, "insert")
                        .ok()
                        .map(|v| match v {
                            inkwell::values::AggregateValueEnum::StructValue(s) => s.into(),
                            inkwell::values::AggregateValueEnum::ArrayValue(a) => a.into(),
                        }),
                    BasicValueEnum::ArrayValue(av) => self
                        .builder
                        .build_insert_value(av, val, *index as u32, "insert")
                        .ok()
                        .map(|v| match v {
                            inkwell::values::AggregateValueEnum::StructValue(s) => s.into(),
                            inkwell::values::AggregateValueEnum::ArrayValue(a) => a.into(),
                        }),
                    _ => None,
                }
            }

            Op::Tuple(values) => {
                let vals: Vec<_> = values.iter().filter_map(|v| self.get_value(*v)).collect();

                let types: Vec<_> = vals.iter().map(|v| v.get_type()).collect();
                let struct_ty = self.context.struct_type(&types, false);
                let mut struct_val = struct_ty.get_undef();

                for (i, val) in vals.iter().enumerate() {
                    struct_val = self
                        .builder
                        .build_insert_value(struct_val, *val, i as u32, "tuple")
                        .ok()?
                        .into_struct_value();
                }

                Some(struct_val.into())
            }

            Op::Array(values) => {
                let vals: Vec<_> = values.iter().filter_map(|v| self.get_value(*v)).collect();

                if vals.is_empty() {
                    return None;
                }

                let elem_ty = vals[0].get_type();
                let len = vals.len() as u32;

                // Create array type based on element type
                let arr_ty = match elem_ty {
                    inkwell::types::BasicTypeEnum::IntType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::FloatType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::PointerType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::ArrayType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::StructType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::VectorType(t) => t.array_type(len),
                };
                let mut arr_val = arr_ty.get_undef();

                for (i, val) in vals.iter().enumerate() {
                    arr_val = self
                        .builder
                        .build_insert_value(arr_val, *val, i as u32, "array")
                        .ok()?
                        .into_array_value();
                }

                Some(arr_val.into())
            }

            Op::Struct { name: _, fields } => {
                let vals: Vec<_> = fields
                    .iter()
                    .filter_map(|(_, v)| self.get_value(*v))
                    .collect();

                let types: Vec<_> = vals.iter().map(|v| v.get_type()).collect();
                let struct_ty = self.context.struct_type(&types, false);
                let mut struct_val = struct_ty.get_undef();

                for (i, val) in vals.iter().enumerate() {
                    struct_val = self
                        .builder
                        .build_insert_value(struct_val, *val, i as u32, "struct")
                        .ok()?
                        .into_struct_value();
                }

                Some(struct_val.into())
            }

            Op::PerformEffect { .. } => {
                // Effects are handled at runtime, not in LLVM IR
                // Could generate calls to effect runtime here
                None
            }
        }
    }

    /// Compile a constant value
    fn compile_constant(&mut self, constant: &HlirConstant) -> Option<BasicValueEnum<'ctx>> {
        match constant {
            HlirConstant::Unit => Some(self.context.struct_type(&[], false).const_zero().into()),

            HlirConstant::Bool(b) => {
                Some(self.context.bool_type().const_int(*b as u64, false).into())
            }

            HlirConstant::Int(n, ty) => {
                let bits = self.types.int_bit_width(ty).unwrap_or(64);
                let int_ty = self.types.int_type_for_bits(bits);
                Some(int_ty.const_int(*n as u64, self.types.is_signed(ty)).into())
            }

            HlirConstant::Float(f, ty) => {
                let float_ty = match ty {
                    HlirType::F32 => self.context.f32_type(),
                    HlirType::F64 => self.context.f64_type(),
                    _ => self.context.f64_type(),
                };
                Some(float_ty.const_float(*f).into())
            }

            HlirConstant::String(s) => {
                if let Some(ptr) = self.strings.get(s) {
                    return Some((*ptr).into());
                }

                let global = self.builder.build_global_string_ptr(s, "str").ok()?;
                let ptr = global.as_pointer_value();
                self.strings.insert(s.clone(), ptr);
                Some(ptr.into())
            }

            HlirConstant::Array(elements) => {
                let vals: Vec<_> = elements
                    .iter()
                    .filter_map(|e| self.compile_constant(e))
                    .collect();

                if vals.is_empty() {
                    return None;
                }

                let elem_ty = vals[0].get_type();
                let len = vals.len() as u32;
                // array_type() is not available on BasicTypeEnum, must match on specific type
                let arr_ty = match elem_ty {
                    inkwell::types::BasicTypeEnum::IntType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::FloatType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::PointerType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::ArrayType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::StructType(t) => t.array_type(len),
                    inkwell::types::BasicTypeEnum::VectorType(t) => t.array_type(len),
                };
                let mut arr_val = arr_ty.get_undef();

                for (i, val) in vals.iter().enumerate() {
                    arr_val = self
                        .builder
                        .build_insert_value(arr_val, *val, i as u32, "const_array")
                        .ok()?
                        .into_array_value();
                }

                Some(arr_val.into())
            }

            HlirConstant::Struct(fields) => {
                let vals: Vec<_> = fields
                    .iter()
                    .filter_map(|f| self.compile_constant(f))
                    .collect();

                let types: Vec<_> = vals.iter().map(|v| v.get_type()).collect();
                let struct_ty = self.context.struct_type(&types, false);
                let mut struct_val = struct_ty.get_undef();

                for (i, val) in vals.iter().enumerate() {
                    struct_val = self
                        .builder
                        .build_insert_value(struct_val, *val, i as u32, "const_struct")
                        .ok()?
                        .into_struct_value();
                }

                Some(struct_val.into())
            }

            HlirConstant::Null(ty) => {
                let llvm_ty = self.types.convert(ty);
                if let inkwell::types::BasicTypeEnum::PointerType(ptr_ty) = llvm_ty {
                    Some(ptr_ty.const_null().into())
                } else {
                    // Use opaque pointer for LLVM 15+ compatibility
                    let ptr_ty = self.context.ptr_type(AddressSpace::default());
                    Some(ptr_ty.const_null().into())
                }
            }

            HlirConstant::Undef(ty) => {
                let llvm_ty = self.types.convert(ty);
                // get_undef() is not available on BasicTypeEnum, must match on specific type
                let undef_val = match llvm_ty {
                    inkwell::types::BasicTypeEnum::IntType(t) => t.get_undef().into(),
                    inkwell::types::BasicTypeEnum::FloatType(t) => t.get_undef().into(),
                    inkwell::types::BasicTypeEnum::PointerType(t) => t.get_undef().into(),
                    inkwell::types::BasicTypeEnum::ArrayType(t) => t.get_undef().into(),
                    inkwell::types::BasicTypeEnum::StructType(t) => t.get_undef().into(),
                    inkwell::types::BasicTypeEnum::VectorType(t) => t.get_undef().into(),
                };
                Some(undef_val)
            }

            HlirConstant::FunctionRef(name) => self
                .functions
                .get(name)
                .map(|f| f.as_global_value().as_pointer_value().into()),

            HlirConstant::GlobalRef(name) => self
                .module
                .get_global(name)
                .map(|g| g.as_pointer_value().into()),
        }
    }

    /// Compile a binary operation
    fn compile_binary_op(
        &mut self,
        op: BinaryOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
        _ty: &HlirType,
    ) -> Option<BasicValueEnum<'ctx>> {
        match op {
            // Integer arithmetic
            BinaryOp::Add => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_add(l, r, "add")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::Sub => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_sub(l, r, "sub")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::Mul => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_mul(l, r, "mul")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::SDiv => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_signed_div(l, r, "sdiv")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::UDiv => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_unsigned_div(l, r, "udiv")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::SRem => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_signed_rem(l, r, "srem")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::URem => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_unsigned_rem(l, r, "urem")
                    .ok()
                    .map(|v| v.into())
            }

            // Float arithmetic
            BinaryOp::FAdd => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_add(l, r, "fadd")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FSub => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_sub(l, r, "fsub")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FMul => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_mul(l, r, "fmul")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FDiv => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_div(l, r, "fdiv")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FRem => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_rem(l, r, "frem")
                    .ok()
                    .map(|v| v.into())
            }

            // Bitwise operations
            BinaryOp::And => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder.build_and(l, r, "and").ok().map(|v| v.into())
            }
            BinaryOp::Or => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder.build_or(l, r, "or").ok().map(|v| v.into())
            }
            BinaryOp::Xor => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder.build_xor(l, r, "xor").ok().map(|v| v.into())
            }
            BinaryOp::Shl => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_left_shift(l, r, "shl")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::AShr => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_right_shift(l, r, true, "ashr")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::LShr => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_right_shift(l, r, false, "lshr")
                    .ok()
                    .map(|v| v.into())
            }

            // Integer comparisons
            BinaryOp::Eq => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::EQ, l, r, "eq")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::Ne => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::NE, l, r, "ne")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::SLt => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::SLT, l, r, "slt")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::SLe => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::SLE, l, r, "sle")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::SGt => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::SGT, l, r, "sgt")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::SGe => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::SGE, l, r, "sge")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::ULt => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::ULT, l, r, "ult")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::ULe => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::ULE, l, r, "ule")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::UGt => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::UGT, l, r, "ugt")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::UGe => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                self.builder
                    .build_int_compare(IntPredicate::UGE, l, r, "uge")
                    .ok()
                    .map(|v| v.into())
            }

            // Float comparisons
            BinaryOp::FOEq => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_compare(FloatPredicate::OEQ, l, r, "foeq")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FONe => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_compare(FloatPredicate::ONE, l, r, "fone")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FOLt => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_compare(FloatPredicate::OLT, l, r, "folt")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FOLe => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_compare(FloatPredicate::OLE, l, r, "fole")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FOGt => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_compare(FloatPredicate::OGT, l, r, "fogt")
                    .ok()
                    .map(|v| v.into())
            }
            BinaryOp::FOGe => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                self.builder
                    .build_float_compare(FloatPredicate::OGE, l, r, "foge")
                    .ok()
                    .map(|v| v.into())
            }

            BinaryOp::Concat => {
                // Array/slice concatenation requires runtime support
                // TODO: Implement proper array concatenation with memcpy
                None
            }
        }
    }

    /// Compile a unary operation
    fn compile_unary_op(
        &mut self,
        op: UnaryOp,
        val: BasicValueEnum<'ctx>,
        _ty: &HlirType,
    ) -> Option<BasicValueEnum<'ctx>> {
        match op {
            UnaryOp::Neg => {
                let v = val.into_int_value();
                self.builder.build_int_neg(v, "neg").ok().map(|v| v.into())
            }
            UnaryOp::FNeg => {
                let v = val.into_float_value();
                self.builder
                    .build_float_neg(v, "fneg")
                    .ok()
                    .map(|v| v.into())
            }
            UnaryOp::Not => {
                let v = val.into_int_value();
                self.builder.build_not(v, "not").ok().map(|v| v.into())
            }
        }
    }

    /// Compile a type cast
    fn compile_cast(
        &mut self,
        val: BasicValueEnum<'ctx>,
        from_ty: &HlirType,
        to_ty: &HlirType,
    ) -> Option<BasicValueEnum<'ctx>> {
        // Handle refinement type unwrapping: if the value is a single-element struct
        // but the type system says it should be a primitive, extract the inner value.
        // This handles cases like `{ r: f64 | constraint }` which may be represented
        // as `{ f64 }` in LLVM IR but should be treated as `f64`.
        let val = self.unwrap_refinement_struct(val, from_ty);

        let from_int = self.types.is_integer_type(from_ty);
        let from_float = self.types.is_float_type(from_ty);
        let to_int = self.types.is_integer_type(to_ty);
        let to_float = self.types.is_float_type(to_ty);

        let from_bits = self.types.int_bit_width(from_ty).unwrap_or(64);
        let to_bits = self.types.int_bit_width(to_ty).unwrap_or(64);

        if from_int && to_int {
            let v = val.into_int_value();
            let target_ty = self.types.int_type_for_bits(to_bits);

            if to_bits > from_bits {
                // Extend
                if self.types.is_signed(from_ty) {
                    self.builder
                        .build_int_s_extend(v, target_ty, "sext")
                        .ok()
                        .map(|v| v.into())
                } else {
                    self.builder
                        .build_int_z_extend(v, target_ty, "zext")
                        .ok()
                        .map(|v| v.into())
                }
            } else if to_bits < from_bits {
                // Truncate
                self.builder
                    .build_int_truncate(v, target_ty, "trunc")
                    .ok()
                    .map(|v| v.into())
            } else {
                // Same size, just return
                Some(val)
            }
        } else if from_float && to_float {
            let v = val.into_float_value();
            let from_f64 = matches!(from_ty, HlirType::F64);
            let to_f64 = matches!(to_ty, HlirType::F64);

            if !from_f64 && to_f64 {
                // f32 -> f64
                self.builder
                    .build_float_ext(v, self.context.f64_type(), "fpext")
                    .ok()
                    .map(|v| v.into())
            } else if from_f64 && !to_f64 {
                // f64 -> f32
                self.builder
                    .build_float_trunc(v, self.context.f32_type(), "fptrunc")
                    .ok()
                    .map(|v| v.into())
            } else {
                Some(val)
            }
        } else if from_int && to_float {
            let v = val.into_int_value();
            let float_ty = self
                .types
                .float_type_for_bits(if matches!(to_ty, HlirType::F32) {
                    32
                } else {
                    64
                });

            if self.types.is_signed(from_ty) {
                self.builder
                    .build_signed_int_to_float(v, float_ty, "sitofp")
                    .ok()
                    .map(|v| v.into())
            } else {
                self.builder
                    .build_unsigned_int_to_float(v, float_ty, "uitofp")
                    .ok()
                    .map(|v| v.into())
            }
        } else if from_float && to_int {
            let v = val.into_float_value();
            let int_ty = self.types.int_type_for_bits(to_bits);

            if self.types.is_signed(to_ty) {
                self.builder
                    .build_float_to_signed_int(v, int_ty, "fptosi")
                    .ok()
                    .map(|v| v.into())
            } else {
                self.builder
                    .build_float_to_unsigned_int(v, int_ty, "fptoui")
                    .ok()
                    .map(|v| v.into())
            }
        } else if matches!(from_ty, HlirType::Ptr(_)) && matches!(to_ty, HlirType::Ptr(_)) {
            // Pointer to pointer cast - with opaque pointers in LLVM 15, no cast needed
            Some(val)
        } else if matches!(from_ty, HlirType::Ptr(_)) || matches!(to_ty, HlirType::Ptr(_)) {
            // Pointer to int or int to pointer
            let target_ty = self.types.convert(to_ty);
            if matches!(from_ty, HlirType::Ptr(_)) {
                // Pointer to int: use ptrtoint
                if let inkwell::types::BasicTypeEnum::IntType(int_ty) = target_ty {
                    self.builder
                        .build_ptr_to_int(val.into_pointer_value(), int_ty, "ptrtoint")
                        .ok()
                        .map(|v| v.into())
                } else {
                    Some(val) // Can't convert, just pass through
                }
            } else {
                // Int to pointer: use inttoptr
                let ptr_ty = self.context.ptr_type(AddressSpace::default());
                self.builder
                    .build_int_to_ptr(val.into_int_value(), ptr_ty, "inttoptr")
                    .ok()
                    .map(|v| v.into())
            }
        } else {
            // Fallback for non-pointer bitcasts (e.g., int <-> float of same size)
            let target_ty = self.types.convert(to_ty);
            // In LLVM 15+, method is named build_bit_cast (with underscore)
            self.builder
                .build_bit_cast(val, target_ty, "bitcast")
                .ok()
                .map(|v| v.into())
        }
    }

    /// Unwrap a refinement type struct if the value is a single-element struct.
    ///
    /// Refinement types like `type OrbitRatio = { r: f64 | 0.25 <= r && r <= 1.0 }`
    /// may be lowered to wrapper structs `{ f64 }` in some code paths, but when
    /// casting to a primitive, we need to extract the inner value.
    ///
    /// This function recursively extracts values from single-element struct wrappers
    /// until we get a non-struct value (int, float, etc.).
    ///
    /// For constant struct values, we use `get_field_at_index` directly.
    /// For non-constant values, we use `build_extract_value` to emit IR.
    fn unwrap_refinement_struct(
        &mut self,
        val: BasicValueEnum<'ctx>,
        _expected_ty: &HlirType,
    ) -> BasicValueEnum<'ctx> {
        let mut current = val;

        // Keep unwrapping single-element structs until we get a primitive
        loop {
            if let BasicValueEnum::StructValue(sv) = current {
                // Check if it has exactly one field (refinement type wrapper)
                if sv.count_fields() != 1 {
                    break;
                }

                // For constant struct values, use get_field_at_index directly
                // (build_extract_value only works for runtime values)
                if sv.is_const() {
                    if let Some(inner) = sv.get_field_at_index(0) {
                        current = inner;
                        continue;
                    } else {
                        break;
                    }
                }

                // For non-constant values, use build_extract_value
                match self.builder.build_extract_value(sv, 0, "unwrap_refinement") {
                    Ok(inner) => {
                        current = inner;
                        continue;
                    }
                    Err(_) => {
                        break;
                    }
                }
            } else {
                // Not a struct, we're done unwrapping
                break;
            }
        }

        current
    }

    /// Compile a terminator
    fn compile_terminator(&mut self, term: &HlirTerminator) {
        match term {
            HlirTerminator::Return(val) => {
                if let Some(val_id) = val {
                    if let Some(ret_val) = self.get_value(*val_id) {
                        let _ = self.builder.build_return(Some(&ret_val));
                    } else {
                        let _ = self.builder.build_return(None);
                    }
                } else {
                    let _ = self.builder.build_return(None);
                }
            }

            HlirTerminator::Branch(target) => {
                if let Some(bb) = self.blocks.get(target) {
                    let _ = self.builder.build_unconditional_branch(*bb);
                }
            }

            HlirTerminator::CondBranch {
                condition,
                then_block,
                else_block,
            } => {
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

            HlirTerminator::Switch {
                value,
                default,
                cases,
            } => {
                if let (Some(val), Some(default_bb)) =
                    (self.get_value(*value), self.blocks.get(default))
                {
                    let val_int = val.into_int_value();
                    let switch_cases: Vec<_> = cases
                        .iter()
                        .filter_map(|(case_val, block_id)| {
                            self.blocks.get(block_id).map(|bb| {
                                let cv = val_int.get_type().const_int(*case_val as u64, true);
                                (cv, *bb)
                            })
                        })
                        .collect();

                    let _ = self
                        .builder
                        .build_switch(val_int, *default_bb, &switch_cases);
                }
            }

            HlirTerminator::Unreachable => {
                let _ = self.builder.build_unreachable();
            }
        }
    }

    /// Get a value from the map
    fn get_value(&self, id: ValueId) -> Option<BasicValueEnum<'ctx>> {
        if id == ValueId::UNIT {
            return Some(self.context.struct_type(&[], false).const_zero().into());
        }
        self.values.get(&id).copied()
    }

    /// Create a C main wrapper that calls the D main function
    fn create_main_wrapper(&mut self) {
        // Check if _start or main already exists
        if self.module.get_function("_main_wrapper").is_some() {
            return;
        }

        let i32_ty = self.context.i32_type();
        let main_type = i32_ty.fn_type(&[], false);
        let main_fn = self.module.add_function("_main_wrapper", main_type, None);

        let entry = self.context.append_basic_block(main_fn, "entry");
        self.builder.position_at_end(entry);

        // Call the D main function
        if let Some(d_main) = self.functions.get("main") {
            let call = self.builder.build_call(*d_main, &[], "result");
            if let Ok(call_val) = call {
                // If main returns a value, convert to i32
                if let Some(ret) = call_val.try_as_basic_value().left() {
                    if ret.is_int_value() {
                        let ret_int = ret.into_int_value();
                        // Truncate to i32 if needed
                        if ret_int.get_type().get_bit_width() > 32 {
                            if let Ok(truncated) =
                                self.builder.build_int_truncate(ret_int, i32_ty, "ret")
                            {
                                let _ = self.builder.build_return(Some(&truncated));
                                return;
                            }
                        } else if ret_int.get_type().get_bit_width() < 32 {
                            if let Ok(extended) =
                                self.builder.build_int_s_extend(ret_int, i32_ty, "ret")
                            {
                                let _ = self.builder.build_return(Some(&extended));
                                return;
                            }
                        } else {
                            let _ = self.builder.build_return(Some(&ret_int));
                            return;
                        }
                    }
                }
            }
        }

        // Return 0 by default
        let zero = i32_ty.const_zero();
        let _ = self.builder.build_return(Some(&zero));
    }

    /// Get the compiled module
    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Print LLVM IR to string
    pub fn print_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }

    /// Write LLVM IR to file
    pub fn write_ir(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.module
            .print_to_file(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    /// Verify the module is well-formed
    pub fn verify(&self) -> Result<(), String> {
        self.module.verify().map_err(|e| e.to_string())
    }

    /// Get optimization level
    pub fn opt_level(&self) -> OptLevel {
        self.opt_level
    }
}
