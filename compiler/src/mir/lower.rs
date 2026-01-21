//! HLIR to MIR lowering
//!
//! This module transforms HLIR (High-Level IR) into MIR (Mid-level IR).

use crate::hlir::{
    HlirModule, HlirFunction, HlirInstr, HlirTerminator, HlirConstant, HlirType,
    BlockId as HlirBlockId, ValueId as HlirValueId, Op, BinaryOp, UnaryOp, HlirTypeDefKind,
};
use crate::mir::builder::{FunctionBuilder, ModuleBuilder};
use crate::mir::types::*;
use crate::mir::{MirModule, MirFunction, MirBinaryOp, MirUnaryOp, MirCompareOp, FuncId, BlockId, ValueId};
use std::collections::HashMap;

/// Lower HLIR to MIR
pub fn lower(hlir: &HlirModule) -> MirModule {
    let mut lowering = HlirToMir::new();
    lowering.lower_module(hlir)
}

/// HLIR to MIR lowering context
struct HlirToMir {
    /// Map from HLIR function names to MIR types
    functions: HashMap<String, MirType>,
    /// Map of struct name to its fields (HLIR types)
    struct_defs: HashMap<String, Vec<(String, HlirType)>>,
    /// Current function ID counter
    next_func_id: usize,
}

impl HlirToMir {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
            struct_defs: HashMap::new(),
            next_func_id: 0,
        }
    }

    fn lower_module(&mut self, hlir: &HlirModule) -> MirModule {
        let mut module_builder = ModuleBuilder::new(&hlir.name);

        // Collect type definitions first (struct layouts)
        self.collect_type_defs(hlir, &mut module_builder);

        // First pass: collect function signatures
        for func in &hlir.functions {
            let sig = self.hlir_sig_to_mir(func);
            self.functions.insert(func.name.clone(), sig);
        }

        // Second pass: lower functions
        for func in &hlir.functions {
            let mir_func = self.lower_function(func);
            module_builder.add_function(mir_func);
        }

        module_builder.build()
    }

    fn collect_type_defs(&mut self, hlir: &HlirModule, module_builder: &mut ModuleBuilder) {
        for ty_def in &hlir.types {
            match &ty_def.kind {
                HlirTypeDefKind::Struct(fields) => {
                    self.struct_defs.insert(ty_def.name.clone(), fields.clone());
                    let mir_fields: Vec<(String, MirType)> = fields
                        .iter()
                        .map(|(name, ty)| (name.clone(), self.hlir_type_to_mir(ty)))
                        .collect();
                    module_builder.add_type(MirTypeDef::Struct {
                        name: ty_def.name.clone(),
                        fields: mir_fields,
                    });
                }
                HlirTypeDefKind::Enum(variants) => {
                    let mir_variants: Vec<(String, Vec<MirType>)> = variants
                        .iter()
                        .map(|(name, tys)| {
                            let mir_tys = tys.iter().map(|ty| self.hlir_type_to_mir(ty)).collect();
                            (name.clone(), mir_tys)
                        })
                        .collect();
                    module_builder.add_type(MirTypeDef::Enum {
                        name: ty_def.name.clone(),
                        variants: mir_variants,
                    });
                }
            }
        }
    }

    fn hlir_sig_to_mir(&self, func: &HlirFunction) -> MirType {
        let params: Vec<MirType> = func
            .params
            .iter()
            .map(|p| self.hlir_type_to_mir(&p.ty))
            .collect();
        let return_type = self.hlir_type_to_mir(&func.return_type);
        MirType::Function {
            params,
            return_type: Box::new(return_type),
        }
    }

    fn lower_function(&mut self, func: &HlirFunction) -> MirFunction {
        let func_id = FuncId(self.next_func_id);
        self.next_func_id += 1;

        let return_type = self.hlir_type_to_mir(&func.return_type);
        let mut builder = FunctionBuilder::new(func_id, func.name.clone(), return_type);

        // Lowering context for this function
        let mut ctx = FunctionLoweringContext::new();

        // Add parameters
        for param in &func.params {
            let mir_type = self.hlir_type_to_mir(&param.ty);
            builder.add_param(param.name.clone(), mir_type);
            let param_value = builder.fresh_value();
            ctx.set_value(param.value, param_value, param.ty.clone());
        }

        // Create blocks (entry block already exists)
        for (i, block) in func.blocks.iter().enumerate() {
            if i == 0 {
                // Entry block already created by FunctionBuilder
                ctx.block_map.insert(block.id, BlockId(0));
            } else {
                let mir_block = builder.create_block(&format!("bb{}", block.id.0));
                ctx.block_map.insert(block.id, mir_block);
            }
        }

        // Lower each block
        for block in &func.blocks {
            let mir_block_id = ctx.block_map.get(&block.id)
                .copied()
                .expect("Block should be mapped");
            builder.switch_to_block(mir_block_id);

            // Lower instructions
            for instr in &block.instructions {
                self.lower_instruction(instr, &mut builder, &mut ctx);
            }

            // Lower terminator
            self.lower_terminator(&block.terminator, &mut builder, &ctx);
        }

        builder.build()
    }

    fn lower_instruction(
        &self,
        instr: &HlirInstr,
        builder: &mut FunctionBuilder,
        ctx: &mut FunctionLoweringContext,
    ) {
        match &instr.op {
            Op::Const(value) => {
                let mir_const = self.hlir_constant_to_mir(value);
                let mir_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let result_id = builder.fresh_value();
                    builder.build_const(result_id, mir_const, mir_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::Copy(source) => {
                // Copy is essentially a no-op in SSA form - just map to the same value
                if let Some(result) = instr.result {
                    let source_value = ctx.get_value(*source);
                    if let Some(source_ty) = ctx.get_type(*source) {
                        ctx.set_value(result, source_value, source_ty.clone());
                    } else {
                        ctx.value_map.insert(result, source_value);
                    }
                }
            }

            Op::Binary { op, left, right } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let left_value = ctx.get_value(*left);
                    let right_value = ctx.get_value(*right);
                    let result_id = builder.fresh_value();

                    match op {
                        BinaryOp::Add | BinaryOp::FAdd => {
                            builder.build_add(result_id, left_value, right_value, result_type);
                        }
                        BinaryOp::Sub | BinaryOp::FSub => {
                            builder.build_sub(result_id, left_value, right_value, result_type);
                        }
                        BinaryOp::Mul | BinaryOp::FMul => {
                            builder.build_mul(result_id, left_value, right_value, result_type);
                        }
                        BinaryOp::SDiv | BinaryOp::UDiv | BinaryOp::FDiv => {
                            builder.build_div(result_id, left_value, right_value, result_type);
                        }
                        BinaryOp::SRem | BinaryOp::URem | BinaryOp::FRem => {
                            builder.build_rem(result_id, left_value, right_value, result_type);
                        }
                        BinaryOp::And => {
                            builder.build_binary(result_id, MirBinaryOp::And, left_value, right_value, result_type);
                        }
                        BinaryOp::Or => {
                            builder.build_binary(result_id, MirBinaryOp::Or, left_value, right_value, result_type);
                        }
                        BinaryOp::Xor => {
                            builder.build_binary(result_id, MirBinaryOp::Xor, left_value, right_value, result_type);
                        }
                        BinaryOp::Shl => {
                            builder.build_binary(result_id, MirBinaryOp::Shl, left_value, right_value, result_type);
                        }
                        BinaryOp::AShr => {
                            builder.build_binary(result_id, MirBinaryOp::AShr, left_value, right_value, result_type);
                        }
                        BinaryOp::LShr => {
                            builder.build_binary(result_id, MirBinaryOp::LShr, left_value, right_value, result_type);
                        }
                        BinaryOp::Eq | BinaryOp::FOEq => {
                            builder.build_compare(result_id, MirCompareOp::Eq, left_value, right_value, MirType::Bool);
                        }
                        BinaryOp::Ne | BinaryOp::FONe => {
                            builder.build_compare(result_id, MirCompareOp::Ne, left_value, right_value, MirType::Bool);
                        }
                        BinaryOp::SLt | BinaryOp::ULt | BinaryOp::FOLt => {
                            builder.build_compare(result_id, MirCompareOp::Lt, left_value, right_value, MirType::Bool);
                        }
                        BinaryOp::SLe | BinaryOp::ULe | BinaryOp::FOLe => {
                            builder.build_compare(result_id, MirCompareOp::Le, left_value, right_value, MirType::Bool);
                        }
                        BinaryOp::SGt | BinaryOp::UGt | BinaryOp::FOGt => {
                            builder.build_compare(result_id, MirCompareOp::Gt, left_value, right_value, MirType::Bool);
                        }
                        BinaryOp::SGe | BinaryOp::UGe | BinaryOp::FOGe => {
                            builder.build_compare(result_id, MirCompareOp::Ge, left_value, right_value, MirType::Bool);
                        }
                        BinaryOp::Concat => {
                            // Array/slice concatenation - lower to a runtime call
                            builder.build_call(Some(result_id), "__concat".to_string(), vec![left_value, right_value], result_type);
                        }
                    }

                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::Unary { op, operand } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let operand_value = ctx.get_value(*operand);
                    let result_id = builder.fresh_value();

                    match op {
                        UnaryOp::Neg => {
                            builder.build_neg(result_id, operand_value, result_type);
                        }
                        UnaryOp::FNeg => {
                            builder.build_fneg(result_id, operand_value, result_type);
                        }
                        UnaryOp::Not => {
                            builder.build_unary(result_id, MirUnaryOp::Not, operand_value, MirType::Bool);
                        }
                    }

                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::Load { ptr } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let ptr_value = ctx.get_value(*ptr);
                    let result_id = builder.fresh_value();
                    builder.build_load(result_id, ptr_value, result_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::Store { ptr, value } => {
                let ptr_value = ctx.get_value(*ptr);
                let value_value = ctx.get_value(*value);
                builder.build_store(ptr_value, value_value);
            }

            Op::Call { func, args } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                let func_value = ctx.get_value(*func);
                let arg_values: Vec<ValueId> = args.iter().map(|&arg| ctx.get_value(arg)).collect();

                if let Some(result) = instr.result {
                    let result_id = builder.fresh_value();
                    builder.build_call_indirect(Some(result_id), func_value, arg_values, result_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                } else {
                    builder.build_call_indirect(None, func_value, arg_values, result_type);
                }
            }

            Op::CallDirect { name, args } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                let arg_values: Vec<ValueId> = args.iter().map(|&arg| ctx.get_value(arg)).collect();

                if let Some(result) = instr.result {
                    let result_id = builder.fresh_value();
                    builder.build_call(Some(result_id), name.clone(), arg_values, result_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                } else {
                    builder.build_call(None, name.clone(), arg_values, result_type);
                }
            }

            Op::Alloca { ty } => {
                let alloc_type = self.hlir_type_to_mir(ty);
                if let Some(result) = instr.result {
                    let result_id = builder.fresh_value();
                    builder.build_alloca(result_id, alloc_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::GetFieldPtr { base, field } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let base_value = ctx.get_value(*base);
                    let base_ty = ctx.get_type(*base);
                    if let Some((offset, _field_ty)) =
                        self.aggregate_offset_for_index(base_ty, *field)
                    {
                        let index_id = builder.fresh_value();
                        builder.build_const(index_id, MirConstant::Int(offset as i64), MirType::I64);
                        let result_id = builder.fresh_value();
                        builder.build_gep(result_id, base_value, vec![index_id], MirType::U8);
                        ctx.set_value(result, result_id, instr.ty.clone());
                    } else {
                        let index_id = builder.fresh_value();
                        builder.build_const(index_id, MirConstant::Int(*field as i64), MirType::I64);
                        let result_id = builder.fresh_value();
                        builder.build_gep(result_id, base_value, vec![index_id], result_type);
                        ctx.set_value(result, result_id, instr.ty.clone());
                    }
                }
            }

            Op::GetElementPtr { base, index } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let base_value = ctx.get_value(*base);
                    let index_value = ctx.get_value(*index);
                    let result_id = builder.fresh_value();
                    let elem_type = self
                        .element_type_for_gep(ctx.get_type(*base))
                        .unwrap_or_else(|| self.hlir_type_to_mir(&instr.ty));
                    builder.build_gep(result_id, base_value, vec![index_value], elem_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::Cast { value, source, target } => {
                let source_type = self.hlir_type_to_mir(source);
                let target_type = self.hlir_type_to_mir(target);
                if let Some(result) = instr.result {
                    let value_value = ctx.get_value(*value);
                    let result_id = builder.fresh_value();
                    builder.build_cast(result_id, value_value, source_type, target_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::Phi { incoming } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let mir_incoming: Vec<(BlockId, ValueId)> = incoming
                        .iter()
                        .map(|(block, value)| {
                            let mir_block = ctx.block_map.get(block)
                                .copied()
                                .expect("Block should be mapped");
                            let mir_value = ctx.get_value(*value);
                            (mir_block, mir_value)
                        })
                        .collect();

                    let result_id = builder.fresh_value();
                    builder.build_phi(result_id, mir_incoming, result_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            // Handle other operations as needed
            Op::ExtractValue { base, index } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let base_value = ctx.get_value(*base);
                    let base_ty = ctx.get_type(*base);
                    if let Some((offset, _field_ty)) =
                        self.aggregate_offset_for_index(base_ty, *index)
                    {
                        let index_id = builder.fresh_value();
                        builder.build_const(index_id, MirConstant::Int(offset as i64), MirType::I64);
                        let ptr_id = builder.fresh_value();
                        builder.build_gep(ptr_id, base_value, vec![index_id], MirType::U8);
                        let result_id = builder.fresh_value();
                        builder.build_load(result_id, ptr_id, result_type);
                        ctx.set_value(result, result_id, instr.ty.clone());
                    } else {
                        let index_id = builder.fresh_value();
                        builder.build_const(index_id, MirConstant::Int(*index as i64), MirType::I64);
                        let ptr_id = builder.fresh_value();
                        builder.build_gep(
                            ptr_id,
                            base_value,
                            vec![index_id],
                            MirType::Ptr(Box::new(result_type.clone())),
                        );
                        let result_id = builder.fresh_value();
                        builder.build_load(result_id, ptr_id, result_type);
                        ctx.set_value(result, result_id, instr.ty.clone());
                    }
                }
            }

            Op::InsertValue { base, value, index } => {
                if let Some(result) = instr.result {
                    let base_value = ctx.get_value(*base);
                    let value_value = ctx.get_value(*value);
                    let base_ty = ctx.get_type(*base);
                    if let Some((offset, _field_ty)) =
                        self.aggregate_offset_for_index(base_ty, *index)
                    {
                        let index_id = builder.fresh_value();
                        builder.build_const(index_id, MirConstant::Int(offset as i64), MirType::I64);
                        let ptr_id = builder.fresh_value();
                        builder.build_gep(ptr_id, base_value, vec![index_id], MirType::U8);
                        builder.build_store(ptr_id, value_value);
                        ctx.set_value(result, base_value, instr.ty.clone());
                    } else {
                        ctx.set_value(result, base_value, instr.ty.clone());
                    }
                }
            }

            Op::Tuple(values) => {
                let agg_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let result_id = builder.fresh_value();
                    builder.build_alloca(result_id, agg_type);
                    if let HlirType::Tuple(elem_tys) = &instr.ty {
                        let mut offset = 0usize;
                        for (idx, value_id) in values.iter().enumerate() {
                            if let Some(elem_ty) = elem_tys.get(idx) {
                                let elem_size =
                                    self.hlir_type_to_mir(elem_ty).size_bytes().unwrap_or(8);
                                let value_val = ctx.get_value(*value_id);
                                let offset_id = builder.fresh_value();
                                builder.build_const(
                                    offset_id,
                                    MirConstant::Int(offset as i64),
                                    MirType::I64,
                                );
                                let ptr_id = builder.fresh_value();
                                builder.build_gep(ptr_id, result_id, vec![offset_id], MirType::U8);
                                builder.build_store(ptr_id, value_val);
                                offset += elem_size;
                            }
                        }
                    }
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::Array(values) => {
                let agg_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let result_id = builder.fresh_value();
                    builder.build_alloca(result_id, agg_type);
                    if let HlirType::Array(elem_ty, _) = &instr.ty {
                        let elem_size = self
                            .hlir_type_to_mir(elem_ty)
                            .size_bytes()
                            .unwrap_or(8);
                        for (idx, value_id) in values.iter().enumerate() {
                            let offset = idx * elem_size;
                            let value_val = ctx.get_value(*value_id);
                            let offset_id = builder.fresh_value();
                            builder.build_const(
                                offset_id,
                                MirConstant::Int(offset as i64),
                                MirType::I64,
                            );
                            let ptr_id = builder.fresh_value();
                            builder.build_gep(ptr_id, result_id, vec![offset_id], MirType::U8);
                            builder.build_store(ptr_id, value_val);
                        }
                    }
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            Op::Struct { name, fields } => {
                let agg_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let result_id = builder.fresh_value();
                    builder.build_alloca(result_id, agg_type);
                    for (field_name, value_id) in fields {
                        if let Some((offset, _field_ty)) =
                            self.struct_field_offset(name, field_name)
                        {
                            let value_val = ctx.get_value(*value_id);
                            let offset_id = builder.fresh_value();
                            builder.build_const(
                                offset_id,
                                MirConstant::Int(offset as i64),
                                MirType::I64,
                            );
                            let ptr_id = builder.fresh_value();
                            builder.build_gep(ptr_id, result_id, vec![offset_id], MirType::U8);
                            builder.build_store(ptr_id, value_val);
                        }
                    }
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }

            _ => {
                // For unhandled operations, emit a warning and create a placeholder
                eprintln!("Warning: Unhandled HLIR instruction: {:?}", instr.op);
                if let Some(result) = instr.result {
                    let result_type = self.hlir_type_to_mir(&instr.ty);
                    let result_id = builder.fresh_value();
                    builder.build_const(result_id, MirConstant::Int(0), result_type);
                    ctx.set_value(result, result_id, instr.ty.clone());
                }
            }
        }
    }

    fn lower_terminator(
        &self,
        terminator: &HlirTerminator,
        builder: &mut FunctionBuilder,
        ctx: &FunctionLoweringContext,
    ) {
        match terminator {
            HlirTerminator::Return(value) => {
                if let Some(value) = value {
                    let value_id = ctx.get_value(*value);
                    builder.build_return(Some(value_id));
                } else {
                    builder.build_return(None);
                }
            }

            HlirTerminator::Branch(target) => {
                let target_block = ctx.block_map.get(target)
                    .copied()
                    .expect("Target block should be mapped");
                builder.build_branch(target_block);
            }

            HlirTerminator::CondBranch { condition, then_block, else_block } => {
                let condition_value = ctx.get_value(*condition);
                let then_block_mir = ctx.block_map.get(then_block)
                    .copied()
                    .expect("Then block should be mapped");
                let else_block_mir = ctx.block_map.get(else_block)
                    .copied()
                    .expect("Else block should be mapped");
                builder.build_cond_branch(condition_value, then_block_mir, else_block_mir);
            }

            HlirTerminator::Switch { value, default, cases } => {
                let value_id = ctx.get_value(*value);
                let default_block = ctx.block_map.get(default)
                    .copied()
                    .expect("Default block should be mapped");
                let case_blocks: Vec<(i64, BlockId)> = cases
                    .iter()
                    .map(|(const_val, block)| {
                        let mir_block = ctx.block_map.get(block)
                            .copied()
                            .expect("Case block should be mapped");
                        (*const_val, mir_block)
                    })
                    .collect();
                builder.build_switch(value_id, default_block, case_blocks);
            }

            HlirTerminator::Unreachable => {
                builder.build_unreachable();
            }
        }
    }

    fn hlir_type_to_mir(&self, ty: &HlirType) -> MirType {
        match ty {
            HlirType::Void => MirType::Void,
            HlirType::Bool => MirType::Bool,
            HlirType::I8 => MirType::I8,
            HlirType::I16 => MirType::I16,
            HlirType::I32 => MirType::I32,
            HlirType::I64 => MirType::I64,
            HlirType::I128 => MirType::I128,
            HlirType::U8 => MirType::U8,
            HlirType::U16 => MirType::U16,
            HlirType::U32 => MirType::U32,
            HlirType::U64 => MirType::U64,
            HlirType::U128 => MirType::U128,
            HlirType::F32 => MirType::F32,
            HlirType::F64 => MirType::F64,
            HlirType::Ptr(inner) => MirType::Ptr(Box::new(self.hlir_type_to_mir(inner))),
            HlirType::Array(elem, size) => MirType::Array(Box::new(self.hlir_type_to_mir(elem)), *size),
            HlirType::Struct(name) => {
                if let Some(fields) = self.struct_defs.get(name) {
                    let mir_fields = fields
                        .iter()
                        .map(|(field, ty)| (field.clone(), self.hlir_type_to_mir(ty)))
                        .collect();
                    MirType::Struct {
                        name: name.clone(),
                        fields: mir_fields,
                    }
                } else {
                    // Structs are lowered to pointers to opaque data if layout is unknown
                    MirType::Ptr(Box::new(MirType::Void))
                }
            },
            HlirType::Tuple(elems) => MirType::Tuple(elems.iter().map(|e| self.hlir_type_to_mir(e)).collect()),
            HlirType::Function { params, return_type } => MirType::Function {
                params: params.iter().map(|p| self.hlir_type_to_mir(p)).collect(),
                return_type: Box::new(self.hlir_type_to_mir(return_type)),
            },
            // Linear algebra types - lower to arrays of floats
            HlirType::Vec2 => MirType::Array(Box::new(MirType::F32), 2),
            HlirType::Vec3 => MirType::Array(Box::new(MirType::F32), 4), // Padded
            HlirType::Vec4 => MirType::Array(Box::new(MirType::F32), 4),
            HlirType::Mat2 => MirType::Array(Box::new(MirType::F32), 4),
            HlirType::Mat3 => MirType::Array(Box::new(MirType::F32), 9),
            HlirType::Mat4 => MirType::Array(Box::new(MirType::F32), 16),
            HlirType::Quat => MirType::Array(Box::new(MirType::F32), 4),
            HlirType::Dual => MirType::Tuple(vec![MirType::F64, MirType::F64]),
        }
    }

    fn hlir_constant_to_mir(&self, constant: &HlirConstant) -> MirConstant {
        match constant {
            HlirConstant::Unit => MirConstant::Unit,
            HlirConstant::Bool(b) => MirConstant::Bool(*b),
            HlirConstant::Int(i, _) => MirConstant::Int(*i),
            HlirConstant::Float(f, _) => MirConstant::Float(f.to_string()),
            HlirConstant::String(s) => MirConstant::String(s.clone()),
            HlirConstant::Array(_) => MirConstant::Int(0), // TODO: array constants
            HlirConstant::Struct(_) => MirConstant::Int(0), // TODO: struct constants
            HlirConstant::Null(_) => MirConstant::Null,
            HlirConstant::Undef(_) => MirConstant::Int(0), // Lower undef to zero
            HlirConstant::FunctionRef(name) => MirConstant::FunctionRef(name.clone()),
            HlirConstant::GlobalRef(name) => MirConstant::GlobalRef(name.clone()),
        }
    }

    fn element_type_for_gep(&self, base_ty: Option<&HlirType>) -> Option<MirType> {
        let ty = base_ty?;
        let elem = match ty {
            HlirType::Ptr(inner) => inner.as_ref(),
            HlirType::Array(inner, _) => inner.as_ref(),
            _ => return None,
        };
        Some(self.hlir_type_to_mir(elem))
    }

    fn aggregate_offset_for_index(
        &self,
        base_ty: Option<&HlirType>,
        index: usize,
    ) -> Option<(usize, MirType)> {
        let ty = base_ty?;
        match ty {
            HlirType::Tuple(elems) => {
                let mut offset = 0usize;
                for (idx, elem_ty) in elems.iter().enumerate() {
                    let elem_size = self.hlir_type_to_mir(elem_ty).size_bytes()?;
                    if idx == index {
                        return Some((offset, self.hlir_type_to_mir(elem_ty)));
                    }
                    offset += elem_size;
                }
                None
            }
            HlirType::Array(elem_ty, _) => {
                let elem_size = self.hlir_type_to_mir(elem_ty).size_bytes()?;
                Some((index * elem_size, self.hlir_type_to_mir(elem_ty)))
            }
            HlirType::Struct(name) => {
                let fields = self.struct_defs.get(name)?;
                let mut offset = 0usize;
                for (idx, (_field, field_ty)) in fields.iter().enumerate() {
                    let field_size = self.hlir_type_to_mir(field_ty).size_bytes()?;
                    if idx == index {
                        return Some((offset, self.hlir_type_to_mir(field_ty)));
                    }
                    offset += field_size;
                }
                None
            }
            HlirType::Ptr(inner) => self.aggregate_offset_for_index(Some(inner.as_ref()), index),
            _ => None,
        }
    }

    fn struct_field_offset(
        &self,
        struct_name: &str,
        field_name: &str,
    ) -> Option<(usize, MirType)> {
        let fields = self.struct_defs.get(struct_name)?;
        let mut offset = 0usize;
        for (name, field_ty) in fields {
            let field_size = self.hlir_type_to_mir(field_ty).size_bytes()?;
            if name == field_name {
                return Some((offset, self.hlir_type_to_mir(field_ty)));
            }
            offset += field_size;
        }
        None
    }
}

/// Per-function lowering context
struct FunctionLoweringContext {
    /// Map from HLIR block IDs to MIR block IDs
    block_map: HashMap<HlirBlockId, BlockId>,
    /// Map from HLIR value IDs to MIR value IDs
    value_map: HashMap<HlirValueId, ValueId>,
    /// Map from HLIR value IDs to HLIR types
    value_types: HashMap<HlirValueId, HlirType>,
}

impl FunctionLoweringContext {
    fn new() -> Self {
        Self {
            block_map: HashMap::new(),
            value_map: HashMap::new(),
            value_types: HashMap::new(),
        }
    }

    fn get_value(&self, hlir_value: HlirValueId) -> ValueId {
        self.value_map.get(&hlir_value)
            .copied()
            .unwrap_or_else(|| {
                eprintln!("Warning: HLIR value {:?} not found in map, using placeholder", hlir_value);
                ValueId(0)
            })
    }

    fn set_value(&mut self, hlir_value: HlirValueId, mir_value: ValueId, ty: HlirType) {
        self.value_map.insert(hlir_value, mir_value);
        self.value_types.insert(hlir_value, ty);
    }

    fn get_type(&self, hlir_value: HlirValueId) -> Option<&HlirType> {
        self.value_types.get(&hlir_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlir_to_mir_basic() {
        // This is a simplified test - in practice we'd need to create proper HLIR
        let hlir_module = HlirModule::new("test");

        let mir_module = lower(&hlir_module);

        assert_eq!(mir_module.name, "test");
        assert!(mir_module.functions.is_empty());
    }
}
