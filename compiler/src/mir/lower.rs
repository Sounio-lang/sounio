//! HLIR to MIR lowering
//!
//! This module transforms HLIR (High-Level IR) into MIR (Mid-level IR).

use crate::hlir::{HlirModule, HlirFunction, HlirBlock, HlirInstr, HlirTerminator, HlirConstant, HlirType, BlockId, ValueId, Op, BinaryOp, UnaryOp};
use crate::mir::builder::MirFunctionBuilder;
use crate::mir::types::*;
use crate::mir::instructions::*;
use crate::mir::MirModule;
use std::collections::HashMap;

/// Lower HLIR to MIR
pub fn lower(hlir: &HlirModule) -> MirModule {
    let lowering = HlirToMir::new();
    lowering.lower_module(hlir)
}

/// HLIR to MIR lowering context
struct HlirToMir {
    /// Map from HLIR function names to MIR types
    functions: HashMap<String, MirType>,
    /// Map from value IDs to their types
    value_types: HashMap<ValueId, MirType>,
}

impl HlirToMir {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
            value_types: HashMap::new(),
        }
    }

    fn lower_module(&mut self, hlir: &HlirModule) -> MirModule {
        let mut module_builder = crate::mir::builder::ModuleBuilder::new();
        
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

    fn hir_sig_to_mir(&self, func: &HlirFunction) -> MirType {
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
        let mut func_builder = MirFunctionBuilder::new(&func.name);
        
        // Add parameters
        for param in &func.params {
            let mir_type = self.hlir_type_to_mir(&param.ty);
            func_builder.add_parameter(mir_type);
        }
        
        // Create blocks
        let mut block_mapping = HashMap::new();
        for block in &func.blocks {
            let mir_block = func_builder.create_block();
            block_mapping.insert(block.id, mir_block);
        }
        
        // Lower each block
        for block in &func.blocks {
            let mir_block = block_mapping.get(&block.id)
                .expect("Block should be mapped");
            func_builder.set_current_block(*mir_block);
            
            // Lower parameters
            for (param_value, param_type) in &block.params {
                let mir_type = self.hlir_type_to_mir(param_type);
                // Store parameter type info
                self.value_types.insert(*param_value, mir_type);
            }
            
            // Lower instructions
            for instr in &block.instructions {
                self.lower_instruction(instr, &mut func_builder, &block_mapping);
            }
            
            // Lower terminator
            self.lower_terminator(&block.terminator, &mut func_builder, &block_mapping);
        }

        func_builder.build()
    }

    fn lower_instruction(&mut self, instr: &HlirInstr, func_builder: &mut MirFunctionBuilder, block_mapping: &HashMap<BlockId, BlockId>) {
        match &instr.op {
            Op::Const(value) => {
                let mir_const = self.hlir_constant_to_mir(value);
                let mir_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let result_value = func_builder.const_(mir_const, mir_type);
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, mir_type);
                }
            }
            
            Op::Binary { op, left, right } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let left_value = func_builder.get_value(*left);
                    let right_value = func_builder.get_value(*right);
                    
                    let result_value = match op {
                        BinaryOp::Add | BinaryOp::FAdd => func_builder.add(left_value, right_value, result_type),
                        BinaryOp::Sub | BinaryOp::FSub => func_builder.sub(left_value, right_value, result_type),
                        BinaryOp::Mul | BinaryOp::FMul => func_builder.mul(left_value, right_value, result_type),
                        BinaryOp::SDiv | BinaryOp::UDiv | BinaryOp::FDiv => func_builder.div(left_value, right_value, result_type),
                        BinaryOp::SRem | BinaryOp::URem => func_builder.rem(left_value, right_value, result_type),
                        BinaryOp::And => func_builder.and(left_value, right_value),
                        BinaryOp::Or => func_builder.or(left_value, right_value),
                        BinaryOp::Xor => func_builder.xor(left_value, right_value),
                        BinaryOp::Shl => func_builder.shl(left_value, right_value),
                        BinaryOp::AShr | BinaryOp::LShr => func_builder.shr(left_value, right_value),
                        BinaryOp::Eq => func_builder.icmp_eq(left_value, right_value),
                        BinaryOp::Ne => func_builder.icmp_ne(left_value, right_value),
                        BinaryOp::SLt | BinaryOp::FOLt => func_builder.icmp_lt(left_value, right_value),
                        BinaryOp::SLe | BinaryOp::FOLe => func_builder.icmp_le(left_value, right_value),
                        BinaryOp::SGt | BinaryOp::FOGt => func_builder.icmp_gt(left_value, right_value),
                        BinaryOp::SGe | BinaryOp::FOGe => func_builder.icmp_ge(left_value, right_value),
                        BinaryOp::ULt => func_builder.icmp_ult(left_value, right_value),
                        BinaryOp::ULe => func_builder.icmp_ule(left_value, right_value),
                        BinaryOp::UGt => func_builder.icmp_ugt(left_value, right_value),
                        BinaryOp::UGe => func_builder.icmp_uge(left_value, right_value),
                        BinaryOp::FOEq => func_builder.icmp_eq(left_value, right_value),
                        BinaryOp::FONe => func_builder.icmp_ne(left_value, right_value),
                        _ => {
                            eprintln!("Warning: Unsupported binary operation: {:?}", op);
                            func_builder.undef(result_type)
                        }
                    };
                    
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, result_type);
                }
            }
            
            Op::Unary { op, operand } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let operand_value = func_builder.get_value(*operand);
                    
                    let result_value = match op {
                        UnaryOp::Neg => func_builder.neg(operand_value),
                        UnaryOp::FNeg => func_builder.fneg(operand_value),
                        UnaryOp::Not => func_builder.not(operand_value),
                    };
                    
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, result_type);
                }
            }
            
            Op::Load { ptr } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let ptr_value = func_builder.get_value(*ptr);
                    let result_value = func_builder.load(result_type, ptr_value);
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, result_type);
                }
            }
            
            Op::Store { ptr, value } => {
                let ptr_value = func_builder.get_value(*ptr);
                let value_value = func_builder.get_value(*value);
                func_builder.store(&instr.ty, ptr_value, value_value);
            }
            
            Op::CallDirect { name, args } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let arg_values: Vec<_> = args.iter().map(|&arg| func_builder.get_value(arg)).collect();
                    let result_value = func_builder.call(name.clone(), &arg_values, result_type);
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, result_type);
                }
            }
            
            Op::Alloca { ty } => {
                let result_type = self.hlir_type_to_mir(ty);
                if let Some(result) = instr.result {
                    let result_value = func_builder.alloca(result_type);
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, MirType::Ptr(Box::new(result_type)));
                }
            }
            
            Op::GetElementPtr { base, index } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let base_value = func_builder.get_value(*base);
                    let index_value = func_builder.get_value(*index);
                    let result_value = func_builder.gep(base_value, &[index_value], result_type);
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, result_type);
                }
            }
            
            Op::Cast { value, source, target } => {
                let result_type = self.hlir_type_to_mir(target);
                if let Some(result) = instr.result {
                    let value_value = func_builder.get_value(*value);
                    let source_type = self.hlir_type_to_mir(source);
                    let result_value = func_builder.cast(value_value, source_type, result_type);
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, result_type);
                }
            }
            
            Op::Phi { incoming } => {
                let result_type = self.hlir_type_to_mir(&instr.ty);
                if let Some(result) = instr.result {
                    let mir_incoming: Vec<_> = incoming
                        .iter()
                        .map(|(block, value)| {
                            let mir_block = block_mapping.get(block)
                                .expect("Block should be mapped");
                            let mir_value = func_builder.get_value(*value);
                            (*mir_block, mir_value)
                        })
                        .collect();
                    
                    let result_value = func_builder.phi(mir_incoming, result_type);
                    func_builder.set_value_in_current_block(result, result_value);
                    self.value_types.insert(result, result_type);
                }
            }
            
            _ => {
                eprintln!("Warning: Unhandled HLIR instruction: {:?}", instr.op);
            }
        }
    }

    fn lower_terminator(&mut self, terminator: &HlirTerminator, func_builder: &mut MirFunctionBuilder, block_mapping: &HashMap<BlockId, BlockId>) {
        match terminator {
            HlirTerminator::Return(value) => {
                if let Some(value) = value {
                    let value_value = func_builder.get_value(*value);
                    func_builder.return_(Some(value_value));
                } else {
                    func_builder.return_(None);
                }
            }
            
            HlirTerminator::Branch(target) => {
                let target_block = block_mapping.get(target)
                    .expect("Target block should be mapped");
                func_builder.branch(*target_block);
            }
            
            HlirTerminator::CondBranch { condition, then_block, else_block } => {
                let condition_value = func_builder.get_value(*condition);
                let then_block_mir = block_mapping.get(then_block)
                    .expect("Then block should be mapped");
                let else_block_mir = block_mapping.get(else_block)
                    .expect("Else block should be mapped");
                func_builder.conditional_branch(condition_value, *then_block_mir, *else_block_mir);
            }
            
            HlirTerminator::Switch { value, default, cases } => {
                let value_value = func_builder.get_value(*value);
                let default_block = block_mapping.get(default)
                    .expect("Default block should be mapped");
                let case_blocks: Vec<_> = cases
                    .iter()
                    .map(|(const_val, block)| {
                        let mir_block = block_mapping.get(block)
                            .expect("Case block should be mapped");
                        (*const_val, *mir_block)
                    })
                    .collect();
                func_builder.switch(value_value, *default_block, &case_blocks);
            }
            
            HlirTerminator::Unreachable => {
                func_builder.unreachable();
            }
        }
    }

    fn hir_type_to_mir(&self, ty: &HlirType) -> MirType {
        match ty {
            HlirType::Void => MirType::Void,
            HlirType::Bool => MirType::I1,
            HlirType::I8 => MirType::I8,
            HlirType::I16 => MirType::I16,
            HlirType::I32 => MirType::I32,
            HlirType::I64 => MirType::I64,
            HlirType::I128 => MirType::I128,
            HlirType::U8 => MirType::I8,
            HlirType::U16 => MirType::I16,
            HlirType::U32 => MirType::I32,
            HlirType::U64 => MirType::I64,
            HlirType::U128 => MirType::I128,
            HlirType::F32 => MirType::F32,
            HlirType::F64 => MirType::F64,
            HlirType::Ptr(inner) => MirType::Ptr(Box::new(self.hlir_type_to_mir(inner))),
            HlirType::Array(elem, size) => MirType::Array {
                size: *size,
                elem: Box::new(self.hlir_type_to_mir(elem)),
            },
            HlirType::Struct(name) => MirType::Struct {
                fields: vec![], // Simplified - would need struct definition info
            },
            HlirType::Tuple(elems) => MirType::Tuple {
                elems: elems.iter().map(|e| self.hlir_type_to_mir(e)).collect(),
            },
            HlirType::Function { params, return_type } => MirType::Function {
                params: params.iter().map(|p| self.hlir_type_to_mir(p)).collect(),
                return_type: Box::new(self.hlir_type_to_mir(return_type)),
            },
            _ => MirType::Void, // For unimplemented types
        }
    }

    fn hir_constant_to_mir(&self, constant: &HlirConstant) -> MirConstant {
        match constant {
            HlirConstant::Unit => MirConstant::Unit,
            HlirConstant::Bool(b) => MirConstant::Bool(*b),
            HlirConstant::Int(i, _) => MirConstant::Int(*i),
            HlirConstant::Float(f, _) => MirConstant::Float(*f),
            HlirConstant::String(s) => MirConstant::String(s.clone()),
            HlirConstant::FunctionRef(name) => MirConstant::FunctionRef(name.clone()),
            HlirConstant::GlobalRef(name) => MirConstant::GlobalRef(name.clone()),
            _ => MirConstant::Int(0, 0), // For unimplemented constants
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlir_to_mir_basic() {
        // This is a simplified test - in practice we'd need to create proper HLIR
        let hlir_module = HlirModule {
            name: "test".to_string(),
            functions: vec![],
            globals: vec![],
            types: vec![],
        };

        let mir_module = lower(&hlir_module);
        
        assert_eq!(mir_module.name, "test");
        assert!(mir_module.functions.is_empty());
    }
}