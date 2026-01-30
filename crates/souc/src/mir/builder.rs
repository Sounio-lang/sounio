//! MIR Builder
//!
//! This module provides the builder pattern for constructing MIR incrementally.

// Section headers use doc comment syntax for consistency with rustdoc rendering
#![allow(clippy::empty_line_after_doc_comments)]

use super::instructions::*;
use super::types::*;

/// Builder for MIR Functions
pub struct FunctionBuilder {
    /// Current function being built
    function: MirFunction,
    /// Next available value ID
    next_value_id: usize,
    /// Next available block ID
    next_block_id: usize,
    /// Current block being built
    current_block: Option<BlockId>,
}

impl FunctionBuilder {
    /// Create a new function builder
    pub fn new(id: FuncId, name: String, return_type: MirType) -> Self {
        let mut builder = Self {
            function: MirFunction::new(id, name, return_type),
            next_value_id: 0,
            next_block_id: 0,
            current_block: None,
        };

        // Create entry block
        let entry_id = builder.alloc_block_id();
        let entry_block = MirBlock::new(entry_id, "entry".to_string());
        builder.function.add_block(entry_block);
        builder.function.entry_block = entry_id;
        builder.switch_to_block(entry_id);

        builder
    }

    /// Generate a fresh value ID
    pub fn fresh_value(&mut self) -> ValueId {
        let id = self.next_value_id;
        self.next_value_id += 1;
        ValueId(id)
    }

    /// Generate a fresh block ID
    pub fn alloc_block_id(&mut self) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id += 1;
        BlockId(id)
    }

    /// Switch to building in a specific block
    pub fn switch_to_block(&mut self, block_id: BlockId) {
        self.current_block = Some(block_id);
    }

    /// Get the current block ID
    pub fn current_block(&self) -> Option<BlockId> {
        self.current_block
    }

    /// Add a parameter to the function
    pub fn add_param(&mut self, name: String, ty: MirType) {
        self.function.add_param(name, ty);
        // Reserve a value ID for this parameter
        self.next_value_id += 1;
    }

    /// Get a parameter value ID by index
    pub fn get_param(&self, index: usize) -> ValueId {
        ValueId(index)
    }

    /// Create a new block and return its ID
    pub fn create_block(&mut self, label: &str) -> BlockId {
        let block_id = self.alloc_block_id();
        let block = MirBlock::new(block_id, label.to_string());
        self.function.add_block(block);
        block_id
    }

    /// Add an instruction to the current block
    pub fn build_instruction(&mut self, instruction: MirInstruction) {
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.get_block_mut(block_id) {
                block.add_instruction(instruction);
            }
        }
    }

    /// Build a load instruction
    pub fn build_load(&mut self, result: ValueId, address: ValueId, ty: MirType) {
        let instruction = MirInstruction::Load {
            result,
            address,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a store instruction
    pub fn build_store(&mut self, address: ValueId, value: ValueId) {
        let instruction = MirInstruction::Store { address, value };
        self.build_instruction(instruction);
    }

    /// Build a get element pointer instruction
    pub fn build_gep(
        &mut self,
        result: ValueId,
        base: ValueId,
        indices: Vec<ValueId>,
        ty: MirType,
    ) {
        let instruction = MirInstruction::GetElementPtr {
            result,
            base,
            indices,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build an allocation instruction
    pub fn build_alloca(&mut self, result: ValueId, ty: MirType) {
        let instruction = MirInstruction::Alloca { result, ty };
        self.build_instruction(instruction);
    }

    /// Build a cast instruction
    pub fn build_cast(
        &mut self,
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    ) {
        let instruction = MirInstruction::Cast {
            result,
            source,
            source_ty,
            target_ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a binary operation
    pub fn build_binary(
        &mut self,
        result: ValueId,
        op: MirBinaryOp,
        left: ValueId,
        right: ValueId,
        ty: MirType,
    ) {
        let instruction = MirInstruction::Binary {
            result,
            op,
            left,
            right,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build an add operation
    pub fn build_add(&mut self, result: ValueId, left: ValueId, right: ValueId, ty: MirType) {
        self.build_binary(result, MirBinaryOp::Add, left, right, ty);
    }

    /// Build a subtraction operation
    pub fn build_sub(&mut self, result: ValueId, left: ValueId, right: ValueId, ty: MirType) {
        self.build_binary(result, MirBinaryOp::Sub, left, right, ty);
    }

    /// Build a multiplication operation
    pub fn build_mul(&mut self, result: ValueId, left: ValueId, right: ValueId, ty: MirType) {
        self.build_binary(result, MirBinaryOp::Mul, left, right, ty);
    }

    /// Build a division operation
    pub fn build_div(&mut self, result: ValueId, left: ValueId, right: ValueId, ty: MirType) {
        self.build_binary(result, MirBinaryOp::Div, left, right, ty);
    }

    /// Build a modulo operation
    pub fn build_rem(&mut self, result: ValueId, left: ValueId, right: ValueId, ty: MirType) {
        self.build_binary(result, MirBinaryOp::Rem, left, right, ty);
    }

    /// Build a comparison operation
    pub fn build_compare(
        &mut self,
        result: ValueId,
        op: MirCompareOp,
        left: ValueId,
        right: ValueId,
        ty: MirType,
    ) {
        let instruction = MirInstruction::Compare {
            result,
            op,
            left,
            right,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build an equality comparison
    pub fn build_eq(&mut self, left: ValueId, right: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.build_compare(result, MirCompareOp::Eq, left, right, MirType::Bool);
        result
    }

    /// Build a not equal comparison
    pub fn build_ne(&mut self, left: ValueId, right: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.build_compare(result, MirCompareOp::Ne, left, right, MirType::Bool);
        result
    }

    /// Build a less than comparison
    pub fn build_lt(&mut self, left: ValueId, right: ValueId, ty: MirType) -> ValueId {
        let result = self.fresh_value();
        self.build_compare(result, MirCompareOp::Lt, left, right, ty);
        result
    }

    /// Build a less than or equal comparison
    pub fn build_le(&mut self, left: ValueId, right: ValueId, ty: MirType) -> ValueId {
        let result = self.fresh_value();
        self.build_compare(result, MirCompareOp::Le, left, right, ty);
        result
    }

    /// Build a greater than comparison
    pub fn build_gt(&mut self, left: ValueId, right: ValueId, ty: MirType) -> ValueId {
        let result = self.fresh_value();
        self.build_compare(result, MirCompareOp::Gt, left, right, ty);
        result
    }

    /// Build a greater than or equal comparison
    pub fn build_ge(&mut self, left: ValueId, right: ValueId, ty: MirType) -> ValueId {
        let result = self.fresh_value();
        self.build_compare(result, MirCompareOp::Ge, left, right, ty);
        result
    }

    /// Build a function call
    pub fn build_call(
        &mut self,
        result: Option<ValueId>,
        func_name: String,
        args: Vec<ValueId>,
        ty: MirType,
    ) {
        let instruction = MirInstruction::Call {
            result,
            func_name,
            args,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build an indirect function call
    pub fn build_call_indirect(
        &mut self,
        result: Option<ValueId>,
        func_ptr: ValueId,
        args: Vec<ValueId>,
        ty: MirType,
    ) {
        let instruction = MirInstruction::CallIndirect {
            result,
            func_ptr,
            args,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a constant instruction
    pub fn build_const(&mut self, result: ValueId, value: MirConstant, ty: MirType) {
        let instruction = MirInstruction::Const { result, value, ty };
        self.build_instruction(instruction);
    }

    /// Build an integer constant
    pub fn build_i32(&mut self, value: i32) -> ValueId {
        let result = self.fresh_value();
        self.build_const(result, MirConstant::Int(value as i64), MirType::I32);
        result
    }

    /// Build an i64 constant
    pub fn build_i64(&mut self, value: i64) -> ValueId {
        let result = self.fresh_value();
        self.build_const(result, MirConstant::Int(value), MirType::I64);
        result
    }

    /// Build a u64 constant
    pub fn build_u64(&mut self, value: u64) -> ValueId {
        let result = self.fresh_value();
        self.build_const(result, MirConstant::UInt(value), MirType::U64);
        result
    }

    /// Build an f64 constant
    pub fn build_f64(&mut self, value: f64) -> ValueId {
        let result = self.fresh_value();
        self.build_const(result, MirConstant::Float(value.to_string()), MirType::F64);
        result
    }

    /// Build a boolean constant
    pub fn build_bool(&mut self, value: bool) -> ValueId {
        let result = self.fresh_value();
        self.build_const(result, MirConstant::Bool(value), MirType::Bool);
        result
    }

    /// Build a unit constant
    pub fn build_unit(&mut self) -> ValueId {
        let result = self.fresh_value();
        self.build_const(result, MirConstant::Unit, MirType::Unit);
        result
    }

    /// Build a null constant
    pub fn build_null(&mut self) -> ValueId {
        let result = self.fresh_value();
        self.build_const(
            result,
            MirConstant::Null,
            MirType::Ptr(Box::new(MirType::Void)),
        );
        result
    }

    /// Build a function reference constant
    pub fn build_function_ref(&mut self, name: String) -> ValueId {
        let result = self.fresh_value();
        self.build_const(
            result,
            MirConstant::FunctionRef(name),
            MirType::Ptr(Box::new(MirType::Void)),
        );
        result
    }

    /// Build a phi instruction
    pub fn build_phi(&mut self, result: ValueId, incoming: Vec<(BlockId, ValueId)>, ty: MirType) {
        let instruction = MirInstruction::Phi {
            result,
            incoming,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a select instruction
    pub fn build_select(
        &mut self,
        result: ValueId,
        condition: ValueId,
        true_value: ValueId,
        false_value: ValueId,
        ty: MirType,
    ) {
        let instruction = MirInstruction::Select {
            result,
            condition,
            true_value,
            false_value,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a unary operation
    pub fn build_unary(&mut self, result: ValueId, op: MirUnaryOp, operand: ValueId, ty: MirType) {
        let instruction = MirInstruction::Unary {
            result,
            op,
            operand,
            ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a negation operation
    pub fn build_neg(&mut self, result: ValueId, operand: ValueId, ty: MirType) {
        self.build_unary(result, MirUnaryOp::Neg, operand, ty);
    }

    /// Build a floating point negation
    pub fn build_fneg(&mut self, result: ValueId, operand: ValueId, ty: MirType) {
        self.build_unary(result, MirUnaryOp::FNeg, operand, ty);
    }

    /// Build a logical NOT
    pub fn build_not(&mut self, operand: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.build_unary(result, MirUnaryOp::Not, operand, MirType::Bool);
        result
    }

    /// Build a bitwise NOT
    pub fn build_bitnot(&mut self, result: ValueId, operand: ValueId, ty: MirType) {
        self.build_unary(result, MirUnaryOp::BitNot, operand, ty);
    }

    /// Build a zero extend
    pub fn build_zext(
        &mut self,
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    ) {
        let instruction = MirInstruction::ZExt {
            result,
            source,
            source_ty,
            target_ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a sign extend
    pub fn build_sext(
        &mut self,
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    ) {
        let instruction = MirInstruction::SExt {
            result,
            source,
            source_ty,
            target_ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a truncate
    pub fn build_trunc(
        &mut self,
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    ) {
        let instruction = MirInstruction::Trunc {
            result,
            source,
            source_ty,
            target_ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a float to int conversion
    pub fn build_fptosi(&mut self, result: ValueId, source: ValueId, target_ty: MirType) {
        let instruction = MirInstruction::FPToSI {
            result,
            source,
            target_ty,
        };
        self.build_instruction(instruction);
    }

    /// Build an int to float conversion
    pub fn build_sitofp(&mut self, result: ValueId, source: ValueId, target_ty: MirType) {
        let instruction = MirInstruction::SIToFP {
            result,
            source,
            target_ty,
        };
        self.build_instruction(instruction);
    }

    /// Build a pointer cast
    pub fn build_ptrcast(
        &mut self,
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    ) {
        let instruction = MirInstruction::PtrCast {
            result,
            source,
            source_ty,
            target_ty,
        };
        self.build_instruction(instruction);
    }

    /// Build terminators

    /// Build an unconditional branch
    pub fn build_branch(&mut self, target: BlockId) {
        let terminator = MirTerminator::Branch { target };
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.get_block_mut(block_id) {
                block.set_terminator(terminator);
            }
        }
    }

    /// Build a conditional branch
    pub fn build_cond_branch(
        &mut self,
        condition: ValueId,
        true_target: BlockId,
        false_target: BlockId,
    ) {
        let terminator = MirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        };
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.get_block_mut(block_id) {
                block.set_terminator(terminator);
            }
        }
    }

    /// Build a return instruction
    pub fn build_return(&mut self, value: Option<ValueId>) {
        let terminator = MirTerminator::Return { value };
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.get_block_mut(block_id) {
                block.set_terminator(terminator);
            }
        }
    }

    /// Build an unreachable instruction
    pub fn build_unreachable(&mut self) {
        let terminator = MirTerminator::Unreachable;
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.get_block_mut(block_id) {
                block.set_terminator(terminator);
            }
        }
    }

    /// Build a switch instruction
    pub fn build_switch(
        &mut self,
        value: ValueId,
        default_target: BlockId,
        cases: Vec<(i64, BlockId)>,
    ) {
        let terminator = MirTerminator::Switch {
            value,
            default_target,
            cases,
        };
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.get_block_mut(block_id) {
                block.set_terminator(terminator);
            }
        }
    }

    /// Build a call that doesn't return
    pub fn build_call_noreturn(&mut self, func_name: String, args: Vec<ValueId>) {
        let terminator = MirTerminator::CallNoReturn { func_name, args };
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.get_block_mut(block_id) {
                block.set_terminator(terminator);
            }
        }
    }

    /// Build the function and return it
    pub fn build(self) -> MirFunction {
        self.function
    }
}

/// Builder for MIR Modules
pub struct ModuleBuilder {
    /// Current module being built
    module: MirModule,
    /// Next available function ID
    next_func_id: usize,
    /// Next available value ID for globals
    next_global_id: usize,
}

impl ModuleBuilder {
    /// Create a new module builder
    pub fn new(name: &str) -> Self {
        Self {
            module: MirModule::new(name.to_string()),
            next_func_id: 0,
            next_global_id: 0,
        }
    }

    /// Generate a fresh function ID
    pub fn fresh_func_id(&mut self) -> FuncId {
        let id = self.next_func_id;
        self.next_func_id += 1;
        FuncId(id)
    }

    /// Generate a fresh global ID
    pub fn fresh_global_id(&mut self) -> ValueId {
        let id = self.next_global_id;
        self.next_global_id += 1;
        ValueId(id)
    }

    /// Create a new function builder
    pub fn create_function(&mut self, name: String, return_type: MirType) -> FunctionBuilder {
        let func_id = self.fresh_func_id();
        FunctionBuilder::new(func_id, name, return_type)
    }

    /// Add a function to the module
    pub fn add_function(&mut self, function: MirFunction) {
        self.module.add_function(function);
    }

    /// Add a global variable
    pub fn add_global(&mut self, global: MirGlobal) {
        self.module.add_global(global);
    }

    /// Create a global variable
    pub fn create_global(
        &mut self,
        name: String,
        ty: MirType,
        init: Option<MirConstant>,
        is_const: bool,
    ) -> ValueId {
        let global_id = self.fresh_global_id();
        let global = MirGlobal {
            id: global_id,
            name,
            ty,
            init,
            is_const,
        };
        self.add_global(global);
        global_id
    }

    /// Add a type definition
    pub fn add_type(&mut self, type_def: MirTypeDef) {
        self.module.add_type(type_def);
    }

    /// Build the module and return it
    pub fn build(self) -> MirModule {
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_builder() {
        let mut builder =
            FunctionBuilder::new(FuncId(0), "test_function".to_string(), MirType::I32);

        // Add parameters
        builder.add_param("x".to_string(), MirType::I32);
        builder.add_param("y".to_string(), MirType::I32);

        // Create some basic instructions
        let const_42 = builder.build_i32(42);
        let add_result = builder.fresh_value();
        let add_left = builder.fresh_value();
        let add_right = builder.fresh_value();

        // Create a new block
        let new_block = builder.create_block("test_block");
        builder.switch_to_block(new_block);

        // Build an add instruction
        builder.build_add(add_result, add_left, add_right, MirType::I32);

        // Build a return
        builder.build_return(Some(add_result));

        let function = builder.build();

        assert_eq!(function.params.len(), 2);
        assert_eq!(function.blocks.len(), 2); // entry + test_block
    }

    #[test]
    fn test_module_builder() {
        let mut module_builder = ModuleBuilder::new("test_module");

        // Create a function
        let mut func_builder = module_builder.create_function("add".to_string(), MirType::I32);

        // Build the function
        let result = func_builder.build_i32(0);
        func_builder.build_return(Some(result));

        // Add to module
        module_builder.add_function(func_builder.build());

        let module = module_builder.build();

        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.name, "test_module");
    }
}
