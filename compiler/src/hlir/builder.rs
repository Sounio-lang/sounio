//! HLIR Builder - convenient API for constructing HLIR
//!
//! The builder provides a high-level API for constructing HLIR functions
//! and basic blocks, managing SSA value numbering automatically.

use super::ir::*;
use crate::ast::Abi;
use std::collections::HashMap;

/// Builder for constructing HLIR modules
pub struct ModuleBuilder {
    module: HlirModule,
    next_func_id: u32,
}

impl ModuleBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            module: HlirModule::new(name),
            next_func_id: 0,
        }
    }

    pub fn add_type_def(&mut self, typedef: HlirTypeDef) {
        self.module.types.push(typedef);
    }

    pub fn add_global(&mut self, global: HlirGlobal) {
        self.module.globals.push(global);
    }

    pub fn add_function(&mut self, func: HlirFunction) {
        self.module.functions.push(func);
    }

    pub fn fresh_func_id(&mut self) -> FunctionId {
        let id = FunctionId(self.next_func_id);
        self.next_func_id += 1;
        id
    }

    pub fn build(self) -> HlirModule {
        self.module
    }
}

/// Builder for constructing HLIR functions
pub struct FunctionBuilder {
    pub func: HlirFunction,
    next_block_id: u32,
    next_value_id: u32,
    current_block: Option<BlockId>,
    /// Map from variable names to their current SSA values
    var_values: HashMap<String, ValueId>,
    /// Map from variable names to their stack slots (for mutable variables)
    var_slots: HashMap<String, ValueId>,
}

impl FunctionBuilder {
    pub fn new(id: FunctionId, name: impl Into<String>, return_type: HlirType) -> Self {
        Self {
            func: HlirFunction {
                id,
                name: name.into(),
                link_name: None,
                params: Vec::new(),
                return_type,
                effects: Vec::new(),
                blocks: Vec::new(),
                is_kernel: false,
                locals: HashMap::new(),
                is_variadic: false,
                abi: Abi::Rust,
                is_exported: false,
            },
            next_block_id: 0,
            next_value_id: 0,
            current_block: None,
            var_values: HashMap::new(),
            var_slots: HashMap::new(),
        }
    }

    /// Set the linker-visible symbol name for this function.
    pub fn set_link_name(&mut self, link_name: Option<String>) {
        self.func.link_name = link_name;
    }

    /// Mark this function as variadic.
    pub fn set_variadic(&mut self, is_variadic: bool) {
        self.func.is_variadic = is_variadic;
    }

    /// Add a parameter to the function
    pub fn add_param(&mut self, name: impl Into<String>, ty: HlirType) -> ValueId {
        let value = self.fresh_value();
        let name = name.into();
        self.var_values.insert(name.clone(), value);
        self.func.params.push(HlirParam { value, name, ty });
        value
    }

    /// Set the ABI for this function (C, System, etc.)
    pub fn set_abi(&mut self, abi: Abi) {
        self.func.abi = abi;
    }

    /// Mark this function as exported (external linkage)
    pub fn set_exported(&mut self, exported: bool) {
        self.func.is_exported = exported;
    }

    /// Create a new basic block
    pub fn create_block(&mut self, label: impl Into<String>) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        self.func.blocks.push(HlirBlock::new(id, label));
        id
    }

    /// Switch to building a different block
    pub fn switch_to_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    /// Get the current block ID
    pub fn current_block(&self) -> Option<BlockId> {
        self.current_block
    }

    /// Get mutable reference to current block
    fn current_block_mut(&mut self) -> &mut HlirBlock {
        let id = self.current_block.expect("No current block");
        self.func.get_block_mut(id).expect("Block not found")
    }

    /// Create a fresh SSA value
    pub fn fresh_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    // ==================== Variable Management ====================

    /// Bind a variable name to an SSA value
    pub fn bind_var(&mut self, name: impl Into<String>, value: ValueId) {
        self.var_values.insert(name.into(), value);
    }

    /// Get the current SSA value for a variable
    pub fn get_var(&self, name: &str) -> Option<ValueId> {
        self.var_values.get(name).copied()
    }

    /// Allocate a stack slot for a mutable variable
    pub fn alloc_var(&mut self, name: impl Into<String>, ty: HlirType) -> ValueId {
        let slot = self.build_alloca(ty.clone());
        let name = name.into();
        self.var_slots.insert(name.clone(), slot);
        self.func.locals.insert(slot, ty);
        slot
    }

    /// Get the stack slot for a mutable variable
    pub fn get_var_slot(&self, name: &str) -> Option<ValueId> {
        self.var_slots.get(name).copied()
    }

    /// Store to a mutable variable
    pub fn store_var(&mut self, name: &str, value: ValueId) {
        if let Some(slot) = self.var_slots.get(name).copied() {
            self.build_store(slot, value);
        }
    }

    /// Load from a mutable variable
    pub fn load_var(&mut self, name: &str, fallback_ty: HlirType) -> Option<ValueId> {
        self.var_slots.get(name).copied().map(|slot| {
            // Use the stored type from locals if available (more reliable than expr type)
            let ty = self.func.locals.get(&slot).cloned().unwrap_or(fallback_ty);
            self.build_load(slot, ty)
        })
    }

    // ==================== Instruction Builders ====================

    fn emit(&mut self, op: Op, ty: HlirType) -> ValueId {
        let result = self.fresh_value();
        self.current_block_mut().instructions.push(HlirInstr {
            result: Some(result),
            op,
            ty,
        });
        result
    }

    fn emit_void(&mut self, op: Op) {
        self.current_block_mut().instructions.push(HlirInstr {
            result: None,
            op,
            ty: HlirType::Void,
        });
    }

    /// Build a constant instruction
    pub fn build_const(&mut self, constant: HlirConstant, ty: HlirType) -> ValueId {
        self.emit(Op::Const(constant), ty)
    }

    /// Build an integer constant
    pub fn build_i64(&mut self, value: i64) -> ValueId {
        self.build_const(HlirConstant::Int(value, HlirType::I64), HlirType::I64)
    }

    /// Build a float constant
    pub fn build_f64(&mut self, value: f64) -> ValueId {
        self.build_const(HlirConstant::Float(value, HlirType::F64), HlirType::F64)
    }

    /// Build a boolean constant
    pub fn build_bool(&mut self, value: bool) -> ValueId {
        self.build_const(HlirConstant::Bool(value), HlirType::Bool)
    }

    /// Build a unit constant
    pub fn build_unit(&mut self) -> ValueId {
        self.build_const(HlirConstant::Unit, HlirType::Void)
    }

    /// Build a copy instruction
    pub fn build_copy(&mut self, value: ValueId, ty: HlirType) -> ValueId {
        self.emit(Op::Copy(value), ty)
    }

    /// Build a binary operation
    pub fn build_binary(
        &mut self,
        op: BinaryOp,
        left: ValueId,
        right: ValueId,
        ty: HlirType,
    ) -> ValueId {
        self.emit(Op::Binary { op, left, right }, ty)
    }

    /// Build an integer add
    pub fn build_add(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::Add, left, right, ty)
    }

    /// Build an integer subtract
    pub fn build_sub(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::Sub, left, right, ty)
    }

    /// Build an integer multiply
    pub fn build_mul(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::Mul, left, right, ty)
    }

    /// Build a signed integer divide
    pub fn build_sdiv(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::SDiv, left, right, ty)
    }

    /// Build a signed integer remainder
    pub fn build_srem(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::SRem, left, right, ty)
    }

    /// Build a float add
    pub fn build_fadd(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::FAdd, left, right, ty)
    }

    /// Build a float subtract
    pub fn build_fsub(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::FSub, left, right, ty)
    }

    /// Build a float multiply
    pub fn build_fmul(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::FMul, left, right, ty)
    }

    /// Build a float divide
    pub fn build_fdiv(&mut self, left: ValueId, right: ValueId, ty: HlirType) -> ValueId {
        self.build_binary(BinaryOp::FDiv, left, right, ty)
    }

    /// Build an equality comparison
    pub fn build_eq(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::Eq, left, right, HlirType::Bool)
    }

    /// Build a not-equal comparison
    pub fn build_ne(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::Ne, left, right, HlirType::Bool)
    }

    /// Build a signed less-than comparison
    pub fn build_slt(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::SLt, left, right, HlirType::Bool)
    }

    /// Build a signed less-or-equal comparison
    pub fn build_sle(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::SLe, left, right, HlirType::Bool)
    }

    /// Build a signed greater-than comparison
    pub fn build_sgt(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::SGt, left, right, HlirType::Bool)
    }

    /// Build a signed greater-or-equal comparison
    pub fn build_sge(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::SGe, left, right, HlirType::Bool)
    }

    /// Build a float ordered equality comparison
    pub fn build_foeq(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::FOEq, left, right, HlirType::Bool)
    }

    /// Build a float ordered less-than comparison
    pub fn build_folt(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::FOLt, left, right, HlirType::Bool)
    }

    /// Build a float ordered greater-than comparison
    pub fn build_fogt(&mut self, left: ValueId, right: ValueId) -> ValueId {
        self.build_binary(BinaryOp::FOGt, left, right, HlirType::Bool)
    }

    /// Build a unary operation
    pub fn build_unary(&mut self, op: UnaryOp, operand: ValueId, ty: HlirType) -> ValueId {
        self.emit(Op::Unary { op, operand }, ty)
    }

    /// Build an integer negation
    pub fn build_neg(&mut self, operand: ValueId, ty: HlirType) -> ValueId {
        self.build_unary(UnaryOp::Neg, operand, ty)
    }

    /// Build a float negation
    pub fn build_fneg(&mut self, operand: ValueId, ty: HlirType) -> ValueId {
        self.build_unary(UnaryOp::FNeg, operand, ty)
    }

    /// Build a logical not
    pub fn build_not(&mut self, operand: ValueId) -> ValueId {
        self.build_unary(UnaryOp::Not, operand, HlirType::Bool)
    }

    /// Build a direct function call
    pub fn build_call(
        &mut self,
        name: impl Into<String>,
        args: Vec<ValueId>,
        ret_ty: HlirType,
    ) -> ValueId {
        self.emit(
            Op::CallDirect {
                name: name.into(),
                args,
            },
            ret_ty,
        )
    }

    /// Build an indirect function call
    pub fn build_call_indirect(
        &mut self,
        func: ValueId,
        args: Vec<ValueId>,
        ret_ty: HlirType,
    ) -> ValueId {
        self.emit(Op::Call { func, args }, ret_ty)
    }

    /// Build a load from memory
    pub fn build_load(&mut self, ptr: ValueId, ty: HlirType) -> ValueId {
        self.emit(Op::Load { ptr }, ty)
    }

    /// Build a store to memory
    pub fn build_store(&mut self, ptr: ValueId, value: ValueId) {
        self.emit_void(Op::Store { ptr, value });
    }

    /// Build a stack allocation
    pub fn build_alloca(&mut self, ty: HlirType) -> ValueId {
        let ptr_ty = HlirType::Ptr(Box::new(ty.clone()));
        self.emit(Op::Alloca { ty }, ptr_ty)
    }

    /// Build a field pointer
    pub fn build_field_ptr(&mut self, base: ValueId, field: usize, field_ty: HlirType) -> ValueId {
        let ptr_ty = HlirType::Ptr(Box::new(field_ty));
        self.emit(Op::GetFieldPtr { base, field }, ptr_ty)
    }

    /// Build an element pointer
    pub fn build_elem_ptr(&mut self, base: ValueId, index: ValueId, elem_ty: HlirType) -> ValueId {
        let ptr_ty = HlirType::Ptr(Box::new(elem_ty));
        self.emit(Op::GetElementPtr { base, index }, ptr_ty)
    }

    /// Build a type cast
    pub fn build_cast(&mut self, value: ValueId, source: HlirType, target: HlirType) -> ValueId {
        self.emit(
            Op::Cast {
                value,
                source,
                target: target.clone(),
            },
            target,
        )
    }

    /// Build a phi node
    pub fn build_phi(&mut self, incoming: Vec<(BlockId, ValueId)>, ty: HlirType) -> ValueId {
        self.emit(Op::Phi { incoming }, ty)
    }

    /// Build an extract value
    pub fn build_extract(&mut self, base: ValueId, index: usize, ty: HlirType) -> ValueId {
        self.emit(Op::ExtractValue { base, index }, ty)
    }

    /// Build an insert value
    pub fn build_insert(
        &mut self,
        base: ValueId,
        value: ValueId,
        index: usize,
        ty: HlirType,
    ) -> ValueId {
        self.emit(Op::InsertValue { base, value, index }, ty)
    }

    /// Build a tuple construction
    pub fn build_tuple(&mut self, values: Vec<ValueId>, ty: HlirType) -> ValueId {
        self.emit(Op::Tuple(values), ty)
    }

    /// Build an array construction
    pub fn build_array(&mut self, values: Vec<ValueId>, ty: HlirType) -> ValueId {
        self.emit(Op::Array(values), ty)
    }

    /// Build a struct construction
    pub fn build_struct(
        &mut self,
        name: impl Into<String>,
        fields: Vec<(String, ValueId)>,
        ty: HlirType,
    ) -> ValueId {
        self.emit(
            Op::Struct {
                name: name.into(),
                fields,
            },
            ty,
        )
    }

    // ==================== Terminator Builders ====================

    /// Set the terminator for the current block
    fn set_terminator(&mut self, term: HlirTerminator) {
        self.current_block_mut().terminator = term;
    }

    /// Build a return
    pub fn build_return(&mut self, value: Option<ValueId>) {
        self.set_terminator(HlirTerminator::Return(value));
    }

    /// Build an unconditional branch
    pub fn build_branch(&mut self, target: BlockId) {
        self.set_terminator(HlirTerminator::Branch(target));
    }

    /// Build a conditional branch
    pub fn build_cond_branch(
        &mut self,
        condition: ValueId,
        then_block: BlockId,
        else_block: BlockId,
    ) {
        self.set_terminator(HlirTerminator::CondBranch {
            condition,
            then_block,
            else_block,
        });
    }

    /// Build a switch
    pub fn build_switch(&mut self, value: ValueId, default: BlockId, cases: Vec<(i64, BlockId)>) {
        self.set_terminator(HlirTerminator::Switch {
            value,
            default,
            cases,
        });
    }

    /// Mark block as unreachable
    pub fn build_unreachable(&mut self) {
        self.set_terminator(HlirTerminator::Unreachable);
    }

    /// Finish building the function
    pub fn build(self) -> HlirFunction {
        self.func
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_function() {
        let mut builder = FunctionBuilder::new(FunctionId(0), "add", HlirType::I64);
        let a = builder.add_param("a", HlirType::I64);
        let b = builder.add_param("b", HlirType::I64);

        let entry = builder.create_block("entry");
        builder.switch_to_block(entry);

        let sum = builder.build_add(a, b, HlirType::I64);
        builder.build_return(Some(sum));

        let func = builder.build();
        assert_eq!(func.name, "add");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_build_conditional() {
        let mut builder = FunctionBuilder::new(FunctionId(0), "abs", HlirType::I64);
        let n = builder.add_param("n", HlirType::I64);

        let entry = builder.create_block("entry");
        let then_block = builder.create_block("then");
        let else_block = builder.create_block("else");

        builder.switch_to_block(entry);
        let zero = builder.build_i64(0);
        let is_neg = builder.build_slt(n, zero);
        builder.build_cond_branch(is_neg, then_block, else_block);

        builder.switch_to_block(then_block);
        let neg_n = builder.build_neg(n, HlirType::I64);
        builder.build_return(Some(neg_n));

        builder.switch_to_block(else_block);
        builder.build_return(Some(n));

        let func = builder.build();
        assert_eq!(func.blocks.len(), 3);
    }
}
