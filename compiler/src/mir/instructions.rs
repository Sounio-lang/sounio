//! MIR Instructions
//!
//! This module defines the instruction set for MIR (Mid-level IR).

use serde::{Deserialize, Serialize};
use super::types::*;

/// Basic block in MIR
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MirBlock {
    /// Block identifier
    pub id: BlockId,
    /// Block label (for debugging)
    pub label: String,
    /// Instructions in this block
    pub instructions: Vec<MirInstruction>,
    /// Block terminator
    pub terminator: MirTerminator,
}

impl MirBlock {
    /// Create a new basic block
    pub fn new(id: BlockId, label: String) -> Self {
        Self {
            id,
            label,
            instructions: Vec::new(),
            terminator: MirTerminator::Unreachable,
        }
    }

    /// Add an instruction to this block
    pub fn add_instruction(&mut self, instruction: MirInstruction) {
        self.instructions.push(instruction);
    }

    /// Set the terminator for this block
    pub fn set_terminator(&mut self, terminator: MirTerminator) {
        self.terminator = terminator;
    }

    /// Check if this block is empty (no instructions except terminator)
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

/// Function in MIR
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MirFunction {
    /// Function identifier
    pub id: FuncId,
    /// Function name
    pub name: String,
    /// Function parameters
    pub params: Vec<(String, MirType)>,
    /// Return type
    pub return_type: MirType,
    /// Basic blocks
    pub blocks: Vec<MirBlock>,
    /// Entry block ID
    pub entry_block: BlockId,
}

impl MirFunction {
    /// Create a new function
    pub fn new(id: FuncId, name: String, return_type: MirType) -> Self {
        Self {
            id,
            name,
            params: Vec::new(),
            return_type,
            blocks: Vec::new(),
            entry_block: BlockId(0),
        }
    }

    /// Add a parameter to this function
    pub fn add_param(&mut self, name: String, ty: MirType) {
        self.params.push((name, ty));
    }

    /// Add a basic block
    pub fn add_block(&mut self, block: MirBlock) {
        self.blocks.push(block);
    }

    /// Get a mutable reference to a block by ID
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut MirBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Get a reference to a block by ID
    pub fn get_block(&self, id: BlockId) -> Option<&MirBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Get the entry block
    pub fn entry(&self) -> &MirBlock {
        self.get_block(self.entry_block).expect("Entry block not found")
    }

    /// Get the entry block (mutable)
    pub fn entry_mut(&mut self) -> &mut MirBlock {
        self.get_block_mut(self.entry_block).expect("Entry block not found")
    }
}

/// MIR Module
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MirModule {
    /// Module name
    pub name: String,
    /// Functions in this module
    pub functions: Vec<MirFunction>,
    /// Global variables
    pub globals: Vec<MirGlobal>,
    /// Type definitions
    pub types: Vec<MirTypeDef>,
}

impl MirModule {
    /// Create a new module
    pub fn new(name: String) -> Self {
        Self {
            name,
            functions: Vec::new(),
            globals: Vec::new(),
            types: Vec::new(),
        }
    }

    /// Add a function to this module
    pub fn add_function(&mut self, function: MirFunction) {
        self.functions.push(function);
    }

    /// Add a global variable
    pub fn add_global(&mut self, global: MirGlobal) {
        self.globals.push(global);
    }

    /// Add a type definition
    pub fn add_type(&mut self, type_def: MirTypeDef) {
        self.types.push(type_def);
    }
}

/// Global variable
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MirGlobal {
    /// Global identifier
    pub id: ValueId,
    /// Variable name
    pub name: String,
    /// Type of the global
    pub ty: MirType,
    /// Initial value (if any)
    pub init: Option<MirConstant>,
    /// Whether this is a constant
    pub is_const: bool,
}

/// MIR Instructions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MirInstruction {
    /// Load from memory
    Load {
        result: ValueId,
        address: ValueId,
        ty: MirType,
    },
    /// Store to memory
    Store {
        address: ValueId,
        value: ValueId,
    },
    /// Get element pointer (GEP)
    GetElementPtr {
        result: ValueId,
        base: ValueId,
        indices: Vec<ValueId>,
        ty: MirType,
    },
    /// Load from global variable
    LoadGlobal {
        result: ValueId,
        global_name: String,
        ty: MirType,
    },
    /// Store to global variable
    StoreGlobal {
        global_name: String,
        value: ValueId,
    },
    /// Allocate on stack
    Alloca {
        result: ValueId,
        ty: MirType,
    },
    /// Cast between types
    Cast {
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Zero extend
    ZExt {
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Sign extend
    SExt {
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Truncate
    Trunc {
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Convert float to int
    FPToSI {
        result: ValueId,
        source: ValueId,
        target_ty: MirType,
    },
    /// Convert int to float
    SIToFP {
        result: ValueId,
        source: ValueId,
        target_ty: MirType,
    },
    /// Convert float to unsigned int
    FPToUI {
        result: ValueId,
        source: ValueId,
        target_ty: MirType,
    },
    /// Convert unsigned int to float
    UIToFP {
        result: ValueId,
        source: ValueId,
        target_ty: MirType,
    },
    /// Pointer cast
    PtrCast {
        result: ValueId,
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Binary operations
    Binary {
        result: ValueId,
        op: MirBinaryOp,
        left: ValueId,
        right: ValueId,
        ty: MirType,
    },
    /// Unary operations
    Unary {
        result: ValueId,
        op: MirUnaryOp,
        operand: ValueId,
        ty: MirType,
    },
    /// Compare operations
    Compare {
        result: ValueId,
        op: MirCompareOp,
        left: ValueId,
        right: ValueId,
        ty: MirType,
    },
    /// Function call
    Call {
        result: Option<ValueId>,
        func_name: String,
        args: Vec<ValueId>,
        ty: MirType,
    },
    /// Function call with function pointer
    CallIndirect {
        result: Option<ValueId>,
        func_ptr: ValueId,
        args: Vec<ValueId>,
        ty: MirType,
    },
    /// Constant value
    Const {
        result: ValueId,
        value: MirConstant,
        ty: MirType,
    },
    /// Phi function (SSA join point)
    Phi {
        result: ValueId,
        incoming: Vec<(BlockId, ValueId)>,
        ty: MirType,
    },
    /// Select (conditional move)
    Select {
        result: ValueId,
        condition: ValueId,
        true_value: ValueId,
        false_value: ValueId,
        ty: MirType,
    },
}

impl MirInstruction {
    /// Get the result value ID of this instruction, if any
    pub fn result(&self) -> Option<ValueId> {
        match self {
            MirInstruction::Load { result, .. } => Some(*result),
            MirInstruction::GetElementPtr { result, .. } => Some(*result),
            MirInstruction::LoadGlobal { result, .. } => Some(*result),
            MirInstruction::Alloca { result, .. } => Some(*result),
            MirInstruction::Cast { result, .. } => Some(*result),
            MirInstruction::ZExt { result, .. } => Some(*result),
            MirInstruction::SExt { result, .. } => Some(*result),
            MirInstruction::Trunc { result, .. } => Some(*result),
            MirInstruction::FPToSI { result, .. } => Some(*result),
            MirInstruction::SIToFP { result, .. } => Some(*result),
            MirInstruction::PtrCast { result, .. } => Some(*result),
            MirInstruction::Binary { result, .. } => Some(*result),
            MirInstruction::Unary { result, .. } => Some(*result),
            MirInstruction::Compare { result, .. } => Some(*result),
            MirInstruction::Call { result, .. } => *result,
            MirInstruction::CallIndirect { result, .. } => *result,
            MirInstruction::Phi { result, .. } => Some(*result),
            MirInstruction::Select { result, .. } => Some(*result),
            MirInstruction::Const { result, .. } => Some(*result),
            MirInstruction::FPToUI { result, .. } => Some(*result),
            MirInstruction::UIToFP { result, .. } => Some(*result),
            // Instructions without results
            MirInstruction::Store { .. } => None,
            MirInstruction::StoreGlobal { .. } => None,
        }
    }

    /// Get all values used by this instruction
    pub fn used_values(&self) -> Vec<ValueId> {
        match self {
            MirInstruction::Load { address, .. } => vec![*address],
            MirInstruction::Store { address, value } => vec![*address, *value],
            MirInstruction::GetElementPtr { base, indices, .. } => {
                let mut values = vec![*base];
                values.extend(indices.iter().copied());
                values
            }
            MirInstruction::LoadGlobal { .. } => vec![],
            MirInstruction::StoreGlobal { value, .. } => vec![*value],
            MirInstruction::Alloca { .. } => vec![],
            MirInstruction::Cast { source, .. } => vec![*source],
            MirInstruction::ZExt { source, .. } => vec![*source],
            MirInstruction::SExt { source, .. } => vec![*source],
            MirInstruction::Trunc { source, .. } => vec![*source],
            MirInstruction::FPToSI { source, .. } => vec![*source],
            MirInstruction::SIToFP { source, .. } => vec![*source],
            MirInstruction::PtrCast { source, .. } => vec![*source],
            MirInstruction::Binary { left, right, .. } => vec![*left, *right],
            MirInstruction::Unary { operand, .. } => vec![*operand],
            MirInstruction::Compare { left, right, .. } => vec![*left, *right],
            MirInstruction::Call { args, .. } => args.clone(),
            MirInstruction::CallIndirect { func_ptr, args, .. } => {
                let mut values = vec![*func_ptr];
                values.extend(args.iter().copied());
                values
            }
            MirInstruction::Phi { incoming, .. } => incoming.iter().map(|(_, v)| *v).collect(),
            MirInstruction::Select { condition, true_value, false_value, .. } => {
                vec![*condition, *true_value, *false_value]
            }
            MirInstruction::Const { .. } => vec![],
            MirInstruction::FPToUI { source, .. } => vec![*source],
            MirInstruction::UIToFP { source, .. } => vec![*source],
        }
    }
}

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MirBinaryOp {
    // Arithmetic
    Add, Sub, Mul, Div, Rem,
    // Bitwise
    And, Or, Xor,
    // Shifts
    Shl, LShr, AShr,
}

impl MirBinaryOp {
    /// Check if this operation is commutative
    pub fn is_commutative(&self) -> bool {
        matches!(self, MirBinaryOp::Add | MirBinaryOp::Mul | MirBinaryOp::And | MirBinaryOp::Or | MirBinaryOp::Xor)
    }

    /// Check if this is an arithmetic operation
    pub fn is_arithmetic(&self) -> bool {
        matches!(self, MirBinaryOp::Add | MirBinaryOp::Sub | MirBinaryOp::Mul | MirBinaryOp::Div | MirBinaryOp::Rem)
    }
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MirUnaryOp {
    /// Negate (arithmetic negation)
    Neg,
    /// Logical NOT
    Not,
    /// Bitwise NOT
    BitNot,
    /// Floating point negation
    FNeg,
}

/// Comparison operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MirCompareOp {
    // Equality
    Eq, Ne,
    // Ordering
    Lt, Le, Gt, Ge,
    // Floating point equality
    FEq, FNe,
    // Floating point ordering
    FLt, FLe, FGt, FGe,
}

/// Terminator instructions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MirTerminator {
    /// Unconditional branch
    Branch {
        target: BlockId,
    },
    /// Conditional branch
    CondBranch {
        condition: ValueId,
        true_target: BlockId,
        false_target: BlockId,
    },
    /// Switch statement
    Switch {
        value: ValueId,
        default_target: BlockId,
        cases: Vec<(i64, BlockId)>,
    },
    /// Return from function
    Return {
        value: Option<ValueId>,
    },
    /// Unreachable instruction
    Unreachable,
    /// Call function that never returns
    CallNoReturn {
        func_name: String,
        args: Vec<ValueId>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block() {
        let mut block = MirBlock::new(BlockId(0), "test_block".to_string());
        
        // Add a constant instruction
        let const_instr = MirInstruction::Const {
            result: ValueId(0),
            value: MirConstant::Int(42),
            ty: MirType::I32,
        };
        block.add_instruction(const_instr);
        
        // Add a branch terminator
        let branch = MirTerminator::Branch {
            target: BlockId(1),
        };
        block.set_terminator(branch);
        
        assert_eq!(block.instructions.len(), 1);
        assert!(matches!(block.terminator, MirTerminator::Branch { .. }));
    }

    #[test]
    fn test_function() {
        let mut func = MirFunction::new(
            FuncId(0),
            "test_function".to_string(),
            MirType::I32,
        );
        
        // Add parameters
        func.add_param("x".to_string(), MirType::I32);
        func.add_param("y".to_string(), MirType::I32);
        
        // Add entry block
        let entry_block = MirBlock::new(BlockId(0), "entry".to_string());
        func.add_block(entry_block);
        
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.return_type, MirType::I32);
    }

    #[test]
    fn test_binary_op_properties() {
        assert!(MirBinaryOp::Add.is_commutative());
        assert!(MirBinaryOp::Mul.is_commutative());
        assert!(!MirBinaryOp::Sub.is_commutative());
        
        assert!(MirBinaryOp::Add.is_arithmetic());
        assert!(!MirBinaryOp::And.is_arithmetic());
    }
}