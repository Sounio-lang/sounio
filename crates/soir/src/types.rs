//! SOIR type definitions
//!
//! This module defines the core data structures for the Sounio IR format.
//! These types closely mirror the self-hosted IR definitions but provide
//! idiomatic Rust representations with proper ownership semantics.

use crate::{IR_MAX_FUNCS, IR_MAX_INSTRS, IR_MAX_PARAMS, IR_MAX_STRINGS};
use crate::{IR_INVALID_FN, IR_INVALID_LABEL, IR_INVALID_REG};

/// Fixed-size string buffer matching Sounio's Name type (128 bytes + length)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Name {
    /// Buffer containing UTF-8 bytes
    pub buf: [u8; 128],
    /// Length of valid data in buffer
    pub len: usize,
}

impl Name {
    /// Create an empty name
    pub fn new() -> Self {
        Self {
            buf: [0; 128],
            len: 0,
        }
    }

    /// Create name from byte slice
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut name = Self::new();
        let copy_len = bytes.len().min(128);
        name.buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
        name.len = copy_len;
        name
    }

    /// Get the name as a string slice
    pub fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(&self.buf[..self.len])
    }

    /// Get the name as bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    /// Check if the name is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for Name {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Self::from_bytes(s.as_bytes())
    }
}

impl From<String> for Name {
    fn from(s: String) -> Self {
        Self::from_bytes(s.as_bytes())
    }
}

/// IR opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IrOpcode {
    IrLoadImm = 0,
    IrLoadFloat = 1,
    IrLoadBool = 2,
    IrLoadString = 3,
    IrCopy = 4,
    IrBinOp = 5,
    IrUnaryOp = 6,
    IrCall = 7,
    IrReturn = 8,
    IrJump = 9,
    IrBranchTrue = 10,
    IrBranchFalse = 11,
    IrFieldGet = 12,
    IrFieldSet = 13,
    IrIndexGet = 14,
    IrIndexSet = 15,
    IrAlloc = 16,
    IrLabel = 17,
    IrNop = 18,
    IrPhi = 19,
}

impl IrOpcode {
    /// Convert from u8 tag
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::IrLoadImm),
            1 => Some(Self::IrLoadFloat),
            2 => Some(Self::IrLoadBool),
            3 => Some(Self::IrLoadString),
            4 => Some(Self::IrCopy),
            5 => Some(Self::IrBinOp),
            6 => Some(Self::IrUnaryOp),
            7 => Some(Self::IrCall),
            8 => Some(Self::IrReturn),
            9 => Some(Self::IrJump),
            10 => Some(Self::IrBranchTrue),
            11 => Some(Self::IrBranchFalse),
            12 => Some(Self::IrFieldGet),
            13 => Some(Self::IrFieldSet),
            14 => Some(Self::IrIndexGet),
            15 => Some(Self::IrIndexSet),
            16 => Some(Self::IrAlloc),
            17 => Some(Self::IrLabel),
            18 => Some(Self::IrNop),
            19 => Some(Self::IrPhi),
            _ => None,
        }
    }

    /// Convert to u8 tag
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BinaryOp {
    OpAdd = 0,
    OpSub = 1,
    OpMul = 2,
    OpDiv = 3,
    OpRem = 4,
    OpEq = 5,
    OpNe = 6,
    OpLt = 7,
    OpLe = 8,
    OpGt = 9,
    OpGe = 10,
    OpAnd = 11,
    OpOr = 12,
    OpBitAnd = 13,
    OpBitOr = 14,
    OpBitXor = 15,
    OpShl = 16,
    OpShr = 17,
    OpConcat = 18,
    OpRange = 19,
    OpRangeInclusive = 20,
}

impl BinaryOp {
    /// Convert from u8 tag
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::OpAdd),
            1 => Some(Self::OpSub),
            2 => Some(Self::OpMul),
            3 => Some(Self::OpDiv),
            4 => Some(Self::OpRem),
            5 => Some(Self::OpEq),
            6 => Some(Self::OpNe),
            7 => Some(Self::OpLt),
            8 => Some(Self::OpLe),
            9 => Some(Self::OpGt),
            10 => Some(Self::OpGe),
            11 => Some(Self::OpAnd),
            12 => Some(Self::OpOr),
            13 => Some(Self::OpBitAnd),
            14 => Some(Self::OpBitOr),
            15 => Some(Self::OpBitXor),
            16 => Some(Self::OpShl),
            17 => Some(Self::OpShr),
            18 => Some(Self::OpConcat),
            19 => Some(Self::OpRange),
            20 => Some(Self::OpRangeInclusive),
            _ => None,
        }
    }

    /// Convert to u8 tag
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

impl Default for BinaryOp {
    fn default() -> Self {
        Self::OpAdd
    }
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UnaryOp {
    OpNeg = 0,
    OpNot = 1,
    OpRef = 2,
    OpRefMut = 3,
    OpDeref = 4,
}

impl UnaryOp {
    /// Convert from u8 tag
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::OpNeg),
            1 => Some(Self::OpNot),
            2 => Some(Self::OpRef),
            3 => Some(Self::OpRefMut),
            4 => Some(Self::OpDeref),
            _ => None,
        }
    }

    /// Convert to u8 tag
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

impl Default for UnaryOp {
    fn default() -> Self {
        Self::OpNeg
    }
}

/// Linked list of register IDs for call arguments
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrRegList {
    pub head: i64,
    pub tail: Option<Box<IrRegList>>,
}

impl IrRegList {
    /// Create a single-element list
    pub fn single(reg: i64) -> Self {
        Self {
            head: reg,
            tail: None,
        }
    }

    /// Append a register to the end of the list
    pub fn append(mut self, reg: i64) -> Self {
        match self.tail {
            Some(tail) => {
                self.tail = Some(Box::new(tail.append(reg)));
                self
            }
            None => Self {
                head: self.head,
                tail: Some(Box::new(Self::single(reg))),
            },
        }
    }

    /// Count the number of registers in the list
    pub fn count(&self) -> i64 {
        1 + self.tail.as_ref().map_or(0, |t| t.count())
    }

    /// Get the first two registers (for compatibility with src1/src2)
    pub fn first_two(&self) -> (i64, i64) {
        let a0 = self.head;
        let a1 = self.tail.as_ref().map_or(IR_INVALID_REG, |t| t.head);
        (a0, a1)
    }
}

/// IR instruction (fixed-size 128-byte encoding)
#[derive(Debug, Clone, PartialEq)]
pub struct IrInstr {
    pub op: IrOpcode,
    pub dst: i64,
    pub src1: i64,
    pub src2: i64,
    pub imm_i64: i64,
    pub imm_f64: f64,
    pub label_id: i64,
    pub fn_id: i64,
    pub field_idx: i64,
    pub bin_op: BinaryOp,
    pub un_op: UnaryOp,
    pub name: Name,
    pub call_args: Option<Box<IrRegList>>,
    pub arg_count: i64,
}

impl IrInstr {
    /// Create a NOP instruction
    pub fn nop() -> Self {
        Self {
            op: IrOpcode::IrNop,
            dst: IR_INVALID_REG,
            src1: IR_INVALID_REG,
            src2: IR_INVALID_REG,
            imm_i64: 0,
            imm_f64: 0.0,
            label_id: IR_INVALID_LABEL,
            fn_id: IR_INVALID_FN,
            field_idx: -1,
            bin_op: BinaryOp::OpAdd,
            un_op: UnaryOp::OpNeg,
            name: Name::new(),
            call_args: None,
            arg_count: 0,
        }
    }
}

impl Default for IrInstr {
    fn default() -> Self {
        Self::nop()
    }
}

/// IR function
#[derive(Debug, Clone, PartialEq)]
pub struct IrFunction {
    pub name: Name,
    pub instrs: Vec<IrInstr>,
    pub instr_count: i64,
    pub reg_count: i64,
    pub label_count: i64,
    pub param_count: i64,
    pub param_regs: [i64; IR_MAX_PARAMS],
}

impl IrFunction {
    /// Create an empty function
    pub fn empty() -> Self {
        Self {
            name: Name::new(),
            instrs: vec![IrInstr::nop(); IR_MAX_INSTRS],
            instr_count: 0,
            reg_count: 0,
            label_count: 0,
            param_count: 0,
            param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
        }
    }
}

impl Default for IrFunction {
    fn default() -> Self {
        Self::empty()
    }
}

/// SOIR module containing functions and string table
#[derive(Debug, Clone, PartialEq)]
pub struct SoirModule {
    pub functions: Vec<IrFunction>,
    pub fn_count: i64,
    pub string_table: Vec<Name>,
    pub string_count: i64,
}

impl SoirModule {
    /// Create an empty module
    pub fn empty() -> Self {
        Self {
            functions: vec![IrFunction::empty(); IR_MAX_FUNCS],
            fn_count: 0,
            string_table: vec![Name::new(); IR_MAX_STRINGS],
            string_count: 0,
        }
    }

    /// Get function by index
    pub fn get_function(&self, idx: usize) -> Option<&IrFunction> {
        if idx < self.fn_count as usize {
            Some(&self.functions[idx])
        } else {
            None
        }
    }

    /// Get string by index
    pub fn get_string(&self, idx: usize) -> Option<&Name> {
        if idx < self.string_count as usize {
            Some(&self.string_table[idx])
        } else {
            None
        }
    }
}

impl Default for SoirModule {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_from_str() {
        let name = Name::from("test");
        assert_eq!(name.len, 4);
        assert_eq!(name.as_str().unwrap(), "test");
    }

    #[test]
    fn test_name_truncation() {
        let long_str = "a".repeat(200);
        let name = Name::from(long_str.as_str());
        assert_eq!(name.len, 128);
    }

    #[test]
    fn test_opcode_roundtrip() {
        for i in 0..=19 {
            let op = IrOpcode::from_u8(i).unwrap();
            assert_eq!(op.to_u8(), i);
        }
    }

    #[test]
    fn test_binary_op_roundtrip() {
        for i in 0..=20 {
            let op = BinaryOp::from_u8(i).unwrap();
            assert_eq!(op.to_u8(), i);
        }
    }

    #[test]
    fn test_reg_list_count() {
        let list = IrRegList::single(1)
            .append(2)
            .append(3);
        assert_eq!(list.count(), 3);
    }

    #[test]
    fn test_empty_module() {
        let module = SoirModule::empty();
        assert_eq!(module.fn_count, 0);
        assert_eq!(module.string_count, 0);
    }
}
