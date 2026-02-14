//! FFI bindings to the Poseidon C VM
//!
//! This module provides low-level unsafe FFI bindings to the C VM.
//! All public APIs are wrapped in safe Rust abstractions in lib.rs.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::c_int;

/// Maximum number of registers in the VM
pub const MAX_REGISTERS: usize = 1024;

/// Maximum call stack depth
pub const MAX_CALL_DEPTH: usize = 1024;

/// Maximum number of functions per module
pub const MAX_FUNCTIONS: usize = 64;

/// Maximum number of instructions per function
pub const MAX_INSTRUCTIONS: usize = 2048;

/// Maximum number of parameters per function
pub const MAX_PARAMS: usize = 64;

/// Size of name buffer
pub const NAME_BUF_SIZE: usize = 128;

/// Opcode enumeration (must match C)
#[repr(i8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Opcode {
    LoadImm = 0,
    LoadFloat = 1,
    LoadBool = 2,
    LoadString = 3,
    Copy = 4,
    BinOp = 5,
    UnaryOp = 6,
    Call = 7,
    Return = 8,
    Jump = 9,
    BranchTrue = 10,
    BranchFalse = 11,
    FieldGet = 12,
    FieldSet = 13,
    IndexGet = 14,
    IndexSet = 15,
    Alloc = 16,
    Label = 17,
    Nop = 18,
    Phi = 19,
}

/// Binary operation enumeration (must match C)
#[repr(i8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Mod = 4,
    Eq = 5,
    Ne = 6,
    Lt = 7,
    Le = 8,
    Gt = 9,
    Ge = 10,
    And = 11,
    Or = 12,
    BitAnd = 13,
    BitOr = 14,
    BitXor = 15,
    Shl = 16,
    Shr = 17,
}

/// Unary operation enumeration (must match C)
#[repr(i8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Neg = 0,
    Not = 1,
}

/// Name structure (must match C layout)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Name {
    pub buf: [i8; NAME_BUF_SIZE],
    pub len: i64,
}

/// Instruction structure (must match C layout, 128 bytes fixed)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Instruction {
    pub op: Opcode,
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
    pub arg_count: i64,
}

/// Function structure (must match C layout)
#[repr(C)]
pub struct Function {
    pub name: Name,
    pub instrs: *mut Instruction,
    pub instr_count: i64,
    pub reg_count: i64,
    pub label_count: i64,
    pub param_count: i64,
    pub param_regs: [i64; MAX_PARAMS],
}

/// Module structure (must match C layout)
#[repr(C)]
pub struct Module {
    pub functions: *mut Function,
    pub fn_count: i64,
    pub string_table: *mut Name,
    pub string_count: i64,
}

/// Call frame structure (must match C layout)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CallFrame {
    pub fn_id: i64,
    pub return_pc: i64,
    pub return_reg: i64,
}

/// VM state structure (must match C layout)
#[repr(C)]
pub struct VMState {
    pub regs: [i64; MAX_REGISTERS],
    pub pc: i64,
    pub call_stack: [CallFrame; MAX_CALL_DEPTH],
    pub call_depth: i64,
    pub current_fn: i64,
    pub halted: bool,
    pub exit_code: i64,
}

unsafe extern "C" {
    /// Initialize VM state with entry function
    pub fn vm_init(vm: *mut VMState, entry_fn: i64);

    /// Execute one instruction
    /// Returns 0 on success, -1 on error
    pub fn vm_step(vm: *mut VMState, module: *const Module) -> c_int;

    /// Run VM until halt or max steps
    /// Returns exit code, -2 on timeout
    pub fn vm_run(vm: *mut VMState, module: *const Module, max_steps: i64) -> i64;

    /// Find main function by name
    /// Returns function ID or -1 if not found
    pub fn vm_find_main_fn(module: *const Module) -> i64;

    /// Find label PC within a function
    pub fn vm_find_label_pc(func: *const Function, label_id: i64) -> i64;

    /// Load SOIR bytecode from memory buffer
    /// Returns NULL on failure
    pub fn load_soir_buffer(buf: *const u8, size: usize) -> *mut Module;

    /// Free loaded module
    pub fn free_module(module: *mut Module);
}
