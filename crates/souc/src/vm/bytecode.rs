//! Bytecode instruction set and value types

use std::collections::HashMap;
use std::fmt;

/// Runtime values in the VM
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Pointer(usize),
    List(Vec<Value>),
    /// Struct/record with named fields
    Struct(HashMap<String, Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "{}", s),
            Value::Pointer(p) => write!(f, "0x{:x}", p),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Struct(fields) => {
                write!(f, "{{ ")?;
                let mut first = true;
                for (name, value) in fields {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, value)?;
                    first = false;
                }
                write!(f, " }}")
            }
        }
    }
}

/// Bytecode instructions
#[derive(Debug, Clone, PartialEq)]
pub enum Bytecode {
    // Stack operations
    Push(Value),
    Pop,
    Dup,
    Swap,

    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,

    // Comparison operations
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical operations
    And,
    Or,
    Not,

    // Bitwise shift operations
    Shl,
    Shr,

    // Control flow
    Jump(usize),
    JumpIf(usize),
    Call(usize),
    Return,

    // Memory operations
    Load(usize),
    Store(usize),
    Alloc(usize),

    // Field and index operations
    /// Store value to a field: target is on stack, field name provided
    StoreField(String),
    /// Store value to an array/index: target and index on stack
    StoreIndex,
    /// Get field from struct on stack
    GetField(String),
    /// Get element from tuple/array at constant index
    GetIndex(usize),
    /// Get element from array using runtime index (array and index on stack)
    IndexOp,

    // Composite type construction
    /// Make an enum variant with given fields on stack
    MakeVariant {
        enum_name: String,
        variant: String,
        field_count: usize,
    },
    /// Make a range from start..end (or start..=end if inclusive) on stack
    MakeRange {
        inclusive: bool,
    },

    // FFI calls: (function_name, argument_count)
    CallExtern(String, i32),
}

impl fmt::Display for Bytecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bytecode::Push(v) => write!(f, "Push({})", v),
            Bytecode::Pop => write!(f, "Pop"),
            Bytecode::Dup => write!(f, "Dup"),
            Bytecode::Swap => write!(f, "Swap"),
            Bytecode::Add => write!(f, "Add"),
            Bytecode::Sub => write!(f, "Sub"),
            Bytecode::Mul => write!(f, "Mul"),
            Bytecode::Div => write!(f, "Div"),
            Bytecode::Mod => write!(f, "Mod"),
            Bytecode::Neg => write!(f, "Neg"),
            Bytecode::Eq => write!(f, "Eq"),
            Bytecode::Ne => write!(f, "Ne"),
            Bytecode::Lt => write!(f, "Lt"),
            Bytecode::Le => write!(f, "Le"),
            Bytecode::Gt => write!(f, "Gt"),
            Bytecode::Ge => write!(f, "Ge"),
            Bytecode::And => write!(f, "And"),
            Bytecode::Or => write!(f, "Or"),
            Bytecode::Not => write!(f, "Not"),
            Bytecode::Shl => write!(f, "Shl"),
            Bytecode::Shr => write!(f, "Shr"),
            Bytecode::Jump(a) => write!(f, "Jump({})", a),
            Bytecode::JumpIf(a) => write!(f, "JumpIf({})", a),
            Bytecode::Call(id) => write!(f, "Call({})", id),
            Bytecode::Return => write!(f, "Return"),
            Bytecode::Load(off) => write!(f, "Load({})", off),
            Bytecode::Store(off) => write!(f, "Store({})", off),
            Bytecode::Alloc(size) => write!(f, "Alloc({})", size),
            Bytecode::StoreField(name) => write!(f, "StoreField({})", name),
            Bytecode::StoreIndex => write!(f, "StoreIndex"),
            Bytecode::GetField(name) => write!(f, "GetField({})", name),
            Bytecode::GetIndex(idx) => write!(f, "GetIndex({})", idx),
            Bytecode::IndexOp => write!(f, "IndexOp"),
            Bytecode::MakeVariant {
                enum_name,
                variant,
                field_count,
            } => write!(
                f,
                "MakeVariant({}::{}, fields={})",
                enum_name, variant, field_count
            ),
            Bytecode::MakeRange { inclusive } => write!(f, "MakeRange(inclusive={})", inclusive),
            Bytecode::CallExtern(name, args) => write!(f, "CallExtern({}, args={})", name, args),
        }
    }
}
