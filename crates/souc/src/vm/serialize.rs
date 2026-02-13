//! Bytecode Serialization
//!
//! Encode and decode bytecode for embedding in binaries and storage.
//! Uses a compact binary format optimized for fast loading.

use super::{Bytecode, Value};
use std::io::{self, Read, Write};

/// Magic bytes identifying a Sounio bytecode file
pub const BYTECODE_MAGIC: &[u8; 4] = b"SOBC";

/// Bytecode format version (for compatibility checking)
pub const BYTECODE_VERSION: u8 = 1;

/// Errors during serialization/deserialization
#[derive(Debug, Clone, PartialEq)]
pub enum SerializeError {
    InvalidMagic,
    UnsupportedVersion(u8),
    UnexpectedEof,
    InvalidOpcode(u8),
    InvalidValueType(u8),
    Utf8Error(String),
    IoError(String),
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "Invalid bytecode magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported bytecode version: {}", v),
            Self::UnexpectedEof => write!(f, "Unexpected end of bytecode"),
            Self::InvalidOpcode(op) => write!(f, "Invalid opcode: 0x{:02x}", op),
            Self::InvalidValueType(t) => write!(f, "Invalid value type: 0x{:02x}", t),
            Self::Utf8Error(e) => write!(f, "UTF-8 error: {}", e),
            Self::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for SerializeError {}

impl From<io::Error> for SerializeError {
    fn from(e: io::Error) -> Self {
        SerializeError::IoError(e.to_string())
    }
}

/// Opcodes for bytecode instructions
#[repr(u8)]
enum Opcode {
    Push = 0x01,
    Pop = 0x02,
    Dup = 0x03,
    Swap = 0x04,
    Add = 0x10,
    Sub = 0x11,
    Mul = 0x12,
    Div = 0x13,
    Mod = 0x14,
    Neg = 0x15,
    Eq = 0x20,
    Ne = 0x21,
    Lt = 0x22,
    Le = 0x23,
    Gt = 0x24,
    Ge = 0x25,
    And = 0x30,
    Or = 0x31,
    Not = 0x32,
    Shl = 0x33,
    Shr = 0x34,
    Xor = 0x35,
    Jump = 0x40,
    JumpIf = 0x41,
    Call = 0x42,
    Return = 0x43,
    Load = 0x50,
    Store = 0x51,
    Alloc = 0x52,
    StoreField = 0x53,
    StoreIndex = 0x54,
    GetField = 0x55,
    GetIndex = 0x56,
    IndexOp = 0x57,
    MakeVariant = 0x58,
    MakeRange = 0x59,
    CallExtern = 0x60,
}

impl Opcode {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Opcode::Push),
            0x02 => Some(Opcode::Pop),
            0x03 => Some(Opcode::Dup),
            0x04 => Some(Opcode::Swap),
            0x10 => Some(Opcode::Add),
            0x11 => Some(Opcode::Sub),
            0x12 => Some(Opcode::Mul),
            0x13 => Some(Opcode::Div),
            0x14 => Some(Opcode::Mod),
            0x15 => Some(Opcode::Neg),
            0x20 => Some(Opcode::Eq),
            0x21 => Some(Opcode::Ne),
            0x22 => Some(Opcode::Lt),
            0x23 => Some(Opcode::Le),
            0x24 => Some(Opcode::Gt),
            0x25 => Some(Opcode::Ge),
            0x30 => Some(Opcode::And),
            0x31 => Some(Opcode::Or),
            0x32 => Some(Opcode::Not),
            0x33 => Some(Opcode::Shl),
            0x34 => Some(Opcode::Shr),
            0x35 => Some(Opcode::Xor),
            0x40 => Some(Opcode::Jump),
            0x41 => Some(Opcode::JumpIf),
            0x42 => Some(Opcode::Call),
            0x43 => Some(Opcode::Return),
            0x50 => Some(Opcode::Load),
            0x51 => Some(Opcode::Store),
            0x52 => Some(Opcode::Alloc),
            0x53 => Some(Opcode::StoreField),
            0x54 => Some(Opcode::StoreIndex),
            0x55 => Some(Opcode::GetField),
            0x56 => Some(Opcode::GetIndex),
            0x57 => Some(Opcode::IndexOp),
            0x58 => Some(Opcode::MakeVariant),
            0x59 => Some(Opcode::MakeRange),
            0x60 => Some(Opcode::CallExtern),
            _ => None,
        }
    }
}

/// Value type tags for serialization
#[repr(u8)]
enum ValueTag {
    Unit = 0x00,
    Bool = 0x01,
    Int = 0x02,
    Float = 0x03,
    String = 0x04,
    Pointer = 0x05,
    List = 0x06,
    Struct = 0x07,
    SparseList = 0x08,
}

impl ValueTag {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x00 => Some(ValueTag::Unit),
            0x01 => Some(ValueTag::Bool),
            0x02 => Some(ValueTag::Int),
            0x03 => Some(ValueTag::Float),
            0x04 => Some(ValueTag::String),
            0x05 => Some(ValueTag::Pointer),
            0x06 => Some(ValueTag::List),
            0x07 => Some(ValueTag::Struct),
            0x08 => Some(ValueTag::SparseList),
            _ => None,
        }
    }
}

/// Bytecode serializer
pub struct BytecodeSerializer<W: Write> {
    writer: W,
}

impl<W: Write> BytecodeSerializer<W> {
    /// Create a new serializer
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Serialize a bytecode program
    pub fn serialize(&mut self, bytecode: &[Bytecode]) -> Result<(), SerializeError> {
        // Write header
        self.writer.write_all(BYTECODE_MAGIC)?;
        self.writer.write_all(&[BYTECODE_VERSION])?;

        // Write instruction count
        self.write_u32(bytecode.len() as u32)?;

        // Write each instruction
        for instr in bytecode {
            self.write_instruction(instr)?;
        }

        Ok(())
    }

    fn write_instruction(&mut self, instr: &Bytecode) -> Result<(), SerializeError> {
        match instr {
            Bytecode::Push(val) => {
                self.writer.write_all(&[Opcode::Push as u8])?;
                self.write_value(val)?;
            }
            Bytecode::Pop => self.writer.write_all(&[Opcode::Pop as u8])?,
            Bytecode::Dup => self.writer.write_all(&[Opcode::Dup as u8])?,
            Bytecode::Swap => self.writer.write_all(&[Opcode::Swap as u8])?,
            Bytecode::Add => self.writer.write_all(&[Opcode::Add as u8])?,
            Bytecode::Sub => self.writer.write_all(&[Opcode::Sub as u8])?,
            Bytecode::Mul => self.writer.write_all(&[Opcode::Mul as u8])?,
            Bytecode::Div => self.writer.write_all(&[Opcode::Div as u8])?,
            Bytecode::Mod => self.writer.write_all(&[Opcode::Mod as u8])?,
            Bytecode::Neg => self.writer.write_all(&[Opcode::Neg as u8])?,
            Bytecode::Eq => self.writer.write_all(&[Opcode::Eq as u8])?,
            Bytecode::Ne => self.writer.write_all(&[Opcode::Ne as u8])?,
            Bytecode::Lt => self.writer.write_all(&[Opcode::Lt as u8])?,
            Bytecode::Le => self.writer.write_all(&[Opcode::Le as u8])?,
            Bytecode::Gt => self.writer.write_all(&[Opcode::Gt as u8])?,
            Bytecode::Ge => self.writer.write_all(&[Opcode::Ge as u8])?,
            Bytecode::And => self.writer.write_all(&[Opcode::And as u8])?,
            Bytecode::Or => self.writer.write_all(&[Opcode::Or as u8])?,
            Bytecode::Not => self.writer.write_all(&[Opcode::Not as u8])?,
            Bytecode::Shl => self.writer.write_all(&[Opcode::Shl as u8])?,
            Bytecode::Shr => self.writer.write_all(&[Opcode::Shr as u8])?,
            Bytecode::Xor => self.writer.write_all(&[Opcode::Xor as u8])?,
            Bytecode::Jump(addr) => {
                self.writer.write_all(&[Opcode::Jump as u8])?;
                self.write_usize(*addr)?;
            }
            Bytecode::JumpIf(addr) => {
                self.writer.write_all(&[Opcode::JumpIf as u8])?;
                self.write_usize(*addr)?;
            }
            Bytecode::Call(fn_id) => {
                self.writer.write_all(&[Opcode::Call as u8])?;
                self.write_usize(*fn_id)?;
            }
            Bytecode::Return => self.writer.write_all(&[Opcode::Return as u8])?,
            Bytecode::Load(offset) => {
                self.writer.write_all(&[Opcode::Load as u8])?;
                self.write_usize(*offset)?;
            }
            Bytecode::Store(offset) => {
                self.writer.write_all(&[Opcode::Store as u8])?;
                self.write_usize(*offset)?;
            }
            Bytecode::Alloc(size) => {
                self.writer.write_all(&[Opcode::Alloc as u8])?;
                self.write_usize(*size)?;
            }
            Bytecode::CallExtern(name, arg_count) => {
                self.writer.write_all(&[Opcode::CallExtern as u8])?;
                self.write_string(name)?;
                self.write_i32(*arg_count)?;
            }
            Bytecode::StoreField(field) => {
                self.writer.write_all(&[Opcode::StoreField as u8])?;
                self.write_string(field)?;
            }
            Bytecode::StoreIndex => {
                self.writer.write_all(&[Opcode::StoreIndex as u8])?;
            }
            Bytecode::GetField(field) => {
                self.writer.write_all(&[Opcode::GetField as u8])?;
                self.write_string(field)?;
            }
            Bytecode::GetIndex(index) => {
                self.writer.write_all(&[Opcode::GetIndex as u8])?;
                self.write_usize(*index)?;
            }
            Bytecode::IndexOp => {
                self.writer.write_all(&[Opcode::IndexOp as u8])?;
            }
            Bytecode::MakeVariant {
                enum_name,
                variant,
                field_count,
            } => {
                self.writer.write_all(&[Opcode::MakeVariant as u8])?;
                self.write_string(enum_name)?;
                self.write_string(variant)?;
                self.write_usize(*field_count)?;
            }
            Bytecode::MakeRange { inclusive } => {
                self.writer.write_all(&[Opcode::MakeRange as u8])?;
                self.writer.write_all(&[if *inclusive { 1 } else { 0 }])?;
            }
        }
        Ok(())
    }

    fn write_value(&mut self, val: &Value) -> Result<(), SerializeError> {
        match val {
            Value::Unit => {
                self.writer.write_all(&[ValueTag::Unit as u8])?;
            }
            Value::Bool(b) => {
                self.writer.write_all(&[ValueTag::Bool as u8])?;
                self.writer.write_all(&[if *b { 1 } else { 0 }])?;
            }
            Value::Int(i) => {
                self.writer.write_all(&[ValueTag::Int as u8])?;
                self.write_i64(*i)?;
            }
            Value::Float(f) => {
                self.writer.write_all(&[ValueTag::Float as u8])?;
                self.write_f64(*f)?;
            }
            Value::String(s) => {
                self.writer.write_all(&[ValueTag::String as u8])?;
                self.write_string(s)?;
            }
            Value::Pointer(p) => {
                self.writer.write_all(&[ValueTag::Pointer as u8])?;
                self.write_usize(*p)?;
            }
            Value::List(items) => {
                self.writer.write_all(&[ValueTag::List as u8])?;
                self.write_u32(items.len() as u32)?;
                for item in items {
                    self.write_value(item)?;
                }
            }
            Value::Struct(fields) => {
                self.writer.write_all(&[ValueTag::Struct as u8])?;
                self.write_u32(fields.len() as u32)?;
                for (key, value) in fields {
                    self.write_string(key)?;
                    self.write_value(value)?;
                }
            }
            Value::SparseList {
                len,
                default,
                overrides,
            } => {
                self.writer.write_all(&[ValueTag::SparseList as u8])?;
                self.write_u32(*len as u32)?;
                self.write_value(default)?;
                self.write_u32(overrides.len() as u32)?;
                for (idx, value) in overrides {
                    self.write_usize(*idx)?;
                    self.write_value(value)?;
                }
            }
        }
        Ok(())
    }

    fn write_u32(&mut self, v: u32) -> Result<(), SerializeError> {
        self.writer.write_all(&v.to_le_bytes())?;
        Ok(())
    }

    fn write_i32(&mut self, v: i32) -> Result<(), SerializeError> {
        self.writer.write_all(&v.to_le_bytes())?;
        Ok(())
    }

    fn write_i64(&mut self, v: i64) -> Result<(), SerializeError> {
        self.writer.write_all(&v.to_le_bytes())?;
        Ok(())
    }

    fn write_f64(&mut self, v: f64) -> Result<(), SerializeError> {
        self.writer.write_all(&v.to_le_bytes())?;
        Ok(())
    }

    fn write_usize(&mut self, v: usize) -> Result<(), SerializeError> {
        // Use 64-bit for portability
        self.writer.write_all(&(v as u64).to_le_bytes())?;
        Ok(())
    }

    fn write_string(&mut self, s: &str) -> Result<(), SerializeError> {
        let bytes = s.as_bytes();
        self.write_u32(bytes.len() as u32)?;
        self.writer.write_all(bytes)?;
        Ok(())
    }
}

/// Bytecode deserializer
pub struct BytecodeDeserializer<R: Read> {
    reader: R,
}

impl<R: Read> BytecodeDeserializer<R> {
    /// Create a new deserializer
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    /// Deserialize a bytecode program
    pub fn deserialize(&mut self) -> Result<Vec<Bytecode>, SerializeError> {
        // Read and verify header
        let mut magic = [0u8; 4];
        self.reader
            .read_exact(&mut magic)
            .map_err(|_| SerializeError::UnexpectedEof)?;
        if &magic != BYTECODE_MAGIC {
            return Err(SerializeError::InvalidMagic);
        }

        let mut version = [0u8; 1];
        self.reader
            .read_exact(&mut version)
            .map_err(|_| SerializeError::UnexpectedEof)?;
        if version[0] != BYTECODE_VERSION {
            return Err(SerializeError::UnsupportedVersion(version[0]));
        }

        // Read instruction count
        let count = self.read_u32()? as usize;

        // Read instructions
        let mut bytecode = Vec::with_capacity(count);
        for _ in 0..count {
            bytecode.push(self.read_instruction()?);
        }

        Ok(bytecode)
    }

    fn read_instruction(&mut self) -> Result<Bytecode, SerializeError> {
        let mut opcode_byte = [0u8; 1];
        self.reader
            .read_exact(&mut opcode_byte)
            .map_err(|_| SerializeError::UnexpectedEof)?;

        let opcode =
            Opcode::from_u8(opcode_byte[0]).ok_or(SerializeError::InvalidOpcode(opcode_byte[0]))?;

        match opcode {
            Opcode::Push => Ok(Bytecode::Push(self.read_value()?)),
            Opcode::Pop => Ok(Bytecode::Pop),
            Opcode::Dup => Ok(Bytecode::Dup),
            Opcode::Swap => Ok(Bytecode::Swap),
            Opcode::Add => Ok(Bytecode::Add),
            Opcode::Sub => Ok(Bytecode::Sub),
            Opcode::Mul => Ok(Bytecode::Mul),
            Opcode::Div => Ok(Bytecode::Div),
            Opcode::Mod => Ok(Bytecode::Mod),
            Opcode::Neg => Ok(Bytecode::Neg),
            Opcode::Eq => Ok(Bytecode::Eq),
            Opcode::Ne => Ok(Bytecode::Ne),
            Opcode::Lt => Ok(Bytecode::Lt),
            Opcode::Le => Ok(Bytecode::Le),
            Opcode::Gt => Ok(Bytecode::Gt),
            Opcode::Ge => Ok(Bytecode::Ge),
            Opcode::And => Ok(Bytecode::And),
            Opcode::Or => Ok(Bytecode::Or),
            Opcode::Not => Ok(Bytecode::Not),
            Opcode::Shl => Ok(Bytecode::Shl),
            Opcode::Shr => Ok(Bytecode::Shr),
            Opcode::Xor => Ok(Bytecode::Xor),
            Opcode::Jump => Ok(Bytecode::Jump(self.read_usize()?)),
            Opcode::JumpIf => Ok(Bytecode::JumpIf(self.read_usize()?)),
            Opcode::Call => Ok(Bytecode::Call(self.read_usize()?)),
            Opcode::Return => Ok(Bytecode::Return),
            Opcode::Load => Ok(Bytecode::Load(self.read_usize()?)),
            Opcode::Store => Ok(Bytecode::Store(self.read_usize()?)),
            Opcode::Alloc => Ok(Bytecode::Alloc(self.read_usize()?)),
            Opcode::StoreField => {
                let name = self.read_string()?;
                Ok(Bytecode::StoreField(name))
            }
            Opcode::StoreIndex => Ok(Bytecode::StoreIndex),
            Opcode::GetField => {
                let name = self.read_string()?;
                Ok(Bytecode::GetField(name))
            }
            Opcode::GetIndex => Ok(Bytecode::GetIndex(self.read_usize()?)),
            Opcode::IndexOp => Ok(Bytecode::IndexOp),
            Opcode::MakeVariant => {
                let enum_name = self.read_string()?;
                let variant = self.read_string()?;
                let field_count = self.read_usize()?;
                Ok(Bytecode::MakeVariant {
                    enum_name,
                    variant,
                    field_count,
                })
            }
            Opcode::MakeRange => {
                let mut b = [0u8; 1];
                self.reader
                    .read_exact(&mut b)
                    .map_err(|_| SerializeError::UnexpectedEof)?;
                Ok(Bytecode::MakeRange {
                    inclusive: b[0] != 0,
                })
            }
            Opcode::CallExtern => {
                let name = self.read_string()?;
                let arg_count = self.read_i32()?;
                Ok(Bytecode::CallExtern(name, arg_count))
            }
        }
    }

    fn read_value(&mut self) -> Result<Value, SerializeError> {
        let mut tag_byte = [0u8; 1];
        self.reader
            .read_exact(&mut tag_byte)
            .map_err(|_| SerializeError::UnexpectedEof)?;

        let tag =
            ValueTag::from_u8(tag_byte[0]).ok_or(SerializeError::InvalidValueType(tag_byte[0]))?;

        match tag {
            ValueTag::Unit => Ok(Value::Unit),
            ValueTag::Bool => {
                let mut b = [0u8; 1];
                self.reader
                    .read_exact(&mut b)
                    .map_err(|_| SerializeError::UnexpectedEof)?;
                Ok(Value::Bool(b[0] != 0))
            }
            ValueTag::Int => Ok(Value::Int(self.read_i64()?)),
            ValueTag::Float => Ok(Value::Float(self.read_f64()?)),
            ValueTag::String => Ok(Value::String(self.read_string()?)),
            ValueTag::Pointer => Ok(Value::Pointer(self.read_usize()?)),
            ValueTag::List => {
                let len = self.read_u32()? as usize;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.read_value()?);
                }
                Ok(Value::List(items))
            }
            ValueTag::Struct => {
                let len = self.read_u32()? as usize;
                let mut fields: rustc_hash::FxHashMap<String, Value> = rustc_hash::FxHashMap::default();
                fields.reserve(len);
                for _ in 0..len {
                    let key = self.read_string()?;
                    let value = self.read_value()?;
                    fields.insert(key, value);
                }
                Ok(Value::Struct(fields))
            }
            ValueTag::SparseList => {
                let len = self.read_u32()? as usize;
                let default = Box::new(self.read_value()?);
                let overrides_len = self.read_u32()? as usize;
                let mut overrides: rustc_hash::FxHashMap<usize, Value> =
                    rustc_hash::FxHashMap::default();
                overrides.reserve(overrides_len);
                for _ in 0..overrides_len {
                    let idx = self.read_usize()?;
                    let value = self.read_value()?;
                    overrides.insert(idx, value);
                }
                Ok(Value::SparseList {
                    len,
                    default,
                    overrides,
                })
            }
        }
    }

    fn read_u32(&mut self) -> Result<u32, SerializeError> {
        let mut buf = [0u8; 4];
        self.reader
            .read_exact(&mut buf)
            .map_err(|_| SerializeError::UnexpectedEof)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32, SerializeError> {
        let mut buf = [0u8; 4];
        self.reader
            .read_exact(&mut buf)
            .map_err(|_| SerializeError::UnexpectedEof)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, SerializeError> {
        let mut buf = [0u8; 8];
        self.reader
            .read_exact(&mut buf)
            .map_err(|_| SerializeError::UnexpectedEof)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, SerializeError> {
        let mut buf = [0u8; 8];
        self.reader
            .read_exact(&mut buf)
            .map_err(|_| SerializeError::UnexpectedEof)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_usize(&mut self) -> Result<usize, SerializeError> {
        let mut buf = [0u8; 8];
        self.reader
            .read_exact(&mut buf)
            .map_err(|_| SerializeError::UnexpectedEof)?;
        Ok(u64::from_le_bytes(buf) as usize)
    }

    fn read_string(&mut self) -> Result<String, SerializeError> {
        let len = self.read_u32()? as usize;
        let mut buf = vec![0u8; len];
        self.reader
            .read_exact(&mut buf)
            .map_err(|_| SerializeError::UnexpectedEof)?;
        String::from_utf8(buf).map_err(|e| SerializeError::Utf8Error(e.to_string()))
    }
}

/// Convenience function to serialize bytecode to bytes
pub fn serialize(bytecode: &[Bytecode]) -> Result<Vec<u8>, SerializeError> {
    let mut buf = Vec::new();
    let mut serializer = BytecodeSerializer::new(&mut buf);
    serializer.serialize(bytecode)?;
    Ok(buf)
}

/// Convenience function to deserialize bytecode from bytes
pub fn deserialize(bytes: &[u8]) -> Result<Vec<Bytecode>, SerializeError> {
    let mut deserializer = BytecodeDeserializer::new(bytes);
    deserializer.deserialize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_simple() {
        let bytecode = vec![
            Bytecode::Push(Value::Int(42)),
            Bytecode::Push(Value::Int(10)),
            Bytecode::Add,
            Bytecode::Return,
        ];

        let serialized = serialize(&bytecode).expect("serialize failed");
        let deserialized = deserialize(&serialized).expect("deserialize failed");

        assert_eq!(bytecode, deserialized);
    }

    #[test]
    fn test_roundtrip_all_instructions() {
        let bytecode = vec![
            Bytecode::Push(Value::Unit),
            Bytecode::Push(Value::Bool(true)),
            Bytecode::Push(Value::Int(123)),
            Bytecode::Push(Value::Float(3.14)),
            Bytecode::Push(Value::String("hello".to_string())),
            Bytecode::Push(Value::Pointer(0x1000)),
            Bytecode::Push(Value::List(vec![Value::Int(1), Value::Int(2)])),
            Bytecode::Pop,
            Bytecode::Dup,
            Bytecode::Swap,
            Bytecode::Add,
            Bytecode::Sub,
            Bytecode::Mul,
            Bytecode::Div,
            Bytecode::Mod,
            Bytecode::Neg,
            Bytecode::Eq,
            Bytecode::Ne,
            Bytecode::Lt,
            Bytecode::Le,
            Bytecode::Gt,
            Bytecode::Ge,
            Bytecode::And,
            Bytecode::Or,
            Bytecode::Not,
            Bytecode::Shl,
            Bytecode::Shr,
            Bytecode::Xor,
            Bytecode::Jump(100),
            Bytecode::JumpIf(200),
            Bytecode::Call(42),
            Bytecode::Return,
            Bytecode::Load(8),
            Bytecode::Store(16),
            Bytecode::Alloc(64),
            Bytecode::CallExtern("__sounio_println".to_string(), 1),
        ];

        let serialized = serialize(&bytecode).expect("serialize failed");
        let deserialized = deserialize(&serialized).expect("deserialize failed");

        assert_eq!(bytecode, deserialized);
    }

    #[test]
    fn test_invalid_magic() {
        let bad_bytes = b"BADC\x01\x00\x00\x00\x00";
        let result = deserialize(bad_bytes);
        assert!(matches!(result, Err(SerializeError::InvalidMagic)));
    }

    #[test]
    fn test_unsupported_version() {
        let bad_bytes = b"SOBC\x99\x00\x00\x00\x00";
        let result = deserialize(bad_bytes);
        assert!(matches!(
            result,
            Err(SerializeError::UnsupportedVersion(0x99))
        ));
    }

    #[test]
    fn test_empty_program() {
        let bytecode: Vec<Bytecode> = vec![];
        let serialized = serialize(&bytecode).expect("serialize failed");
        let deserialized = deserialize(&serialized).expect("deserialize failed");
        assert!(deserialized.is_empty());
    }
}
