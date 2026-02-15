//! # SOIR - Sounio IR Binary Format
//!
//! SOIR (Sounio Intermediate Representation) is a binary serialization format
//! for the Sounio compiler's intermediate representation. It provides:
//!
//! - **Deterministic serialization** - Byte-identical output for equivalent IR
//! - **Normalization** - Canonical form for semantic equivalence checking
//! - **Versioned format** - Magic bytes and version number for compatibility
//! - **Compact encoding** - Fixed-size instruction format for efficient processing
//!
//! ## Format Overview
//!
//! ```text
//! SOIR v1 Binary Format:
//! ──────────────────────
//! Header (8 bytes):
//!   - Magic: "SOIR" (0x534F4952)
//!   - Version: 1 (1 byte)
//!   - Reserved: 0x00 (3 bytes)
//!
//! Body:
//!   - fn_count: i64
//!   - functions[]: IrFunction[]
//!   - string_count: i64
//!   - strings[]: Name[]
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use soir::{SoirModule, serialize, deserialize, normalize};
//!
//! // Serialize IR module to bytes
//! let bytes = serialize(&ir_module)?;
//!
//! // Deserialize from bytes
//! let module = deserialize(&bytes)?;
//!
//! // Normalize for deterministic comparison
//! let normalized = normalize(&module);
//!
//! // Compare two normalized modules
//! assert!(normalized == normalized2);
//! ```

use std::io;
use thiserror::Error;

pub mod compare;
pub mod deserialize;
pub mod normalize;
pub mod serialize;
pub mod types;

pub use compare::compare;
pub use deserialize::deserialize;
pub use normalize::normalize;
pub use serialize::serialize;
pub use types::{BinaryOp, IrFunction, IrInstr, IrOpcode, IrRegList, Name, SoirModule, UnaryOp};

/// SOIR format version
pub const SOIR_VERSION: u8 = 1;

/// SOIR magic bytes "SOIR"
pub const SOIR_MAGIC: [u8; 4] = [0x53, 0x4F, 0x49, 0x52];

/// Maximum module size (128KB)
pub const SOIR_MAX_SIZE: usize = 128 * 1024;

/// Maximum number of functions per module
pub const IR_MAX_FUNCS: usize = 64;

/// Maximum number of string table entries
pub const IR_MAX_STRINGS: usize = 256;

/// Maximum number of instructions per function
pub const IR_MAX_INSTRS: usize = 2048;

/// Maximum number of parameters per function
pub const IR_MAX_PARAMS: usize = 64;

/// Invalid register sentinel value
pub const IR_INVALID_REG: i64 = -1;

/// Invalid label sentinel value
pub const IR_INVALID_LABEL: i64 = -1;

/// Invalid function ID sentinel value
pub const IR_INVALID_FN: i64 = -1;

/// Errors that can occur during SOIR operations
#[derive(Error, Debug)]
pub enum SoirError {
    #[error("Invalid magic bytes: expected SOIR, got {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("Unsupported version: {0} (expected {1})")]
    UnsupportedVersion(u8, u8),

    #[error("Module too large: {0} bytes (max {1})")]
    ModuleTooLarge(usize, usize),

    #[error("Invalid opcode: {0}")]
    InvalidOpcode(u8),

    #[error("Invalid binary op: {0}")]
    InvalidBinaryOp(u8),

    #[error("Invalid unary op: {0}")]
    InvalidUnaryOp(u8),

    #[error("Unexpected end of data at offset {0}")]
    UnexpectedEof(usize),

    #[error("Function count overflow: {0} (max {1})")]
    FunctionCountOverflow(i64, usize),

    #[error("String count overflow: {0} (max {1})")]
    StringCountOverflow(i64, usize),

    #[error("Instruction count overflow: {0} (max {1})")]
    InstrCountOverflow(i64, usize),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, SoirError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_bytes() {
        assert_eq!(&SOIR_MAGIC, b"SOIR");
    }

    #[test]
    fn test_version() {
        assert_eq!(SOIR_VERSION, 1);
    }
}
