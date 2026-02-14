//! Error types for the Poseidon VM wrapper

use thiserror::Error;

/// Result type for Poseidon VM operations
pub type Result<T> = std::result::Result<T, VmError>;

/// Errors that can occur during VM operations
#[derive(Debug, Error)]
pub enum VmError {
    #[error("Failed to load bytecode: {0}")]
    LoadError(String),

    #[error("Execution failed: {0}")]
    ExecutionError(String),

    #[error("Invalid register index: {0} (max {1})")]
    InvalidRegister(usize, usize),

    #[error("Stack overflow (max depth {0})")]
    StackOverflow(usize),

    #[error("Invalid bytecode: {0}")]
    InvalidBytecode(String),

    #[error("Execution timeout after {0} steps")]
    Timeout(i64),

    #[error("VM already halted with exit code {0}")]
    AlreadyHalted(i64),

    #[error("No main function found in module")]
    NoMainFunction,

    #[error("SOIR error: {0}")]
    SoirError(#[from] soir::SoirError),
}
