//! LLVM Backend for Sounio
//!
//! Translates HLIR to LLVM IR for ahead-of-time compilation.
//!
//! # Architecture
//!
//! ```text
//! HLIR → LLVMCodegen → LLVM IR → LLVM Passes → Object File → Linker → Executable
//! ```
//!
//! # Features
//!
//! - Full type mapping from D types to LLVM types
//! - SSA-based code generation from HLIR
//! - Multiple optimization levels (O0-O3, Os, Oz)
//! - Debug information generation (DWARF)
//! - Cross-compilation support via target triples
//! - Native executable linking
//!
//! # Example
//!
//! ```ignore
//! use sounio::codegen::llvm::{LLVMCodegen, OptLevel};
//! use inkwell::context::Context;
//!
//! let context = Context::create();
//! let mut codegen = LLVMCodegen::new(&context, "my_module", OptLevel::O2, false);
//! let module = codegen.compile(&hlir);
//! ```
//!
//! # References
//!
//! - LLVM Language Reference: <https://llvm.org/docs/LangRef.html>
//! - Inkwell Documentation: <https://thedan64.github.io/inkwell/>

#[cfg(feature = "llvm")]
pub mod codegen;
#[cfg(feature = "llvm")]
pub mod debug;
#[cfg(feature = "llvm")]
pub mod gpu;
#[cfg(feature = "llvm")]
pub mod linker;
#[cfg(feature = "llvm")]
pub mod passes;
#[cfg(feature = "llvm")]
pub mod target;
#[cfg(feature = "llvm")]
pub mod types;

#[cfg(feature = "llvm")]
pub use codegen::{LLVMCodegen, OptLevel};
#[cfg(feature = "llvm")]
pub use gpu::LlvmGpuCodegen;
#[cfg(feature = "llvm")]
pub use linker::{LinkError, Linker};
#[cfg(feature = "llvm")]
pub use target::{GpuTargetConfig, TargetConfig};

// Stub implementations when LLVM is not available
#[cfg(not(feature = "llvm"))]
pub mod stub {
    use crate::hlir::HlirModule;
    use std::path::Path;

    /// Optimization level (stub)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum OptLevel {
        O0,
        O1,
        O2,
        O3,
        Os,
        Oz,
    }

    /// LLVM codegen stub when feature is disabled
    pub struct LLVMCodegen;

    impl LLVMCodegen {
        pub fn compile(_hlir: &HlirModule) -> Result<(), String> {
            Err("LLVM backend not enabled. Rebuild with: cargo build --features llvm".to_string())
        }
    }

    /// Linker stub
    pub struct Linker;

    impl Linker {
        pub fn link(_objects: &[&Path], _output: &Path) -> Result<(), String> {
            Err("LLVM backend not enabled. Rebuild with: cargo build --features llvm".to_string())
        }
    }
}

#[cfg(not(feature = "llvm"))]
pub use stub::*;
