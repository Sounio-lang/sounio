//! Sounio Programming Language Compiler
//!
//! A novel L0 systems + scientific programming language with:
//! - Full algebraic effects with handlers
//! - Linear and affine types for safe resource management
//! - Units of measure with compile-time dimensional analysis
//! - Refinement types with SMT-backed verification
//! - GPU-native computation
//!
//! # Architecture
//!
//! ```text
//! Source → Lexer → Parser → AST → Type Checker → HIR → HLIR → Codegen
//! ```
//!
//! # Example
//!
//! ```d
//! module example
//!
//! let dose: mg = 500.0
//! let volume: mL = 10.0
//! let concentration: mg/mL = dose / volume
//!
//! fn simulate(params: PKParams) -> Vec<f64> with Prob, Alloc {
//!     let eta = sample(Normal(0.0, 0.3))
//!     // ...
//! }
//! ```

#![allow(dead_code)]
#![allow(unused_variables)]

pub mod analyze;
pub mod ast;
pub mod bio;
pub mod build;
pub mod causal;
pub mod check;
pub mod codegen;
pub mod common;
pub mod dependent;
pub mod diagnostic;
pub mod diagnostics;
#[cfg(feature = "distributed")]
pub mod distributed;
pub mod doc;
pub mod effects;
pub mod epistemic;
pub mod fmt;
pub mod geometry;
pub mod hir;
pub mod hlir;
pub mod interop;
pub mod interp;
pub mod layout;
pub mod lexer;
pub mod linear;
pub mod lint;
pub mod llm;
pub mod locality;
#[cfg(feature = "lsp")]
pub mod lsp;
pub mod macro_system;
pub mod mlir;
pub mod module_loader;
pub mod ontology;
pub mod optimizer;
pub mod ownership;
pub mod parser;
pub mod pkg;
pub mod profiling;
pub mod quantum;
pub mod refactor;
pub mod refinement;
pub mod repl;
pub mod resolve;
pub mod rl;
pub mod runtime;
pub mod semantic_diagnostics;
pub mod smt;
pub mod sourcemap;
pub mod target;
pub mod temporal;
pub mod test;
pub mod typeck;
pub mod types;
pub mod units;
pub mod watch;

// Re-export diagnostics for convenience
pub use diagnostics::{CompileError, Reporter, SourceFile};

// Re-exports for convenience
pub use ast::Ast;
pub use hir::Hir;
pub use hlir::HlirModule;
pub use types::Type;

/// Compiler version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compile source code to an executable (native code via LLVM/Cranelift)
pub fn compile(source: &str) -> miette::Result<Vec<u8>> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens, source)?;
    let hir = check::check(&ast)?;
    let hlir = hlir::lower(&hir);

    // Use Cranelift backend for JIT compilation
    #[cfg(feature = "jit")]
    {
        let code = codegen::cranelift::compile(&hlir)
            .map_err(|e| miette::miette!("Cranelift codegen failed: {}", e))?;
        return Ok(code);
    }

    // Use LLVM backend for AOT compilation
    #[cfg(feature = "llvm")]
    {
        use inkwell::context::Context;
        let context = Context::create();
        let mut codegen =
            codegen::llvm::LLVMCodegen::new(&context, "main", codegen::llvm::OptLevel::O2, false);
        let _module = codegen.compile(&hlir);
        // TODO: Actually emit machine code
        return Ok(vec![]);
    }

    #[cfg(not(any(feature = "jit", feature = "llvm")))]
    Err(miette::miette!(
        "No native backend enabled. Use --features jit or --features llvm"
    ))
}

/// Compile source code to PTX for GPU execution
pub fn compile_to_gpu(source: &str, sm_version: (u32, u32)) -> miette::Result<String> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens, source)?;
    let hir = check::check(&ast)?;
    let hlir = hlir::lower(&hir);

    // Lower to GPU IR and generate PTX
    let ptx = codegen::gpu::compile_to_ptx(&hlir, sm_version);
    Ok(ptx)
}

/// Compile source code to PTX with epistemic state tracking
///
/// This is the revolutionary feature of Sounio: tracking uncertainty
/// through GPU computation using shadow registers.
pub fn compile_to_gpu_epistemic(source: &str, sm_version: (u32, u32)) -> miette::Result<String> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens, source)?;
    let hir = check::check(&ast)?;
    let hlir = hlir::lower(&hir);

    // Lower to GPU IR with epistemic tracking enabled
    let ptx = codegen::gpu::compile_to_ptx_epistemic(&hlir, sm_version, true);
    Ok(ptx)
}

/// Type-check source code without compiling
pub fn typecheck(source: &str) -> miette::Result<Hir> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens, source)?;
    check::check(&ast)
}

/// Parse source code to AST
pub fn parse(source: &str) -> miette::Result<Ast> {
    let tokens = lexer::lex(source)?;
    parser::parse(&tokens, source)
}

/// Interpret source code directly
pub fn interpret(source: &str) -> miette::Result<interp::Value> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens, source)?;
    let hir = check::check(&ast)?;
    let mut interpreter = interp::Interpreter::new();
    interpreter.interpret(&hir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
