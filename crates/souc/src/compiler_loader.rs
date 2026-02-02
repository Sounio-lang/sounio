//! Self-hosted Sounio Compiler Loader
//!
//! Loads and compiles the Sounio compiler (written in Sounio) and executes it via the bytecode VM.

use crate::vm::{Bytecode, BytecodeVM, Value};
use std::collections::HashMap;

/// Result type for compiler loader operations
pub type LoadResult<T> = Result<T, CompilerLoaderError>;

/// Errors that can occur during compiler loading and execution
#[derive(Debug, Clone, PartialEq)]
pub enum CompilerLoaderError {
    LoadError(String),
    ParseError(String),
    CompileError(String),
    ExecutionError(String),
    IoError(String),
}

impl std::fmt::Display for CompilerLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadError(msg) => write!(f, "Load error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::CompileError(msg) => write!(f, "Compile error: {}", msg),
            Self::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for CompilerLoaderError {}

/// Self-hosted Sounio compiler
///
/// Executes the Sounio compiler (which is written in Sounio) to compile source code.
/// The compiler modules are loaded from the filesystem or embedded in the binary.
pub struct SounioCompiler {
    /// Virtual machine for executing compiled compiler bytecode
    vm: BytecodeVM,
    /// Cache of compiled compiler modules
    module_cache: HashMap<String, Vec<Bytecode>>,
    /// Path to stdlib/compiler/ directory
    stdlib_path: String,
}

impl SounioCompiler {
    /// Creates a new self-hosted compiler instance
    ///
    /// Initializes the bytecode VM and loads the compiler modules from the filesystem.
    ///
    /// # Arguments
    /// * `stdlib_path` - Path to the stdlib/compiler/ directory containing .sio files
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if modules cannot be loaded or initialized
    pub fn new(stdlib_path: &str) -> LoadResult<Self> {
        tracing::info!("Initializing SounioCompiler from: {}", stdlib_path);

        let vm = BytecodeVM::new();
        let module_cache = HashMap::new();

        let compiler = Self {
            vm,
            module_cache,
            stdlib_path: stdlib_path.to_string(),
        };

        // In a full implementation, we would verify that the compiler modules exist
        // and are readable here
        tracing::info!("SounioCompiler initialized successfully");

        Ok(compiler)
    }

    /// Compiles Sounio source code to bytecode (deferred - requires full compiler integration)
    ///
    /// This is a placeholder that would integrate with the full compiler pipeline.
    /// In the full implementation, this would:
    /// 1. Tokenize the source code
    /// 2. Parse to AST
    /// 3. Type check
    /// 4. Generate bytecode
    ///
    /// # Arguments
    /// * `source` - The Sounio source code to compile
    ///
    /// # Errors
    /// Returns `CompilerLoaderError::CompileError` if compilation fails
    pub fn compile(&self, source: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling {} bytes of Sounio source", source.len());

        // This is where we would invoke the self-hosted compiler
        // For now, this is a placeholder that shows the intended interface

        // TODO: In a complete implementation:
        // 1. Load lexer module and call __sounio_lex(source) -> tokens
        // 2. Load parser module and call __sounio_parse(tokens) -> ast
        // 3. Load checker module and call __sounio_check(ast) -> checked_ast
        // 4. Load codegen module and call __sounio_codegen(checked_ast) -> bytecode

        tracing::warn!("Self-hosted compiler compilation not yet fully integrated");

        Err(CompilerLoaderError::CompileError(
            "Self-hosted compiler integration in progress".to_string(),
        ))
    }

    /// Compiles a Sounio source file to bytecode
    ///
    /// # Arguments
    /// * `path` - Path to the Sounio source file
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if file cannot be read or compilation fails
    pub fn compile_file(&self, path: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling file: {}", path);

        let source = std::fs::read_to_string(path)
            .map_err(|e| CompilerLoaderError::IoError(e.to_string()))?;

        self.compile(&source)
    }

    /// Loads a compiler module from the filesystem
    ///
    /// Looks for files in stdlib/compiler/ directory with the pattern:
    /// - `stdlib/compiler/{module}.sio` - single file module
    /// - `stdlib/compiler/{module}/mod.sio` - multi-file module
    ///
    /// # Arguments
    /// * `module_name` - Name of the module to load (e.g., "lexer", "parser")
    ///
    /// # Errors
    /// Returns `CompilerLoaderError::LoadError` if module cannot be found
    pub fn load_module(&self, module_name: &str) -> LoadResult<String> {
        tracing::debug!("Loading compiler module: {}", module_name);

        // Try single file first: stdlib/compiler/{module}.sio
        let single_path = format!("{}/{}.sio", self.stdlib_path, module_name);
        if std::path::Path::new(&single_path).exists() {
            let content = std::fs::read_to_string(&single_path)
                .map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;
            tracing::trace!("Loaded module {} from {}", module_name, single_path);
            return Ok(content);
        }

        // Try directory: stdlib/compiler/{module}/mod.sio
        let dir_path = format!("{}/{}/mod.sio", self.stdlib_path, module_name);
        if std::path::Path::new(&dir_path).exists() {
            let content = std::fs::read_to_string(&dir_path)
                .map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;
            tracing::trace!("Loaded module {} from {}", module_name, dir_path);
            return Ok(content);
        }

        Err(CompilerLoaderError::LoadError(format!(
            "Module '{}' not found in {}",
            module_name, self.stdlib_path
        )))
    }

    /// Lists all available compiler modules
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if the stdlib directory cannot be read
    pub fn list_modules(&self) -> LoadResult<Vec<String>> {
        tracing::debug!("Listing available modules in: {}", self.stdlib_path);

        let entries = std::fs::read_dir(&self.stdlib_path)
            .map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;

        let mut modules = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "sio") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    modules.push(name.to_string());
                }
            } else if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    modules.push(name.to_string());
                }
            }
        }

        tracing::info!("Found {} compiler modules", modules.len());

        Ok(modules)
    }

    /// Executes the self-hosted compiler on the given source
    ///
    /// This is the main entry point that would orchestrate the compilation process
    /// by calling the appropriate compiler modules in sequence.
    ///
    /// # Arguments
    /// * `source` - The Sounio source code to compile
    ///
    /// # Returns
    /// The bytecode result of compilation
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if any stage fails
    pub fn execute(&mut self, source: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Executing self-hosted compiler on {} bytes", source.len());

        // TODO: When fully integrated, this would:
        // 1. Load all compiler modules
        // 2. Execute lexer bytecode
        // 3. Execute parser bytecode on tokens
        // 4. Execute checker bytecode on AST
        // 5. Execute codegen bytecode on checked AST
        // 6. Return the generated bytecode

        self.compile(source)
    }
}

impl std::fmt::Debug for SounioCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SounioCompiler")
            .field("stdlib_path", &self.stdlib_path)
            .field("cached_modules", &self.module_cache.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = SounioCompiler::new("stdlib/compiler");
        assert!(compiler.is_ok());
    }

    #[test]
    fn test_module_listing() {
        let compiler = SounioCompiler::new("stdlib/compiler").unwrap();
        let modules = compiler.list_modules();
        assert!(modules.is_ok());
        let module_list = modules.unwrap();
        assert!(!module_list.is_empty(), "Should find compiler modules");
    }
}
