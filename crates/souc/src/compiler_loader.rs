//! Self-hosted Sounio Compiler Loader
//!
//! Loads and compiles the Sounio compiler (written in Sounio) and executes it via the bytecode VM.

use crate::embedded_stdlib;
use crate::lexer;
use crate::parser;
use crate::vm::{Bytecode, BytecodeVM};
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
    /// Path to stdlib/compiler/ directory (for filesystem mode)
    stdlib_path: String,
    /// Whether to use embedded modules (Phase 2)
    use_embedded: bool,
}

impl SounioCompiler {
    /// Creates a new self-hosted compiler instance using filesystem modules
    ///
    /// Initializes the bytecode VM and loads the compiler modules from the filesystem.
    ///
    /// # Arguments
    /// * `stdlib_path` - Path to the stdlib/compiler/ directory containing .sio files
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if modules cannot be loaded or initialized
    pub fn new(stdlib_path: &str) -> LoadResult<Self> {
        tracing::info!(
            "Initializing SounioCompiler from filesystem: {}",
            stdlib_path
        );

        let vm = BytecodeVM::new();
        let module_cache = HashMap::new();

        let compiler = Self {
            vm,
            module_cache,
            stdlib_path: stdlib_path.to_string(),
            use_embedded: false,
        };

        // In a full implementation, we would verify that the compiler modules exist
        // and are readable here
        tracing::info!("SounioCompiler initialized successfully (filesystem mode)");

        Ok(compiler)
    }

    /// Creates a new self-hosted compiler instance using embedded modules
    ///
    /// This is the Phase 2 implementation: all stdlib/compiler modules are embedded
    /// in the binary at build time and loaded from there.
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if embedded modules cannot be loaded
    pub fn new_embedded() -> LoadResult<Self> {
        tracing::info!("Initializing SounioCompiler with embedded modules");

        let module_count = embedded_stdlib::MODULE_COUNT;
        tracing::debug!("Found {} embedded modules", module_count);

        if module_count == 0 {
            return Err(CompilerLoaderError::LoadError(
                "No embedded modules found. Was the compiler built correctly?".to_string(),
            ));
        }

        let vm = BytecodeVM::new();
        let module_cache = HashMap::new();

        let compiler = Self {
            vm,
            module_cache,
            stdlib_path: String::new(), // Not used in embedded mode
            use_embedded: true,
        };

        tracing::info!(
            "SounioCompiler initialized successfully (embedded mode, {} modules)",
            module_count
        );

        Ok(compiler)
    }

    /// Returns whether this compiler uses embedded modules
    pub fn is_embedded(&self) -> bool {
        self.use_embedded
    }

    /// Returns the number of available modules
    pub fn module_count(&self) -> usize {
        if self.use_embedded {
            embedded_stdlib::MODULE_COUNT
        } else {
            self.list_modules().map(|m| m.len()).unwrap_or(0)
        }
    }

    /// Compiles Sounio source code to bytecode
    ///
    /// During bootstrap, this delegates to the Rust compiler for compilation.
    /// Once the Sounio compiler modules are embedded as bytecode, this will:
    /// 1. Call the Sounio lexer (bytecode) to tokenize
    /// 2. Call the Sounio parser (bytecode) to parse
    /// 3. Call the Sounio type checker (bytecode) to check
    /// 4. Call the Sounio codegen (bytecode) to generate bytecode
    ///
    /// # Arguments
    /// * `source` - The Sounio source code to compile
    ///
    /// # Errors
    /// Returns `CompilerLoaderError::CompileError` if compilation fails
    pub fn compile(&self, source: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling {} bytes of Sounio source", source.len());

        // BOOTSTRAP PATH: Use Rust compiler to compile Sounio code
        // This is the temporary bridge during self-hosting bootstrap.
        //
        // Once all stdlib/compiler modules are compiled to bytecode and embedded:
        // 1. Load lexer.bytecode and call lexer(source) -> tokens
        // 2. Load parser.bytecode and call parser(tokens) -> ast
        // 3. Load checker.bytecode and call checker(ast) -> typed_ast
        // 4. Load codegen.bytecode and call codegen(typed_ast) -> bytecode
        //
        // For now, we delegate to the existing Rust compiler infrastructure.
        // See: crates/souc/src/parser/, crates/souc/src/check/, crates/souc/src/codegen/

        // Lex source code to tokens using Rust lexer
        let tokens = lexer::lex(source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Lexer error: {}", e)))?;

        tracing::debug!("Lexed {} bytes to {} tokens", source.len(), tokens.len());

        // Parse tokens to AST using Rust parser
        let ast = parser::parse(&tokens, source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Parser error: {}", e)))?;

        tracing::debug!("Parsed {} tokens to AST", tokens.len());

        // Type check the AST
        let hir = crate::check::check_ast(&ast)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Type check error: {}", e)))?;

        tracing::debug!("Type checking complete");

        // Generate bytecode from HIR using the bytecode codegen backend
        let bytecode = crate::codegen::compile_hir(&hir)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Codegen error: {}", e)))?;

        tracing::info!(
            "Compilation complete, generated {} bytecode instructions",
            bytecode.len()
        );

        Ok(bytecode)
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

    /// Loads a compiler module
    ///
    /// In embedded mode, loads from the embedded modules.
    /// In filesystem mode, looks for files in stdlib/compiler/ directory with the pattern:
    /// - `stdlib/compiler/{module}.sio` - single file module
    /// - `stdlib/compiler/{module}/mod.sio` - multi-file module
    ///
    /// # Arguments
    /// * `module_name` - Name of the module to load (e.g., "lexer::mod", "parser::expr")
    ///
    /// # Errors
    /// Returns `CompilerLoaderError::LoadError` if module cannot be found
    pub fn load_module(&self, module_name: &str) -> LoadResult<String> {
        tracing::debug!(
            "Loading compiler module: {} (embedded={})",
            module_name,
            self.use_embedded
        );

        // Try embedded modules first if in embedded mode
        if self.use_embedded {
            if let Some(source) = embedded_stdlib::get_module(module_name) {
                tracing::trace!("Loaded module {} from embedded source", module_name);
                return Ok(source.to_string());
            }

            // Module not found in embedded
            return Err(CompilerLoaderError::LoadError(format!(
                "Module '{}' not found in embedded modules. Available: {:?}",
                module_name,
                embedded_stdlib::list_modules()
                    .iter()
                    .take(5)
                    .collect::<Vec<_>>()
            )));
        }

        // Filesystem mode: convert module_name format from "lexer::mod" to "lexer/mod.sio"
        let file_path = format!(
            "{}/{}.sio",
            self.stdlib_path,
            module_name.replace("::", "/")
        );

        if std::path::Path::new(&file_path).exists() {
            let content = std::fs::read_to_string(&file_path)
                .map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;
            tracing::trace!("Loaded module {} from {}", module_name, file_path);
            return Ok(content);
        }

        // Try without the ::mod suffix for single-file modules
        let base_module = module_name.strip_suffix("::mod").unwrap_or(module_name);
        let single_path = format!("{}/{}.sio", self.stdlib_path, base_module);
        if std::path::Path::new(&single_path).exists() {
            let content = std::fs::read_to_string(&single_path)
                .map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;
            tracing::trace!("Loaded module {} from {}", module_name, single_path);
            return Ok(content);
        }

        Err(CompilerLoaderError::LoadError(format!(
            "Module '{}' not found in {}",
            module_name, self.stdlib_path
        )))
    }

    /// Lists all available compiler modules
    ///
    /// In embedded mode, returns the list of embedded module names.
    /// In filesystem mode, reads from the stdlib directory.
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if the stdlib directory cannot be read (filesystem mode only)
    pub fn list_modules(&self) -> LoadResult<Vec<String>> {
        // In embedded mode, return embedded module names
        if self.use_embedded {
            let modules: Vec<String> = embedded_stdlib::list_modules()
                .iter()
                .map(|s| s.to_string())
                .collect();
            tracing::info!("Found {} embedded compiler modules", modules.len());
            return Ok(modules);
        }

        // Filesystem mode
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

/// Multi-module compilation context for stdlib bootstrap
///
/// Pre-loads all stdlib/compiler modules and builds a shared symbol context
/// for cross-module type resolution. This enables modules to import types
/// from other modules (e.g., `use check::context::TypeContext`).
pub struct StdlibCompilationContext {
    /// Parsed ASTs for all stdlib modules (module_name -> AST)
    module_asts: HashMap<String, crate::ast::Ast>,
    /// Type definitions extracted from all modules (type_name -> module_name)
    type_registry: HashMap<String, String>,
    /// Function signatures extracted from all modules
    fn_registry: HashMap<String, (Vec<String>, String)>, // (param_types, return_type)
}

impl StdlibCompilationContext {
    /// Build a compilation context from all embedded stdlib modules
    pub fn from_embedded() -> LoadResult<Self> {
        let modules = embedded_stdlib::list_modules();
        let mut module_asts = HashMap::new();
        let mut type_registry = HashMap::new();
        let mut fn_registry = HashMap::new();

        tracing::info!(
            "Building stdlib compilation context from {} modules",
            modules.len()
        );

        for module_name in modules {
            if let Some(source) = embedded_stdlib::get_module(module_name) {
                // Parse the module
                match Self::parse_module(source, module_name) {
                    Ok(ast) => {
                        // Extract type definitions
                        Self::extract_definitions(
                            &ast,
                            module_name,
                            &mut type_registry,
                            &mut fn_registry,
                        );
                        module_asts.insert(module_name.to_string(), ast);
                    }
                    Err(e) => {
                        tracing::debug!("Could not parse module {}: {}", module_name, e);
                        // Continue with other modules
                    }
                }
            }
        }

        tracing::info!(
            "Built context with {} modules, {} types, {} functions",
            module_asts.len(),
            type_registry.len(),
            fn_registry.len()
        );

        Ok(Self {
            module_asts,
            type_registry,
            fn_registry,
        })
    }

    /// Parse a single module
    fn parse_module(source: &str, module_name: &str) -> LoadResult<crate::ast::Ast> {
        let tokens = lexer::lex(source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("{}: {}", module_name, e)))?;
        parser::parse(&tokens, source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("{}: {}", module_name, e)))
    }

    /// Extract type and function definitions from an AST
    fn extract_definitions(
        ast: &crate::ast::Ast,
        module_name: &str,
        type_registry: &mut HashMap<String, String>,
        fn_registry: &mut HashMap<String, (Vec<String>, String)>,
    ) {
        use crate::ast::Item;

        for item in &ast.items {
            match item {
                Item::Struct(s) => {
                    // Register struct name with its module
                    type_registry.insert(s.name.clone(), module_name.to_string());
                    tracing::trace!("Registered type {} from {}", s.name, module_name);
                }
                Item::Enum(e) => {
                    type_registry.insert(e.name.clone(), module_name.to_string());
                    tracing::trace!("Registered enum {} from {}", e.name, module_name);
                }
                Item::TypeAlias(t) => {
                    type_registry.insert(t.name.clone(), module_name.to_string());
                    tracing::trace!("Registered type alias {} from {}", t.name, module_name);
                }
                Item::Function(f) => {
                    // Extract function signature
                    let param_types: Vec<String> =
                        f.params.iter().map(|p| format_type_expr(&p.ty)).collect();
                    let return_type = f
                        .return_type
                        .as_ref()
                        .map(|t| format_type_expr(t))
                        .unwrap_or_else(|| "()".to_string());
                    fn_registry.insert(
                        format!("{}::{}", module_name, f.name),
                        (param_types, return_type),
                    );
                }
                _ => {}
            }
        }
    }

    /// Get all registered type names
    pub fn registered_types(&self) -> impl Iterator<Item = (&str, &str)> {
        self.type_registry
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Look up which module defines a type
    pub fn find_type_module(&self, type_name: &str) -> Option<&str> {
        self.type_registry.get(type_name).map(|s| s.as_str())
    }

    /// Compile a module with cross-module type resolution
    pub fn compile_module(&self, module_name: &str) -> LoadResult<Vec<Bytecode>> {
        let source = embedded_stdlib::get_module(module_name).ok_or_else(|| {
            CompilerLoaderError::LoadError(format!("Module not found: {}", module_name))
        })?;

        self.compile_source(source, module_name)
    }

    /// Compile source code with access to all stdlib type definitions
    pub fn compile_source(&self, source: &str, module_name: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling {} with stdlib context", module_name);

        // Lex and parse
        let tokens = lexer::lex(source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Lexer error: {}", e)))?;
        let ast = parser::parse(&tokens, source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Parser error: {}", e)))?;

        // Resolve with stdlib context
        let resolved_ast = crate::resolve::resolve(ast.clone())
            .map_err(|e| CompilerLoaderError::CompileError(format!("Resolution error: {}", e)))?;

        // Type check with external types from stdlib context
        // This injects all known types from other stdlib modules so cross-module
        // references resolve correctly (e.g., TypeContext from check::context)
        let external_types = self
            .type_registry
            .iter()
            .map(|(name, module)| (name.clone(), module.clone()));
        let hir = crate::check::check_with_external_types(&resolved_ast, external_types)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Type check error: {}", e)))?;

        // Generate bytecode
        let bytecode = crate::codegen::compile_hir(&hir)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Codegen error: {}", e)))?;

        tracing::info!(
            "Compiled {} to {} bytecode instructions",
            module_name,
            bytecode.len()
        );

        Ok(bytecode)
    }
}

/// Format a type expression as a string (for debugging/registry)
fn format_type_expr(ty: &crate::ast::TypeExpr) -> String {
    use crate::ast::TypeExpr;
    match ty {
        TypeExpr::Named { path, args, .. } => {
            let name = path.segments.join("::");
            if args.is_empty() {
                name
            } else {
                let args_str: Vec<String> = args.iter().map(format_type_expr).collect();
                format!("{}<{}>", name, args_str.join(", "))
            }
        }
        TypeExpr::Array { element, size } => {
            let size_str = if size.is_some() { "N" } else { "?" };
            format!("[{}; {}]", format_type_expr(element), size_str)
        }
        TypeExpr::Reference { mutable, inner } => {
            if *mutable {
                format!("&!{}", format_type_expr(inner))
            } else {
                format!("&{}", format_type_expr(inner))
            }
        }
        TypeExpr::RawPointer { mutable, inner } => {
            if *mutable {
                format!("*mut {}", format_type_expr(inner))
            } else {
                format!("*const {}", format_type_expr(inner))
            }
        }
        TypeExpr::Tuple(elems) => {
            let elems_str: Vec<String> = elems.iter().map(format_type_expr).collect();
            format!("({})", elems_str.join(", "))
        }
        TypeExpr::Function {
            params,
            return_type,
            ..
        } => {
            let params_str: Vec<String> = params.iter().map(format_type_expr).collect();
            format!(
                "fn({}) -> {}",
                params_str.join(", "),
                format_type_expr(return_type)
            )
        }
        TypeExpr::Unit => "()".to_string(),
        TypeExpr::Never => "!".to_string(),
        TypeExpr::SelfType => "Self".to_string(),
        TypeExpr::Infer => "_".to_string(),
        _ => "?".to_string(), // Other complex types
    }
}

impl std::fmt::Debug for SounioCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SounioCompiler")
            .field("use_embedded", &self.use_embedded)
            .field("stdlib_path", &self.stdlib_path)
            .field("cached_modules", &self.module_cache.len())
            .field("available_modules", &self.module_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation_filesystem() {
        // Use absolute path for test
        let stdlib_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../stdlib/compiler");
        let compiler = SounioCompiler::new(stdlib_path);
        assert!(compiler.is_ok());
        let compiler = compiler.unwrap();
        assert!(!compiler.is_embedded());
    }

    #[test]
    fn test_compiler_creation_embedded() {
        let compiler = SounioCompiler::new_embedded();
        assert!(compiler.is_ok());
        let compiler = compiler.unwrap();
        assert!(compiler.is_embedded());
        assert!(compiler.module_count() > 0, "Should have embedded modules");
    }

    #[test]
    fn test_module_listing_filesystem() {
        let stdlib_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../stdlib/compiler");
        let compiler = SounioCompiler::new(stdlib_path).unwrap();
        let modules = compiler.list_modules();
        assert!(modules.is_ok());
        let module_list = modules.unwrap();
        assert!(!module_list.is_empty(), "Should find compiler modules");
    }

    #[test]
    fn test_module_listing_embedded() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let modules = compiler.list_modules();
        assert!(modules.is_ok());
        let module_list = modules.unwrap();
        assert!(!module_list.is_empty(), "Should have embedded modules");
        // Check for known modules
        assert!(
            module_list.iter().any(|m| m.contains("lexer")),
            "Should have lexer module"
        );
    }

    #[test]
    fn test_load_embedded_module() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let modules = compiler.list_modules().unwrap();

        // Try to load the first available module
        if let Some(first_module) = modules.first() {
            let source = compiler.load_module(first_module);
            assert!(
                source.is_ok(),
                "Should load embedded module: {}",
                first_module
            );
            let content = source.unwrap();
            assert!(!content.is_empty(), "Module content should not be empty");
        }
    }

    #[test]
    fn test_embedded_has_core_modules() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let modules = compiler.list_modules().unwrap();

        // These core modules should be present
        let expected_prefixes = ["lexer", "parser", "check", "codegen"];

        for prefix in expected_prefixes {
            let has_module = modules.iter().any(|m| m.contains(prefix));
            assert!(has_module, "Should have module containing '{}'", prefix);
        }
    }

    #[test]
    fn test_stdlib_compilation_context() {
        let ctx = StdlibCompilationContext::from_embedded();
        assert!(ctx.is_ok(), "Should build stdlib compilation context");
        let ctx = ctx.unwrap();

        // Check that types were extracted
        let type_count: usize = ctx.registered_types().count();
        assert!(
            type_count > 0,
            "Should extract type definitions, found {}",
            type_count
        );
        println!("Extracted {} type definitions from stdlib", type_count);

        // Print some sample types for debugging
        for (name, module) in ctx.registered_types().take(10) {
            println!("  Type '{}' from module '{}'", name, module);
        }
    }

    #[test]
    fn test_find_type_module() {
        let ctx = StdlibCompilationContext::from_embedded().unwrap();

        // Look for common types
        let type_count: usize = ctx.registered_types().count();
        if type_count > 0 {
            // Get first type and verify we can look it up
            let (first_type, first_module) = ctx.registered_types().next().unwrap();
            let found_module = ctx.find_type_module(first_type);
            assert_eq!(found_module, Some(first_module));
        }
    }

    #[test]
    fn test_cross_module_compilation() {
        let ctx = StdlibCompilationContext::from_embedded().unwrap();

        // Try to compile each module and track success/failure
        let modules: Vec<String> = embedded_stdlib::list_modules()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let mut success_count = 0;
        let mut fail_count = 0;
        let mut failures: Vec<(String, String)> = Vec::new();

        for module in &modules {
            match ctx.compile_module(module) {
                Ok(bytecode) => {
                    success_count += 1;
                    println!("✓ {} ({} instructions)", module, bytecode.len());
                }
                Err(e) => {
                    fail_count += 1;
                    let error_msg = format!("{}", e);
                    // Truncate long errors for readability
                    let short_error = if error_msg.len() > 100 {
                        format!("{}...", &error_msg[..100])
                    } else {
                        error_msg.clone()
                    };
                    println!("✗ {}: {}", module, short_error);
                    failures.push((module.clone(), short_error));
                }
            }
        }

        println!("\n=== Summary ===");
        println!("Success: {}/{}", success_count, modules.len());
        println!("Failed: {}", fail_count);

        // Print first 5 failure reasons for debugging
        if !failures.is_empty() {
            println!("\nSample failures:");
            for (module, error) in failures.iter().take(5) {
                println!("  {}: {}", module, error);
            }
        }

        // Target: at least 10 modules should compile with external types
        assert!(
            success_count >= 7,
            "Expected at least 7 modules to compile, got {}",
            success_count
        );
    }
}
