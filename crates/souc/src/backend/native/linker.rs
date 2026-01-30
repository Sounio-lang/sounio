//! # Sounio Compiler: Linker Integration
//!
//! This module provides integration with system linkers (ld, lld) to produce
//! executable files and shared libraries from ELF object files.
//!
//! ## Workflow
//!
//! ```text
//! SIR → metrics → alloc → asm/emit → ELF .o → [linker.rs] → executable/.so
//! ```
//!
//! ## Supported Output Formats
//!
//! - Shared library (.so) - default for FFI
//! - Executable - with runtime support
//! - Object file (.o) - for external linking
//!
//! ## Usage
//!
//! ```rust,ignore
//! let config = LinkerConfig::default();
//! let linker = Linker::new(config)?;
//!
//! // Link object file to shared library
//! linker.link(&["module.o"], "output.so", LinkMode::SharedLib)?;
//!
//! // Link with runtime for executable
//! linker.link_executable(&["main.o"], "myprogram")?;
//! ```
//!
//! ## Author
//!
//! Demetrios Chiuratto Agourakis <demetrios@chiuratto.ai>

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Linker configuration
#[derive(Debug, Clone)]
pub struct LinkerConfig {
    /// Path to linker executable (ld, lld, gold, mold)
    pub linker_path: PathBuf,
    /// Linker flavor
    pub flavor: LinkerFlavor,
    /// Target triple
    pub target: String,
    /// Library search paths (-L)
    pub library_paths: Vec<PathBuf>,
    /// Libraries to link (-l)
    pub libraries: Vec<String>,
    /// Additional linker flags
    pub extra_flags: Vec<String>,
    /// Generate position-independent code
    pub pic: bool,
    /// Strip debug symbols
    pub strip: bool,
    /// Optimization level for LTO
    pub lto: bool,
    /// Verbose output
    pub verbose: bool,
    /// Use system C runtime
    pub use_crt: bool,
    /// Path to Demetrios runtime library
    pub runtime_path: Option<PathBuf>,
}

impl Default for LinkerConfig {
    fn default() -> Self {
        Self {
            linker_path: find_linker().unwrap_or_else(|| PathBuf::from("ld")),
            flavor: LinkerFlavor::Gnu,
            target: "x86_64-unknown-linux-gnu".to_string(),
            library_paths: vec![
                PathBuf::from("/lib/x86_64-linux-gnu"),
                PathBuf::from("/usr/lib/x86_64-linux-gnu"),
                PathBuf::from("/usr/lib"),
            ],
            libraries: Vec::new(),
            extra_flags: Vec::new(),
            pic: true,
            strip: false,
            lto: false,
            verbose: false,
            use_crt: true,
            runtime_path: None,
        }
    }
}

impl LinkerConfig {
    /// Configuration for shared library output
    pub fn shared_lib() -> Self {
        Self {
            pic: true,
            use_crt: false,
            ..Default::default()
        }
    }

    /// Configuration for standalone executable
    pub fn executable() -> Self {
        Self {
            pic: false,
            use_crt: true,
            ..Default::default()
        }
    }

    /// Configuration for FFI library (Julia, Python, etc.)
    pub fn ffi_library() -> Self {
        Self {
            pic: true,
            use_crt: false,
            libraries: vec!["m".to_string()], // Link math library
            ..Default::default()
        }
    }

    /// Add a library to link
    pub fn with_library(mut self, lib: &str) -> Self {
        self.libraries.push(lib.to_string());
        self
    }

    /// Add a library search path
    pub fn with_library_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.library_paths.push(path.into());
        self
    }

    /// Set runtime path
    pub fn with_runtime(mut self, path: impl Into<PathBuf>) -> Self {
        self.runtime_path = Some(path.into());
        self
    }
}

/// Linker flavor (affects command-line arguments)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkerFlavor {
    /// GNU ld
    Gnu,
    /// LLVM lld
    Lld,
    /// GNU gold
    Gold,
    /// Modern mold linker
    Mold,
    /// macOS ld64
    Darwin,
}

impl LinkerFlavor {
    /// Get linker executable name
    pub fn executable_name(&self) -> &'static str {
        match self {
            LinkerFlavor::Gnu => "ld",
            LinkerFlavor::Lld => "ld.lld",
            LinkerFlavor::Gold => "ld.gold",
            LinkerFlavor::Mold => "mold",
            LinkerFlavor::Darwin => "ld",
        }
    }
}

/// Link mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkMode {
    /// Produce executable
    Executable,
    /// Produce shared library (.so / .dylib)
    SharedLib,
    /// Produce static library (.a)
    StaticLib,
    /// Produce relocatable object (partial link)
    Relocatable,
}

// ============================================================================
// LINKER ERROR
// ============================================================================

/// Linker error type
#[derive(Debug)]
pub enum LinkerError {
    /// Linker not found
    LinkerNotFound(String),
    /// Linker execution failed
    ExecutionFailed { command: String, stderr: String },
    /// Input file not found
    InputNotFound(PathBuf),
    /// Output write failed
    OutputFailed(io::Error),
    /// Invalid configuration
    InvalidConfig(String),
    /// Missing runtime
    MissingRuntime,
}

impl std::fmt::Display for LinkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkerError::LinkerNotFound(path) => {
                write!(f, "Linker not found: {}", path)
            }
            LinkerError::ExecutionFailed { command, stderr } => {
                write!(
                    f,
                    "Linker failed:\n  Command: {}\n  Error: {}",
                    command, stderr
                )
            }
            LinkerError::InputNotFound(path) => {
                write!(f, "Input file not found: {}", path.display())
            }
            LinkerError::OutputFailed(e) => {
                write!(f, "Failed to write output: {}", e)
            }
            LinkerError::InvalidConfig(msg) => {
                write!(f, "Invalid linker configuration: {}", msg)
            }
            LinkerError::MissingRuntime => {
                write!(f, "Demetrios runtime library not found")
            }
        }
    }
}

impl std::error::Error for LinkerError {}

// ============================================================================
// LINKER
// ============================================================================

/// System linker wrapper
pub struct Linker {
    config: LinkerConfig,
}

impl Linker {
    /// Create a new linker with configuration
    pub fn new(config: LinkerConfig) -> Result<Self, LinkerError> {
        // Verify linker exists
        if !config.linker_path.exists() && which(&config.linker_path).is_none() {
            return Err(LinkerError::LinkerNotFound(
                config.linker_path.display().to_string(),
            ));
        }

        Ok(Self { config })
    }

    /// Create with default configuration
    pub fn default_linker() -> Result<Self, LinkerError> {
        Self::new(LinkerConfig::default())
    }

    /// Link object files into output
    pub fn link(
        &self,
        inputs: &[impl AsRef<Path>],
        output: impl AsRef<Path>,
        mode: LinkMode,
    ) -> Result<LinkResult, LinkerError> {
        // Verify inputs exist
        for input in inputs {
            if !input.as_ref().exists() {
                return Err(LinkerError::InputNotFound(input.as_ref().to_path_buf()));
            }
        }

        // Build command
        let mut cmd = Command::new(&self.config.linker_path);

        // Add mode-specific flags
        match mode {
            LinkMode::Executable => {
                self.add_executable_flags(&mut cmd)?;
            }
            LinkMode::SharedLib => {
                self.add_shared_lib_flags(&mut cmd);
            }
            LinkMode::StaticLib => {
                return self.create_static_lib(inputs, output);
            }
            LinkMode::Relocatable => {
                cmd.arg("-r");
            }
        }

        // Output file
        cmd.arg("-o").arg(output.as_ref());

        // Input files
        for input in inputs {
            cmd.arg(input.as_ref());
        }

        // Library paths
        for path in &self.config.library_paths {
            cmd.arg("-L").arg(path);
        }

        // Libraries
        for lib in &self.config.libraries {
            cmd.arg(format!("-l{}", lib));
        }

        // Extra flags
        for flag in &self.config.extra_flags {
            cmd.arg(flag);
        }

        // Strip if requested
        if self.config.strip {
            cmd.arg("-s");
        }

        // Verbose
        if self.config.verbose {
            cmd.arg("-v");
        }

        // Execute
        let cmd_str = format!("{:?}", cmd);
        let output_result = cmd.output().map_err(|e| LinkerError::ExecutionFailed {
            command: cmd_str.clone(),
            stderr: e.to_string(),
        })?;

        if !output_result.status.success() {
            return Err(LinkerError::ExecutionFailed {
                command: cmd_str,
                stderr: String::from_utf8_lossy(&output_result.stderr).to_string(),
            });
        }

        Ok(LinkResult {
            output_path: output.as_ref().to_path_buf(),
            mode,
            warnings: parse_warnings(&output_result.stderr),
        })
    }

    /// Link into executable with Demetrios runtime
    pub fn link_executable(
        &self,
        inputs: &[impl AsRef<Path>],
        output: impl AsRef<Path>,
    ) -> Result<LinkResult, LinkerError> {
        // For executable, we may need runtime support
        let mut all_inputs: Vec<PathBuf> =
            inputs.iter().map(|p| p.as_ref().to_path_buf()).collect();

        // Add runtime if specified
        if let Some(runtime) = &self.config.runtime_path {
            if !runtime.exists() {
                return Err(LinkerError::MissingRuntime);
            }
            all_inputs.push(runtime.clone());
        }

        let input_refs: Vec<&Path> = all_inputs.iter().map(|p| p.as_path()).collect();
        self.link(&input_refs, output, LinkMode::Executable)
    }

    /// Link into shared library
    pub fn link_shared(
        &self,
        inputs: &[impl AsRef<Path>],
        output: impl AsRef<Path>,
    ) -> Result<LinkResult, LinkerError> {
        self.link(inputs, output, LinkMode::SharedLib)
    }

    /// Create static library using ar
    fn create_static_lib(
        &self,
        inputs: &[impl AsRef<Path>],
        output: impl AsRef<Path>,
    ) -> Result<LinkResult, LinkerError> {
        let mut cmd = Command::new("ar");
        cmd.arg("rcs");
        cmd.arg(output.as_ref());

        for input in inputs {
            cmd.arg(input.as_ref());
        }

        let cmd_str = format!("{:?}", cmd);
        let result = cmd.output().map_err(|e| LinkerError::ExecutionFailed {
            command: cmd_str.clone(),
            stderr: e.to_string(),
        })?;

        if !result.status.success() {
            return Err(LinkerError::ExecutionFailed {
                command: cmd_str,
                stderr: String::from_utf8_lossy(&result.stderr).to_string(),
            });
        }

        Ok(LinkResult {
            output_path: output.as_ref().to_path_buf(),
            mode: LinkMode::StaticLib,
            warnings: vec![],
        })
    }

    /// Add flags for executable output
    fn add_executable_flags(&self, cmd: &mut Command) -> Result<(), LinkerError> {
        if self.config.use_crt {
            // Dynamic linker
            cmd.arg("-dynamic-linker");
            cmd.arg("/lib64/ld-linux-x86-64.so.2");

            // C runtime files
            let crt_paths = [
                "/usr/lib/x86_64-linux-gnu",
                "/lib/x86_64-linux-gnu",
                "/usr/lib",
            ];

            let crt1 = find_file(&crt_paths, "crt1.o");
            let crti = find_file(&crt_paths, "crti.o");
            let crtn = find_file(&crt_paths, "crtn.o");

            if let Some(crt1) = crt1 {
                cmd.arg(&crt1);
            }
            if let Some(crti) = crti {
                cmd.arg(&crti);
            }

            // Link against libc
            cmd.arg("-lc");

            if let Some(crtn) = crtn {
                cmd.arg(&crtn);
            }
        }

        // PIC/PIE
        if self.config.pic {
            cmd.arg("-pie");
        }

        Ok(())
    }

    /// Add flags for shared library output
    fn add_shared_lib_flags(&self, cmd: &mut Command) {
        cmd.arg("-shared");

        if self.config.pic {
            // Already handled by -shared, but explicit
        }

        // Allow undefined symbols (will be resolved at runtime)
        cmd.arg("--allow-shlib-undefined");
    }
}

/// Link operation result
#[derive(Debug)]
pub struct LinkResult {
    /// Path to output file
    pub output_path: PathBuf,
    /// Link mode used
    pub mode: LinkMode,
    /// Any warnings from linker
    pub warnings: Vec<String>,
}

impl LinkResult {
    /// Check if there were warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

// ============================================================================
// COMPLETE PIPELINE
// ============================================================================

/// Complete compilation pipeline from code segment to output file
pub struct NativePipeline {
    /// Working directory for intermediate files
    work_dir: PathBuf,
    /// Linker configuration
    linker_config: LinkerConfig,
    /// Keep intermediate files
    keep_intermediates: bool,
    /// Verbose output
    verbose: bool,
}

impl NativePipeline {
    /// Create a new pipeline
    pub fn new(work_dir: impl Into<PathBuf>) -> Self {
        Self {
            work_dir: work_dir.into(),
            linker_config: LinkerConfig::default(),
            keep_intermediates: false,
            verbose: false,
        }
    }

    /// Set linker configuration
    pub fn with_linker_config(mut self, config: LinkerConfig) -> Self {
        self.linker_config = config;
        self
    }

    /// Keep intermediate files for debugging
    pub fn keep_intermediates(mut self) -> Self {
        self.keep_intermediates = true;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self.linker_config.verbose = verbose;
        self
    }

    /// Compile machine code to object file
    pub fn emit_object(
        &self,
        code: &[u8],
        symbols: &[SymbolDef],
        output_name: &str,
    ) -> Result<PathBuf, LinkerError> {
        use super::elf::{ElfConfig, ElfWriter, SymbolBinding};

        let mut elf = ElfWriter::new(ElfConfig::default());

        // Add text section with code
        elf.add_text_section(code);

        // Add symbols
        for sym in symbols {
            let binding = if sym.is_global {
                SymbolBinding::Global
            } else {
                SymbolBinding::Local
            };
            elf.add_function(&sym.name, sym.offset, sym.size, sym.is_global);
        }

        // Generate ELF
        let elf_bytes = elf.finish().map_err(|e| LinkerError::OutputFailed(e))?;

        // Write to file
        let obj_path = self.work_dir.join(format!("{}.o", output_name));
        fs::create_dir_all(&self.work_dir).map_err(|e| LinkerError::OutputFailed(e))?;
        fs::write(&obj_path, &elf_bytes).map_err(|e| LinkerError::OutputFailed(e))?;

        if self.verbose {
            println!("Generated object file: {}", obj_path.display());
        }

        Ok(obj_path)
    }

    /// Assemble assembly file to object file
    pub fn assemble(
        &self,
        asm_path: impl AsRef<Path>,
        output_name: &str,
    ) -> Result<PathBuf, LinkerError> {
        let obj_path = self.work_dir.join(format!("{}.o", output_name));

        let mut cmd = Command::new("as");
        cmd.arg("-o").arg(&obj_path);
        cmd.arg(asm_path.as_ref());

        if self.verbose {
            println!("Assembling: {:?}", cmd);
        }

        let result = cmd.output().map_err(|e| LinkerError::ExecutionFailed {
            command: format!("{:?}", cmd),
            stderr: e.to_string(),
        })?;

        if !result.status.success() {
            return Err(LinkerError::ExecutionFailed {
                command: format!("{:?}", cmd),
                stderr: String::from_utf8_lossy(&result.stderr).to_string(),
            });
        }

        Ok(obj_path)
    }

    /// Link object files to final output
    pub fn link(
        &self,
        objects: &[impl AsRef<Path>],
        output: impl AsRef<Path>,
        mode: LinkMode,
    ) -> Result<LinkResult, LinkerError> {
        let linker = Linker::new(self.linker_config.clone())?;
        linker.link(objects, output, mode)
    }

    /// Complete pipeline: code → object → linked output
    pub fn compile_and_link(
        &self,
        code: &[u8],
        symbols: &[SymbolDef],
        output: impl AsRef<Path>,
        mode: LinkMode,
    ) -> Result<PipelineResult, LinkerError> {
        // Step 1: Emit object file
        let obj_path = self.emit_object(code, symbols, "module")?;

        // Step 2: Link
        let link_result = self.link(&[&obj_path], &output, mode)?;

        // Cleanup if requested
        if !self.keep_intermediates {
            let _ = fs::remove_file(&obj_path);
        }

        Ok(PipelineResult {
            output_path: link_result.output_path,
            object_path: if self.keep_intermediates {
                Some(obj_path)
            } else {
                None
            },
            mode,
            warnings: link_result.warnings,
        })
    }

    /// Pipeline from assembly: asm → object → linked output
    pub fn assemble_and_link(
        &self,
        asm_path: impl AsRef<Path>,
        output: impl AsRef<Path>,
        mode: LinkMode,
    ) -> Result<PipelineResult, LinkerError> {
        // Step 1: Assemble
        let obj_path = self.assemble(&asm_path, "module")?;

        // Step 2: Link
        let link_result = self.link(&[&obj_path], &output, mode)?;

        // Cleanup if requested
        if !self.keep_intermediates {
            let _ = fs::remove_file(&obj_path);
        }

        Ok(PipelineResult {
            output_path: link_result.output_path,
            object_path: if self.keep_intermediates {
                Some(obj_path)
            } else {
                None
            },
            mode,
            warnings: link_result.warnings,
        })
    }
}

/// Symbol definition for code emission
#[derive(Debug, Clone)]
pub struct SymbolDef {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub is_global: bool,
}

/// Pipeline execution result
#[derive(Debug)]
pub struct PipelineResult {
    /// Final output path
    pub output_path: PathBuf,
    /// Object file path (if kept)
    pub object_path: Option<PathBuf>,
    /// Link mode
    pub mode: LinkMode,
    /// Warnings
    pub warnings: Vec<String>,
}

// ============================================================================
// UTILITIES
// ============================================================================

/// Find system linker
fn find_linker() -> Option<PathBuf> {
    // Try in order of preference
    // Note: mold crashes on our ELF output, so prefer ld/lld for now
    let candidates = ["ld", "ld.lld", "ld.gold"];

    for candidate in candidates {
        if let Some(path) = which(candidate) {
            return Some(path);
        }
    }
    None
}

/// Find executable in PATH
fn which(name: impl AsRef<OsStr>) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;

    for dir in std::env::split_paths(&path_var) {
        let full_path = dir.join(name.as_ref());
        if full_path.exists() && full_path.is_file() {
            return Some(full_path);
        }
    }
    None
}

/// Find file in search paths
fn find_file(paths: &[&str], filename: &str) -> Option<PathBuf> {
    for path in paths {
        let full = PathBuf::from(path).join(filename);
        if full.exists() {
            return Some(full);
        }
    }
    None
}

/// Parse warnings from linker stderr
fn parse_warnings(stderr: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(stderr);
    text.lines()
        .filter(|line| line.contains("warning:"))
        .map(|s| s.to_string())
        .collect()
}

// ============================================================================
// MODULE EXPORTS
// ============================================================================

pub mod prelude {
    pub use super::{
        LinkMode, LinkResult, Linker, LinkerConfig, LinkerError, LinkerFlavor, NativePipeline,
        PipelineResult, SymbolDef,
    };
}
