//! Linker Integration System
//!
//! This module provides platform-specific linker integration, including linker
//! discovery, invocation, and linker script generation.

use super::spec::{LinkerFlavor, LtoMode, OperatingSystem, RelocationModel, TargetSpec};
use super::sysroot::Sysroot;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use thiserror::Error;

/// Errors that can occur during linking.
#[derive(Debug, Error)]
pub enum LinkerError {
    #[error("Linker not found: {0}")]
    NotFound(String),

    #[error("Linker execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Linker returned error: {0}")]
    LinkFailed(String),

    #[error("Unsupported linker flavor for target: {flavor:?} on {target}")]
    UnsupportedFlavor {
        flavor: LinkerFlavor,
        target: String,
    },

    #[error("Missing required library: {0}")]
    MissingLibrary(String),

    #[error("Invalid linker script: {0}")]
    InvalidScript(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for linker operations.
pub type LinkerResult<T> = Result<T, LinkerError>;

/// Output artifact type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputType {
    /// Executable binary
    Executable,
    /// Dynamic/shared library
    DynamicLib,
    /// Static library (archive)
    StaticLib,
    /// Relocatable object file
    Object,
    /// LLVM bitcode
    Bitcode,
    /// Assembly
    Assembly,
    /// LLVM IR
    LlvmIr,
}

impl OutputType {
    /// Get the default file extension for this output type.
    pub fn extension<'a>(&self, spec: &'a TargetSpec) -> &'a str {
        match self {
            Self::Executable => spec.exe_suffix(),
            Self::DynamicLib => spec.dylib_suffix(),
            Self::StaticLib => spec.staticlib_suffix(),
            Self::Object => ".o",
            Self::Bitcode => ".bc",
            Self::Assembly => ".s",
            Self::LlvmIr => ".ll",
        }
    }
}

/// Linker input file.
#[derive(Debug, Clone)]
pub struct LinkerInput {
    /// Path to the input file
    pub path: PathBuf,
    /// Type of input
    pub kind: LinkerInputKind,
}

/// Kind of linker input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkerInputKind {
    /// Object file (.o)
    Object,
    /// Static library (.a, .lib)
    StaticLib,
    /// Dynamic library (.so, .dylib, .dll)
    DynamicLib,
    /// Bitcode file (.bc)
    Bitcode,
    /// Linker script
    Script,
    /// Archive file
    Archive,
}

/// Linker invocation builder.
#[derive(Debug)]
pub struct Linker {
    /// Target specification
    spec: TargetSpec,
    /// Linker executable path
    executable: PathBuf,
    /// Linker flavor
    flavor: LinkerFlavor,
    /// Input files
    inputs: Vec<LinkerInput>,
    /// Output file
    output: Option<PathBuf>,
    /// Output type
    output_type: OutputType,
    /// Library search paths
    lib_paths: Vec<PathBuf>,
    /// Libraries to link
    libraries: Vec<String>,
    /// Framework search paths (macOS)
    framework_paths: Vec<PathBuf>,
    /// Frameworks to link (macOS)
    frameworks: Vec<String>,
    /// Extra arguments
    extra_args: Vec<String>,
    /// Whether to link statically
    static_link: bool,
    /// Whether to strip symbols
    strip: bool,
    /// Link-time optimization
    lto: Option<LtoMode>,
    /// Linker script
    script: Option<PathBuf>,
    /// Sysroot
    sysroot: Option<PathBuf>,
}

impl Linker {
    /// Create a new linker for the given target.
    pub fn new(spec: &TargetSpec) -> LinkerResult<Self> {
        let flavor = spec.linker.flavor;
        let executable = Self::find_linker(spec, flavor)?;

        Ok(Self {
            spec: spec.clone(),
            executable,
            flavor,
            inputs: Vec::new(),
            output: None,
            output_type: OutputType::Executable,
            lib_paths: Vec::new(),
            libraries: Vec::new(),
            framework_paths: Vec::new(),
            frameworks: Vec::new(),
            extra_args: Vec::new(),
            static_link: false,
            strip: false,
            lto: None,
            script: None,
            sysroot: None,
        })
    }

    /// Create with a specific linker path.
    pub fn with_path(spec: &TargetSpec, path: PathBuf) -> Self {
        Self {
            spec: spec.clone(),
            executable: path,
            flavor: spec.linker.flavor,
            inputs: Vec::new(),
            output: None,
            output_type: OutputType::Executable,
            lib_paths: Vec::new(),
            libraries: Vec::new(),
            framework_paths: Vec::new(),
            frameworks: Vec::new(),
            extra_args: Vec::new(),
            static_link: false,
            strip: false,
            lto: None,
            script: None,
            sysroot: None,
        }
    }

    /// Find the linker for the given target and flavor.
    fn find_linker(spec: &TargetSpec, flavor: LinkerFlavor) -> LinkerResult<PathBuf> {
        // Check if user specified a path
        if let Some(ref path) = spec.linker.path
            && path.exists()
        {
            return Ok(path.clone());
        }

        // Try to find linker based on flavor
        let candidates = match flavor {
            LinkerFlavor::Gcc => vec![
                format!("{}-gcc", spec.triple),
                "gcc".to_string(),
                "cc".to_string(),
            ],
            LinkerFlavor::Gnu => vec![format!("{}-ld", spec.triple), "ld".to_string()],
            LinkerFlavor::Lld | LinkerFlavor::LldLink => vec![
                "ld.lld".to_string(),
                "lld".to_string(),
                "lld-link".to_string(),
            ],
            LinkerFlavor::LldCoff => vec!["lld-link".to_string()],
            LinkerFlavor::LldMacho => vec!["ld64.lld".to_string()],
            LinkerFlavor::WasmLd => vec!["wasm-ld".to_string(), "wasm-ld-14".to_string()],
            LinkerFlavor::Msvc => vec!["link.exe".to_string(), "lld-link.exe".to_string()],
            LinkerFlavor::Darwin => vec!["ld".to_string(), "ld64".to_string()],
            LinkerFlavor::Em => vec!["emcc".to_string()],
            LinkerFlavor::Ptx => vec!["ptxas".to_string()],
            LinkerFlavor::Bpf => vec!["bpf-linker".to_string()],
        };

        for candidate in &candidates {
            if let Ok(path) = which::which(candidate) {
                return Ok(path);
            }
        }

        Err(LinkerError::NotFound(format!(
            "No linker found for flavor {:?}. Tried: {:?}",
            flavor, candidates
        )))
    }

    /// Add an input file.
    pub fn add_input(&mut self, path: PathBuf, kind: LinkerInputKind) -> &mut Self {
        self.inputs.push(LinkerInput { path, kind });
        self
    }

    /// Add an object file.
    pub fn add_object(&mut self, path: PathBuf) -> &mut Self {
        self.add_input(path, LinkerInputKind::Object)
    }

    /// Set the output file.
    pub fn output(&mut self, path: PathBuf) -> &mut Self {
        self.output = Some(path);
        self
    }

    /// Set the output type.
    pub fn output_type(&mut self, ty: OutputType) -> &mut Self {
        self.output_type = ty;
        self
    }

    /// Add a library search path.
    pub fn add_lib_path(&mut self, path: PathBuf) -> &mut Self {
        self.lib_paths.push(path);
        self
    }

    /// Add a library to link.
    pub fn add_library(&mut self, name: &str) -> &mut Self {
        self.libraries.push(name.to_string());
        self
    }

    /// Add a framework search path (macOS only).
    pub fn add_framework_path(&mut self, path: PathBuf) -> &mut Self {
        self.framework_paths.push(path);
        self
    }

    /// Add a framework to link (macOS only).
    pub fn add_framework(&mut self, name: &str) -> &mut Self {
        self.frameworks.push(name.to_string());
        self
    }

    /// Add extra linker arguments.
    pub fn add_arg(&mut self, arg: &str) -> &mut Self {
        self.extra_args.push(arg.to_string());
        self
    }

    /// Add multiple arguments.
    pub fn add_args(&mut self, args: &[&str]) -> &mut Self {
        for arg in args {
            self.extra_args.push((*arg).to_string());
        }
        self
    }

    /// Enable static linking.
    pub fn static_link(&mut self, enable: bool) -> &mut Self {
        self.static_link = enable;
        self
    }

    /// Enable symbol stripping.
    pub fn strip(&mut self, enable: bool) -> &mut Self {
        self.strip = enable;
        self
    }

    /// Enable LTO.
    pub fn lto(&mut self, mode: LtoMode) -> &mut Self {
        self.lto = Some(mode);
        self
    }

    /// Set linker script.
    pub fn script(&mut self, path: PathBuf) -> &mut Self {
        self.script = Some(path);
        self
    }

    /// Set sysroot.
    pub fn sysroot(&mut self, path: PathBuf) -> &mut Self {
        self.sysroot = Some(path);
        self
    }

    /// Apply settings from a Sysroot.
    pub fn with_sysroot(&mut self, sysroot: &Sysroot) -> &mut Self {
        for path in &sysroot.lib_paths {
            self.add_lib_path(path.clone());
        }
        self.sysroot = Some(sysroot.path.clone());
        self
    }

    /// Build the linker command.
    pub fn build_command(&self) -> LinkerResult<Command> {
        let mut cmd = Command::new(&self.executable);

        match self.flavor {
            LinkerFlavor::Gcc => self.build_gcc_args(&mut cmd)?,
            LinkerFlavor::Gnu | LinkerFlavor::Lld | LinkerFlavor::LldLink => {
                self.build_gnu_args(&mut cmd)?
            }
            LinkerFlavor::Msvc | LinkerFlavor::LldCoff => self.build_msvc_args(&mut cmd)?,
            LinkerFlavor::Darwin | LinkerFlavor::LldMacho => self.build_darwin_args(&mut cmd)?,
            LinkerFlavor::WasmLd => self.build_wasm_args(&mut cmd)?,
            LinkerFlavor::Em => self.build_emscripten_args(&mut cmd)?,
            LinkerFlavor::Ptx | LinkerFlavor::Bpf => self.build_special_args(&mut cmd)?,
        }

        Ok(cmd)
    }

    /// Build arguments for GCC-style linker.
    fn build_gcc_args(&self, cmd: &mut Command) -> LinkerResult<()> {
        // Pre-link args from spec
        for arg in &self.spec.linker.pre_link_args {
            cmd.arg(arg);
        }

        // Sysroot
        if let Some(ref sysroot) = self.sysroot {
            cmd.arg(format!("--sysroot={}", sysroot.display()));
        }

        // Target triple
        cmd.arg("-target");
        cmd.arg(self.spec.llvm_target());

        // Output type
        match self.output_type {
            OutputType::DynamicLib => {
                cmd.arg("-shared");
            }
            OutputType::StaticLib => {
                // GCC doesn't directly create static libs, use ar
                return Err(LinkerError::UnsupportedFlavor {
                    flavor: self.flavor,
                    target: "static library (use ar instead)".to_string(),
                });
            }
            OutputType::Object => {
                cmd.arg("-c");
            }
            _ => {}
        }

        // Relocation model
        match self.spec.env.relocation_model {
            RelocationModel::Pic => {
                cmd.arg("-fPIC");
            }
            RelocationModel::Pie => {
                cmd.arg("-fPIE");
                cmd.arg("-pie");
            }
            RelocationModel::Static => {
                cmd.arg("-static");
            }
            _ => {}
        }

        // Static linking
        if self.static_link {
            cmd.arg("-static");
        }

        // LTO
        if let Some(lto) = &self.lto {
            match lto {
                LtoMode::Thin => {
                    cmd.arg("-flto=thin");
                }
                LtoMode::Fat => {
                    cmd.arg("-flto");
                }
                LtoMode::Off => {}
            }
        }

        // Library paths
        for path in &self.lib_paths {
            cmd.arg(format!("-L{}", path.display()));
        }

        // Input files
        for input in &self.inputs {
            match input.kind {
                LinkerInputKind::Object | LinkerInputKind::Bitcode => {
                    cmd.arg(&input.path);
                }
                LinkerInputKind::StaticLib => {
                    cmd.arg("-Wl,--whole-archive");
                    cmd.arg(&input.path);
                    cmd.arg("-Wl,--no-whole-archive");
                }
                LinkerInputKind::Archive => {
                    cmd.arg(&input.path);
                }
                LinkerInputKind::Script => {
                    cmd.arg(format!("-T{}", input.path.display()));
                }
                _ => {
                    cmd.arg(&input.path);
                }
            }
        }

        // Linker script
        if let Some(ref script) = self.script {
            cmd.arg(format!("-T{}", script.display()));
        }

        // Libraries
        for lib in &self.libraries {
            cmd.arg(format!("-l{}", lib));
        }

        // Libraries from spec
        for lib in &self.spec.linker.libraries {
            cmd.arg(format!("-l{}", lib));
        }

        // Output file
        if let Some(ref output) = self.output {
            cmd.arg("-o");
            cmd.arg(output);
        }

        // Strip
        if self.strip {
            cmd.arg("-s");
        }

        // Extra args
        for arg in &self.extra_args {
            cmd.arg(arg);
        }

        // Post-link args from spec
        for arg in &self.spec.linker.post_link_args {
            cmd.arg(arg);
        }

        // Args from spec
        for arg in &self.spec.linker.args {
            cmd.arg(arg);
        }

        Ok(())
    }

    /// Build arguments for GNU ld-style linker.
    fn build_gnu_args(&self, cmd: &mut Command) -> LinkerResult<()> {
        // Emulation mode based on target
        if let Some(emulation) = self.get_gnu_emulation() {
            cmd.arg("-m");
            cmd.arg(emulation);
        }

        // Sysroot
        if let Some(ref sysroot) = self.sysroot {
            cmd.arg(format!("--sysroot={}", sysroot.display()));
        }

        // Output type
        match self.output_type {
            OutputType::DynamicLib => {
                cmd.arg("-shared");
            }
            OutputType::Executable => {
                if self.spec.os.requires_pie {
                    cmd.arg("-pie");
                }
            }
            _ => {}
        }

        // Static linking
        if self.static_link {
            cmd.arg("-static");
        }

        // Library paths
        for path in &self.lib_paths {
            cmd.arg(format!("-L{}", path.display()));
        }

        // Input files
        for input in &self.inputs {
            match input.kind {
                LinkerInputKind::StaticLib => {
                    cmd.arg("--whole-archive");
                    cmd.arg(&input.path);
                    cmd.arg("--no-whole-archive");
                }
                LinkerInputKind::Script => {
                    cmd.arg("-T");
                    cmd.arg(&input.path);
                }
                _ => {
                    cmd.arg(&input.path);
                }
            }
        }

        // Linker script
        if let Some(ref script) = self.script {
            cmd.arg("-T");
            cmd.arg(script);
        }

        // Libraries
        for lib in &self.libraries {
            cmd.arg(format!("-l{}", lib));
        }

        // Output file
        if let Some(ref output) = self.output {
            cmd.arg("-o");
            cmd.arg(output);
        }

        // Strip
        if self.strip {
            cmd.arg("-s");
        }

        // Extra args
        for arg in &self.extra_args {
            cmd.arg(arg);
        }

        Ok(())
    }

    /// Build arguments for MSVC linker.
    fn build_msvc_args(&self, cmd: &mut Command) -> LinkerResult<()> {
        // Output type
        if self.output_type == OutputType::DynamicLib {
            cmd.arg("/DLL");
        }

        // Library paths
        for path in &self.lib_paths {
            cmd.arg(format!("/LIBPATH:{}", path.display()));
        }

        // Input files
        for input in &self.inputs {
            cmd.arg(&input.path);
        }

        // Libraries
        for lib in &self.libraries {
            if lib.ends_with(".lib") {
                cmd.arg(lib);
            } else {
                cmd.arg(format!("{}.lib", lib));
            }
        }

        // Output file
        if let Some(ref output) = self.output {
            cmd.arg(format!("/OUT:{}", output.display()));
        }

        // Strip (MSVC doesn't have -s, debugging info is separate)
        if self.strip {
            cmd.arg("/DEBUG:NONE");
        }

        // Extra args
        for arg in &self.extra_args {
            cmd.arg(arg);
        }

        Ok(())
    }

    /// Build arguments for Darwin/macOS linker.
    fn build_darwin_args(&self, cmd: &mut Command) -> LinkerResult<()> {
        // Architecture
        cmd.arg("-arch");
        cmd.arg(match self.spec.arch.arch.name() {
            "x86_64" => "x86_64",
            "aarch64" => "arm64",
            "arm" => "armv7",
            _ => self.spec.arch.arch.name(),
        });

        // Sysroot (SDK path)
        if let Some(ref sysroot) = self.sysroot {
            cmd.arg("-syslibroot");
            cmd.arg(sysroot);
        }

        // Output type
        match self.output_type {
            OutputType::DynamicLib => {
                cmd.arg("-dylib");
            }
            OutputType::Executable => {
                cmd.arg("-execute");
            }
            _ => {}
        }

        // Library paths
        for path in &self.lib_paths {
            cmd.arg("-L");
            cmd.arg(path);
        }

        // Framework paths
        for path in &self.framework_paths {
            cmd.arg("-F");
            cmd.arg(path);
        }

        // Input files
        for input in &self.inputs {
            cmd.arg(&input.path);
        }

        // Libraries
        for lib in &self.libraries {
            cmd.arg("-l");
            cmd.arg(lib);
        }

        // Frameworks
        for fw in &self.frameworks {
            cmd.arg("-framework");
            cmd.arg(fw);
        }

        // Output file
        if let Some(ref output) = self.output {
            cmd.arg("-o");
            cmd.arg(output);
        }

        // Strip
        if self.strip {
            cmd.arg("-S"); // Strip debugging symbols
        }

        // Extra args
        for arg in &self.extra_args {
            cmd.arg(arg);
        }

        Ok(())
    }

    /// Build arguments for wasm-ld.
    fn build_wasm_args(&self, cmd: &mut Command) -> LinkerResult<()> {
        // No entry point for libraries
        if self.output_type == OutputType::DynamicLib {
            cmd.arg("--no-entry");
            cmd.arg("--export-dynamic");
        }

        // Allow undefined symbols by default for WASM
        cmd.arg("--allow-undefined");

        // Library paths
        for path in &self.lib_paths {
            cmd.arg("-L");
            cmd.arg(path);
        }

        // Input files
        for input in &self.inputs {
            cmd.arg(&input.path);
        }

        // Libraries
        for lib in &self.libraries {
            cmd.arg("-l");
            cmd.arg(lib);
        }

        // Output file
        if let Some(ref output) = self.output {
            cmd.arg("-o");
            cmd.arg(output);
        }

        // Strip
        if self.strip {
            cmd.arg("--strip-all");
        }

        // Extra args
        for arg in &self.extra_args {
            cmd.arg(arg);
        }

        Ok(())
    }

    /// Build arguments for Emscripten.
    fn build_emscripten_args(&self, cmd: &mut Command) -> LinkerResult<()> {
        // Similar to GCC but with emcc specifics
        self.build_gcc_args(cmd)?;

        // Add WASM-specific flags
        cmd.arg("-s");
        cmd.arg("WASM=1");

        Ok(())
    }

    /// Build arguments for special targets (PTX, BPF).
    fn build_special_args(&self, cmd: &mut Command) -> LinkerResult<()> {
        // Input files
        for input in &self.inputs {
            cmd.arg(&input.path);
        }

        // Output file
        if let Some(ref output) = self.output {
            cmd.arg("-o");
            cmd.arg(output);
        }

        // Extra args
        for arg in &self.extra_args {
            cmd.arg(arg);
        }

        Ok(())
    }

    /// Get GNU ld emulation mode for target.
    fn get_gnu_emulation(&self) -> Option<&'static str> {
        let arch = self.spec.arch.arch.name();
        let os = &self.spec.os.os;

        match (arch, os) {
            ("x86_64", OperatingSystem::Linux) => Some("elf_x86_64"),
            ("x86", OperatingSystem::Linux) => Some("elf_i386"),
            ("aarch64", OperatingSystem::Linux) => Some("aarch64linux"),
            ("arm", OperatingSystem::Linux) => Some("armelf_linux_eabi"),
            ("riscv64", OperatingSystem::Linux) => Some("elf64lriscv"),
            ("riscv32", OperatingSystem::Linux) => Some("elf32lriscv"),
            _ => None,
        }
    }

    /// Execute the linker.
    pub fn link(&self) -> LinkerResult<Output> {
        let mut cmd = self.build_command()?;

        let output = cmd
            .output()
            .map_err(|e| LinkerError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LinkerError::LinkFailed(format!(
                "Linker returned exit code {:?}:\n{}",
                output.status.code(),
                stderr
            )));
        }

        Ok(output)
    }

    /// Get the linker command as a string (for debugging).
    pub fn command_string(&self) -> LinkerResult<String> {
        let cmd = self.build_command()?;
        let prog = cmd.get_program().to_string_lossy();
        let args: Vec<_> = cmd.get_args().map(|a| a.to_string_lossy()).collect();
        Ok(format!("{} {}", prog, args.join(" ")))
    }
}

/// Linker script generator.
#[derive(Debug, Default)]
pub struct LinkerScriptBuilder {
    /// Entry point symbol
    entry: Option<String>,
    /// Memory regions
    memory_regions: Vec<MemoryRegion>,
    /// Sections
    sections: Vec<LinkerSection>,
    /// Extra content
    extra: Vec<String>,
}

/// Memory region definition.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Region name
    pub name: String,
    /// Attributes (rwx)
    pub attributes: String,
    /// Origin address
    pub origin: u64,
    /// Length in bytes
    pub length: u64,
}

/// Linker section definition.
#[derive(Debug, Clone)]
pub struct LinkerSection {
    /// Section name
    pub name: String,
    /// Load address
    pub load_address: Option<String>,
    /// Virtual address
    pub virtual_address: Option<String>,
    /// Memory region
    pub region: Option<String>,
    /// Input sections
    pub inputs: Vec<String>,
    /// Alignment
    pub alignment: Option<u64>,
}

impl LinkerScriptBuilder {
    /// Create a new linker script builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the entry point.
    pub fn entry(mut self, symbol: &str) -> Self {
        self.entry = Some(symbol.to_string());
        self
    }

    /// Add a memory region.
    pub fn memory_region(mut self, region: MemoryRegion) -> Self {
        self.memory_regions.push(region);
        self
    }

    /// Add a section.
    pub fn section(mut self, section: LinkerSection) -> Self {
        self.sections.push(section);
        self
    }

    /// Add raw content.
    pub fn raw(mut self, content: &str) -> Self {
        self.extra.push(content.to_string());
        self
    }

    /// Build the linker script.
    pub fn build(&self) -> String {
        let mut script = String::new();

        // Entry point
        if let Some(ref entry) = self.entry {
            script.push_str(&format!("ENTRY({})\n\n", entry));
        }

        // Memory regions
        if !self.memory_regions.is_empty() {
            script.push_str("MEMORY\n{\n");
            for region in &self.memory_regions {
                script.push_str(&format!(
                    "    {} ({}) : ORIGIN = {:#x}, LENGTH = {:#x}\n",
                    region.name, region.attributes, region.origin, region.length
                ));
            }
            script.push_str("}\n\n");
        }

        // Sections
        script.push_str("SECTIONS\n{\n");
        for section in &self.sections {
            script.push_str(&format!("    {} ", section.name));

            if let Some(ref addr) = section.load_address {
                script.push_str(&format!("{} ", addr));
            }

            script.push_str(": {\n");

            if let Some(align) = section.alignment {
                script.push_str(&format!("        . = ALIGN({});\n", align));
            }

            for input in &section.inputs {
                script.push_str(&format!("        {}\n", input));
            }

            script.push_str("    }");

            if let Some(ref region) = section.region {
                script.push_str(&format!(" > {}", region));
            }

            script.push('\n');
        }
        script.push_str("}\n");

        // Extra content
        for content in &self.extra {
            script.push_str(content);
            script.push('\n');
        }

        script
    }

    /// Write the script to a file.
    pub fn write(&self, path: &Path) -> LinkerResult<()> {
        std::fs::write(path, self.build())?;
        Ok(())
    }
}

/// Common linker script templates.
pub mod templates {
    use super::*;

    /// Generate a basic embedded linker script.
    pub fn embedded(flash_start: u64, flash_size: u64, ram_start: u64, ram_size: u64) -> String {
        LinkerScriptBuilder::new()
            .entry("_start")
            .memory_region(MemoryRegion {
                name: "FLASH".to_string(),
                attributes: "rx".to_string(),
                origin: flash_start,
                length: flash_size,
            })
            .memory_region(MemoryRegion {
                name: "RAM".to_string(),
                attributes: "rwx".to_string(),
                origin: ram_start,
                length: ram_size,
            })
            .section(LinkerSection {
                name: ".text".to_string(),
                load_address: None,
                virtual_address: None,
                region: Some("FLASH".to_string()),
                inputs: vec![
                    "*(.text._start)".to_string(),
                    "*(.text*)".to_string(),
                    "*(.rodata*)".to_string(),
                ],
                alignment: Some(4),
            })
            .section(LinkerSection {
                name: ".data".to_string(),
                load_address: None,
                virtual_address: None,
                region: Some("RAM".to_string()),
                inputs: vec!["*(.data*)".to_string()],
                alignment: Some(4),
            })
            .section(LinkerSection {
                name: ".bss".to_string(),
                load_address: None,
                virtual_address: None,
                region: Some("RAM".to_string()),
                inputs: vec!["*(.bss*)".to_string(), "*(COMMON)".to_string()],
                alignment: Some(4),
            })
            .build()
    }

    /// Generate a kernel linker script.
    pub fn kernel(kernel_base: u64) -> String {
        LinkerScriptBuilder::new()
            .entry("_start")
            .raw(&format!(". = {:#x};", kernel_base))
            .section(LinkerSection {
                name: ".text".to_string(),
                load_address: None,
                virtual_address: None,
                region: None,
                inputs: vec!["*(.text.boot)".to_string(), "*(.text*)".to_string()],
                alignment: Some(4096),
            })
            .section(LinkerSection {
                name: ".rodata".to_string(),
                load_address: None,
                virtual_address: None,
                region: None,
                inputs: vec!["*(.rodata*)".to_string()],
                alignment: Some(4096),
            })
            .section(LinkerSection {
                name: ".data".to_string(),
                load_address: None,
                virtual_address: None,
                region: None,
                inputs: vec!["*(.data*)".to_string()],
                alignment: Some(4096),
            })
            .section(LinkerSection {
                name: ".bss".to_string(),
                load_address: None,
                virtual_address: None,
                region: None,
                inputs: vec!["*(.bss*)".to_string(), "*(COMMON)".to_string()],
                alignment: Some(4096),
            })
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_type_extension() {
        let spec = TargetSpec::from_triple("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(OutputType::Executable.extension(&spec), "");
        assert_eq!(OutputType::DynamicLib.extension(&spec), ".so");
        assert_eq!(OutputType::StaticLib.extension(&spec), ".a");
        assert_eq!(OutputType::Object.extension(&spec), ".o");

        let spec = TargetSpec::from_triple("x86_64-pc-windows-msvc").unwrap();
        assert_eq!(OutputType::Executable.extension(&spec), ".exe");
        assert_eq!(OutputType::DynamicLib.extension(&spec), ".dll");
        assert_eq!(OutputType::StaticLib.extension(&spec), ".lib");
    }

    #[test]
    fn test_linker_script_builder() {
        let script = LinkerScriptBuilder::new()
            .entry("main")
            .memory_region(MemoryRegion {
                name: "ROM".to_string(),
                attributes: "rx".to_string(),
                origin: 0x0800_0000,
                length: 0x0010_0000,
            })
            .section(LinkerSection {
                name: ".text".to_string(),
                load_address: None,
                virtual_address: None,
                region: Some("ROM".to_string()),
                inputs: vec!["*(.text*)".to_string()],
                alignment: Some(4),
            })
            .build();

        assert!(script.contains("ENTRY(main)"));
        assert!(script.contains("ROM (rx)"));
        assert!(script.contains("ORIGIN = 0x8000000"));
        assert!(script.contains(".text"));
    }

    #[test]
    fn test_embedded_template() {
        let script = templates::embedded(0x0800_0000, 0x0010_0000, 0x2000_0000, 0x0002_0000);

        assert!(script.contains("ENTRY(_start)"));
        assert!(script.contains("FLASH (rx)"));
        assert!(script.contains("RAM (rwx)"));
        assert!(script.contains(".text"));
        assert!(script.contains(".data"));
        assert!(script.contains(".bss"));
    }

    #[test]
    fn test_kernel_template() {
        let script = templates::kernel(0xFFFF_FFFF_8000_0000);

        assert!(script.contains("ENTRY(_start)"));
        assert!(script.contains("0xffffffff80000000"));
        assert!(script.contains(".text.boot"));
    }

    #[test]
    fn test_linker_flavor_candidates() {
        // Just test that the linker can be created without panicking
        // (actual linking requires the linker to be installed)
        let spec = TargetSpec::from_triple("x86_64-unknown-linux-gnu").unwrap();
        // Note: This might fail if no linker is installed, which is okay for unit tests
        let _ = Linker::new(&spec);
    }
}
