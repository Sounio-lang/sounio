//! SIR Module
//!
//! A SIR module is the top-level compilation unit, containing functions,
//! global variables, and type definitions.

use super::blocks::{Linkage, SirFunction};
use super::types::SirType;
use super::values::{Constant, FuncId};
use std::collections::HashMap;
use std::fmt;

/// A global variable in SIR
#[derive(Debug, Clone)]
pub struct SirGlobal {
    /// Name of the global
    pub name: String,
    /// Type of the global
    pub ty: SirType,
    /// Initial value (if any)
    pub initializer: Option<Constant>,
    /// Is this a constant?
    pub constant: bool,
    /// Linkage
    pub linkage: Linkage,
    /// Alignment override
    pub align: Option<u32>,
    /// Section (for linking)
    pub section: Option<String>,
}

impl SirGlobal {
    pub fn new(name: impl Into<String>, ty: SirType) -> Self {
        Self {
            name: name.into(),
            ty,
            initializer: None,
            constant: false,
            linkage: Linkage::External,
            align: None,
            section: None,
        }
    }

    pub fn with_initializer(mut self, init: Constant) -> Self {
        self.initializer = Some(init);
        self
    }

    pub fn constant(mut self) -> Self {
        self.constant = true;
        self
    }

    pub fn internal(mut self) -> Self {
        self.linkage = Linkage::Internal;
        self
    }
}

/// External function declaration
#[derive(Debug, Clone)]
pub struct ExternFunction {
    /// Name of the external function
    pub name: String,
    /// Parameter types
    pub params: Vec<SirType>,
    /// Return type
    pub ret_ty: SirType,
    /// Is this variadic?
    pub variadic: bool,
}

impl ExternFunction {
    pub fn new(name: impl Into<String>, params: Vec<SirType>, ret_ty: SirType) -> Self {
        Self {
            name: name.into(),
            params,
            ret_ty,
            variadic: false,
        }
    }

    pub fn variadic(mut self) -> Self {
        self.variadic = true;
        self
    }
}

/// Target triple for code generation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetTriple {
    pub arch: Architecture,
    pub vendor: String,
    pub os: OperatingSystem,
    pub env: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    AArch64,
    Riscv64,
    Wasm32,
    Wasm64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingSystem {
    Linux,
    MacOS,
    Windows,
    FreeBSD,
    Wasi,
    None, // Bare metal
}

impl TargetTriple {
    pub fn native() -> Self {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        return Self {
            arch: Architecture::X86_64,
            vendor: "unknown".into(),
            os: OperatingSystem::Linux,
            env: Some("gnu".into()),
        };

        #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
        return Self {
            arch: Architecture::X86_64,
            vendor: "apple".into(),
            os: OperatingSystem::MacOS,
            env: None,
        };

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        return Self {
            arch: Architecture::AArch64,
            vendor: "apple".into(),
            os: OperatingSystem::MacOS,
            env: None,
        };

        #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
        return Self {
            arch: Architecture::X86_64,
            vendor: "pc".into(),
            os: OperatingSystem::Windows,
            env: Some("msvc".into()),
        };

        #[cfg(not(any(
            all(target_arch = "x86_64", target_os = "linux"),
            all(target_arch = "x86_64", target_os = "macos"),
            all(target_arch = "aarch64", target_os = "macos"),
            all(target_arch = "x86_64", target_os = "windows"),
        )))]
        Self {
            arch: Architecture::X86_64,
            vendor: "unknown".into(),
            os: OperatingSystem::Linux,
            env: Some("gnu".into()),
        }
    }

    pub fn pointer_size(&self) -> usize {
        match self.arch {
            Architecture::X86_64
            | Architecture::AArch64
            | Architecture::Riscv64
            | Architecture::Wasm64 => 8,
            Architecture::Wasm32 => 4,
        }
    }

    pub fn is_64bit(&self) -> bool {
        self.pointer_size() == 8
    }
}

impl fmt::Display for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arch = match self.arch {
            Architecture::X86_64 => "x86_64",
            Architecture::AArch64 => "aarch64",
            Architecture::Riscv64 => "riscv64",
            Architecture::Wasm32 => "wasm32",
            Architecture::Wasm64 => "wasm64",
        };
        let os = match self.os {
            OperatingSystem::Linux => "linux",
            OperatingSystem::MacOS => "darwin",
            OperatingSystem::Windows => "windows",
            OperatingSystem::FreeBSD => "freebsd",
            OperatingSystem::Wasi => "wasi",
            OperatingSystem::None => "none",
        };

        write!(f, "{}-{}-{}", arch, self.vendor, os)?;
        if let Some(env) = &self.env {
            write!(f, "-{}", env)?;
        }
        Ok(())
    }
}

/// A SIR module (compilation unit)
#[derive(Debug, Clone)]
pub struct SirModule {
    /// Module name
    pub name: String,
    /// Source file path
    pub source_file: Option<String>,
    /// Target triple
    pub target: TargetTriple,

    /// Type definitions (named structs)
    pub types: HashMap<String, SirType>,
    /// Global variables
    pub globals: Vec<SirGlobal>,
    /// External function declarations
    pub externs: Vec<ExternFunction>,
    /// Function definitions
    pub functions: Vec<SirFunction>,

    /// Next function ID
    next_func_id: u32,

    // === Module Metadata ===
    /// Data layout string (like LLVM)
    pub data_layout: Option<String>,
    /// Module flags
    pub flags: ModuleFlags,
}

#[derive(Debug, Clone, Default)]
pub struct ModuleFlags {
    /// Enable debug info
    pub debug: bool,
    /// Enable epistemic optimizations
    pub optimize_epistemic: bool,
    /// Enable probability distribution fusion
    pub optimize_prob: bool,
    /// Enable ODE solver fusion
    pub optimize_ode: bool,
    /// Enable SIMD vectorization
    pub vectorize: bool,
    /// Optimization level (0-3)
    pub opt_level: u8,
}

impl SirModule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source_file: None,
            target: TargetTriple::native(),
            types: HashMap::new(),
            globals: vec![],
            externs: vec![],
            functions: vec![],
            next_func_id: 0,
            data_layout: None,
            flags: ModuleFlags::default(),
        }
    }

    pub fn with_target(mut self, target: TargetTriple) -> Self {
        self.target = target;
        self
    }

    pub fn with_source(mut self, path: impl Into<String>) -> Self {
        self.source_file = Some(path.into());
        self
    }

    /// Add a named type
    pub fn add_type(&mut self, name: impl Into<String>, ty: SirType) {
        self.types.insert(name.into(), ty);
    }

    /// Add a global variable
    pub fn add_global(&mut self, global: SirGlobal) {
        self.globals.push(global);
    }

    /// Add an external function declaration
    pub fn add_extern(&mut self, ext: ExternFunction) {
        self.externs.push(ext);
    }

    /// Create a new function
    pub fn create_function(
        &mut self,
        name: impl Into<String>,
        params: Vec<(String, SirType)>,
        ret_ty: SirType,
    ) -> FuncId {
        let id = FuncId(self.next_func_id);
        self.next_func_id += 1;

        let func = SirFunction::new(id, name, params, ret_ty);
        self.functions.push(func);
        id
    }

    /// Get function by ID
    pub fn function(&self, id: FuncId) -> Option<&SirFunction> {
        self.functions.iter().find(|f| f.id == id)
    }

    /// Get function by ID mutably
    pub fn function_mut(&mut self, id: FuncId) -> Option<&mut SirFunction> {
        self.functions.iter_mut().find(|f| f.id == id)
    }

    /// Get function by name
    pub fn function_by_name(&self, name: &str) -> Option<&SirFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Get the main function (if any)
    pub fn main_function(&self) -> Option<&SirFunction> {
        self.function_by_name("main")
    }

    /// Validate the module
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = vec![];

        for func in &self.functions {
            // Check all blocks have terminators
            for block in &func.blocks {
                if block.terminator.is_none() {
                    errors.push(format!(
                        "Function '{}' block {} has no terminator",
                        func.name, block.id
                    ));
                }
            }

            // Check function has at least one block
            if func.blocks.is_empty() {
                errors.push(format!("Function '{}' has no blocks", func.name));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl fmt::Display for SirModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "; ModuleID = '{}'", self.name)?;
        writeln!(f, "; Target = {}", self.target)?;
        if let Some(src) = &self.source_file {
            writeln!(f, "; Source = {}", src)?;
        }
        writeln!(f)?;

        // Type definitions
        if !self.types.is_empty() {
            writeln!(f, "; Type definitions")?;
            for (name, ty) in &self.types {
                writeln!(f, "%{} = type {}", name, ty)?;
            }
            writeln!(f)?;
        }

        // Globals
        if !self.globals.is_empty() {
            writeln!(f, "; Global variables")?;
            for g in &self.globals {
                let const_str = if g.constant { "constant" } else { "global" };
                write!(f, "@{} = {} {}", g.name, const_str, g.ty)?;
                if let Some(init) = &g.initializer {
                    write!(f, " {}", init)?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
        }

        // External declarations
        if !self.externs.is_empty() {
            writeln!(f, "; External declarations")?;
            for ext in &self.externs {
                write!(f, "declare {} @{}(", ext.ret_ty, ext.name)?;
                for (i, ty) in ext.params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                if ext.variadic {
                    write!(f, ", ...")?;
                }
                writeln!(f, ")")?;
            }
            writeln!(f)?;
        }

        // Functions
        for func in &self.functions {
            writeln!(f, "{}", func)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        let mut module = SirModule::new("test");

        let func_id = module.create_function("main", vec![], SirType::i32());

        assert!(module.function(func_id).is_some());
        assert!(module.main_function().is_some());
    }

    #[test]
    fn test_target_triple() {
        let target = TargetTriple::native();
        assert!(target.is_64bit());
        println!("Native target: {}", target);
    }
}
