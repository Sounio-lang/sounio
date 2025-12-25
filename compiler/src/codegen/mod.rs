//! Code generation backends
//!
//! D supports multiple code generation backends:
//! - LLVM: For optimized native code (requires `--features llvm`)
//! - Cranelift: For fast JIT compilation (requires `--features jit`)
//! - GPU: For CUDA/SPIR-V compute kernels (requires `--features gpu`)
//! - Debug: DWARF debug information and source maps

pub mod autodiff;
pub mod cranelift;
pub mod debug;
pub mod gpu;
pub mod simd;

// The LLVM backend is in a subdirectory when the feature is enabled
#[cfg(feature = "llvm")]
#[path = "llvm/mod.rs"]
pub mod llvm;

// Provide stub module when LLVM is not enabled
#[cfg(not(feature = "llvm"))]
pub mod llvm {
    //! LLVM backend stub (feature not enabled)

    use crate::hlir::HlirModule;
    use std::path::Path;

    /// Optimization level (stub)
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum OptLevel {
        O0,
        O1,
        #[default]
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

    #[derive(Debug)]
    pub enum LinkError {
        NotEnabled,
    }

    impl std::fmt::Display for LinkError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "LLVM backend not enabled")
        }
    }

    impl std::error::Error for LinkError {}

    impl Linker {
        pub fn link(_objects: &[&Path], _output: &Path) -> Result<(), LinkError> {
            Err(LinkError::NotEnabled)
        }
    }
}

/// Backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// LLVM backend for optimized native code
    LLVM,
    /// Cranelift backend for fast JIT compilation
    Cranelift,
    /// GPU backend for compute kernels
    GPU,
}

impl Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Backend::LLVM => "llvm",
            Backend::Cranelift => "cranelift",
            Backend::GPU => "gpu",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Target triple for code generation
#[derive(Debug, Clone)]
pub struct Target {
    pub arch: Architecture,
    pub os: OperatingSystem,
    pub env: Environment,
}

#[derive(Debug, Clone, Copy)]
pub enum Architecture {
    X86_64,
    AArch64,
    Wasm32,
    Wasm64,
    NVPTX64, // NVIDIA PTX
    SPIRV64, // SPIR-V
}

#[derive(Debug, Clone, Copy)]
pub enum OperatingSystem {
    Linux,
    MacOS,
    Windows,
    None, // For bare metal / GPU
}

#[derive(Debug, Clone, Copy)]
pub enum Environment {
    GNU,
    MSVC,
    Musl,
    None,
}

impl Target {
    pub fn host() -> Self {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        return Self {
            arch: Architecture::X86_64,
            os: OperatingSystem::Linux,
            env: Environment::GNU,
        };

        #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
        return Self {
            arch: Architecture::X86_64,
            os: OperatingSystem::MacOS,
            env: Environment::None,
        };

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        return Self {
            arch: Architecture::AArch64,
            os: OperatingSystem::MacOS,
            env: Environment::None,
        };

        #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
        return Self {
            arch: Architecture::X86_64,
            os: OperatingSystem::Windows,
            env: Environment::MSVC,
        };

        // Default fallback
        #[allow(unreachable_code)]
        Self {
            arch: Architecture::X86_64,
            os: OperatingSystem::Linux,
            env: Environment::GNU,
        }
    }

    pub fn triple(&self) -> String {
        let arch = match self.arch {
            Architecture::X86_64 => "x86_64",
            Architecture::AArch64 => "aarch64",
            Architecture::Wasm32 => "wasm32",
            Architecture::Wasm64 => "wasm64",
            Architecture::NVPTX64 => "nvptx64",
            Architecture::SPIRV64 => "spirv64",
        };

        let os = match self.os {
            OperatingSystem::Linux => "linux",
            OperatingSystem::MacOS => "darwin",
            OperatingSystem::Windows => "windows",
            OperatingSystem::None => "unknown",
        };

        let env = match self.env {
            Environment::GNU => "gnu",
            Environment::MSVC => "msvc",
            Environment::Musl => "musl",
            Environment::None => "",
        };

        if env.is_empty() {
            format!("{}-{}", arch, os)
        } else {
            format!("{}-{}-{}", arch, os, env)
        }
    }
}
