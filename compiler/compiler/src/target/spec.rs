//! Target Specification System
//!
//! This module provides comprehensive target specification support for cross-compilation,
//! including target triple parsing, architecture specifications, ABI configuration,
//! and feature management.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use thiserror::Error;

/// Errors that can occur during target specification parsing or validation.
#[derive(Debug, Error)]
pub enum TargetSpecError {
    #[error("Invalid target triple format: {0}")]
    InvalidTriple(String),

    #[error("Unknown architecture: {0}")]
    UnknownArchitecture(String),

    #[error("Unknown operating system: {0}")]
    UnknownOperatingSystem(String),

    #[error("Unknown environment/ABI: {0}")]
    UnknownEnvironment(String),

    #[error("Unknown vendor: {0}")]
    UnknownVendor(String),

    #[error("Incompatible feature combination: {0}")]
    IncompatibleFeatures(String),

    #[error("Missing required feature for target: {0}")]
    MissingRequiredFeature(String),

    #[error("Invalid target specification file: {0}")]
    InvalidSpecFile(String),

    #[error("Target not found: {0}")]
    TargetNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Result type for target specification operations.
pub type TargetResult<T> = Result<T, TargetSpecError>;

/// Target triple representation (arch-vendor-os-env).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetTriple {
    /// The target architecture (e.g., x86_64, aarch64, riscv64).
    pub arch: String,
    /// The vendor (e.g., unknown, apple, pc).
    pub vendor: String,
    /// The operating system (e.g., linux, windows, macos).
    pub os: String,
    /// The environment/ABI (e.g., gnu, musl, msvc).
    pub env: Option<String>,
}

impl TargetTriple {
    /// Create a new target triple.
    pub fn new(arch: &str, vendor: &str, os: &str, env: Option<&str>) -> Self {
        Self {
            arch: arch.to_string(),
            vendor: vendor.to_string(),
            os: os.to_string(),
            env: env.map(String::from),
        }
    }

    /// Parse a target triple from a string.
    pub fn parse(s: &str) -> TargetResult<Self> {
        let parts: Vec<&str> = s.split('-').collect();

        match parts.len() {
            2 => {
                // arch-os (e.g., wasm32-unknown)
                Ok(Self::new(parts[0], "unknown", parts[1], None))
            }
            3 => {
                // arch-vendor-os or arch-os-env
                // Heuristic: if middle part looks like a vendor, treat as arch-vendor-os
                if Self::is_vendor(parts[1]) {
                    Ok(Self::new(parts[0], parts[1], parts[2], None))
                } else {
                    Ok(Self::new(parts[0], "unknown", parts[1], Some(parts[2])))
                }
            }
            4 => {
                // arch-vendor-os-env (full form)
                Ok(Self::new(parts[0], parts[1], parts[2], Some(parts[3])))
            }
            _ => Err(TargetSpecError::InvalidTriple(format!(
                "Expected 2-4 components, got {}: {}",
                parts.len(),
                s
            ))),
        }
    }

    /// Check if a string looks like a vendor name.
    fn is_vendor(s: &str) -> bool {
        matches!(
            s,
            "unknown" | "pc" | "apple" | "nvidia" | "amd" | "ibm" | "arm" | "riscv" | "wasi"
        )
    }

    /// Get the canonical string representation.
    pub fn to_string_canonical(&self) -> String {
        match &self.env {
            Some(env) => format!("{}-{}-{}-{}", self.arch, self.vendor, self.os, env),
            None => format!("{}-{}-{}", self.arch, self.vendor, self.os),
        }
    }

    /// Check if this is a bare-metal target (no OS).
    pub fn is_bare_metal(&self) -> bool {
        self.os == "none" || self.os == "unknown" || self.os == "bare"
    }

    /// Check if this is a WebAssembly target.
    pub fn is_wasm(&self) -> bool {
        self.arch.starts_with("wasm")
    }

    /// Check if this is a Windows target.
    pub fn is_windows(&self) -> bool {
        self.os == "windows"
    }

    /// Check if this is a macOS/Darwin target.
    pub fn is_macos(&self) -> bool {
        self.os == "macos" || self.os == "darwin"
    }

    /// Check if this is a Linux target.
    pub fn is_linux(&self) -> bool {
        self.os == "linux"
    }
}

impl fmt::Display for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_canonical())
    }
}

impl FromStr for TargetTriple {
    type Err = TargetSpecError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// CPU architecture enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Architecture {
    /// 64-bit x86 (AMD64/Intel 64)
    X86_64,
    /// 32-bit x86 (i686)
    X86,
    /// 64-bit ARM (AArch64)
    Aarch64,
    /// 32-bit ARM
    Arm,
    /// 64-bit RISC-V
    Riscv64,
    /// 32-bit RISC-V
    Riscv32,
    /// 32-bit WebAssembly
    Wasm32,
    /// 64-bit WebAssembly
    Wasm64,
    /// 64-bit PowerPC
    Powerpc64,
    /// 32-bit PowerPC
    Powerpc,
    /// 64-bit MIPS
    Mips64,
    /// 32-bit MIPS
    Mips,
    /// 64-bit SPARC
    Sparc64,
    /// 64-bit IBM Z (s390x)
    S390x,
    /// Hexagon DSP
    Hexagon,
    /// NVIDIA PTX (CUDA)
    Nvptx64,
    /// AMD GCN (GPU)
    Amdgcn,
    /// Unknown/custom architecture
    Unknown(String),
}

impl Architecture {
    /// Parse an architecture from string.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "x86_64" | "amd64" | "x64" => Self::X86_64,
            "x86" | "i686" | "i386" | "i586" => Self::X86,
            "aarch64" | "arm64" => Self::Aarch64,
            "arm" | "armv7" | "armv7a" | "armv7l" | "armv6" | "thumbv7" => Self::Arm,
            "riscv64" | "riscv64gc" | "riscv64imac" => Self::Riscv64,
            "riscv32" | "riscv32i" | "riscv32imac" => Self::Riscv32,
            "wasm32" => Self::Wasm32,
            "wasm64" => Self::Wasm64,
            "powerpc64" | "ppc64" | "powerpc64le" | "ppc64le" => Self::Powerpc64,
            "powerpc" | "ppc" => Self::Powerpc,
            "mips64" | "mips64el" => Self::Mips64,
            "mips" | "mipsel" => Self::Mips,
            "sparc64" | "sparcv9" => Self::Sparc64,
            "s390x" => Self::S390x,
            "hexagon" => Self::Hexagon,
            "nvptx64" => Self::Nvptx64,
            "amdgcn" => Self::Amdgcn,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// Get the pointer width in bits.
    pub fn pointer_width(&self) -> u32 {
        match self {
            Self::X86_64
            | Self::Aarch64
            | Self::Riscv64
            | Self::Wasm64
            | Self::Powerpc64
            | Self::Mips64
            | Self::Sparc64
            | Self::S390x
            | Self::Nvptx64
            | Self::Amdgcn => 64,
            Self::X86
            | Self::Arm
            | Self::Riscv32
            | Self::Wasm32
            | Self::Powerpc
            | Self::Mips
            | Self::Hexagon => 32,
            Self::Unknown(_) => 64, // Default assumption
        }
    }

    /// Get the default data layout string for LLVM.
    pub fn data_layout(&self) -> &'static str {
        match self {
            Self::X86_64 => {
                "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            }
            Self::X86 => {
                "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
            }
            Self::Aarch64 => "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
            Self::Arm => "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64",
            Self::Riscv64 => "e-m:e-p:64:64-i64:64-i128:128-n64-S128",
            Self::Riscv32 => "e-m:e-p:32:32-i64:64-n32-S128",
            Self::Wasm32 => "e-m:e-p:32:32-i64:64-n32:64-S128",
            Self::Wasm64 => "e-m:e-p:64:64-i64:64-n32:64-S128",
            Self::Powerpc64 => "e-m:e-i64:64-n32:64-S128",
            Self::Powerpc => "E-m:e-p:32:32-i64:64-n32",
            Self::Mips64 => "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128",
            Self::Mips => "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64",
            Self::Sparc64 => "E-m:e-i64:64-n32:64-S128",
            Self::S390x => "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64",
            Self::Hexagon => {
                "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
            }
            Self::Nvptx64 => "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
            Self::Amdgcn => {
                "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
            }
            Self::Unknown(_) => "e-m:e-i64:64-n32:64-S128",
        }
    }

    /// Get the canonical name.
    pub fn name(&self) -> &str {
        match self {
            Self::X86_64 => "x86_64",
            Self::X86 => "x86",
            Self::Aarch64 => "aarch64",
            Self::Arm => "arm",
            Self::Riscv64 => "riscv64",
            Self::Riscv32 => "riscv32",
            Self::Wasm32 => "wasm32",
            Self::Wasm64 => "wasm64",
            Self::Powerpc64 => "powerpc64",
            Self::Powerpc => "powerpc",
            Self::Mips64 => "mips64",
            Self::Mips => "mips",
            Self::Sparc64 => "sparc64",
            Self::S390x => "s390x",
            Self::Hexagon => "hexagon",
            Self::Nvptx64 => "nvptx64",
            Self::Amdgcn => "amdgcn",
            Self::Unknown(name) => name,
        }
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Operating system enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperatingSystem {
    /// Linux-based systems
    Linux,
    /// Microsoft Windows
    Windows,
    /// Apple macOS
    MacOs,
    /// Apple iOS
    Ios,
    /// Google Android
    Android,
    /// FreeBSD
    FreeBsd,
    /// NetBSD
    NetBsd,
    /// OpenBSD
    OpenBsd,
    /// DragonFlyBSD
    DragonFly,
    /// illumos (OpenSolaris derivative)
    Illumos,
    /// Oracle Solaris
    Solaris,
    /// Haiku OS
    Haiku,
    /// Fuchsia OS
    Fuchsia,
    /// Redox OS
    Redox,
    /// WebAssembly System Interface
    Wasi,
    /// Nintendo Switch
    Horizon,
    /// PlayStation Vita
    Vita,
    /// No OS (bare metal)
    None,
    /// Unknown/custom OS
    Unknown(String),
}

impl OperatingSystem {
    /// Parse an OS from string.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "linux" | "gnu" => Self::Linux,
            "windows" | "win32" | "win64" => Self::Windows,
            "macos" | "darwin" | "osx" => Self::MacOs,
            "ios" | "iphoneos" => Self::Ios,
            "android" => Self::Android,
            "freebsd" => Self::FreeBsd,
            "netbsd" => Self::NetBsd,
            "openbsd" => Self::OpenBsd,
            "dragonfly" => Self::DragonFly,
            "illumos" => Self::Illumos,
            "solaris" => Self::Solaris,
            "haiku" => Self::Haiku,
            "fuchsia" => Self::Fuchsia,
            "redox" => Self::Redox,
            "wasi" | "wasip1" | "wasip2" => Self::Wasi,
            "horizon" | "switch" => Self::Horizon,
            "vita" | "psvita" => Self::Vita,
            "none" | "bare" | "unknown" => Self::None,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// Get the canonical name.
    pub fn name(&self) -> &str {
        match self {
            Self::Linux => "linux",
            Self::Windows => "windows",
            Self::MacOs => "macos",
            Self::Ios => "ios",
            Self::Android => "android",
            Self::FreeBsd => "freebsd",
            Self::NetBsd => "netbsd",
            Self::OpenBsd => "openbsd",
            Self::DragonFly => "dragonfly",
            Self::Illumos => "illumos",
            Self::Solaris => "solaris",
            Self::Haiku => "haiku",
            Self::Fuchsia => "fuchsia",
            Self::Redox => "redox",
            Self::Wasi => "wasi",
            Self::Horizon => "horizon",
            Self::Vita => "vita",
            Self::None => "none",
            Self::Unknown(name) => name,
        }
    }

    /// Check if this OS uses Unix-like conventions.
    pub fn is_unix_like(&self) -> bool {
        matches!(
            self,
            Self::Linux
                | Self::MacOs
                | Self::Ios
                | Self::Android
                | Self::FreeBsd
                | Self::NetBsd
                | Self::OpenBsd
                | Self::DragonFly
                | Self::Illumos
                | Self::Solaris
                | Self::Haiku
                | Self::Fuchsia
                | Self::Redox
        )
    }

    /// Get the default executable extension.
    pub fn exe_extension(&self) -> &str {
        match self {
            Self::Windows => ".exe",
            Self::Wasi => ".wasm",
            _ => "",
        }
    }

    /// Get the default shared library extension.
    pub fn dylib_extension(&self) -> &str {
        match self {
            Self::Windows => ".dll",
            Self::MacOs | Self::Ios => ".dylib",
            _ => ".so",
        }
    }

    /// Get the default static library extension.
    pub fn staticlib_extension(&self) -> &str {
        match self {
            Self::Windows => ".lib",
            _ => ".a",
        }
    }
}

impl fmt::Display for OperatingSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Environment/ABI enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    /// GNU libc (glibc)
    Gnu,
    /// GNU libc with extended ABI
    Gnueabi,
    /// GNU libc with hard-float ABI
    Gnueabihf,
    /// GNU libc with 64-bit ABI on x32
    Gnux32,
    /// musl libc
    Musl,
    /// musl with EABI
    Musleabi,
    /// musl with hard-float EABI
    Musleabihf,
    /// Microsoft Visual C++ runtime
    Msvc,
    /// Android NDK
    Android,
    /// Android NDK with EABI
    Androideabi,
    /// macOS/iOS
    Macho,
    /// Embedded ABI (bare metal)
    Eabi,
    /// Embedded ABI with hard-float
    Eabihf,
    /// SGX enclave
    Sgx,
    /// UEFI
    Uefi,
    /// No specific environment
    None,
    /// Unknown/custom environment
    Unknown(String),
}

impl Environment {
    /// Parse an environment from string.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gnu" => Self::Gnu,
            "gnueabi" => Self::Gnueabi,
            "gnueabihf" => Self::Gnueabihf,
            "gnux32" => Self::Gnux32,
            "musl" => Self::Musl,
            "musleabi" => Self::Musleabi,
            "musleabihf" => Self::Musleabihf,
            "msvc" => Self::Msvc,
            "android" => Self::Android,
            "androideabi" => Self::Androideabi,
            "macho" => Self::Macho,
            "eabi" => Self::Eabi,
            "eabihf" => Self::Eabihf,
            "sgx" => Self::Sgx,
            "uefi" => Self::Uefi,
            "none" | "" => Self::None,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// Get the canonical name.
    pub fn name(&self) -> &str {
        match self {
            Self::Gnu => "gnu",
            Self::Gnueabi => "gnueabi",
            Self::Gnueabihf => "gnueabihf",
            Self::Gnux32 => "gnux32",
            Self::Musl => "musl",
            Self::Musleabi => "musleabi",
            Self::Musleabihf => "musleabihf",
            Self::Msvc => "msvc",
            Self::Android => "android",
            Self::Androideabi => "androideabi",
            Self::Macho => "macho",
            Self::Eabi => "eabi",
            Self::Eabihf => "eabihf",
            Self::Sgx => "sgx",
            Self::Uefi => "uefi",
            Self::None => "none",
            Self::Unknown(name) => name,
        }
    }

    /// Check if this environment uses static linking by default.
    pub fn prefers_static(&self) -> bool {
        matches!(
            self,
            Self::Musl | Self::Musleabi | Self::Musleabihf | Self::Eabi | Self::Eabihf
        )
    }

    /// Check if this environment uses hard-float ABI.
    pub fn is_hard_float(&self) -> bool {
        matches!(self, Self::Gnueabihf | Self::Musleabihf | Self::Eabihf)
    }
}

impl fmt::Display for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Detailed architecture-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchSpec {
    /// Base architecture
    pub arch: Architecture,
    /// CPU model (e.g., "generic", "skylake", "cortex-a72")
    pub cpu: String,
    /// CPU features to enable (e.g., "sse4.2", "avx2", "neon")
    pub features: HashSet<String>,
    /// CPU features to disable
    pub disabled_features: HashSet<String>,
    /// Minimum supported CPU for this target
    pub min_cpu: Option<String>,
    /// Target feature level (e.g., "baseline", "v2", "v3", "v4" for x86-64)
    pub feature_level: Option<String>,
    /// Whether to use soft-float ABI
    pub soft_float: bool,
    /// Vector width preference (in bits)
    pub vector_width: Option<u32>,
}

impl Default for ArchSpec {
    fn default() -> Self {
        Self {
            arch: Architecture::X86_64,
            cpu: "generic".to_string(),
            features: HashSet::new(),
            disabled_features: HashSet::new(),
            min_cpu: None,
            feature_level: None,
            soft_float: false,
            vector_width: None,
        }
    }
}

impl ArchSpec {
    /// Create a new architecture specification.
    pub fn new(arch: Architecture) -> Self {
        Self {
            arch,
            ..Default::default()
        }
    }

    /// Set the CPU model.
    pub fn with_cpu(mut self, cpu: &str) -> Self {
        self.cpu = cpu.to_string();
        self
    }

    /// Enable a CPU feature.
    pub fn with_feature(mut self, feature: &str) -> Self {
        self.features.insert(feature.to_string());
        self.disabled_features.remove(feature);
        self
    }

    /// Disable a CPU feature.
    pub fn without_feature(mut self, feature: &str) -> Self {
        self.disabled_features.insert(feature.to_string());
        self.features.remove(feature);
        self
    }

    /// Get LLVM target features string.
    pub fn llvm_features(&self) -> String {
        let mut features = Vec::new();

        for f in &self.features {
            features.push(format!("+{}", f));
        }

        for f in &self.disabled_features {
            features.push(format!("-{}", f));
        }

        if self.soft_float {
            features.push("+soft-float".to_string());
        }

        features.join(",")
    }
}

/// Operating system-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsSpec {
    /// Base operating system
    pub os: OperatingSystem,
    /// Minimum OS version supported
    pub min_version: Option<String>,
    /// Target SDK version
    pub sdk_version: Option<String>,
    /// System root path (sysroot)
    pub sysroot: Option<PathBuf>,
    /// Default stack size
    pub stack_size: Option<usize>,
    /// Whether position-independent executables are required
    pub requires_pie: bool,
    /// Whether stack protector is enabled by default
    pub stack_protector: bool,
    /// Default panic strategy ("abort" or "unwind")
    pub panic_strategy: PanicStrategy,
}

impl Default for OsSpec {
    fn default() -> Self {
        Self {
            os: OperatingSystem::Linux,
            min_version: None,
            sdk_version: None,
            sysroot: None,
            stack_size: None,
            requires_pie: false,
            stack_protector: true,
            panic_strategy: PanicStrategy::Unwind,
        }
    }
}

/// Panic handling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum PanicStrategy {
    /// Unwind the stack on panic
    #[default]
    Unwind,
    /// Abort immediately on panic
    Abort,
}

/// Environment-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvSpec {
    /// Base environment/ABI
    pub env: Environment,
    /// C runtime to link against
    pub crt: CRuntime,
    /// Whether to link C runtime statically
    pub static_crt: bool,
    /// Default relocation model
    pub relocation_model: RelocationModel,
    /// Default code model
    pub code_model: CodeModel,
    /// Thread-local storage model
    pub tls_model: TlsModel,
}

impl Default for EnvSpec {
    fn default() -> Self {
        Self {
            env: Environment::Gnu,
            crt: CRuntime::default(),
            static_crt: false,
            relocation_model: RelocationModel::Pic,
            code_model: CodeModel::Small,
            tls_model: TlsModel::GeneralDynamic,
        }
    }
}

/// C runtime library selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum CRuntime {
    /// GNU libc
    #[default]
    Glibc,
    /// musl libc
    Musl,
    /// Microsoft Visual C++ runtime
    Msvcrt,
    /// UCRT (Universal C Runtime)
    Ucrt,
    /// Newlib (embedded)
    Newlib,
    /// No C runtime
    None,
}

/// Code relocation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum RelocationModel {
    /// Position-independent code
    #[default]
    Pic,
    /// Position-independent executable
    Pie,
    /// Static (no relocation)
    Static,
    /// Dynamic, no PIC
    DynamicNoPic,
    /// Read-only position independence
    Ropi,
    /// Read-write position independence
    Rwpi,
    /// Combined ROPI and RWPI
    RopiRwpi,
}

/// Code model for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum CodeModel {
    /// Tiny code model (very limited address range)
    Tiny,
    /// Small code model (default, 2GB address range)
    #[default]
    Small,
    /// Kernel code model (high addresses)
    Kernel,
    /// Medium code model (large data)
    Medium,
    /// Large code model (unlimited address range)
    Large,
}

/// Thread-local storage model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum TlsModel {
    /// General dynamic (most compatible)
    #[default]
    GeneralDynamic,
    /// Local dynamic (multiple TLS variables)
    LocalDynamic,
    /// Initial exec (loaded at startup)
    InitialExec,
    /// Local exec (same module)
    LocalExec,
}

/// Target CPU features configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TargetFeatures {
    /// Explicitly enabled features
    pub enabled: HashSet<String>,
    /// Explicitly disabled features
    pub disabled: HashSet<String>,
    /// Features implied by the target
    pub implied: HashSet<String>,
    /// Feature dependencies (feature -> required features)
    pub dependencies: HashMap<String, Vec<String>>,
}

impl TargetFeatures {
    /// Create empty feature set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable a feature.
    pub fn enable(&mut self, feature: &str) {
        self.enabled.insert(feature.to_string());
        self.disabled.remove(feature);
    }

    /// Disable a feature.
    pub fn disable(&mut self, feature: &str) {
        self.disabled.insert(feature.to_string());
        self.enabled.remove(feature);
    }

    /// Check if a feature is enabled.
    pub fn is_enabled(&self, feature: &str) -> bool {
        (self.enabled.contains(feature) || self.implied.contains(feature))
            && !self.disabled.contains(feature)
    }

    /// Get all effective features.
    pub fn effective_features(&self) -> HashSet<String> {
        let mut result: HashSet<String> = self.enabled.union(&self.implied).cloned().collect();
        for f in &self.disabled {
            result.remove(f);
        }
        result
    }

    /// Resolve feature dependencies.
    pub fn resolve_dependencies(&mut self) {
        let mut to_enable = Vec::new();

        for feature in &self.enabled {
            if let Some(deps) = self.dependencies.get(feature) {
                to_enable.extend(deps.clone());
            }
        }

        for dep in to_enable {
            self.enabled.insert(dep);
        }
    }
}

/// Linker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkerSpec {
    /// Linker flavor (gnu, gcc, msvc, lld, wasm-ld)
    pub flavor: LinkerFlavor,
    /// Linker executable path (if not using default)
    pub path: Option<PathBuf>,
    /// Extra linker arguments
    pub args: Vec<String>,
    /// Pre-link arguments (before objects)
    pub pre_link_args: Vec<String>,
    /// Post-link arguments (after objects)
    pub post_link_args: Vec<String>,
    /// Libraries to link
    pub libraries: Vec<String>,
    /// Library search paths
    pub library_paths: Vec<PathBuf>,
    /// Linker script path
    pub script: Option<PathBuf>,
    /// Whether to use LTO
    pub lto: bool,
    /// LTO mode (thin or fat)
    pub lto_mode: LtoMode,
}

impl Default for LinkerSpec {
    fn default() -> Self {
        Self {
            flavor: LinkerFlavor::Gcc,
            path: None,
            args: Vec::new(),
            pre_link_args: Vec::new(),
            post_link_args: Vec::new(),
            libraries: Vec::new(),
            library_paths: Vec::new(),
            script: None,
            lto: false,
            lto_mode: LtoMode::Thin,
        }
    }
}

/// Linker flavor/type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum LinkerFlavor {
    /// GNU ld
    Gnu,
    /// GCC (uses system linker)
    #[default]
    Gcc,
    /// Microsoft link.exe
    Msvc,
    /// LLVM lld (generic)
    Lld,
    /// LLVM lld (ELF flavor)
    LldLink,
    /// LLVM lld (COFF flavor)
    LldCoff,
    /// LLVM lld (Mach-O flavor)
    LldMacho,
    /// LLVM lld (Wasm flavor)
    WasmLd,
    /// macOS linker
    Darwin,
    /// Emscripten
    Em,
    /// PTX assembler
    Ptx,
    /// BPF linker
    Bpf,
}

/// LTO (Link-Time Optimization) mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum LtoMode {
    /// Thin LTO (faster, parallel)
    #[default]
    Thin,
    /// Fat LTO (slower, better optimization)
    Fat,
    /// No LTO
    Off,
}

/// Code generation options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodegenSpec {
    /// Optimization level (0-3)
    pub opt_level: OptLevel,
    /// Debug info level
    pub debug_info: DebugInfo,
    /// Whether to generate debug assertions
    pub debug_assertions: bool,
    /// Whether to enable overflow checks
    pub overflow_checks: bool,
    /// Whether to emit LLVM IR
    pub emit_llvm_ir: bool,
    /// Whether to emit assembly
    pub emit_asm: bool,
    /// Inline threshold
    pub inline_threshold: Option<u32>,
    /// Whether to use frame pointers
    pub frame_pointer: FramePointer,
    /// Strip symbols
    pub strip: StripMode,
}

impl Default for CodegenSpec {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            debug_info: DebugInfo::None,
            debug_assertions: false,
            overflow_checks: true,
            emit_llvm_ir: false,
            emit_asm: false,
            inline_threshold: None,
            frame_pointer: FramePointer::MayOmit,
            strip: StripMode::None,
        }
    }
}

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OptLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Default optimization
    #[default]
    O2,
    /// Aggressive optimization
    O3,
    /// Optimize for size
    Os,
    /// Aggressively optimize for size
    Oz,
}

/// Debug information level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum DebugInfo {
    /// No debug info
    #[default]
    None,
    /// Line tables only
    LineTablesOnly,
    /// Limited debug info
    Limited,
    /// Full debug info
    Full,
}

/// Frame pointer behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum FramePointer {
    /// Always keep frame pointer
    Always,
    /// Only keep for non-leaf functions
    NonLeaf,
    /// May omit frame pointer
    #[default]
    MayOmit,
}

/// Symbol stripping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum StripMode {
    /// Don't strip
    #[default]
    None,
    /// Strip debug info only
    Debuginfo,
    /// Strip all symbols
    Symbols,
}

/// Additional target options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TargetOptions {
    /// Whether this is a builtin target
    pub is_builtin: bool,
    /// Default calling convention
    pub default_calling_convention: Option<String>,
    /// Whether the target supports atomics
    pub max_atomic_width: Option<u32>,
    /// Whether the target supports 128-bit integers
    pub supports_i128: bool,
    /// Whether the target has native floating point
    pub has_float: bool,
    /// Minimum alignment for functions
    pub function_alignment: Option<u32>,
    /// Default visibility
    pub default_visibility: Visibility,
    /// Whether to use LLVM's built-in CRT
    pub use_llvm_crt: bool,
    /// Whether the target supports thread-local storage
    pub has_tls: bool,
    /// Whether this target requires cross-compilation
    pub requires_cross: bool,
    /// Custom LLVM target name
    pub llvm_target: Option<String>,
    /// Extra metadata
    pub metadata: HashMap<String, String>,
}

/// Symbol visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    #[default]
    Default,
    Hidden,
    Protected,
}

/// Complete target specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSpec {
    /// Target triple
    pub triple: TargetTriple,
    /// Architecture specification
    pub arch: ArchSpec,
    /// OS specification
    pub os: OsSpec,
    /// Environment specification
    pub env: EnvSpec,
    /// Target features
    pub features: TargetFeatures,
    /// Linker specification
    pub linker: LinkerSpec,
    /// Codegen specification
    pub codegen: CodegenSpec,
    /// Additional options
    pub options: TargetOptions,
}

impl TargetSpec {
    /// Create a new target specification from a triple.
    pub fn new(triple: TargetTriple) -> Self {
        let arch = Architecture::parse(&triple.arch);
        let os = OperatingSystem::parse(&triple.os);
        let env = triple
            .env
            .as_ref()
            .map(|e| Environment::parse(e))
            .unwrap_or(Environment::None);

        Self {
            triple,
            arch: ArchSpec::new(arch),
            os: OsSpec {
                os,
                ..Default::default()
            },
            env: EnvSpec {
                env,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec::default(),
            codegen: CodegenSpec::default(),
            options: TargetOptions::default(),
        }
    }

    /// Parse a target specification from a triple string.
    pub fn from_triple(triple: &str) -> TargetResult<Self> {
        let triple = TargetTriple::parse(triple)?;
        Ok(Self::new(triple))
    }

    /// Load a target specification from a JSON file.
    pub fn from_file(path: &std::path::Path) -> TargetResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let spec: Self = serde_json::from_str(&content)?;
        Ok(spec)
    }

    /// Save target specification to a JSON file.
    pub fn to_file(&self, path: &std::path::Path) -> TargetResult<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get the LLVM target triple string.
    pub fn llvm_target(&self) -> String {
        self.options
            .llvm_target
            .clone()
            .unwrap_or_else(|| self.triple.to_string())
    }

    /// Get the data layout string.
    pub fn data_layout(&self) -> &str {
        self.arch.arch.data_layout()
    }

    /// Get pointer width in bits.
    pub fn pointer_width(&self) -> u32 {
        self.arch.arch.pointer_width()
    }

    /// Check if this is a 64-bit target.
    pub fn is_64bit(&self) -> bool {
        self.pointer_width() == 64
    }

    /// Check if this requires cross-compilation from the host.
    pub fn requires_cross(&self) -> bool {
        self.options.requires_cross || self.triple.to_string() != host_triple()
    }

    /// Get the executable file extension.
    pub fn exe_suffix(&self) -> &str {
        self.os.os.exe_extension()
    }

    /// Get the dynamic library extension.
    pub fn dylib_suffix(&self) -> &str {
        self.os.os.dylib_extension()
    }

    /// Get the static library extension.
    pub fn staticlib_suffix(&self) -> &str {
        self.os.os.staticlib_extension()
    }

    /// Apply overrides from another partial spec.
    pub fn apply_overrides(&mut self, overrides: &TargetOverrides) {
        if let Some(ref cpu) = overrides.cpu {
            self.arch.cpu = cpu.clone();
        }
        if let Some(ref features) = overrides.features {
            for f in features {
                if f.starts_with('-') {
                    self.features.disable(&f[1..]);
                } else if f.starts_with('+') {
                    self.features.enable(&f[1..]);
                } else {
                    self.features.enable(f);
                }
            }
        }
        if let Some(opt) = overrides.opt_level {
            self.codegen.opt_level = opt;
        }
        if let Some(lto) = overrides.lto {
            self.linker.lto = lto;
        }
    }
}

/// Partial overrides for a target specification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TargetOverrides {
    /// CPU model override
    pub cpu: Option<String>,
    /// Feature overrides (+feature or -feature)
    pub features: Option<Vec<String>>,
    /// Optimization level override
    pub opt_level: Option<OptLevel>,
    /// LTO override
    pub lto: Option<bool>,
    /// Extra linker args
    pub linker_args: Option<Vec<String>>,
}

/// Get the host target triple.
pub fn host_triple() -> String {
    // Use compile-time detection
    #[cfg(target_os = "linux")]
    let os = "linux";
    #[cfg(target_os = "windows")]
    let os = "windows";
    #[cfg(target_os = "macos")]
    let os = "macos";
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    let os = "unknown";

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";
    #[cfg(target_arch = "arm")]
    let arch = "arm";
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
    let arch = "unknown";

    #[cfg(target_env = "gnu")]
    let env = Some("gnu");
    #[cfg(target_env = "musl")]
    let env = Some("musl");
    #[cfg(target_env = "msvc")]
    let env = Some("msvc");
    #[cfg(not(any(target_env = "gnu", target_env = "musl", target_env = "msvc")))]
    let env = None::<&str>;

    match env {
        Some(e) => format!("{}-unknown-{}-{}", arch, os, e),
        None => format!("{}-unknown-{}", arch, os),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_triple_parse() {
        // Full triple
        let triple = TargetTriple::parse("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(triple.arch, "x86_64");
        assert_eq!(triple.vendor, "unknown");
        assert_eq!(triple.os, "linux");
        assert_eq!(triple.env, Some("gnu".to_string()));

        // Three-part triple
        let triple = TargetTriple::parse("aarch64-apple-darwin").unwrap();
        assert_eq!(triple.arch, "aarch64");
        assert_eq!(triple.vendor, "apple");
        assert_eq!(triple.os, "darwin");
        assert_eq!(triple.env, None);

        // WASM triple
        let triple = TargetTriple::parse("wasm32-unknown-wasi").unwrap();
        assert_eq!(triple.arch, "wasm32");
        assert!(triple.is_wasm());
    }

    #[test]
    fn test_architecture_parse() {
        assert_eq!(Architecture::parse("x86_64"), Architecture::X86_64);
        assert_eq!(Architecture::parse("aarch64"), Architecture::Aarch64);
        assert_eq!(Architecture::parse("arm64"), Architecture::Aarch64);
        assert_eq!(Architecture::parse("riscv64gc"), Architecture::Riscv64);
        assert_eq!(Architecture::parse("wasm32"), Architecture::Wasm32);
    }

    #[test]
    fn test_architecture_properties() {
        assert_eq!(Architecture::X86_64.pointer_width(), 64);
        assert_eq!(Architecture::X86.pointer_width(), 32);
        assert_eq!(Architecture::Wasm32.pointer_width(), 32);
        assert_eq!(Architecture::Aarch64.pointer_width(), 64);
    }

    #[test]
    fn test_os_properties() {
        assert!(OperatingSystem::Linux.is_unix_like());
        assert!(OperatingSystem::MacOs.is_unix_like());
        assert!(!OperatingSystem::Windows.is_unix_like());

        assert_eq!(OperatingSystem::Windows.exe_extension(), ".exe");
        assert_eq!(OperatingSystem::Linux.exe_extension(), "");
        assert_eq!(OperatingSystem::Wasi.exe_extension(), ".wasm");
    }

    #[test]
    fn test_target_spec_creation() {
        let spec = TargetSpec::from_triple("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(spec.arch.arch, Architecture::X86_64);
        assert_eq!(spec.os.os, OperatingSystem::Linux);
        assert_eq!(spec.env.env, Environment::Gnu);
        assert!(spec.is_64bit());
    }

    #[test]
    fn test_target_features() {
        let mut features = TargetFeatures::new();
        features.enable("avx2");
        features.enable("sse4.2");
        features.disable("avx512f");

        assert!(features.is_enabled("avx2"));
        assert!(features.is_enabled("sse4.2"));
        assert!(!features.is_enabled("avx512f"));
    }

    #[test]
    fn test_arch_spec_llvm_features() {
        let spec = ArchSpec::new(Architecture::X86_64)
            .with_cpu("skylake")
            .with_feature("avx2")
            .with_feature("fma")
            .without_feature("avx512f");

        let features = spec.llvm_features();
        assert!(features.contains("+avx2"));
        assert!(features.contains("+fma"));
        assert!(features.contains("-avx512f"));
    }

    #[test]
    fn test_target_overrides() {
        let mut spec = TargetSpec::from_triple("x86_64-unknown-linux-gnu").unwrap();
        let overrides = TargetOverrides {
            cpu: Some("skylake".to_string()),
            features: Some(vec!["+avx2".to_string(), "-sse3".to_string()]),
            opt_level: Some(OptLevel::O3),
            lto: Some(true),
            linker_args: None,
        };

        spec.apply_overrides(&overrides);

        assert_eq!(spec.arch.cpu, "skylake");
        assert!(spec.features.is_enabled("avx2"));
        assert!(!spec.features.is_enabled("sse3"));
        assert_eq!(spec.codegen.opt_level, OptLevel::O3);
        assert!(spec.linker.lto);
    }

    #[test]
    fn test_target_triple_helpers() {
        let linux = TargetTriple::parse("x86_64-unknown-linux-gnu").unwrap();
        assert!(linux.is_linux());
        assert!(!linux.is_windows());
        assert!(!linux.is_bare_metal());

        let windows = TargetTriple::parse("x86_64-pc-windows-msvc").unwrap();
        assert!(windows.is_windows());
        assert!(!windows.is_linux());

        let bare = TargetTriple::parse("thumbv7m-none-eabi").unwrap();
        assert!(bare.is_bare_metal());
    }

    #[test]
    fn test_host_triple() {
        let triple = host_triple();
        // Should at least produce a valid triple
        assert!(triple.contains('-'));
        let parsed = TargetTriple::parse(&triple);
        assert!(parsed.is_ok());
    }
}
