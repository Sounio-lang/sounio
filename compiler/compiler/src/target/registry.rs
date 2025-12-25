//! Target Registry
//!
//! This module provides a registry of built-in and custom target specifications.
//! It includes predefined targets for common platforms and allows loading custom
//! target specifications from files.

use super::spec::{
    ArchSpec, Architecture, CRuntime, CodeModel, CodegenSpec, EnvSpec, Environment, LinkerFlavor,
    LinkerSpec, OperatingSystem, OsSpec, PanicStrategy, RelocationModel, TargetFeatures,
    TargetOptions, TargetResult, TargetSpec, TargetSpecError, TargetTriple,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Global target registry.
static REGISTRY: RwLock<Option<TargetRegistry>> = RwLock::new(None);

/// Get the global target registry, initializing it if necessary.
pub fn global_registry() -> Arc<TargetRegistry> {
    {
        let guard = REGISTRY.read().unwrap();
        if let Some(ref registry) = *guard {
            return Arc::new(registry.clone());
        }
    }

    let mut guard = REGISTRY.write().unwrap();
    if guard.is_none() {
        *guard = Some(TargetRegistry::with_builtins());
    }
    Arc::new(guard.clone().unwrap())
}

/// Registry of target specifications.
#[derive(Debug, Clone)]
pub struct TargetRegistry {
    /// Built-in targets
    builtins: HashMap<String, TargetSpec>,
    /// Custom targets loaded from files
    custom: HashMap<String, TargetSpec>,
    /// Search paths for custom target files
    search_paths: Vec<PathBuf>,
}

impl Default for TargetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TargetRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            builtins: HashMap::new(),
            custom: HashMap::new(),
            search_paths: Vec::new(),
        }
    }

    /// Create a registry with all built-in targets.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register_builtins();
        registry
    }

    /// Register all built-in targets.
    fn register_builtins(&mut self) {
        // Linux targets
        self.register_builtin(targets::x86_64_unknown_linux_gnu());
        self.register_builtin(targets::x86_64_unknown_linux_musl());
        self.register_builtin(targets::i686_unknown_linux_gnu());
        self.register_builtin(targets::aarch64_unknown_linux_gnu());
        self.register_builtin(targets::aarch64_unknown_linux_musl());
        self.register_builtin(targets::arm_unknown_linux_gnueabihf());
        self.register_builtin(targets::riscv64gc_unknown_linux_gnu());

        // macOS targets
        self.register_builtin(targets::x86_64_apple_darwin());
        self.register_builtin(targets::aarch64_apple_darwin());

        // Windows targets
        self.register_builtin(targets::x86_64_pc_windows_msvc());
        self.register_builtin(targets::x86_64_pc_windows_gnu());
        self.register_builtin(targets::i686_pc_windows_msvc());
        self.register_builtin(targets::i686_pc_windows_gnu());
        self.register_builtin(targets::aarch64_pc_windows_msvc());

        // WebAssembly targets
        self.register_builtin(targets::wasm32_unknown_unknown());
        self.register_builtin(targets::wasm32_wasi());
        self.register_builtin(targets::wasm64_unknown_unknown());

        // Embedded/bare metal targets
        self.register_builtin(targets::thumbv7m_none_eabi());
        self.register_builtin(targets::thumbv7em_none_eabihf());
        self.register_builtin(targets::riscv32imac_unknown_none_elf());
        self.register_builtin(targets::riscv64gc_unknown_none_elf());
        self.register_builtin(targets::aarch64_unknown_none());

        // GPU targets
        self.register_builtin(targets::nvptx64_nvidia_cuda());
        self.register_builtin(targets::amdgcn_amd_amdhsa());

        // BSD targets
        self.register_builtin(targets::x86_64_unknown_freebsd());
        self.register_builtin(targets::x86_64_unknown_netbsd());
        self.register_builtin(targets::x86_64_unknown_openbsd());

        // Android targets
        self.register_builtin(targets::aarch64_linux_android());
        self.register_builtin(targets::x86_64_linux_android());
    }

    /// Register a built-in target.
    fn register_builtin(&mut self, spec: TargetSpec) {
        let name = spec.triple.to_string();
        self.builtins.insert(name, spec);
    }

    /// Add a search path for custom targets.
    pub fn add_search_path(&mut self, path: PathBuf) {
        if !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
    }

    /// Look up a target by name.
    pub fn get(&self, name: &str) -> TargetResult<TargetSpec> {
        // Check custom targets first
        if let Some(spec) = self.custom.get(name) {
            return Ok(spec.clone());
        }

        // Check built-ins
        if let Some(spec) = self.builtins.get(name) {
            return Ok(spec.clone());
        }

        // Try to load from file
        if let Ok(spec) = self.load_from_file(name) {
            return Ok(spec);
        }

        // Try to parse as a triple
        if let Ok(spec) = TargetSpec::from_triple(name) {
            return Ok(spec);
        }

        Err(TargetSpecError::TargetNotFound(name.to_string()))
    }

    /// Try to load a target from a file.
    fn load_from_file(&self, name: &str) -> TargetResult<TargetSpec> {
        // Check if name is a path
        let path = Path::new(name);
        if path.exists() {
            return TargetSpec::from_file(path);
        }

        // Search in search paths
        let filename = format!("{}.json", name);
        for search_path in &self.search_paths {
            let path = search_path.join(&filename);
            if path.exists() {
                return TargetSpec::from_file(&path);
            }
        }

        Err(TargetSpecError::TargetNotFound(name.to_string()))
    }

    /// Register a custom target.
    pub fn register(&mut self, spec: TargetSpec) {
        let name = spec.triple.to_string();
        self.custom.insert(name, spec);
    }

    /// Register a custom target from a JSON file.
    pub fn register_from_file(&mut self, path: &Path) -> TargetResult<()> {
        let spec = TargetSpec::from_file(path)?;
        self.register(spec);
        Ok(())
    }

    /// List all available targets.
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<_> = self
            .builtins
            .keys()
            .chain(self.custom.keys())
            .map(|s| s.as_str())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// List only built-in targets.
    pub fn list_builtins(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.builtins.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// List only custom targets.
    pub fn list_custom(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.custom.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Get targets matching a filter.
    pub fn filter<F>(&self, predicate: F) -> Vec<&TargetSpec>
    where
        F: Fn(&TargetSpec) -> bool,
    {
        self.builtins
            .values()
            .chain(self.custom.values())
            .filter(|spec| predicate(spec))
            .collect()
    }

    /// Get all targets for a specific OS.
    pub fn targets_for_os(&self, os: OperatingSystem) -> Vec<&TargetSpec> {
        self.filter(|spec| spec.os.os == os)
    }

    /// Get all targets for a specific architecture.
    pub fn targets_for_arch(&self, arch: Architecture) -> Vec<&TargetSpec> {
        self.filter(|spec| spec.arch.arch == arch)
    }

    /// Check if a target exists.
    pub fn has(&self, name: &str) -> bool {
        self.builtins.contains_key(name) || self.custom.contains_key(name)
    }
}

/// Built-in target definitions.
pub mod targets {
    use super::*;

    // ========================================================================
    // Linux targets
    // ========================================================================

    /// x86_64-unknown-linux-gnu
    pub fn x86_64_unknown_linux_gnu() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("x86_64", "unknown", "linux", Some("gnu")),
            arch: ArchSpec {
                arch: Architecture::X86_64,
                cpu: "x86-64".to_string(),
                features: ["sse2".to_string()].into_iter().collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Linux,
                requires_pie: true,
                stack_protector: true,
                panic_strategy: PanicStrategy::Unwind,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Gnu,
                crt: CRuntime::Glibc,
                relocation_model: RelocationModel::Pic,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    /// x86_64-unknown-linux-musl
    pub fn x86_64_unknown_linux_musl() -> TargetSpec {
        let mut spec = x86_64_unknown_linux_gnu();
        spec.triple = TargetTriple::new("x86_64", "unknown", "linux", Some("musl"));
        spec.env.env = Environment::Musl;
        spec.env.crt = CRuntime::Musl;
        spec.env.static_crt = true;
        spec
    }

    /// i686-unknown-linux-gnu
    pub fn i686_unknown_linux_gnu() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("i686", "unknown", "linux", Some("gnu")),
            arch: ArchSpec {
                arch: Architecture::X86,
                cpu: "pentium4".to_string(),
                features: ["sse2".to_string()].into_iter().collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Linux,
                requires_pie: true,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Gnu,
                crt: CRuntime::Glibc,
                relocation_model: RelocationModel::Pic,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: false,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    /// aarch64-unknown-linux-gnu
    pub fn aarch64_unknown_linux_gnu() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("aarch64", "unknown", "linux", Some("gnu")),
            arch: ArchSpec {
                arch: Architecture::Aarch64,
                cpu: "generic".to_string(),
                features: ["neon".to_string()].into_iter().collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Linux,
                requires_pie: true,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Gnu,
                crt: CRuntime::Glibc,
                relocation_model: RelocationModel::Pic,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(128),
                ..Default::default()
            },
        }
    }

    /// aarch64-unknown-linux-musl
    pub fn aarch64_unknown_linux_musl() -> TargetSpec {
        let mut spec = aarch64_unknown_linux_gnu();
        spec.triple = TargetTriple::new("aarch64", "unknown", "linux", Some("musl"));
        spec.env.env = Environment::Musl;
        spec.env.crt = CRuntime::Musl;
        spec.env.static_crt = true;
        spec
    }

    /// arm-unknown-linux-gnueabihf
    pub fn arm_unknown_linux_gnueabihf() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("arm", "unknown", "linux", Some("gnueabihf")),
            arch: ArchSpec {
                arch: Architecture::Arm,
                cpu: "generic".to_string(),
                features: ["vfp3".to_string(), "neon".to_string()]
                    .into_iter()
                    .collect(),
                soft_float: false,
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Linux,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Gnueabihf,
                crt: CRuntime::Glibc,
                relocation_model: RelocationModel::Pic,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: false,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    /// riscv64gc-unknown-linux-gnu
    pub fn riscv64gc_unknown_linux_gnu() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("riscv64gc", "unknown", "linux", Some("gnu")),
            arch: ArchSpec {
                arch: Architecture::Riscv64,
                cpu: "generic-rv64".to_string(),
                features: [
                    "m".to_string(),
                    "a".to_string(),
                    "f".to_string(),
                    "d".to_string(),
                    "c".to_string(),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Linux,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Gnu,
                crt: CRuntime::Glibc,
                relocation_model: RelocationModel::Pic,
                code_model: CodeModel::Medium,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(64),
                llvm_target: Some("riscv64-unknown-linux-gnu".to_string()),
                ..Default::default()
            },
        }
    }

    // ========================================================================
    // macOS targets
    // ========================================================================

    /// x86_64-apple-darwin
    pub fn x86_64_apple_darwin() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("x86_64", "apple", "darwin", None),
            arch: ArchSpec {
                arch: Architecture::X86_64,
                cpu: "core2".to_string(),
                features: ["sse3".to_string(), "ssse3".to_string()]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::MacOs,
                min_version: Some("10.12".to_string()),
                requires_pie: true,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Macho,
                relocation_model: RelocationModel::Pic,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Darwin,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    /// aarch64-apple-darwin (Apple Silicon)
    pub fn aarch64_apple_darwin() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("aarch64", "apple", "darwin", None),
            arch: ArchSpec {
                arch: Architecture::Aarch64,
                cpu: "apple-m1".to_string(),
                features: ["neon".to_string(), "fp-armv8".to_string()]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::MacOs,
                min_version: Some("11.0".to_string()),
                requires_pie: true,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Macho,
                relocation_model: RelocationModel::Pic,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Darwin,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(128),
                ..Default::default()
            },
        }
    }

    // ========================================================================
    // Windows targets
    // ========================================================================

    /// x86_64-pc-windows-msvc
    pub fn x86_64_pc_windows_msvc() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("x86_64", "pc", "windows", Some("msvc")),
            arch: ArchSpec {
                arch: Architecture::X86_64,
                cpu: "x86-64".to_string(),
                features: ["sse2".to_string()].into_iter().collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Windows,
                panic_strategy: PanicStrategy::Unwind,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Msvc,
                crt: CRuntime::Msvcrt,
                relocation_model: RelocationModel::Pic,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Msvc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    /// x86_64-pc-windows-gnu (MinGW)
    pub fn x86_64_pc_windows_gnu() -> TargetSpec {
        let mut spec = x86_64_pc_windows_msvc();
        spec.triple = TargetTriple::new("x86_64", "pc", "windows", Some("gnu"));
        spec.env.env = Environment::Gnu;
        spec.env.crt = CRuntime::Glibc; // MinGW runtime
        spec.linker.flavor = LinkerFlavor::Gcc;
        spec
    }

    /// i686-pc-windows-msvc
    pub fn i686_pc_windows_msvc() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("i686", "pc", "windows", Some("msvc")),
            arch: ArchSpec {
                arch: Architecture::X86,
                cpu: "pentium4".to_string(),
                features: ["sse2".to_string()].into_iter().collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Windows,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Msvc,
                crt: CRuntime::Msvcrt,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Msvc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: false,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    /// i686-pc-windows-gnu
    pub fn i686_pc_windows_gnu() -> TargetSpec {
        let mut spec = i686_pc_windows_msvc();
        spec.triple = TargetTriple::new("i686", "pc", "windows", Some("gnu"));
        spec.env.env = Environment::Gnu;
        spec.linker.flavor = LinkerFlavor::Gcc;
        spec
    }

    /// aarch64-pc-windows-msvc
    pub fn aarch64_pc_windows_msvc() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("aarch64", "pc", "windows", Some("msvc")),
            arch: ArchSpec {
                arch: Architecture::Aarch64,
                cpu: "generic".to_string(),
                features: ["neon".to_string()].into_iter().collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Windows,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Msvc,
                crt: CRuntime::Msvcrt,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Msvc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    // ========================================================================
    // WebAssembly targets
    // ========================================================================

    /// wasm32-unknown-unknown
    pub fn wasm32_unknown_unknown() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("wasm32", "unknown", "unknown", None),
            arch: ArchSpec {
                arch: Architecture::Wasm32,
                cpu: "generic".to_string(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::None,
                panic_strategy: PanicStrategy::Abort,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::None,
                crt: CRuntime::None,
                relocation_model: RelocationModel::Static,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::WasmLd,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: false,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    /// wasm32-wasi
    pub fn wasm32_wasi() -> TargetSpec {
        let mut spec = wasm32_unknown_unknown();
        spec.triple = TargetTriple::new("wasm32", "wasi", "wasi", None);
        spec.os.os = OperatingSystem::Wasi;
        spec.os.panic_strategy = PanicStrategy::Abort;
        spec
    }

    /// wasm64-unknown-unknown
    pub fn wasm64_unknown_unknown() -> TargetSpec {
        let mut spec = wasm32_unknown_unknown();
        spec.triple = TargetTriple::new("wasm64", "unknown", "unknown", None);
        spec.arch.arch = Architecture::Wasm64;
        spec
    }

    // ========================================================================
    // Embedded / Bare metal targets
    // ========================================================================

    /// thumbv7m-none-eabi (Cortex-M3)
    pub fn thumbv7m_none_eabi() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("thumbv7m", "none", "none", Some("eabi")),
            arch: ArchSpec {
                arch: Architecture::Arm,
                cpu: "cortex-m3".to_string(),
                features: ["thumb-mode".to_string()].into_iter().collect(),
                soft_float: true,
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::None,
                panic_strategy: PanicStrategy::Abort,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Eabi,
                crt: CRuntime::Newlib,
                relocation_model: RelocationModel::Static,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: false,
                has_float: false,
                has_tls: false,
                max_atomic_width: Some(32),
                llvm_target: Some("thumbv7m-none-eabi".to_string()),
                ..Default::default()
            },
        }
    }

    /// thumbv7em-none-eabihf (Cortex-M4/M7 with FPU)
    pub fn thumbv7em_none_eabihf() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("thumbv7em", "none", "none", Some("eabihf")),
            arch: ArchSpec {
                arch: Architecture::Arm,
                cpu: "cortex-m4".to_string(),
                features: [
                    "thumb-mode".to_string(),
                    "dsp".to_string(),
                    "vfp4".to_string(),
                ]
                .into_iter()
                .collect(),
                soft_float: false,
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::None,
                panic_strategy: PanicStrategy::Abort,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Eabihf,
                crt: CRuntime::Newlib,
                relocation_model: RelocationModel::Static,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: false,
                has_float: true,
                has_tls: false,
                max_atomic_width: Some(32),
                llvm_target: Some("thumbv7em-none-eabihf".to_string()),
                ..Default::default()
            },
        }
    }

    /// riscv32imac-unknown-none-elf
    pub fn riscv32imac_unknown_none_elf() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("riscv32imac", "unknown", "none", Some("elf")),
            arch: ArchSpec {
                arch: Architecture::Riscv32,
                cpu: "generic-rv32".to_string(),
                features: ["m".to_string(), "a".to_string(), "c".to_string()]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::None,
                panic_strategy: PanicStrategy::Abort,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::None,
                crt: CRuntime::None,
                relocation_model: RelocationModel::Static,
                code_model: CodeModel::Medium,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: false,
                has_float: false,
                has_tls: false,
                max_atomic_width: Some(32),
                llvm_target: Some("riscv32-unknown-none-elf".to_string()),
                ..Default::default()
            },
        }
    }

    /// riscv64gc-unknown-none-elf
    pub fn riscv64gc_unknown_none_elf() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("riscv64gc", "unknown", "none", Some("elf")),
            arch: ArchSpec {
                arch: Architecture::Riscv64,
                cpu: "generic-rv64".to_string(),
                features: [
                    "m".to_string(),
                    "a".to_string(),
                    "f".to_string(),
                    "d".to_string(),
                    "c".to_string(),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::None,
                panic_strategy: PanicStrategy::Abort,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::None,
                crt: CRuntime::None,
                relocation_model: RelocationModel::Static,
                code_model: CodeModel::Medium,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: false,
                max_atomic_width: Some(64),
                llvm_target: Some("riscv64-unknown-none-elf".to_string()),
                ..Default::default()
            },
        }
    }

    /// aarch64-unknown-none (bare metal AArch64)
    pub fn aarch64_unknown_none() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("aarch64", "unknown", "none", None),
            arch: ArchSpec {
                arch: Architecture::Aarch64,
                cpu: "generic".to_string(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::None,
                panic_strategy: PanicStrategy::Abort,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::None,
                crt: CRuntime::None,
                relocation_model: RelocationModel::Static,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: false,
                max_atomic_width: Some(128),
                ..Default::default()
            },
        }
    }

    // ========================================================================
    // GPU targets
    // ========================================================================

    /// nvptx64-nvidia-cuda (NVIDIA CUDA)
    pub fn nvptx64_nvidia_cuda() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("nvptx64", "nvidia", "cuda", None),
            arch: ArchSpec {
                arch: Architecture::Nvptx64,
                cpu: "sm_30".to_string(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::None,
                panic_strategy: PanicStrategy::Abort,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::None,
                crt: CRuntime::None,
                relocation_model: RelocationModel::Static,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Ptx,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: false,
                has_float: true,
                has_tls: false,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    /// amdgcn-amd-amdhsa (AMD GPU)
    pub fn amdgcn_amd_amdhsa() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("amdgcn", "amd", "amdhsa", None),
            arch: ArchSpec {
                arch: Architecture::Amdgcn,
                cpu: "gfx900".to_string(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::None,
                panic_strategy: PanicStrategy::Abort,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::None,
                crt: CRuntime::None,
                relocation_model: RelocationModel::Static,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Lld,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: false,
                has_float: true,
                has_tls: false,
                max_atomic_width: Some(64),
                ..Default::default()
            },
        }
    }

    // ========================================================================
    // BSD targets
    // ========================================================================

    /// x86_64-unknown-freebsd
    pub fn x86_64_unknown_freebsd() -> TargetSpec {
        let mut spec = x86_64_unknown_linux_gnu();
        spec.triple = TargetTriple::new("x86_64", "unknown", "freebsd", None);
        spec.os.os = OperatingSystem::FreeBsd;
        spec
    }

    /// x86_64-unknown-netbsd
    pub fn x86_64_unknown_netbsd() -> TargetSpec {
        let mut spec = x86_64_unknown_linux_gnu();
        spec.triple = TargetTriple::new("x86_64", "unknown", "netbsd", None);
        spec.os.os = OperatingSystem::NetBsd;
        spec
    }

    /// x86_64-unknown-openbsd
    pub fn x86_64_unknown_openbsd() -> TargetSpec {
        let mut spec = x86_64_unknown_linux_gnu();
        spec.triple = TargetTriple::new("x86_64", "unknown", "openbsd", None);
        spec.os.os = OperatingSystem::OpenBsd;
        spec
    }

    // ========================================================================
    // Android targets
    // ========================================================================

    /// aarch64-linux-android
    pub fn aarch64_linux_android() -> TargetSpec {
        TargetSpec {
            triple: TargetTriple::new("aarch64", "linux", "android", None),
            arch: ArchSpec {
                arch: Architecture::Aarch64,
                cpu: "generic".to_string(),
                features: ["neon".to_string()].into_iter().collect(),
                ..Default::default()
            },
            os: OsSpec {
                os: OperatingSystem::Android,
                min_version: Some("21".to_string()), // API level
                requires_pie: true,
                ..Default::default()
            },
            env: EnvSpec {
                env: Environment::Android,
                relocation_model: RelocationModel::Pic,
                ..Default::default()
            },
            features: TargetFeatures::new(),
            linker: LinkerSpec {
                flavor: LinkerFlavor::Gcc,
                ..Default::default()
            },
            codegen: CodegenSpec::default(),
            options: TargetOptions {
                is_builtin: true,
                supports_i128: true,
                has_float: true,
                has_tls: true,
                max_atomic_width: Some(128),
                ..Default::default()
            },
        }
    }

    /// x86_64-linux-android
    pub fn x86_64_linux_android() -> TargetSpec {
        let mut spec = aarch64_linux_android();
        spec.triple = TargetTriple::new("x86_64", "linux", "android", None);
        spec.arch = ArchSpec {
            arch: Architecture::X86_64,
            cpu: "x86-64".to_string(),
            features: ["sse4.2".to_string(), "popcnt".to_string()]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        spec.options.max_atomic_width = Some(64);
        spec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_with_builtins() {
        let registry = TargetRegistry::with_builtins();
        assert!(registry.has("x86_64-unknown-linux-gnu"));
        assert!(registry.has("aarch64-apple-darwin"));
        assert!(registry.has("wasm32-unknown-unknown"));
    }

    #[test]
    fn test_registry_get() {
        let registry = TargetRegistry::with_builtins();

        let spec = registry.get("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(spec.arch.arch, Architecture::X86_64);
        assert_eq!(spec.os.os, OperatingSystem::Linux);
    }

    #[test]
    fn test_registry_list() {
        let registry = TargetRegistry::with_builtins();
        let targets = registry.list();

        assert!(targets.contains(&"x86_64-unknown-linux-gnu"));
        assert!(targets.contains(&"aarch64-apple-darwin"));
        assert!(targets.len() >= 20); // We have many built-in targets
    }

    #[test]
    fn test_registry_filter() {
        let registry = TargetRegistry::with_builtins();

        let linux_targets = registry.targets_for_os(OperatingSystem::Linux);
        assert!(!linux_targets.is_empty());
        for spec in &linux_targets {
            assert_eq!(spec.os.os, OperatingSystem::Linux);
        }

        let x86_targets = registry.targets_for_arch(Architecture::X86_64);
        assert!(!x86_targets.is_empty());
        for spec in &x86_targets {
            assert_eq!(spec.arch.arch, Architecture::X86_64);
        }
    }

    #[test]
    fn test_registry_custom() {
        let mut registry = TargetRegistry::with_builtins();

        let custom = TargetSpec::from_triple("myarch-myvendor-myos").unwrap();
        let canonical_name = custom.triple.to_string();
        registry.register(custom);

        // The canonical form is arch-vendor-os
        assert!(
            registry.has(&canonical_name),
            "Expected to find target with name: {}",
            canonical_name
        );
    }

    #[test]
    fn test_linux_targets() {
        let spec = targets::x86_64_unknown_linux_gnu();
        assert_eq!(spec.triple.to_string(), "x86_64-unknown-linux-gnu");
        assert!(spec.options.is_builtin);
        assert!(spec.options.has_tls);
    }

    #[test]
    fn test_macos_targets() {
        let spec = targets::aarch64_apple_darwin();
        assert_eq!(spec.triple.vendor, "apple");
        assert_eq!(spec.os.os, OperatingSystem::MacOs);
        assert_eq!(spec.linker.flavor, LinkerFlavor::Darwin);
    }

    #[test]
    fn test_windows_targets() {
        let spec = targets::x86_64_pc_windows_msvc();
        assert_eq!(spec.os.os, OperatingSystem::Windows);
        assert_eq!(spec.env.env, Environment::Msvc);
        assert_eq!(spec.linker.flavor, LinkerFlavor::Msvc);
    }

    #[test]
    fn test_wasm_targets() {
        let spec = targets::wasm32_unknown_unknown();
        assert_eq!(spec.arch.arch, Architecture::Wasm32);
        assert!(!spec.options.has_tls);
        assert_eq!(spec.linker.flavor, LinkerFlavor::WasmLd);
    }

    #[test]
    fn test_embedded_targets() {
        let spec = targets::thumbv7m_none_eabi();
        assert_eq!(spec.os.os, OperatingSystem::None);
        assert_eq!(spec.os.panic_strategy, PanicStrategy::Abort);
        assert!(!spec.options.has_tls);
    }

    #[test]
    fn test_global_registry() {
        let registry = global_registry();
        assert!(registry.has("x86_64-unknown-linux-gnu"));
    }
}
