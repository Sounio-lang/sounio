//! Target machine configuration for LLVM
//!
//! This module handles target triple configuration, target machine creation,
//! and object file generation.

use inkwell::OptimizationLevel;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};

use std::path::Path;

use super::codegen::OptLevel;

/// Initialize all LLVM targets
pub fn initialize_all_targets() {
    Target::initialize_all(&InitializationConfig::default());
}

/// Initialize only the native target
pub fn initialize_native_target() {
    Target::initialize_native(&InitializationConfig::default())
        .expect("Failed to initialize native target");
}

/// Get the native target triple
pub fn native_triple() -> TargetTriple {
    TargetMachine::get_default_triple()
}

/// Target configuration
#[derive(Debug, Clone)]
pub struct TargetConfig {
    /// Target triple string
    pub triple: String,
    /// CPU name (e.g., "generic", "x86-64", "apple-m1")
    pub cpu: String,
    /// CPU features (e.g., "+sse4.2,+avx")
    pub features: String,
    /// Relocation mode
    pub reloc_mode: RelocMode,
    /// Code model
    pub code_model: CodeModel,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            triple: TargetMachine::get_default_triple()
                .as_str()
                .to_string_lossy()
                .to_string(),
            cpu: "generic".to_string(),
            features: String::new(),
            reloc_mode: RelocMode::PIC,
            code_model: CodeModel::Default,
        }
    }
}

impl TargetConfig {
    /// Create config for native target
    pub fn native() -> Self {
        Self::default()
    }

    /// Create config for a specific triple
    pub fn for_triple(triple: &str) -> Self {
        Self {
            triple: triple.to_string(),
            ..Default::default()
        }
    }

    /// Set CPU
    pub fn with_cpu(mut self, cpu: &str) -> Self {
        self.cpu = cpu.to_string();
        self
    }

    /// Set features
    pub fn with_features(mut self, features: &str) -> Self {
        self.features = features.to_string();
        self
    }

    /// Set relocation mode
    pub fn with_reloc_mode(mut self, mode: RelocMode) -> Self {
        self.reloc_mode = mode;
        self
    }

    /// Set code model
    pub fn with_code_model(mut self, model: CodeModel) -> Self {
        self.code_model = model;
        self
    }

    /// Get target triple
    pub fn target_triple(&self) -> TargetTriple {
        TargetTriple::create(&self.triple)
    }

    /// Create target machine from this config
    pub fn create_target_machine(&self, opt_level: OptLevel) -> Result<TargetMachine, String> {
        let triple = self.target_triple();
        let target =
            Target::from_triple(&triple).map_err(|e| format!("Invalid target triple: {}", e))?;

        let opt = opt_level.to_inkwell();

        target
            .create_target_machine(
                &triple,
                &self.cpu,
                &self.features,
                opt,
                self.reloc_mode,
                self.code_model,
            )
            .ok_or_else(|| "Failed to create target machine".to_string())
    }
}

/// Create a target machine for the native platform
pub fn create_native_target_machine(opt_level: OptLevel) -> Result<TargetMachine, String> {
    initialize_native_target();
    TargetConfig::native().create_target_machine(opt_level)
}

/// Create a target machine for a specific triple
pub fn create_target_machine(triple: &str, opt_level: OptLevel) -> Result<TargetMachine, String> {
    initialize_all_targets();
    TargetConfig::for_triple(triple).create_target_machine(opt_level)
}

/// Compile module to object file
pub fn compile_to_object(
    module: &Module,
    target: &TargetMachine,
    output: &Path,
) -> Result<(), String> {
    target
        .write_to_file(module, FileType::Object, output)
        .map_err(|e| format!("Failed to write object file: {}", e))
}

/// Compile module to assembly
pub fn compile_to_asm(
    module: &Module,
    target: &TargetMachine,
    output: &Path,
) -> Result<(), String> {
    target
        .write_to_file(module, FileType::Assembly, output)
        .map_err(|e| format!("Failed to write assembly file: {}", e))
}

/// Compile module to memory buffer
pub fn compile_to_memory(
    module: &Module,
    target: &TargetMachine,
    file_type: FileType,
) -> Result<inkwell::memory_buffer::MemoryBuffer, String> {
    target
        .write_to_memory_buffer(module, file_type)
        .map_err(|e| format!("Failed to compile to memory: {}", e))
}

/// Get object file extension for target
pub fn object_extension(triple: &str) -> &'static str {
    if triple.contains("windows") {
        "obj"
    } else {
        "o"
    }
}

/// Get executable extension for target
pub fn executable_extension(triple: &str) -> &'static str {
    if triple.contains("windows") {
        "exe"
    } else {
        ""
    }
}

/// Get assembly extension for target
pub fn assembly_extension(_triple: &str) -> &'static str {
    "s"
}

/// Get shared library extension for target
pub fn shared_lib_extension(triple: &str) -> &'static str {
    if triple.contains("windows") {
        "dll"
    } else if triple.contains("darwin") || triple.contains("macos") {
        "dylib"
    } else {
        "so"
    }
}

/// Create a target machine configured for shared library compilation (with PIC)
pub fn create_shared_target_machine(
    triple: &str,
    opt_level: OptLevel,
) -> Result<TargetMachine, String> {
    initialize_all_targets();

    let config = TargetConfig::for_triple(triple).with_reloc_mode(RelocMode::PIC);

    config.create_target_machine(opt_level)
}

/// Create a native shared library target machine
pub fn create_native_shared_target_machine(opt_level: OptLevel) -> Result<TargetMachine, String> {
    initialize_native_target();

    let config = TargetConfig::native().with_reloc_mode(RelocMode::PIC);

    config.create_target_machine(opt_level)
}

/// Common target triples
pub mod triples {
    /// x86_64 Linux GNU
    pub const X86_64_LINUX_GNU: &str = "x86_64-unknown-linux-gnu";
    /// x86_64 Linux musl
    pub const X86_64_LINUX_MUSL: &str = "x86_64-unknown-linux-musl";
    /// x86_64 macOS
    pub const X86_64_MACOS: &str = "x86_64-apple-darwin";
    /// AArch64 macOS (Apple Silicon)
    pub const AARCH64_MACOS: &str = "aarch64-apple-darwin";
    /// x86_64 Windows MSVC
    pub const X86_64_WINDOWS_MSVC: &str = "x86_64-pc-windows-msvc";
    /// x86_64 Windows GNU
    pub const X86_64_WINDOWS_GNU: &str = "x86_64-pc-windows-gnu";
    /// AArch64 Linux GNU
    pub const AARCH64_LINUX_GNU: &str = "aarch64-unknown-linux-gnu";
    /// WebAssembly 32-bit
    pub const WASM32: &str = "wasm32-unknown-unknown";
    /// WebAssembly 32-bit WASI
    pub const WASM32_WASI: &str = "wasm32-wasi";
    /// RISC-V 64-bit
    pub const RISCV64_LINUX_GNU: &str = "riscv64-unknown-linux-gnu";

    // ========== GPU Targets ==========

    /// NVIDIA PTX 64-bit (CUDA)
    pub const NVPTX64: &str = "nvptx64-nvidia-cuda";
    /// NVIDIA PTX 32-bit (legacy CUDA)
    pub const NVPTX: &str = "nvptx-nvidia-cuda";
    /// AMD GPU (ROCm/HIP)
    pub const AMDGCN: &str = "amdgcn-amd-amdhsa";
    /// SPIR-V 64-bit (Vulkan/OpenCL)
    pub const SPIRV64: &str = "spirv64-unknown-unknown";

    /// Get triple for current host
    pub fn host() -> String {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        return X86_64_LINUX_GNU.to_string();

        #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
        return X86_64_MACOS.to_string();

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        return AARCH64_MACOS.to_string();

        #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
        return X86_64_WINDOWS_MSVC.to_string();

        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        return AARCH64_LINUX_GNU.to_string();

        #[allow(unreachable_code)]
        "unknown-unknown-unknown".to_string()
    }
}

/// Target information
#[derive(Debug, Clone)]
pub struct TargetInfo {
    /// Pointer size in bits
    pub pointer_size: u32,
    /// Default data alignment
    pub default_align: u32,
    /// Stack alignment
    pub stack_align: u32,
    /// Is big endian
    pub is_big_endian: bool,
    /// Is GPU target
    pub is_gpu: bool,
    /// GPU compute capability (for NVIDIA)
    pub gpu_compute_capability: Option<(u32, u32)>,
}

impl TargetInfo {
    /// Get target info for a triple
    pub fn for_triple(triple: &str) -> Self {
        // Default to 64-bit little endian
        let mut info = Self {
            pointer_size: 64,
            default_align: 8,
            stack_align: 16,
            is_big_endian: false,
            is_gpu: false,
            gpu_compute_capability: None,
        };

        if triple.contains("wasm32") {
            info.pointer_size = 32;
            info.default_align = 4;
        } else if triple.contains("i686") || triple.contains("i386") {
            info.pointer_size = 32;
            info.default_align = 4;
        } else if triple.contains("powerpc") && !triple.contains("powerpc64") {
            info.pointer_size = 32;
            info.default_align = 4;
            info.is_big_endian = true;
        } else if triple.contains("powerpc64") {
            info.is_big_endian = !triple.contains("le");
        } else if triple.contains("mips") && !triple.contains("mips64") {
            info.pointer_size = 32;
            info.default_align = 4;
            info.is_big_endian = !triple.contains("el");
        } else if triple.contains("nvptx64") {
            // NVIDIA PTX 64-bit
            info.is_gpu = true;
            info.gpu_compute_capability = Some((7, 5)); // Default to sm_75 (Turing)
        } else if triple.contains("nvptx") {
            // NVIDIA PTX 32-bit (legacy)
            info.pointer_size = 32;
            info.default_align = 4;
            info.is_gpu = true;
            info.gpu_compute_capability = Some((5, 0)); // Default to sm_50
        } else if triple.contains("amdgcn") {
            // AMD GCN GPU
            info.is_gpu = true;
        } else if triple.contains("spirv") {
            // SPIR-V (GPU compute)
            info.is_gpu = true;
        }

        info
    }

    /// Create target info for NVIDIA GPU with specific compute capability
    pub fn nvptx(compute_capability: (u32, u32)) -> Self {
        Self {
            pointer_size: 64,
            default_align: 8,
            stack_align: 16,
            is_big_endian: false,
            is_gpu: true,
            gpu_compute_capability: Some(compute_capability),
        }
    }

    /// Create target info for AMD GPU
    pub fn amdgpu() -> Self {
        Self {
            pointer_size: 64,
            default_align: 8,
            stack_align: 16,
            is_big_endian: false,
            is_gpu: true,
            gpu_compute_capability: None,
        }
    }
}

/// GPU target configuration
#[derive(Debug, Clone)]
pub struct GpuTargetConfig {
    /// Target triple
    pub triple: String,
    /// GPU architecture (e.g., "sm_75", "gfx1030")
    pub arch: String,
    /// Additional features
    pub features: String,
}

impl GpuTargetConfig {
    /// Create NVIDIA CUDA target config
    pub fn cuda(compute_capability: (u32, u32)) -> Self {
        Self {
            triple: triples::NVPTX64.to_string(),
            arch: format!("sm_{}{}", compute_capability.0, compute_capability.1),
            features: "+ptx75".to_string(), // PTX ISA 7.5
        }
    }

    /// Create AMD ROCm target config
    pub fn rocm(gfx_arch: &str) -> Self {
        Self {
            triple: triples::AMDGCN.to_string(),
            arch: gfx_arch.to_string(),
            features: String::new(),
        }
    }

    /// Create target config for sm_75 (Turing)
    pub fn sm_75() -> Self {
        Self::cuda((7, 5))
    }

    /// Create target config for sm_80 (Ampere)
    pub fn sm_80() -> Self {
        Self::cuda((8, 0))
    }

    /// Create target config for sm_86 (Ampere consumer)
    pub fn sm_86() -> Self {
        Self::cuda((8, 6))
    }

    /// Create target config for sm_89 (Ada Lovelace)
    pub fn sm_89() -> Self {
        Self::cuda((8, 9))
    }

    /// Create target config for sm_90 (Hopper)
    pub fn sm_90() -> Self {
        Self::cuda((9, 0))
    }

    /// Create target config for sm_100 (Blackwell)
    pub fn sm_100() -> Self {
        Self::cuda((10, 0))
    }

    /// Create target machine for this GPU config
    pub fn create_target_machine(&self, opt_level: OptLevel) -> Result<TargetMachine, String> {
        initialize_all_targets();

        let triple = TargetTriple::create(&self.triple);
        let target =
            Target::from_triple(&triple).map_err(|e| format!("Invalid GPU target: {}", e))?;

        let opt = opt_level.to_inkwell();

        target
            .create_target_machine(
                &triple,
                &self.arch,
                &self.features,
                opt,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| format!("Failed to create GPU target machine for {}", self.arch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_config_default() {
        let config = TargetConfig::default();
        assert_eq!(config.cpu, "generic");
        assert!(config.features.is_empty());
    }

    #[test]
    fn test_object_extension() {
        assert_eq!(object_extension("x86_64-unknown-linux-gnu"), "o");
        assert_eq!(object_extension("x86_64-pc-windows-msvc"), "obj");
    }

    #[test]
    fn test_executable_extension() {
        assert_eq!(executable_extension("x86_64-unknown-linux-gnu"), "");
        assert_eq!(executable_extension("x86_64-pc-windows-msvc"), "exe");
    }

    #[test]
    fn test_target_info() {
        let info = TargetInfo::for_triple("x86_64-unknown-linux-gnu");
        assert_eq!(info.pointer_size, 64);
        assert!(!info.is_big_endian);
        assert!(!info.is_gpu);

        let info32 = TargetInfo::for_triple("wasm32-unknown-unknown");
        assert_eq!(info32.pointer_size, 32);
    }

    #[test]
    fn test_host_triple() {
        let host = triples::host();
        assert!(!host.is_empty());
    }

    #[test]
    fn test_gpu_target_info() {
        let nvptx = TargetInfo::for_triple(triples::NVPTX64);
        assert!(nvptx.is_gpu);
        assert_eq!(nvptx.gpu_compute_capability, Some((7, 5)));

        let amd = TargetInfo::for_triple(triples::AMDGCN);
        assert!(amd.is_gpu);
        assert_eq!(amd.gpu_compute_capability, None);
    }

    #[test]
    fn test_gpu_target_config() {
        let sm80 = GpuTargetConfig::sm_80();
        assert_eq!(sm80.arch, "sm_80");
        assert!(sm80.triple.contains("nvptx64"));

        let sm100 = GpuTargetConfig::sm_100();
        assert_eq!(sm100.arch, "sm_100");
    }
}
