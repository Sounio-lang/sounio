//! GPU Runtime Bridge
//!
//! Provides a global singleton GpuRuntime that bridges the effect handler system
//! to actual GPU execution. This module enables:
//!
//! - Global GPU runtime singleton (OnceLock + Mutex pattern)
//! - Kernel registry (map kernel IDs to loaded kernels)
//! - Buffer registry (map buffer IDs to device buffers)
//! - C-callable dispatch functions for effect system integration
//!
//! # Architecture
//!
//! ```text
//! Handler Stack                    GPU Bridge                    GPU Runtime
//! +----------------+              +----------------+            +----------------+
//! | Effect Dispatch| ----------> | Kernel Registry| ---------> | CUDA/Vulkan    |
//! | gpu_launch()   |              | Buffer Registry|            | launch()       |
//! +----------------+              +----------------+            +----------------+
//! ```
//!
//! # Usage
//!
//! The bridge is automatically initialized when first accessed. Kernel and buffer
//! IDs are assigned sequentially and can be used from effect handlers.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::codegen::gpu::runtime::{
    DeviceBuffer, GpuBackend, GpuError, GpuRuntime, Kernel, KernelArg, LaunchConfig,
};
use crate::codegen::gpu::epistemic_gemm::{EpistemicGemmConfig, generate_epistemic_gemm_ptx_sm};
#[cfg(feature = "gpu")]
use crate::codegen::gpu::epistemic_gemm::{
    format_gemm_profile, profile_gemm_launch, suggest_tile_config,
};

/// Global GPU runtime singleton
static GPU_RUNTIME: OnceLock<Mutex<GpuRuntimeBridge>> = OnceLock::new();

/// Get the global GPU runtime bridge
///
/// Initializes the bridge on first access using simulated backend by default.
/// Use `init_gpu_bridge` for explicit initialization with a specific backend.
pub fn get_gpu_bridge() -> &'static Mutex<GpuRuntimeBridge> {
    GPU_RUNTIME.get_or_init(|| {
        let bridge = GpuRuntimeBridge::new_simulated();
        Mutex::new(bridge)
    })
}

/// Initialize GPU bridge with a specific backend
///
/// Must be called before any GPU operations. Returns error if already initialized.
pub fn init_gpu_bridge(backend: GpuBackend, device_id: u32) -> Result<(), GpuBridgeError> {
    let bridge = GpuRuntimeBridge::new(backend, device_id)?;

    GPU_RUNTIME
        .set(Mutex::new(bridge))
        .map_err(|_| GpuBridgeError::AlreadyInitialized)
}

/// Reset the GPU bridge (for testing)
///
/// # Safety
///
/// This is unsafe because it invalidates any existing kernel/buffer IDs.
/// Only use in tests or when you're certain no GPU operations are in progress.
#[cfg(test)]
pub unsafe fn reset_gpu_bridge() {
    // OnceLock doesn't have a reset, so we work around this in tests
    // by just clearing the registries
    if let Some(bridge) = GPU_RUNTIME.get() {
        if let Ok(mut b) = bridge.lock() {
            b.kernels.clear();
            b.buffers.clear();
        }
    }
}

/// GPU Runtime Bridge
///
/// Wraps GpuRuntime with registries for kernel and buffer management.
pub struct GpuRuntimeBridge {
    /// The underlying GPU runtime
    runtime: GpuRuntime,

    /// Kernel registry: ID -> Kernel
    kernels: HashMap<u64, Kernel>,

    /// Buffer registry: ID -> DeviceBuffer
    buffers: HashMap<u64, DeviceBuffer>,

    /// Next kernel ID
    next_kernel_id: u64,

    /// Next buffer ID
    next_buffer_id: u64,

    /// Statistics
    stats: BridgeStats,
}

// Safety: GpuRuntimeBridge is safe to send between threads because:
// 1. The underlying GPU runtime (CUDA/Vulkan/etc.) handles its own thread synchronization
// 2. All access goes through a Mutex, ensuring exclusive access
// 3. Raw pointers in DeviceBuffer/Kernel are device memory handles managed by the GPU driver
// 4. The simulated backend has no actual threading concerns
//
// This is the standard pattern for FFI GPU runtimes (cudarc, wgpu, vulkano all do this).
unsafe impl Send for GpuRuntimeBridge {}
unsafe impl Sync for GpuRuntimeBridge {}

/// Bridge statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct BridgeStats {
    /// Total kernels loaded
    pub kernels_loaded: u64,
    /// Total buffers allocated
    pub buffers_allocated: u64,
    /// Total bytes allocated
    pub bytes_allocated: u64,
    /// Total kernel launches
    pub kernel_launches: u64,
    /// Total host-to-device copies
    pub htod_copies: u64,
    /// Total device-to-host copies
    pub dtoh_copies: u64,
}

/// GPU Bridge Error
#[derive(Debug, Clone)]
pub enum GpuBridgeError {
    /// Bridge already initialized
    AlreadyInitialized,
    /// GPU runtime error
    RuntimeError(String),
    /// Kernel not found
    KernelNotFound(u64),
    /// Buffer not found
    BufferNotFound(u64),
    /// Invalid configuration
    InvalidConfig(String),
}

impl std::fmt::Display for GpuBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBridgeError::AlreadyInitialized => write!(f, "GPU bridge already initialized"),
            GpuBridgeError::RuntimeError(msg) => write!(f, "GPU runtime error: {}", msg),
            GpuBridgeError::KernelNotFound(id) => write!(f, "Kernel {} not found", id),
            GpuBridgeError::BufferNotFound(id) => write!(f, "Buffer {} not found", id),
            GpuBridgeError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
        }
    }
}

impl std::error::Error for GpuBridgeError {}

impl From<GpuError> for GpuBridgeError {
    fn from(err: GpuError) -> Self {
        GpuBridgeError::RuntimeError(format!("{:?}", err))
    }
}

impl GpuRuntimeBridge {
    /// Create a new GPU bridge with the specified backend
    pub fn new(backend: GpuBackend, device_id: u32) -> Result<Self, GpuBridgeError> {
        let runtime = GpuRuntime::new(backend, device_id)?;

        Ok(Self {
            runtime,
            kernels: HashMap::new(),
            buffers: HashMap::new(),
            next_kernel_id: 1, // Start from 1, 0 is invalid
            next_buffer_id: 1,
            stats: BridgeStats::default(),
        })
    }

    /// Create a bridge with simulated backend (for testing)
    pub fn new_simulated() -> Self {
        let runtime = GpuRuntime::new(GpuBackend::Simulated, 0)
            .expect("Simulated backend should always succeed");

        Self {
            runtime,
            kernels: HashMap::new(),
            buffers: HashMap::new(),
            next_kernel_id: 1,
            next_buffer_id: 1,
            stats: BridgeStats::default(),
        }
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.runtime.backend()
    }

    /// Get statistics
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }

    // === Kernel Management ===

    /// Load a PTX kernel and return its ID
    pub fn load_ptx(&mut self, ptx: &str, name: &str) -> Result<u64, GpuBridgeError> {
        let kernel = self.runtime.load_ptx(ptx, name)?;
        let id = self.next_kernel_id;
        self.next_kernel_id += 1;
        self.kernels.insert(id, kernel);
        self.stats.kernels_loaded += 1;
        Ok(id)
    }

    /// Load a SPIR-V kernel and return its ID
    pub fn load_spirv(&mut self, spirv: &[u8], name: &str) -> Result<u64, GpuBridgeError> {
        let kernel = self.runtime.load_spirv(spirv, name)?;
        let id = self.next_kernel_id;
        self.next_kernel_id += 1;
        self.kernels.insert(id, kernel);
        self.stats.kernels_loaded += 1;
        Ok(id)
    }

    /// Launch a kernel by ID
    pub fn launch_kernel(
        &mut self,
        kernel_id: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[KernelArg],
    ) -> Result<(), GpuBridgeError> {
        let kernel = self
            .kernels
            .get(&kernel_id)
            .ok_or(GpuBridgeError::KernelNotFound(kernel_id))?;

        let config = LaunchConfig::new(grid, block);
        self.runtime.launch(kernel, &config, args)?;
        self.stats.kernel_launches += 1;
        Ok(())
    }

    /// Launch a kernel by ID with 1D configuration
    pub fn launch_kernel_1d(
        &mut self,
        kernel_id: u64,
        num_elements: u32,
        block_size: u32,
        args: &[KernelArg],
    ) -> Result<(), GpuBridgeError> {
        let grid_size = (num_elements + block_size - 1) / block_size;
        self.launch_kernel(kernel_id, (grid_size, 1, 1), (block_size, 1, 1), args)
    }

    /// Get kernel info
    pub fn kernel_info(&self, kernel_id: u64) -> Option<(&str, usize)> {
        self.kernels
            .get(&kernel_id)
            .map(|k| (k.name(), k.param_count()))
    }

    /// Unload a kernel
    pub fn unload_kernel(&mut self, kernel_id: u64) -> Result<(), GpuBridgeError> {
        self.kernels
            .remove(&kernel_id)
            .ok_or(GpuBridgeError::KernelNotFound(kernel_id))?;
        Ok(())
    }

    // === Buffer Management ===

    /// Allocate a device buffer and return its ID
    pub fn alloc(&mut self, size: usize) -> Result<u64, GpuBridgeError> {
        let buffer = self.runtime.alloc(size)?;
        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(id, buffer);
        self.stats.buffers_allocated += 1;
        self.stats.bytes_allocated += size as u64;
        Ok(id)
    }

    /// Free a device buffer by ID
    pub fn free(&mut self, buffer_id: u64) -> Result<(), GpuBridgeError> {
        let buffer = self
            .buffers
            .remove(&buffer_id)
            .ok_or(GpuBridgeError::BufferNotFound(buffer_id))?;
        self.runtime.free(buffer)?;
        Ok(())
    }

    /// Copy data from host to device buffer
    pub fn copy_htod<T>(&mut self, buffer_id: u64, data: &[T]) -> Result<(), GpuBridgeError> {
        let buffer = self
            .buffers
            .get(&buffer_id)
            .ok_or(GpuBridgeError::BufferNotFound(buffer_id))?;
        self.runtime.copy_to_device(buffer, data)?;
        self.stats.htod_copies += 1;
        Ok(())
    }

    /// Copy data from device buffer to host
    pub fn copy_dtoh<T>(&mut self, buffer_id: u64, data: &mut [T]) -> Result<(), GpuBridgeError> {
        let buffer = self
            .buffers
            .get(&buffer_id)
            .ok_or(GpuBridgeError::BufferNotFound(buffer_id))?;
        self.runtime.copy_to_host(data, buffer)?;
        self.stats.dtoh_copies += 1;
        Ok(())
    }

    /// Get buffer size
    pub fn buffer_size(&self, buffer_id: u64) -> Option<usize> {
        self.buffers.get(&buffer_id).map(|b| b.size())
    }

    /// Get buffer as kernel argument
    pub fn buffer_as_arg(&self, buffer_id: u64) -> Option<KernelArg> {
        self.buffers
            .get(&buffer_id)
            .map(|b| KernelArg::Buffer(b.as_ptr()))
    }

    // === Epistemic GEMM dispatch ===

    /// Compile, load, and launch an epistemic GEMM kernel using the device's
    /// SM version for PTX selection and emit a roofline profile line afterwards.
    ///
    /// The PTX path (single-buffer vs. double-buffered cp.async vs. WMMA)
    /// is chosen automatically from `DeviceInfo::compute_capability`.
    ///
    /// Returns `(kernel_id, elapsed_ms)` so the caller can reuse the loaded
    /// kernel on subsequent calls.
    pub fn launch_epistemic_gemm(
        &mut self,
        config: &EpistemicGemmConfig,
        kernel_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[KernelArg],
    ) -> Result<(u64, f64), GpuBridgeError> {
        let sm_major = self.runtime.device_info().sm_major();
        let sm_minor = self.runtime.device_info().sm_minor();
        let mem_bw = self.runtime.device_info().mem_bandwidth_gbs;
        let peak_fp32 = self.runtime.device_info().peak_fp32_tflops;

        // SM-aware PTX generation (Goal 1)
        let ptx = generate_epistemic_gemm_ptx_sm(config, kernel_name, sm_major, sm_minor);

        let kernel_id = self.load_ptx(&ptx, kernel_name)?;

        let start = std::time::Instant::now();
        self.launch_kernel(kernel_id, grid, block, args)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Roofline profiler hook (Goal 2)
        // Double-buffering is active when sm_major >= 8 and precision is F32
        let uses_double_buffer = sm_major >= 8
            && !config.use_tensor_cores;
        #[cfg(feature = "gpu")]
        {
            let profile = profile_gemm_launch(config, elapsed_ms, mem_bw, peak_fp32, uses_double_buffer);
            eprintln!("{}", format_gemm_profile(kernel_name, config, &profile));
            if let Some(hint) = suggest_tile_config(&profile) {
                eprintln!("{hint}");
            }
        }
        // Suppress unused-variable warnings outside the `gpu` feature gate.
        let _ = (mem_bw, peak_fp32, uses_double_buffer);

        Ok((kernel_id, elapsed_ms))
    }

    // === Synchronization ===

    /// Synchronize all GPU operations
    pub fn synchronize(&self) -> Result<(), GpuBridgeError> {
        self.runtime.synchronize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = GpuRuntimeBridge::new_simulated();
        assert_eq!(bridge.backend(), GpuBackend::Simulated);
        assert_eq!(bridge.stats.kernels_loaded, 0);
        assert_eq!(bridge.stats.buffers_allocated, 0);
    }

    #[test]
    fn test_buffer_allocation() {
        let mut bridge = GpuRuntimeBridge::new_simulated();

        let buf_id = bridge.alloc(1024).expect("allocation should succeed");
        assert!(buf_id > 0);
        assert_eq!(bridge.buffer_size(buf_id), Some(1024));
        assert_eq!(bridge.stats.buffers_allocated, 1);
        assert_eq!(bridge.stats.bytes_allocated, 1024);

        bridge.free(buf_id).expect("free should succeed");
        assert_eq!(bridge.buffer_size(buf_id), None);
    }

    #[test]
    fn test_multiple_buffers() {
        let mut bridge = GpuRuntimeBridge::new_simulated();

        let buf1 = bridge.alloc(512).unwrap();
        let buf2 = bridge.alloc(1024).unwrap();
        let buf3 = bridge.alloc(256).unwrap();

        assert_ne!(buf1, buf2);
        assert_ne!(buf2, buf3);
        assert_eq!(bridge.stats.buffers_allocated, 3);
        assert_eq!(bridge.stats.bytes_allocated, 512 + 1024 + 256);

        // Free in different order
        bridge.free(buf2).unwrap();
        bridge.free(buf1).unwrap();
        bridge.free(buf3).unwrap();
    }

    #[test]
    fn test_invalid_buffer_id() {
        let mut bridge = GpuRuntimeBridge::new_simulated();

        let result = bridge.free(999);
        assert!(matches!(result, Err(GpuBridgeError::BufferNotFound(999))));
    }

    #[test]
    fn test_kernel_not_found() {
        let mut bridge = GpuRuntimeBridge::new_simulated();

        let result = bridge.launch_kernel(999, (1, 1, 1), (1, 1, 1), &[]);
        assert!(matches!(result, Err(GpuBridgeError::KernelNotFound(999))));
    }

    #[test]
    fn test_simulated_kernel_load() {
        let mut bridge = GpuRuntimeBridge::new_simulated();

        // Simulated backend should accept any PTX
        let ptx = r#"
            .version 7.0
            .target sm_70
            .entry kernel_add() { ret; }
        "#;

        let kernel_id = bridge.load_ptx(ptx, "kernel_add");
        // Simulated backend may or may not support load
        if let Ok(id) = kernel_id {
            assert!(id > 0);
            assert_eq!(bridge.stats.kernels_loaded, 1);
        }
    }

    #[test]
    fn test_bridge_stats() {
        let bridge = GpuRuntimeBridge::new_simulated();
        let stats = bridge.stats();

        assert_eq!(stats.kernels_loaded, 0);
        assert_eq!(stats.buffers_allocated, 0);
        assert_eq!(stats.kernel_launches, 0);
    }

    #[test]
    fn test_launch_epistemic_gemm_roofline() {
        use crate::codegen::gpu::epistemic_gemm::MatrixPrecision;
        let mut bridge = GpuRuntimeBridge::new_simulated();

        let config = EpistemicGemmConfig {
            m: 64,
            k: 64,
            n: 64,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };

        let result = bridge.launch_epistemic_gemm(
            &config,
            "roofline_test_gemm",
            (1, 1, 1),
            (16, 16, 1),
            &[],
        );
        assert!(result.is_ok(), "launch_epistemic_gemm must succeed on simulated backend");
        let (kernel_id, elapsed_ms) = result.unwrap();
        assert!(kernel_id > 0, "kernel id must be nonzero");
        assert!(elapsed_ms >= 0.0, "elapsed must be non-negative");
        assert_eq!(bridge.stats.kernel_launches, 1);
    }

    #[test]
    fn test_suggest_tile_config_memory_bound() {
        use crate::codegen::gpu::epistemic_gemm::{suggest_tile_config, GemmLaunchProfile};
        let p = GemmLaunchProfile {
            flops: 2.0 * 64.0 * 64.0 * 64.0,
            achieved_gflops: 14.0,
            arithmetic_intensity: 0.3,
            ridge_point_flops_per_byte: 20.0,
            memory_bound: true,
            efficiency: 0.0005,
            double_buffered: false,
        };
        let hint = suggest_tile_config(&p);
        assert!(hint.is_some(), "memory-bound single-buffer must produce a hint");
        let s = hint.unwrap();
        assert!(s.contains("memory-bound"), "hint must mention memory-bound");
        assert!(s.contains("128"), "hint must suggest 128×128 tile");
    }

    #[test]
    fn test_suggest_tile_config_compute_bound() {
        use crate::codegen::gpu::epistemic_gemm::{suggest_tile_config, GemmLaunchProfile};
        let p = GemmLaunchProfile {
            flops: 2.0 * 4096.0 * 4096.0 * 4096.0,
            achieved_gflops: 10000.0,
            arithmetic_intensity: 256.0,
            ridge_point_flops_per_byte: 20.0,
            memory_bound: false,
            efficiency: 0.5,
            double_buffered: true,
        };
        assert!(suggest_tile_config(&p).is_none(), "compute-bound path must return None");
    }

    // C-callable dispatch tests moved to runtime::ffi::ffi_gpu
}
