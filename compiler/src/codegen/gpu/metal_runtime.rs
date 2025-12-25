//! Native Metal Runtime for Apple GPUs
//!
//! This module provides the Metal-specific runtime bindings for executing
//! Sounio GPU kernels on Apple Silicon and Intel Macs.
//!
//! # Architecture
//!
//! ```text
//! Sounio MSL → Metal Library → Pipeline State → Command Encoder → GPU
//! ```
//!
//! # Key Metal Concepts
//!
//! - **MTLDevice**: Represents a GPU (Apple Silicon or Intel)
//! - **MTLCommandQueue**: Queue for submitting work
//! - **MTLBuffer**: GPU memory allocation
//! - **MTLLibrary**: Compiled MSL shaders
//! - **MTLComputePipelineState**: Compiled kernel ready for execution
//! - **MTLCommandBuffer**: Container for GPU commands
//! - **MTLComputeCommandEncoder**: Encodes compute kernel dispatches
//!
//! # Example
//!
//! ```ignore
//! use sounio::codegen::gpu::metal_runtime::*;
//!
//! // Initialize Metal runtime
//! let runtime = MetalRuntime::new(MetalGpuFamily::Apple8)?;
//!
//! // Compile MSL to library
//! let library = runtime.compile_msl(msl_source)?;
//!
//! // Get kernel function
//! let kernel = runtime.get_kernel(&library, "epistemic_add")?;
//!
//! // Allocate buffers
//! let input_a = runtime.alloc_buffer(size, MetalStorageMode::Shared)?;
//! let input_b = runtime.alloc_buffer(size, MetalStorageMode::Shared)?;
//! let output = runtime.alloc_buffer(size, MetalStorageMode::Shared)?;
//!
//! // Launch kernel
//! runtime.dispatch_kernel(&kernel, &[&input_a, &input_b, &output], grid_size, threadgroup_size)?;
//!
//! // Synchronize
//! runtime.synchronize()?;
//! ```

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::ptr;

use super::ir::MetalGpuFamily;
use super::runtime::{GpuError, LaunchConfig};

/// Metal storage mode for buffers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalStorageMode {
    /// Shared memory accessible by both CPU and GPU (unified memory)
    /// Best for Apple Silicon
    Shared,
    /// Private GPU memory (not accessible by CPU)
    /// Best for intermediate computation buffers
    Private,
    /// Managed memory with explicit synchronization
    /// For Intel Macs with discrete GPUs
    Managed,
    /// Memory-mapped storage for streaming
    Memoryless,
}

impl MetalStorageMode {
    /// Get the Metal API storage mode value
    pub fn to_metal_value(&self) -> u32 {
        match self {
            MetalStorageMode::Shared => 0,     // MTLStorageModeShared
            MetalStorageMode::Private => 2,    // MTLStorageModePrivate
            MetalStorageMode::Managed => 1,    // MTLStorageModeManaged
            MetalStorageMode::Memoryless => 3, // MTLStorageModeMemoryless
        }
    }

    /// Get recommended storage mode for GPU family
    pub fn recommended_for(family: MetalGpuFamily, cpu_access_needed: bool) -> Self {
        match family {
            MetalGpuFamily::Mac2 => {
                if cpu_access_needed {
                    MetalStorageMode::Managed
                } else {
                    MetalStorageMode::Private
                }
            }
            _ => {
                // Apple Silicon has unified memory
                if cpu_access_needed {
                    MetalStorageMode::Shared
                } else {
                    MetalStorageMode::Private
                }
            }
        }
    }
}

/// Metal resource options
#[derive(Debug, Clone, Copy)]
pub struct MetalResourceOptions {
    /// Storage mode
    pub storage_mode: MetalStorageMode,
    /// CPU cache mode
    pub cpu_cache_mode: MetalCpuCacheMode,
    /// Hazard tracking mode
    pub hazard_tracking: MetalHazardTracking,
}

impl Default for MetalResourceOptions {
    fn default() -> Self {
        Self {
            storage_mode: MetalStorageMode::Shared,
            cpu_cache_mode: MetalCpuCacheMode::DefaultCache,
            hazard_tracking: MetalHazardTracking::Default,
        }
    }
}

/// Metal CPU cache mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalCpuCacheMode {
    DefaultCache,
    WriteCombined,
}

/// Metal hazard tracking mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalHazardTracking {
    Default,
    Untracked,
    Tracked,
}

/// Metal buffer handle
#[derive(Debug)]
pub struct MetalBuffer {
    /// Opaque handle to MTLBuffer
    handle: *mut c_void,
    /// Size in bytes
    size: usize,
    /// Storage mode
    storage_mode: MetalStorageMode,
    /// Label for debugging
    label: String,
    /// Is buffer valid
    valid: bool,
}

impl MetalBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn storage_mode(&self) -> MetalStorageMode {
        self.storage_mode
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle
    }
}

/// Metal library (compiled MSL)
#[derive(Debug)]
pub struct MetalLibrary {
    /// Opaque handle to MTLLibrary
    handle: *mut c_void,
    /// Source MSL (for debugging)
    source: String,
    /// Function names in library
    functions: Vec<String>,
}

impl MetalLibrary {
    pub fn functions(&self) -> &[String] {
        &self.functions
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle
    }
}

/// Metal compute kernel (pipeline state)
#[derive(Debug)]
pub struct MetalKernel {
    /// Opaque handle to MTLComputePipelineState
    handle: *mut c_void,
    /// Function name
    name: String,
    /// Maximum total threads per threadgroup
    max_total_threads_per_threadgroup: u32,
    /// Thread execution width (simd width)
    thread_execution_width: u32,
    /// Static threadgroup memory size
    static_threadgroup_memory_length: u32,
}

impl MetalKernel {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn max_total_threads_per_threadgroup(&self) -> u32 {
        self.max_total_threads_per_threadgroup
    }

    pub fn thread_execution_width(&self) -> u32 {
        self.thread_execution_width
    }

    pub fn static_threadgroup_memory_length(&self) -> u32 {
        self.static_threadgroup_memory_length
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle
    }
}

/// Metal dispatch size (grid configuration)
#[derive(Debug, Clone, Copy)]
pub struct MetalDispatchSize {
    /// Threadgroups per grid (equivalent to CUDA grid)
    pub threadgroups: (u32, u32, u32),
    /// Threads per threadgroup (equivalent to CUDA block)
    pub threads_per_threadgroup: (u32, u32, u32),
}

impl MetalDispatchSize {
    pub fn new(threadgroups: (u32, u32, u32), threads_per_threadgroup: (u32, u32, u32)) -> Self {
        Self {
            threadgroups,
            threads_per_threadgroup,
        }
    }

    /// Create 1D dispatch
    pub fn new_1d(num_threadgroups: u32, threadgroup_size: u32) -> Self {
        Self::new((num_threadgroups, 1, 1), (threadgroup_size, 1, 1))
    }

    /// Create 2D dispatch
    pub fn new_2d(threadgroups: (u32, u32), threads_per_threadgroup: (u32, u32)) -> Self {
        Self::new(
            (threadgroups.0, threadgroups.1, 1),
            (threads_per_threadgroup.0, threads_per_threadgroup.1, 1),
        )
    }

    /// Convert from generic LaunchConfig
    pub fn from_launch_config(config: &LaunchConfig) -> Self {
        Self {
            threadgroups: config.grid,
            threads_per_threadgroup: config.block,
        }
    }

    /// Total threads dispatched
    pub fn total_threads(&self) -> u64 {
        let tg =
            self.threadgroups.0 as u64 * self.threadgroups.1 as u64 * self.threadgroups.2 as u64;
        let tpt = self.threads_per_threadgroup.0 as u64
            * self.threads_per_threadgroup.1 as u64
            * self.threads_per_threadgroup.2 as u64;
        tg * tpt
    }

    /// Total threadgroups
    pub fn total_threadgroups(&self) -> u64 {
        self.threadgroups.0 as u64 * self.threadgroups.1 as u64 * self.threadgroups.2 as u64
    }

    /// Threads per threadgroup
    pub fn threads_per_group(&self) -> u64 {
        self.threads_per_threadgroup.0 as u64
            * self.threads_per_threadgroup.1 as u64
            * self.threads_per_threadgroup.2 as u64
    }
}

/// Metal device information
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    /// Device name
    pub name: String,
    /// GPU family
    pub family: MetalGpuFamily,
    /// Recommended maximum working set size
    pub recommended_max_working_set_size: u64,
    /// Maximum buffer length
    pub max_buffer_length: u64,
    /// Maximum threads per threadgroup
    pub max_threads_per_threadgroup: u32,
    /// Maximum threadgroup memory size
    pub max_threadgroup_memory_length: u32,
    /// Supports Apple Silicon features
    pub supports_apple_silicon_features: bool,
    /// Supports simdgroup functions
    pub supports_simdgroup: bool,
    /// Supports simdgroup matrix operations
    pub supports_simdgroup_matrix: bool,
    /// Registry ID
    pub registry_id: u64,
    /// Is low power device
    pub is_low_power: bool,
    /// Is headless (no display)
    pub is_headless: bool,
}

impl MetalDeviceInfo {
    /// Create default info for a GPU family
    pub fn default_for(family: MetalGpuFamily) -> Self {
        let (name, apple_silicon, simdgroup_matrix) = match family {
            MetalGpuFamily::Apple7 => ("Apple M1 GPU", true, true),
            MetalGpuFamily::Apple8 => ("Apple M2 GPU", true, true),
            MetalGpuFamily::Apple9 => ("Apple M3 GPU", true, true),
            MetalGpuFamily::Apple10 => ("Apple M4 GPU", true, true),
            MetalGpuFamily::Mac2 => ("Intel Mac GPU", false, false),
            MetalGpuFamily::Common => ("Metal Common GPU", false, false),
        };

        Self {
            name: name.to_string(),
            family,
            recommended_max_working_set_size: match family {
                MetalGpuFamily::Apple10 => 64 * 1024 * 1024 * 1024, // 64 GB (M4 Max)
                MetalGpuFamily::Apple9 => 48 * 1024 * 1024 * 1024,  // 48 GB
                MetalGpuFamily::Apple8 => 32 * 1024 * 1024 * 1024,  // 32 GB
                MetalGpuFamily::Apple7 => 16 * 1024 * 1024 * 1024,  // 16 GB
                _ => 8 * 1024 * 1024 * 1024,                        // 8 GB
            },
            max_buffer_length: 1024 * 1024 * 1024, // 1 GB
            max_threads_per_threadgroup: family.max_threads_per_threadgroup(),
            max_threadgroup_memory_length: family.max_threadgroup_memory(),
            supports_apple_silicon_features: apple_silicon,
            supports_simdgroup: true,
            supports_simdgroup_matrix: simdgroup_matrix,
            registry_id: 0,
            is_low_power: false,
            is_headless: false,
        }
    }
}

/// Metal runtime for GPU execution
pub struct MetalRuntime {
    /// GPU family
    gpu_family: MetalGpuFamily,
    /// Device handle (MTLDevice)
    device: *mut c_void,
    /// Command queue handle (MTLCommandQueue)
    command_queue: *mut c_void,
    /// Device info
    device_info: MetalDeviceInfo,
    /// Compiled libraries cache
    libraries: HashMap<String, MetalLibrary>,
    /// Compiled kernels cache
    kernels: HashMap<String, MetalKernel>,
    /// Allocated buffers (for tracking)
    allocated_bytes: u64,
}

impl MetalRuntime {
    /// Create new Metal runtime for specified GPU family
    pub fn new(gpu_family: MetalGpuFamily) -> Result<Self, MetalError> {
        // In real implementation, would:
        // 1. Call MTLCreateSystemDefaultDevice() or select specific device
        // 2. Create MTLCommandQueue
        // 3. Query device capabilities

        Ok(Self {
            gpu_family,
            device: ptr::null_mut(),
            command_queue: ptr::null_mut(),
            device_info: MetalDeviceInfo::default_for(gpu_family),
            libraries: HashMap::new(),
            kernels: HashMap::new(),
            allocated_bytes: 0,
        })
    }

    /// Get GPU family
    pub fn gpu_family(&self) -> MetalGpuFamily {
        self.gpu_family
    }

    /// Get device info
    pub fn device_info(&self) -> &MetalDeviceInfo {
        &self.device_info
    }

    /// Get allocated bytes
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Compile MSL source to Metal library
    pub fn compile_msl(&mut self, source: &str) -> Result<MetalLibrary, MetalError> {
        // In real implementation:
        // 1. Create MTLCompileOptions
        // 2. Call device.makeLibrary(source:options:)
        // 3. Handle compilation errors

        // Parse function names from source (simple regex-like scan)
        let functions = Self::extract_function_names(source);

        let library = MetalLibrary {
            handle: ptr::null_mut(),
            source: source.to_string(),
            functions,
        };

        Ok(library)
    }

    /// Compile MSL from file
    pub fn compile_msl_file(&mut self, path: &str) -> Result<MetalLibrary, MetalError> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| MetalError::CompilationFailed(format!("Failed to read file: {}", e)))?;
        self.compile_msl(&source)
    }

    /// Load pre-compiled Metal library (.metallib)
    pub fn load_metallib(&mut self, _path: &str) -> Result<MetalLibrary, MetalError> {
        // In real implementation:
        // Call device.makeLibrary(filepath:)

        Ok(MetalLibrary {
            handle: ptr::null_mut(),
            source: String::new(),
            functions: Vec::new(),
        })
    }

    /// Get kernel from library
    pub fn get_kernel(
        &mut self,
        library: &MetalLibrary,
        function_name: &str,
    ) -> Result<MetalKernel, MetalError> {
        // Check cache
        if let Some(kernel) = self.kernels.get(function_name) {
            return Ok(MetalKernel {
                handle: kernel.handle,
                name: kernel.name.clone(),
                max_total_threads_per_threadgroup: kernel.max_total_threads_per_threadgroup,
                thread_execution_width: kernel.thread_execution_width,
                static_threadgroup_memory_length: kernel.static_threadgroup_memory_length,
            });
        }

        // Check if function exists
        if !library.functions.contains(&function_name.to_string()) && !library.functions.is_empty()
        {
            return Err(MetalError::FunctionNotFound(function_name.to_string()));
        }

        // In real implementation:
        // 1. Call library.makeFunction(name:)
        // 2. Call device.makeComputePipelineState(function:)

        let kernel = MetalKernel {
            handle: ptr::null_mut(),
            name: function_name.to_string(),
            max_total_threads_per_threadgroup: self.device_info.max_threads_per_threadgroup,
            thread_execution_width: self.gpu_family.simd_width(),
            static_threadgroup_memory_length: 0,
        };

        self.kernels.insert(
            function_name.to_string(),
            MetalKernel {
                handle: kernel.handle,
                name: kernel.name.clone(),
                max_total_threads_per_threadgroup: kernel.max_total_threads_per_threadgroup,
                thread_execution_width: kernel.thread_execution_width,
                static_threadgroup_memory_length: kernel.static_threadgroup_memory_length,
            },
        );

        Ok(kernel)
    }

    /// Allocate buffer
    pub fn alloc_buffer(
        &mut self,
        size: usize,
        storage_mode: MetalStorageMode,
    ) -> Result<MetalBuffer, MetalError> {
        self.alloc_buffer_with_label(size, storage_mode, "")
    }

    /// Allocate buffer with label
    pub fn alloc_buffer_with_label(
        &mut self,
        size: usize,
        storage_mode: MetalStorageMode,
        label: &str,
    ) -> Result<MetalBuffer, MetalError> {
        if size == 0 {
            return Err(MetalError::InvalidSize);
        }

        if size as u64 > self.device_info.max_buffer_length {
            return Err(MetalError::BufferTooLarge);
        }

        // In real implementation:
        // Call device.makeBuffer(length:options:)

        self.allocated_bytes += size as u64;

        Ok(MetalBuffer {
            handle: ptr::null_mut(),
            size,
            storage_mode,
            label: label.to_string(),
            valid: true,
        })
    }

    /// Allocate typed buffer
    pub fn alloc_typed<T>(
        &mut self,
        count: usize,
        storage_mode: MetalStorageMode,
    ) -> Result<MetalBuffer, MetalError> {
        self.alloc_buffer(count * std::mem::size_of::<T>(), storage_mode)
    }

    /// Free buffer
    pub fn free_buffer(&mut self, buffer: MetalBuffer) -> Result<(), MetalError> {
        if buffer.valid {
            self.allocated_bytes = self.allocated_bytes.saturating_sub(buffer.size as u64);
        }
        // In real implementation: buffer goes out of scope and ARC releases it
        Ok(())
    }

    /// Copy data to buffer (for Shared storage mode)
    pub fn copy_to_buffer<T>(&self, buffer: &MetalBuffer, data: &[T]) -> Result<(), MetalError> {
        let size = std::mem::size_of_val(data);
        if size > buffer.size {
            return Err(MetalError::BufferTooSmall);
        }

        if buffer.storage_mode == MetalStorageMode::Private {
            return Err(MetalError::InvalidStorageMode);
        }

        // In real implementation:
        // 1. Get buffer.contents() pointer
        // 2. memcpy data to pointer
        // 3. If Managed: call buffer.didModifyRange()

        Ok(())
    }

    /// Copy data from buffer (for Shared storage mode)
    pub fn copy_from_buffer<T: Clone + Default>(
        &self,
        buffer: &MetalBuffer,
        count: usize,
    ) -> Result<Vec<T>, MetalError> {
        let size = count * std::mem::size_of::<T>();
        if size > buffer.size {
            return Err(MetalError::BufferTooSmall);
        }

        if buffer.storage_mode == MetalStorageMode::Private {
            return Err(MetalError::InvalidStorageMode);
        }

        // In real implementation:
        // 1. Get buffer.contents() pointer
        // 2. Copy data from pointer

        Ok(vec![T::default(); count])
    }

    /// Dispatch kernel
    pub fn dispatch_kernel(
        &self,
        kernel: &MetalKernel,
        buffers: &[&MetalBuffer],
        dispatch_size: MetalDispatchSize,
    ) -> Result<(), MetalError> {
        self.dispatch_kernel_with_threadgroup_memory(kernel, buffers, dispatch_size, 0)
    }

    /// Dispatch kernel with threadgroup memory
    pub fn dispatch_kernel_with_threadgroup_memory(
        &self,
        kernel: &MetalKernel,
        buffers: &[&MetalBuffer],
        dispatch_size: MetalDispatchSize,
        threadgroup_memory: u32,
    ) -> Result<(), MetalError> {
        // Validate dispatch parameters
        let threads_per_group = dispatch_size.threads_per_group();
        if threads_per_group > kernel.max_total_threads_per_threadgroup as u64 {
            return Err(MetalError::InvalidDispatchSize(format!(
                "Threads per threadgroup ({}) exceeds maximum ({})",
                threads_per_group, kernel.max_total_threads_per_threadgroup
            )));
        }

        let total_threadgroup_memory = threadgroup_memory + kernel.static_threadgroup_memory_length;
        if total_threadgroup_memory > self.device_info.max_threadgroup_memory_length {
            return Err(MetalError::ThreadgroupMemoryExceeded);
        }

        // In real implementation:
        // 1. Create MTLCommandBuffer from command_queue
        // 2. Create MTLComputeCommandEncoder
        // 3. Set compute pipeline state
        // 4. Set buffers with setBuffer(_:offset:index:)
        // 5. Set threadgroup memory if needed
        // 6. Call dispatchThreadgroups(_:threadsPerThreadgroup:)
        // 7. End encoding
        // 8. Commit command buffer

        Ok(())
    }

    /// Dispatch kernel with automatic threadgroup sizing
    pub fn dispatch_threads(
        &self,
        kernel: &MetalKernel,
        buffers: &[&MetalBuffer],
        total_threads: (u32, u32, u32),
    ) -> Result<(), MetalError> {
        // Calculate optimal threadgroup size
        let threads_per_threadgroup = self.calculate_threadgroup_size(kernel, total_threads);

        // Calculate number of threadgroups
        let threadgroups = (
            total_threads.0.div_ceil(threads_per_threadgroup.0),
            total_threads.1.div_ceil(threads_per_threadgroup.1),
            total_threads.2.div_ceil(threads_per_threadgroup.2),
        );

        let dispatch_size = MetalDispatchSize::new(threadgroups, threads_per_threadgroup);
        self.dispatch_kernel(kernel, buffers, dispatch_size)
    }

    /// Calculate optimal threadgroup size for kernel
    fn calculate_threadgroup_size(
        &self,
        kernel: &MetalKernel,
        _total_threads: (u32, u32, u32),
    ) -> (u32, u32, u32) {
        // Use thread execution width as base
        let simd_width = kernel.thread_execution_width;
        let max_threads = kernel.max_total_threads_per_threadgroup;

        // Simple 1D optimal size
        let size = max_threads.min(simd_width * 8); // 8 simdgroups
        (size, 1, 1)
    }

    /// Synchronize (wait for all GPU work to complete)
    pub fn synchronize(&self) -> Result<(), MetalError> {
        // In real implementation:
        // 1. Create a command buffer
        // 2. Commit and waitUntilCompleted

        Ok(())
    }

    /// Create a command buffer for manual control
    pub fn create_command_buffer(&self) -> Result<MetalCommandBuffer, MetalError> {
        Ok(MetalCommandBuffer {
            handle: ptr::null_mut(),
            committed: false,
        })
    }

    /// Extract function names from MSL source (simple parser)
    fn extract_function_names(source: &str) -> Vec<String> {
        let mut functions = Vec::new();

        // Look for kernel functions
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("kernel void") || trimmed.starts_with("[[kernel]]") {
                // Extract function name
                if let Some(start) = trimmed.find("void ") {
                    let rest = &trimmed[start + 5..];
                    if let Some(end) = rest.find('(') {
                        let name = rest[..end].trim();
                        functions.push(name.to_string());
                    }
                }
            }
        }

        functions
    }
}

/// Metal command buffer for explicit control
#[derive(Debug)]
pub struct MetalCommandBuffer {
    handle: *mut c_void,
    committed: bool,
}

impl MetalCommandBuffer {
    pub fn is_committed(&self) -> bool {
        self.committed
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle
    }
}

/// Metal-specific errors
#[derive(Debug)]
pub enum MetalError {
    DeviceNotFound,
    CompilationFailed(String),
    FunctionNotFound(String),
    PipelineCreationFailed(String),
    BufferAllocationFailed,
    BufferTooSmall,
    BufferTooLarge,
    InvalidSize,
    InvalidStorageMode,
    InvalidDispatchSize(String),
    ThreadgroupMemoryExceeded,
    CommandBufferFailed,
    ExecutionFailed(String),
    Timeout,
    GpuError(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::DeviceNotFound => write!(f, "Metal device not found"),
            MetalError::CompilationFailed(msg) => write!(f, "MSL compilation failed: {}", msg),
            MetalError::FunctionNotFound(name) => {
                write!(f, "Function '{}' not found in library", name)
            }
            MetalError::PipelineCreationFailed(msg) => {
                write!(f, "Pipeline creation failed: {}", msg)
            }
            MetalError::BufferAllocationFailed => write!(f, "Buffer allocation failed"),
            MetalError::BufferTooSmall => write!(f, "Buffer too small"),
            MetalError::BufferTooLarge => write!(f, "Buffer exceeds maximum size"),
            MetalError::InvalidSize => write!(f, "Invalid size"),
            MetalError::InvalidStorageMode => write!(f, "Invalid storage mode for operation"),
            MetalError::InvalidDispatchSize(msg) => write!(f, "Invalid dispatch size: {}", msg),
            MetalError::ThreadgroupMemoryExceeded => write!(f, "Threadgroup memory limit exceeded"),
            MetalError::CommandBufferFailed => write!(f, "Command buffer execution failed"),
            MetalError::ExecutionFailed(msg) => write!(f, "Kernel execution failed: {}", msg),
            MetalError::Timeout => write!(f, "GPU operation timed out"),
            MetalError::GpuError(msg) => write!(f, "GPU error: {}", msg),
        }
    }
}

impl std::error::Error for MetalError {}

impl From<MetalError> for GpuError {
    fn from(e: MetalError) -> Self {
        match e {
            MetalError::DeviceNotFound => GpuError::DeviceNotFound,
            MetalError::CompilationFailed(msg) => GpuError::KernelLoadFailed(msg),
            MetalError::FunctionNotFound(name) => {
                GpuError::KernelLoadFailed(format!("Function not found: {}", name))
            }
            MetalError::PipelineCreationFailed(msg) => GpuError::KernelLoadFailed(msg),
            MetalError::BufferAllocationFailed => GpuError::AllocationFailed,
            MetalError::BufferTooSmall => GpuError::BufferTooSmall,
            MetalError::BufferTooLarge => GpuError::OutOfMemory,
            MetalError::InvalidSize => GpuError::InvalidSize,
            MetalError::InvalidStorageMode => {
                GpuError::InvalidConfig("Invalid storage mode".to_string())
            }
            MetalError::InvalidDispatchSize(msg) => GpuError::InvalidConfig(msg),
            MetalError::ThreadgroupMemoryExceeded => {
                GpuError::InvalidConfig("Threadgroup memory exceeded".to_string())
            }
            MetalError::CommandBufferFailed => GpuError::LaunchFailed,
            MetalError::ExecutionFailed(msg) => GpuError::DriverError(msg),
            MetalError::Timeout => GpuError::SyncFailed,
            MetalError::GpuError(msg) => GpuError::DriverError(msg),
        }
    }
}

/// High-level epistemic kernel runner for Metal
pub struct EpistemicMetalRunner {
    runtime: MetalRuntime,
    epistemic_library: Option<MetalLibrary>,
}

impl EpistemicMetalRunner {
    /// Create new epistemic runner
    pub fn new(gpu_family: MetalGpuFamily) -> Result<Self, MetalError> {
        let runtime = MetalRuntime::new(gpu_family)?;
        Ok(Self {
            runtime,
            epistemic_library: None,
        })
    }

    /// Load epistemic MSL library
    pub fn load_epistemic_library(&mut self, msl_source: &str) -> Result<(), MetalError> {
        let library = self.runtime.compile_msl(msl_source)?;
        self.epistemic_library = Some(library);
        Ok(())
    }

    /// Run epistemic addition kernel
    pub fn epistemic_add(
        &mut self,
        a: &[f32],
        epsilon_a: &[f32],
        b: &[f32],
        epsilon_b: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
        let library = self
            .epistemic_library
            .as_ref()
            .ok_or(MetalError::FunctionNotFound(
                "epistemic library not loaded".to_string(),
            ))?;

        let kernel = self.runtime.get_kernel(library, "epistemic_add_kernel")?;

        let n = a.len();
        let storage = MetalStorageMode::Shared;

        // Allocate buffers
        let buf_a = self.runtime.alloc_typed::<f32>(n, storage)?;
        let buf_eps_a = self.runtime.alloc_typed::<f32>(n, storage)?;
        let buf_b = self.runtime.alloc_typed::<f32>(n, storage)?;
        let buf_eps_b = self.runtime.alloc_typed::<f32>(n, storage)?;
        let buf_out = self.runtime.alloc_typed::<f32>(n, storage)?;
        let buf_eps_out = self.runtime.alloc_typed::<f32>(n, storage)?;

        // Copy inputs
        self.runtime.copy_to_buffer(&buf_a, a)?;
        self.runtime.copy_to_buffer(&buf_eps_a, epsilon_a)?;
        self.runtime.copy_to_buffer(&buf_b, b)?;
        self.runtime.copy_to_buffer(&buf_eps_b, epsilon_b)?;

        // Dispatch
        let dispatch = MetalDispatchSize::new_1d((n as u32).div_ceil(256), 256);
        self.runtime.dispatch_kernel(
            &kernel,
            &[
                &buf_a,
                &buf_eps_a,
                &buf_b,
                &buf_eps_b,
                &buf_out,
                &buf_eps_out,
            ],
            dispatch,
        )?;

        // Synchronize and read results
        self.runtime.synchronize()?;

        let result = self.runtime.copy_from_buffer(&buf_out, n)?;
        let epsilon_out = self.runtime.copy_from_buffer(&buf_eps_out, n)?;

        // Free buffers
        self.runtime.free_buffer(buf_a)?;
        self.runtime.free_buffer(buf_eps_a)?;
        self.runtime.free_buffer(buf_b)?;
        self.runtime.free_buffer(buf_eps_b)?;
        self.runtime.free_buffer(buf_out)?;
        self.runtime.free_buffer(buf_eps_out)?;

        Ok((result, epsilon_out))
    }

    /// Get runtime reference
    pub fn runtime(&self) -> &MetalRuntime {
        &self.runtime
    }

    /// Get mutable runtime reference
    pub fn runtime_mut(&mut self) -> &mut MetalRuntime {
        &mut self.runtime
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_runtime_creation() {
        let runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();
        assert_eq!(runtime.gpu_family(), MetalGpuFamily::Apple8);
        assert_eq!(runtime.allocated_bytes(), 0);
    }

    #[test]
    fn test_device_info() {
        let info = MetalDeviceInfo::default_for(MetalGpuFamily::Apple9);
        assert_eq!(info.family, MetalGpuFamily::Apple9);
        assert!(info.supports_simdgroup_matrix);
        assert!(info.supports_apple_silicon_features);
    }

    #[test]
    fn test_buffer_allocation() {
        let mut runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();

        let buffer = runtime
            .alloc_buffer(1024, MetalStorageMode::Shared)
            .unwrap();
        assert_eq!(buffer.size(), 1024);
        assert!(buffer.is_valid());
        assert_eq!(runtime.allocated_bytes(), 1024);

        runtime.free_buffer(buffer).unwrap();
        assert_eq!(runtime.allocated_bytes(), 0);
    }

    #[test]
    fn test_storage_mode_recommendation() {
        // Apple Silicon prefers Shared for CPU access
        assert_eq!(
            MetalStorageMode::recommended_for(MetalGpuFamily::Apple8, true),
            MetalStorageMode::Shared
        );

        // Intel Mac prefers Managed for CPU access
        assert_eq!(
            MetalStorageMode::recommended_for(MetalGpuFamily::Mac2, true),
            MetalStorageMode::Managed
        );

        // Private for no CPU access
        assert_eq!(
            MetalStorageMode::recommended_for(MetalGpuFamily::Apple8, false),
            MetalStorageMode::Private
        );
    }

    #[test]
    fn test_dispatch_size() {
        let dispatch = MetalDispatchSize::new_1d(64, 256);
        assert_eq!(dispatch.total_threads(), 64 * 256);
        assert_eq!(dispatch.total_threadgroups(), 64);
        assert_eq!(dispatch.threads_per_group(), 256);
    }

    #[test]
    fn test_compile_msl() {
        let mut runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();

        let msl = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void test_kernel(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint id [[thread_position_in_grid]]
            ) {
                output[id] = input[id] * 2.0f;
            }
        "#;

        let library = runtime.compile_msl(msl).unwrap();
        assert!(library.functions().contains(&"test_kernel".to_string()));
    }

    #[test]
    fn test_get_kernel() {
        let mut runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();

        let msl = "kernel void my_kernel() {}";
        let library = runtime.compile_msl(msl).unwrap();

        let kernel = runtime.get_kernel(&library, "my_kernel").unwrap();
        assert_eq!(kernel.name(), "my_kernel");
        assert_eq!(kernel.thread_execution_width(), 32);
    }

    #[test]
    fn test_metal_error_conversion() {
        let metal_err = MetalError::DeviceNotFound;
        let gpu_err: GpuError = metal_err.into();

        match gpu_err {
            GpuError::DeviceNotFound => {}
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_epistemic_runner() {
        let runner = EpistemicMetalRunner::new(MetalGpuFamily::Apple8).unwrap();
        assert_eq!(runner.runtime().gpu_family(), MetalGpuFamily::Apple8);
    }
}
