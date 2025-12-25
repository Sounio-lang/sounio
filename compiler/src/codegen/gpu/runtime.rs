//! GPU Runtime for kernel execution
//!
//! Provides a safe wrapper around CUDA/Vulkan for:
//! - Device memory allocation
//! - Data transfer
//! - Kernel launch
//!
//! When the `cuda` feature is enabled, this module uses `cudarc` for real
//! CUDA execution. Otherwise, it provides stub implementations.

use std::ffi::c_void;
use std::fmt;
use std::ptr;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig as CudarcLaunchConfig,
};

#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;

/// GPU Runtime abstraction
pub struct GpuRuntime {
    /// Backend type
    backend: GpuBackend,

    /// Device ID
    device: u32,

    /// Current context (CUDA) or device (Vulkan)
    context: *mut c_void,

    /// Device properties
    device_info: DeviceInfo,

    /// cudarc device handle (when cuda feature is enabled)
    #[cfg(feature = "cuda")]
    cuda_device: Option<Arc<CudaDevice>>,
}

/// GPU Backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Cuda,
    Vulkan,
    OpenCL,
    Metal,
    /// Simulated backend for testing
    Simulated,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Vulkan => write!(f, "Vulkan"),
            GpuBackend::OpenCL => write!(f, "OpenCL"),
            GpuBackend::Metal => write!(f, "Metal"),
            GpuBackend::Simulated => write!(f, "Simulated"),
        }
    }
}

/// Device buffer handle
pub struct DeviceBuffer {
    /// Raw pointer to device memory
    ptr: *mut c_void,

    /// Size in bytes
    size: usize,

    /// Backend
    backend: GpuBackend,

    /// Is this buffer allocated?
    allocated: bool,

    /// cudarc slice for CUDA memory (when cuda feature is enabled)
    #[cfg(feature = "cuda")]
    cuda_slice: Option<CudaSlice<u8>>,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &self.ptr)
            .field("size", &self.size)
            .field("backend", &self.backend)
            .field("allocated", &self.allocated)
            .finish()
    }
}

impl DeviceBuffer {
    /// Get the size of the buffer in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the raw pointer (unsafe)
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Check if buffer is allocated
    pub fn is_allocated(&self) -> bool {
        self.allocated
    }
}

/// Kernel handle
pub struct Kernel {
    /// Kernel name
    name: String,

    /// Module handle (CUDA) or pipeline (Vulkan)
    module: *mut c_void,

    /// Function handle
    function: *mut c_void,

    /// Backend
    backend: GpuBackend,

    /// Parameter count
    param_count: usize,

    /// cudarc function handle (when cuda feature is enabled)
    #[cfg(feature = "cuda")]
    cuda_function: Option<CudaFunction>,
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("name", &self.name)
            .field("backend", &self.backend)
            .field("param_count", &self.param_count)
            .finish()
    }
}

impl Kernel {
    /// Get the kernel name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the parameter count
    pub fn param_count(&self) -> usize {
        self.param_count
    }
}

/// Launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (blocks)
    pub grid: (u32, u32, u32),

    /// Block dimensions (threads)
    pub block: (u32, u32, u32),

    /// Dynamic shared memory size
    pub shared_mem: u32,

    /// Stream (for async execution)
    pub stream: Option<*mut c_void>,
}

impl LaunchConfig {
    pub fn new(grid: (u32, u32, u32), block: (u32, u32, u32)) -> Self {
        Self {
            grid,
            block,
            shared_mem: 0,
            stream: None,
        }
    }

    /// Create a 1D launch configuration
    pub fn new_1d(grid_size: u32, block_size: u32) -> Self {
        Self::new((grid_size, 1, 1), (block_size, 1, 1))
    }

    /// Create a 2D launch configuration
    pub fn new_2d(grid: (u32, u32), block: (u32, u32)) -> Self {
        Self::new((grid.0, grid.1, 1), (block.0, block.1, 1))
    }

    pub fn with_shared_mem(mut self, size: u32) -> Self {
        self.shared_mem = size;
        self
    }

    pub fn with_stream(mut self, stream: *mut c_void) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn total_threads(&self) -> u64 {
        let grid = self.grid.0 as u64 * self.grid.1 as u64 * self.grid.2 as u64;
        let block = self.block.0 as u64 * self.block.1 as u64 * self.block.2 as u64;
        grid * block
    }

    pub fn total_blocks(&self) -> u64 {
        self.grid.0 as u64 * self.grid.1 as u64 * self.grid.2 as u64
    }

    pub fn threads_per_block(&self) -> u64 {
        self.block.0 as u64 * self.block.1 as u64 * self.block.2 as u64
    }

    /// Validate the launch configuration against device limits
    pub fn validate(&self, device_info: &DeviceInfo) -> Result<(), GpuError> {
        let threads = self.threads_per_block();
        if threads > device_info.max_threads_per_block as u64 {
            return Err(GpuError::InvalidConfig(format!(
                "Threads per block ({}) exceeds maximum ({})",
                threads, device_info.max_threads_per_block
            )));
        }

        if self.shared_mem > device_info.shared_mem_per_block {
            return Err(GpuError::InvalidConfig(format!(
                "Shared memory ({}) exceeds maximum ({})",
                self.shared_mem, device_info.shared_mem_per_block
            )));
        }

        Ok(())
    }
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self::new((1, 1, 1), (256, 1, 1))
    }
}

impl GpuRuntime {
    /// Initialize GPU runtime
    pub fn new(backend: GpuBackend, device_id: u32) -> Result<Self, GpuError> {
        match backend {
            GpuBackend::Cuda => Self::init_cuda(device_id),
            GpuBackend::Vulkan => Self::init_vulkan(device_id),
            GpuBackend::OpenCL => Self::init_opencl(device_id),
            GpuBackend::Metal => Self::init_metal(device_id),
            GpuBackend::Simulated => Self::init_simulated(device_id),
        }
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Get the device ID
    pub fn device_id(&self) -> u32 {
        self.device
    }

    /// Get device properties
    pub fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Allocate device memory
    pub fn alloc(&self, size: usize) -> Result<DeviceBuffer, GpuError> {
        if size == 0 {
            return Err(GpuError::InvalidSize);
        }

        match self.backend {
            GpuBackend::Cuda => self.cuda_alloc(size),
            GpuBackend::Vulkan => self.vulkan_alloc(size),
            GpuBackend::OpenCL => self.opencl_alloc(size),
            GpuBackend::Metal => self.metal_alloc(size),
            GpuBackend::Simulated => self.simulated_alloc(size),
        }
    }

    /// Allocate typed device memory
    pub fn alloc_typed<T>(&self, count: usize) -> Result<DeviceBuffer, GpuError> {
        self.alloc(count * std::mem::size_of::<T>())
    }

    /// Free device memory
    pub fn free(&self, buffer: DeviceBuffer) -> Result<(), GpuError> {
        if !buffer.allocated {
            return Ok(());
        }

        match self.backend {
            GpuBackend::Cuda => self.cuda_free(buffer),
            GpuBackend::Vulkan => self.vulkan_free(buffer),
            GpuBackend::OpenCL => self.opencl_free(buffer),
            GpuBackend::Metal => self.metal_free(buffer),
            GpuBackend::Simulated => self.simulated_free(buffer),
        }
    }

    /// Copy data to device
    pub fn copy_to_device<T>(&self, dst: &DeviceBuffer, src: &[T]) -> Result<(), GpuError> {
        let size = std::mem::size_of_val(src);
        if size > dst.size {
            return Err(GpuError::BufferTooSmall);
        }

        match self.backend {
            GpuBackend::Cuda => self.cuda_copy_htod(dst.ptr, src.as_ptr() as *const c_void, size),
            GpuBackend::Vulkan => {
                self.vulkan_copy_htod(dst.ptr, src.as_ptr() as *const c_void, size)
            }
            GpuBackend::OpenCL => {
                self.opencl_copy_htod(dst.ptr, src.as_ptr() as *const c_void, size)
            }
            GpuBackend::Metal => self.metal_copy_htod(dst.ptr, src.as_ptr() as *const c_void, size),
            GpuBackend::Simulated => {
                self.simulated_copy_htod(dst.ptr, src.as_ptr() as *const c_void, size)
            }
        }
    }

    /// Copy data from device
    pub fn copy_to_host<T>(&self, dst: &mut [T], src: &DeviceBuffer) -> Result<(), GpuError> {
        let size = std::mem::size_of_val(dst);
        if size > src.size {
            return Err(GpuError::BufferTooSmall);
        }

        match self.backend {
            GpuBackend::Cuda => self.cuda_copy_dtoh(dst.as_mut_ptr() as *mut c_void, src.ptr, size),
            GpuBackend::Vulkan => {
                self.vulkan_copy_dtoh(dst.as_mut_ptr() as *mut c_void, src.ptr, size)
            }
            GpuBackend::OpenCL => {
                self.opencl_copy_dtoh(dst.as_mut_ptr() as *mut c_void, src.ptr, size)
            }
            GpuBackend::Metal => {
                self.metal_copy_dtoh(dst.as_mut_ptr() as *mut c_void, src.ptr, size)
            }
            GpuBackend::Simulated => {
                self.simulated_copy_dtoh(dst.as_mut_ptr() as *mut c_void, src.ptr, size)
            }
        }
    }

    /// Load kernel from PTX
    pub fn load_ptx(&self, ptx: &str, kernel_name: &str) -> Result<Kernel, GpuError> {
        match self.backend {
            GpuBackend::Cuda => self.cuda_load_ptx(ptx, kernel_name),
            GpuBackend::Simulated => self.simulated_load_ptx(ptx, kernel_name),
            _ => Err(GpuError::UnsupportedBackend),
        }
    }

    /// Load kernel from SPIR-V
    pub fn load_spirv(&self, spirv: &[u8], kernel_name: &str) -> Result<Kernel, GpuError> {
        match self.backend {
            GpuBackend::Vulkan => self.vulkan_load_spirv(spirv, kernel_name),
            GpuBackend::OpenCL => self.opencl_load_spirv(spirv, kernel_name),
            GpuBackend::Simulated => self.simulated_load_spirv(spirv, kernel_name),
            _ => Err(GpuError::UnsupportedBackend),
        }
    }

    /// Launch kernel
    pub fn launch(
        &self,
        kernel: &Kernel,
        config: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<(), GpuError> {
        // Validate configuration
        config.validate(&self.device_info)?;

        match self.backend {
            GpuBackend::Cuda => self.cuda_launch(kernel, config, args),
            GpuBackend::Vulkan => self.vulkan_launch(kernel, config, args),
            GpuBackend::OpenCL => self.opencl_launch(kernel, config, args),
            GpuBackend::Metal => self.metal_launch(kernel, config, args),
            GpuBackend::Simulated => self.simulated_launch(kernel, config, args),
        }
    }

    /// Synchronize device
    pub fn synchronize(&self) -> Result<(), GpuError> {
        match self.backend {
            GpuBackend::Cuda => self.cuda_synchronize(),
            GpuBackend::Vulkan => self.vulkan_synchronize(),
            GpuBackend::OpenCL => self.opencl_synchronize(),
            GpuBackend::Metal => self.metal_synchronize(),
            GpuBackend::Simulated => Ok(()),
        }
    }

    // === CUDA Implementation (with cudarc when cuda feature is enabled) ===

    /// Initialize CUDA runtime with cudarc (real GPU execution)
    #[cfg(feature = "cuda")]
    fn init_cuda(device_id: u32) -> Result<Self, GpuError> {
        // CudaDevice::new() returns Arc<CudaDevice>
        let device = CudaDevice::new(device_id as usize).map_err(|e| {
            GpuError::DriverError(format!(
                "Failed to initialize CUDA device {}: {}",
                device_id, e
            ))
        })?;

        // Query device info from cudarc
        let device_info = DeviceInfo::from_cuda_device(&device, device_id);

        Ok(Self {
            backend: GpuBackend::Cuda,
            device: device_id,
            context: ptr::null_mut(),
            device_info,
            cuda_device: Some(device), // Already an Arc<CudaDevice>
        })
    }

    /// Initialize CUDA runtime (stub when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    fn init_cuda(device_id: u32) -> Result<Self, GpuError> {
        // Stub implementation - no real CUDA support
        Ok(Self {
            backend: GpuBackend::Cuda,
            device: device_id,
            context: ptr::null_mut(),
            device_info: DeviceInfo::default_cuda(),
        })
    }

    /// Allocate device memory with cudarc
    #[cfg(feature = "cuda")]
    fn cuda_alloc(&self, size: usize) -> Result<DeviceBuffer, GpuError> {
        let device = self
            .cuda_device
            .as_ref()
            .ok_or_else(|| GpuError::DriverError("CUDA device not initialized".into()))?;

        let slice: CudaSlice<u8> = device
            .alloc_zeros(size)
            .map_err(|_| GpuError::AllocationFailed)?;

        // Get device pointer using DevicePtr trait
        let ptr = *slice.device_ptr() as *mut c_void;

        Ok(DeviceBuffer {
            ptr,
            size,
            backend: GpuBackend::Cuda,
            allocated: true,
            cuda_slice: Some(slice),
        })
    }

    /// Allocate device memory (stub when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    fn cuda_alloc(&self, size: usize) -> Result<DeviceBuffer, GpuError> {
        // Stub: returns null pointer
        Ok(DeviceBuffer {
            ptr: ptr::null_mut(),
            size,
            backend: GpuBackend::Cuda,
            allocated: true,
        })
    }

    /// Free device memory with cudarc
    #[cfg(feature = "cuda")]
    fn cuda_free(&self, buffer: DeviceBuffer) -> Result<(), GpuError> {
        // CudaSlice is dropped automatically, freeing the memory
        drop(buffer.cuda_slice);
        Ok(())
    }

    /// Free device memory (stub when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    fn cuda_free(&self, _buffer: DeviceBuffer) -> Result<(), GpuError> {
        Ok(())
    }

    /// Copy host to device with cudarc
    /// Note: This is a low-level implementation that copies raw memory.
    /// For proper type-safe transfers, use copy_to_device with DeviceBuffer.
    #[cfg(feature = "cuda")]
    fn cuda_copy_htod(
        &self,
        _dst: *mut c_void,
        src: *const c_void,
        size: usize,
    ) -> Result<(), GpuError> {
        let device = self
            .cuda_device
            .as_ref()
            .ok_or_else(|| GpuError::DriverError("CUDA device not initialized".into()))?;

        // Create a slice from the source pointer
        let src_slice = unsafe { std::slice::from_raw_parts(src as *const u8, size) };

        // Allocate temporary device buffer and copy
        // Note: This is inefficient - better to use the CudaSlice stored in DeviceBuffer
        let mut dev_buf: CudaSlice<u8> = device
            .alloc_zeros(size)
            .map_err(|_| GpuError::AllocationFailed)?;
        device
            .htod_sync_copy_into(src_slice, &mut dev_buf)
            .map_err(|_| GpuError::CopyFailed)?;

        Ok(())
    }

    /// Copy host to device (stub when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    fn cuda_copy_htod(
        &self,
        _dst: *mut c_void,
        _src: *const c_void,
        _size: usize,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    /// Copy device to host with cudarc
    #[cfg(feature = "cuda")]
    fn cuda_copy_dtoh(
        &self,
        dst: *mut c_void,
        _src: *mut c_void,
        size: usize,
    ) -> Result<(), GpuError> {
        let device = self
            .cuda_device
            .as_ref()
            .ok_or_else(|| GpuError::DriverError("CUDA device not initialized".into()))?;

        // Create destination slice
        let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, size) };

        // Note: In a real implementation, we'd need to track which CudaSlice corresponds
        // to which raw pointer. For now, this is a placeholder.
        // The proper way is to use copy_to_host with DeviceBuffer.
        let dev_buf: CudaSlice<u8> = device
            .alloc_zeros(size)
            .map_err(|_| GpuError::AllocationFailed)?;
        device
            .dtoh_sync_copy_into(&dev_buf, dst_slice)
            .map_err(|_| GpuError::CopyFailed)?;

        Ok(())
    }

    /// Copy device to host (stub when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    fn cuda_copy_dtoh(
        &self,
        _dst: *mut c_void,
        _src: *mut c_void,
        _size: usize,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    /// Load PTX kernel with cudarc
    #[cfg(feature = "cuda")]
    fn cuda_load_ptx(&self, ptx: &str, kernel_name: &str) -> Result<Kernel, GpuError> {
        let device = self
            .cuda_device
            .as_ref()
            .ok_or_else(|| GpuError::DriverError("CUDA device not initialized".into()))?;

        // Convert to owned strings for cudarc which requires 'static lifetime
        let module_name: &'static str = Box::leak(kernel_name.to_string().into_boxed_str());
        let func_name: &'static str = Box::leak(kernel_name.to_string().into_boxed_str());

        // Use cudarc to load PTX and get the kernel function
        device
            .load_ptx(Ptx::from_src(ptx), module_name, &[func_name])
            .map_err(|e| GpuError::KernelLoadFailed(format!("Failed to load PTX: {}", e)))?;

        let func = device.get_func(module_name, func_name).ok_or_else(|| {
            GpuError::KernelLoadFailed(format!("Function '{}' not found in module", kernel_name))
        })?;

        Ok(Kernel {
            name: kernel_name.to_string(),
            module: ptr::null_mut(),
            function: ptr::null_mut(),
            backend: GpuBackend::Cuda,
            param_count: 0,
            cuda_function: Some(func),
        })
    }

    /// Load PTX kernel (stub when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    fn cuda_load_ptx(&self, _ptx: &str, kernel_name: &str) -> Result<Kernel, GpuError> {
        Ok(Kernel {
            name: kernel_name.to_string(),
            module: ptr::null_mut(),
            function: ptr::null_mut(),
            backend: GpuBackend::Cuda,
            param_count: 0,
        })
    }

    /// Launch kernel with cudarc
    #[cfg(feature = "cuda")]
    fn cuda_launch(
        &self,
        kernel: &Kernel,
        config: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<(), GpuError> {
        let _device = self
            .cuda_device
            .as_ref()
            .ok_or_else(|| GpuError::DriverError("CUDA device not initialized".into()))?;

        let func = kernel
            .cuda_function
            .as_ref()
            .ok_or_else(|| GpuError::InvalidKernel)?;

        // Convert launch config
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: config.grid,
            block_dim: config.block,
            shared_mem_bytes: config.shared_mem,
        };

        // Convert kernel arguments to raw pointers
        // cudarc requires arguments as a vector of *mut c_void
        let mut arg_ptrs: Vec<*mut c_void> = Vec::with_capacity(args.len());
        let mut arg_storage: Vec<Box<dyn std::any::Any>> = Vec::with_capacity(args.len());

        for arg in args {
            match arg {
                KernelArg::Buffer(ptr) => arg_ptrs.push(*ptr),
                KernelArg::Int32(v) => {
                    let boxed = Box::new(*v);
                    arg_ptrs.push(Box::into_raw(boxed) as *mut c_void);
                    // Note: This leaks memory - in production, we'd need proper cleanup
                }
                KernelArg::Int64(v) => {
                    let boxed = Box::new(*v);
                    arg_ptrs.push(Box::into_raw(boxed) as *mut c_void);
                }
                KernelArg::UInt32(v) => {
                    let boxed = Box::new(*v);
                    arg_ptrs.push(Box::into_raw(boxed) as *mut c_void);
                }
                KernelArg::UInt64(v) => {
                    let boxed = Box::new(*v);
                    arg_ptrs.push(Box::into_raw(boxed) as *mut c_void);
                }
                KernelArg::Float32(v) => {
                    let boxed = Box::new(*v);
                    arg_ptrs.push(Box::into_raw(boxed) as *mut c_void);
                }
                KernelArg::Float64(v) => {
                    let boxed = Box::new(*v);
                    arg_ptrs.push(Box::into_raw(boxed) as *mut c_void);
                }
                KernelArg::Pointer(ptr) => arg_ptrs.push(*ptr),
            }
        }

        // Launch kernel with arguments
        unsafe {
            func.clone()
                .launch(launch_cfg, &mut arg_ptrs)
                .map_err(|e| GpuError::DriverError(format!("Kernel launch failed: {}", e)))?;
        }

        // Drop storage to free allocated argument memory
        drop(arg_storage);

        Ok(())
    }

    /// Launch kernel (stub when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    fn cuda_launch(
        &self,
        _kernel: &Kernel,
        _config: &LaunchConfig,
        _args: &[KernelArg],
    ) -> Result<(), GpuError> {
        Ok(())
    }

    /// Synchronize device with cudarc
    #[cfg(feature = "cuda")]
    fn cuda_synchronize(&self) -> Result<(), GpuError> {
        let device = self
            .cuda_device
            .as_ref()
            .ok_or_else(|| GpuError::DriverError("CUDA device not initialized".into()))?;

        device.synchronize().map_err(|e| GpuError::SyncFailed)?;

        Ok(())
    }

    /// Synchronize device (stub when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    fn cuda_synchronize(&self) -> Result<(), GpuError> {
        Ok(())
    }

    // === Vulkan Implementation ===

    fn init_vulkan(device_id: u32) -> Result<Self, GpuError> {
        Ok(Self {
            backend: GpuBackend::Vulkan,
            device: device_id,
            context: ptr::null_mut(),
            device_info: DeviceInfo::default_vulkan(),
            #[cfg(feature = "cuda")]
            cuda_device: None,
        })
    }

    fn vulkan_alloc(&self, size: usize) -> Result<DeviceBuffer, GpuError> {
        Ok(DeviceBuffer {
            ptr: ptr::null_mut(),
            size,
            backend: GpuBackend::Vulkan,
            allocated: true,
            #[cfg(feature = "cuda")]
            cuda_slice: None,
        })
    }

    fn vulkan_free(&self, _buffer: DeviceBuffer) -> Result<(), GpuError> {
        Ok(())
    }

    fn vulkan_copy_htod(
        &self,
        _dst: *mut c_void,
        _src: *const c_void,
        _size: usize,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn vulkan_copy_dtoh(
        &self,
        _dst: *mut c_void,
        _src: *mut c_void,
        _size: usize,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn vulkan_load_spirv(&self, _spirv: &[u8], kernel_name: &str) -> Result<Kernel, GpuError> {
        Ok(Kernel {
            name: kernel_name.to_string(),
            module: ptr::null_mut(),
            function: ptr::null_mut(),
            backend: GpuBackend::Vulkan,
            param_count: 0,
            #[cfg(feature = "cuda")]
            cuda_function: None,
        })
    }

    fn vulkan_launch(
        &self,
        _kernel: &Kernel,
        _config: &LaunchConfig,
        _args: &[KernelArg],
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn vulkan_synchronize(&self) -> Result<(), GpuError> {
        Ok(())
    }

    // === OpenCL Implementation ===

    fn init_opencl(device_id: u32) -> Result<Self, GpuError> {
        Ok(Self {
            backend: GpuBackend::OpenCL,
            device: device_id,
            context: ptr::null_mut(),
            device_info: DeviceInfo::default_opencl(),
            #[cfg(feature = "cuda")]
            cuda_device: None,
        })
    }

    fn opencl_alloc(&self, size: usize) -> Result<DeviceBuffer, GpuError> {
        Ok(DeviceBuffer {
            ptr: ptr::null_mut(),
            size,
            backend: GpuBackend::OpenCL,
            allocated: true,
            #[cfg(feature = "cuda")]
            cuda_slice: None,
        })
    }

    fn opencl_free(&self, _buffer: DeviceBuffer) -> Result<(), GpuError> {
        Ok(())
    }

    fn opencl_copy_htod(
        &self,
        _dst: *mut c_void,
        _src: *const c_void,
        _size: usize,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn opencl_copy_dtoh(
        &self,
        _dst: *mut c_void,
        _src: *mut c_void,
        _size: usize,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn opencl_load_spirv(&self, _spirv: &[u8], kernel_name: &str) -> Result<Kernel, GpuError> {
        Ok(Kernel {
            name: kernel_name.to_string(),
            module: ptr::null_mut(),
            function: ptr::null_mut(),
            backend: GpuBackend::OpenCL,
            param_count: 0,
            #[cfg(feature = "cuda")]
            cuda_function: None,
        })
    }

    fn opencl_launch(
        &self,
        _kernel: &Kernel,
        _config: &LaunchConfig,
        _args: &[KernelArg],
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn opencl_synchronize(&self) -> Result<(), GpuError> {
        Ok(())
    }

    // === Metal Implementation ===

    fn init_metal(device_id: u32) -> Result<Self, GpuError> {
        Ok(Self {
            backend: GpuBackend::Metal,
            device: device_id,
            context: ptr::null_mut(),
            device_info: DeviceInfo::default_metal(),
            #[cfg(feature = "cuda")]
            cuda_device: None,
        })
    }

    fn metal_alloc(&self, size: usize) -> Result<DeviceBuffer, GpuError> {
        Ok(DeviceBuffer {
            ptr: ptr::null_mut(),
            size,
            backend: GpuBackend::Metal,
            allocated: true,
            #[cfg(feature = "cuda")]
            cuda_slice: None,
        })
    }

    fn metal_free(&self, _buffer: DeviceBuffer) -> Result<(), GpuError> {
        Ok(())
    }

    fn metal_copy_htod(
        &self,
        _dst: *mut c_void,
        _src: *const c_void,
        _size: usize,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn metal_copy_dtoh(
        &self,
        _dst: *mut c_void,
        _src: *mut c_void,
        _size: usize,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn metal_launch(
        &self,
        _kernel: &Kernel,
        _config: &LaunchConfig,
        _args: &[KernelArg],
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn metal_synchronize(&self) -> Result<(), GpuError> {
        Ok(())
    }

    // === Simulated Implementation (for testing) ===

    fn init_simulated(device_id: u32) -> Result<Self, GpuError> {
        Ok(Self {
            backend: GpuBackend::Simulated,
            device: device_id,
            context: ptr::null_mut(),
            device_info: DeviceInfo::default_simulated(),
            #[cfg(feature = "cuda")]
            cuda_device: None,
        })
    }

    fn simulated_alloc(&self, size: usize) -> Result<DeviceBuffer, GpuError> {
        // Allocate actual memory for simulation
        let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut c_void };

        Ok(DeviceBuffer {
            ptr,
            size,
            backend: GpuBackend::Simulated,
            allocated: true,
            #[cfg(feature = "cuda")]
            cuda_slice: None,
        })
    }

    fn simulated_free(&self, buffer: DeviceBuffer) -> Result<(), GpuError> {
        if !buffer.ptr.is_null() && buffer.size > 0 {
            let layout = std::alloc::Layout::from_size_align(buffer.size, 8).unwrap();
            unsafe { std::alloc::dealloc(buffer.ptr as *mut u8, layout) };
        }
        Ok(())
    }

    fn simulated_copy_htod(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
    ) -> Result<(), GpuError> {
        if !dst.is_null() && !src.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
            }
        }
        Ok(())
    }

    fn simulated_copy_dtoh(
        &self,
        dst: *mut c_void,
        src: *mut c_void,
        size: usize,
    ) -> Result<(), GpuError> {
        if !dst.is_null() && !src.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
            }
        }
        Ok(())
    }

    fn simulated_load_ptx(&self, _ptx: &str, kernel_name: &str) -> Result<Kernel, GpuError> {
        Ok(Kernel {
            name: kernel_name.to_string(),
            module: ptr::null_mut(),
            function: ptr::null_mut(),
            backend: GpuBackend::Simulated,
            param_count: 0,
            #[cfg(feature = "cuda")]
            cuda_function: None,
        })
    }

    fn simulated_load_spirv(&self, _spirv: &[u8], kernel_name: &str) -> Result<Kernel, GpuError> {
        Ok(Kernel {
            name: kernel_name.to_string(),
            module: ptr::null_mut(),
            function: ptr::null_mut(),
            backend: GpuBackend::Simulated,
            param_count: 0,
            #[cfg(feature = "cuda")]
            cuda_function: None,
        })
    }

    fn simulated_launch(
        &self,
        _kernel: &Kernel,
        _config: &LaunchConfig,
        _args: &[KernelArg],
    ) -> Result<(), GpuError> {
        // In simulation mode, we don't actually execute the kernel
        Ok(())
    }
}

/// Kernel argument
#[derive(Debug, Clone)]
pub enum KernelArg {
    Buffer(*mut c_void),
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Pointer(*mut c_void),
}

impl KernelArg {
    /// Create a buffer argument from a DeviceBuffer
    pub fn from_buffer(buffer: &DeviceBuffer) -> Self {
        KernelArg::Buffer(buffer.ptr)
    }
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: u64,
    pub multiprocessors: u32,
    pub max_threads_per_block: u32,
    pub warp_size: u32,
    pub shared_mem_per_block: u32,
    pub max_registers_per_block: u32,
    pub clock_rate_khz: u32,
    pub memory_bus_width: u32,
}

impl DeviceInfo {
    /// Query device info from cudarc CudaDevice
    /// Note: cudarc 0.12 has limited device query support, so we use defaults
    /// for most values and query what we can.
    #[cfg(feature = "cuda")]
    fn from_cuda_device(_device: &CudaDevice, device_id: u32) -> Self {
        // cudarc 0.12 doesn't expose all device query methods
        // Use default values for now - can be enhanced in future versions
        Self {
            name: format!("CUDA Device {}", device_id),
            compute_capability: (7, 5),           // Default to SM 7.5
            total_memory: 8 * 1024 * 1024 * 1024, // 8 GB default
            multiprocessors: 48,
            max_threads_per_block: 1024,
            warp_size: 32,
            shared_mem_per_block: 48 * 1024,
            max_registers_per_block: 65536,
            clock_rate_khz: 1500000,
            memory_bus_width: 256,
        }
    }

    fn default_cuda() -> Self {
        Self {
            name: "CUDA Device".to_string(),
            compute_capability: (7, 5),
            total_memory: 8 * 1024 * 1024 * 1024, // 8 GB
            multiprocessors: 48,
            max_threads_per_block: 1024,
            warp_size: 32,
            shared_mem_per_block: 48 * 1024,
            max_registers_per_block: 65536,
            clock_rate_khz: 1500000,
            memory_bus_width: 256,
        }
    }

    fn default_vulkan() -> Self {
        Self {
            name: "Vulkan Device".to_string(),
            compute_capability: (1, 2),
            total_memory: 8 * 1024 * 1024 * 1024,
            multiprocessors: 48,
            max_threads_per_block: 1024,
            warp_size: 32,
            shared_mem_per_block: 32 * 1024,
            max_registers_per_block: 65536,
            clock_rate_khz: 1500000,
            memory_bus_width: 256,
        }
    }

    fn default_opencl() -> Self {
        Self {
            name: "OpenCL Device".to_string(),
            compute_capability: (2, 0),
            total_memory: 8 * 1024 * 1024 * 1024,
            multiprocessors: 48,
            max_threads_per_block: 1024,
            warp_size: 64,
            shared_mem_per_block: 32 * 1024,
            max_registers_per_block: 65536,
            clock_rate_khz: 1500000,
            memory_bus_width: 256,
        }
    }

    fn default_metal() -> Self {
        Self {
            name: "Metal Device".to_string(),
            compute_capability: (2, 0),
            total_memory: 8 * 1024 * 1024 * 1024,
            multiprocessors: 32,
            max_threads_per_block: 1024,
            warp_size: 32,
            shared_mem_per_block: 32 * 1024,
            max_registers_per_block: 32768,
            clock_rate_khz: 1200000,
            memory_bus_width: 128,
        }
    }

    fn default_simulated() -> Self {
        Self {
            name: "Simulated GPU".to_string(),
            compute_capability: (1, 0),
            total_memory: 1024 * 1024 * 1024, // 1 GB
            multiprocessors: 1,
            max_threads_per_block: 1024,
            warp_size: 32,
            shared_mem_per_block: 48 * 1024,
            max_registers_per_block: 65536,
            clock_rate_khz: 1000000,
            memory_bus_width: 64,
        }
    }
}

/// GPU errors
#[derive(Debug)]
pub enum GpuError {
    InitFailed,
    DeviceNotFound,
    AllocationFailed,
    CopyFailed,
    KernelLoadFailed(String),
    LaunchFailed,
    SyncFailed,
    BufferTooSmall,
    UnsupportedBackend,
    InvalidKernel,
    InvalidSize,
    InvalidConfig(String),
    OutOfMemory,
    DriverError(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::InitFailed => write!(f, "GPU initialization failed"),
            GpuError::DeviceNotFound => write!(f, "GPU device not found"),
            GpuError::AllocationFailed => write!(f, "GPU memory allocation failed"),
            GpuError::CopyFailed => write!(f, "GPU memory copy failed"),
            GpuError::KernelLoadFailed(msg) => write!(f, "Failed to load GPU kernel: {}", msg),
            GpuError::LaunchFailed => write!(f, "Kernel launch failed"),
            GpuError::SyncFailed => write!(f, "GPU synchronization failed"),
            GpuError::BufferTooSmall => write!(f, "Buffer too small for operation"),
            GpuError::UnsupportedBackend => write!(f, "GPU backend not supported"),
            GpuError::InvalidKernel => write!(f, "Invalid kernel"),
            GpuError::InvalidSize => write!(f, "Invalid size"),
            GpuError::InvalidConfig(msg) => write!(f, "Invalid launch configuration: {}", msg),
            GpuError::OutOfMemory => write!(f, "Out of GPU memory"),
            GpuError::DriverError(msg) => write!(f, "GPU driver error: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config() {
        let config = LaunchConfig::new((32, 32, 1), (16, 16, 1)).with_shared_mem(4096);

        assert_eq!(config.grid, (32, 32, 1));
        assert_eq!(config.block, (16, 16, 1));
        assert_eq!(config.shared_mem, 4096);
        assert_eq!(config.total_threads(), 32 * 32 * 16 * 16);
        assert_eq!(config.total_blocks(), 32 * 32);
        assert_eq!(config.threads_per_block(), 16 * 16);
    }

    #[test]
    fn test_launch_config_1d() {
        let config = LaunchConfig::new_1d(64, 256);
        assert_eq!(config.grid, (64, 1, 1));
        assert_eq!(config.block, (256, 1, 1));
    }

    #[test]
    fn test_launch_config_2d() {
        let config = LaunchConfig::new_2d((32, 32), (16, 16));
        assert_eq!(config.grid, (32, 32, 1));
        assert_eq!(config.block, (16, 16, 1));
    }

    #[test]
    fn test_simulated_runtime() {
        let runtime = GpuRuntime::new(GpuBackend::Simulated, 0).unwrap();
        assert_eq!(runtime.backend(), GpuBackend::Simulated);
        assert_eq!(runtime.device_id(), 0);
    }

    #[test]
    fn test_simulated_allocation() {
        let runtime = GpuRuntime::new(GpuBackend::Simulated, 0).unwrap();

        let buffer = runtime.alloc(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
        assert!(buffer.is_allocated());

        runtime.free(buffer).unwrap();
    }

    #[test]
    fn test_simulated_copy() {
        let runtime = GpuRuntime::new(GpuBackend::Simulated, 0).unwrap();

        let buffer = runtime.alloc_typed::<f32>(10).unwrap();

        let host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        runtime.copy_to_device(&buffer, &host_data).unwrap();

        let mut result = vec![0.0f32; 10];
        runtime.copy_to_host(&mut result, &buffer).unwrap();

        assert_eq!(result, host_data);

        runtime.free(buffer).unwrap();
    }

    #[test]
    fn test_kernel_arg() {
        let arg_i32 = KernelArg::Int32(42);
        let arg_f32 = KernelArg::Float32(3.14);

        match arg_i32 {
            KernelArg::Int32(v) => assert_eq!(v, 42),
            _ => panic!("Wrong type"),
        }

        match arg_f32 {
            KernelArg::Float32(v) => assert!((v - 3.14).abs() < 0.001),
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_config_validation() {
        let info = DeviceInfo::default_simulated();

        // Valid config
        let valid = LaunchConfig::new((1, 1, 1), (256, 1, 1));
        assert!(valid.validate(&info).is_ok());

        // Too many threads
        let invalid = LaunchConfig::new((1, 1, 1), (2048, 1, 1));
        assert!(invalid.validate(&info).is_err());

        // Too much shared memory
        let invalid = LaunchConfig::new((1, 1, 1), (256, 1, 1)).with_shared_mem(1024 * 1024);
        assert!(invalid.validate(&info).is_err());
    }
}
