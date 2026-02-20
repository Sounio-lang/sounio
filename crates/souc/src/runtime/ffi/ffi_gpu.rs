//! GPU FFI Functions for Sounio Runtime
//!
//! C-callable GPU dispatch functions for the Sounio effect system.
//!
//! - `__sounio_gpu_available` — probe for GPU runtime
//! - `__sounio_gpu_init` — initialize GPU backend
//! - `__sounio_gpu_alloc` — allocate device buffer
//! - `__sounio_gpu_free` — free device buffer
//! - `__sounio_gpu_copy_htod` — host-to-device copy
//! - `__sounio_gpu_copy_dtoh` — device-to-host copy
//! - `__sounio_gpu_load_ptx` — load a PTX kernel
//! - `__sounio_gpu_launch` — launch a kernel
//! - `__sounio_gpu_sync` — synchronize device

use std::process::Command;

use crate::runtime::gpu_bridge::{get_gpu_bridge, init_gpu_bridge};
use crate::codegen::gpu::runtime::{GpuBackend, KernelArg};

/// Check whether a CUDA or Vulkan GPU runtime is available.
///
/// Probes the system for `nvidia-smi` (CUDA) and `vulkaninfo` (Vulkan).
/// Returns 1 if at least one GPU runtime is detected, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_gpu_available() -> i32 {
    let span = tracing::trace_span!("ffi::gpu_available");
    let _guard = span.enter();

    // Check for CUDA via nvidia-smi
    let cuda = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output();

    if let Ok(output) = cuda {
        if output.status.success() {
            tracing::trace!("GPU available via CUDA (nvidia-smi)");
            return 1;
        }
    }

    // Check for Vulkan via vulkaninfo
    let vulkan = Command::new("vulkaninfo")
        .arg("--summary")
        .output();

    if let Ok(output) = vulkan {
        if output.status.success() {
            tracing::trace!("GPU available via Vulkan (vulkaninfo)");
            return 1;
        }
    }

    tracing::trace!("no GPU runtime detected");
    0
}

/// Initialize the GPU runtime with a specific backend.
///
/// `backend`: 0 = CUDA, 1 = Vulkan, 2 = OpenCL, 3 = Metal, 4 = Simulated
/// `device_id`: GPU device index
///
/// Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_gpu_init(backend: i32, device_id: u32) -> i32 {
    let gpu_backend = match backend {
        0 => GpuBackend::Cuda,
        1 => GpuBackend::Vulkan,
        2 => GpuBackend::OpenCL,
        3 => GpuBackend::Metal,
        4 => GpuBackend::Simulated,
        _ => return -1,
    };

    match init_gpu_bridge(gpu_backend, device_id) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Free a device buffer by ID.
///
/// Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_gpu_free(buffer_id: u64) -> i32 {
    let bridge = get_gpu_bridge();
    match bridge.lock() {
        Ok(mut b) => match b.free(buffer_id) {
            Ok(()) => 0,
            Err(_) => -1,
        },
        Err(_) => -1,
    }
}

/// Copy data from host to device buffer.
///
/// `buffer_id`: target device buffer
/// `src`: pointer to host data
/// `size`: number of bytes to copy
///
/// Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// `src` must point to at least `size` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_gpu_copy_htod(
    buffer_id: u64,
    src: *const u8,
    size: usize,
) -> i32 {
    if src.is_null() || size == 0 {
        return -1;
    }

    let data = unsafe { std::slice::from_raw_parts(src, size) };
    let bridge = get_gpu_bridge();
    match bridge.lock() {
        Ok(mut b) => match b.copy_htod(buffer_id, data) {
            Ok(()) => 0,
            Err(_) => -1,
        },
        Err(_) => -1,
    }
}

/// Copy data from device buffer to host.
///
/// `buffer_id`: source device buffer
/// `dst`: pointer to host destination
/// `size`: number of bytes to copy
///
/// Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// `dst` must point to at least `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_gpu_copy_dtoh(
    buffer_id: u64,
    dst: *mut u8,
    size: usize,
) -> i32 {
    if dst.is_null() || size == 0 {
        return -1;
    }

    let data = unsafe { std::slice::from_raw_parts_mut(dst, size) };
    let bridge = get_gpu_bridge();
    match bridge.lock() {
        Ok(mut b) => match b.copy_dtoh(buffer_id, data) {
            Ok(()) => 0,
            Err(_) => -1,
        },
        Err(_) => -1,
    }
}

/// Load a PTX kernel.
///
/// `ptx_ptr`/`ptx_len`: PTX source as UTF-8 string
/// `name_ptr`/`name_len`: kernel function name as UTF-8 string
///
/// Returns kernel ID (> 0) on success, 0 on failure.
///
/// # Safety
///
/// Both pointers must be valid UTF-8 strings of the given lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_gpu_load_ptx(
    ptx_ptr: *const u8,
    ptx_len: usize,
    name_ptr: *const u8,
    name_len: usize,
) -> u64 {
    if ptx_ptr.is_null() || name_ptr.is_null() {
        return 0;
    }

    let ptx_slice = unsafe { std::slice::from_raw_parts(ptx_ptr, ptx_len) };
    let name_slice = unsafe { std::slice::from_raw_parts(name_ptr, name_len) };

    let ptx_str = match std::str::from_utf8(ptx_slice) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let name_str = match std::str::from_utf8(name_slice) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let bridge = get_gpu_bridge();
    match bridge.lock() {
        Ok(mut b) => match b.load_ptx(ptx_str, name_str) {
            Ok(id) => id,
            Err(_) => 0,
        },
        Err(_) => 0,
    }
}

/// Launch a kernel.
///
/// `kernel_id`: kernel to launch
/// `grid_x/y/z`: grid dimensions
/// `block_x/y/z`: block dimensions
/// `args`: pointer to array of `KernelArgFFI` structs
/// `num_args`: number of arguments
///
/// Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// `args` must point to `num_args` valid `KernelArgFFI` structs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_gpu_launch(
    kernel_id: u64,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    args: *const KernelArgFFI,
    num_args: usize,
) -> i32 {
    let kernel_args: Vec<KernelArg> = if args.is_null() || num_args == 0 {
        Vec::new()
    } else {
        let ffi_args = unsafe { std::slice::from_raw_parts(args, num_args) };
        ffi_args.iter().map(|a| unsafe { a.to_kernel_arg() }).collect()
    };

    let bridge = get_gpu_bridge();
    match bridge.lock() {
        Ok(mut b) => {
            match b.launch_kernel(
                kernel_id,
                (grid_x, grid_y, grid_z),
                (block_x, block_y, block_z),
                &kernel_args,
            ) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// Synchronize the GPU device (wait for all pending operations).
///
/// Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_gpu_sync() -> i32 {
    let bridge = get_gpu_bridge();
    match bridge.lock() {
        Ok(b) => match b.synchronize() {
            Ok(()) => 0,
            Err(_) => -1,
        },
        Err(_) => -1,
    }
}

// =============================================================================
// FFI kernel argument type
// =============================================================================

/// C-compatible kernel argument for FFI dispatch.
///
/// `tag`: 0 = buffer_id (u64), 1 = f32, 2 = f64, 3 = i32, 4 = u32
#[repr(C)]
pub struct KernelArgFFI {
    pub tag: i32,
    pub value: KernelArgValue,
}

/// Union for kernel argument values.
#[repr(C)]
pub union KernelArgValue {
    pub buffer_id: u64,
    pub f32_val: f32,
    pub f64_val: f64,
    pub i32_val: i32,
    pub u32_val: u32,
}

impl KernelArgFFI {
    unsafe fn to_kernel_arg(&self) -> KernelArg {
        match self.tag {
            0 => KernelArg::Buffer(unsafe { self.value.buffer_id } as *mut std::ffi::c_void),
            1 => KernelArg::Float32(unsafe { self.value.f32_val }),
            2 => KernelArg::Float64(unsafe { self.value.f64_val }),
            3 => KernelArg::Int32(unsafe { self.value.i32_val }),
            4 => KernelArg::UInt32(unsafe { self.value.u32_val }),
            _ => KernelArg::Int32(0),
        }
    }
}
