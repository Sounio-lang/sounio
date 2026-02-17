//! GPU Detection FFI Functions
//!
//! Implements GPU availability detection:
//! - __sounio_gpu_available

use std::process::Command;

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
