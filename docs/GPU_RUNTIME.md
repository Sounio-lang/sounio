# Sounio GPU Runtime

This document describes the GPU runtime infrastructure for Sounio, including the runtime bridge, kernel launch, memory management, and backend support.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [GPU Runtime Bridge](#gpu-runtime-bridge)
- [Memory Management](#memory-management)
- [Kernel Management](#kernel-management)
- [C-Callable API](#c-callable-api)
- [Usage Examples](#usage-examples)
- [Backend Support](#backend-support)

---

## Overview

Sounio's GPU runtime provides:

- **Global singleton pattern**: Thread-safe `OnceLock<Mutex<...>>` for GPU access
- **Kernel registry**: Map kernel IDs to loaded GPU kernels
- **Buffer registry**: Map buffer IDs to device memory allocations
- **C-callable dispatch functions**: Integration with effect handler system
- **Multiple backends**: CUDA, Vulkan/SPIR-V, and simulated (testing)

The runtime bridges the effect handler system to actual GPU execution, enabling the `GPU` effect in Sounio programs.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Sounio Program                               │
│                    (fn ... with GPU { ... })                        │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Effect Handler Stack                           │
│              (__sounio_dispatch_gpu_launch, etc.)                   │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                       GpuRuntimeBridge                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ Kernel Registry │  │ Buffer Registry │  │    Statistics   │     │
│  │   ID → Kernel   │  │  ID → Buffer    │  │  launches, etc. │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                         GpuRuntime                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐         │
│  │    CUDA     │  │   Vulkan    │  │     Simulated       │         │
│  │  (cudarc)   │  │  (SPIR-V)   │  │     (testing)       │         │
│  └─────────────┘  └─────────────┘  └─────────────────────┘         │
└────────────────────────────────────────────────────────────────────┘
```

---

## GPU Runtime Bridge

The `GpuRuntimeBridge` is a global singleton that manages GPU resources.

### Initialization

```rust
use sounio::runtime::gpu_bridge::{init_gpu_bridge, get_gpu_bridge};
use sounio::codegen::gpu::runtime::GpuBackend;

// Explicit initialization (optional)
init_gpu_bridge(GpuBackend::Cuda, 0)?;

// Or use auto-initialization (simulated backend)
let bridge = get_gpu_bridge();
```

### Thread Safety

The bridge uses `OnceLock<Mutex<GpuRuntimeBridge>>` for thread-safe access:

```rust
static GPU_RUNTIME: OnceLock<Mutex<GpuRuntimeBridge>> = OnceLock::new();

pub fn get_gpu_bridge() -> &'static Mutex<GpuRuntimeBridge> {
    GPU_RUNTIME.get_or_init(|| {
        Mutex::new(GpuRuntimeBridge::new_simulated())
    })
}
```

### Statistics

```rust
let bridge = get_gpu_bridge().lock().unwrap();
let stats = bridge.stats();

println!("Kernels loaded: {}", stats.kernels_loaded);
println!("Buffers allocated: {}", stats.buffers_allocated);
println!("Bytes allocated: {}", stats.bytes_allocated);
println!("Kernel launches: {}", stats.kernel_launches);
println!("HtoD copies: {}", stats.htod_copies);
println!("DtoH copies: {}", stats.dtoh_copies);
```

---

## Memory Management

### Buffer Allocation

```rust
let mut bridge = get_gpu_bridge().lock().unwrap();

// Allocate 1024 bytes on device
let buffer_id = bridge.alloc(1024)?;

// Check buffer size
let size = bridge.buffer_size(buffer_id); // Some(1024)

// Free when done
bridge.free(buffer_id)?;
```

### Data Transfer

```rust
let mut bridge = get_gpu_bridge().lock().unwrap();

// Allocate buffer
let buf_id = bridge.alloc(1024)?;

// Host to device
let host_data: [f32; 256] = [0.0; 256];
bridge.copy_htod(buf_id, &host_data)?;

// Device to host
let mut result: [f32; 256] = [0.0; 256];
bridge.copy_dtoh(buf_id, &mut result)?;
```

### Buffer as Kernel Argument

```rust
// Get buffer pointer for kernel launch
let arg = bridge.buffer_as_arg(buffer_id);
// Returns Option<KernelArg>
```

---

## Kernel Management

### Loading PTX Kernels

```rust
let ptx = r#"
    .version 7.0
    .target sm_70
    .address_size 64

    .visible .entry vector_add(
        .param .u64 a,
        .param .u64 b,
        .param .u64 c,
        .param .u32 n
    ) {
        // kernel code
    }
"#;

let kernel_id = bridge.load_ptx(ptx, "vector_add")?;
```

### Loading SPIR-V Kernels

```rust
let spirv_bytes: Vec<u8> = load_spirv_from_file("shader.spv");
let kernel_id = bridge.load_spirv(&spirv_bytes, "main")?;
```

### Launching Kernels

```rust
// Full 3D launch
bridge.launch_kernel(
    kernel_id,
    (grid_x, grid_y, grid_z),   // Grid dimensions
    (block_x, block_y, block_z), // Block dimensions
    &[arg1, arg2, arg3],         // Kernel arguments
)?;

// Simplified 1D launch
bridge.launch_kernel_1d(
    kernel_id,
    num_elements,  // Total elements
    block_size,    // Threads per block
    &[arg1, arg2],
)?;
```

### Kernel Info

```rust
if let Some((name, param_count)) = bridge.kernel_info(kernel_id) {
    println!("Kernel '{}' has {} parameters", name, param_count);
}
```

### Unloading Kernels

```rust
bridge.unload_kernel(kernel_id)?;
```

---

## C-Callable API

The GPU runtime provides C-callable functions for integration with generated code and the effect handler system.

### Memory Functions

```c
// Allocate device memory (returns buffer ID, 0 on failure)
uint64_t __sounio_gpu_alloc(size_t size);

// Free device memory (returns 1 on success, 0 on failure)
uint64_t __sounio_gpu_free(uint64_t buffer_id);

// Copy host to device (returns 1 on success, 0 on failure)
uint64_t __sounio_gpu_copy_htod(uint64_t buffer_id, const uint8_t* src, size_t size);

// Copy device to host (returns 1 on success, 0 on failure)
uint64_t __sounio_gpu_copy_dtoh(uint64_t buffer_id, uint8_t* dst, size_t size);
```

### Kernel Functions

```c
// Load PTX kernel (returns kernel ID, 0 on failure)
uint64_t __sounio_gpu_load_ptx(
    const uint8_t* ptx, size_t ptx_len,
    const uint8_t* name, size_t name_len
);

// Launch kernel (returns 1 on success, 0 on failure)
uint64_t __sounio_gpu_launch(
    uint64_t kernel_id,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    const uint64_t* buffer_ids, size_t arg_count
);
```

### Synchronization

```c
// Synchronize GPU (returns 1 on success, 0 on failure)
uint64_t __sounio_gpu_sync();
```

---

## Usage Examples

### Vector Addition (Sounio)

```sio
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32]) {
    let i = gpu.thread_id.x
    c[i] = a[i] + b[i]
}

fn main() with GPU, IO {
    let n = 1024
    let a: [f32; 1024] = [1.0; 1024]
    let b: [f32; 1024] = [2.0; 1024]
    var c: [f32; 1024] = [0.0; 1024]

    gpu.launch(vector_add, n, 256, &a, &b, &!c)
    gpu.sync()

    println("c[0] = ", c[0])  // 3.0
}
```

### Matrix Multiplication (Rust API)

```rust
use sounio::runtime::gpu_bridge::get_gpu_bridge;
use sounio::codegen::gpu::runtime::KernelArg;

fn gpu_matmul() -> Result<(), GpuBridgeError> {
    let mut bridge = get_gpu_bridge().lock().unwrap();

    // Load kernel
    let kernel_id = bridge.load_ptx(MATMUL_PTX, "matmul")?;

    // Allocate buffers
    let n = 1024;
    let size = n * n * std::mem::size_of::<f32>();
    let a_buf = bridge.alloc(size)?;
    let b_buf = bridge.alloc(size)?;
    let c_buf = bridge.alloc(size)?;

    // Copy input data
    let a_data: Vec<f32> = vec![1.0; n * n];
    let b_data: Vec<f32> = vec![1.0; n * n];
    bridge.copy_htod(a_buf, &a_data)?;
    bridge.copy_htod(b_buf, &b_data)?;

    // Launch kernel
    let block_size = 16;
    let grid_size = (n as u32 + block_size - 1) / block_size;

    let args = vec![
        bridge.buffer_as_arg(a_buf).unwrap(),
        bridge.buffer_as_arg(b_buf).unwrap(),
        bridge.buffer_as_arg(c_buf).unwrap(),
        KernelArg::U32(n as u32),
    ];

    bridge.launch_kernel(
        kernel_id,
        (grid_size, grid_size, 1),
        (block_size, block_size, 1),
        &args,
    )?;

    // Synchronize and read result
    bridge.synchronize()?;
    let mut c_data: Vec<f32> = vec![0.0; n * n];
    bridge.copy_dtoh(c_buf, &mut c_data)?;

    // Cleanup
    bridge.free(a_buf)?;
    bridge.free(b_buf)?;
    bridge.free(c_buf)?;
    bridge.unload_kernel(kernel_id)?;

    Ok(())
}
```

---

## Backend Support

### CUDA Backend

```bash
cargo build --features "gpu,cuda"
```

Requires:
- CUDA Toolkit 11.0+
- NVIDIA GPU with compute capability 6.0+

```rust
init_gpu_bridge(GpuBackend::Cuda, 0)?;  // Device 0
```

### Vulkan/SPIR-V Backend

```bash
cargo build --features gpu
```

SPIR-V code generation works without runtime. Execution requires Vulkan support.

### Simulated Backend

Default backend for testing. No GPU required.

```rust
let bridge = GpuRuntimeBridge::new_simulated();
```

Features:
- Accepts any PTX/SPIR-V
- Allocates host memory for "device" buffers
- Launches are no-ops
- Useful for CI/testing

---

## Error Handling

```rust
#[derive(Debug)]
pub enum GpuBridgeError {
    AlreadyInitialized,        // Bridge initialized twice
    RuntimeError(String),      // Underlying GPU error
    KernelNotFound(u64),       // Invalid kernel ID
    BufferNotFound(u64),       // Invalid buffer ID
    InvalidConfig(String),     // Bad configuration
}
```

Example:

```rust
match bridge.launch_kernel(kernel_id, grid, block, &args) {
    Ok(()) => println!("Kernel launched"),
    Err(GpuBridgeError::KernelNotFound(id)) => {
        eprintln!("Kernel {} not loaded", id);
    }
    Err(e) => eprintln!("GPU error: {}", e),
}
```

---

## Performance Tips

1. **Batch transfers**: Minimize host-device copies by batching data
2. **Reuse buffers**: Allocate once, reuse for multiple kernels
3. **Async launches**: Multiple kernel launches can overlap
4. **Block size**: Use 256 or 512 threads per block for most workloads
5. **Occupancy**: Check `kernel_info()` for parameter counts

---

## Testing

```bash
# Unit tests
cargo test gpu_bridge

# With simulated backend (no GPU required)
cargo test --features gpu gpu_

# With CUDA (requires NVIDIA GPU)
cargo test --features "gpu,cuda" gpu_
```

---

## Related Documentation

- [Feature Flags](FEATURE_FLAGS.md) - GPU build configuration
- [LLVM Codegen](LLVM_CODEGEN.md) - Native code generation
- [Async Runtime](ASYNC_RUNTIME.md) - Async GPU operations
- [GPU Numerical README](../compiler/src/codegen/gpu/numerical_README.md) - GPU codegen details

---

*Last updated: January 2026 (v1.0.0)*
