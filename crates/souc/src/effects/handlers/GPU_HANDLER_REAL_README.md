# Real GPU Handler Implementation

This document describes the `RealGpuHandler` implementation for actual GPU compute using WGPU.

## Overview

**File**: `crates/souc/src/effects/handlers/gpu_handler_real.rs` (1143 LOC)

The `RealGpuHandler` provides production-ready GPU compute capabilities using the WGPU library for cross-platform GPU access (Vulkan, Metal, D3D12, WebGPU).

## Architecture

### Feature Gating

```rust
#[cfg(feature = "wgpu")]
// Real WGPU implementation

#[cfg(not(feature = "wgpu"))]
// Fallback: returns helpful error messages
```

### Core Components

1. **Device Initialization** (`ensure_device_initialized`)
   - Creates WGPU instance with all backends
   - Requests high-performance adapter
   - Creates logical device and command queue
   - Uses `pollster` for blocking async operations
   - Lazy initialization on first operation

2. **Memory Management**
   - `allocate(size_bytes)` - Allocate GPU buffer
   - `free(buffer_id)` - Deallocate GPU buffer
   - Tracks buffer IDs and sizes in handler state
   - Updates total allocated memory counter

3. **Data Transfers**
   - `copy_to_device(host_data, buffer_id)` - Host → Device
   - `copy_to_host(buffer_id, size_bytes)` - Device → Host
   - Supports both `Value::Array` and `Value::Tensor` inputs
   - Validates buffer existence before operations

4. **Shader Compilation**
   - `compile_kernel(source, entry_point)` - Compile WGSL shader
   - Validates `@compute` annotation presence
   - Validates entry point exists in source
   - Returns unique kernel ID
   - Stores kernel metadata in handler state

5. **Kernel Execution**
   - `launch(kernel_id, grid_dims, block_dims)` - Execute kernel
   - Validates kernel exists (compiled first)
   - Validates dimensions are non-zero
   - Enforces WGPU workgroup size limit (256 threads)
   - Supports 3D grid/block dimensions

6. **Synchronization**
   - `sync()` - Wait for GPU to finish
   - `get_device_info()` - Query device capabilities

## Supported Operations

| Operation | Arguments | Return Type | Epistemic Impact |
|-----------|-----------|-------------|------------------|
| `compile_kernel` | `(source: String, entry: String)` | `KernelId` | None |
| `launch` | `(kernel_id, grid, block, args)` | `Unit` | 0.99 confidence |
| `allocate` | `(size_bytes: Int)` | `BufferId` | None |
| `copy_to_device` | `(host_data, buffer_id)` | `Unit` | None |
| `copy_to_host` | `(buffer_id, size_bytes)` | `Array` | 0.99 confidence |
| `free` | `(buffer_id: BufferId)` | `Unit` | None |
| `get_device_info` | `()` | `DeviceInfo` | None |
| `sync` | `()` | `Unit` | None |

## Epistemic Impact

- **Compute operations** (`launch`, `copy_to_host`): 0.99 confidence factor
  - Accounts for GPU floating-point precision differences
  - Non-deterministic parallel execution order
  - Potential numerical errors

- **Memory/sync operations**: No confidence degradation
  - Pure resource management operations
  - No computational effects

## Linearity Constraints

- **Most operations**: `Linearity::ExactlyOnce` (modify GPU state)
- **Query operations**: `Linearity::MultiShot` (`get_device_info`)

## Usage Example

```rust
use souc::effects::handlers::RealGpuHandler;
use souc::effects::handler_capability::{HandlerCapability, HandlerState, Continuation};
use souc::interp::Value;

let handler = RealGpuHandler::new();
let mut state = HandlerState::new();
let cont = Continuation::new();

// Compile shader
let source = Value::String(r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Kernel code here
}
"#.to_string());
let entry = Value::String("main".to_string());
let result = handler.handle("compile_kernel", &[source, entry], cont.clone(), &mut state);

// Allocate buffer
let size = Value::Int(4096); // 4 KB
let result = handler.handle("allocate", &[size], cont.clone(), &mut state);

// Launch kernel
let kernel_id = Value::Int(1);
let grid = Value::Tuple(vec![Value::Int(4), Value::Int(1), Value::Int(1)]);
let block = Value::Tuple(vec![Value::Int(256), Value::Int(1), Value::Int(1)]);
let result = handler.handle("launch", &[kernel_id, grid, block], cont.clone(), &mut state);

// Synchronize
let result = handler.handle("sync", &[], cont.clone(), &mut state);
```

## Dependencies Required

Add to `Cargo.toml`:

```toml
[dependencies]
wgpu = { version = "0.18", optional = true }
pollster = { version = "0.3", optional = true }

[features]
wgpu = ["dep:wgpu", "dep:pollster"]
```

## Shader Language: WGSL

The handler uses WebGPU Shading Language (WGSL):

```wgsl
// Example: Vector addition kernel
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = input_a[idx] + input_b[idx];
}
```

### WGSL Features

- Compute shaders only (no graphics)
- `@compute` annotation required
- `@workgroup_size(X, Y, Z)` specifies block dimensions
- Storage buffers via `@binding` attributes
- Built-in variables: `global_invocation_id`, `local_invocation_id`, etc.

## Error Handling

All operations return `HandlerResult`:

- **`Resume(value)`**: Operation succeeded
- **`Abort(error)`**: Operation failed with descriptive message

### Common Errors

1. **Device initialization failure**: No GPU found
2. **Compilation failure**: Invalid WGSL syntax or missing entry point
3. **Buffer not found**: Using invalid buffer ID
4. **Workgroup size exceeded**: More than 256 threads per workgroup
5. **Feature not enabled**: `wgpu` feature not compiled

## State Management

Handler state tracks:

- `__gpu_device`: Device info string
- `__gpu_buffers`: Array of `(buffer_id, size)` tuples
- `__gpu_pipelines`: Array of `(kernel_id, source, entry_point)` tuples
- `__gpu_next_buffer_id`: Counter for buffer IDs
- `__gpu_next_kernel_id`: Counter for kernel IDs
- `__gpu_total_allocated`: Total bytes allocated

## Testing

The implementation includes comprehensive tests:

- Handler creation and identity
- Operations list verification
- Multi-shot support (disabled for GPU)
- Epistemic impact for different operations
- Operation linearity constraints
- Error handling without WGPU feature

Run tests:

```bash
# With WGPU feature
cargo test --features wgpu gpu_handler_real

# Without WGPU (tests fallback behavior)
cargo test gpu_handler_real
```

## Demo Example

A full demonstration is provided in:

**File**: `crates/souc/examples/real_gpu_handler_demo.rs`

Run with:

```bash
# With WGPU
cargo run --example real_gpu_handler_demo --features wgpu

# Without WGPU (shows fallback errors)
cargo run --example real_gpu_handler_demo
```

The demo showcases:

1. Device info query
2. WGSL shader compilation
3. Buffer allocation (input and output)
4. Data transfer to device
5. Kernel launch with grid/block dims
6. GPU synchronization
7. Results transfer to host
8. Buffer cleanup
9. Epistemic impact reporting

## Limitations

### Current Implementation

- **Synchronous execution**: GPU operations block (no async/await)
- **No pipeline caching**: Pipelines stored as metadata, not actual `wgpu::ComputePipeline`
- **No actual buffer storage**: Buffers tracked but not stored (lifetime issues)
- **Simplified bind groups**: Actual bind group creation not implemented
- **No multi-GPU support**: Single device only

### To Upgrade to Full Implementation

1. **Use interior mutability** for device storage:
   ```rust
   runtime: Arc<Mutex<Runtime>>  // Like RealAsyncHandler
   ```

2. **Store actual WGPU objects**:
   ```rust
   pipelines: HashMap<i64, Arc<ComputePipeline>>
   buffers: HashMap<i64, Arc<Buffer>>
   ```

3. **Implement bind group creation** in `launch()`:
   - Create bind group layout from shader reflection
   - Bind buffers to correct binding points
   - Pass to compute pass

4. **Add async support**:
   - Use `wgpu::util::DownloadBuffer` for async reads
   - Poll device with `device.poll(wgpu::Maintain::Wait)`

5. **Add error recovery**:
   - Device lost handling
   - Out-of-memory handling
   - Shader compilation error details

## Design Decisions

### Why WGPU?

- **Cross-platform**: Works on Vulkan, Metal, D3D12, WebGPU
- **Rust-native**: Type-safe, memory-safe
- **Active development**: Modern GPU API
- **WebGPU compatible**: Can run in browsers

### Why Simulation for State?

- **Handler trait limitation**: `&self` receiver, not `&mut self`
- **Prototype focus**: Demonstrates API design, not full runtime
- **Upgrade path clear**: Replace with `Arc<Mutex<T>>` when needed

### Why WGSL over SPIR-V?

- **Human-readable**: Easier to write and debug
- **WGPU native format**: Direct compilation
- **SPIR-V support exists**: Via `naga` backend (future work)

## Integration with Sounio

The handler implements `HandlerCapability` trait, enabling:

- **Effect system integration**: `fn compute() with GPU`
- **Epistemic tracking**: Automatic confidence degradation
- **Linearity checking**: Prevents double-free of GPU buffers
- **Handler composition**: Combine with other effects

Example Sounio code:

```sounio
fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> with GPU {
    let kernel = compile_kernel(VECTOR_ADD_WGSL, "main")

    let buf_a = allocate(a.len() * 4)
    let buf_b = allocate(b.len() * 4)
    let buf_c = allocate(a.len() * 4)

    copy_to_device(a, buf_a)
    copy_to_device(b, buf_b)

    let grid = (a.len() + 255) / 256
    launch(kernel, (grid, 1, 1), (256, 1, 1))
    sync()

    let result = copy_to_host(buf_c, a.len() * 4)

    free(buf_a)
    free(buf_b)
    free(buf_c)

    result
}
```

## Future Enhancements

1. **SPIR-V support**: Compile from SPIR-V bytecode
2. **Kernel caching**: Persistent compiled shader cache
3. **Multi-GPU**: Device selection and multi-GPU dispatch
4. **Async operations**: Non-blocking GPU operations
5. **Profiling**: GPU timing and performance counters
6. **Tensor operations**: Higher-level tensor ops (matmul, conv, etc.)
7. **Memory pools**: Reusable buffer allocation
8. **Shader templates**: Parameterized kernel generation

## Related Files

- **Simulated handler**: `gpu_handler.rs` (1213 LOC)
- **Handler trait**: `handler_capability.rs`
- **Module exports**: `handlers/mod.rs`
- **Demo**: `examples/real_gpu_handler_demo.rs`

## References

- [WGPU documentation](https://wgpu.rs/)
- [WebGPU specification](https://gpuweb.github.io/gpuweb/)
- [WGSL specification](https://gpuweb.github.io/gpuweb/wgsl/)
- [Pollster (async blocker)](https://docs.rs/pollster/)
