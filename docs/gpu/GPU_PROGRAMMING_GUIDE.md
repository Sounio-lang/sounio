# Sounio GPU Programming Guide

Sounio has a complete, multi-target GPU backend (PTX/CUDA, Metal/MSL, SPIR-V/Vulkan).
All three backends are implemented in `self-hosted/gpu/` with no stubs.

## Architecture Overview

```
Sounio source
     │
     ▼
 HLIR (SSA)
     │  hlir_to_gpu.sio
     ▼
GpuKernelIr  ──┬── lower_to_ptx.sio ──► PTX text (.ptx)   [NVIDIA CUDA]
               ├── metal.sio         ──► MSL text (.metal) [Apple Metal]
               └── spirv.sio         ──► SPIR-V binary (.spv) [Vulkan]
               └── portable.sio      ──► all three at once
```

## GPU Effects

GPU kernels must be annotated with the `GPU` effect. Effects compose:

```sounio
fn vector_add(a: &[f32; 1024], b: &[f32; 1024], c: &![f32; 1024], n: i32)
    -> i32 with GPU, Mut, Div, Panic
{
    // kernel body
    0
}
```

## Compiler Flags

```bash
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit

# Compile to PTX (NVIDIA sm_80 = Ampere)
$SOUC run input.sio --gpu-target cuda-sm80 --output kernel.ptx

# Compile to Metal
$SOUC run input.sio --gpu-target metal-gen4 --output kernel.metal

# Compile to SPIR-V (Vulkan 1.3)
$SOUC run input.sio --gpu-target spirv-vk13 --output kernel.spv

# Strict parity check (validates CUDA/ROCm ABI compatibility)
$SOUC run input.sio --gpu-target cuda-sm90 --gpu-strict-parity --output kernel.ptx

# Supported formats
$SOUC run input.sio --gpu-target cuda-sm80 --gpu-binary-format fatbin
```

### `--gpu-target` values

| Flag value      | Backend  | Architecture       |
|-----------------|----------|--------------------|
| `cuda-sm50`     | PTX      | Maxwell (GTX 900)  |
| `cuda-sm70`     | PTX      | Volta (V100)       |
| `cuda-sm80`     | PTX      | Ampere (A100)      |
| `cuda-sm90`     | PTX      | Hopper (H100)      |
| `cuda-sm100`    | PTX      | Blackwell (B100)   |
| `rocm-gfx906`   | PTX      | ROCm Vega 20       |
| `metal-gen1`    | Metal    | Apple A11          |
| `metal-gen3`    | Metal    | Apple M2           |
| `metal-gen4`    | Metal    | Apple M3/A17       |
| `metal-gen5`    | Metal    | Apple M4           |
| `spirv-vk13`    | SPIR-V   | Vulkan 1.3         |

## GPU Kernel IR

The `GpuKernelIr` struct (`self-hosted/gpu/kernel_ir.sio`) is the intermediate
representation for GPU kernels, targeting 100+ opcodes across all three backends.

Key constants:

```sounio
// Thread/block index builtins (from hlir_to_gpu.sio)
let HLIR_GPU_BUILTIN_TID_X: i64 = 0   // threadIdx.x
let HLIR_GPU_BUILTIN_TID_Y: i64 = 1   // threadIdx.y
let HLIR_GPU_BUILTIN_TID_Z: i64 = 2   // threadIdx.z
let HLIR_GPU_BUILTIN_BID_X: i64 = 3   // blockIdx.x
let HLIR_GPU_BUILTIN_BDIM_X: i64 = 6  // blockDim.x
let HLIR_GPU_BUILTIN_SYNC_THREADS: i64 = 7  // __syncthreads()
```

### Standard 1D kernel prologue

`hlir_emit_kernel_prologue(kernel, num_params)` automatically emits the canonical
index computation:

```
tid         = threadIdx.x          (GpuGetTid axis=0)
bid         = blockIdx.x           (GpuGetBid axis=0)
bdim        = blockDim.x           (GpuGetNtid axis=0)
global_idx  = bid * bdim + tid
```

Virtual registers `num_params..num_params+4` are reserved for this pattern.

## Multi-Target Compilation

`portable.sio` provides write-once-compile-anywhere dispatch:

```sounio
// Compile to all three backends simultaneously
let result = gpu_compile_to_all(kernel)

// Compile to a single backend
let result = gpu_compile_to_target(kernel, GpuTarget::GpuTargetPtx)

// Validate against hardware limits
let caps = gpu_default_capabilities(GpuTarget::GpuTargetPtx)
let ok   = gpu_validate_kernel(kernel, caps)
```

### Capability matrix

```sounio
// Explicit capabilities for sm_80 (Ampere)
let caps = gpu_ptx_capabilities(8, 0)
// caps.has_f64          = true
// caps.has_tensor_cores = true
// caps.max_shared_bytes = 163840 (160KB)
// caps.warp_size        = 32
```

## Epistemic GPU (World-First Feature)

Sounio is the only language with native Knowledge<T> on GPU — uncertainty
propagates through tensor cores using GUM arithmetic.

### Shadow register model

Every `Knowledge<T>` value maps to **four GPU registers**:

| Register | Holds |
|----------|-------|
| `val`    | Central value (f32/f64) |
| `eps`    | GUM uncertainty (εᵢ) |
| `valid`  | Validity predicate (bool) |
| `prov`   | Provenance hash (u32) |

### GUM propagation rules on GPU

| Operation | Rule |
|-----------|------|
| `a + b`   | `eps_c = sqrt(eps_a² + eps_b²)` (quadrature) |
| `a * b`   | `eps_c = |a|·eps_b + |b|·eps_a` (first-order) |
| `FMA(a,b,c)` | Heron step refinement |
| Warp reduce | `vote.sync` + `shfl.sync` aggregation |

See `self-hosted/gpu/epistemic_ptx.sio` for the full implementation.

### WMMA tensor core epistemic operations

`epistemic_tensor_core.sio` implements 16×16×16 tiled epistemic matrix multiply
with per-element Merkle provenance tracking:

```
Uncertainty(C_tile) = GUM_merge(Uncertainty(A_tile), Uncertainty(B_tile))
Provenance(C_tile)  = merkle_merge(Prov(A_tile), Prov(B_tile), lane_id)
```

Baseline overhead vs cuBLAS (L4 GPU): **2.71×** (projected < 2.0× after
async pipeline + FP16 TF32 optimizations — see
`self-hosted/gpu/OPTIMIZATION_REPORT.md`).

## Standard Library GPU Modules

`stdlib/gpu/` provides CPU-side data-parallel algorithms that mirror GPU patterns:

| Module | Contents |
|--------|----------|
| `fft.sio` | Cooley-Tukey radix-2 DIT FFT (forward + inverse) |
| `smooth.sio` | Gaussian smoothing, moving average, bilateral filter |
| `stats.sio` | Sum, mean, variance, min, max, histogram on f64 arrays |

```sounio
use stdlib::gpu::fft::{fft_forward, fft_inverse}
use stdlib::gpu::smooth::{smooth_gaussian_1d}
use stdlib::gpu::stats::{stats_mean, stats_variance}
```

## Available Optimization Passes

Located in `self-hosted/gpu/opt/`:

| Pass | File | Purpose |
|------|------|---------|
| Async pipeline | `async_pipeline.sio` | `cp.async` double-buffered HBM→shared overlap |
| Auto-tuner | `autotune.sio` | Runtime configuration optimization |
| Kernel fusion | `fusion.sio` | Merge adjacent kernels to reduce launch overhead |
| Divergence | `divergence.sio` | Warp divergence analysis + mitigation |
| Quantization | `quantize.sio` | int8/int4 weight compression |
| Autodiff | `autodiff/autodiff.sio` | Automatic differentiation for GPU kernels |
| Multi-GPU | `multi/multi_gpu.sio` | Multi-device synchronization |

## Backend Source Files

| File | LOC | Purpose |
|------|-----|---------|
| `kernel_ir.sio` | 5,373 | GpuKernelIr data types + 100+ opcodes |
| `hlir_to_gpu.sio` | 2,250 | HLIR → GpuKernelIr lowering |
| `ptx_advanced.sio` | 3,486 | Full PTX codegen with register allocation |
| `epistemic_ptx.sio` | 3,092 | Knowledge<T> shadow registers on GPU |
| `metal.sio` | 1,538 | MSL (Metal Shading Language) codegen |
| `spirv.sio` | 1,219 | SPIR-V 1.0/1.5 binary emitter |
| `spirv_text.sio` | 2,249 | SPIR-V text emitter (human-readable) |
| `lower_to_ptx.sio` | 1,166 | GpuKernelIr → PTX driver |
| `portable.sio` | 1,302 | Write-once-compile-anywhere dispatcher |
| `epistemic_tensor_core.sio` | 620 | WMMA epistemic tensor operations |

## Example: Vector Add Kernel (PTX path)

```sounio
// Build a minimal parallel-add kernel programmatically
fn build_vector_add_kernel() -> GpuKernelIr with Mut, Panic {
    var kernel = gpu_kernel_ir_new()
    kernel.name = "vector_add"
    kernel.num_params = 3           // a_ptr, b_ptr, c_ptr

    // Emit standard 1D prologue: tid, bid, bdim, global_idx
    kernel = hlir_emit_kernel_prologue(kernel, 3)

    // global_idx is at vreg num_params+4 = 7
    let idx_reg = 7

    // Load a[idx], b[idx], store to c[idx]
    var load_a = gpu_op_new()
    load_a.opcode = GpuOpcode::GpuLoad
    load_a.dst_reg = 8
    load_a.src_reg = 0          // param 0 = a_ptr
    load_a.index_reg = idx_reg
    load_a.ty = GpuType::GpuF32
    kernel = gpu_kernel_append_op(kernel, load_a)

    // ... (load b, fadd, store c)
    kernel
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    let kernel = build_vector_add_kernel()
    let buf = gpu_lower_to_ptx(kernel)
    println(ptx_buf_to_string(buf))
    0
}
```

Run: `$SOUC run examples/gpu_vector_add.sio`
