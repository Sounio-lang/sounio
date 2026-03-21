---
title: "GPU Code Generation"
description: "Sounio's GPU compilation pipeline: PTX (NVIDIA), Metal (Apple Silicon), and SPIR-V backends with epistemic shadow registers."
---

## GPU Code Generation

Sounio compiles `kernel fn` functions directly to GPU machine code, bypassing CUDA/OpenCL toolchains entirely. The GPU codegen supports three targets: **PTX** (NVIDIA), **Metal** (Apple Silicon), and **SPIR-V** (Vulkan/OpenCL).

### GPU Compilation Flow

```
Source (.sio)
  |
  v
kernel fn annotation detected
  |
  v
Effect analysis (must have GPU effect)
  |
  v
HLIR kernel extraction
  |
  v
GPU IR (unified representation)
  |
  +---> PTX Codegen (NVIDIA Volta-Blackwell)
  |       |
  |       v
  |     .ptx file --> NVIDIA Driver --> GPU Execution
  |
  +---> Metal Codegen (Apple M1/M2/M3)
  |       |
  |       v
  |     .metal source --> metallib --> GPU Execution
  |
  +---> SPIR-V Codegen (Vulkan 1.0-1.2)
          |
          v
        .spv binary --> Vulkan/OpenCL Runtime --> GPU Execution
```

### HLIR to GPU Lowering

**File**: `compiler/src/codegen/gpu/hlir_to_gpu.rs`

The critical bridge between the compiler's SSA IR and GPU-specific IR:

```
HLIR (SSA) --> GPU IR --> PTX / SPIR-V / Metal
                |
                +--> Epistemic Extension (shadow registers)
```

**Epistemic state lowering** transforms `Knowledge<T>` into GPU-friendly form:

```
Knowledge[T, epsilon, delta, Phi] --> {
    value: T,            // Actual data
    epsilon: f32,        // Uncertainty bound (shadow register)
    validity: pred,      // Predicate register
    provenance: u64,     // Bit-packed lineage
}
```

**Configuration** (`LoweringConfig`):
- `target`: CUDA, Metal, or Vulkan
- `epistemic_enabled`: Shadow register tracking on/off
- `max_threads_per_block`: Default 256
- `shared_memory_hint`: Default 48KB
- Optimization phases: auto-tune, fusion, pipelining

### PTX Backend (NVIDIA)

**File**: `compiler/src/codegen/gpu/ptx.rs` (7,861 lines)

The PTX backend directly generates NVIDIA's Parallel Thread Execution ISA:

- **Architecture support**: Volta (sm_70) through Blackwell (sm_100)
- **Register allocation**: Type-specific counters for precise register pressure control
- **Entry point**: `PtxCodegen::new()` at line 92

Compute capability mapping (lines 107-120):

| GPU Architecture | Compute Capability |
|------------------|--------------------|
| Volta (V100) | sm_70 |
| Turing (RTX 2000) | sm_75 |
| Ampere (A100, RTX 3000) | sm_80, sm_86 |
| Hopper (H100) | sm_90 |
| Blackwell (B100) | sm_100 |

### Metal Backend (Apple)

**File**: `compiler/src/codegen/gpu/metal.rs` (174KB)

Generates **Metal Shading Language (MSL)** source code:

- **Architecture support**: Apple8+ (M1, M2, M3) and Intel Macs
- **Thread model**: Maps CUDA-style threadgroups/simdgroups to Metal equivalents
- **Entry point**: `MetalCodegen` struct with MSL source generation

### SPIR-V Backend (Vulkan/OpenCL)

**File**: `compiler/src/codegen/gpu/spirv.rs` (37KB)

Generates SPIR-V binary using the `rspirv` builder:

- **Target APIs**: Vulkan 1.0-1.2, OpenCL 1.2-2.0
- **Entry point**: `SpirvCodegen` at line 48

### Epistemic GPU Computing

**File**: `compiler/src/codegen/gpu/ptx_epistemic_bridge.rs`

The epistemic bridge extends PTX codegen with uncertainty propagation:

- **Shadow registers**: Each `Knowledge<T>` value gets a shadow `f32` register for `epsilon` (uncertainty bound)
- **ValueId -> EpistemicShadowRegs** mapping tracks which registers carry epistemic metadata
- **Propagation operations**:
  - `emit_epistemic_add()` (line 84): Uncertainty adds in quadrature
  - `emit_epistemic_mul()` (line 112): Relative uncertainties combine
  - `emit_epistemic_div()` (line 126): Division uncertainty propagation

### GPU Runtime

**File**: `compiler/src/codegen/gpu/runtime.rs` (40KB)

The GPU runtime handles:
- Device discovery and selection
- Host-device memory transfers
- Kernel launch configuration
- Synchronization barriers
- Error handling and recovery

### Advanced GPU Optimizations

| Optimization | File | Description |
|-------------|------|-------------|
| **Kernel Fusion** | `fusion.rs` (80KB) | Merges adjacent kernels to reduce launch overhead |
| **Autotune** | `autotune.rs` (44KB) | Automatic launch configuration tuning |
| **Async Pipeline** | `async_memory.rs` (53KB) | Overlapped data transfers and computation |
| **Warp Divergence** | `warp_divergence.rs` (34KB) | Control flow optimization for SIMT |
| **Quantization** | `ptq.rs`, `qat.rs`, `quantize.rs` | Int4/Int8 quantization for inference |
| **Roofline Model** | `roofline.rs` (22KB) | Performance bound analysis |
| **Profiler** | `profiler.rs` (33KB) | Kernel execution profiling |
| **Numerical Stability** | `numerical.rs` (46KB) | Floating-point accuracy analysis |

### Benchmark Results

Octonion operations (120 FLOPs per multiply):

| Operation | CPU (Ryzen 9 7950X) | GPU PTX (RTX 4090) | GPU Metal (M2 Ultra) |
|-----------|--------------------|--------------------|---------------------|
| Multiply | 8.5 GFLOPS | 142.7 GFLOPS (16.8x) | 156.3 GFLOPS (18.4x) |
| Add | 12.3 GFLOPS | 189.4 GFLOPS (15.4x) | 201.7 GFLOPS (16.4x) |
| Norm | 9.7 GFLOPS | 156.8 GFLOPS (16.2x) | 168.2 GFLOPS (17.3x) |
| Conjugate | 11.2 GFLOPS | 178.3 GFLOPS (15.9x) | 192.5 GFLOPS (17.2x) |

Moufang identity validation throughput:
- CPU: 76,337 tests/sec
- GPU PTX (RTX 4090): 1,541,850 tests/sec (20.2x)
- GPU Metal (M2 Ultra): 1,391,649 tests/sec (18.2x)
