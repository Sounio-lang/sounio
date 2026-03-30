<!-- docs:meta
topic_id: repo.docs.gpu.gpu-programming-guide
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.gpu-programming-guide
-->

# GPU Programming Guide

This guide describes the broader GPU implementation tree and the intended
programming model. It is not, by itself, proof that every described surface is
public in the checked GPU artifact.

For the evidence-backed public contract and the canonical support classes, use:

- `docs/features/GPU_RUNTIME.md`
- `docs/compiler/GPU_KERNELS.md`
- `docs/implementation/GPU_CAPABILITY_MODEL.md`

Sounio GPU computing is a first-class part of the language. Kernels are declared with the `kernel fn` syntax, type-checked with the same bidirectional inference as the rest of the language, and lowered through a dedicated GPU IR pipeline (HLIR → GpuKernelIr) to three backends: PTX (CUDA), Metal (MSL), and SPIR-V (Vulkan/OpenCL). Epistemic uncertainty — `Knowledge<T>` and GUM-compliant shadow registers — propagates transparently through kernel execution.

**Pipeline:**

```
Source → Lexer → Parser → AST → Check → HIR → HLIR (SSA)
       → hlir_to_gpu → GpuKernelIr
       ├─► ptx_codegen_kernel()  → .ptx text  (NVIDIA CUDA)
       ├─► gpu_lower_to_metal()  → MSL text   (Apple Metal)
       └─► spv_emit_*()          → SPIR-V blob (Vulkan/OpenCL)
```

---

## 1. Kernel Function Syntax

### Declaration

A GPU kernel is declared with `kernel fn`. The `GPU` effect is mandatory; the compiler will reject a `kernel fn` body that lacks it, and in the simplest cases it is auto-injected.

```sio
kernel fn vec_add(n: i64) with GPU {
}

kernel fn vec_scale(factor: f64, n: i64) with GPU, Div {
}
```

Multiple effects may be combined. Common combinations seen in the codebase:

| Effect combination | Meaning |
|--------------------|---------|
| `with GPU` | Pure GPU kernel, no division |
| `with GPU, Div` | Kernel that performs integer division |
| `with GPU, Div, Mut, Panic` | Kernel using mutable state and bounds-checked indexing |

### Effect rules

- `GPU` must always be present on a `kernel fn`.
- `Div` is required when the kernel body uses `/` or `%` on integer types.
- `Mut` is required when mutating values through `var` bindings or `&!` references inside the kernel.
- `Panic` is required when array indexing with `as usize` is used (the compiler inserts bounds checks).
- Host functions that launch kernels must also declare `with GPU`.

### Parameters

Kernel parameters map directly to PTX `.param` space entries. In the emitted PTX, each parameter becomes a `ld.param.*` instruction at kernel entry. Only scalar types (`i32`, `i64`, `f32`, `f64`) and pointer-sized values appear as kernel parameters in the current implementation. Array data is passed as pointers to global memory.

```sio
// Emits: .visible .entry vec_scale(.param .b64 param_factor, .param .b64 param_n)
kernel fn vec_scale(factor: f64, n: i64) with GPU, Div, Mut, Panic {
    var i: i64 = 0
    while i < n {
        i = i + 1
    }
}
```

---

## 2. GPU Builtins

The builtins in this section describe the implementation-facing GPU model.
They should not be read as proof that the checked public GPU artifact already
accepts every `gpu.*` form today.

Inside a `kernel fn` body the following built-in names are available. They map to specific `GpuKernelIr` opcodes which the PTX emitter lowers to `%tid`, `%ctaid`, and `%ntid` special registers.

| Sounio name | GpuKernelIr opcode | PTX register | Meaning |
|---|---|---|---|
| `gpu.thread_id.x` | `GpuGetTid axis=0` | `%tid.x` | Thread index within block, X axis |
| `gpu.thread_id.y` | `GpuGetTid axis=1` | `%tid.y` | Thread index within block, Y axis |
| `gpu.thread_id.z` | `GpuGetTid axis=2` | `%tid.z` | Thread index within block, Z axis |
| `gpu.block_id.x` | `GpuGetBid axis=0` | `%ctaid.x` | Block index within grid, X axis |
| `gpu.block_id.y` | `GpuGetBid axis=1` | `%ctaid.y` | Block index within grid, Y axis |
| `gpu.block_id.z` | `GpuGetBid axis=2` | `%ctaid.z` | Block index within grid, Z axis |
| `gpu.block_dim.x` | `GpuGetNtid axis=0` | `%ntid.x` | Block dimension (threads per block), X |
| `gpu.block_dim.y` | `GpuGetNtid axis=1` | `%ntid.y` | Block dimension, Y |
| `gpu.block_dim.z` | `GpuGetNtid axis=2` | `%ntid.z` | Block dimension, Z |
| `gpu.sync_threads` | `GpuBarrierSync` | `bar.sync 0` | Thread block barrier (shared memory fence) |

The standard 1D global thread index pattern — used in virtually every 1D kernel — is:

```
global_idx = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x
```

This is documented in `examples/kernel_source_level.sio`:

```sio
// gpu.thread_id.x  → GpuGetTid  axis=0
// gpu.thread_id.y  → GpuGetTid  axis=1
// gpu.block_id.x   → GpuGetBid  axis=0
// gpu.block_dim.x  → GpuGetNtid axis=0
// gpu.sync_threads → GpuBarrierSync
```

### Shared memory

Load/store to shared (threadgroup) memory uses the `GpuLoadShared` / `GpuStoreShared` opcodes at the IR level. Bounds are tracked via `shared_addr` offsets. Shared memory size for a kernel is declared in `GpuKernelIr.shared_bytes` and validated against `GpuCapabilities.max_shared_bytes` before code emission.

### Warp-level operations

The IR includes the full set of warp primitives (available on NVIDIA, no-op or emulated on others):

- `GpuShflDown` / `GpuShflUp` / `GpuShflBfly` / `GpuShflIdx` — warp shuffle
- `GpuVote` / `GpuBallot` — warp vote
- `GpuAtomicAdd` — atomic addition to global memory

### Math intrinsics

Single-instruction GPU math (maps to PTX `sqrt.approx.f32`, `rsqrt.approx`, `sin.approx`, etc.):

`GpuSqrt`, `GpuRsqrt`, `GpuExp2`, `GpuLg2`, `GpuSin`, `GpuCos`, `GpuAbs`, `GpuRcp`

---

## 3. 1D Vector Addition — Complete Example

The canonical GPU example in Sounio demonstrates all the pieces together: kernel declaration, launch syntax, and GPU effect threading through the host.

```sio
// examples/gpu.sio — verified public GPU surface

kernel fn vector_add(n: i64) with GPU {
}

kernel fn scale_vector(factor: f64, n: i64) with GPU, Div {
}

fn main() with GPU, IO {
    let n: i64 = 1024
    let grid  = (16, 1, 1)   // 16 blocks
    let block = (64, 1, 1)   // 64 threads per block = 1024 threads total

    perform GPU.launch(vector_add, grid, block)(n)
    perform GPU.launch(scale_vector, grid, block)(2.0, n)
    perform GPU.sync()
}
```

**Launch syntax:**

```sio
perform GPU.launch(<kernel_fn>, <grid_tuple>, <block_tuple>)(<kernel_args...>)
perform GPU.sync()
```

- `grid` and `block` are 3-tuples of `i64` representing (x, y, z) dimensions.
- `GPU.sync()` blocks the host until all previously launched kernels complete.
- `GPU.launch` requires the `GPU` effect on the calling function.

**CPU fallback pattern** (used in `tests/run-pass/gpu_kernel_basic.sio`):

The test suite exercises kernel logic on CPU when no GPU hardware is present. The idiom is:

```sio
fn gpu_available() -> bool {
    // Returns false in CI/test environment — triggers CPU fallback path
    false
}

fn parallel_add(a: [i64; 64], b: [i64; 64], n: i64) -> [i64; 64] with Mut, Panic, Div {
    var result: [i64; 64] = [0; 64]

    if gpu_available() {
        // GPU path (would launch kernel)
        var i: i64 = 0
        while i < n {
            result[i as usize] = a[i as usize] + b[i as usize]
            i = i + 1
        }
    } else {
        // CPU fallback — identical logic, serial execution
        var i: i64 = 0
        while i < n {
            result[i as usize] = a[i as usize] + b[i as usize]
            i = i + 1
        }
    }

    result
}
```

This pattern ensures tests are portable and runnable without GPU hardware. The test annotations follow the standard Sounio test format:

```sio
//@ run-pass
```

---

## 4. Three Backends

The abstraction layer is `self-hosted/gpu/portable.sio`. The `GpuTarget` enum selects the backend:

```sio
enum GpuTarget {
    GpuTargetPtx,       // NVIDIA CUDA (PTX)
    GpuTargetMetal,     // Apple Metal (MSL)
    GpuTargetSpirv,     // Vulkan/OpenCL (SPIR-V)
}
```

The `GpuCompileResult` struct carries output from all three simultaneously:

```sio
struct GpuCompileResult {
    ptx_buf:     PtxBuf,
    metal_buf:   MetalBuf,
    spirv_buf:   SpvBuf,
    ptx_valid:   bool,
    metal_valid: bool,
    spirv_valid: bool,
}
```

### 4.1 PTX — NVIDIA CUDA

**Source files:** `self-hosted/gpu/ptx.sio`, `ptx_advanced.sio`, `ptx_emitter.sio`

**Compiler flag:** `--gpu-target cuda-sm80` (or other SM versions)

**Generated file format:** PTX text, version 6.4+

```
$SOUC run self-hosted/compiler/main.sio -- --gpu-target cuda-sm80 input.sio
```

The emitted PTX follows NVIDIA's ISA conventions directly. From `examples/kernel_epistemic_wmma_matmul.ptx`:

```ptx
.version 6.4
.target sm_75
.address_size 64

.visible .entry epi_wmma_mm16(
    .param .b64 param_n
)
.maxntid 128, 1, 1
{
    .reg .pred  p<64>;
    .reg .b32   r32_<128>;
    .reg .b64   r64_<128>;
    .reg .f32   f32_<128>;
    .reg .f64   f64_<64>;
    ...
}
```

**Hardware capability matrix** (`GpuCapabilities` from `portable.sio`):

| Architecture | SM version | Tensor cores | Shared memory | Threads/block |
|---|---|---|---|---|
| Maxwell | sm_50 | No | 48 KB | 1024 |
| Volta | sm_70 | Yes | 96 KB | 1024 |
| Ampere | sm_80 | Yes | ~164 KB | 1024 |
| Ada Lovelace | sm_89 | Yes | 100 KB | 1024 |
| Hopper | sm_90 | Yes | 228 KB | 1024 |
| Blackwell | sm_100 | Yes | 228 KB | 1024 |

Tensor cores (`GpuWmma`, `GpuMma`) are available from sm_70 onward. Tensor memory access (`GpuTmaLoad`, `GpuTmaStore`) requires sm_90+.

### 4.2 Metal — Apple MSL

**Source files:** `self-hosted/gpu/metal.sio`, `metal_render.sio`

**Compiler flag:** `--gpu-target metal`

Metal has important constraints relative to PTX:

- **No f64 in compute kernels.** `has_f64 = false` for all Metal targets. Software emulation is possible but imposes severe performance cost.
- **No tensor cores.** Apple's equivalent (ANE/AMX) is not exposed through Metal compute.
- **SIMD width = 32** (SIMD groups, analogous to NVIDIA warps).
- **Threadgroup memory** (shared memory equivalent): 16 KB on GPU Family 1-2, 32 KB on Family 3+.

GPU family capabilities:

| Apple GPU Family | Devices | Max threads/threadgroup | Threadgroup mem |
|---|---|---|---|
| 1-2 | A7-A9 | 512 | 16 KB |
| 4+ | A11+ | 1024 | 32 KB |
| 7 (M1+) | M1 | 1024 | 32 KB |
| 8 (M2+) | M2 | 1024 | 32 KB |
| 9 (M3+) | M3 | 1024 | 32 KB |

### 4.3 SPIR-V — Vulkan / OpenCL

**Source files:** `self-hosted/gpu/spirv.sio`, `spirv_lower.sio`, `spirv_render.sio`, `spirv_text.sio`

**Compiler flag:** `--gpu-target spirv`

SPIR-V targets the widest hardware range (any Vulkan 1.0 driver). Conservative defaults are used because capabilities vary across drivers:

```sio
fn gpu_spirv_capabilities() -> GpuCapabilities {
    GpuCapabilities {
        has_f64: false,              // Vulkan f64 is optional, assume absent
        has_tensor_cores: false,     // No standard tensor core support in SPIR-V
        has_shared_memory: true,
        max_threads_per_block: 1024, // minmax: maxComputeWorkGroupInvocations >= 128
        max_shared_bytes: 32768,     // minmax: maxComputeSharedMemorySize >= 16384
        warp_size: 32,               // Subgroup size varies; 32 is common
        compute_units: 1,            // Unknown at compile time
    }
}
```

### Targeting all backends at once

Use `gpu_compile_to_all()` to produce PTX + Metal + SPIR-V in one pass:

```sio
let kernel = gpu_build_my_kernel()
let result = gpu_compile_to_all(kernel)        // all three backends

// Or target a single backend:
let result = gpu_compile_to_target(kernel, GpuTarget::GpuTargetPtx)

// Validate against hardware limits before launch:
let caps = gpu_default_capabilities(GpuTarget::GpuTargetPtx)
let ok = gpu_validate_kernel(kernel, caps)
```

---

## 5. Epistemic GPU — Knowledge<T> Through GPU Kernels

Sounio's epistemic type system extends into GPU kernels. `Knowledge<T>` values carry uncertainty through GPU execution using GUM (JCGM 100:2008) uncertainty propagation rules. This is the world-first GUM-compliant uncertainty propagation through GPU tensor core WMMA operations.

### How it works

Each `Knowledge<T>` value is lowered to **four shadow registers** in `GpuKernelIr`:

1. **value** — the primary datum (f32 or f64)
2. **epsilon** — GUM standard uncertainty (u_c)
3. **valid** — predicate register (`.pred`) for validity tracking
4. **provenance** — 64-bit Merkle hash for data lineage

This is defined in `self-hosted/gpu/hlir_to_gpu.sio`:

```
// Epistemic Knowledge<T> values get four shadow registers each.
// Counterfactual mode replicates lanes across the warp.
```

The type tag `HLIR_TY_KNOWLEDGE = 11` signals the lowering pass to emit the four-register layout rather than a single register.

### Uncertainty propagation formulas

From `self-hosted/gpu/tensor_epistemic.sio`:

| Operation | GUM propagation formula |
|---|---|
| GEMM (quadrature) | `epsilon_c = sqrt(k * (a^2 * epsilon_b^2 + b^2 * epsilon_a^2))` |
| GEMM (max-propagation) | `epsilon_c = max(epsilon_a, epsilon_b)` |
| Warp sum (32 threads) | `epsilon_sum = epsilon * sqrt(32)` (approx 5.657x) |
| Softmax output `i` | `epsilon_out_i = softmax_i * (1 - softmax_i) * epsilon_in` |
| LayerNorm | Chain rule through mean, variance, and normalization steps |

The `TensorEpistemicConfig` controls which formula is used:

```sio
struct TensorEpistemicConfig {
    sm_version: i64,           // 80, 86, 89, 90, 100
    use_tensor_cores: bool,    // wmma/mma when sm >= 70
    epistemic_enabled: bool,
    quadrature_mode: bool,     // true = GUM quadrature, false = max-propagation
    confidence_threshold: f32,
    warp_size: i64,            // always 32
}
```

### WMMA tensor core epistemic kernel

`examples/kernel_epistemic_wmma_matmul.sio` declares a 16x16 epistemic WMMA kernel:

```sio
// World-first: GUM-compliant uncertainty propagation through GPU tensor core
// WMMA operations, implemented at the compiler level.
//
// The emitted PTX contains:
//   - mma.sync.aligned.m16n8k16 (tensor core data path, ptxas-valid)
//   - sqrt.approx.f32            (GUM uncertainty shadow path)
//   - and.pred                   (validity conjunction)
//   - xor.b64                    (provenance Merkle merge)

kernel fn epi_wmma_mm16(n: i64) with GPU, Div, Mut, Panic {
    var i: i64 = 0
    while i < n {
        i = i + 1
    }
}
```

The PTX emitter maps this through the epistemic tensor state machine in `tensor_epistemic.sio`. Every arithmetic operation emits a parallel shadow computation that updates `f32_epsilon` registers according to the GUM formula.

### Binary format modes

The `--gpu-binary-format` flag selects how epistemic metadata is packaged into the output:

| Flag value | Description |
|---|---|
| `epistemic` | Full Knowledge<T> shadow registers; uncertainty in output |
| `tuned` | Profile-guided layout; shadow registers with tuned block sizes |
| `fused` | Fused epistemic and data paths for throughput |
| *(default)* | Standard mode; no epistemic shadow registers |

---

## 6. Stdlib GPU Modules

The `stdlib/gpu/` directory provides these modules:

- `stdlib/gpu/fft.sio`
- `stdlib/gpu/smooth.sio`
- `stdlib/gpu/stats.sio`
- `stdlib/gpu/tilelang.sio`
- `stdlib/gpu/lib.sio`

### fft.sio — Cooley-Tukey FFT

In-place radix-2 DIT FFT on fixed-size arrays of complex numbers (real/imag interleaved).

Reference: Cooley & Tukey, "An algorithm for the machine calculation of complex Fourier series," Math. Comp. 1965.

Provided functions include `fft_sin`, `fft_cos`, `fft_bit_reverse`, forward and inverse transforms. All math is implemented in pure Sounio (no external intrinsics) using Taylor series expansions — this makes the module portable across all three GPU backends.

```sio
// Bit-reversal permutation (from stdlib/gpu/fft.sio):
fn fft_bit_reverse(index: i32, log2n: i32) -> i32 {
    var result = 0
    var val = index
    var i = 0
    while i < log2n {
        result = result * 2 + (val - (val / 2) * 2)
        val = val / 2
        i = i + 1
    }
    return result
}
```

### smooth.sio — Signal Smoothing

1D Gaussian smoothing, moving average, and bilateral filtering on fixed-size `f64` arrays.

Reference: Paris & Durand, "A fast approximation of the bilateral filter using a signal processing approach," ECCV 2006.

Key function:

```sio
fn smooth_gaussian_1d(
    input:  &[f64; 256],
    output: &![f64; 256],
    n:      i32,
    sigma:  f64
) with Mut, Div, Panic
```

The `&![f64; 256]` parameter uses Sounio's exclusive reference syntax for the output buffer. Kernel radius is clamped to `min(3*sigma, 32)` to bound the work.

### stats.sio — Parallel Statistical Reductions

Sum, mean, variance, min, max, histogram, dot product, and vector norms. Designed to mirror GPU-style data-parallel reduction patterns (tree-based reduction with warp shuffles at the IR level).

Reference: Harris, "Optimizing Parallel Reduction in CUDA," NVIDIA 2007.

The result struct:

```sio
struct GpuStats {
    mean:     f64,
    variance: f64,
    min:      f64,
    max:      f64,
    n:        i32,
}
```

Key functions: `gpu_sum`, `gpu_mean`, and `gpu_stats` which computes all fields in one pass.

**Note on the JIT &! limitation:** As documented in `tests/stdlib/gpu/test_gpu.sio`, array-mutating operations that use `&!` references across call boundaries do not reflect mutations in the caller's stack frame under the Cranelift JIT. The workaround is to inline the mutation logic or use value-return patterns. This is a JIT-only limitation; native-compiled code is unaffected.

---

## 7. Compilation Workflow

### Type-checking only

```bash
SOUC=./bin/souc

$SOUC check examples/kernel_vec_add.sio
$SOUC check examples/gpu.sio --show-ast
$SOUC check examples/gpu.sio --show-types
```

### Running with JIT

```bash
$SOUC run examples/kernel_epistemic_wmma_matmul.sio
```

### GPU binary compilation

```bash
# PTX for NVIDIA Ampere (sm_80)
$SOUC run self-hosted/compiler/main.sio -- --gpu-target cuda-sm80 input.sio

# Epistemic binary format (Knowledge<T> shadow registers active)
$SOUC run self-hosted/compiler/main.sio -- \
    --gpu-target cuda-sm80 \
    --gpu-binary-format epistemic \
    input.sio

# All three backends simultaneously
$SOUC run self-hosted/compiler/main.sio -- \
    --gpu-target all \
    input.sio
```

### GPU-profile binary (souc-linux-x86_64-gpu)

The GPU-profile binary supports the `build --backend gpu` command directly:

```bash
# Verified public GPU surface commands:
souc-linux-x86_64-gpu check examples/gpu.sio
souc-linux-x86_64-gpu build examples/gpu.sio --backend gpu -o /tmp/sounio-gpu.ptx
souc-linux-x86_64-gpu check examples/kernel_vec_add.sio
souc-linux-x86_64-gpu build examples/kernel_vec_add.sio --backend gpu -o /tmp/kernel_vec_add.ptx
```

### Kernel resource estimation

Before launching, use the portable layer's helpers to validate a kernel against hardware limits:

```sio
let caps   = gpu_ptx_capabilities(8, 0)    // Ampere sm_80
let kernel = gpu_build_my_kernel()

// Check shared memory usage vs. hardware limit
let shm_used = gpu_kernel_shared_mem_usage(kernel)
let ok       = gpu_validate_kernel(kernel, caps)

// Estimate register pressure (higher = lower occupancy)
let reg_pressure = gpu_kernel_register_pressure(kernel)
```

---

## 8. GpuKernelIr — The Internal Representation

Understanding the IR helps when reading compiler output or writing backend extensions. The IR is defined in `self-hosted/gpu/kernel_ir.sio`.

### Types

```sio
enum GpuType {
    GpuU32, GpuU64, GpuF32, GpuF64,
    GpuI32, GpuI64,
    GpuPtr,    // .u64 pointer
    GpuBool,   // .pred register
    GpuTf32,   // TF32 (sm_80+)
    GpuF16,    // Half precision
    GpuB32, GpuB64,  // Untyped bitwise
}
```

### Opcode categories

| Category | Opcodes |
|---|---|
| Thread indexing | `GpuGetTid`, `GpuGetBid`, `GpuGetNtid` |
| Arithmetic | `GpuAdd`, `GpuSub`, `GpuMul`, `GpuDiv`, `GpuFma`, `GpuAddImm`, `GpuMulImm` |
| Memory | `GpuLoadParam`, `GpuLoadGlobal`, `GpuStoreGlobal`, `GpuLoadShared`, `GpuStoreShared`, `GpuLdGlobalCached`, `GpuPrefetch` |
| Predicates | `GpuSetpLt`, `GpuSetpLe`, `GpuSetpEq`, `GpuSetpGe`, `GpuSetpGt`, `GpuSetpNe`, `GpuSelp` |
| Control flow | `GpuBra`, `GpuExit`, `GpuBarrierSync`, `GpuRet` |
| Warp | `GpuShflDown`, `GpuShflUp`, `GpuShflBfly`, `GpuShflIdx`, `GpuVote`, `GpuBallot`, `GpuAtomicAdd` |
| Tensor core | `GpuWmma`, `GpuMma`, `GpuTmaLoad` (sm_90+), `GpuTmaStore` (sm_90+) |
| Math intrinsics | `GpuSqrt`, `GpuRsqrt`, `GpuExp2`, `GpuLg2`, `GpuSin`, `GpuCos`, `GpuAbs`, `GpuRcp` |
| Conversion | `GpuCvt` |

---

## 9. Limitations and Known Issues

### JIT &! reference visibility

Mutations through exclusive references (`&!T`) passed across call boundaries are not reflected in the caller's stack frame under the Cranelift JIT backend. This affects `smooth_gaussian_1d` and similar functions in `stdlib/gpu/smooth.sio` when called from JIT-compiled code.

**Workaround:** Return the modified value by value rather than mutating through a reference. Native-compiled code does not have this restriction.

### Metal has no f64

`GpuCapabilities.has_f64 = false` for all Metal targets. Kernels that use `f64` arithmetic will not compile to Metal. Use `f32` throughout, or check `has_f64` before targeting Metal.

### SPIR-V f64 is optional

Vulkan 1.0 does not require `Float64` capability in the driver. The SPIR-V backend defaults to `has_f64 = false`. Enable it only when you have verified driver support via `VkPhysicalDeviceFeatures.shaderFloat64`.

### Tensor cores require sm_70+

`GpuWmma` and `GpuMma` opcodes are only available on Volta and later (sm_70+). The compiler checks `GpuCapabilities.has_tensor_cores` and will reject tensor core kernels targeting older hardware.

### Closure literals in kernel bodies

The current compiler does not support closure literals (`|x| x + 1`) inside `kernel fn` bodies. Use named function references instead:

```sio
// Does not work in kernel fn:
// let f = |x: f64| x * 2.0

// Works:
fn double(x: f64) -> f64 { x * 2.0 }
let f = double
```

See `docs/compiler/KNOWN_LIMITATIONS.md` for the full list.

### Shared memory arrays

Bare `&![T; N]` array mutation in shared memory must be wrapped in a struct due to the same limitation as bare array mutation elsewhere. See `KNOWN_LIMITATIONS.md`.

### Fixed-capacity data structures

All GPU IR data structures use fixed-capacity arrays (no heap allocation at compile time). Kernel op count is bounded by the `ops` array size in `GpuKernelIr`. Kernels that exceed the op limit will be rejected at IR construction time.

### JIT memory explosion with large self-hosted compilations

Running `$SOUC run self-hosted/compiler/main.sio -- --native-compile` or `--gpu-target` with the full self-hosted compiler as input causes Cranelift JIT to compile all self-hosted compiler functions into memory, growing to 14-35 GB RSS. Type-checking (`--check`) and IR dumping (`--ir-dump`) complete in under 5 seconds and are unaffected.

---

## 10. Quick Reference

```sio
// Declare a kernel
kernel fn my_kernel(n: i64) with GPU { }
kernel fn my_kernel_div(n: i64) with GPU, Div { }
kernel fn my_kernel_full(n: i64) with GPU, Div, Mut, Panic { }

// Launch from host
fn main() with GPU, IO {
    let grid  = (32, 1, 1)
    let block = (32, 1, 1)
    perform GPU.launch(my_kernel, grid, block)(1024)
    perform GPU.sync()
}

// GPU builtins (inside kernel fn only):
//   gpu.thread_id.x / .y / .z
//   gpu.block_id.x  / .y / .z
//   gpu.block_dim.x / .y / .z
//   gpu.sync_threads
//
// Standard 1D global index:
//   idx = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x

// Compile commands
SOUC=./bin/souc

$SOUC check file.sio
$SOUC run self-hosted/compiler/main.sio -- \
    --gpu-target cuda-sm80 file.sio
$SOUC run self-hosted/compiler/main.sio -- \
    --gpu-target cuda-sm80 \
    --gpu-binary-format epistemic file.sio

souc-linux-x86_64-gpu build file.sio --backend gpu -o out.ptx
```

---

*Sources: `self-hosted/gpu/portable.sio`, `kernel_ir.sio`, `hlir_to_gpu.sio`, `tensor_epistemic.sio`; `examples/gpu.sio`, `kernel_vec_add.sio`, `kernel_source_level.sio`, `kernel_epistemic_wmma_matmul.sio`; `tests/run-pass/gpu_kernel_basic.sio`, `kernel_fn_gpu_effect.sio`; `stdlib/gpu/fft.sio`, `smooth.sio`, `stats.sio`.*
