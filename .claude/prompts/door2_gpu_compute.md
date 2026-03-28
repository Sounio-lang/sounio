# Door 2: Wire GPU Kernel Dispatch into the Native Compiler

## Mission

Enable `lean_single.sio` to compile and execute GPU kernels end-to-end: parse `kernel fn`, emit PTX, load via CUDA Driver API, launch with grid/block dimensions, transfer data host↔device. The self-hosted compiler already has extensive GPU infrastructure (52 files, 60K+ lines) — this prompt wires it into the native pipeline.

## Context

### What EXISTS (self-hosted checker + IR pipeline)

The self-hosted type checker (`check.sio`) already:
- Parses `kernel fn` via `TokenKind::Kernel` + `FnDef.is_kernel`
- Auto-injects GPU effect (ID 5) on kernel functions
- Enforces kernel constraints: E70 (only GPU/Panic/Div effects), E71 (scalar or array ref params), E72 (no return values)

The self-hosted GPU pipeline has:
- **GpuKernelIr** (`self-hosted/gpu/kernel_ir.sio`): 147+ opcodes including GpuGetTid, GpuWmma, atomics, barriers
- **PTX emission** (`self-hosted/gpu/ptx.sio` + `lower_to_ptx.sio`): GpuKernelIr → PTX text in PtxBuf
- **CUDA Driver API** (`self-hosted/gpu/runtime/cuda.sio`): FFI bindings for cuInit, cuCtxCreate, cuModuleLoadData, cuModuleGetFunction, cuMemAlloc, cuMemcpyHtoD/DtoH, cuLaunchKernel, cuCtxSynchronize
- **Launch orchestration** (`self-hosted/gpu/runtime/launch.sio`): PtxBuf → module → function → launch
- **Epistemic WMMA** (`self-hosted/gpu/epistemic_tensor_core.sio`): GUM uncertainty through tensor cores (13/13 gate PASS)
- **Three backends**: PTX (NVIDIA), Metal (Apple), SPIR-V (Vulkan)

### What DOESN'T exist (lean_single gap)

`lean_single.sio` has NO GPU awareness:
- No `kernel` keyword recognition
- No PTX emission path
- No CUDA runtime calls
- No device memory management
- No kernel launch codegen

### What this prompt builds

A minimal but real GPU path in `lean_single.sio` that:
1. Recognizes `kernel fn` declarations
2. Emits PTX from kernel function bodies
3. Generates host-side CUDA Driver API calls for launch
4. Handles host↔device data transfer for array parameters

## Architecture

### Compilation flow

```
Source: kernel fn vec_add(n: i64, a: &[f64], b: &[f64], c: &![f64]) with GPU { ... }

lean_single.sio:
  Pass 0a: recognize `kernel` keyword (token 73), register kernel functions
  Pass 1: register kernel fn names alongside regular fns
  Pass 2: for kernel fns, emit PTX instead of x86. Store PTX in a BSS buffer.

  For host call sites calling kernel fns:
    1. cuInit(0)
    2. cuCtxCreate(&ctx, 0, 0)
    3. cuModuleLoadData(&module, ptx_buf)
    4. cuModuleGetFunction(&func, module, "kernel_name")
    5. For each array param: cuMemAlloc + cuMemcpyHtoD
    6. cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, 0, args, 0)
    7. cuCtxSynchronize()
    8. For &! output params: cuMemcpyDtoH
    9. Cleanup: cuMemFree, cuModuleUnload, cuCtxDestroy
```

### PTX emission (minimal viable)

For Phase 1, emit a SUBSET of PTX — enough for vector operations:

```ptx
.version 7.0
.target sm_70
.address_size 64

.visible .entry vec_add(
    .param .u64 n,
    .param .u64 a,
    .param .u64 b,
    .param .u64 c
) {
    .reg .u64 %rd<8>;
    .reg .f64 %fd<4>;
    .reg .pred %p1;

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.u64 %rd0, %r2, %r1, %r0;   // global thread id

    // bounds check: if tid >= n, return
    ld.param.u64 %rd1, [n];
    setp.ge.u64 %p1, %rd0, %rd1;
    @%p1 ret;

    // c[tid] = a[tid] + b[tid]
    ld.param.u64 %rd2, [a];
    ld.param.u64 %rd3, [b];
    ld.param.u64 %rd4, [c];
    shl.b64 %rd5, %rd0, 3;            // tid * 8 (sizeof f64)
    add.u64 %rd2, %rd2, %rd5;
    add.u64 %rd3, %rd3, %rd5;
    add.u64 %rd4, %rd4, %rd5;
    ld.global.f64 %fd0, [%rd2];
    ld.global.f64 %fd1, [%rd3];
    add.f64 %fd2, %fd0, %fd1;
    st.global.f64 [%rd4], %fd2;
    ret;
}
```

### CUDA Driver API syscall interface

Since Sounio compiles to native ELF, CUDA calls are made via `dlopen("libcuda.so")` + `dlsym`:

```
// At program init: load libcuda.so
dlopen("libcuda.so.1", RTLD_NOW) → handle
dlsym(handle, "cuInit") → fn_ptr
dlsym(handle, "cuCtxCreate_v2") → fn_ptr
// ... etc for all needed functions
```

Alternatively, link directly against `-lcuda` at ELF emission time by adding a PT_INTERP + DT_NEEDED entry. The simpler approach for Phase 1 is syscall-based `dlopen`.

## Required Changes

### Phase 1: Kernel recognition in lean_single (token + skip)

Add `kernel` keyword (token 73):
```sio
if src_match(s, l, "kernel") { k = 73 }
```

In Pass 0a, when token 73 is followed by token 1 (`fn`), mark the function as a kernel:
```sio
// Track kernel functions
var IS_KERNEL: [i64; 4096] = [0; 4096]  // 1 if fn is kernel
```

In Pass 1 (function registration), set `IS_KERNEL[fn_idx] = 1` when preceded by token 73.

### Phase 2: PTX emission for kernel bodies

Add a `compile_kernel_ptx()` function that:
1. Emits PTX header into a BSS buffer (`PTX_BUF: [i8; 65536]`)
2. For each statement in the kernel body:
   - Arithmetic → PTX arithmetic instructions (add.f64, mul.f64, etc.)
   - Array access → PTX global memory loads/stores (ld.global, st.global)
   - Thread ID → mov.u32 %r, %tid.x
   - Bounds check → setp + @pred ret
3. Emits PTX footer (ret + closing brace)

Map Sounio operations to PTX:
| Sounio | PTX |
|--------|-----|
| `a + b` (f64) | `add.f64 %fd, %fd, %fd` |
| `a * b` (f64) | `mul.f64 %fd, %fd, %fd` |
| `a / b` (f64) | `div.rn.f64 %fd, %fd, %fd` |
| `arr[i]` (f64 load) | `ld.global.f64 %fd, [%rd]` |
| `arr[i] = v` (f64 store) | `st.global.f64 [%rd], %fd` |
| Thread ID | `mov.u32 %r, %tid.x` |
| Block ID | `mov.u32 %r, %ctaid.x` |
| Block dim | `mov.u32 %r, %ntid.x` |
| Barrier | `bar.sync 0` |

### Phase 3: Host-side launch codegen

When a regular function CALLS a kernel function, emit CUDA Driver API calls:

1. **Lazy CUDA init**: First kernel call emits `dlopen("libcuda.so.1")` + `dlsym` for all needed functions, cached in globals
2. **Context creation**: `cuInit(0)`, `cuDeviceGet(&dev, 0)`, `cuCtxCreate(&ctx, 0, dev)`
3. **Module load**: `cuModuleLoadData(&mod, ptx_buf_addr)` where ptx_buf_addr points to the BSS PTX buffer
4. **Function lookup**: `cuModuleGetFunction(&func, mod, "kernel_name")`
5. **Memory alloc + transfer**: For each array param, emit `cuMemAlloc` + `cuMemcpyHtoD`
6. **Launch**: `cuLaunchKernel(func, grid_x, 1, 1, 256, 1, 1, 0, 0, args_ptr, 0)`
7. **Sync**: `cuCtxSynchronize()`
8. **Readback**: For `&!` output params, `cuMemcpyDtoH`
9. **Cleanup**: `cuMemFree` for each buffer

Grid calculation: `grid_x = (n + 255) / 256` (256 threads per block default).

### Phase 4: Test and verify

Create test files:
```sio
// tests/run-pass/gpu_vec_add.sio
kernel fn vec_add(n: i64, a: &[f64], b: &[f64], c: &![f64]) with GPU {
    let tid = gpu_thread_id_x()
    if tid < n {
        c[tid] = a[tid] + b[tid]
    }
}

fn main() -> i32 with IO, Mut, Panic, GPU {
    var a: [f64; 1024] = [0.0; 1024]
    var b: [f64; 1024] = [0.0; 1024]
    var c: [f64; 1024] = [0.0; 1024]

    var i = 0
    while i < 1024 {
        a[i as usize] = (i as f64)
        b[i as usize] = (i as f64) * 2.0
        i = i + 1
    }

    vec_add(1024, &a, &b, &!c)

    // Verify: c[0] = 0, c[1] = 3, c[1023] = 3069
    if c[0] < 0.01 && c[1] > 2.99 && c[1] < 3.01 && c[1023] > 3068.0 {
        println("PASS: GPU vec_add")
    } else { println("FAIL") }
    0
}
```

## Hard Constraints

- **Self-host must preserve**: gen2==gen3 after every change (kernel code is unreachable during bootstrap)
- **No regressions**: All existing run-pass tests must pass
- **CUDA is optional**: If libcuda.so is not found at runtime, print "GPU unavailable" and skip (don't crash)
- **Sounio syntax**: `kernel fn`, `with GPU`, `&!` for output params
- **Phase 1 target**: Vector operations only (add, mul, div on f64 arrays)
- **Do NOT rewrite** self-hosted/gpu/ — build the lean_single path independently, referencing self-hosted patterns

## Verification

1. Self-host chain: gen2==gen3 (kernel code unreachable during bootstrap)
2. Non-GPU tests: all existing run-pass tests still pass
3. GPU test (requires NVIDIA GPU + CUDA):
   ```bash
   ./bin/souc run tests/run-pass/gpu_vec_add.sio
   # Expected: "PASS: GPU vec_add"
   ```
4. Graceful degradation (no GPU):
   ```bash
   # On machine without CUDA:
   ./bin/souc run tests/run-pass/gpu_vec_add.sio
   # Expected: "GPU unavailable: libcuda.so not found" (exit 0, not crash)
   ```

## Files to Modify

| File | Change |
|------|--------|
| `self-hosted/compiler/lean_single.sio` | Add kernel token, PTX emission, CUDA launch codegen |
| `artifacts/self-hosted/souc-self-hosted-x86_64` | Rebuilt binary |

## Reference Files (read, don't modify)

| File | Why |
|------|-----|
| `self-hosted/gpu/kernel_ir.sio` | GPU opcode definitions, GpuType enum |
| `self-hosted/gpu/lower_to_ptx.sio` | PTX emission patterns |
| `self-hosted/gpu/runtime/cuda.sio` | CUDA Driver API FFI signatures |
| `self-hosted/gpu/runtime/launch.sio` | Launch orchestration pattern |
| `stdlib/gpu/clifford_kernel.sio` | Working kernel fn pattern (212 lines) |
| `self-hosted/gpu/ptx.sio` | PtxBuf struct, PTX string accumulation |

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| GPU kernels | Type-checked only | **Compile + execute** |
| PTX emission | Self-hosted IR only | **Native lean_single** |
| CUDA launch | Not in native path | **dlopen + Driver API** |
| Vector ops | CPU only | **GPU-accelerated** |
| Graceful degrade | N/A | **"GPU unavailable" on non-CUDA** |

## What This Unblocks (Door 3)

- Octonion GEMM on GPU (8×8 = 64 FLOPs per element, massively parallel)
- Epistemic tensor core matmul (WMMA with GUM uncertainty — already proven in self-hosted)
- State space model parallel scans
- Real training loops on GPU (forward + backward on device)
- Hypercomplex attention mechanisms
