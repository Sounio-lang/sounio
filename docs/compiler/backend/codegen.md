# Code Generation

The code generation stage transforms HLIR (or SIR) into executable code. Multiple backends are supported for different use cases.

## Overview

The codegen module is located in `compiler/src/codegen/`. It provides:

- **Cranelift JIT**: Fast compilation for development and scripting
- **LLVM**: Optimized ahead-of-time compilation
- **GPU Backends**: PTX (CUDA), SPIR-V (Vulkan/OpenCL), Metal (Apple)

## Backend Selection

```rust
pub enum Backend {
    Native,    // SIR direct emission (x86-64, epistemic-aware)
    LLVM,      // LLVM for optimized native code
    Cranelift, // Fast JIT compilation
    GPU,       // GPU compute kernels
}
```

## Target Architecture

```rust
pub struct Target {
    pub arch: Architecture,
    pub os: OperatingSystem,
    pub env: Environment,
}

pub enum Architecture {
    X86_64,
    AArch64,
    Wasm32,
    Wasm64,
    NVPTX64,  // NVIDIA PTX
    SPIRV64,  // SPIR-V
}

pub enum OperatingSystem {
    Linux,
    MacOS,
    Windows,
    None,  // Bare metal / GPU
}
```

## Cranelift JIT Backend

**Location**: `compiler/src/codegen/cranelift.rs`

**Feature Flag**: `--features jit`

Cranelift provides fast JIT compilation, ideal for:

- Development iteration
- REPL
- Scripting use cases

### Effect Dispatch

The JIT supports algebraic effects through:

1. **Direct runtime functions**: Specialized functions like `runtime_prob_sample`, `runtime_io_print`
2. **Registry-based dispatch**: Configurable handlers via `HandlerRegistry`

```rust
// Configure handlers
configure_jit_registry(HandlerRegistry::with_defaults());

let jit = CraneliftJit::new();
let result = jit.compile_and_run(&module)?;
```

### Runtime Functions

The JIT exposes several runtime functions:

```rust
extern "C" fn runtime_print_i64(val: i64);
extern "C" fn runtime_print_f64(val: f64);
extern "C" fn runtime_print_newline();
extern "C" fn runtime_print_str(ptr: *const u8, len: usize);
extern "C" fn runtime_print_bool(val: i8);
```

### Effect Handlers

Predefined handler IDs are organized by effect type:

```rust
// Prob effect handlers (10-19)
const HANDLER_PROB_DETERMINISTIC: u32 = 10;
const HANDLER_PROB_IMPORTANCE: u32 = 11;
const HANDLER_PROB_ENUMERATE: u32 = 12;

// Causal effect handlers (20-29)
const HANDLER_CAUSAL_SCM: u32 = 20;
const HANDLER_CAUSAL_BACKDOOR: u32 = 21;
const HANDLER_CAUSAL_FRONTDOOR: u32 = 22;

// Grad effect handlers (30-39)
const HANDLER_GRAD_FORWARD: u32 = 30;
const HANDLER_GRAD_REVERSE: u32 = 31;
const HANDLER_GRAD_NUMERIC: u32 = 32;
```

## LLVM Backend

**Location**: `compiler/src/codegen/llvm/`

**Feature Flag**: `--features llvm`

The LLVM backend provides optimized ahead-of-time compilation using the LLVM infrastructure (requires LLVM 15+).

### Structure

```
codegen/llvm/
+-- mod.rs       # Module exports
+-- codegen.rs   # Main LLVM codegen
+-- types.rs     # Type mapping
+-- passes.rs    # Optimization passes
+-- debug.rs     # Debug info generation
+-- linker.rs    # Linking support
+-- target.rs    # Target configuration
+-- gpu.rs       # GPU-specific LLVM lowering
```

### Optimization Levels

```rust
pub enum OptLevel {
    O0,  // No optimization
    O1,  // Basic optimization
    O2,  // Standard optimization (default)
    O3,  // Aggressive optimization
    Os,  // Size optimization
    Oz,  // Aggressive size optimization
}
```

### Usage

```rust
use inkwell::context::Context;

let context = Context::create();
let mut codegen = LLVMCodegen::new(
    &context,
    "main",
    OptLevel::O2,
    false,  // debug_info
);
let module = codegen.compile(&hlir);
```

## GPU Backends

**Location**: `compiler/src/codegen/gpu/`

**Feature Flag**: `--features gpu`

Sounio provides comprehensive GPU code generation:

### Supported Targets

- **PTX**: NVIDIA CUDA (compute capability 3.0+)
- **SPIR-V**: Vulkan, OpenCL
- **Metal**: Apple Silicon (MSL)

### Architecture

```
HLIR
  | hlir_to_gpu::lower()
  v
GpuIR (GPU-specific intermediate representation)
  | PtxCodegen / SpirvCodegen / MetalCodegen
  v
PTX / SPIR-V / MSL
  | Driver
  v
GPU Execution
```

### GPU IR Types

```rust
pub struct GpuModule {
    pub name: String,
    pub target: GpuTarget,
    pub functions: Vec<GpuFunction>,
    pub kernels: Vec<GpuKernel>,
    pub shared_mem: Vec<SharedMemDecl>,
    pub constants: Vec<GpuConstant>,
}

pub struct GpuKernel {
    pub name: String,
    pub params: Vec<GpuParam>,
    pub blocks: Vec<GpuBlock>,
    pub shared_mem_size: usize,
    pub max_threads: (u32, u32, u32),
}

pub enum GpuTarget {
    Cuda { compute_capability: (u32, u32) },
    Vulkan { version: (u32, u32) },
    OpenCL { version: (u32, u32) },
    Metal { gpu_family: MetalGpuFamily },
}
```

### Epistemic GPU Computing

Sounio is the first language to track epistemic state through GPU computation:

- **Shadow registers**: Track uncertainty (epsilon) values alongside data
- **Validity predicates**: Propagate validity conditions
- **Provenance tracking**: Maintain data lineage on GPU

```rust
// Epistemic PTX emission
pub struct EpistemicPtxConfig {
    pub enable_shadow_regs: bool,
    pub confidence_threshold: f64,
    pub propagation_mode: PropagationMode,
}

pub struct EpistemicPtxEmitter {
    config: EpistemicPtxConfig,
    shadow_regs: EpistemicShadowRegs,
}
```

Example SIR with epistemic tracking:

```
; Input: epistemic value x with confidence 0.9
%x.val = sir.const.f64 0.5
%x.conf = sir.const.f64 0.9
%two = sir.const.f64 2.0
%y.val = sir.mul.f64 %x.val, %two
%y.conf = sir.epistemic.propagate.mul %x.conf, 1.0
```

### PTX Codegen

```rust
pub struct PtxCodegen {
    compute_capability: (u32, u32),
    features: CudaFeatures,
}

impl PtxCodegen {
    pub fn new(compute_capability: (u32, u32)) -> Self;
    pub fn generate(&self, module: &GpuModule) -> String;
}

// Usage
let ptx = PtxCodegen::new((8, 0)).generate(&gpu_module);
```

### SPIR-V Codegen

```rust
pub struct SpirvCodegen {
    target: SpirvTarget,
}

impl SpirvCodegen {
    pub fn generate(&self, module: &GpuModule) -> Vec<u32>;
}
```

### Metal Codegen

```rust
pub struct MetalCodegen {
    config: MetalCodegenConfig,
}

pub struct MetalCodegenConfig {
    pub gpu_family: MetalGpuFamily,
    pub language_version: (u32, u32),
    pub enable_fast_math: bool,
}
```

### GPU Optimization Features

The GPU backend includes sophisticated optimization:

**Kernel Fusion**:
```rust
pub use fusion::{
    FusionAnalysis, FusionCandidate, FusionConfig,
    FusionTransformer, analyze_and_fuse_kernels,
};
```

**Auto-tuning**:
```rust
pub use autotune::{
    AutoTuner, KernelAnalyzer, OccupancyCalculator,
    TuningStrategy, tune_module,
};
```

**Async Memory Pipeline**:
```rust
pub use async_pipeline::{
    AsyncPipeline, PipelineScheduler, apply_pipeline,
};
```

**Warp Divergence Analysis**:
```rust
pub use divergence::{
    WarpDivergenceAnalyzer, ControlFlowOptimizer,
    PredicateCompiler,
};
```

## Cross-Platform Portable IR

Write-once, compile-anywhere GPU kernels:

```rust
pub struct UnifiedKernel {
    pub name: String,
    pub params: Vec<UnifiedParam>,
    pub ops: Vec<PortableGpuOp>,
}

// Compile to all targets
let results = compile_to_all(&kernel, &available_backends);
```

## Entry Points

Main compilation entry points in `compiler/src/lib.rs`:

```rust
/// Compile to native code (requires jit or llvm feature)
pub fn compile(source: &str) -> miette::Result<Vec<u8>>

/// Compile to PTX
pub fn compile_to_gpu(source: &str, sm_version: (u32, u32)) -> miette::Result<String>

/// Compile to PTX with epistemic tracking
pub fn compile_to_gpu_epistemic(
    source: &str,
    sm_version: (u32, u32)
) -> miette::Result<String>
```

## Debug Information

The debug module generates DWARF debug info:

```rust
pub mod debug {
    pub mod cfi;        // Call frame information
    pub mod source_map; // Source mapping
}
```

Source mapping for GPU:

```rust
pub use sourcemap::{
    GpuSourceMapper, LocationTrace,
    PtxDebugEmitter, SpanTracker,
};
```

## Files

```
compiler/src/codegen/
+-- mod.rs              # Backend exports and Target types
+-- cranelift.rs        # Cranelift JIT backend
+-- autodiff.rs         # Automatic differentiation
+-- simd.rs             # SIMD code generation
+-- llvm/               # LLVM backend
|   +-- mod.rs
|   +-- codegen.rs
|   +-- types.rs
|   +-- passes.rs
|   +-- debug.rs
|   +-- linker.rs
|   +-- target.rs
|   +-- gpu.rs
+-- gpu/                # GPU backends
|   +-- mod.rs          # GPU module exports
|   +-- ir.rs           # GPU IR types
|   +-- hlir_to_gpu.rs  # HLIR to GPU lowering
|   +-- ptx.rs          # PTX codegen
|   +-- spirv.rs        # SPIR-V codegen
|   +-- metal.rs        # Metal codegen
|   +-- epistemic_ptx.rs # Epistemic-aware PTX
|   +-- fusion.rs       # Kernel fusion
|   +-- autotune.rs     # Auto-tuning
|   +-- async_pipeline.rs # Async memory
|   +-- divergence.rs   # Warp divergence
|   +-- ... (40+ files)
+-- debug/              # Debug info generation
    +-- mod.rs
    +-- cfi.rs
    +-- source_map.rs
```
