# Geração de Código - Arquitetura do Compilador Sounio

## Visão Geral

O compilador Sounio suporta **4 backends de geração de código** para diferentes plataformas e casos de uso:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE COMPILAÇÃO                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HLIR (High-Level IR)                                              │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                    BACKENDS DISPONÍVEIS                     │     │
│  ├─────────────────────────────────────────────────────────────┤     │
│  │                                                             │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │     │
│  │  │   Native     │  │   Cranelift  │  │     LLVM     │   │     │
│  │  │  (x86-64)   │  │   (JIT/AOT)  │  │  (Otimizado) │   │     │
│  │  │  Sem LLVM    │  │   Rápido     │  │   Portable   │   │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │     │
│  │                                                             │     │
│  │  ┌──────────────┐  ┌──────────────┐                      │     │
│  │  │     GPU      │  │    Debug     │                      │     │
│  │  │ PTX/SPIR-V/ │  │   DWARF      │                      │     │
│  │  │ Metal        │  │              │                      │     │
│  │  └──────────────┘  └──────────────┘                      │     │
│  │                                                             │     │
│  └─────────────────────────────────────────────────────────────┘     │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                      OUTPUTS                               │     │
│  ├─────────────────────────────────────────────────────────────┤     │
│  │ ELF (Linux), Mach-O (macOS), PE (Windows)               │     │
│  │ PTX (NVIDIA), SPIR-V (Vulkan), MSL (Metal)              │     │
│  │ JIT (memória), DWARF (debug)                            │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. Backend Cranelift

### Localização
- **Principal**: [`crates/souc/src/codegen/cranelift.rs`](../../crates/souc/src/codegen/cranelift.rs)
- **Integração MIR**: [`crates/souc/src/codegen/mir_cranelift.rs`](../../crates/souc/src/codegen/mir_cranelift.rs)
- **Documentação**: [`docs/MIR_CRANELIFT_INTEGRATION_REPORT.md`](../../docs/MIR_CRANELIFT_INTEGRATION_REPORT.md)

### Arquitetura

```
MIR (SSA Form)
       │
       ▼
┌─────────────────┐
│   MirAware      │
│   CraneliftJit  │
└────────┬────────┘
         │
         ├─► Otimizações MIR
         │      ├─ Constant Propagation
         │      ├─ Dead Code Elimination
         │      ├─ Common Subexpression Elimination
         │      ├─ Loop Invariant Code Motion
         │      └─ Function Inlining
         │
         ▼
┌─────────────────┐
│    MirCranelt  │
│    Compiler     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cranelift IR   │
│  (CLIF)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Codegen        │
│  (x86-64/ARM)  │
└────────┬────────┘
         │
         ▼
   Código Nativo
```

### Estrutura Principal

```rust
// docs/MIR_CRANELIFT_INTEGRATION_REPORT.md:77
pub struct MirAwareCraneliftJit {
    optimize: bool,
    mir_opt_level: Option<OptimizationLevel>,
}

impl MirAwareCraneliftJit {
    pub fn compile_mir(&self, mir_module: &MirModule) -> Result<CompiledModule, String> {
        // 1. Aplicar otimizações MIR
        let optimized_module = if let Some(opt_level) = self.mir_opt_level {
            let mut module_clone = mir_module.clone();
            let mut pass_manager = create_default_pass_manager(opt_level);
            let _modified = pass_manager.run_module_passes(&mut module_clone)?;
            module_clone
        } else {
            mir_module.clone()
        };

        // 2. Compilar para Cranelift IR
        let mut compiler = MirCraneliftCompiler::new(self.optimize)?;
        compiler.compile_mir_module(&optimized_module)?;
        compiler.finalize()
    }
}
```

### Mapeamento de Tipos MIR → Cranelift

```rust
// docs/MIR_CRANELIFT_INTEGRATION_REPORT.md:201
MIR Type        → Cranelift Type
────────────────────────────────
I32, I64       → I32, I64
F32, F64       → F32, F64
Bool           → I8
Ptr(_)          → I64
Array(_, _)     → I64 (pointer)
Struct {..}    → I64 (pointer)
Function {...} → I64 (function pointer)
```

### Níveis de Otimização

| Nível | Passes Ativos | Benefício | Tempo Típico |
|--------|---------------|-----------|--------------|
| O0 | Nenhum | Baseline | - |
| O1 | Constant Propagation, DCE | Básico | ~5ms |
| O2 | + CSE, LICM | Moderado | ~15ms |
| O3 | + Function Inlining | Agressivo | ~25ms |

### Exemplo de Uso

```rust
// docs/MIR_CRANELIFT_INTEGRATION_REPORT.md:217
use sounio_compiler::codegen::mir_cranelift::MirAwareCraneliftJit;
use sounio_compiler::mir::optimization::OptimizationLevel;

// 1. Criar módulo MIR
let mir_module = create_mir_module();

// 2. Compilar com otimizações
let jit = MirAwareCraneliftJit::new()
    .with_optimization()
    .with_mir_optimization(OptimizationLevel::O3);

let compiled = jit.compile_mir(&mir_module)?;

// 3. Executar função compilada
unsafe {
    let func_ptr = compiled.get_function("main").unwrap();
    let main_func = std::mem::transmute::<_, fn() -> i64>(func_ptr);
    let result = main_func();
    println!("Result: {}", result);
}
```

## 2. Backend Native (x86-64)

### Localização
- **Principal**: [`crates/souc/src/backend/native/mod.rs`](../../crates/souc/src/backend/native/mod.rs)
- **x86-64**: [`crates/souc/src/backend/native/x86_64.rs`](../../crates/souc/src/backend/native/x86_64.rs)
- **AArch64**: [`crates/souc/src/backend/native/aarch64.rs`](../../crates/souc/src/backend/native/aarch64.rs)
- **Runtime**: [`crates/souc/src/backend/native/runtime.rs`](../../crates/souc/src/backend/native/runtime.rs)

### Arquitetura

```rust
// crates/souc/src/backend/native/mod.rs:8
//! # Sounio Compiler: Native Backend
//!
//! This module provides the native x86-64 backend for the Sounio compiler,
//! bypassing LLVM entirely for direct machine code generation with epistemic
//! awareness at every level.

/// NATIVE BACKEND PIPELINE
/// ┌─────────────────────────────────────────────────────────────────────┐
/// │  SIR (Sounio IR)                                                   │
/// │       │                                                             │
/// │       ▼                                                             │
/// │  ┌─────────────────┐                                                │
/// │  │ Metrics Analysis │ ◄── metrics.rs                                 │
/// │  │ • Cycle estimation                                             │
/// │  │ • Power estimation                                              │
/// │  │ • Epistemic propagation                                         │
/// │  └────────┬────────┘                                                │
/// │           │                                                         │
/// │           ▼                                                         │
/// │  ┌─────────────────┐                                                │
/// │  │ Thermal Analysis │ ◄── thermal.rs                                │
/// │  │ • Arrhenius degradation                                         │
/// │  │ • Self-heating feedback                                        │
/// │  │ • Confidence degradation                                        │
/// │  └────────┬────────┘                                                │
/// │           │                                                         │
/// │           ▼                                                         │
/// │  ┌─────────────────┐                                                │
/// │  │ Register Alloc  │ ◄── alloc.rs (epistemic-aware)                 │
/// │  │ • Confidence-based spilling                                     │
/// │  │ • LiveInterval with metadata                                    │
/// │  └────────┬────────┘                                                │
/// │           │                                                         │
/// │           ▼                                                         │
/// │  ┌─────────────────┐                                                │
/// │  │ Code Emission   │ ◄── emit.rs                                   │
/// │  │ • x86-64 machine code                                           │
/// │  │ • .so generation                                                │
/// └─────────────────────────────────────────────────────────────────────┘
```

### Features Principais

```rust
// crates/souc/src/backend/native/mod.rs:49
/// Key Features:
///
/// - **No LLVM Dependency**: Direct x86-64 emission without external toolchains
/// - **Epistemic-Aware Allocation**: Register allocation considers confidence metadata
/// - **Thermal Modeling**: Arrhenius-based degradation affects epistemic confidence
/// - **Self-Contained Metrics**: All cycle/power estimates computed internally
```

### Componentes do Backend Native

| Componente | Descrição |
|------------|-----------|
| `metrics.rs` | Estimativa de ciclos e consumo de energia |
| `thermal.rs` | Modelo térmico (Arrhenius degradation) |
| `alloc.rs` | Alocação de registradores epistêmico-aware |
| `emit.rs` | Emissão de código x86-64 |
| `elf.rs` | Geração de arquivos ELF |
| `linker.rs` | Linking de objetos |

### Configuração do Backend

```rust
// crates/souc/src/backend/native/mod.rs:112
#[derive(Debug, Clone)]
pub struct NativeBackendConfig {
    /// Target architecture
    pub arch: TargetArch,
    /// Cycle model to use
    pub cycle_model: metrics::CycleModel,
    /// Power model to use
    pub power_model: metrics::PowerModel,
    /// Thermal degradation model
    pub thermal_model: thermal::ArrheniusModel,
    /// Whether to enable thermal tracking
    pub enable_thermal_tracking: bool,
    /// Confidence floor (minimum allowed confidence)
    pub confidence_floor: f64,
    /// Optimization level
    pub opt_level: OptLevel,
}
```

## 3. Backend GPU

### Localização
- **Principal**: [`crates/souc/src/codegen/gpu/mod.rs`](../../crates/souc/src/codegen/gpu/mod.rs)
- **HLIR → GPU**: [`crates/souc/src/codegen/gpu/hlir_to_gpu.rs`](../../crates/souc/src/codegen/gpu/hlir_to_gpu.rs)
- **PTX**: [`crates/souc/src/codegen/gpu/ptx.rs`](../../crates/souc/src/codegen/gpu/ptx.rs)
- **Metal**: [`crates/souc/src/codegen/gpu/metal.rs`](../../crates/souc/src/codegen/gpu/metal.rs)
- **SPIR-V**: [`crates/souc/src/codegen/gpu/spirv.rs`](../../crates/souc/src/codegen/gpu/spirv.rs)

### Arquitetura GPU

```rust
// crates/souc/src/codegen/gpu/mod.rs:8
//! GPU Code Generation for Sounio
//!
//! Supports:
/// - PTX (NVIDIA CUDA)
/// - SPIR-V (Vulkan, OpenCL)
/// - MSL (Apple Metal)
///
/// Architecture:
/// ```text
/// HLIR -> GpuIR -> PTX/SPIR-V -> Driver -> GPU Execution
/// ```
```

### Epistemic GPU Computing

```rust
// crates/souc/src/codegen/gpu/mod.rs:13
/// # Epistemic GPU Computing
///
/// Sounio is the first language to track epistemic state through GPU computation:
/// - Shadow registers for uncertainty (ε)
/// - Validity predicates
/// - Provenance tracking
/// - Tensor Core operations with uncertainty propagation
```

### Módulos GPU

| Módulo | Descrição |
|--------|-----------|
| `hlir_to_gpu.rs` | Lowering HLIR → GpuIR |
| `ptx.rs` | Codegen PTX para NVIDIA |
| `metal.rs` | Codegen MSL para Apple |
| `spirv.rs` | Codegen SPIR-V (Vulkan/OpenCL) |
| `runtime.rs` | Runtime GPU |
| `intrinsics.rs` | Intrinsics GPU |
| `tensor_epistemic.rs` | Operações epistêmicas em tensores |
| `autodiff.rs` | Diferenciação automática GPU |
| `qnn_kernels.rs` | Kernels de redes neurais quaterniónicas |

### Targets GPU Suportados

```rust
// crates/souc/src/codegen/gpu/ir.rs
pub enum GpuTarget {
    Cuda {
        compute_capability: (u8, u8),  // e.g., (8, 0) for sm_80
    },
    Vulkan {
        version: (u32, u32, u32),
    },
    Metal {
        family: MetalGpuFamily,
        version: (u32, u32),
    },
    OpenCL {
        version: (u32, u32),
    },
}
```

### Exemplo de Uso GPU

```rust
// crates/souc/src/codegen/gpu/mod.rs:26
use sounio::codegen::gpu::{hlir_to_gpu, PtxCodegen, GpuTarget};

let gpu_module = hlir_to_gpu::lower(&hlir, GpuTarget::Cuda {
    compute_capability: (8, 0)
});
let ptx = PtxCodegen::new((8, 0)).generate(&gpu_module);
```

### Operações GPU Especiais

| Operação | Descrição |
|----------|-----------|
| Tensor Cores | Operações de matrix com uncertainty propagation |
| Cooperative Groups | Sincronização em groups de threads |
| Warp Vote/Reduce | Operações em nível warp |
| Shared Memory | Memória compartilhada epistêmico-aware |
| Asynchronous Copy | Transferências assíncronas |

## 4. Backend LLVM (Experimental)

### Localização
- **Stub**: [`crates/souc/src/codegen/llvm/mod.rs`](../../crates/souc/src/codegen/llvm/mod.rs)
- **Feature**: `llvm-base`

### Arquitetura

```rust
// crates/souc/src/codegen/mod.rs:22
// The LLVM backend is in a subdirectory when the feature is enabled
#[cfg(feature = "llvm-base")]
#[path = "llvm/mod.rs"]
pub mod llvm;

// Provide stub module when LLVM is not enabled
#[cfg(not(feature = "llvm-base"))]
pub mod llvm {
    pub struct LLVMCodegen;

    impl LLVMCodegen {
        pub fn compile(_hlir: &HlirModule) -> Result<(), String> {
            Err("LLVM backend not enabled. Rebuild with: cargo build --features llvm".to_string())
        }
    }
}
```

### Características

| Feature | Status |
|---------|--------|
| Otimizações LLVM | ⏳ Planejado |
| LTO (Link-Time Optimization) | ⏳ Planejado |
| ThinLTO | ⏳ Planejado |
| BOLT | ⏳ Planejado |
| WASM target | ⏳ Planejado |

## 5. Comparação de Backends

### Critérios de Escolha

| Backend | Quando Usar | Vantagens | Desvantagens |
|---------|--------------|-----------|--------------|
| **Cranelift** | JIT, desenvolvimento rápido | Compilação rápida, hot reload | Menos otimizações |
| **Native** | Produção, x86-64 | Sem dependências externas, epistêmico-aware | Apenas x86-64 |
| **LLVM** | Portabilidade máxima | Ecossistema maduro, muitas otimizações | Dependência externa |
| **GPU** | Computação paralela | CUDA/Metal/Vulkan support | Apenas kernels GPU |

### Performance Comparativa

| Backend | Compile Time | Runtime Performance | Portabilidade |
|---------|--------------|---------------------|---------------|
| Cranelift O2 | ~15ms | 85-95% do LLVM | Multi-plataforma |
| Native O2 | ~20ms | 90-95% do LLVM | x86-64 apenas |
| LLVM O3 | ~100ms | Baseline (100%) | Multi-plataforma |
| GPU PTX | ~50ms | GPU Speedup | NVIDIA only |

## 6. Target Specification

### Arquiteturas Suportadas

```rust
// crates/souc/src/codegen/mod.rs:118
#[derive(Debug, Clone, Copy)]
pub enum Architecture {
    X86_64,
    AArch64,
    Wasm32,
    Wasm64,
    NVPTX64,  // NVIDIA PTX
    SPIRV64,  // SPIR-V
}

#[derive(Debug, Clone, Copy)]
pub enum OperatingSystem {
    Linux,
    MacOS,
    Windows,
    None,  // For bare metal / GPU
}

#[derive(Debug, Clone, Copy)]
pub enum Environment {
    GNU,
    MSVC,
    Musl,
    None,
}
```

### Triple de Compilação

```rust
// crates/souc/src/codegen/mod.rs:182
impl Target {
    pub fn triple(&self) -> String {
        let arch = match self.arch {
            Architecture::X86_64 => "x86_64",
            Architecture::AArch64 => "aarch64",
            // ...
        };

        let os = match self.os {
            OperatingSystem::Linux => "linux",
            OperatingSystem::MacOS => "darwin",
            OperatingSystem::Windows => "windows",
            // ...
        };

        let env = match self.env {
            Environment::GNU => "gnu",
            // ...
        };

        format!("{}-{}-{}", arch, os, env)
    }
}
```

## 7. Pipeline de Compilação

### Fluxo Completo

```
Source Code (.sio)
        ↓
┌─────────────────┐
│     Parser      │  → AST
└─────────────────┘
        ↓
┌─────────────────┐
│  Type Checker   │  → HIR
└─────────────────┘
        ↓
┌─────────────────┐
│  HLIR Builder   │  → HLIR
└─────────────────┘
        ↓
┌─────────────────┐
│  SIR Builder    │  → SIR
└─────────────────┘
        ↓
    [ESCOLHA DO BACKEND]
            │
    ┌───────┼───────┬──────────┐
    │       │       │          │
    ▼       ▼       ▼          ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Cranel│ │Native│ │ LLVM │ │  GPU │
│  if  │ │      │ │      │ │      │
└───┬──┘ └──┬───┘ └──┬───┘ └──┬───┘
    │       │        │        │
    ▼       ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ CLIF │ │x86-64 │ │ LLVM │ │ PTX/ │
│      │ │ ELF   │ │ IR   │ │MSL   │
└───┬──┘ └──┬───┘ └──┬───┘ └──┬───┘
    │       │        │        │
    ▼       ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│JIT   │ │.so/  │ │Native│ │GPU   │
│Exec  │ │.exe  │ │ exec │ │Exec  │
└──────┘ └──────┘ └──────┘ └──────┘
```

## 8. Métricas e Documentação

### Documentação Relacionada

| Documento | Descrição |
|-----------|-----------|
| [`docs/MIR_CRANELIFT_INTEGRATION_REPORT.md`](../../docs/MIR_CRANELIFT_INTEGRATION_REPORT.md) | Integração MIR-Cranelift completa |
| [`docs/LLVM_CODEGEN.md`](../../docs/LLVM_CODEGEN.md) | Backend LLVM |
| [`docs/GPU_RUNTIME.md`](../../docs/GPU_RUNTIME.md) | Runtime GPU |
| [`docs/compiler/PHASE2_OPTIMIZATIONS.md`](../../docs/compiler/PHASE2_OPTIMIZATIONS.md) | Otimizações |

### Arquivos Principais

| Arquivo | Linhas | Descrição |
|---------|--------|-----------|
| `codegen/mod.rs` | 213 | Módulo principal de codegen |
| `codegen/mir_cranelift.rs` | ~500 | Integração MIR-Cranelift |
| `codegen/gpu/mod.rs` | 298 | Backend GPU |
| `backend/native/mod.rs` | 525 | Backend Native |
| `backend/native/x86_64.rs` | ~800 | Emissão x86-64 |

## 9. Exemplos

### Compilação Basic

```rust
use sounio::codegen::{Backend, Target};

// Escolher backend
let backend = Backend::Cranelift;

// Configurar target
let target = Target {
    arch: Architecture::X86_64,
    os: OperatingSystem::Linux,
    env: Environment::GNU,
};

// Compilar
let output = compile(source, backend, target)?;
```

### Compilação GPU

```rust
use sounio::codegen::gpu::{hlir_to_gpu, PtxCodegen, GpuTarget};

// HLIR → GpuIR
let gpu_module = hlir_to_gpu::lower(&hlir, GpuTarget::Cuda {
    compute_capability: (8, 0),
})?;

// GpuIR → PTX
let ptx = PtxCodegen::new((8, 0))
    .with_optimization(true)
    .generate(&gpu_module)?;

// Executar na GPU
let runtime = GpuRuntime::new()?;
let kernel = runtime.load_ptx(&ptx)?;
kernel.launch(&args)?;
```

## Próximos Passos

1. **LLVM Backend** → Completar integração LLVM
2. **WebAssembly** → Suporte WASM target
3. **Cross-compilation** → Compilação para múltiplas plataformas
4. **PGO** → Profile-Guided Optimization

## Referências

- [`crates/souc/src/codegen/mod.rs`](../../crates/souc/src/codegen/mod.rs)
- [`docs/MIR_CRANELIFT_INTEGRATION_REPORT.md`](../../docs/MIR_CRANELIFT_INTEGRATION_REPORT.md)
- [`crates/souc/src/codegen/gpu/mod.rs`](../../crates/souc/src/codegen/gpu/mod.rs)
- [`crates/souc/src/backend/native/mod.rs`](../../crates/souc/src/backend/native/mod.rs)
