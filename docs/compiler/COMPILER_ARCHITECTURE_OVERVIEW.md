# Arquitetura Completa do Compilador Sounio

## Visão Geral

O **Compilador Sounio** (`souc`) é um compilador multi-estágio para a linguagem de programação **Sounio** - uma linguagem de sistemas para computação epistêmica. O compilador é escrito em Rust (edition 2024) e suporta múltiplos backends de geração de código.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COMPILADOR SOUNIO - PIPELINE COMPLETO                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Source Code (.sio)                                                              │
│        │                                                                        │
│        ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐      │
│  │                           FRONTEND                                        │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │      │
│  │  │    Lexer    │→ │   Parser   │→ │   AST      │→ │ Name       │  │      │
│  │  │  (Logos)    │  │ (Recursive │  │  Definition │  │ Resolver   │  │      │
│  │  │             │  │  Descent)   │  │             │  │            │  │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │      │
│  │         │               │               │               │               │      │
│  │         └───────────────┴───────────────┴───────────────┘               │      │
│  │                              │                                            │      │
│  │                              ▼                                            │      │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │      │
│  │  │                    TYPE CHECKER                                   │   │      │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │      │
│  │  │  │ Type       │→ │ Effect      │→ │ Unit       │              │   │      │
│  │  │  │ Inference  │  │ Checking    │  │ Checking   │              │   │      │
│  │  │  │ (Bidirect) │  │             │  │            │              │   │      │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │      │
│  │  │                              │                                    │   │      │
│  │  └──────────────────────────────┴────────────────────────────────────┘   │      │
│  │                                    │                                        │      │
│  └────────────────────────────────────┼────────────────────────────────────┘      │
│                                       │                                             │
│                                       ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐      │
│  │                         MIDDLE-END                                        │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │      │
│  │  │    HIR      │→ │   HLIR     │→ │    SIR      │→ │    MIR      │  │      │
│  │  │ High-Level  │  │ Polyhedral │  │ Scientific │  │ SSA-based   │  │      │
│  │  │    IR       │  │    IR      │  │    IR       │  │    IR       │  │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │      │
│  │         │               │               │               │               │      │
│  │         └───────────────┴───────────────┴───────────────┘               │      │
│  │                              │                                            │      │
│  │                              ▼                                            │      │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │      │
│  │  │                    OPTIMIZATION PASSES                            │   │      │
│  │  │  ┌─────────────────────────────────────────────────────────┐   │   │      │
│  │  │  │ Constant Propagation │ DCE │ CSE │ LICM │ Inlining   │   │   │      │
│  │  │  └─────────────────────────────────────────────────────────┘   │   │      │
│  │  └───────────────────────────────────────────────────────────────┘   │      │
│  │                                    │                                    │      │
│  └────────────────────────────────────┼────────────────────────────────────┘      │
│                                       │                                            │
│                                       ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐      │
│  │                           BACKEND                                          │      │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │      │
│  │  │                      BACKENDS DISPONÍVEIS                         │   │      │
│  │  ├──────────────────────────────────────────────────────────────────┤   │      │
│  │  │                                                                  │   │      │
│  │  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │      │
│  │  │   │   Cranelift │  │   Native    │  │    LLVM     │            │   │      │
│  │  │   │   (JIT)    │  │  (x86-64)  │  │  (Experimental) │            │   │      │
│  │  │   │  ~15ms      │  │  Sem LLVM   │  │ Portável    │            │   │      │
│  │  │   └─────────────┘  └─────────────┘  └─────────────┘            │   │      │
│  │  │                                                                  │   │      │
│  │  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │      │
│  │  │   │     GPU     │  │   Debug     │  │   bytecode   │            │   │      │
│  │  │   │ PTX/SPIR-V │  │   DWARF     │  │  (Self-host) │            │   │      │
│  │  │   │  / Metal   │  │             │  │             │            │   │      │
│  │  │   └─────────────┘  └─────────────┘  └─────────────┘            │   │      │
│  │  │                                                                  │   │      │
│  │  └──────────────────────────────────────────────────────────────────┘   │      │
│  │                                    │                                    │      │
│  └────────────────────────────────────┼────────────────────────────────────┘      │
│                                       │                                            │
│                                       ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐      │
│  │                          OUTPUTS                                           │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │      │
│  │  │ ELF/.so    │  │ PTX/SPIR-V  │  │  DWARF     │  │   JIT      │  │      │
│  │  │ (Linux)    │  │ (GPU)       │  │  (Debug)   │  │  (Memory)  │  │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │      │
│  └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Estatísticas do Projeto

| Métrica | Valor |
|---------|-------|
| **Linguagem** | Rust 2024 Edition |
| **Versão** | 0.100.0 |
| **Linhas do Compilador** | ~566,719 |
| **Linhas da Stdlib** | 215,000+ |
| **Módulos Principais** | 50+ |
| **Backends** | 4 (Cranelift, Native, LLVM, GPU) |

## 1. Frontend

### 1.1 Lexer (Tokenização)

**Localização**: [`crates/souc/src/lexer/`](../../crates/souc/src/lexer/)

**Tecnologia**: Logos (gerador de lexer baseado em regex)

**Componentes**:
- [`mod.rs`](../../crates/souc/src/lexer/mod.rs) - Função principal `lex()`
- [`tokens.rs`](../../crates/souc/src/lexer/tokens.rs) - 799 linhas, 150+ `TokenKind`

**Tokens Suportados**:
- Keywords: `module`, `fn`, `let`, `struct`, `enum`, `effect`, `handler`, `with`, etc.
- Literais: `i32`, `f64`, `string`, `unit literals` (`500_mg`)
- Operadores: `+`, `-`, `*`, `/`, `==`, `!=`, `<=`, `>=`, `&&`, `||`
- Doc Comments: `///`, `//!`, `/** ... */`

**Fluxo**:
```
Source Code
    ↓
lex(source) → Vec<Token>
    ↓
Token { kind, span, text }
```

### 1.2 Parser (Análise Sintática)

**Localização**: [`crates/souc/src/parser/`](../../crates/souc/src/parser/)

**Técnica**: Recursive Descent + Pratt Parsing

**Componentes**:
- [`mod.rs`](../../crates/souc/src/parser/mod.rs) - 5,851 linhas
- [`errors.rs`](../../crates/souc/src/parser/errors.rs) - Error handling
- [`recovery.rs`](../../crates/souc/src/parser/recovery.rs) - Error recovery

**Estrutura do Parser**:
```rust
pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    id_gen: IdGenerator,
    allow_struct_literals: bool,
    node_spans: HashMap<NodeId, Span>,
    pending_gt: bool,  // Para splitting de >>
    source: &'a str,
}
```

**Features**:
- Desambiguação de struct literals vs blocks
- Splitting automático de `>>` para generics aninhados
- Suporte a doc comments
- Error recovery limitado

### 1.3 AST (Abstract Syntax Tree)

**Localização**: [`crates/souc/src/ast/mod.rs`](../../crates/souc/src/ast/mod.rs) (2,003 linhas)

**Estrutura Raiz**:
```rust
pub struct Ast {
    pub module_name: Option<Path>,
    pub items: Vec<Item>,
    pub inner_doc: Option<String>,
    pub node_spans: HashMap<NodeId, Span>,
}
```

**Items Suportados**:
```rust
pub enum Item {
    Function(FnDef),
    Struct(StructDef),
    Enum(EnumDef),
    Trait(TraitDef),
    Impl(ImplDef),
    TypeAlias(TypeAliasDef),
    Effect(EffectDef),
    Handler(HandlerDef),
    Import(ImportDef),
    Export(ExportDef),
    Extern(ExternBlock),
    Global(GlobalDef),
    // Scientific DSL
    OntologyImport(OntologyImportDef),
    AlignDecl(AlignDef),
    OdeDef(OdeDef),
    PdeDef(PdeDef),
    CausalModel(CausalModelDef),
    Unit(UnitDef),
    Module(ModuleDef),
}
```

## 2. Sistema de Tipos

### 2.1 Tipos Epistêmicos

**Localização**: [`crates/souc/src/epistemic/`](../../crates/souc/src/epistemic/)

**Tipo Principal - Knowledge<T>**:
```rust
pub struct Knowledge {
    pub content: Box<Type>,      // Tipo de conteúdo
    pub temporal: ContextTime,   // τ: Contexto temporal
    pub epistemic: EpistemicStatus,  // ε: Status epistêmico
    pub domain: OntologyBinding,   // δ: Binding ontológico
    pub provenance: Provenance,    // Φ: Traço de functor
    pub span: Span,
}
```

**Arquitetura Ontológica (4 Camadas)**:
```
L1: Primitive (BFO, RO, COB) → ~850 termos, compilados
L2: Foundation (PATO, UO, IAO) → ~8,000 termos, stdlib
L3: Domain (ChEBI, GO, DOID) → ~500,000 termos, lazy load
L4: Federated (BioPortal) → ~15M termos, runtime
```

### 2.2 Sistema de Unidades

**Localização**: [`crates/souc/src/units/`](../../crates/souc/src/units/)

**Dimensões SI (7 Grandezas Base)**:
```rust
pub struct Dimension {
    pub mass: i8,        // [M]
    pub length: i8,      // [L]
    pub time: i8,        // [T]
    pub current: i8,     // [I]
    pub temperature: i8, // [Θ]
    pub amount: i8,       // [N]
    pub luminosity: i8,  // [J]
}
```

**Unidades PK/PD**:
- `mg`, `mL`, `L`, `kg`
- `mg/mL`, `mg/L`, `L/h`
- `mg*h/L` (AUC)

### 2.3 Type Checker

**Localização**: [`crates/souc/src/check/mod.rs`](../../crates/souc/src/check/mod.rs) (8,385 linhas)

**Estrutura**:
```rust
pub struct TypeChecker {
    env: TypeEnv,
    type_defs: HashMap<String, TypeDef>,
    effects: EffectInference,
    units: UnitChecker,
    next_type_var: u32,
    next_effect_var: u32,
    constraints: Vec<TypeConstraint>,
    ontology_resolver: Option<OntologyResolver>,
    conformal_checker: Option<ConformalTypeChecker>,
    // ... mais campos
}
```

**Features**:
- Inferência bidirecional (synthesis + checking)
- Unificação de tipos
- Verificação de efeitos
- Verificação de unidades
- Compatibilidade semântica ontológica
- Refinement types
- Conformal prediction

## 3. Sistema de Efeitos

### 3.1 Efeitos Suportados

**Localização**: [`crates/souc/src/effects/`](../../crates/souc/src/effects/)

| Efeito | Handler | Operações |
|--------|---------|-----------|
| `IO` | `IOHandler` | `print`, `read_file`, `write_file` |
| `Mut` | `MutHandler` | `get`, `set`, `modify` |
| `Alloc` | `AllocHandler` | `alloc`, `dealloc` |
| `Panic` | `PanicHandler` | `panic`, `assert` |
| `Async` | `AsyncHandler` | `spawn`, `await`, `join`, `select` |
| `GPU` | `GpuHandler` | `launch`, `sync` |
| `Prob` | `ProbHandler` | `sample`, `observe` |
| `Div` | `DivHandler` | `div`, `checked_div` |
| `Exn` | `ExnHandler` | `throw`, `try_catch` |
| `Epistemic` | `EpistemicHandler` | `degrade`, `assert_confidence` |

### 3.2 Effect Handlers

**Localização**: [`crates/souc/src/effects/handlers/`](../../crates/souc/src/effects/handlers/)

**Handlers Implementados**:
- `IOHandler` - Input/output com tracking epistêmico
- `AsyncHandler` - Concorrência estruturada
- `MutHandler` - Estado mutável
- `ProbHandler` - Operações probabilísticas
- `EpistemicHandler` - Tracking epistêmico
- `GpuHandler` - Operações GPU
- E mais 10+ handlers

### 3.3 Continuations (CPS)

**Localização**: [`crates/souc/src/effects/continuation.rs`](../../crates/souc/src/effects/continuation.rs) (1,229 linhas)

**Base Teórica**: Plotkin & Pretnar (2009), Leijen (2017)

**Tipos de Continuation**:
```rust
pub enum ResumePoint {
    InterpreterClosure { resume_fn: OneShotResumeFn },
    InterpreterMultiShot { resume_fn: MultiShotResumeFn },
    Jit { return_address: usize, saved_registers: Vec<u64> },
    Stub,
}
```

## 4. Representações Intermediárias (IRs)

### 4.1 HIR (High-Level IR)

**Localização**: [`crates/souc/src/hir/mod.rs`](../../crates/souc/src/hir/mod.rs) (1,348 linhas)

**Características**:
- Tipos resolvidos
- Nomes resolvidos
- Desugar de constructs
- Informação de ownership

```rust
pub struct Hir {
    pub items: Vec<HirItem>,
    pub externs: Vec<HirExternBlock>,
}
```

### 4.2 HLIR (Polyhedral IR)

**Localização**: [`crates/souc/src/hlir/`](../../crates/souc/src/hlir/)

**Features**:
- Análise de loop nests
- Transformações afins
- Otimização de locality
- Polyhedral model

### 4.3 SIR (Scientific IR)

**Localização**: [`crates/souc/src/sir/`](../../crates/souc/src/sir/)

**Features**:
- Operações ODE/PDE
- Operações tensor
- Diferenciação automática
- Kernels GPU

### 4.4 MIR (SSA-based IR)

**Localização**: [`crates/souc/src/mir/`](../../crates/souc/src/mir/)

**Features**:
- SSA form
- CFG (Control Flow Graph)
- Otimizações baseadas em análise de dados

## 5. Otimizações

### 5.1 Passes MIR

**Localização**: [`crates/souc/src/mir/optimization/`](../../crates/souc/src/mir/optimization/)

**Passes Implementados**:
| Pass | Descrição |
|------|-----------|
| `ConstantPropagation` | Propagação de constantes |
| `DeadCodeElimination` | Eliminação de código morto |
| `CommonSubexpressionElimination` | CSE |
| `LoopInvariantCodeMotion` | LICM |
| `FunctionInlining` | Inlining de funções |

**Níveis de Otimização**:
- **O0**: Nenhum
- **O1**: Constant Propagation, DCE
- **O2**: + CSE, LICM
- **O3**: + Function Inlining

### 5.2 Otimizações GPU

**Localização**: [`crates/souc/src/codegen/gpu/`](../../crates/souc/src/codegen/gpu/)

**Features**:
- Kernel fusion
- Auto-tuning
- Async memory pipeline
- Cooperative groups
- Tensor cores

## 6. Backends de Geração de Código

### 6.1 Cranelift (Primary)

**Localização**: [`crates/souc/src/codegen/cranelift.rs`](../../crates/souc/src/codegen/cranelift.rs)

**Features**:
- JIT compilation rápida (~15ms)
- Otimizações MIR
- Multi-plataforma

**Pipeline**:
```
MIR → Otimizações MIR → Cranelift IR → Código Nativo
```

### 6.2 Native (x86-64)

**Localização**: [`crates/souc/src/backend/native/`](../../crates/souc/src/backend/native/)

**Features**:
- Sem dependência LLVM
- Emission x86-64 direta
- Epistemic-aware register allocation
- Thermal modeling
- Cycle/power estimation

**Componentes**:
- [`x86_64.rs`](../../crates/souc/src/backend/native/x86_64.rs) - Emission
- [`alloc.rs`](../../crates/souc/src/backend/native/alloc.rs) - Register allocation
- [`metrics.rs`](../../crates/souc/src/backend/native/metrics.rs) - Métricas
- [`elf.rs`](../../crates/souc/src/backend/native/elf.rs) - ELF generation

### 6.3 LLVM (Experimental)

**Localização**: [`crates/souc/src/codegen/llvm/`](../../crates/souc/src/codegen/llvm/)

**Features**:
- Suporte completo a LLVM
- LTO (planejado)
- Multi-plataforma

### 6.4 GPU

**Localização**: [`crates/souc/src/codegen/gpu/`](../../crates/souc/src/codegen/gpu/)

**Targets**:
| Target | Descrição |
|--------|-----------|
| PTX | NVIDIA CUDA |
| SPIR-V | Vulkan/OpenCL |
| MSL | Apple Metal |

**Features**:
- Epistemic GPU computing
- Tensor cores
- Quaternionic neural networks
- Counterfactual execution

## 7. Funcionalidades Científicas

### 7.1 Álgebra Hypercomplexa

**Quaternions** (`Quat`):
- 4 componentes: a + bi + cj + dk
- Multiplicação: 16 FLOPs
- SLERP (spherical interpolation)

**Octonions** (`Octonion`):
- 8 componentes: a + bi + cj + dk + el + fil + gjl + hkl
- Multiplicação: 64 FLOPs
- Não-associativo
- Norma multiplicativa

**Sedenions** (`Sedenion`):
- 16 componentes
- Possui zero divisors

### 7.2 Quaternionic Neural Networks (QNN)

**Tipos HLIR**:
```rust
QuatLinear,   // Fully connected quaternion layer
QuatConv2d,   // Quaternionic 2D convolution
QuatRnnState, // RNN state
QuatGate,     // Gate representation
```

**Vantagens**:
- 4× eficiência de parâmetros
- Natural 3D rotation handling
- SIMD intrinsics eficientes

### 7.3 ODE/PDE Solvers

**DSL Integrada**:
```sio
ode LotkaVolterra {
    state { x: f64, y: f64 }
    params { alpha: f64 = 1.1, beta: f64 = 0.4 }
    equations {
        dx/dt = alpha * x - beta * x * y,
        dy/dt = delta * x * y - gamma * y
    }
}
```

**Métodos**:
- Euler, RK4, RK45
- BS5 (Bogacki-Shampine)
- CVODE (BDF)

### 7.4 Computação Quântica

**Localização**: [`crates/souc/src/quantum/`](../../crates/souc/src/quantum/)

**Features**:
- Circuit representation
- VQE (Variational Quantum Eigensolver)
- UCCSD ansatz
- PennyLane integration
- Hamiltonian simulation

**Portas**:
H, X, Y, Z, T, S, CNOT, CZ, SWAP, RX, RY, RZ, U3, ISWAP

### 7.5 Machine Learning Epistêmico

**Gaussian Processes**:
- Kernels: RBF, Matern32/52, Periodic
- Predição epistêmica

**MCMC Samplers**:
- Metropolis-Hastings
- Hamiltonian MC
- NUTS

**PCE (Polynomial Chaos Expansion)**:
- Hermite, Legendre, Laguerre
- Multi-index support

## 8. Documentação Criada

### Documentos de Arquitetura

| Documento | Descrição | Linhas |
|-----------|-----------|--------|
| [`LEXER_PARSER_ARCHITECTURE.md`](LEXER_PARSER_ARCHITECTURE.md) | Lexer, Parser, AST | ~350 |
| [`TYPE_SYSTEM_ARCHITECTURE.md`](TYPE_SYSTEM_ARCHITECTURE.md) | Sistema de Tipos | ~400 |
| [`EFFECT_SYSTEM_ARCHITECTURE.md`](EFFECT_SYSTEM_ARCHITECTURE.md) | Sistema de Efeitos | ~300 |
| [`CODE_GENERATION_ARCHITECTURE.md`](CODE_GENERATION_ARCHITECTURE.md) | Backends | ~400 |
| [`SCIENTIFIC_FEATURES_ARCHITECTURE.md`](SCIENTIFIC_FEATURES_ARCHITECTURE.md) | Funcionalidades Científicas | ~500 |

### Documentos de Referência Existentes

| Documento | Descrição |
|-----------|-----------|
| [`docs/compiler/ARCHITECTURE.md`](../../docs/compiler/ARCHITECTURE.md) | Arquitetura geral |
| [`docs/MIR_CRANELIFT_INTEGRATION_REPORT.md`](../../docs/MIR_CRANELIFT_INTEGRATION_REPORT.md) | Integração MIR-Cranelift |
| [`docs/compiler/OCTONION_ALGEBRA.md`](../../docs/compiler/OCTONION_ALGEBRA.md) | Álgebra de Octonions |
| [`hugo/content/architecture/compiler-pipeline.md`](../../hugo/content/architecture/compiler-pipeline.md) | Pipeline do Compilador |

## 9. Estrutura de Diretórios

```
sounio/
├── crates/
│   └── souc/
│       └── src/
│           ├── lexer/              # Tokenização
│           ├── parser/             # Parsing
│           ├── ast/                # AST
│           ├── resolve/            # Name resolution
│           ├── check/              # Type checking
│           ├── types/              # Type system
│           ├── epistemic/          # Knowledge types
│           ├── units/              # Units of measure
│           ├── effects/            # Effect system
│           ├── hir/                # High-level IR
│           ├── hlir/               # Polyhedral IR
│           ├── sir/                # Scientific IR
│           ├── mir/                # SSA IR
│           ├── codegen/            # Code generation
│           │   ├── cranelift.rs
│           │   ├── gpu/
│           │   └── llvm/
│           ├── backend/
│           │   └── native/
│           ├── linear/             # Linear algebra
│           ├── geometry/           # Geometric algebra
│           ├── quantum/            # Quantum computing
│           ├── optimizer/           # Optimization
│           └── ...
├── stdlib/                         # Standard library (215K+ lines)
├── tests/                          # Integration tests
├── benches/                         # Benchmarks
└── docs/
    └── compiler/                   # Documentation
```

## 10. Métricas de Componentes

| Componente | Arquivo | Linhas |
|------------|---------|--------|
| Lexer tokens | `lexer/tokens.rs` | 799 |
| Parser | `parser/mod.rs` | 5,851 |
| AST | `ast/mod.rs` | 2,003 |
| Type Checker | `check/mod.rs` | 8,385 |
| Effects inference | `effects/inference.rs` | 1,335 |
| Continuation | `effects/continuation.rs` | 1,229 |
| HLIR IR | `hlir/ir.rs` | 648 |
| HIR | `hir/mod.rs` | 1,348 |
| GPU codegen | `codegen/gpu/mod.rs` | 298 |
| Native backend | `backend/native/mod.rs` | 525 |

## 11. Fluxo de Compilação

### Compilação AOT (Ahead-of-Time)

```bash
# Compilar para executável
cargo build -p souc --release

# Compilar programa
./target/release/souc compile input.sio -o output

# Geração de código
./target/release/souc compile input.sio --backend native -o output
```

### Compilação JIT

```rust
use sounio_compiler::codegen::mir_cranelift::MirAwareCraneliftJit;

let jit = MirAwareCraneliftJit::new()
    .with_optimization()
    .with_mir_optimization(OptimizationLevel::O3);

let compiled = jit.compile_mir(&mir_module)?;
```

### Targets Suportados

| Arquitetura | Sistema Operacional | Ambiente |
|-------------|---------------------|----------|
| x86_64 | Linux, macOS, Windows | GNU, MSVC, Musl |
| AArch64 | Linux, macOS | GNU |
| NVPTX64 | CUDA | - |
| SPIRV64 | Vulkan | - |
| Wasm32/Wasm64 | Browsers | - |

## 12. Recursos Adicionais

### Tutoriais
- [`docs/getting-started.md`](../../docs/getting-started.md)
- [`docs/guide/tutorial.md`](../../docs/guide/tutorial.md)

### Exemplos
- [`examples/`](../../examples/)
- [`crates/souc/examples/`](../../crates/souc/examples/)
- [`tests/`](../../tests/)

### Benchmarks
- [`benches/compiler/`](../../benches/compiler/)

### Documentação da API
- [`docs/reference/STDLIB_REFERENCE.md`](../../docs/reference/STDLIB_REFERENCE.md)

---

## Referências

1. **Lexer/Parser**: [`crates/souc/src/lexer/`](../../crates/souc/src/lexer/), [`crates/souc/src/parser/`](../../crates/souc/src/parser/)
2. **Sistema de Tipos**: [`crates/souc/src/types/`](../../crates/souc/src/types/), [`crates/souc/src/epistemic/`](../../crates/souc/src/epistemic/)
3. **Type Checker**: [`crates/souc/src/check/mod.rs`](../../crates/souc/src/check/mod.rs)
4. **Sistema de Efeitos**: [`crates/souc/src/effects/`](../../crates/souc/src/effects/)
5. **IRs**: [`crates/souc/src/hir/`](../../crates/souc/src/hir/), [`crates/souc/src/hlir/`](../../crates/souc/src/hlir/), [`crates/souc/src/mir/`](../../crates/souc/src/mir/)
6. **Backends**: [`crates/souc/src/codegen/`](../../crates/souc/src/codegen/), [`crates/souc/src/backend/`](../../crates/souc/src/backend/)
7. **Funcionalidades Científicas**: [`crates/souc/src/linear/`](../../crates/souc/src/linear/), [`crates/souc/src/quantum/`](../../crates/souc/src/quantum/)
