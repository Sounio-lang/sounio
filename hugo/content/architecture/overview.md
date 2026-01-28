---
title: "Architecture Overview"
description: "High-level architecture of the Sounio compiler: an 8-stage pipeline from source to machine code with 5 backend targets."
---

## Compiler Architecture

Sounio's compiler follows an 8-stage pipeline, progressively lowering source code through typed intermediate representations to machine code. The design prioritizes **epistemic correctness**---uncertainty and provenance metadata flows through every IR layer, from parsing to native code emission.

```
Source (.sio) --> Lexer --> Parser --> AST --> Type Checker --> HIR --> HLIR (SSA) --> Codegen
                                                                         |
                                                              +---------+---------+
                                                              |         |         |
                                                            LLVM   Cranelift   Native
                                                              |         |         |
                                                           Binary     JIT    ELF/Mach-O
                                                              |
                                                          SIR (Scientific IR)
                                                              |
                                                     +--------+--------+
                                                     |                 |
                                                  GPU IR           x86-64
                                                     |
                                              +------+------+
                                              |      |      |
                                            PTX   SPIR-V  Metal
```

### Pipeline Stages

| Stage | Module | Key File | Purpose |
|-------|--------|----------|---------|
| **1. Lexer** | `lexer/` | `mod.rs:14` | Tokenization via Logos |
| **2. Parser** | `parser/` | `mod.rs:19` | Recursive descent, 5,656 lines |
| **3. AST** | `ast/` | `mod.rs:113` | Untyped abstract syntax tree |
| **4. Type Checker** | `check/` | `mod.rs:37` | Bidirectional inference, 7,153 lines |
| **5. HIR** | `hir/` | `mod.rs:20` | Typed high-level IR with effects |
| **6. HLIR** | `hlir/` | `ir.rs:13` | SSA-based low-level IR |
| **7. SIR** | `sir/` | `mod.rs:19` | Domain-aware scientific IR |
| **8. Codegen** | `codegen/`, `backend/` | Various | Multi-target code generation |

### Entry Points

The public API is defined in `compiler/src/lib.rs`:

- **`compile(source)`** (line 114): Full pipeline, source to executable
- **`parse(source)`** (line 208): Lex + parse to AST
- **`typecheck(source)`** (line 201): Lex + parse + type check to HIR
- **`compile_to_gpu(source)`** (line 156): GPU variant with PTX output

### Backend Targets

Sounio supports **5 code generation backends**:

| Backend | Use Case | Key Strength |
|---------|----------|-------------|
| **LLVM** | Production builds | Peak optimization (O2/O3) |
| **Cranelift JIT** | Development/REPL | Fast compilation |
| **Native x86-64** | Zero-dependency builds | No LLVM required |
| **PTX (NVIDIA)** | GPU compute | Direct CUDA kernel generation |
| **Metal (Apple)** | Apple GPU compute | M1/M2/M3 acceleration |

### Key Design Decisions

1. **Epistemic types flow through all IRs**: `Knowledge<T>` is not erased at any stage. The HLIR encodes epistemic metadata (`mode`, `epsilon_bound`, `provenance_id`), and GPU backends use shadow registers for uncertainty tracking.

2. **Domain-specific SIR**: Unlike LLVM IR, the Scientific IR preserves ODE patterns, matrix operations, probability primitives, and autodiff structure for domain-specific optimizations.

3. **Effect system as type-level metadata**: Effects (`IO`, `Mut`, `GPU`, `Prob`, etc.) are tracked through the HIR and inform backend selection---a function with the `GPU` effect triggers GPU codegen.

4. **Multiple IR layers for different optimizations**: HIR enables semantic optimizations (effect inference, linearity checking), HLIR enables control-flow optimizations (SSA), and SIR enables scientific optimizations (ODE fusion, tensor layout).

### Module Organization

The compiler is organized into functional groups:

- **Frontend**: `lexer`, `parser`, `ast`, `resolve`
- **Analysis**: `check`, `typeck`, `types`, `effects`, `ownership`, `linear`
- **IR Lowering**: `hir`, `hlir`, `sir`
- **Codegen**: `codegen` (LLVM, Cranelift, GPU), `backend` (Native)
- **Domain**: `units`, `refinement`, `epistemic`, `ontology`, `bio`
- **Advanced**: `smt` (Z3), `temporal`, `causal`, `quantum`
- **Tools**: `lsp`, `pkg`, `repl`, `lint`, `doc`, `profiling`

### Further Reading

- [Compiler Pipeline](/architecture/compiler-pipeline/) --- Detailed stage-by-stage walkthrough
- [Type System](/architecture/type-system/) --- Bidirectional inference and linear types
- [Effect System](/architecture/effect-system/) --- Algebraic effects with handlers
- [GPU Codegen](/architecture/gpu-codegen/) --- PTX and Metal backend details
- [Epistemic Types](/architecture/epistemic-types/) --- `Knowledge<T>` implementation
- [Ontology Integration](/architecture/ontology-integration/) --- 15M+ scientific terms
