# Sounio Compiler Architecture

This document provides an overview of the Sounio compiler architecture for contributors who want to understand, maintain, or extend the compiler.

## Overview

The Sounio compiler (`souc`) transforms Sounio source code (`.sio` files) into executable programs. It is implemented in Rust and features a multi-stage pipeline designed to support Sounio's unique features:

- **Epistemic computing** with uncertainty propagation
- **Algebraic effects** with row polymorphism
- **Units of measure** with compile-time dimensional analysis
- **GPU-native computation** with PTX/SPIR-V/Metal backends
- **Refinement types** with SMT-backed verification

## Design Philosophy

### Domain-Aware Compilation

Unlike general-purpose compilers, Sounio preserves domain-specific semantic information throughout compilation. This enables optimizations impossible in traditional compilers:

- **Epistemic awareness**: Confidence and provenance flow through every IR stage
- **Numerical semantics**: IEEE 754 guarantees and precision requirements
- **Probability primitives**: First-class distribution sampling
- **Scientific patterns**: ODE solver steps, compartment models

### Key Architectural Decisions

1. **Multi-Stage IR**: The compiler uses multiple intermediate representations (HIR, HLIR, SIR) to enable different optimization opportunities at each level.

2. **Bidirectional Type Inference**: Type checking uses bidirectional inference, allowing type information to flow both up and down the AST.

3. **Effect Row Polymorphism**: Effects are tracked using row polymorphism, enabling functions to be polymorphic over their computational effects.

4. **Pluggable Backends**: The codegen stage supports multiple backends (LLVM, Cranelift, GPU) through a unified interface.

## Module Map

The compiler source is organized in `compiler/src/`:

```
compiler/src/
+-- lib.rs              # Main library, re-exports, compile/interpret entry points
|
+-- Frontend
|   +-- lexer/          # Logos-based tokenization
|   +-- parser/         # Recursive descent + Pratt parsing
|   +-- ast/            # Abstract Syntax Tree types
|
+-- Middle End (Analysis & Transformation)
|   +-- check/          # Type checking, bidirectional inference
|   +-- types/          # Type system types and operations
|   +-- typeck/         # Additional type checking utilities
|   +-- effects/        # Algebraic effect system
|   +-- linear/         # Linear/affine type checking
|   +-- ownership/      # Ownership analysis
|   +-- units/          # Dimensional analysis
|   +-- refinement/     # Refinement type constraints
|   +-- smt/            # Z3 SMT solver integration
|   +-- epistemic/      # Knowledge<T> type handling
|   +-- ontology/       # Scientific ontology (15M+ terms)
|   +-- resolve/        # Name resolution
|
+-- Intermediate Representations
|   +-- hir/            # High-level IR (typed AST)
|   +-- hlir/           # SSA-based low-level IR
|   +-- sir/            # Scientific IR (domain-specific)
|
+-- Backend (Code Generation)
|   +-- codegen/        # Backend dispatcher
|   +-- codegen/llvm/   # LLVM backend (AOT)
|   +-- codegen/cranelift.rs  # Cranelift JIT backend
|   +-- codegen/gpu/    # GPU backends (PTX, SPIR-V, Metal)
|
+-- Runtime & Tools
|   +-- interp/         # Tree-walking interpreter
|   +-- repl/           # Interactive REPL
|   +-- lsp/            # Language Server Protocol
|   +-- pkg/            # Package manager
|   +-- fmt/            # Code formatter
|   +-- lint/           # Linting infrastructure
|   +-- doc/            # Documentation generator
|
+-- Supporting Infrastructure
|   +-- common/         # Common types (Span, NodeId, etc.)
|   +-- diagnostic/     # Error reporting (miette integration)
|   +-- diagnostics/    # Additional diagnostic utilities
|   +-- sourcemap/      # Source mapping for debugging
|   +-- layout/         # Memory layout optimization
|   +-- locality/       # Cache-aware data placement
|   +-- runtime/        # Runtime support library
```

## Compilation Pipeline Overview

```
Source (.sio)
     |
     v
  [Lexer] -----> Token Stream
     |
     v
  [Parser] ----> AST (Abstract Syntax Tree)
     |
     v
  [TypeChecker] -> HIR (High-level IR, typed AST)
     |
     v
  [HLIR Lower] -> HLIR (SSA form, explicit control flow)
     |
     v
  [SIR Lower] --> SIR (Scientific IR, domain metadata)
     |
     v
  [Codegen] ----> Machine Code / PTX / SPIR-V
```

See [pipeline.md](pipeline.md) for detailed documentation of each stage.

## Feature Flags

The compiler supports several feature flags for optional functionality:

| Flag | Description |
|------|-------------|
| `jit` | Cranelift JIT compilation |
| `llvm` | LLVM backend (requires LLVM 15+) |
| `lsp` | Language Server Protocol |
| `smt` | Z3 refinement type verification |
| `gpu` | GPU codegen (PTX/SPIR-V) |
| `cuda` | CUDA runtime (requires CUDA toolkit) |
| `ontology` | Scientific ontology (15M+ terms) |
| `llm` | LLM integration |
| `pkg` | Package manager |
| `wasm` | WebAssembly target |
| `full` | All features |

## Further Reading

- [Compilation Pipeline](pipeline.md) - Detailed pipeline documentation
- [Frontend](frontend/) - Lexer, parser, AST documentation
- [Middle End](middle/) - Type checking, effects, analysis
- [Backend](backend/) - Code generation documentation
- [Contributing](contributing/) - Building, testing, development workflow
