---
title: "Architecture"
description: "Deep dive into Sounio's compiler architecture, type system, effect system, GPU codegen, epistemic types, and ontology integration."
---

# Architecture & Technical Design

Explore the internal workings of Sounio's compiler, runtime, and type system.

## Compiler Pipeline

The Sounio compiler implements an 8-stage pipeline from source code to machine code:

1. **Lexer** - Tokenization using the Logos library
2. **Parser** - Recursive descent parser with 5,656 lines of Rust code
3. **AST** - Untyped abstract syntax tree
4. **Type Checker** - Bidirectional inference with 7,153 lines of code
5. **HIR** - High-level IR with effect tracking
6. **HLIR** - SSA-based low-level IR
7. **SIR** - Scientific IR with domain-specific optimizations
8. **Codegen** - Multi-target code generation (LLVM, Cranelift, Native, PTX, Metal)

## Key Components

- **[Overview](/architecture/overview/)** - High-level architecture and design decisions
- **[Compiler Pipeline](/architecture/compiler-pipeline/)** - Stage-by-stage walkthrough
- **[Type System](/architecture/type-system/)** - Bidirectional inference, linear types, units
- **[Effect System](/architecture/effect-system/)** - 10 effects with row polymorphism
- **[GPU Codegen](/architecture/gpu-codegen/)** - PTX, Metal, and SPIR-V backends
- **[Epistemic Types](/architecture/epistemic-types/)** - Knowledge<T> implementation
- **[Ontology Integration](/architecture/ontology-integration/)** - 15M+ scientific terms

## Technical Diagrams

Visual representations of key systems:

- **[Compiler Pipeline](/diagrams/compiler-pipeline.svg)** - 8-stage pipeline flow
- **[Hypercomplex Hierarchy](/diagrams/hypercomplex-hierarchy.svg)** - Cayley-Dickson construction
- **[GPU Flow](/diagrams/gpu-flow.svg)** - Kernel compilation process
- **[Uncertainty Propagation](/diagrams/uncertainty-propagation.svg)** - Knowledge<T> workflow
- **[Moufang Validation](/diagrams/moufang-validation.svg)** - Identity verification

## Design Philosophy

Sounio's architecture prioritizes:

- **Epistemic correctness** - Uncertainty quantification at every stage
- **Scientific accuracy** - GUM-compliant uncertainty propagation
- **Performance** - GPU acceleration and multi-target codegen
- **Type safety** - Linear types and effect tracking
- **Reproducibility** - Provenance tracking for all computations

## Technical Specifications

| Component | Lines of Code | Purpose |
|-----------|--------------|---------|
| Parser | 5,656 | Recursive descent parsing |
| Type Checker | 7,153 | Bidirectional inference |
| LLVM Codegen | ~60K | Production compilation |
| Cranelift JIT | ~25K | Fast development iteration |
| GPU Backend (PTX) | ~360K | NVIDIA GPU support |
| GPU Backend (Metal) | ~174K | Apple Silicon support |
| Total | 145K+ | Full compiler stack |