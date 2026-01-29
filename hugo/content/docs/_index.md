---
title: "Documentation"
description: "Comprehensive documentation for the Sounio programming language"
weight: 1
---

# Sounio Documentation

Welcome to the Sounio documentation. Sounio is a systems programming language for epistemic computing with native uncertainty quantification, GPU acceleration, and scientific standards compliance.

## Quick Start

- **[Getting Started](/docs/getting-started/)** — Install Sounio and write your first program
- **[Minimum Viable Sounio](/docs/minimum-viable/)** — What works today (implementation status)

## Language Reference

- **[Language Guide](/docs/language/)** — Complete syntax and semantics reference
- **[Standard Library](/docs/stdlib/)** — API documentation for 49 stdlib modules
- **[API Reference](/docs/api/)** — Compiler and runtime APIs

## Core Concepts

### Epistemic Computing

Sounio's `Knowledge<T>` type tracks measurement uncertainty through computations:

```sio
let dose: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
let volume: Knowledge<L> = measure(2.0, uncertainty: 0.1)
let concentration = dose / volume  // Uncertainty propagates automatically
```

### Effects System

All side effects are tracked in the type system:

```sio
fn read_file(path: string) -> string with IO { ... }
fn mutate_array(arr: &![i32]) with Mut { arr[0] = 42 }
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32]) with GPU { ... }
```

### Units of Measure

Type-safe dimensional analysis prevents unit errors at compile time:

```sio
let mass: mg = 500.0
let volume: L = 2.0
let concentration: mg/L = mass / volume  // Type-safe division
```

## Architecture Deep Dives

For compiler internals and implementation details:

- **[Architecture Overview](/architecture/overview/)** — High-level system design
- **[Compiler Pipeline](/architecture/compiler-pipeline/)** — 8-stage compilation process
- **[Type System](/architecture/type-system/)** — Bidirectional inference, linear types
- **[Effect System](/architecture/effect-system/)** — Algebraic effects with handlers
- **[GPU Codegen](/architecture/gpu-codegen/)** — PTX, Metal, SPIR-V backends
- **[Epistemic Types](/architecture/epistemic-types/)** — `Knowledge<T>` implementation
- **[Ontology Integration](/architecture/ontology-integration/)** — 15M+ scientific terms

## Domain Applications

- **[Pharmaceutical Sciences](/showcases/pharma/)** — PK/PD modeling with GUM compliance
- **[Quantum Chemistry](/showcases/quantum/)** — VQE algorithms with octonion algebra
- **[Climate Modeling](/showcases/climate/)** — Multi-model ensembles with uncertainty
- **[Financial Risk](/showcases/finance/)** — GPU-accelerated VaR calculations

## Validation & Testing

- **[Test Report](/validation/test-report/)** — 487 tests, 87% coverage
- **[Moufang Validation](/validation/moufang-tests/)** — 7 mathematical identity proofs

## External Resources

- **[GitHub Repository](https://github.com/sounio-lang/sounio)** — Source code and issues
- **[Interactive Playground](https://sounio-lang.github.io/playground)** — Try Sounio in your browser
- **[Technical Report (PDF)](/papers/technical-report.pdf)** — Academic preprint with 25+ citations
