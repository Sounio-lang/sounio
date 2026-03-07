<!-- docs:meta
topic_id: repo.docs.feature-flags
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.feature-flags
-->

# Sounio Feature Flags Reference

This document provides a comprehensive reference for all Cargo feature flags available in the Sounio compiler.

## Table of Contents

- [Overview](#overview)
- [Feature Matrix](#feature-matrix)
- [Code Generation Backends](#code-generation-backends)
- [Optional Features](#optional-features)
- [Build Profiles](#build-profiles)
- [Common Configurations](#common-configurations)

---

## Overview

Sounio uses Cargo feature flags to enable optional functionality. This allows users to build only what they need, reducing compile time and binary size.

**Default build** (no features):
```bash
cargo build --release
```

The default build includes the core compiler (lexer, parser, type checker, HIR/SIR/HLIR) but no code generation backends.

---

## Feature Matrix

| Feature | Description | Dependencies | Build Time Impact |
|---------|-------------|--------------|-------------------|
| `jit` | Cranelift JIT compilation | Cranelift crates | +30s |
| `llvm` | LLVM native codegen (default: LLVM 15) | LLVM 15, libzstd | +60s |
| `llvm14` | LLVM 14 backend | LLVM 14, libzstd | +60s |
| `llvm15` | LLVM 15 backend | LLVM 15, libzstd | +60s |
| `llvm16` | LLVM 16 backend | LLVM 16, libzstd | +60s |
| `llvm17` | LLVM 17 backend | LLVM 17, libzstd | +60s |
| `gpu` | GPU codegen (PTX, SPIR-V) | rspirv, spirv | +15s |
| `cuda` | CUDA runtime execution | cudarc, CUDA toolkit | +20s |
| `smt` | Z3 refinement type verification | z3, cmake | +45s |
| `lsp` | Language Server Protocol | tower-lsp, tokio | +25s |
| `ontology` | Scientific ontology (15M+ terms) | rusqlite | +10s |
| `ontology-build` | OWL/RDF parsing for ontology builds | rio_turtle, rio_xml | +10s |
| `pkg` | Package manager with registry | tokio, reqwest, tar | +20s |
| `distributed` | Distributed build system | axum, hyper, tower | +25s |
| `wasm` | WebAssembly target | wasm-bindgen | +15s |
| `llm` | LLM integration | regex | +5s |
| `glm` | GLM-4.7 ML-guided optimization | regex, lazy_static | +5s |
| `full` | All features combined | All dependencies | +180s |

---

## Code Generation Backends

### Cranelift JIT (`jit`)

Fast JIT compilation using Cranelift. Recommended for development and REPL.

```bash
cargo build --features jit
```

**Enables:**
- `cranelift-codegen` - Code generation
- `cranelift-frontend` - IR builder
- `cranelift-jit` - JIT execution
- `cranelift-module` - Module management
- `cranelift-native` - Host target support
- `cranelift-object` - Object file emission
- `target-lexicon` - Target triple parsing

**No external dependencies required.**

### LLVM Backend (`llvm`, `llvm14`-`llvm17`)

Optimized native code generation via LLVM.

```bash
# Default (LLVM 15)
cargo build --features llvm

# Specific version
cargo build --features llvm17
```

**Version selection:**

| Feature | LLVM Version | Environment Variable |
|---------|--------------|----------------------|
| `llvm14` | 14.x | `LLVM_SYS_140_PREFIX` |
| `llvm15` | 15.x | `LLVM_SYS_150_PREFIX` |
| `llvm16` | 16.x | `LLVM_SYS_160_PREFIX` |
| `llvm17` | 17.x | `LLVM_SYS_170_PREFIX` |

**Requirements:**
- LLVM installation (matching version)
- `libzstd-dev` for compression support
- C++ compiler for LLVM bindings

**Features are mutually exclusive** - only enable one LLVM version.

### GPU Backend (`gpu`)

PTX (NVIDIA) and SPIR-V code generation for GPU kernels.

```bash
cargo build --features gpu
```

**Enables:**
- `rspirv` - SPIR-V manipulation
- `spirv` - SPIR-V types

**No runtime required** - generates shader code that can be executed elsewhere.

### CUDA Runtime (`cuda`)

Execute GPU kernels on NVIDIA GPUs.

```bash
cargo build --features "gpu,cuda"
```

**Enables:**
- `cudarc` - CUDA Driver API bindings

**Requirements:**
- CUDA toolkit 11.0+
- NVIDIA GPU with compute capability 6.0+

---

## Optional Features

### SMT Solver (`smt`)

Z3-backed refinement type verification.

```bash
cargo build --features smt
```

**Enables:**
- Compile-time verification of refinement predicates
- Z3 constraint solving

**Requirements:**
- Z3 library 4.8+
- CMake (for z3-sys build)

**Without `smt`**: Refinement types fall back to runtime assertions.

### Language Server (`lsp`)

LSP server for IDE integration.

```bash
cargo build --features lsp

# Run LSP server
cargo run --features lsp --bin sounio-lsp
```

**Enables:**
- `tower-lsp` - LSP protocol implementation
- `tokio` - Async runtime
- `dashmap` - Concurrent hash maps
- `ropey` - Rope data structure for text

### Scientific Ontology (`ontology`)

Access to 15M+ scientific terms from BioPortal, ChEBI, GO, etc.

```bash
cargo build --features ontology
```

**Enables:**
- `rusqlite` - SQLite backend for term storage
- Ontology querying and validation

### Ontology Build Tools (`ontology-build`)

Parse OWL/RDF files to build ontology databases.

```bash
cargo build --features "ontology,ontology-build"
cargo run --bin sounio-ontology-build
```

**Enables:**
- `rio_turtle` - Turtle/N-Triples parsing
- `rio_xml` - RDF/XML parsing
- `rio_api` - RDF API

### Package Manager (`pkg`)

Full package manager with HTTP registry support.

```bash
cargo build --features pkg
```

**Enables:**
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `tar` - Archive handling
- `async-compression` - Gzip compression
- `futures` - Async utilities
- `base64` - Encoding

### Distributed Builds (`distributed`)

Remote compilation and caching infrastructure.

```bash
cargo build --features distributed
```

**Enables:**
- `axum` - Web framework
- `tower` - Service abstractions
- `tower-http` - HTTP utilities
- `hyper` - HTTP implementation

### WebAssembly (`wasm`)

Build for browser-based playground.

```bash
cargo build --target wasm32-unknown-unknown --features wasm
```

**Enables:**
- `wasm-bindgen` - JS interop
- `js-sys` - JS system bindings
- `web-sys` - Web API bindings
- `console_error_panic_hook` - Debug support

### LLM Integration (`llm`, `glm`)

LLM-assisted features for code generation and optimization.

```bash
cargo build --features llm
cargo build --features glm  # GLM-4.7 specific
```

---

## Build Profiles

### Debug Profile

```bash
cargo build
# or
cargo build --profile dev
```

**Settings:**
- `opt-level = 0` - No optimization
- `debug = true` - Full debug info
- Fast compile times

### Release Profile

```bash
cargo build --release
```

**Settings:**
- `lto = true` - Link-time optimization
- `codegen-units = 1` - Single codegen unit for better optimization
- `panic = "abort"` - Smaller binary, no unwinding

---

## Common Configurations

### Minimal (Parsing only)

```bash
cargo build --release
```

No code generation, just parsing and type checking.

### Development

```bash
cargo build --features jit
```

Fast compilation with JIT execution for quick iteration.

### Production

```bash
cargo build --release --features "llvm17,smt,gpu"
```

Optimized native code, refinement verification, GPU support.

### IDE Development

```bash
cargo build --features "jit,lsp"
```

JIT backend plus LSP server for IDE integration.

### Scientific Computing

```bash
cargo build --release --features "llvm17,gpu,ontology,smt"
```

All scientific features: native code, GPU, ontology, refinement types.

### Full Build

```bash
cargo build --release --features full
```

Everything enabled. Requires all dependencies.

### CI/CD Pipeline

```bash
# Quick check (no codegen)
cargo check --lib

# Tests without LLVM
cargo test --features "jit,gpu,ontology"

# Full tests (requires LLVM)
cargo test --features "jit,llvm17,gpu,ontology,smt"
```

---

## Feature Combinations

### Valid Combinations

```bash
# JIT + GPU
--features "jit,gpu"

# LLVM + SMT
--features "llvm17,smt"

# Everything except WASM
--features "jit,llvm17,gpu,cuda,smt,lsp,ontology,pkg"
```

### Invalid Combinations

```bash
# Multiple LLVM versions (ERROR)
--features "llvm15,llvm17"

# CUDA without GPU (no error but useless)
--features "cuda"
```

---

## Environment Variables

| Variable | Feature | Purpose |
|----------|---------|---------|
| `LLVM_SYS_140_PREFIX` | `llvm14` | LLVM 14 installation path |
| `LLVM_SYS_150_PREFIX` | `llvm15` | LLVM 15 installation path |
| `LLVM_SYS_160_PREFIX` | `llvm16` | LLVM 16 installation path |
| `LLVM_SYS_170_PREFIX` | `llvm17` | LLVM 17 installation path |
| `SOUNIO_STDLIB_PATH` | - | Standard library path |
| `CUDA_PATH` | `cuda` | CUDA toolkit path |
| `Z3_SYS_Z3_HEADER` | `smt` | Z3 header path (if non-standard) |

---

## Conditional Compilation

In Rust code, check features with:

```rust
#[cfg(feature = "jit")]
fn run_jit() { ... }

#[cfg(feature = "llvm-base")]  // Any LLVM version
fn run_llvm() { ... }

#[cfg(feature = "gpu")]
fn compile_gpu_kernel() { ... }

#[cfg(feature = "smt")]
fn verify_refinement() { ... }

#[cfg(not(any(feature = "jit", feature = "llvm-base")))]
compile_error!("Enable jit or llvm feature for code generation");
```

---

## Related Documentation

- [Installation Guide](INSTALLATION.md) - Dependency setup
- [LLVM Codegen](LLVM_CODEGEN.md) - LLVM backend details
- [GPU Runtime](GPU_RUNTIME.md) - GPU features
- [Getting Started](getting-started.md) - First steps

---

*Last updated: January 2026 (v1.0.0)*
