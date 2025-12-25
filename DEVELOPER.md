# DEVELOPER.md

This file provides guidance when working with code in this repository.

## Project Identity

**Sounio** is a novel L0 systems + scientific programming language created by Demetrios Chiuratto Agourakis. This is NOT a dialect of Rust, Julia, or any existing language. Sounio has its own syntax, semantics, and design philosophy.

## Build Commands

```bash
# Build (from repo root or compiler/)
cd compiler && cargo build

# Run all tests
cargo test

# Run specific test
cargo test test_name
cargo test --test integration_semantic_types
cargo test --test integration_ontology_e2e

# Run with output
cargo test -- --nocapture

# Lint and format
cargo clippy
cargo fmt

# Check a Sounio file
cargo run -- check examples/hello.d
cargo run -- check examples/hello.d --show-ast --show-types

# Run with JIT (requires --features jit)
cargo run --features jit -- run examples/hello.d

# Run benchmarks
cargo bench --bench layout_bench
cargo bench --bench locality_bench
cargo bench --bench ontology_bench

# Build with features
cargo build --features jit           # Cranelift JIT
cargo build --features llvm          # LLVM backend (requires LLVM)
cargo build --features lsp           # Language Server
cargo build --features smt           # Z3 refinement types
cargo build --features gpu           # GPU codegen
cargo build --features ontology      # Ontology support
cargo build --features full          # All features

# Run LSP server (requires --features lsp)
cargo run --features lsp --bin sounio-lsp

# Build ontology database
cargo run --bin dc-ontology-build
```

## Repository Structure

```
sounio/
├── compiler/           # Rust compiler (main codebase)
│   ├── Cargo.toml      # Dependencies and features
│   ├── src/
│   │   ├── main.rs     # CLI entry point (dc)
│   │   ├── lib.rs      # Library root
│   │   ├── lexer/      # Tokenization (Logos)
│   │   ├── parser/     # Recursive descent + Pratt parsing
│   │   ├── ast/        # Abstract syntax tree
│   │   ├── check/      # Type checking
│   │   ├── typeck/     # Additional type checking
│   │   ├── types/      # Type system core
│   │   ├── effects/    # Algebraic effect system
│   │   ├── hir/        # High-level IR (typed AST)
│   │   ├── hlir/       # SSA-based low-level IR
│   │   ├── codegen/    # LLVM/Cranelift/GPU backends
│   │   ├── interp/     # Interpreter
│   │   ├── ontology/   # Scientific ontology (15M+ terms)
│   │   ├── epistemic/  # Knowledge types and confidence
│   │   ├── smt/        # Z3 SMT solver integration
│   │   ├── refinement/ # Refinement type verification
│   │   ├── linear/     # Linear/affine type checking
│   │   ├── ownership/  # Ownership analysis
│   │   ├── units/      # Units of measure
│   │   ├── lsp/        # Language Server Protocol
│   │   ├── locality/   # Cache optimization
│   │   ├── layout/     # Memory layout synthesis
│   │   ├── optimizer/  # Optimizations
│   │   ├── fmt/        # Code formatter
│   │   └── ...
│   ├── tests/          # Integration tests
│   └── benches/        # Performance benchmarks
├── stdlib/             # Standard library (Sounio code)
├── spec/               # Language specification
├── docs/               # Documentation
├── examples/           # Example programs
├── editors/            # IDE integrations (VS Code)
└── tests/              # Language test suite
    ├── compile-fail/   # Should fail to compile
    ├── run-pass/       # Should compile and run
    └── ui/             # Error message tests
```

## Language Syntax Quick Reference

```d
// Variables
let x = 5              // immutable
var y = 10             // mutable
const PI = 3.14159     // compile-time constant

// Functions with effects
fn read_file(path: string) -> string with IO, Panic { ... }
fn simulate() -> f64 with Prob, Alloc { ... }

// References (NOT &mut like Rust)
&T                     // shared reference
&!T                    // exclusive reference

// Linear/affine types
linear struct FileHandle { fd: i32 }
affine struct Buffer { ptr: *u8 }

// GPU kernels
kernel fn vector_add(a: &[f32], b: &[f32], c: &!mut [f32]) {
    let i = gpu.thread_id.x
    c[i] = a[i] + b[i]
}

// Units of measure
let dose: mg = 500.0
let volume: mL = 10.0
let conc: mg/mL = dose / volume

// Refinement types
type Positive = { x: i32 | x > 0 }
type Percentage = { x: f64 | 0.0 <= x && x <= 100.0 }
```

## Built-in Effects

`IO`, `Mut`, `Alloc`, `Panic`, `Async`, `GPU`, `Prob`, `Div`

## Compiler Pipeline

Source -> Lexer (Logos) -> Parser -> AST -> Type Checker -> HIR -> HLIR (SSA) -> MLIR -> Codegen (LLVM/Cranelift/GPU)

## Coding Standards

### Rust Code (Compiler)
- Use `thiserror` for error types, `miette` for diagnostics with source spans
- No `unwrap()` in library code—use `?` or proper error handling
- All public items need doc comments
- Use `logos` for lexing patterns

### Commit Messages
```
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, ontology, epistemic
```

## Key Architectural Decisions

1. **Effects are first-class** — every function has an effect signature
2. **Linear types matter** — track resource usage carefully in `linear/` and `ownership/`
3. **Ontology-aware types** — 15M+ scientific terms as first-class types via `ontology/`
4. **Epistemic computing** — confidence and provenance tracking in `epistemic/`
5. **Bidirectional type inference** — types flow both up and down the AST
