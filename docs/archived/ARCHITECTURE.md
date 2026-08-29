<!-- docs:meta
topic_id: repo.docs.archived.architecture
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.architecture
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

> Status note: this page describes an older Rust-first architecture map. For the
> current contributor-facing compiler map, start with
> [COMPILER_ARCHITECTURE_OVERVIEW.md](COMPILER_ARCHITECTURE_OVERVIEW.md).

# Sounio Compiler Architecture

High-level overview of the Sounio compiler design for contributors.

## Table of Contents

1. [Overview](#overview)
2. [Compilation Pipeline](#compilation-pipeline)
3. [Module Organization](#module-organization)
4. [Key Components](#key-components)
5. [Intermediate Representations](#intermediate-representations)
6. [Type System](#type-system)
7. [Effect System](#effect-system)
8. [Code Generation](#code-generation)
9. [Contributing Guide](#contributing-guide)

---

## Overview

The Sounio compiler (`souc`) is a multi-stage ahead-of-time compiler with optional JIT support. It compiles `.sio` source files to native executables, with experimental support for LLVM IR and WebAssembly.

### Design Principles

1. **Correctness First** - Type safety, memory safety, epistemic integrity
2. **Explicit Over Implicit** - Effects, mutability, uncertainty all visible
3. **Zero-Cost Abstractions** - High-level features with no runtime overhead
4. **Scientific Computing Focus** - Optimized for uncertainty propagation and domain-specific operations

### Technology Stack

- **Language**: Rust (edition 2024)
- **Lexer**: Logos (fast lexer generator)
- **Parser**: Recursive descent with operator precedence
- **Type Inference**: Bidirectional (synthesis + checking)
- **Codegen**: Cranelift (JIT/AOT), Native (ELF/Mach-O), LLVM (experimental)
- **Dependencies**: historical Rust dependencies lived in component-local Cargo manifests; current public/compiler orientation starts with `self-hosted/` and the checked artifacts under `artifacts/omega/`

---

## Compilation Pipeline

```
Source (.sio)
    ↓
┌───────────────┐
│    Lexer      │  → Tokens
│ (lexer/)      │
└───────────────┘
    ↓
┌───────────────┐
│    Parser     │  → AST (Abstract Syntax Tree)
│ (parser/)     │
└───────────────┘
    ↓
┌───────────────┐
│ Name Resolver │  → Resolved names, imports
│ (resolve/)    │
└───────────────┘
    ↓
┌───────────────┐
│ Type Checker  │  → Typed AST
│ (check/)      │  → Type errors
│ (types/)      │
└───────────────┘
    ↓
┌───────────────┐
│ Effect Check  │  → Effect-checked AST
│ (effects/)    │
└───────────────┘
    ↓
┌───────────────┐
│ HIR Lowering  │  → High-Level IR
│ (hir/)        │
└───────────────┘
    ↓
┌───────────────┐
│ HLIR          │  → Polyhedral analysis
│ (hlir/)       │  → Loop optimization
└───────────────┘
    ↓
┌───────────────┐
│ SIR           │  → Domain-specific IR
│ (sir/)        │  → ODE, tensor, GPU ops
└───────────────┘
    ↓
┌───────────────┐
│ MIR           │  → SSA form
│ (mir/)        │  → Optimization passes
└───────────────┘
    ↓
┌───────────────┐
│ Codegen       │  → Native code
│ (codegen/)    │  → ELF/Mach-O/WASM
│ (backend/)    │
└───────────────┘
```

---

## Module Organization

The compiler is organized as a Rust workspace:

```
sounio/
├── Cargo.toml              # Workspace manifest
├── crates/
│   ├── souc/              # Main compiler
│   │   ├── src/
│   │   │   ├── lexer/     # Tokenization
│   │   │   ├── parser/    # Syntax analysis
│   │   │   ├── ast/       # AST definition
│   │   │   ├── resolve/   # Name resolution
│   │   │   ├── check/     # Type checking
│   │   │   ├── types/     # Type system
│   │   │   ├── effects/   # Effect system
│   │   │   ├── hir/       # High-level IR
│   │   │   ├── hlir/      # Polyhedral IR
│   │   │   ├── sir/       # Scientific IR
│   │   │   ├── mir/       # Mid-level IR
│   │   │   ├── codegen/   # Code generation
│   │   │   └── backend/   # Native backends
│   │   └── tests/         # (moved to workspace tests/)
│   └── runtime/           # Runtime library
├── stdlib/                 # Standard library
├── tests/                  # Integration tests
├── benches/                # Benchmarks
└── docs/                   # Documentation
```

---

## Key Components

### Lexer ([src/lexer/](../../crates/souc/src/lexer/))

**Purpose**: Convert source text into tokens

**Implementation**: Logos-based lexer generator

**Key Files**:
- `mod.rs` - Token definitions
- `keywords.rs` - Keyword handling

**Example**:
```rust
// Input: "let x = 42"
// Output: [LET, IDENT("x"), EQ, INT_LIT(42)]
```

### Parser ([src/parser/](../../crates/souc/src/parser/))

**Purpose**: Build Abstract Syntax Tree from tokens

**Implementation**: Recursive descent with Pratt parsing for expressions

**Key Files**:
- `mod.rs` - Parser driver
- `expr.rs` - Expression parsing
- `stmt.rs` - Statement parsing
- `item.rs` - Top-level items (functions, structs, etc.)

**Error Recovery**: Synchronization on statement boundaries

### Type Checker ([src/check/](../../crates/souc/src/check/), [src/types/](../../crates/souc/src/types/))

**Purpose**: Verify type correctness, infer types

**Algorithm**: Bidirectional type inference
- **Synthesis**: Infer type from expression
- **Checking**: Verify expression matches expected type

**Key Features**:
- Generic type parameters
- Type unification
- Subtyping for Knowledge<T>
- Units of measure checking
- Refinement types (optional, requires SMT)

**Key Files**:
- `check/mod.rs` - Type checking driver
- `types/inference.rs` - Type inference
- `types/unification.rs` - Unification algorithm

### Effect System ([src/effects/](../../crates/souc/src/effects/))

**Purpose**: Track computational side effects

**Implementation**: Algebraic effects with handlers

**Built-in Effects**:
- `IO` - Input/output
- `Mut` - Mutable state
- `Alloc` - Memory allocation
- `Panic` - Can panic/error
- `Async` - Asynchronous operations
- `GPU` - GPU execution
- `Prob` - Probabilistic operations
- `Div` - Divergence (non-termination)

**Key Files**:
- `mod.rs` - Effect definitions
- `inference.rs` - Effect inference
- `handlers.rs` - Effect handlers

---

## Intermediate Representations

### HIR (High-Level IR)

**Purpose**: First IR after type checking, close to source

**Features**:
- Still has high-level constructs
- Fully typed
- Async lowering applied

### HLIR (Higher-Level IR)

**Purpose**: Polyhedral optimization

**Features**:
- Loop nest analysis
- Affine transformations
- Locality optimization

### SIR (Scientific IR)

**Purpose**: Domain-specific operations

**Specialized Constructs**:
- ODE systems
- Tensor operations
- Automatic differentiation
- GPU kernels
- Epistemic operations

**Why SIR?**: Scientific code has patterns not captured by general-purpose IRs

### MIR (Mid-Level IR)

**Purpose**: Optimization and lowering

**Features**:
- SSA form (Static Single Assignment)
- Control flow graph (CFG)
- Dataflow analysis
- Optimization passes:
  - Constant propagation
  - Dead code elimination
  - Loop invariant code motion
  - Inlining

---

## Type System

### Core Types

```sio
// Primitives
i8, i16, i32, i64, i128
u8, u16, u32, u64, u128
f32, f64
bool, char, string

// Compound
[T]           // Array
(T, U, ...)   // Tuple
struct { }    // Struct
enum { }      // Enum

// Epistemic
Knowledge<T>  // Value + uncertainty + provenance

// References
&T            // Shared reference
&!T           // Exclusive (mutable) reference

// Functions
fn(T, U) -> V with Effects

// Generics
Vec<T>
Option<T>
Result<T, E>
```

### Type Inference

**Algorithm**: Constraint-based bidirectional inference

**Steps**:
1. Generate constraints from expressions
2. Unify type variables
3. Check for cycles (occurs check)
4. Substitute unified types

**Example**:
```sio
let x = 42        // Infer: x: i32
let y = x + 1.0   // Error: i32 + f64 mismatch
```

### Units of Measure

**Implementation**: Phantom types + compile-time checking

```rust
struct Meter<T>(T);
struct Second<T>(T);
struct Velocity<T>(T);  // Meter<T> / Second<T>
```

**Verification**: Type-level dimensional analysis

---

## Effect System

### Effect Inference

```sio
fn read_file(path: string) -> string {
    fs.read(path)  // Infers: with IO
}

fn process(data: &! Data) {
    data.value = 42  // Infers: with Mut
}
```

### Effect Handlers

**Mechanism**: Delimited continuations

```sio
effect Log {
    fn log(msg: string) -> ()
}

fn computation() -> i32 with Log {
    do Log.log("starting")
    42
}

// Handle the effect
let result = handle computation() {
    Log.log(msg) => {
        print("[LOG] ", msg)
        resume(())  // Continue computation
    }
}
```

---

## Code Generation

### Backends

1. **Cranelift** (Primary)
   - Fast JIT compilation
   - AOT for production
   - Location: `src/codegen/cranelift/`

2. **Native** (Custom)
   - Direct ELF/Mach-O generation
   - No LLVM dependency
   - Location: `src/backend/native/`

3. **LLVM** (Experimental)
   - Leverage LLVM optimizations
   - Location: `src/codegen/llvm/`

4. **GPU** (Specialized)
   - CUDA PTX generation
   - SPIR-V for Vulkan/Metal
   - Location: `src/codegen/gpu/`

### Code Generation Pipeline

```
MIR
 ↓
Register Allocation
 ↓
Instruction Selection
 ↓
Binary Encoding
 ↓
Linking
 ↓
Executable
```

---

## Contributing Guide

### Getting Started

1. **Clone and build**:
   ```bash
   git clone https://github.com/sounio-lang/sounio.git
   cd sounio
   ./bin/souc info
   ```

2. **Run tests**:
   ```bash
   cargo test --workspace
   ./scripts/fast_gate.sh
   ```

3. **Explore the codebase**:
   - Start with `src/main.rs` - compiler entry point
   - Read `src/lib.rs` - public API
   - Check `tests/` for examples

### Adding a Feature

**Example: Add a new operator**

1. **Lexer**: Add token in `src/lexer/mod.rs`
2. **Parser**: Parse in `src/parser/expr.rs`
3. **AST**: Add to `src/ast/expr.rs`
4. **Type Check**: Handle in `src/check/expr.rs`
5. **HIR**: Lower in `src/hir/lower.rs`
6. **Codegen**: Generate code in `src/codegen/`
7. **Test**: Add test in `tests/`

### Code Review Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No compiler warnings
- [ ] Clippy passes
- [ ] Fast gate passes
- [ ] Example added (if applicable)

### Resources

- **Technical Report**: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- **Known Limitations**: [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)
- **Debugging Guide**: [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md)
- **Style Guide**: [../STYLE_GUIDE.md](../STYLE_GUIDE.md)

---

## Performance Considerations

### Compilation Speed

- **Parallel type checking** (per-module)
- **Incremental compilation** (planned)
- **Fast lexer** (Logos)

### Runtime Performance

- **Zero-cost abstractions** - No overhead for unused features
- **Specialization** - Monomorphization like Rust
- **SIMD** - Vectorization where possible
- **GPU offload** - Automatic for large arrays

---

## Testing Strategy

### Unit Tests
- Inline `#[test]` in Rust modules
- Test individual components

### Integration Tests
- `tests/integration/` - Full compiler pipeline
- `tests/run-pass/` - Programs that should run
- `tests/compile-fail/` - Programs that should fail
- `tests/ui/` - Error message quality

### Fuzzing
- Lexer fuzzing
- Parser fuzzing
- Type checker fuzzing

### Benchmarks
- `benches/compiler/` - Compilation speed
- `benches/runtime/` - Generated code performance

---

## Future Directions

### Short Term (Pre-1.0)
- Complete LSP implementation
- Stabilize LLVM backend
- Package manager (`siopkg`)
- Improve error messages

### Long Term (Post-1.0)
- Incremental compilation
- Parallel compilation
- Formal verification tools
- More domain-specific IRs

---

*For detailed technical information, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).*
