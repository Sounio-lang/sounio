---
title: "API Reference"
description: "Compiler, runtime, and tooling APIs for Sounio"
weight: 4
---

# API Reference

This section covers the Sounio compiler API, runtime services, and tooling interfaces.

## Compiler API

### Command Line Interface

```bash
# Type checking
souc check program.sio
souc check program.sio --show-ast --show-types

# Compilation
souc build program.sio -o program
souc build program.sio --release -o program
souc build program.sio --target wasm32

# JIT execution
souc run program.sio
souc run program.sio --features gpu

# REPL
souc repl

# Package management
souc pkg init
souc pkg add dependency@1.0
souc pkg build

# Formatting
souc fmt program.sio
souc fmt --check .

# Analysis
souc analyze program.sio --metrics
souc analyze program.sio --dead-code
```

### Compiler Flags

| Flag | Description |
|------|-------------|
| `--show-ast` | Print parsed AST |
| `--show-hir` | Print typed HIR |
| `--show-sir` | Print Scientific IR |
| `--show-hlir` | Print SSA HLIR |
| `--show-types` | Print inferred types |
| `--show-effects` | Print effect signatures |
| `--emit-llvm` | Output LLVM IR |
| `--emit-ptx` | Output CUDA PTX |
| `--emit-spirv` | Output SPIR-V |

### Feature Flags

```bash
# Enable at compile time
souc build --features jit      # Cranelift JIT
souc build --features llvm     # LLVM backend
souc build --features gpu      # GPU codegen
souc build --features smt      # Z3 refinement types
souc build --features ontology # Scientific ontology
souc build --features full     # All features
```

---

## Compiler Internals

### Pipeline Stages

The Sounio compiler processes source through 8 stages:

```
Source → Lexer → Parser → AST → TypeCheck → HIR → SIR → HLIR → Codegen
```

| Stage | Module | Output |
|-------|--------|--------|
| 1. Lexing | `lexer/` | Token stream |
| 2. Parsing | `parser/` | Untyped AST |
| 3. Name Resolution | `resolve/` | Resolved AST |
| 4. Type Checking | `check/`, `typeck/` | Typed HIR |
| 5. Effect Inference | `effects/` | Effect-annotated HIR |
| 6. SIR Lowering | `sir/` | Scientific IR |
| 7. HLIR Lowering | `hlir/` | SSA form |
| 8. Codegen | `codegen/`, `backend/` | Native/LLVM/GPU |

### Type System API

```rust
// compiler/src/types/mod.rs

/// Core type representation
pub enum Type {
    Primitive(PrimitiveType),
    Array(Box<Type>, usize),
    Slice(Box<Type>),
    Struct(StructDef),
    Function(FunctionType),
    Knowledge(Box<Type>),  // Epistemic wrapper
    Unit(Box<Type>, UnitExpr),  // Dimensional type
    Refinement(Box<Type>, Predicate),  // Refinement type
    Linear(Box<Type>),  // Linear/affine type
}

/// Type checking context
pub struct TypeContext {
    pub bindings: HashMap<Symbol, Type>,
    pub effects: EffectSet,
    pub constraints: Vec<Constraint>,
}

/// Bidirectional type inference
pub fn infer(ctx: &mut TypeContext, expr: &Expr) -> Result<Type>;
pub fn check(ctx: &mut TypeContext, expr: &Expr, expected: &Type) -> Result<()>;
```

### Effect System API

```rust
// compiler/src/effects/mod.rs

/// Effect types
pub enum Effect {
    IO,      // File/network I/O
    Mut,     // Mutation
    Alloc,   // Heap allocation
    Panic,   // Can panic
    Async,   // Asynchronous
    GPU,     // GPU computation
    Prob,    // Probabilistic
    Div,     // Can diverge
}

/// Effect set (algebraic effects)
pub struct EffectSet(HashSet<Effect>);

impl EffectSet {
    pub fn pure() -> Self;  // Empty set
    pub fn union(&self, other: &Self) -> Self;
    pub fn subsumes(&self, other: &Self) -> bool;
}

/// Effect inference
pub fn infer_effects(func: &Function) -> EffectSet;
```

### Epistemic Type API

```rust
// compiler/src/epistemic/mod.rs

/// Knowledge type metadata
pub struct KnowledgeInfo {
    pub inner_type: Type,
    pub confidence_level: f64,
    pub provenance: Option<ProvenanceChain>,
}

/// GUM-compliant uncertainty propagation
pub fn propagate_uncertainty(
    op: BinaryOp,
    lhs: &KnowledgeInfo,
    rhs: &KnowledgeInfo,
) -> KnowledgeInfo;

/// Confidence interval computation
pub fn confidence_interval(
    value: f64,
    std_uncertainty: f64,
    coverage: f64,
) -> (f64, f64);
```

---

## Runtime API

### Memory Management

```sio
// Allocation
let ptr = alloc<T>(count)        // Heap allocation
let stack_arr = [0; 1024]        // Stack allocation
defer free(ptr)                  // Deferred cleanup

// Linear types (compile-time enforced)
linear struct Resource { handle: i32 }
fn consume(r: Resource) { ... }  // Takes ownership
fn borrow(r: &Resource) { ... }  // Borrows
```

### Effect Handlers

```sio
// Define custom effect
effect Logger {
    fn log(msg: string) -> ()
}

// Handler implementation
handler ConsoleLogger for Logger {
    fn log(msg: string) -> () {
        print("[LOG] ", msg)
    }
}

// Use with handler
fn main() with IO {
    with ConsoleLogger {
        do_work()  // log() calls handled by ConsoleLogger
    }
}
```

### GPU Runtime

```sio
// Device query
let devices = gpu.list_devices()
let device = gpu.select_device(0)
print("Using: ", device.name())
print("Memory: ", device.memory_gb(), " GB")
print("Compute: ", device.compute_capability())

// Memory management
let d_data = gpu.alloc<f32>(n)
gpu.copy_to_device(d_data, h_data)
// ... kernel execution ...
gpu.copy_to_host(h_result, d_result)
gpu.free(d_data)

// Synchronization
gpu.sync()
gpu.sync_stream(stream)
```

---

## Language Server Protocol

Sounio provides an LSP server for IDE integration.

### Starting the Server

```bash
# Start LSP server
souc lsp

# With logging
souc lsp --log-level debug --log-file /tmp/sounio-lsp.log
```

### Supported Features

| Feature | Status |
|---------|--------|
| Hover | ✅ Full |
| Go to Definition | ✅ Full |
| Find References | ✅ Full |
| Completion | ✅ Full |
| Signature Help | ✅ Full |
| Diagnostics | ✅ Full |
| Code Actions | 🔶 Partial |
| Rename | 🔶 Partial |
| Formatting | ✅ Full |
| Semantic Tokens | ✅ Full |

### VS Code Extension

```json
// .vscode/settings.json
{
    "sounio.lsp.path": "/usr/local/bin/souc",
    "sounio.lsp.args": ["lsp"],
    "sounio.trace.server": "verbose"
}
```

---

## Package Manager API

### Package Manifest

```toml
# sounio.toml
[package]
name = "my-project"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
license = "MIT"
edition = "2024"

[dependencies]
std-scientific = "1.0"
gpu-primitives = "0.5"

[dev-dependencies]
test-framework = "2.0"

[features]
default = ["gpu"]
gpu = ["gpu-primitives"]
```

### CLI Commands

```bash
# Initialize project
souc pkg init my-project

# Add dependencies
souc pkg add std-scientific@1.0
souc pkg add --dev test-framework

# Build and run
souc pkg build
souc pkg run

# Testing
souc pkg test
souc pkg test --filter "unit_*"

# Publishing
souc pkg publish --registry sounio-registry.org
```

---

## Ontology API

### Querying Scientific Terms

```sio
use std::ontology::*

fn main() with IO {
    // Look up a term
    let term = ontology.lookup("dopamine")
    print("Definition: ", term.definition())
    print("Synonyms: ", term.synonyms())
    print("Parent: ", term.parent())

    // Check relationships
    let is_neurotransmitter = ontology.is_a("dopamine", "neurotransmitter")

    // Fuzzy search
    let matches = ontology.search("seroton*", limit: 10)
}
```

### Supported Ontologies

| Ontology | Terms | Domain |
|----------|-------|--------|
| ChEBI | 170,000+ | Chemical entities |
| GO | 45,000+ | Gene function |
| UBERON | 25,000+ | Anatomy |
| SNOMED-CT | 350,000+ | Clinical terms |
| LOINC | 98,000+ | Lab observations |
| RXNORM | 115,000+ | Drug names |
| **Total** | **15M+** | Multi-domain |

---

## Diagnostic Codes

### Error Categories

| Range | Category |
|-------|----------|
| E0001-E0099 | Syntax errors |
| E0100-E0199 | Type errors |
| E0200-E0299 | Effect errors |
| E0300-E0399 | Linear type errors |
| E0400-E0499 | Unit errors |
| E0500-E0599 | Refinement errors |
| E0600-E0699 | GPU errors |
| E0700-E0799 | Import/module errors |

### Common Errors

```
E0101: Type mismatch
  expected `i32`, found `f64`

E0201: Effect not declared
  function calls `read_file` which requires `IO` effect

E0301: Linear value used twice
  `handle` was already consumed on line 42

E0401: Unit mismatch
  cannot add `kg` and `m`

E0501: Refinement violation
  value -5 does not satisfy predicate `x > 0`
```

### Explaining Errors

```bash
souc explain E0201
```

---

## See Also

- **[Standard Library](/docs/stdlib/)** — Module reference
- **[Language Guide](/docs/language/)** — Syntax and semantics
- **[Architecture](/architecture/)** — Compiler internals
- **[Validation](/validation/)** — Test suite and benchmarks
