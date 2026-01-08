# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**For comprehensive Sounio syntax and programming patterns, see [docs/LLM_PROGRAMMING_GUIDE.md](docs/LLM_PROGRAMMING_GUIDE.md)**

## Project Identity

**Sounio** is a novel L0 systems + scientific programming language for epistemic computing. This is NOT a dialect of Rust, Julia, or any existing language—Sounio has its own syntax, semantics, and design philosophy.

## Working Principles (MANDATORY)

1. **No AI attribution** — Never add "Co-Authored-By", "Generated with", or similar footers to commits
2. **Sounio Native syntax** — Write `.sio` files in Sounio's native idioms (`&!` not `&mut`, `var` not `let mut`), never Rust-like patterns
3. **Atomic commits** — One logical change per commit, focused and reviewable
4. **Token efficiency** — Use parallel agents, concise operations, minimize redundant work
5. **YOLO mode** — Execute routine operations without asking; move fast
6. **Q1+ research first** — Deep literature review (SOTA, peer-reviewed, Q1+ journals) before major architectural decisions
7. **No drift to mean** — Excellence only; reject mediocre or "good enough" solutions
8. **Epistemic honesty** — Be rigorous, cite sources, acknowledge uncertainty, no hallucinated claims
9. **Edge of novelty** — We are building something genuinely new; do not copy existing languages or settle for conventional approaches

**Core philosophy:** Every value carries not just data, but knowledge of its own uncertainty. Sounio makes uncertainty a first-class citizen with automatic propagation and confidence-based control flow.

**Binary name:** `souc` (Sounio compiler)
**File extensions:** `.sio` (primary), `.d` (legacy support)

## Build Commands

```bash
# Build (from repo root or compiler/ directory)
cd compiler && cargo build
cargo build --release

# Run tests
cargo test
cargo test test_name                    # specific test
cargo test --test integration_semantic_types
cargo test --test integration_ontology_e2e
cargo test -- --nocapture               # with output

# Check/run Sounio programs
cargo run -- check examples/hello.sio
cargo run -- check examples/hello.sio --show-ast --show-types
cargo run -- run examples/hello.sio     # Interpreter
cargo run --features jit -- run examples/hello.sio   # JIT execution
cargo run --features llvm -- build examples/hello.sio  # Native compile

# REPL
cargo run -- repl

# Lint and format
cargo clippy
cargo fmt

# Run benchmarks
cargo bench --bench layout_bench
cargo bench --bench locality_bench
cargo bench --bench ontology_bench
cargo bench --bench compiler_bench
cargo bench --bench sir_gpu_bench

# LSP server (requires --features lsp)
cargo run --features lsp --bin sounio-lsp

# Build ontology database
cargo run --bin sounio-ontology-build

# Feature flags
cargo build --features jit      # Cranelift JIT
cargo build --features llvm     # LLVM backend (requires LLVM 15+)
cargo build --features lsp      # Language Server
cargo build --features smt      # Z3 refinement types
cargo build --features gpu      # GPU codegen (PTX/SPIR-V)
cargo build --features cuda     # CUDA runtime (requires CUDA toolkit)
cargo build --features ontology # Scientific ontology (15M+ terms)
cargo build --features llm      # LLM integration
cargo build --features pkg      # Package manager
cargo build --features wasm     # WASM target for browser
cargo build --features full     # All features
```

## Compiler Architecture

**Pipeline:** Source → Lexer (Logos) → Parser → AST → Type Checker → HIR → HLIR (SSA) → Codegen

**Key modules in `compiler/src/`:**
- `lexer/`, `parser/`, `ast/` — Frontend (Logos-based tokenization, recursive descent + Pratt parsing)
- `check/`, `types/`, `typeck/` — Bidirectional type inference
- `effects/` — Algebraic effect system (IO, Mut, Alloc, Panic, Async, GPU, Prob, Div)
- `linear/`, `ownership/` — Linear/affine type checking for resource safety
- `units/` — Dimensional analysis (mg, mL, h, etc.) with compile-time checking
- `refinement/`, `smt/` — Z3-backed refinement type verification
- `epistemic/` — Knowledge<T> types with confidence and provenance tracking
- `ontology/` — Scientific ontology integration (15M+ terms from OWL/RDF)
- `hir/` — Typed high-level IR
- `hlir/` — SSA-based low-level IR
- `sir/` — Scientific IR for domain-specific optimizations
- `codegen/` — LLVM, Cranelift JIT, GPU (PTX/SPIR-V) backends
- `interp/` — Tree-walking interpreter
- `lsp/` — Language Server Protocol
- `pkg/` — Package manager
- `repl/` — Interactive REPL
- `layout/`, `locality/` — Cache-aware memory layout optimization
- `analyze/` — Program analysis
- `fmt/` — Code formatter

**Standard library (`stdlib/`):** 151,000+ lines of Sounio code organized by domain (epistemic, medlang, fmri, causal, connectivity, gpu, etc.)

## Sounio Language Syntax (NOT Rust)

**CRITICAL SYNTAX DIFFERENCES:**

```sio
// Variables
let x = 5              // immutable
var y = 10             // mutable
const PI = 3.14159     // compile-time constant

// References: Sounio uses &! for mutable, NOT &mut
&T                     // shared reference
&!T                    // exclusive/mutable reference (NOT &mut!)

// Functions with effects
fn read_file(path: string) -> string with IO, Panic { ... }
fn simulate() -> f64 with Prob, Alloc { ... }

// Linear/affine types
linear struct FileHandle { fd: i32 }
affine struct Buffer { ptr: *u8 }

// Units of measure (compile-time dimensional analysis)
let dose: mg = 500.0
let volume: mL = 10.0
let conc: mg/mL = dose / volume

// Epistemic types (core to Sounio's philosophy)
let measurement = Knowledge::new(
    value: 42.0,
    uncertainty: 0.5,
    confidence: 0.95,
    source: "laboratory"
)

// Confidence gates
if measurement.confidence > 0.95 {
    proceed(measurement)
} else {
    require_confirmation(measurement)
}

// Array/slice operations
let head = arr[..k]    // first k elements
let tail = arr[k..]    // from k to end
let combined = a ++ b  // concatenation

// GPU kernels
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32]) {
    let i = gpu.thread_id.x
    c[i] = a[i] + b[i]
}

// Refinement types (Z3-verified)
type Positive = { x: i32 | x > 0 }
type Percentage = { x: f64 | 0.0 <= x && x <= 100.0 }
```

**What does NOT work in Sounio:**
- `&mut` — use `&!` instead
- `assert!()`, `println!()` — no Rust macros
- `#[test]`, `#[derive()]` — no attribute macros
- `let (a, b) = tuple` — no tuple destructuring
- `|(x, y)| expr` — no tuple destructuring in closures

## Test Organization

**Rust integration tests:**
- `compiler/tests/` — Rust-based integration tests
- `compiler/benches/` — Performance benchmarks

**Sounio test suites:**
- `tests/run-pass/` — Should compile and run successfully
- `tests/compile-fail/` — Should fail to compile (type errors, etc.)
- `tests/ui/` — Error message verification
- `tests/ffi/` — Foreign function interface tests

**Test annotations in Sounio files:**
```sio
//@ run-pass
//@ compile-fail
//@ error-pattern: <text>
```

**Running specific tests:**
```bash
cargo test test_name
cargo test --test integration_semantic_types
cargo test --test integration_ontology_e2e
cargo test -- --nocapture  # show output
```

## Coding Standards

**Rust (compiler code):**
- Use `thiserror` for error types, `miette` for diagnostics with source spans
- No `unwrap()` in library code—use `?` or proper error handling
- All public items need doc comments with `///`
- Use `logos` for lexing patterns
- Run `cargo fmt` and `cargo clippy` before committing

**Sounio (stdlib code):**
- Use `Knowledge<T>` for values with uncertainty
- Include doc comments for public functions
- Propagate uncertainty automatically through computations
- Use appropriate effect annotations (`with IO`, `with Prob`, etc.)

## Commit Format

```
[component] Brief description

Components: lexer, parser, ast, check, types, typeck, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, ontology, epistemic, lsp,
           medlang, fmri, causal, gpu, sir, units, refinement, pkg

Examples:
  [parser] Add support for Knowledge<T> generic syntax
  [stdlib] Implement bootstrap_correlation in connectivity module
  [docs] Update README with MedLang integration examples
  [epistemic] Fix uncertainty propagation in division
```
