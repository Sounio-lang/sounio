# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**START HERE**: [docs/MINIMUM_VIABLE_SOUNIO.md](docs/MINIMUM_VIABLE_SOUNIO.md) — What actually works today

**For comprehensive syntax reference (mix of implemented + aspirational)**, see [docs/LLM_PROGRAMMING_GUIDE.md](docs/LLM_PROGRAMMING_GUIDE.md)

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

## Build Commands

```bash
# Build (from compiler/ directory)
cd compiler && cargo build
cargo build --release

# Run tests
cargo test
cargo test test_name                    # specific test
cargo test --test integration_semantic_types
cargo test -- --nocapture               # with output

# Check/run Sounio programs
cargo run -- check examples/hello.sio
cargo run -- check examples/hello.sio --show-ast --show-types
cargo run --features jit -- run examples/hello.sio   # JIT execution

# REPL
cargo run -- repl

# Lint and format
cargo clippy
cargo fmt

# Run benchmarks
cargo bench --bench layout_bench
cargo bench --bench ontology_bench

# Feature flags
cargo build --features jit      # Cranelift JIT
cargo build --features llvm     # LLVM backend (requires LLVM 15+)
cargo build --features lsp      # Language Server
cargo build --features smt      # Z3 refinement types
cargo build --features gpu      # GPU codegen (PTX/SPIR-V)
cargo build --features ontology # Scientific ontology support
cargo build --features pkg      # Package manager
cargo build --features full     # All features

# LSP server
cargo run --features lsp --bin sounio-lsp

# Ontology database builder
cargo run --bin sounio-ontology-build
```

## Environment Configuration

### Stdlib Path Setup

When working outside the repo root, always set the stdlib path:

```bash
export SOUNIO_STDLIB_PATH=/home/demetrios/sounio-1/stdlib
```

Verify stdlib resolution:

```bash
souc sysroot stdlib-paths
```

This shows all search locations the compiler checks and which ones exist.

### Common Issues

**"Import not found" when running programs outside repo:**
- Ensure `SOUNIO_STDLIB_PATH` is set to the stdlib directory
- Run `souc sysroot stdlib-paths` to verify the path
- Check that stdlib modules have proper entry points (`mod.sio` or `lib.sio`)

**Module system not yet implemented:**
- `use` statements don't work (planned for v0.70.0)
- Workaround: Use single-file programs
- See [docs/MINIMUM_VIABLE_SOUNIO.md](docs/MINIMUM_VIABLE_SOUNIO.md) for details

## Compiler Architecture

**Pipeline:** Source → Lexer (Logos) → Parser → AST → Type Checker → HIR → SIR → HLIR (SSA) → Codegen

Key modules in `compiler/src/`:

- `lexer/`, `parser/`, `ast/` — Frontend
- `check/`, `types/`, `typeck/` — Bidirectional type inference
- `effects/` — Algebraic effect system (IO, Mut, Alloc, Panic, Async, GPU, Prob, Div)
- `linear/`, `ownership/` — Linear/affine type checking
- `units/` — Dimensional analysis (mg, mL, h, etc.)
- `refinement/`, `smt/` — Z3-backed refinement types
- `epistemic/` — Confidence and provenance tracking (Knowledge<T>)
- `ontology/` — Scientific ontology integration (15M+ terms)
- `hir/` — Typed high-level IR
- `sir/` — Scientific IR for domain-specific optimizations (ODEs, tensors, autodiff)
- `hlir/` — SSA-based low-level IR
- `backend/` — Native code generation (ELF/Mach-O binary output)
- `codegen/` — LLVM, Cranelift JIT, GPU (PTX/SPIR-V) backends
- `interp/` — Interpreter
- `lsp/` — Language Server Protocol
- `pkg/` — Package manager
- `repl/` — Interactive REPL

**See also:** [compiler/docs/KNOWN_LIMITATIONS.md](compiler/docs/KNOWN_LIMITATIONS.md) for current language limitations

## Sounio Language Syntax (NOT Rust)

**CRITICAL SYNTAX DIFFERENCES:**

```sio
// Variables
let x = 5              // immutable
var y = 10             // mutable

// References: Sounio uses &! for mutable, NOT &mut
&T                     // shared reference
&!T                    // exclusive/mutable reference (NOT &mut!)

// Functions with effects
fn read_file(path: string) -> string with IO { ... }

// Linear types
linear struct FileHandle { fd: i32 }

// Units of measure
let dose: mg = 500.0
let conc: mg/L = dose / volume

// Array/slice operations (Darwin Atlas syntax)
let head = arr[..k]    // first k elements
let tail = arr[k..]    // from k to end
let combined = a ++ b  // concatenation

// GPU kernels
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32]) {
    let i = gpu.thread_id.x
    c[i] = a[i] + b[i]
}

// Refinement types
type Positive = { x: i32 | x > 0 }

// Epistemic types
let measurement: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
```

**What does NOT work in Sounio:**
- `&mut` - use `&!` instead
- `assert!()`, `println!()` - no Rust macros
- `#[test]`, `#[derive()]` - no attribute macros
- `let (a, b) = tuple` - no tuple destructuring
- `|(x, y)| expr` - no tuple destructuring in closures
- `pub` - visibility modifiers not yet implemented
- Forward references - functions must be defined before use (helpers go at top)

## Test Organization

- `compiler/tests/` — Integration tests (Rust)
  - `integration_semantic_types.rs` — Core type system tests
  - `integration_ontology_e2e.rs` — Scientific ontology integration
  - `epistemic_integration.rs` — Knowledge<T> and uncertainty
  - `jit_effects.rs` — Effect system with JIT
  - `gpu_*.rs` — GPU codegen and execution tests
  - `native_*.rs` — Native backend tests
- `tests/ui/` — Error message verification
- `tests/run-pass/` — Should compile and run
- `tests/compile-fail/` — Should fail to compile

Test annotations in Sounio files:
```sio
//@ run-pass
//@ compile-fail
//@ error-pattern: <text>
//@ ignore             — skip this test
```

## Coding Standards

- Use `thiserror` for error types, `miette` for diagnostics with source spans
- No `unwrap()` in library code—use `?` or proper error handling
- All public items need doc comments
- Use `logos` for lexer patterns

## Commit Format

```
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, backend, cli, docs, stdlib, tests, ontology, epistemic,
           lsp, pkg, sir, units, refinement
```

## LLM Offload

Use `llm-offload` for bulk generation to save Anthropic tokens:
- `llm-offload -t expand -p local` - Expand outline (free, local Mistral)
- `llm-offload -t paraphrase -p grok` - Rewrite text (Grok)
- `llm-offload -t scaffold -p local` - Code boilerplate
- `llm-offload -t variations -p minimax` - Generate alternatives
- `llm-offload --list-templates` - See all templates

**Workflow**: Claude designs → llm-offload expands → Claude critiques

