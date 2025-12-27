# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**For comprehensive Sounio syntax and programming patterns, see [docs/LLM_PROGRAMMING_GUIDE.md](docs/LLM_PROGRAMMING_GUIDE.md)**

## Project Identity

**Sounio** is a novel L0 systems + scientific programming language. This is NOT a dialect of Rust, Julia, or any existing language—Sounio has its own syntax, semantics, and design philosophy.

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
cargo run -- check examples/hello.d
cargo run -- check examples/hello.d --show-ast --show-types
cargo run --features jit -- run examples/hello.d   # JIT execution

# Lint and format
cargo clippy
cargo fmt

# Feature flags
cargo build --features jit      # Cranelift JIT
cargo build --features llvm     # LLVM backend (requires LLVM)
cargo build --features lsp      # Language Server
cargo build --features smt      # Z3 refinement types
cargo build --features gpu      # GPU codegen
cargo build --features full     # All features
```

## Compiler Architecture

**Pipeline:** Source → Lexer (Logos) → Parser → AST → Type Checker → HIR → HLIR (SSA) → Codegen

Key modules in `compiler/src/`:
- `lexer/`, `parser/`, `ast/` — Frontend
- `check/`, `types/` — Bidirectional type inference
- `effects/` — Algebraic effect system (IO, Mut, Alloc, Panic, Async, GPU, Prob, Div)
- `linear/`, `ownership/` — Linear/affine type checking
- `units/` — Dimensional analysis (mg, mL, h, etc.)
- `refinement/`, `smt/` — Z3-backed refinement types
- `epistemic/` — Confidence and provenance tracking
- `ontology/` — Scientific ontology integration (15M+ terms)
- `hir/` — Typed high-level IR
- `hlir/` — SSA-based low-level IR
- `codegen/` — LLVM, Cranelift JIT, GPU backends
- `interp/` — Interpreter
- `lsp/` — Language Server Protocol

## Sounio Language Syntax (NOT Rust)

**CRITICAL SYNTAX DIFFERENCES:**

```d
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
```

**What does NOT work in Sounio:**
- `&mut` - use `&!` instead
- `assert!()`, `println!()` - no Rust macros
- `#[test]`, `#[derive()]` - no attribute macros
- `let (a, b) = tuple` - no tuple destructuring
- `|(x, y)| expr` - no tuple destructuring in closures

## Test Organization

- `compiler/tests/` — Integration tests (Rust)
- `tests/ui/` — Error message verification
- `tests/run-pass/` — Should compile and run
- `tests/compile-fail/` — Should fail to compile

Test annotations in Sounio files:
```d
//@ run-pass
//@ compile-fail
//@ error-pattern: <text>
```

## Coding Standards

- Use `thiserror` for error types, `miette` for diagnostics with source spans
- No `unwrap()` in library code—use `?` or proper error handling
- All public items need doc comments

## Commit Format

```
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, ontology, epistemic
```
