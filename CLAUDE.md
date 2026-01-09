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

## Versioning Policy

**Semantic Versioning (SemVer):** `MAJOR.MINOR.PATCH`

**Current version:** 0.97.0 (pre-1.0 development)

### Pre-1.0 Rules (current phase)
- `0.MINOR.PATCH` — MINOR bumps may include breaking changes
- PATCH bumps are for bug fixes and non-breaking additions
- API stability is not guaranteed between MINOR versions
- Target: 1.0.0 when language semantics, type system, and core stdlib are stable

### Post-1.0 Rules
- **MAJOR** — Breaking changes to language syntax, semantics, or public compiler APIs
- **MINOR** — New features, stdlib additions, non-breaking enhancements
- **PATCH** — Bug fixes, performance improvements, documentation

### Version Bump Triggers
| Change Type | Pre-1.0 | Post-1.0 |
|-------------|---------|----------|
| Breaking syntax/semantics | MINOR | MAJOR |
| New language feature | MINOR | MINOR |
| New stdlib module | PATCH | MINOR |
| Bug fix | PATCH | PATCH |
| Performance improvement | PATCH | PATCH |
| Documentation only | PATCH | PATCH |

## Tag and Release Policy

### Tag Format
```
v{MAJOR}.{MINOR}.{PATCH}[-{prerelease}]

Examples:
  v0.97.0        # Stable release
  v0.98.0-alpha  # Alpha pre-release
  v0.98.0-beta.1 # Beta with iteration
  v0.98.0-rc.1   # Release candidate
  v1.0.0         # First stable release
```

### Release Process

**1. Pre-release checklist:**
```bash
# All tests must pass
cd compiler && cargo test --all-features
cargo clippy --all-features
cargo fmt --check

# Verify version in Cargo.toml matches intended release
grep '^version' Cargo.toml
```

**2. Version bump (edit Cargo.toml):**
```bash
# Update version in compiler/Cargo.toml
# Commit with format: [release] Bump version to X.Y.Z
```

**3. Create annotated tag:**
```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z

Highlights:
- Feature 1
- Feature 2
- Bug fixes"
```

**4. Push release:**
```bash
git push origin main
git push origin vX.Y.Z
```

### Release Cadence
- **Alpha/Beta:** As needed during active development
- **Stable:** When significant features land or critical bugs are fixed
- **No fixed schedule** — quality over frequency

### Breaking Change Policy
- Document all breaking changes in release notes
- Provide migration guidance for syntax/API changes
- For post-1.0: deprecate first, remove in next MAJOR

## Commit Format

```
[component] Brief description

Components: lexer, parser, ast, check, types, typeck, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, ontology, epistemic, lsp,
           medlang, fmri, causal, gpu, sir, units, refinement, pkg, release

Examples:
  [parser] Add support for Knowledge<T> generic syntax
  [stdlib] Implement bootstrap_correlation in connectivity module
  [docs] Update README with MedLang integration examples
  [epistemic] Fix uncertainty propagation in division
  [release] Bump version to 0.98.0
```

### Commit Types by Impact
- **feat:** New feature (triggers MINOR bump consideration)
- **fix:** Bug fix (triggers PATCH bump)
- **perf:** Performance improvement (PATCH)
- **refactor:** Code restructuring, no behavior change (no bump needed)
- **docs:** Documentation only (PATCH if released)
- **test:** Test additions/fixes (no bump needed)
- **chore:** Build/tooling changes (no bump needed)

### Breaking Change Commits
When a commit introduces breaking changes, append `!` or note in body:
```
[parser]! Remove deprecated `unsafe` block syntax

BREAKING: The `unsafe { }` syntax has been removed.
Use `trust { }` instead.

Migration: Replace all `unsafe` keywords with `trust`.
```
