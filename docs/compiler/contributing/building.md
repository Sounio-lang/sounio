# Building the Compiler

This guide covers building the Sounio compiler, running tests, and the development workflow.

## Prerequisites

- **Rust**: 1.70 or later (install via [rustup](https://rustup.rs/))
- **Cargo**: Comes with Rust

### Optional Dependencies

- **LLVM 15+**: Required for `--features llvm`
- **CUDA Toolkit**: Required for `--features cuda`
- **Z3**: Required for `--features smt`

## Building

### Basic Build

```bash
# From repository root
cd compiler && cargo build

# Or from compiler directory
cargo build

# Release build (optimized)
cargo build --release
```

### Feature Flags

The compiler supports several optional features:

```bash
# JIT compilation (Cranelift)
cargo build --features jit

# LLVM backend (requires LLVM 15+)
cargo build --features llvm

# Language Server Protocol
cargo build --features lsp

# Z3 refinement types
cargo build --features smt

# GPU codegen (PTX/SPIR-V)
cargo build --features gpu

# CUDA runtime (requires CUDA toolkit)
cargo build --features cuda

# Scientific ontology (15M+ terms)
cargo build --features ontology

# LLM integration
cargo build --features llm

# Package manager
cargo build --features pkg

# WebAssembly target
cargo build --features wasm

# All features
cargo build --features full
```

### Running the Compiler

```bash
# Check a Sounio program
cargo run -- check examples/hello.sio

# Show AST and types
cargo run -- check examples/hello.sio --show-ast --show-types

# Run via interpreter
cargo run -- run examples/hello.sio

# Run via JIT (requires --features jit)
cargo run --features jit -- run examples/hello.sio

# Native compilation (requires --features llvm)
cargo run --features llvm -- build examples/hello.sio

# Start REPL
cargo run -- repl
```

## Running Tests

### All Tests

```bash
cargo test
```

### Specific Tests

```bash
# Run a specific test
cargo test test_name

# Run a specific integration test file
cargo test --test integration_semantic_types
cargo test --test integration_ontology_e2e

# Show test output
cargo test -- --nocapture
```

### Test Organization

**Rust integration tests**: `compiler/tests/`

**Sounio test suites**:
- `tests/run-pass/` - Should compile and run successfully
- `tests/compile-fail/` - Should fail to compile
- `tests/ui/` - Error message verification
- `tests/ffi/` - Foreign function interface tests

**Test annotations in Sounio files**:

```sio
//@ run-pass
//@ compile-fail
//@ error-pattern: <expected error text>
```

### Benchmarks

```bash
cargo bench --bench layout_bench
cargo bench --bench locality_bench
cargo bench --bench ontology_bench
cargo bench --bench compiler_bench
cargo bench --bench sir_gpu_bench
```

## Code Quality

### Formatting

```bash
# Format code
cargo fmt

# Check formatting without modifying
cargo fmt --check
```

### Linting

```bash
# Run clippy
cargo clippy

# With all features
cargo clippy --all-features
```

## LSP Server

```bash
# Build and run LSP server (requires --features lsp)
cargo run --features lsp --bin sounio-lsp
```

## Ontology Database

```bash
# Build ontology database (requires --features ontology)
cargo run --bin sounio-ontology-build
```

## Development Workflow

### 1. Making Changes

1. Create a feature branch
2. Make changes
3. Run tests: `cargo test`
4. Format: `cargo fmt`
5. Lint: `cargo clippy`

### 2. Testing Changes

For parser changes:
```bash
cargo test --lib parser
cargo test --test integration_parser
```

For type checker changes:
```bash
cargo test --lib check
cargo test --test integration_semantic_types
```

For codegen changes:
```bash
cargo test --lib codegen
cargo test --test integration_codegen
```

### 3. Adding New Features

When adding a new language feature:

1. **Lexer**: Add new tokens to `compiler/src/lexer/tokens.rs`
2. **Parser**: Add parsing in `compiler/src/parser/mod.rs`
3. **AST**: Add AST nodes to `compiler/src/ast/mod.rs`
4. **Type Checker**: Add type checking in `compiler/src/check/mod.rs`
5. **HIR**: Add HIR nodes to `compiler/src/hir/mod.rs`
6. **HLIR**: Add lowering in `compiler/src/hlir/`
7. **Codegen**: Add code generation for each backend
8. **Tests**: Add tests for the new feature

### 4. Commit Format

```
[component] Brief description

Components: lexer, parser, ast, check, types, typeck, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, ontology, epistemic, lsp,
           medlang, fmri, causal, gpu, sir, units, refinement, pkg

Examples:
  [parser] Add support for Knowledge<T> generic syntax
  [stdlib] Implement bootstrap_correlation in connectivity module
  [epistemic] Fix uncertainty propagation in division
```

## Debugging

### Debug Builds

Debug builds include additional checks:

```bash
cargo build  # Debug by default
```

### Verbose Output

```bash
# Show compilation stages
RUST_LOG=debug cargo run -- check examples/hello.sio
```

### Running with Debugger

```bash
# Generate debug info
cargo build

# Run with lldb
lldb target/debug/souc -- check examples/hello.sio
```

## Project Structure

```
compiler/
+-- Cargo.toml       # Package manifest
+-- src/
|   +-- lib.rs       # Library entry point
|   +-- main.rs      # CLI entry point (souc)
|   +-- lexer/       # Tokenization
|   +-- parser/      # Parsing
|   +-- ast/         # AST types
|   +-- check/       # Type checking
|   +-- types/       # Type system
|   +-- effects/     # Effect system
|   +-- hir/         # High-level IR
|   +-- hlir/        # SSA-based IR
|   +-- sir/         # Scientific IR
|   +-- codegen/     # Code generation
|   +-- interp/      # Interpreter
|   +-- lsp/         # Language server
|   +-- ...
+-- tests/           # Integration tests
+-- benches/         # Benchmarks
```

## Troubleshooting

### LLVM Not Found

If LLVM is not found:

```bash
# macOS
brew install llvm@15
export LLVM_SYS_150_PREFIX=$(brew --prefix llvm@15)

# Ubuntu
sudo apt install llvm-15-dev
```

### Cranelift Compilation Issues

Ensure you have a recent Rust version:

```bash
rustup update
```

### Feature Conflicts

Some features may conflict. Use `--features full` carefully, or specify only needed features.

## Continuous Integration

The CI pipeline runs:

1. `cargo fmt --check`
2. `cargo clippy --all-features`
3. `cargo test --all-features`
4. `cargo build --release --all-features`

Ensure all checks pass before submitting PRs.
