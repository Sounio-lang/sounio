# Sounio Tooling

This section covers the development tools that make up the Sounio ecosystem. From compilation to editing, these tools are designed to support epistemic computing workflows while providing a familiar developer experience.

## Tools Overview

| Tool | Command | Description |
|------|---------|-------------|
| **Compiler** | `souc` | The Sounio compiler - check, build, and run Sounio programs |
| **Language Server** | `sounio-lsp` | LSP server for IDE integration |
| **REPL** | `souc repl` | Interactive shell with epistemic value visualization |
| **Package Manager** | `souc pkg` | Dependency management and project scaffolding |
| **Formatter** | `souc fmt` | Code formatting with configurable style |

## Quick Reference

### Compile and Run

```bash
# Type check a file
souc check src/main.sio

# Run with interpreter
souc run src/main.sio

# Build with native backend (default)
souc build src/main.sio -o output

# Build with LLVM backend
souc build --backend=llvm src/main.sio -o output

# Build with JIT (requires --features jit)
souc run --backend=cranelift src/main.sio
```

### Development Workflow

```bash
# Format code
souc fmt src/

# Check formatting without modifying
souc fmt --check src/

# Start REPL
souc repl

# Initialize new project
souc pkg init my-project

# Add dependency
souc pkg add statistics ^1.0

# Build project
souc pkg build --release
```

### Debugging

```bash
# Show AST during compilation
souc check src/main.sio --show-ast

# Show types
souc check src/main.sio --show-types

# Show HIR (typed intermediate representation)
souc check src/main.sio --show-hir

# Emit intermediate stages
souc build --emit=ast,sir,asm src/main.sio
```

## Feature Flags

The Sounio compiler supports optional features enabled at build time:

| Feature | Description |
|---------|-------------|
| `jit` | Cranelift JIT backend for fast development |
| `llvm` | LLVM backend for optimized native compilation |
| `lsp` | Language Server Protocol support |
| `smt` | Z3-backed refinement type verification |
| `gpu` | GPU codegen (PTX/SPIR-V) |
| `cuda` | CUDA runtime integration |
| `ontology` | Scientific ontology (15M+ terms) |
| `pkg` | Package manager |
| `wasm` | WebAssembly target |
| `full` | All features |

Enable features when building the compiler:

```bash
cargo build --features jit,lsp,gpu
cargo build --features full
```

## Getting Help

### Command Help

```bash
# General help
souc --help

# Command-specific help
souc check --help
souc build --help
souc pkg --help

# LSP help
sounio-lsp --help
```

### Version Information

```bash
# Compiler version
souc --version

# LSP version
sounio-lsp --version
```

### Documentation

- [CLI Reference](cli.md) - Complete `souc` command documentation
- [Language Server](lsp.md) - Editor integration guide
- [REPL Guide](repl.md) - Interactive development
- [Package Manager](package-manager.md) - Project and dependency management
- [Formatter](formatter.md) - Code style configuration

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOUNIO_HOME` | Sounio installation directory | Platform-specific |
| `SOUNIO_CACHE` | Package cache location | `~/.sounio/cache` |
| `SOUNIO_REGISTRY` | Default package registry | `https://registry.sounio.dev` |
| `RUST_LOG` | Logging level for debugging | `warn` |

## Exit Codes

All Sounio tools use consistent exit codes:

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Compilation error (type errors, parse errors) |
| `2` | Command-line argument error |
| `3` | I/O error (file not found, permission denied) |
| `101` | Internal compiler error |

## Next Steps

- [Get started with the CLI](cli.md)
- [Set up your editor](lsp.md)
- [Try the REPL](repl.md)
