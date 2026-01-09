# Sounio CLI (`souc`)

The `souc` command is the primary interface to the Sounio compiler. It handles type checking, compilation, execution, and auxiliary tasks like formatting and package management.

## Installation

After building the compiler:

```bash
cd compiler
cargo build --release

# The binary is at target/release/souc
# Add to PATH or create symlink
ln -s $(pwd)/target/release/souc ~/.local/bin/souc
```

## Commands

### `souc check`

Type-check a Sounio source file without generating code.

```bash
souc check <file.sio> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--show-ast` | Print the parsed AST |
| `--show-types` | Print inferred types for expressions |
| `--show-hir` | Print the typed HIR |

**Examples:**

```bash
# Basic type check
souc check src/main.sio

# Debug parsing issues
souc check src/main.sio --show-ast

# Inspect type inference
souc check src/main.sio --show-types

# Full compilation pipeline inspection
souc check src/main.sio --show-ast --show-types --show-hir
```

### `souc run`

Execute a Sounio program using the interpreter or JIT.

```bash
souc run <file.sio> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--backend=<name>` | Execution backend: `interpreter` (default), `cranelift` |

**Examples:**

```bash
# Run with interpreter (default)
souc run examples/hello.sio

# Run with JIT (requires --features jit)
souc run --backend=cranelift examples/compute.sio
```

### `souc build`

Compile a Sounio program to native code.

```bash
souc build <file.sio> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-o, --output <path>` | Output file path |
| `--backend=<name>` | Compiler backend: `native`, `llvm`, `cranelift`, `gpu` |
| `-O<level>` | Optimization level: `0`, `1`, `2`, `3`, `s` |
| `--thermal=<model>` | Thermal model: `none`, `7nm`, `5nm`, `conservative` |
| `--no-thermal` | Disable thermal modeling |
| `--alloc=<strategy>` | Register allocation: `epistemic`, `linear`, `graph` |
| `-g, --debug` | Emit debug information |
| `--emit=<stages>` | Emit intermediate representations |
| `--timing` | Show compilation timing breakdown |
| `-v, --verbose` | Verbose output |

**Backends:**

| Backend | Description | Feature Flag |
|---------|-------------|--------------|
| `native` | Direct x86-64 emission with epistemic-aware allocation | Always available |
| `llvm` | LLVM for multi-architecture support | `--features llvm` |
| `cranelift` | Fast JIT compilation | `--features jit` |
| `gpu` | NVIDIA PTX emission | `--features gpu` |

**Examples:**

```bash
# Build executable (native backend)
souc build src/main.sio -o myapp

# Build shared library
souc build src/lib.sio -o libmylib.so

# Optimized release build
souc build src/main.sio -O3 -o myapp

# Debug build with symbols
souc build src/main.sio -O0 -g -o myapp_debug

# LLVM backend for cross-compilation
souc build --backend=llvm src/main.sio -o myapp

# GPU kernel compilation
souc build --backend=gpu src/kernel.sio -o kernel.ptx

# Emit intermediate representations
souc build --emit=ast,sir,asm src/main.sio

# Show timing breakdown
souc build --timing src/main.sio -o myapp
```

**Native Backend Options:**

The native backend supports epistemic-aware optimizations:

```bash
# Use epistemic register allocation (default)
souc build --alloc=epistemic src/main.sio

# Traditional linear scan allocation
souc build --alloc=linear src/main.sio

# Thermal-aware compilation for 7nm process
souc build --thermal=7nm src/main.sio

# Disable thermal modeling for faster compilation
souc build --no-thermal src/main.sio
```

### `souc repl`

Start the interactive REPL (Read-Eval-Print Loop).

```bash
souc repl [options]
```

See [REPL Guide](repl.md) for detailed documentation.

### `souc fmt`

Format Sounio source files.

```bash
souc fmt <path> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--check` | Check formatting without modifying files |
| `--diff` | Show diff of formatting changes |
| `--config <path>` | Path to configuration file |

See [Formatter Guide](formatter.md) for configuration options.

### `souc pkg`

Package management commands.

```bash
souc pkg <subcommand> [options]
```

See [Package Manager Guide](package-manager.md) for detailed documentation.

### `souc backend`

Display information about compiler backends.

```bash
# List available backends
souc backend --list

# Show detailed info for a backend
souc backend --info native
```

**Example output:**

```
Available backends:

  [checkmark] native       Native SIR backend with epistemic-aware allocation (x86-64)
  [checkmark] llvm         LLVM backend for AOT compilation (multi-arch)
              Warning: May have issues with refinement types + FFI
  [x] cranelift    Cranelift JIT backend for fast compilation
  [x] gpu          GPU backend for NVIDIA PTX emission
```

## Diagnostic Output

### Error Format

Sounio uses rich diagnostic output with source context:

```
error[E0308]: type mismatch
  --> src/main.sio:15:12
   |
15 |     let x: i32 = "hello"
   |            ^^^   ^^^^^^^ expected `i32`, found `string`
   |            |
   |            expected due to this type annotation
   |
help: consider using a numeric literal
   |
15 |     let x: i32 = 42
   |                  ~~
```

### Warning Levels

Control warning output:

```bash
# Show all warnings
souc check src/main.sio -W all

# Treat warnings as errors
souc check src/main.sio -W error

# Suppress specific warnings
souc check src/main.sio -W no-unused-variables
```

## Configuration Files

### `sounio.toml`

Project configuration file (see [Package Manager](package-manager.md)):

```toml
[package]
name = "my-project"
version = "0.1.0"

[build]
target-dir = "target"
incremental = true

[profile.release]
opt-level = 3
debug = false
```

### `.souniofmt.toml`

Formatter configuration (see [Formatter](formatter.md)):

```toml
max_width = 100
indent_width = 4
use_tabs = false
```

## Shell Completion

Generate shell completions:

```bash
# Bash
souc completions bash > ~/.local/share/bash-completion/completions/souc

# Zsh
souc completions zsh > ~/.zsh/completions/_souc

# Fish
souc completions fish > ~/.config/fish/completions/souc.fish
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SOUNIO_HOME` | Installation directory |
| `SOUNIO_CACHE` | Build cache location |
| `RUST_LOG` | Logging level (`debug`, `info`, `warn`, `error`) |
| `NO_COLOR` | Disable colored output |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Compilation error |
| `2` | Command-line error |
| `3` | I/O error |
| `101` | Internal error (bug) |

## Troubleshooting

### Common Issues

**"Backend not available"**

The requested backend feature is not compiled in:

```bash
# Check available backends
souc backend --list

# Rebuild with feature
cd compiler && cargo build --features jit
```

**"Parse error: unexpected token"**

Check syntax - Sounio uses `&!` for mutable references, not `&mut`:

```sio
// Correct
fn mutate(x: &!i32) { ... }

// Incorrect (Rust syntax)
fn mutate(x: &mut i32) { ... }
```

**"Type error: unknown type"**

Ensure imports are correct:

```sio
use std::collections::HashMap
use epistemic::Knowledge
```

### Debug Mode

Enable verbose logging:

```bash
RUST_LOG=debug souc check src/main.sio
```

## See Also

- [Language Server](lsp.md) - IDE integration
- [REPL](repl.md) - Interactive development
- [Package Manager](package-manager.md) - Project management
- [Formatter](formatter.md) - Code style
