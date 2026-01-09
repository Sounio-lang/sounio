---
title: Installing Sounio
description: Build and install the Sounio compiler (souc) on your system
prerequisites: None
reading_time: 10 minutes
---

# Installing Sounio

This guide walks you through building the Sounio compiler (`souc`) from source and verifying your installation.

## Prerequisites

Sounio is written in Rust and requires the Rust toolchain to build:

- **Rust 1.70 or later** - Install via [rustup](https://rustup.rs/)
- **Git** - For cloning the repository
- **A C compiler** (optional) - Required for LLVM backend

### Installing Rust

If you do not have Rust installed, run:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the prompts to complete the installation, then restart your terminal or run:

```bash
source $HOME/.cargo/env
```

Verify your Rust installation:

```bash
rustc --version
# rustc 1.70.0 (90c541806 2023-05-31) or later
```

## Building from Source

### 1. Clone the Repository

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio
```

### 2. Build the Compiler

For development (faster compilation, slower runtime):

```bash
cd compiler
cargo build
```

For release (slower compilation, optimized runtime):

```bash
cd compiler
cargo build --release
```

The compiled binary will be at:
- Development: `compiler/target/debug/souc`
- Release: `compiler/target/release/souc`

### 3. Install the Binary (Optional)

To install `souc` system-wide:

**macOS and Linux:**

```bash
# From the compiler directory
sudo cp target/release/souc /usr/local/bin/

# Or add to your PATH in ~/.bashrc or ~/.zshrc
export PATH="$PATH:/path/to/sounio/compiler/target/release"
```

**Windows:**

Add the path to `compiler\target\release` to your system PATH environment variable.

## Verify Installation

Check that Sounio is installed correctly:

```bash
souc --version
# souc 0.97.0
```

Run the help command to see available options:

```bash
souc --help
```

You should see output listing commands like `check`, `run`, `build`, and `repl`.

## Feature Flags

Sounio supports optional features that can be enabled during compilation:

### JIT Compilation (Cranelift)

For faster iteration during development, enable the JIT backend:

```bash
cargo build --release --features jit
```

Then run programs with:

```bash
souc run program.sio  # Uses JIT by default when available
```

### LLVM Backend

For native compilation to optimized machine code:

```bash
# Requires LLVM 15+ installed on your system
cargo build --release --features llvm
```

Compile to native executable:

```bash
souc build program.sio -o program
./program
```

### Language Server Protocol (LSP)

For editor integration:

```bash
cargo build --release --features lsp
```

See [Editor Setup](./editor-setup.md) for configuration details.

### All Features

Enable everything (requires all dependencies):

```bash
cargo build --release --features full
```

## Platform-Specific Notes

### macOS

On Apple Silicon (M1/M2/M3), ensure you are using the native arm64 Rust toolchain:

```bash
rustup default stable-aarch64-apple-darwin
```

For LLVM backend, install via Homebrew:

```bash
brew install llvm@15
export LLVM_SYS_150_PREFIX=$(brew --prefix llvm@15)
```

### Linux

Most distributions work out of the box. For LLVM backend:

**Ubuntu/Debian:**

```bash
sudo apt-get install llvm-15-dev libclang-15-dev
```

**Fedora:**

```bash
sudo dnf install llvm15-devel clang15-devel
```

**Arch Linux:**

```bash
sudo pacman -S llvm clang
```

### Windows

Sounio works on Windows with the MSVC toolchain:

```bash
rustup default stable-x86_64-pc-windows-msvc
```

For LLVM backend, download pre-built binaries from [LLVM releases](https://releases.llvm.org/) and add to PATH.

## Running Your First Program

Create a file called `hello.sio`:

```sio
fn main() -> i32 {
    print("Hello, Sounio!")
    println()
    0
}
```

Run it:

```bash
souc run hello.sio
# Output: Hello, Sounio!
```

## Troubleshooting

### "souc: command not found"

The binary is not in your PATH. Either:
1. Use the full path: `./target/release/souc`
2. Add the directory to your PATH
3. Copy the binary to `/usr/local/bin/`

### Build fails with "could not find LLVM"

LLVM is only required if you use `--features llvm`. For basic usage, omit this flag:

```bash
cargo build --release  # No LLVM needed
```

### Permission denied on Linux/macOS

Make the binary executable:

```bash
chmod +x target/release/souc
```

### Cargo not found after installing Rust

Run `source $HOME/.cargo/env` or restart your terminal.

## Next Steps

- [Hello World](./hello-world.md) - Write your first Sounio program
- [Your First Uncertainty](./your-first-uncertainty.md) - Learn Sounio's unique feature
- [Editor Setup](./editor-setup.md) - Configure your editor for Sounio

## See Also

- [Project Structure](./project-structure.md) - How Sounio projects are organized
- [Language Reference](../LLM_PROGRAMMING_GUIDE.md) - Complete syntax guide
