# Sounio Installation Guide

> **For SoftwareX Reviewers**: This document provides step-by-step installation instructions for evaluating the Sounio compiler. A quick-start reviewer guide is provided in the [Reviewer Quick Start](#reviewer-quick-start) section below.

This document provides comprehensive installation instructions for the Sounio compiler across Linux, macOS, and Windows platforms.

## Table of Contents

- [Reviewer Quick Start](#reviewer-quick-start)
- [Prerequisites](#prerequisites)
- [Linux Installation (Ubuntu 22.04/24.04)](#linux-installation-ubuntu-220424)
- [macOS Installation (Sonoma/Sequoia)](#macos-installation-sonomasequoia)
- [Windows Installation (WSL2)](#windows-installation-wsl2)
- [Feature Flags](#feature-flags)
- [Verification](#verification)
- [Running Examples](#running-examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Reviewer Quick Start

**Estimated time**: 15-20 minutes for minimal installation and verification.

This section provides streamlined instructions for SoftwareX reviewers to quickly install and verify the Sounio compiler.

### Minimal Installation (Linux/Ubuntu)

```bash
# 1. Install Rust (only required dependency)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 2. Install build tools
sudo apt update
sudo apt install -y build-essential git

# 3. Clone repository
git clone https://github.com/sounio-lang/sounio.git
cd sounio

# 4. Build compiler (takes ~5-10 minutes on first build)
cargo build --release

# 5. Verify installation
./target/release/souc --version
# Expected output: souc 0.99.0
```

### Quick Verification

```bash
# Set stdlib path
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

# Check basic examples
./target/release/souc check examples/hello.sio
./target/release/souc check examples/fibonacci.sio
./target/release/souc check examples/structs.sio

# Run test suite (~2-3 minutes)
cargo test --lib --release
```

**Expected results**:
- ✓ All examples should type-check successfully
- ✓ Library tests should pass (some integration tests may be skipped without optional features)
- ✓ No compilation errors or warnings

### Minimal macOS Installation

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 2. Install Xcode Command Line Tools (if needed)
xcode-select --install

# 3. Clone and build
git clone https://github.com/sounio-lang/sounio.git
cd sounio
cargo build --release

# 4. Verify
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./target/release/souc --version
./target/release/souc check examples/hello.sio
```

### What You Can Evaluate

With the minimal installation, reviewers can:

1. **Core Language Features**:
   - Type checking and inference
   - Epistemic types (`Knowledge<T>`)
   - Dimensional analysis (units)
   - Algebraic effects
   - Linear types
   - Refinement types (runtime assertions)

2. **Native Code Generation**:
   - Native backend produces ELF (Linux) or Mach-O (macOS) binaries
   - Example: `./target/release/souc compile examples/fibonacci.sio`

3. **Standard Library**:
   - 151,000+ lines of scientific computing code
   - Located in `stdlib/` directory
   - Examples using stdlib: `uncertainty.sio`, `units_simple.sio`, `ode_demo.sio`

4. **Test Suite**:
   - `cargo test --lib` runs unit tests (~400 tests)
   - `cargo test --release` runs full integration suite (3800+ tests, requires optional features)

### Key Examples for Reviewers

| Example File | Demonstrates | Command |
|--------------|--------------|---------|
| `hello.sio` | Basic syntax | `souc check examples/hello.sio` |
| `fibonacci.sio` | Recursion, basic types | `souc check examples/fibonacci.sio` |
| `structs.sio` | User-defined types | `souc check examples/structs.sio` |
| `uncertainty.sio` | Epistemic types | `souc check examples/uncertainty.sio` |
| `units_simple.sio` | Dimensional analysis | `souc check examples/units_simple.sio` |
| `ode_demo.sio` | Scientific computing | `souc check examples/ode_demo.sio` |
| `effects_simple.sio` | Algebraic effects | `souc check examples/effects_simple.sio` |

All examples are located in the `examples/` directory at the repository root.

### Optional Features for Extended Evaluation

For a more complete evaluation, you may optionally install:

```bash
# LLVM backend (optimized native code generation)
sudo apt install -y llvm-15 llvm-15-dev libzstd-dev
export LLVM_SYS_150_PREFIX=/usr/lib/llvm-15
cargo build --release --features llvm15

# JIT/Cranelift (for `run` command - execute programs)
cargo build --release --features jit
./target/release/souc run examples/fibonacci.sio

# GPU support (PTX/SPIR-V code generation)
cargo build --release --features gpu

# All tests with features
cargo test --release --features "jit,gpu"
```

See [Feature Flags](#feature-flags) section for complete list.

### Platform Notes

- **Linux**: Fully supported (Ubuntu 22.04+, Debian 12+, Fedora 39+, Arch)
- **macOS**: Fully supported (Sonoma 14.x+, Sequoia 15.x+, both Intel and Apple Silicon)
- **Windows**: Use WSL2 (Windows Subsystem for Linux)

### Expected Build Times

- **First build**: 5-10 minutes (downloads and compiles dependencies)
- **Incremental builds**: 1-2 minutes (after code changes)
- **Test suite**: 2-3 minutes (library tests), 5-10 minutes (full integration tests)

---

---

## Prerequisites

### Required

| Component | Version | Notes |
|-----------|---------|-------|
| Rust | 1.70+ | The Sounio compiler is written in Rust |
| Cargo | 1.70+ | Included with Rust installation |
| C Compiler | GCC 9+ or Clang 11+ | Required for native code generation |
| Git | 2.0+ | For cloning the repository |

### Optional Dependencies

| Component | Version | Purpose | Feature Flag |
|-----------|---------|---------|--------------|
| LLVM | 15, 16, or 17 | Optimized native code generation | `llvm15`, `llvm16`, `llvm17` |
| Z3 | 4.8+ | SMT solver for refinement type verification | `smt` |
| CMake | 3.20+ | Required for building Z3 bindings | `smt` |
| libzstd | 1.4+ | Compression library (required by LLVM) | `llvm*` |
| CUDA Toolkit | 11.0+ | GPU kernel execution | `cuda` |

---

## Linux Installation (Ubuntu 22.04/24.04)

### Step 1: Install System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential git curl pkg-config
```

### Step 2: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup update stable
```

Verify installation:

```bash
rustc --version   # Should show 1.70.0 or later
cargo --version
```

### Step 3: Clone and Build

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio
cargo build --release
```

The compiled binary will be at `./target/release/souc`.

### Step 4: Install to PATH (Optional)

```bash
cargo install --path .
```

This installs `souc` to `~/.cargo/bin/`, which should already be in your PATH.

### Step 5: Configure Standard Library Path

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib' >> ~/.bashrc
source ~/.bashrc
```

Replace `/path/to/sounio` with your actual installation path.

### Optional: Install LLVM Backend

For optimized native code generation:

```bash
# Install LLVM 15 and dependencies
sudo apt install -y llvm-15 llvm-15-dev libllvm15 clang-15 lld-15 libzstd-dev

# Set environment variable
export LLVM_SYS_150_PREFIX=/usr/lib/llvm-15

# Build with LLVM support
cargo build --release --features llvm15
```

For LLVM 17 (if available in your distribution):

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-17 main"
sudo apt update
sudo apt install -y llvm-17 llvm-17-dev libllvm17 clang-17 lld-17 libzstd-dev
export LLVM_SYS_170_PREFIX=/usr/lib/llvm-17
cargo build --release --features llvm17
```

### Optional: Install Z3 for Refinement Types

```bash
sudo apt install -y z3 libz3-dev cmake
cargo build --release --features smt
```

---

## macOS Installation (Sonoma/Sequoia)

### Step 1: Install Xcode Command Line Tools

```bash
xcode-select --install
```

### Step 2: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 3: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup update stable
```

Verify installation:

```bash
rustc --version   # Should show 1.70.0 or later
cargo --version
```

### Step 4: Clone and Build

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio
cargo build --release
```

### Step 5: Install to PATH (Optional)

```bash
cargo install --path .
```

### Step 6: Configure Standard Library Path

```bash
# Add to ~/.zshrc (default shell on macOS)
echo 'export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib' >> ~/.zshrc
source ~/.zshrc
```

### Optional: Install LLVM Backend

```bash
brew install llvm@15 zstd
export LLVM_SYS_150_PREFIX=$(brew --prefix llvm@15)
export PATH="$(brew --prefix llvm@15)/bin:$PATH"
cargo build --release --features llvm15
```

For LLVM 17:

```bash
brew install llvm@17 zstd
export LLVM_SYS_170_PREFIX=$(brew --prefix llvm@17)
export PATH="$(brew --prefix llvm@17)/bin:$PATH"
cargo build --release --features llvm17
```

### Optional: Install Z3 for Refinement Types

```bash
brew install z3 cmake
cargo build --release --features smt
```

---

## Windows Installation (WSL2)

Sounio is developed and tested primarily on Unix-like systems. For Windows, we recommend using WSL2 (Windows Subsystem for Linux).

### Step 1: Install WSL2

Open PowerShell as Administrator and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart your computer when prompted.

### Step 2: Follow Linux Instructions

After WSL2 is installed, open the Ubuntu terminal and follow the [Linux Installation](#linux-installation-ubuntu-220424) instructions above.

For more information on WSL2, see the [official documentation](https://learn.microsoft.com/en-us/windows/wsl/install).

---

## Feature Flags

Sounio uses Cargo feature flags to enable optional functionality. Features can be combined as needed.

### Available Features

| Feature | Description | External Dependencies |
|---------|-------------|----------------------|
| `jit` | Cranelift JIT compilation | None |
| `llvm15` | LLVM 15 native codegen | LLVM 15, libzstd |
| `llvm16` | LLVM 16 native codegen | LLVM 16, libzstd |
| `llvm17` | LLVM 17 native codegen | LLVM 17, libzstd |
| `smt` | Z3 refinement type verification | Z3, CMake |
| `gpu` | GPU codegen (PTX, SPIR-V) | None |
| `cuda` | CUDA kernel execution | CUDA Toolkit |
| `lsp` | Language Server Protocol support | None |
| `ontology` | Scientific ontology (15M+ terms) | None |
| `pkg` | Package manager support | None |
| `full` | All features | All optional dependencies |

### Build Examples

```bash
# Minimal build (Cranelift JIT only)
cargo build --release

# With JIT and GPU codegen
cargo build --release --features "jit,gpu"

# With LLVM 15 and SMT verification
cargo build --release --features "llvm15,smt"

# Full feature build (requires all dependencies)
cargo build --release --features full
```

Note: LLVM version features (`llvm14`, `llvm15`, `llvm16`, `llvm17`) are mutually exclusive. Only enable one at a time.

---

## Running Examples

After installation, you can run the provided examples to verify functionality and explore Sounio's features.

### Basic Type Checking

The `check` command performs type checking without code generation:

```bash
cd sounio

# Basic examples (no stdlib required)
./target/release/souc check examples/hello.sio
./target/release/souc check examples/fibonacci.sio
./target/release/souc check examples/arithmetic.sio
./target/release/souc check examples/structs.sio
```

### Examples Using Standard Library

For examples that import stdlib modules, set the stdlib path:

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

# Epistemic computing examples
./target/release/souc check examples/uncertainty.sio
./target/release/souc check examples/beta_epistemic.sio

# Dimensional analysis
./target/release/souc check examples/units_simple.sio

# Scientific computing
./target/release/souc check examples/ode_demo.sio
./target/release/souc check examples/pharmacokinetics.sio

# Effects and control flow
./target/release/souc check examples/effects_simple.sio
./target/release/souc check examples/effects.sio
```

### Viewing AST and Type Information

Use flags to see compiler internals:

```bash
# Show parsed AST
./target/release/souc check examples/fibonacci.sio --show-ast

# Show inferred types
./target/release/souc check examples/fibonacci.sio --show-types

# Both
./target/release/souc check examples/fibonacci.sio --show-ast --show-types
```

### Native Code Generation

The native backend compiles to ELF (Linux) or Mach-O (macOS):

```bash
# Compile to native binary
./target/release/souc compile examples/fibonacci.sio -o fib

# Run the generated binary
./fib
```

**Note**: The `compile` command is available in the default build. It uses the native backend (not LLVM).

### JIT Execution (Requires `jit` Feature)

To execute programs directly, build with the JIT feature:

```bash
cargo build --release --features jit

# Run programs with JIT
./target/release/souc run examples/fibonacci.sio
./target/release/souc run examples/minimal.sio
```

### GPU Code Generation (Requires `gpu` Feature)

To generate PTX or SPIR-V code:

```bash
cargo build --release --features gpu

# Generate GPU kernels
./target/release/souc check examples/gpu.sio
```

### Interactive REPL

The compiler includes an interactive REPL:

```bash
./target/release/souc repl
```

Example session:

```sio
sounio> let x: i32 = 42
x: i32 = 42
sounio> let y: i32 = x + 8
y: i32 = 50
sounio> :type x
i32
sounio> :quit
```

### Advanced Examples

**Causal inference**:
```bash
./target/release/souc check examples/causal_model.sio
```

**Autodifferentiation**:
```bash
./target/release/souc check examples/autodiff.sio
./target/release/souc check examples/wave2_autodiff_simple.sio
```

**Scientific modeling**:
```bash
./target/release/souc check examples/wave1_ode_exponential_decay.sio
./target/release/souc check examples/wave2_neural_ode.sio
```

**PK/PD (Pharmacokinetics/Pharmacodynamics)**:
```bash
./target/release/souc check examples/pkpd.sio
```

For a complete list of examples, see the `examples/` directory.

---

## Verification

### Step 1: Check Compiler Version

```bash
./target/release/souc --version
```

Expected output:

```
souc 0.99.0
```

### Step 2: Run the Test Suite

```bash
cargo test --release
```

Expected output (summary):

```
test result: ok. XXX passed; 0 failed; 0 ignored
```

### Step 3: Check a Source File

Create a test file `test.sio`:

```sio
fn main() -> i32 {
    let x: i32 = 42
    return x
}
```

Run the type checker:

```bash
./target/release/souc check test.sio
```

Expected output:

```
Check passed: test.sio
```

### Step 4: Run an Example with JIT

```bash
cargo build --release --features jit
./target/release/souc run examples/minimal.sio
```

Expected output:

```
42
```

### Step 5: Verify Standard Library Path

```bash
./target/release/souc sysroot stdlib-paths
```

This displays all stdlib search locations and indicates which paths exist.

---

## Troubleshooting

### Error: `unable to find library -lzstd`

**Cause**: LLVM requires the zstd compression library.

**Solution**:

```bash
# Ubuntu/Debian
sudo apt install libzstd-dev

# macOS
brew install zstd

# Fedora/RHEL
sudo dnf install libzstd-devel
```

### Error: `LLVM_SYS_XXX_PREFIX not set`

**Cause**: The llvm-sys crate cannot locate your LLVM installation.

**Solution**:

```bash
# Find LLVM prefix
llvm-config --prefix

# Set the appropriate environment variable
# For LLVM 15:
export LLVM_SYS_150_PREFIX=$(llvm-config --prefix)

# For LLVM 17:
export LLVM_SYS_170_PREFIX=$(llvm-config-17 --prefix)
```

### Error: `z3.h not found`

**Cause**: Z3 development headers are not installed.

**Solution**:

```bash
# Ubuntu/Debian
sudo apt install libz3-dev

# macOS
brew install z3

# Fedora/RHEL
sudo dnf install z3-devel
```

### Error: `cmake not found`

**Cause**: CMake is required for building Z3 bindings.

**Solution**:

```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake

# Fedora/RHEL
sudo dnf install cmake
```

### Error: `Import not found` when running programs

**Cause**: The standard library path is not configured.

**Solution**:

```bash
# Set the stdlib path
export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib

# Verify the path
./target/release/souc sysroot stdlib-paths
```

### Error: `feature X requires feature Y`

**Cause**: Incompatible or missing feature dependencies.

**Solution**: Ensure only one LLVM version is enabled:

```bash
# Correct: single LLVM version
cargo build --features llvm17

# Incorrect: multiple LLVM versions
cargo build --features "llvm15,llvm17"  # This will fail
```

### Build is slow

**Cause**: Debug build compiles without optimizations.

**Solution**: Use release mode for faster execution:

```bash
cargo build --release
```

### Tests fail with timeout

**Cause**: Some tests require significant computation time.

**Solution**: Increase the test timeout or run specific tests:

```bash
# Run a specific test
cargo test test_name --release

# Run with output
cargo test --release -- --nocapture
```

---

## Environment Variables Reference

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `SOUNIO_STDLIB_PATH` | Standard library location | `/home/user/sounio/stdlib` |
| `LLVM_SYS_150_PREFIX` | LLVM 15 installation path | `/usr/lib/llvm-15` |
| `LLVM_SYS_160_PREFIX` | LLVM 16 installation path | `/usr/lib/llvm-16` |
| `LLVM_SYS_170_PREFIX` | LLVM 17 installation path | `/usr/lib/llvm-17` |
| `Z3_SYS_Z3_HEADER` | Custom Z3 header location | `/opt/z3/include/z3.h` |
| `CUDA_PATH` | CUDA toolkit location | `/usr/local/cuda` |
| `RUST_LOG` | Enable debug logging | `sounio=debug` |

---

## Advanced Configuration

### Custom LLVM Installation

If LLVM is installed in a non-standard location:

```bash
# Point to custom LLVM installation
export LLVM_SYS_170_PREFIX=/opt/llvm-17

# Or use llvm-config
export LLVM_SYS_170_PREFIX=$(llvm-config-17 --prefix)

# Build with custom LLVM
cargo build --release --features llvm17
```

### Cross-Compilation

Sounio supports cross-compilation for different targets:

```bash
# List available targets
rustup target list

# Install target
rustup target add aarch64-unknown-linux-gnu

# Cross-compile
cargo build --release --target aarch64-unknown-linux-gnu
```

### Optimization Levels

Control optimization during build:

```bash
# Debug build (fast compile, slow execution)
cargo build

# Release build (slow compile, fast execution)
cargo build --release

# Custom optimization in Cargo.toml
[profile.custom]
opt-level = 2
lto = false
```

### Parallel Build Configuration

Speed up compilation by adjusting parallelism:

```bash
# Use more CPU cores (default is #cores)
cargo build --release -j 16

# Limit cores to avoid overloading
cargo build --release -j 4
```

### Development vs Production Builds

**Development** (faster iteration):
```bash
cargo build --features jit
export RUST_LOG=sounio=debug
```

**Production** (maximum performance):
```bash
cargo build --release --features "llvm17,smt,gpu,ontology"
export RUST_LOG=warn
```

### Cargo Cache Management

Sounio's dependencies require significant disk space (~2-3 GB). To manage:

```bash
# Check cache size
du -sh ~/.cargo

# Clean build artifacts
cargo clean

# Remove old cached packages
cargo install cargo-cache
cargo cache --autoclean
```

### Running Tests Selectively

```bash
# Run only unit tests (fast)
cargo test --lib

# Run specific test module
cargo test --lib types::

# Run integration tests with features
cargo test --test native_e2e_test --features jit

# Run with verbose output
cargo test -- --nocapture

# Run single test
cargo test test_fibonacci --release
```

### Enabling Logging

Sounio uses the `tracing` crate for diagnostics:

```bash
# Debug level logging
export RUST_LOG=sounio=debug
./target/release/souc check examples/fibonacci.sio

# Trace level (very verbose)
export RUST_LOG=sounio=trace

# Log only specific modules
export RUST_LOG=sounio::check=debug,sounio::types=trace
```

### Performance Profiling

For benchmarking and optimization work:

```bash
# Build with profiling enabled
cargo build --release --features jit

# Run benchmarks
cargo bench

# Specific benchmark suite
cargo bench --bench compiler_bench
```

### IDE Integration

**Visual Studio Code** with rust-analyzer:

```bash
# Install rust-analyzer extension
code --install-extension rust-lang.rust-analyzer

# Configure settings.json
{
  "rust-analyzer.cargo.features": ["jit", "gpu"],
  "rust-analyzer.checkOnSave.command": "clippy"
}
```

**Vim/Neovim** with coc.nvim:

```vim
" In coc-settings.json
{
  "rust-analyzer.cargo.features": ["jit", "gpu"]
}
```

### Binary Installation (Alternative to Building)

For users who don't want to build from source:

```bash
# Using cargo-binstall (if binary releases are available)
cargo install cargo-binstall
cargo binstall souc

# Or install directly from source (builds on your machine)
cargo install --git https://github.com/sounio-lang/sounio.git souc
```

**Note**: Binary releases are not yet available for Sounio v0.99. Building from source is currently required.

---

## Additional Resources

- **[README.md](README.md)** - Project overview and quick start
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Comprehensive installation guide with all dependencies
- **[docs/LLM_PROGRAMMING_GUIDE.md](docs/LLM_PROGRAMMING_GUIDE.md)** - Complete language syntax reference
- **[docs/MINIMUM_VIABLE_SOUNIO.md](docs/MINIMUM_VIABLE_SOUNIO.md)** - Current feature status and roadmap
- **[docs/FEATURE_FLAGS.md](docs/FEATURE_FLAGS.md)** - Detailed feature flag documentation
- **[compiler/docs/KNOWN_LIMITATIONS.md](compiler/docs/KNOWN_LIMITATIONS.md)** - Known issues and limitations
- **[CLAUDE.md](CLAUDE.md)** - Quick reference for LLM-assisted development
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

---

## Support and Contact

- **GitHub Issues**: https://github.com/sounio-lang/sounio/issues
- **Email**: demetrios@sounio-lang.org
- **Documentation**: https://sounio-lang.org (when available)

---

## Citation

If you use Sounio in your research, please cite:

```bibtex
@software{sounio2025,
  title = {Sounio: A Systems Language for Epistemic Computing},
  author = {Agourakis, Demetrios Chiuratto},
  year = {2025},
  version = {0.99.0},
  url = {https://github.com/sounio-lang/sounio},
  note = {Under review at SoftwareX}
}
```

---

*Last updated: January 2026*
*Sounio version: 0.99.0*
*For questions or issues, please open an issue on GitHub: https://github.com/sounio-lang/sounio/issues*
