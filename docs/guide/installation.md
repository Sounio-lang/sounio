<!-- docs:meta
topic_id: repo.docs.guide.installation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.installation
-->

# Sounio Installation Guide

This guide covers all installation scenarios for the Sounio compiler, from basic setup to full-featured builds with all optional dependencies.

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Basic Installation](#basic-installation)
- [Optional Dependencies](#optional-dependencies)
  - [LLVM Backend](#llvm-backend-llvm-15-17)
  - [SMT Solver (Z3)](#smt-solver-z3)
  - [GPU Support (CUDA)](#gpu-support-cuda)
- [Feature Flags](#feature-flags)
- [Build Configurations](#build-configurations)
- [Troubleshooting](#troubleshooting)

---

> **⚠️ Rewritten 2026-07-11 (doc-reality audit).** The current Sounio toolchain is **self-hosted and prebuilt** — you do **not** build it with `cargo`/Rust, there is no Cranelift/LLVM feature-flag build, and there is no `./target/release/souc`. The compiler binaries are checked into `bin/`. The Rust / `cargo` / LLVM 15–17 / Z3 / feature-flag sections further down describe the **retired** Rust build and no longer apply to this checkout; they are kept for historical reference only. Use the Quick Start below.

## Quick Start

The compiler ships **prebuilt** in `bin/`. No build step is needed to use it.

```bash
# Clone the repository
git clone https://github.com/Sounio-lang/sounio.git
cd sounio

# bin/souc routes to the self-hosted Madaros engine (no Rust/cargo needed)
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"   # required when running from outside the repo root
./bin/souc --version                        # -> Madaros v0.80.0 (an un-rebuilt checkout may still print "Madares")
./bin/souc run examples/hello.sio           # smoke test -> "Hello, Sounio"
```

### Building the compiler from source (optional)

Only needed to *rebuild* the compiler itself. It is self-hosted in Sounio and bootstrapped from a small C stage0 — **no Rust, no cargo**:

```bash
make build          # boot chain gen1 -> gen2 -> gen3; verifies the gen2 == gen3 fixed point
make build-madaros  # (re)build the modular Madaros compiler from self-hosted/compiler/main.sio
make check          # type-check the compiler + CI gates
```

---

## System Requirements

### Minimum Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Rust | 1.70+ | `rustup update stable` |
| Cargo | 1.70+ | Comes with Rust |
| C Compiler | GCC 9+ / Clang 11+ | For native code generation |

### Recommended (Full Features)

| Component | Version | Purpose |
|-----------|---------|---------|
| LLVM | 15, 16, or 17 | Native codegen backend |
| Z3 | 4.8+ | SMT solving for refinement types |
| CMake | 3.20+ | Required for Z3 binding |
| libzstd | 1.4+ | LLVM compression support |
| CUDA Toolkit | 11.0+ | GPU kernel execution |

---

## Basic Installation

### Step 1: Clone (the compiler is prebuilt — no Rust needed)

```bash
git clone https://github.com/Sounio-lang/sounio.git
cd sounio

# The compiler binaries are checked into bin/; nothing to install.
./bin/souc --version
```

### Step 2: (optional) Rebuild the self-hosted compiler

Only if you want to rebuild it from source. Sounio is self-hosted and bootstrapped from a C stage0 — **no Rust, no cargo, no rustup**:

```bash
make build          # gen1 -> gen2 -> gen3, verifies gen2 == gen3 fixed point
make install        # optional: place souc on PATH
```

### Step 3: Configure Stdlib Path

Set the stdlib path for programs that use standard library modules:

```bash
# Add to ~/.bashrc or ~/.zshrc
export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib

# Verify stdlib resolution by type-checking a program that imports the stdlib
./bin/souc check examples/hello.sio    # (there is no `souc sysroot` subcommand)
```

---

## Optional Dependencies

### LLVM Backend (LLVM 15-17)

The LLVM backend provides optimized native code generation. Sounio supports LLVM 15, 16, and 17.

#### Ubuntu/Debian (LLVM 17)

```bash
# Add LLVM repository
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-17 main"
sudo apt update

# Install LLVM 17 and dependencies
sudo apt install llvm-17 llvm-17-dev libllvm17 clang-17 lld-17

# CRITICAL: Install zstd library (required for linking)
sudo apt install libzstd-dev

# Set environment variables
export LLVM_SYS_170_PREFIX=/usr/lib/llvm-17
export PATH="/usr/lib/llvm-17/bin:$PATH"
```

#### macOS (Homebrew)

```bash
brew install llvm@17

# Set environment variables
export LLVM_SYS_170_PREFIX=$(brew --prefix llvm@17)
export PATH="$(brew --prefix llvm@17)/bin:$PATH"
```

#### Fedora/RHEL

```bash
sudo dnf install llvm17 llvm17-devel clang17 lld17 libzstd-devel
export LLVM_SYS_170_PREFIX=/usr
```

#### Build with LLVM

```bash
# LLVM 17 (recommended)
cargo build --release --features llvm17

# LLVM 16
cargo build --release --features llvm16

# LLVM 15
cargo build --release --features llvm15
```

#### Verify LLVM Installation

```bash
# Check LLVM version
llvm-config --version

# Check llvm-sys finds LLVM
cargo build --features llvm17 2>&1 | grep -i llvm
```

---

### SMT Solver (Z3)

Z3 enables compile-time verification of refinement types. Without Z3, refinement types fall back to runtime assertions.

#### Ubuntu/Debian

```bash
# Install Z3 and build dependencies
sudo apt install z3 libz3-dev cmake

# Verify Z3
z3 --version
```

#### macOS

```bash
brew install z3 cmake
```

#### Fedora/RHEL

```bash
sudo dnf install z3 z3-devel cmake
```

#### Build with SMT Support

```bash
cargo build --release --features smt
```

---

### GPU Support (CUDA)

GPU support enables PTX code generation and CUDA kernel execution.

#### Ubuntu/Debian

```bash
# Install CUDA toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Or via apt (if NVIDIA repo configured)
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
```

#### Build with GPU Support

```bash
cargo build --release --features gpu
```

**Note**: GPU codegen works without CUDA installed. CUDA is only required for actual kernel execution.

---

## Feature Flags

> **⚠️ This entire section describes the retired Rust tree — measured 2026-08-27.**
> There is no `Cargo.toml` in this repository; the Rust crates were removed on
> 2026-02-26 by `79acc192e1`. Every `cargo build --features …` line below, and every
> row of the table, is a record of how the compiler used to be built, not an
> instruction you can follow.
>
> Two rows are worth naming because they are quoted elsewhere as if current:
> `jit` is marked "(default)" and **Cranelift was never compiled into any shipped
> artifact** — `souc info` prints `[-] Cranelift JIT - rebuild with --features jit`,
> and the binary exports no Cranelift symbol. The same holds for `llvm15/16/17`.
>
> To build today, see the self-hosted path at the top of this document; the shipped
> engine is native-v2 (Madaros) per `docs/RELEASE_POLICY.md`.

Sounio uses Cargo feature flags to enable optional functionality:

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `jit` | Cranelift JIT compilation (default) | None |
| `llvm15` | LLVM 15 native codegen | LLVM 15, libzstd |
| `llvm16` | LLVM 16 native codegen | LLVM 16, libzstd |
| `llvm17` | LLVM 17 native codegen | LLVM 17, libzstd |
| `llvm-base` | Core LLVM support (auto-enabled by llvm1x) | LLVM, libzstd |
| `smt` | Z3 refinement type verification | Z3, cmake |
| `gpu` | GPU codegen (PTX, Metal, SPIR-V) | None (CUDA for execution) |
| `ontology` | Scientific ontology (15M+ terms) | None |
| `pkg` | Package manager | None |
| `lsp` | Language Server Protocol | None |
| `full` | All features | All dependencies |

### Common Feature Combinations

```bash
# Minimal (Cranelift JIT only)
cargo build --release

# Development (JIT + GPU codegen)
cargo build --release --features "jit,gpu"

# Production (LLVM + SMT + GPU)
cargo build --release --features "llvm17,smt,gpu"

# Full scientific (all features)
cargo build --release --features full

# Check compilation without building binary
cargo check --lib --features "jit,llvm17,gpu,ontology,pkg"
```

---

## Build Configurations

### Minimal Build (No External Dependencies)

```bash
cargo build --release
```

Uses Cranelift JIT. No LLVM, Z3, or CUDA required.

### Development Build

```bash
cargo build --features "jit,gpu,ontology"
```

Includes JIT, GPU codegen, and scientific ontology.

### Production Build

```bash
# Install all dependencies first (see above)
cargo build --release --features "llvm17,smt,gpu,ontology,pkg"
```

### Full Feature Build

```bash
cargo build --release --features full
```

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

### Error: `LLVM_SYS_170_PREFIX not set`

**Cause**: llvm-sys crate cannot find LLVM installation.

**Solution**:
```bash
# Find LLVM installation
llvm-config --prefix

# Set environment variable
export LLVM_SYS_170_PREFIX=$(llvm-config --prefix)
```

### Error: `z3.h not found`

**Cause**: Z3 development headers not installed.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install libz3-dev

# macOS
brew install z3
```

### Error: `cmake not found`

**Cause**: CMake required for z3-sys crate build.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake
```

### Tests Fail with `Import not found`

**Cause**: Stdlib path not configured.

**Solution**:
```bash
export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib
souc info                  # Verify configuration: the `stdlib:` line echoes the resolved path
```

### Build Fails with `feature X requires feature Y`

**Cause**: Incompatible feature combinations.

**Solution**: Check the feature dependency graph:
- `llvm15`, `llvm16`, `llvm17` are mutually exclusive
- `llvm-base` is auto-enabled by any `llvm1x` feature
- `full` enables everything

```bash
# Correct: only one LLVM version
cargo build --features llvm17

# Incorrect: multiple LLVM versions
cargo build --features "llvm15,llvm17"  # Error!
```

---

## Verifying Installation

After cloning, verify the prebuilt toolchain and run the test suite (no cargo):

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

# Quick smoke test
./bin/souc run examples/hello.sio

# Full .sio test suite
bash scripts/run_sio_test_suite.sh        # or: make test

# Single test by pattern
bash scripts/run_sio_test_suite.sh vancomycin --verbose
```

---

## Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `SOUNIO_STDLIB_PATH` | Stdlib location | `/home/user/sounio/stdlib` |
| `LLVM_SYS_150_PREFIX` | LLVM 15 location | `/usr/lib/llvm-15` |
| `LLVM_SYS_160_PREFIX` | LLVM 16 location | `/usr/lib/llvm-16` |
| `LLVM_SYS_170_PREFIX` | LLVM 17 location | `/usr/lib/llvm-17` |
| `Z3_SYS_Z3_HEADER` | Z3 header path (if non-standard) | `/opt/z3/include/z3.h` |
| `CUDA_PATH` | CUDA toolkit location | `/usr/local/cuda` |

---

## Next Steps

- [Getting Started](getting-started.md) — Write your first Sounio program
- [Programming Guide](programming.md) — Complete syntax reference
- [Feature Flags](../FEATURE_FLAGS.md) — Detailed feature documentation
- [Minimum Viable Sounio](MINIMUM_VIABLE_SOUNIO.md) — What works today

---

*Last updated: 2026-07-11 (Madaros v0.80.0). Quick Start reflects the prebuilt self-hosted toolchain; the Rust/LLVM dependency sections are retained as historical reference for the retired Rust build.*
