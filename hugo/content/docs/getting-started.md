---
title: "Getting Started"
description: "Quick start guide for Sounio - install, write your first program, and explore epistemic computing."
---

# Getting Started with Sounio

## Installation

### Prerequisites

- **Rust 1.70+** (for the compiler)
- **LLVM 16+** (optional, for LLVM backend)
- **CUDA Toolkit 12+** (optional, for NVIDIA GPU support)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Sounio-lang/sounio.git
cd sounio

# Build the compiler
cd compiler
cargo build --release

# Install the sounio binary
cargo install --path .

# Verify installation
souc --version
```

### Pre-built Binaries

Download from the [releases page](https://github.com/Sounio-lang/sounio/releases/latest):

```bash
# Linux (x86-64)
wget https://github.com/Sounio-lang/sounio/releases/latest/download/sounio-x86_64-linux.tar.gz
tar -xzf sounio-x86_64-linux.tar.gz
./sounio-installer

# macOS (Apple Silicon)
wget https://github.com/Sounio-lang/sounio/releases/latest/download/sounio-aarch64-darwin.tar.gz
tar -xzf sounio-aarch64-darwin.tar.gz
./sounio-installer
```

## Your First Sounio Program

Create a file named `hello.sio`:

```sio
// Simple hello world with uncertainty
fn main() {
    let message: Knowledge<string> = measure("Hello, Sounio!", uncertainty: 0.0)
    print(message.value)
}
```

Run it:

```bash
souc hello.sio
output: Hello, Sounio!
```

## Epistemic Computing Basics

### Knowledge Type

Every measurement knows its uncertainty:

```sio
let temperature: Knowledge<kelvin> = measure(298.15, uncertainty: 0.1)
let pressure: Knowledge<pascal> = measure(101325.0, uncertainty: 5.0)

// Uncertainty propagates automatically
let ideal_gas = temperature * pressure
// Result includes combined uncertainty
```

### Type-Safe Units

Prevent unit errors at compile time:

```sio
let mass: kg = 500.0
let volume: m³ = 2.0
let density: kg/m³ = mass / volume  // Correct!

// This won't compile:
// let bad: kg = volume  // Error: can't assign m³ to kg
```

### GPU Acceleration

Simple GPU kernel syntax:

```sio
kernel fn vector_add(
    a: &[f64],
    b: &[f64], 
    c: &![f64]
) {
    let i = gpu.thread_id.x
    c[i] = a[i] + b[i]
}
```

## Learning Paths

Choose your path based on your background:

### [Scientific Computing](/#learning-paths)
For researchers and scientists working with measurements and uncertainty

### [Machine Learning](/#learning-paths)  
For ML engineers interested in octonion neural networks and GPU acceleration

### [Systems Programming](/#learning-paths)
For compiler and systems engineers wanting to understand the internals

### [Domain Applications](/#learning-paths)
For pharma, physics, climate, and finance professionals

## Next Steps

1. **Try the [Playground](/playground)** - No installation required
2. **Read the [Language Guide](/docs/language/)** - Comprehensive documentation
3. **Explore [Examples](/examples)** - Real-world use cases
4. **Check [Architecture](/architecture/)** - Understand the internals

## Help & Community

- **GitHub Issues** - Report bugs and request features
- **Discord** - Join the community chat
- **Documentation** - [API Reference](/docs/api/), [Standard Library](/docs/stdlib/)
- **Research** - [Technical Reports](/research/), [Papers](/docs/papers/)
