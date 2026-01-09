---
title: Getting Started with Sounio
description: Choose your learning path for Sounio, the epistemic computing language
reading_time: 3 minutes
---

# Getting Started with Sounio

Welcome to **Sounio**, a systems programming language for epistemic computing. Sounio is built on a radical premise: every value should carry not just data, but *knowledge of its own uncertainty*.

Whether you are a scientist seeking computational rigor, a programmer learning a new paradigm, or migrating from another language, this guide will help you get started.

## Choose Your Learning Path

### Path A: I am a scientist who wants to compute with uncertainty

You work with experimental data, simulations, or measurements. You know that every value has error bars, but your current tools force you to track uncertainty manually (or worse, ignore it). Sounio makes uncertainty a first-class citizen.

**Start here:**

1. [Installation](./installation.md) - Get Sounio running on your machine
2. [Your First Uncertainty](./your-first-uncertainty.md) - Learn how `Knowledge<T>` tracks uncertainty automatically
3. [Hello World](./hello-world.md) - Basic syntax and program structure

**Key concepts for you:**
- `Knowledge<T>` - Values that carry uncertainty and confidence
- Automatic GUM-compliant uncertainty propagation
- Confidence gates that control execution based on data quality
- Provenance tracking for reproducibility

---

### Path B: I am a programmer learning Sounio

You have experience with systems languages (C, C++, Rust) or scientific languages (Python, Julia, MATLAB). You want to understand Sounio's syntax and semantics.

**Start here:**

1. [Installation](./installation.md) - Get Sounio running on your machine
2. [Hello World](./hello-world.md) - Basic syntax, functions, and types
3. [Project Structure](./project-structure.md) - How Sounio projects are organized
4. [Your First Uncertainty](./your-first-uncertainty.md) - Sounio's unique feature

**Key concepts for you:**
- `let` for immutable bindings, `var` for mutable
- `&!T` for exclusive references (not `&mut` like Rust)
- Effect system: functions declare their side effects with `with`
- Linear and affine types for resource safety

---

### Path C: I am migrating from another language

You are coming from a specific language background and want to understand the differences.

#### From Rust

Sounio shares some syntax with Rust but is a distinct language:

| Rust | Sounio | Notes |
|------|--------|-------|
| `&mut T` | `&!T` | Exclusive/mutable reference |
| `let mut x` | `var x` | Mutable binding |
| `assert!()` | `assert()` | No macros in Sounio |
| `println!()` | `print()` | No macros in Sounio |
| `#[test]` | Test files in `tests/` | No attribute macros |

**Start here:** [Hello World](./hello-world.md), then [Your First Uncertainty](./your-first-uncertainty.md)

#### From Python/Julia

You are used to scientific computing with NumPy, SciPy, or Julia's ecosystem:

| Python/Julia | Sounio | Notes |
|--------------|--------|-------|
| `x = 5` | `let x = 5` | Immutable by default |
| Numpy arrays | `[T; N]` or `Vec<T>` | Typed arrays |
| `uncertainties` package | `Knowledge<T>` | Built into the type system |
| `import numpy` | `import stdlib.linalg` | Module imports |

**Start here:** [Installation](./installation.md), then [Your First Uncertainty](./your-first-uncertainty.md)

#### From C/C++

You want low-level control with high-level safety:

| C/C++ | Sounio | Notes |
|-------|--------|-------|
| `int* p` | `&!i32` or `*mut i32` | References vs raw pointers |
| `malloc`/`free` | Linear types | Resources freed automatically |
| Header files | Modules | `import` and `module` keywords |
| `#define` | `const` | Compile-time constants |

**Start here:** [Hello World](./hello-world.md), then [Project Structure](./project-structure.md)

---

## Quick Reference

### Essential Commands

```bash
# Build the compiler from source
cd compiler && cargo build --release

# Check a Sounio file (type-check without running)
souc check your_file.sio

# Run a Sounio program
souc run your_file.sio

# Start the interactive REPL
souc repl

# Show AST and types (debugging)
souc check your_file.sio --show-ast --show-types
```

### File Extensions

- `.sio` - Sounio source files (primary)
- `sounio.toml` - Project manifest file

### Documentation

- [Language Reference](../LLM_PROGRAMMING_GUIDE.md) - Complete syntax guide
- [Standard Library](../../stdlib/) - Browse the 151,000+ line stdlib
- [Examples](../../examples/) - Working code examples
- [Manifesto](../../MANIFESTO.md) - Design philosophy

---

## Next Steps

Pick the first tutorial from your learning path above. If you are unsure, start with [Installation](./installation.md) followed by [Your First Uncertainty](./your-first-uncertainty.md) - it demonstrates what makes Sounio unique.

---

*At the horizon of certainty, where ancient columns meet the endless sea.*

**SOUNIO** - Compute at the Horizon of Certainty
