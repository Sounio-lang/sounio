<!-- docs:meta
topic_id: repo.docs.getting-started
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.getting-started
-->

# Getting Started with Sounio

Sounio is an L0 systems and scientific programming language for epistemic computing. It features algebraic effects, linear types, units of measure, epistemic uncertainty propagation, and 16-dimensional hypercomplex (sedenion) algebra.

## Build from Source

```bash
git clone https://github.com/Chiuratto-AI/sounio.git
cd sounio

# Build the compiler (release mode recommended)
cargo build -p souc --release

# The binary is at target/release/souc
# Optionally add to your PATH:
export PATH="$PWD/target/release:$PATH"
```

### Feature Flags

The default build includes the Cranelift JIT backend. Optional features:

| Flag | Requires | What it adds |
|------|----------|-------------|
| `jit` | (included) | Cranelift-based JIT execution |
| `llvm` | LLVM 15+ | LLVM codegen with O0-O3 optimization |
| `smt` | Z3 + cmake | Compile-time refinement type proofs |
| `gpu` | CUDA toolkit | GPU kernel execution (PTX/Metal/SPIR-V) |
| `lsp` | (included) | Language server for editor integration |
| `full` | All above | Everything |

```bash
# Build with all optional features
cargo build -p souc --release --features full
```

## Hello World

Create `hello.sio`:

```sio
fn main() with IO {
    println("Hello, Sounio!")
}
```

Run it:

```bash
souc run hello.sio
```

Type-check without running:

```bash
souc check hello.sio
```

## Key Syntax

Sounio has its own syntax. It is **not** Rust.

```sio
// Immutable binding
let x = 42

// Mutable binding (NOT 'let mut')
var y = 10
y = y + 1

// Functions declare their effects
fn compute(a: f64, b: f64) -> f64 with Mut, Div, Panic {
    var result = a / b
    result
}

// Structs
struct Point { x: f64, y: f64 }
let p = Point { x: 1.0, y: 2.0 }

// Arrays
let nums = [1, 2, 3, 4, 5]
let first = nums[0]

// Linear types (must be consumed exactly once)
linear struct FileHandle { fd: i32 }

// Exclusive references use &! (NOT &mut)
fn modify(r: &!i32) with Mut { }
```

### Effects

Every side effect must be declared in the function signature:

| Effect | Triggered by |
|--------|-------------|
| `IO` | `println`, `print`, file I/O |
| `Mut` | `var` bindings, mutation |
| `Div` | Division operator (`/`) |
| `Panic` | Division, `as` casts, array indexing |
| `Alloc` | Heap allocation |
| `Async` | Async operations |
| `GPU` | GPU kernel launch |
| `Prob` | Probabilistic operations |

Effects propagate: if you call a function `with Div, Panic`, your function must also declare those effects.

```sio
fn sqrt(x: f64) -> f64 with Mut, Div, Panic {
    if x <= 0.0 { return 0.0 }
    var g = x / 2.0
    var i = 0
    while i < 60 { g = (g + x / g) / 2.0; i = i + 1 }
    g
}

fn main() with IO, Mut, Div, Panic {
    println(sqrt(2.0))
}
```

## The REPL

```bash
souc repl
```

The REPL supports epistemic badges, 17 commands, and JIT execution. Type `:help` for available commands.

## Explore the Showcase

Ten self-contained examples demonstrate Sounio's capabilities:

```bash
# Pharmacokinetic drug dose modeling with uncertainty
souc run examples/showcase/drug_dose_optimizer.sio

# DNA motif scanning with epistemic confidence
souc run examples/showcase/genome_motif_scanner.sio

# 16D sedenion knowledge graph embeddings
souc run examples/showcase/knowledge_graph_trainer.sio

# Linear types for resource safety
souc run examples/showcase/linear_file_server.sio

# ISO GUM uncertainty propagation
souc run examples/showcase/measurement_lab.sio
```

See [examples/showcase/README.md](../examples/showcase/README.md) for the full list.

## Format Your Code

```bash
# Format a file
souc fmt myfile.sio

# Check formatting without modifying
souc fmt --check myfile.sio

# Show diff
souc fmt --diff myfile.sio

# Format all files in a directory
souc fmt src/
```

## Editor Support

### VSCode

Install the Sounio extension:

```bash
code --install-extension editors/vscode/sounio-*.vsix
```

Features: syntax highlighting, semantic tokens, hover documentation, go-to-definition, completion, and diagnostic reporting via the built-in LSP.

## Next Steps

- Read the [showcase README](../examples/showcase/README.md) for example descriptions
- Browse `tests/run-pass/` for more language feature examples
- See `docs/compiler/KNOWN_LIMITATIONS.md` for current status of advanced features
- Check `docs/STYLE_GUIDE.md` for coding conventions
