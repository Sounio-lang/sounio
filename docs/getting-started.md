# Getting Started with Sounio

Welcome to **Sounio**, a systems programming language for epistemic computing — where every value can carry its uncertainty.

## Installation

### From Binary (Recommended)

Download the latest release for your platform:

```bash
# Linux/macOS
curl -sSf https://souniolang.org/install.sh | sh

# Or download directly
wget https://github.com/sounio-lang/sounio/releases/latest/download/souc-linux-x64.tar.gz
tar xzf souc-linux-x64.tar.gz
sudo mv souc /usr/local/bin/
```

### From Source

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio/compiler
cargo build --release
sudo cp target/release/souc /usr/local/bin/
```

### Verify Installation

```bash
souc --version
# souc 0.93.0
```

## Your First Program

Create a file `hello.sio`:

```sounio
fn main() -> i32 {
    print("Hello, Sounio!")
    println()
    0
}
```

Compile and run:

```bash
souc run hello.sio
# Output: Hello, Sounio!
```

Or just check types:

```bash
souc check hello.sio
```

## Key Concepts

### 1. Epistemic Types

Sounio's signature feature is the `Knowledge<T>` type — values that carry their uncertainty:

```sounio
import sounio::epistemic::*

fn main() -> i32 {
    // Value with uncertainty
    let measurement = Knowledge::new(
        value: 42.0,
        uncertainty: 0.5,
        confidence: 0.95
    )

    // Uncertainty propagates through operations
    let doubled = measurement.mul(Knowledge::exact(2.0))

    print(doubled.to_string())
    // Output: 84.0000 +/- 1.9600 (95% CI)

    0
}
```

### 2. Variables

```sounio
let x = 5              // immutable
var y = 10             // mutable

y = y + 1              // OK: y is mutable
// x = 6               // Error: x is immutable
```

### 3. References

Sounio uses `&!` for mutable references (not `&mut` like Rust):

```sounio
fn increment(x: &!i32) {
    *x = *x + 1
}

fn main() -> i32 {
    var value = 10
    increment(&!value)
    print(value)  // 11
    0
}
```

### 4. Physical Units

Type-safe dimensional analysis:

```sounio
let distance: f64<m> = 100.0 m
let time: f64<s> = 9.58 s
let speed = distance / time  // Type: f64<m/s>

// Compile error: can't add meters and seconds
// let invalid = distance + time
```

### 5. Effects

Functions declare their side effects:

```sounio
fn read_file(path: &str) -> String with IO {
    // Can perform I/O
}

fn pure_function(x: i32) -> i32 {
    // No effects allowed
    x * 2
}
```

### 6. MedLang DSL

Domain-specific syntax for pharmacometrics:

```sounio
import sounio::medlang::*

model OneCompartment {
    param CL: Knowledge<f64> = Knowledge::new(
        value: 10.0,
        uncertainty: 3.0,
        confidence: 0.95
    )
    param V: Knowledge<f64> = Knowledge::new(
        value: 50.0,
        uncertainty: 12.5,
        confidence: 0.95
    )

    compartment Central { volume: V }
    flow Central -> Elimination: CL

    observe Cp = Central.concentration
}
```

## Project Structure

A typical Sounio project:

```
my_project/
├── src/
│   ├── main.sio
│   └── lib.sio
├── tests/
│   └── test_main.sio
├── examples/
│   └── demo.sio
└── sounio.toml
```

## Command Reference

```bash
# Type-check a file
souc check file.sio

# Run a file (JIT compilation)
souc run file.sio

# Compile to executable
souc build file.sio -o output

# Show AST
souc check file.sio --show-ast

# Show types
souc check file.sio --show-types

# Watch mode (recompile on changes)
souc watch file.sio

# Get help
souc --help
```

## Examples

The `examples/` directory contains many working examples:

| File | Description |
|------|-------------|
| `hello.sio` | Hello World |
| `fibonacci.sio` | Recursive and iterative Fibonacci |
| `uncertainty.sio` | Knowledge<T> uncertainty propagation |
| `pkpd.sio` | Two-compartment PK model |
| `effects.sio` | Algebraic effects demo |
| `gpu.sio` | GPU kernel example |
| `ode_demo.sio` | ODE solving |
| `autodiff.sio` | Automatic differentiation |

Run any example:

```bash
cd examples
souc run hello.sio
souc run fibonacci.sio
souc run uncertainty.sio
```

## Next Steps

- [Language Reference](./LLM_PROGRAMMING_GUIDE.md) — Complete syntax guide
- [Standard Library](../stdlib/) — Browse the stdlib
- [Examples](../examples/) — Working code examples
- [CHANGELOG](../CHANGELOG.md) — Version history

## Getting Help

- **GitHub Issues**: [sounio-lang/sounio](https://github.com/sounio-lang/sounio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)
- **Website**: [souniolang.org](https://souniolang.org)

---

🏛️ **Sounio** — Compute at the Horizon of Certainty
