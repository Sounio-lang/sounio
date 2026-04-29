# Sounio

A systems programming language for epistemic computing — scientific code that knows its own uncertainty.

## Quick Links

- [Getting Started](getting-started.md) — install and run your first program
- [Tutorial](tutorial.md) — step-by-step learning guide
- [Cookbook](cookbook.md) — task-oriented recipes
- [Language Specification](spec.md) — formal syntax and semantics

## What Makes Sounio Different?

- **Epistemic types** (`Knowledge<T>`) track uncertainty automatically
- **Algebraic effects** make side effects explicit in the type system
- **Units of measure** prevent dimensional errors at compile time
- **Self-hosted compiler** written in Sounio itself
- **GPU backend** with PTX emission

```sio
fn main() with IO {
    println("Hello, Sounio!")
}
```
