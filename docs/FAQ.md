# Frequently Asked Questions

Common questions about Sounio and epistemic computing.

## Table of Contents

- [General](#general)
- [Language Features](#language-features)
- [Epistemic Computing](#epistemic-computing)
- [Tooling](#tooling)
- [Performance](#performance)
- [Interoperability](#interoperability)
- [Contributing](#contributing)

---

## General

### What is Sounio?

Sounio is a systems programming language designed for **epistemic computing**—computation that explicitly tracks uncertainty, confidence, and provenance. It's particularly suited for scientific computing, where knowing *how confident* you are in a result is as important as the result itself.

### Why create a new language?

Existing languages treat uncertainty as an afterthought (manual error propagation, separate libraries). Sounio makes epistemic integrity **foundational**:
- `Knowledge<T>` types track uncertainty automatically
- Effect system tracks computational side effects
- Units of measure prevent dimensional errors
- First-class support for scientific workflows

### Is Sounio production-ready?

**Current status (v0.99.0)**: Pre-1.0, suitable for research and evaluation.
- ✅ Core language features stable
- ✅ 151K+ lines standard library
- 🚧 Some advanced features experimental (LLVM backend, LSP)
- 📋 API may change before 1.0

See [Project Status](../README.md#project-status) for details.

### How does Sounio compare to Rust?

| Feature | Sounio | Rust |
|---------|--------|------|
| Memory safety | ✅ Yes | ✅ Yes |
| Ownership system | Linear/affine types | Borrow checker |
| Mutability syntax | `var x` | `let mut x` |
| Reference syntax | `&!T` (exclusive) | `&mut T` |
| Epistemic types | ✅ Built-in | ❌ Manual libraries |
| Effect system | ✅ Algebraic effects | ❌ (traits only) |
| Units of measure | ✅ First-class | ❌ (F# has it) |
| Macros | 🚧 Planned | ✅ Proc macros |
| Ecosystem | 🌱 Growing | 🌳 Mature |

**When to use Sounio**: Scientific computing, uncertainty quantification, PK/PD modeling
**When to use Rust**: General systems programming, mature ecosystem needed

---

## Language Features

### Why `var` instead of `let mut`?

Sounio emphasizes immutability-by-default more strongly:
```sio
let x = 10      // Immutable (like Rust's `let`)
var y = 20      // Mutable (Rust's `let mut`)
```

`var` is shorter and more distinct, making mutable variables visually obvious.

### Why `&!T` instead of `&mut T`?

Consistency and clarity:
- `&T` - shared (immutable) reference
- `&!T` - exclusive (mutable) reference

The `!` visually indicates "caution: mutation happening."

### Does Sounio have macros?

Not yet. Planned for post-1.0. Design goals:
- Hygienic (like Rust)
- Syntax-aware (not text substitution)
- Support for DSL embedding (like MedLang)

### Can I use async/await?

Yes! Async is part of the effect system:
```sio
fn fetch_data(url: string) -> Result<Data> with Async {
    let response = http.get(url).await
    parse(response)
}
```

### Does Sounio have a package manager?

Planned (`siopkg`), currently in design phase. For now:
- Use Git submodules
- Copy modules into your project
- Path-based imports

---

## Epistemic Computing

### What is "epistemic computing"?

Computing that explicitly represents **what we know** and **how well we know it**. Every value carries:
- The value itself
- Uncertainty (measurement error)
- Confidence level (statistical confidence)
- Provenance (where it came from)

### Do I have to use `Knowledge<T>` everywhere?

No! Use it where uncertainty matters:
```sio
// Computational geometry - exact
let angle = 90.0  // degrees

// Scientific measurement - uncertain
let temperature = Knowledge::new(
    value: 37.2,
    uncertainty: 0.1,
    source: "thermometer_A"
)
```

### How is uncertainty propagated?

Automatically, using the **GUM** (Guide to Uncertainty in Measurement) standard:
```sio
let x = Knowledge::new(10.0, uncertainty: 0.5)
let y = Knowledge::new(5.0, uncertainty: 0.2)

// Addition: σ_sum = sqrt(σ_x² + σ_y²)
let sum = x + y  // uncertainty: sqrt(0.5² + 0.2²) = 0.539

// Multiplication uses partial derivatives
let product = x * y  // GUM formula applied automatically
```

### What if I don't know the uncertainty?

Be explicit about it:
```sio
// Unknown uncertainty - use conservative estimate
let guess = Knowledge::new(
    value: 42.0,
    uncertainty: LARGE_VALUE,
    confidence: 0.50,  // Low confidence
    source: "rough_estimate"
)

// Or use raw values when uncertainty truly doesn't matter
let count: i32 = 5  // Counting objects - no uncertainty
```

### Can I compare `Knowledge<T>` values?

Yes, but comparisons are probabilistic:
```sio
let a = Knowledge::new(10.0, uncertainty: 1.0)
let b = Knowledge::new(12.0, uncertainty: 1.0)

// Deterministic comparison (point values)
if a.value < b.value { }

// Probabilistic comparison (accounts for uncertainty)
if a < b with_confidence 0.95 { }
```

---

## Tooling

### How do I install Sounio?

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio
cargo build -p souc --release

# Add to PATH
export PATH=$PATH:$(pwd)/target/release
```

### Is there IDE support?

Work in progress:
- **LSP** (Language Server Protocol): 80% complete
- **VS Code extension**: Basic support
- **Vim/Emacs**: Syntax highlighting available

### How do I debug Sounio programs?

```bash
# Print debugging
print("debug: ", value)

# Compiler debugging
souc check --show-types file.sio
souc check --show-ast file.sio

# GDB/LLDB support (when using native backend)
souc build --debug file.sio
gdb ./file
```

### Can I use Sounio in Jupyter notebooks?

Yes! See `/home/demetrios/sounio-1/jupyter/` for setup.

---

## Performance

### Is Sounio fast?

Yes. Performance comparable to Rust/C++:
- Native code generation (ELF/Mach-O)
- LLVM backend (experimental)
- Cranelift JIT for fast iteration
- GPU acceleration for parallel workloads

### Does `Knowledge<T>` have runtime overhead?

Minimal:
- `Knowledge<f64>` ≈ 3 additional f64 fields (value, uncertainty, confidence)
- Propagation adds ~2-5% overhead vs raw arithmetic
- GPU kernels can vectorize uncertainty calculations

### When should I use GPU acceleration?

For data-parallel operations:
- Matrix operations (> 1000×1000)
- FFT on large signals (> 10,000 points)
- Monte Carlo simulations
- ODE systems with many particles

```sio
// Automatic GPU dispatch for large arrays
let large_matrix = Matrix::new(5000, 5000)
let result = large_matrix.multiply(other)  // Runs on GPU if available
```

### Can I disable uncertainty propagation for performance?

Yes, compile with optimization flags:
```bash
souc build --release --unsafe-fast-math file.sio
```

Or use raw types where uncertainty doesn't matter:
```sio
let fast_computation: f64 = ...  // No Knowledge<> wrapper
```

---

## Interoperability

### Can I call C/C++ code from Sounio?

Yes, via Foreign Function Interface (FFI):
```sio
extern "C" {
    fn c_function(x: i32) -> f64
}

fn main() {
    let result = unsafe { c_function(42) }
}
```

### Can I call Sounio from Python?

Yes, via PyO3 bindings (experimental):
```python
import sounio

knowledge = sounio.Knowledge(value=10.0, uncertainty=0.5)
result = knowledge + sounio.Knowledge(5.0, 0.2)
print(f"Result: {result.value} ± {result.uncertainty}")
```

### Can I use Rust crates?

Not directly (yet). Planned for post-1.0:
- FFI to compiled Rust libraries
- Automatic binding generation
- Cargo integration for dependencies

### Can Sounio target WebAssembly?

Experimental support:
```bash
souc build --target wasm32 file.sio
```

Current limitations:
- No threading
- Limited stdlib support
- File I/O not available

---

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md). Quick start:
- 🐛 **Bug reports**: Open an issue
- 📝 **Documentation**: Always welcome
- 🧪 **Tests**: Add test cases
- ⚡ **Performance**: Benchmark & optimize
- 🎨 **Examples**: Share your code

### Where should I ask questions?

- **GitHub Discussions**: General questions
- **GitHub Issues**: Bug reports, feature requests
- **Discord**: Real-time chat (link in repo)

### What's the contribution workflow?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `cargo test --workspace`
5. Run fast gate: `./scripts/fast_gate.sh`
6. Submit pull request

### Do you accept papers/citations?

Yes! Academic contributions welcome:
- Algorithm implementations
- Formal verification
- Performance analysis
- Domain-specific applications

---

## Advanced Questions

### How does the effect system work?

Algebraic effects track side effects in types:
```sio
fn pure_function(x: i32) -> i32 {
    x * 2  // No effects
}

fn impure_function() -> string with IO {
    fs.read_to_string("file.txt")  // IO effect
}
```

Effects compose and propagate through the call graph automatically.

### What's the type inference algorithm?

**Bidirectional type inference** (similar to OCaml/Haskell):
- Synthesis: infer types bottom-up
- Checking: verify against expected types top-down
- Local type inference (not global like Hindley-Milner)

### Can I write custom effects?

Yes (advanced feature):
```sio
effect State<S> {
    fn get() -> S
    fn put(s: S) -> ()
}

fn stateful_computation() -> i32 with State<i32> {
    let x = do State.get()
    do State.put(x + 1)
    x
}
```

### Are refinement types fully supported?

Experimental (requires SMT solver):
```sio
type Positive = { x: i32 | x > 0 }

fn sqrt(x: Positive) -> f64 {
    // Compile-time verification via Z3
    math.sqrt(x as f64)
}
```

Enable with: `cargo build -p souc --features smt`

---

*Still have questions? Ask on [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)!*
