<!-- docs:meta
topic_id: repo.docs.faq
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.faq
-->

# Frequently Asked Questions

Common questions about Sounio and epistemic computing.

> **Canonical epistemic API.** The checked public surface for epistemic values
> is `epistemic::knowledge` (free-fn form: `ep_measured`, `ep_val`, `ep_std`,
> `ep_add`, `ep_mul`, `ep_div`, `ep_merge`, `ep_is_credible`, `ep_*_cov` for
> covariance-aware GUM), anchored by
> `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio` and
> `tests/run-pass/ep_gum_covariance.sio`. The legacy
> `epistemic::lib` surface (`epistemic_std`, `add_epistemic`, `mul_epistemic`,
> `fuse_measurements`) is exercised by
> `tests/stdlib/epistemic/test_core_e2e.sio` (a `//@ run-pass` test collected by
> `scripts/dev/run_sio_test_suite.sh`) but is a legacy surface separate from the
> canonical `epistemic::knowledge` API. `with_confidence` operators and
> units-as-type-parameters are aspirational in source and absent from the
> checked public surface — see `docs/compiler/KNOWN_LIMITATIONS.md`.

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
- `Epistemic` (stdlib `epistemic::knowledge`) tracks point estimate, variance, and integer confidence 0–1000; free-fn arithmetic (`ep_add`, `ep_mul`, ...) propagates variance via GUM delta method
- Effect system tracks computational side effects
- Units of measure prevent dimensional errors
- First-class support for scientific workflows

### Is Sounio production-ready?

**Current checked-artifact status (`souc 1.0.0-beta.4`)**: suitable for
research, evaluation, and artifact-backed workflows, but still not "everything
in the source tree is equally public".
- ✅ Core language and science lanes are artifact-backed
- ✅ Checked GPU compiler profile exists
- 🚧 No JIT tier — Cranelift is not compiled in any checked artifact (measured 2026-08-27)
- ✅ The repo-wide checkpoint now includes self-hosted render/bootstrap proofs and the skills/dispatch wave
- 🚧 Advanced or alternate-build features such as LLVM, SMT, and LSP remain profile-dependent
- 📋 Public docs should still track the exact artifact and gate behind each claim

See [Current State](../README.md#current-state) for details.

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

Partially. Local package support is checked: `tools/sounio-pkg/sounio-pkg`
can create, build, check, and test local packages, and the compiler has a
gated local package-import path for packages under `packages/*`.

There is no launched public package registry yet. Publishing, login, hosted
search, and broad dependency-resolution workflows remain design/prototype
surfaces.

---

## Epistemic Computing

### What is "epistemic computing"?

Computing that explicitly represents **what we know** and **how well we know it**. The shipped `Epistemic` value carries a point estimate, its variance, and an integer confidence score (see the Epistemic Types section). The broader paradigm also tracks provenance (where data came from); the current `Epistemic` struct does not expose a provenance field — `stdlib/epistemic/prov.sio` is a separate, private model (no `pub` symbols).

### Do I have to use Epistemic everywhere?

No! Use it where uncertainty matters:
```sio
use epistemic::knowledge::{ep_measured}

// Computational geometry - exact
let angle = 90.0  // degrees

// Scientific measurement - uncertain (canonical free-fn form)
let temperature = ep_measured(37.2, 0.1)
// Note: ep_measured stores variance = std^2; Epistemic has no provenance field,
// and provenance is NOT part of the checked public surface — stdlib/epistemic/prov.sio
// is a private/internal PROV model (no `pub` symbols) and cannot be imported as a
// user-facing module.
```

### How is uncertainty propagated?

Automatically, using the **GUM** (Guide to Uncertainty in Measurement) standard:
```sio
use epistemic::knowledge::{Epistemic, ep_add, ep_mul}

// Constructor (struct-literal form) — variance = std^2, confidence integer 0..1000.
let x = Epistemic { val: 10.0, variance: 0.25, confidence: 900 }
let y = Epistemic { val:  5.0, variance: 0.04, confidence: 900 }

// GUM delta method, uncorrelated: Var(X+Y) = Var(X) + Var(Y).
let sum = ep_add(&x, &y)
// Multiplication uses partial derivatives via ep_mul.
let product = ep_mul(&x, &y)
// Numerical check: Var(sum) = 0.29, Var(product) computed by ep_mul.
```

### What if I don't know the uncertainty?

Be explicit about it:
```sio
use epistemic::knowledge::{Epistemic}

// Unknown uncertainty - encode as high-variance, low-confidence struct literal.
let guess = Epistemic { val: 42.0, variance: 1.0e6, confidence: 500 }

// Or use raw values when uncertainty truly does not matter.
let count: i32 = 5  // Counting objects - no uncertainty
```

### Can I compare Epistemic values?

Yes — compare the point values directly, and gate on confidence separately:
```sio
use epistemic::knowledge::{Epistemic, ep_val, ep_is_credible}

let a = Epistemic { val: 10.0, variance: 1.0, confidence: 900 }
let b = Epistemic { val: 12.0, variance: 1.0, confidence: 900 }

// Deterministic comparison (point values).
if ep_val(&a) < ep_val(&b) { }

// Combined credibility gate: require both values to carry at least 0.95
// (950/1000) confidence. ep_is_credible compares each value's integer
// confidence score against the threshold; it does not compute the probability
// that a < b.
if ep_is_credible(&a, 950) && ep_is_credible(&b, 950) {
    // proceed with the comparison as a measured assertion
}
```

> **Note.** `if a < b with_confidence 0.95 { }` is **not** part of the checked
> surface. The canonical equivalent uses `ep_is_credible(&e, 950)` (boolean)
> and `ep_val(&e)` for the point comparison. See `tests/stdlib/epistemic/test_knowledge_stdlib.sio`.

---

## Tooling

### How do I install Sounio?

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio
export SOUC_BIN="$(pwd)/bin/souc"
"$SOUC_BIN" info
"$SOUC_BIN" check examples/hello.sio
```

### Does the self-hosted compiler work today?

Yes. `self-hosted/compiler/main.sio` is the authoritative repo-checkpoint
driver, and the stabilized contributor-facing modes are `--check`,
`--ir-dump`, `--ir-roundtrip`, and `--native-compile`.

The current gates use that driver to validate all 7 render fixtures and the
`triangle_basic.sio` bootstrap-native render proof.

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
- Fast iteration via the checked `bin/souc` artifact — AOT only, no JIT tier (measured 2026-08-27)
- GPU acceleration for parallel workloads

### Does `Epistemic` have runtime overhead?

Minimal:
- `Epistemic` is `val: f64` + `variance: f64` + `confidence: i64` (24 bytes total). See `stdlib/epistemic/knowledge.sio`.
- Propagation adds a small, operation-dependent overhead per operation: relative to a raw `f64`, `Epistemic` carries one extra `f64` variance field and one `i64` confidence field, plus a handful of GUM δ-method arithmetic ops; the exact cost depends on the operation mix and the compiler path.
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
let fast_computation: f64 = ...  // No Epistemic wrapper
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

Yes, via PyO3 bindings (experimental — not part of the checked public artifact):
```python
import sounio

# Experimental PyO3 binding shape: sounio.Knowledge(value, uncertainty, confidence, unit, prov),
# where uncertainty is the standard uncertainty (k=1, sigma) and confidence is an f64 in [0,1]
# (NOT the stdlib's integer 0..1000 score). This is a standalone prototype API, not the
# canonical stdlib surface (which uses integer confidence 0..1000 and variance = sigma**2).
e1 = sounio.Knowledge(10.0, uncertainty=0.5, confidence=0.95)
e2 = sounio.Knowledge(5.0,  uncertainty=0.2, confidence=0.95)
print(f"Result: {e1.value} ± {e1.uncertainty} (conf {e1.confidence})")
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
5. Run fast gate: `./scripts/dev/fast_gate.sh`
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

The checked public artifacts currently report SMT as disabled. Treat SMT as a
rebuild-only capability family and confirm it with `souc info` on the exact
binary you built before documenting it as available.

---

*Still have questions? Ask on [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)!*
