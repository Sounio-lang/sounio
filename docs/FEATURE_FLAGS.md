<!-- docs:meta
topic_id: repo.docs.feature-flags
authority: repo_only
audience: users
last_validated: 2026-03-10
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.feature-flags
-->

# Sounio Feature Flags Reference

This page now describes the real capability surface users can rely on in this
repository. The public compiler contract is artifact-based, not "run one root
Cargo build and inherit every feature flag from the README".

## 1. Public profiles you can verify today

The checked compiler artifacts live under `artifacts/omega/souc-bin/`.

| Profile | Artifact | What `souc info` proves |
|---------|----------|-------------------------|
| Default JIT profile | `souc-linux-x86_64-jit` | Cranelift JIT enabled; LLVM and GPU codegen disabled |
| GPU profile | `souc-linux-x86_64-gpu` | GPU codegen enabled; Cranelift JIT disabled; PTX emission via `build --backend gpu` |

Recommended verification:

```bash
./artifacts/omega/souc-bin/souc-linux-x86_64-jit info
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu info
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu build examples/gpu.sio --backend gpu -o /tmp/sounio-gpu.ptx
```

## 2. What "feature flags" mean now

For the main public compiler workflow, feature names such as `jit`, `llvm`,
`gpu`, `smt`, `lsp`, `ontology`, `distributed`, and `pkg` are best understood
as rebuild-only capability families reported by `souc info`.

That means:

- they are real compiler capability groups
- they are not exposed through a single root-level `Cargo.toml` in this checkout
- they must be confirmed on the exact binary you are documenting

## 3. Source-build guidance

If you are rebuilding internal components or historical Rust subtrees:

- use the local manifest that actually exists in that subtree
- treat its features as component-local, not as the public Sounio compiler contract
- verify the rebuilt compiler with `souc info` before documenting its behavior

This repository currently has Rust manifests for subcomponents such as:

- `bootstrap/poseidon/rust/Cargo.toml`
- `tests/jit/Cargo.toml`

Those are not a substitute for the public compiler artifact contract.

## 4. GPU-specific rule

For public GPU documentation, use the checked GPU profile and the public CLI
path:

```bash
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu build examples/gpu.sio --backend gpu -o /tmp/sounio-gpu.ptx
```

Do not describe top-level `gpu-emit` as a public checked command, and do not
describe older `gpu.*` intrinsic-heavy examples as if they already passed in the
checked public artifact.

## 5. Documentation rule of thumb

- Cite the exact artifact or binary you tested.
- Use `souc info` as the first proof point.
- Treat source-tree presence as implementation evidence, not as proof that the checked public binary exposes the same feature.

### Scientific Ontology (`ontology`)

Access to 15M+ scientific terms from BioPortal, ChEBI, GO, etc.

```bash
cargo build --features ontology
```

**Enables:**
- `rusqlite` - SQLite backend for term storage
- Ontology querying and validation

### Ontology Build Tools (`ontology-build`)

Parse OWL/RDF files to build ontology databases.

```bash
cargo build --features "ontology,ontology-build"
cargo run --bin sounio-ontology-build
```

**Enables:**
- `rio_turtle` - Turtle/N-Triples parsing
- `rio_xml` - RDF/XML parsing
- `rio_api` - RDF API

### Package Manager (`pkg`)

Full package manager with HTTP registry support.

```bash
cargo build --features pkg
```

**Enables:**
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `tar` - Archive handling
- `async-compression` - Gzip compression
- `futures` - Async utilities
- `base64` - Encoding

### Distributed Builds (`distributed`)

Remote compilation and caching infrastructure.

```bash
cargo build --features distributed
```

**Enables:**
- `axum` - Web framework
- `tower` - Service abstractions
- `tower-http` - HTTP utilities
- `hyper` - HTTP implementation

### WebAssembly (`wasm`)

Build for browser-based playground.

```bash
cargo build --target wasm32-unknown-unknown --features wasm
```

**Enables:**
- `wasm-bindgen` - JS interop
- `js-sys` - JS system bindings
- `web-sys` - Web API bindings
- `console_error_panic_hook` - Debug support

### LLM Integration (`llm`, `glm`)

LLM-assisted features for code generation and optimization.

```bash
cargo build --features llm
cargo build --features glm  # GLM-4.7 specific
```

---

## Build Profiles

### Debug Profile

```bash
cargo build
# or
cargo build --profile dev
```

**Settings:**
- `opt-level = 0` - No optimization
- `debug = true` - Full debug info
- Fast compile times

### Release Profile

```bash
cargo build --release
```

**Settings:**
- `lto = true` - Link-time optimization
- `codegen-units = 1` - Single codegen unit for better optimization
- `panic = "abort"` - Smaller binary, no unwinding

---

## Common Configurations

### Minimal (Parsing only)

```bash
cargo build --release
```

No code generation, just parsing and type checking.

### Development

```bash
cargo build --features jit
```

Fast compilation with JIT execution for quick iteration.

### Production

```bash
cargo build --release --features "llvm17,smt,gpu"
```

Optimized native code, refinement verification, GPU support.

### IDE Development

```bash
cargo build --features "jit,lsp"
```

JIT backend plus LSP server for IDE integration.

### Scientific Computing

```bash
cargo build --release --features "llvm17,gpu,ontology,smt"
```

All scientific features: native code, GPU, ontology, refinement types.

### Full Build

```bash
cargo build --release --features full
```

Everything enabled. Requires all dependencies.

### CI/CD Pipeline

```bash
# Quick check (no codegen)
cargo check --lib

# Tests without LLVM
cargo test --features "jit,gpu,ontology"

# Full tests (requires LLVM)
cargo test --features "jit,llvm17,gpu,ontology,smt"
```

---

## Feature Combinations

### Valid Combinations

```bash
# JIT + GPU
--features "jit,gpu"

# LLVM + SMT
--features "llvm17,smt"

# Everything except WASM
--features "jit,llvm17,gpu,cuda,smt,lsp,ontology,pkg"
```

### Invalid Combinations

```bash
# Multiple LLVM versions (ERROR)
--features "llvm15,llvm17"

# CUDA without GPU (no error but useless)
--features "cuda"
```

---

## Environment Variables

| Variable | Feature | Purpose |
|----------|---------|---------|
| `LLVM_SYS_140_PREFIX` | `llvm14` | LLVM 14 installation path |
| `LLVM_SYS_150_PREFIX` | `llvm15` | LLVM 15 installation path |
| `LLVM_SYS_160_PREFIX` | `llvm16` | LLVM 16 installation path |
| `LLVM_SYS_170_PREFIX` | `llvm17` | LLVM 17 installation path |
| `SOUNIO_STDLIB_PATH` | - | Standard library path |
| `CUDA_PATH` | `cuda` | CUDA toolkit path |
| `Z3_SYS_Z3_HEADER` | `smt` | Z3 header path (if non-standard) |

---

## Conditional Compilation

In Rust code, check features with:

```rust
#[cfg(feature = "jit")]
fn run_jit() { ... }

#[cfg(feature = "llvm-base")]  // Any LLVM version
fn run_llvm() { ... }

#[cfg(feature = "gpu")]
fn compile_gpu_kernel() { ... }

#[cfg(feature = "smt")]
fn verify_refinement() { ... }

#[cfg(not(any(feature = "jit", feature = "llvm-base")))]
compile_error!("Enable jit or llvm feature for code generation");
```

---

## Related Documentation

- [Installation Guide](INSTALLATION.md) - Dependency setup
- [LLVM Codegen](LLVM_CODEGEN.md) - LLVM backend details
- [GPU Runtime](GPU_RUNTIME.md) - GPU features
- [Getting Started](getting-started.md) - First steps

---

*Last updated: January 2026 (v1.0.0)*
