<!-- docs:meta
topic_id: website.docs.getting-started
authority: dual
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.getting-started
-->

# Getting Started with Sounio

> **Other guides**: [Scientists' Quick Start](../QUICK_START_GUIDE.md) | [LLM Quick Start](SOUNIO_QUICK_START.md) | [Conservative contract](MINIMUM_VIABLE_SOUNIO.md)

Welcome to **Sounio**, a programming language and research platform for scientific code that needs explicit uncertainty, provenance, and gate-backed validation.

This guide is intentionally conservative. It reflects the repository state validated on March 9, 2026.

## 1. Use A Real Compiler Artifact

For this checkout, the easiest path is the signed Linux `x86_64` JIT artifact already committed under `artifacts/omega/souc-bin/`:

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" sysroot stdlib-paths
```

In this repo snapshot, that artifact reports `souc 1.0.0-beta.4`.

There is also a separate checked GPU artifact for GPU-specific workflows:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
"$SOUC_GPU_BIN" info
"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

If you need the repo to resolve a pinned binary path for you:

```bash
scripts/omega/omega_resolve_souc_bin.sh --print-path --allow-local-fallback
```

## 2. Start With `check`

The most reliable way to validate language features in this repo is `souc check`.

```bash
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" check tests/run-pass/covid_2020_kernel.sio
"$SOUC_BIN" check tests/run-pass/vancomycin_propagation.sio
"$SOUC_BIN" check tests/compile-fail/vancomycin_low_conf.sio
```

Expected behavior:

- `examples/hello.sio` passes
- `covid_2020_kernel.sio` passes
- `vancomycin_propagation.sio` passes
- `vancomycin_low_conf.sio` fails with the expected confidence-bound type error

## 3. Your First Program

Create a file `hello.sio`:

```sounio
fn main() with IO {
    println("Hello, Sounio!")
}
```

Type-check it:

```bash
"$SOUC_BIN" check hello.sio
```

If your selected `souc` variant supports the runtime path you need, you can then try:

```bash
"$SOUC_BIN" run hello.sio
```

## 4. What Is Actually Verified Today

The gate-backed public summary in this repo is:

- `artifacts/stdlib/stdlib_reliability_status.v1.json`: `81 pass / 0 fail / 1 skip / 82 total`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`: `pass` for `fmri` and `darwin_pbpk`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`: `pass` for 7 required hyper lanes
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`: `pass` for the current GPU runtime smoke set on the checked GPU lane
- local science runtime regression probes are still recorded in `soft` mode unless strict CI enforcement is enabled

For the full conservative contract, read [Minimum Viable Sounio](MINIMUM_VIABLE_SOUNIO.md).

## 5. Key Concepts

### 1. Epistemic Types

Sounio's signature feature is the `Knowledge<T>` type:

```sounio
let risky = Knowledge { value: 15.0, epsilon: 0.4 }
let safe = Knowledge { value: 15.0, epsilon: 0.9 }
```

### 2. Variables

```sounio
let x = 5
var y = 10

y = y + 1
```

### 3. References

Sounio uses `&!` for mutable references:

```sounio
fn increment(x: &!i32) {
    *x = *x + 1
}
```

### 4. Physical Units

```sounio
let distance: f64<m> = 100.0 m
let time: f64<s> = 9.58 s
let speed = distance / time
```

### 5. Effects

```sounio
fn read_file(path: &str) -> String with IO {
    "demo"
}
```

## 6. Command Reference

```bash
souc check file.sio
souc run file.sio
souc build file.sio -o output
souc check file.sio --show-ast
souc check file.sio --show-types
souc sysroot stdlib-paths
souc info
```

## 7. Examples

Prefer these when validating the repo:

| File | Description |
|------|-------------|
| `examples/hello.sio` | Hello World |
| `tests/run-pass/covid_2020_kernel.sio` | Typed epistemic and temporal acceptance |
| `tests/run-pass/vancomycin_propagation.sio` | Confidence propagation |
| `tests/compile-fail/vancomycin_low_conf.sio` | Compile-time refusal on weak evidence |

Do not assume every file under `examples/` is equally runnable. Some are exploratory, backend-dependent, or represent partially implemented surfaces.

## Next Steps

- [Minimum Viable Sounio](MINIMUM_VIABLE_SOUNIO.md)
- [Installation Guide](../../INSTALL.md)
- [Standard Library Reference](../reference/STDLIB_REFERENCE.md)
- [Examples](../../tests/README.md)
