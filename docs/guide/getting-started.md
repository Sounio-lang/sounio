<!-- docs:meta
topic_id: website.docs.getting-started
authority: dual
audience: users
last_validated: 2026-04-12
validated_by: human
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.getting-started
-->

> **Status**: Production | **Last validated**: 2026-04-12 | **Source**: `tests/run-pass/`, committed artifacts

# Getting Started with Sounio

Welcome to **Sounio**, a programming language for scientific code that needs explicit uncertainty, provenance, and gate-backed validation.

This is the **canonical getting-started guide**. For the conservative "what actually works" contract, see [Minimum Viable Sounio](MINIMUM_VIABLE_SOUNIO.md). For LLMs writing Sounio, see the [LLM Programming Guide](LLM_PROGRAMMING_GUIDE.md).

---

## 1. Install and Verify

For this repository snapshot, use the checked compiler artifact under `artifacts/omega/souc-bin/`:

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
```

Current checked-artifact status (`souc 1.0.0-beta.4`):

- Cranelift JIT enabled
- LLVM and GPU codegen disabled in this artifact
- SMT, LSP, ontology, distributed, and package-manager features disabled

For GPU-specific workflows, there is a separate artifact:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
"$SOUC_GPU_BIN" info
"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" build examples/gpu.sio --backend gpu -o /tmp/sounio-gpu.ptx
```

**There is no Rust build step.** Do not run `cargo build` at the repo root.

---

## 2. Your First Program

Create `hello.sio`:

```sio
fn main() with IO {
    println("Hello, Sounio!")
}
```

Type-check it:

```bash
"$SOUC_BIN" check hello.sio
```

Run it (if your artifact supports the runtime path):

```bash
"$SOUC_BIN" run hello.sio
```

---

## 3. Key Concepts in 60 Seconds

### Variables — no semicolons, `var` for mutable

```sio
let x = 5
var y = 10
y = y + 1
```

### Effects — required on functions with side effects

```sio
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }
fn mutate(x: &!i32) with Mut { *x = 42 }
fn hello() with IO { println("hi") }
```

### Mutable references — `&!` not `&mut`

```sio
fn increment(x: &!i32) with Mut { *x = *x + 1 }
var counter: i32 = 0
increment(&!counter)
```

### Epistemic types — uncertainty is tracked

```sio
let risky = Knowledge { value: 15.0, epsilon: 0.4 }
let safe = Knowledge { value: 15.0, epsilon: 0.9 }
```

### Structs, enums, arrays

```sio
struct Point { x: f64, y: f64 }
let p = Point { x: 1.0, y: 2.0 }

enum Color { Red, Green, Blue }

var buffer: [u8; 256] = [0; 256]
let data: [i64; 4] = [1, 2, 3, 4]
```

---

## 4. Sounio is NOT Rust

The syntax looks similar but semantics differ. These mistakes cause compile errors:

| Wrong (Rust) | Right (Sounio) |
|---|---|
| `let x = 5;` | `let x = 5` (no semicolons) |
| `let mut x = 5` | `var x = 5` |
| `&mut T` | `&!T` |
| `assert!(cond)` | `assert(cond)` |
| `println!("hi")` | `println("hi")` |
| `-42` | `0 - 42` (no unary minus) |
| `\|x\| x + 1` | named fn refs only: `let f = square` |
| `x >> 4` | `x >> 4u8` (bitshifts require u8) |

See [Sounio Gotchas](SOUNIO_GOTCHAS.md) for the full list.

---

## 5. Command Reference

```bash
souc check file.sio                          # type-check only
souc run file.sio                            # compile to temp ELF, execute
souc compile file.sio -o output.elf          # compile to named ELF
souc build file.sio --backend gpu -o out.ptx # GPU PTX emission
souc fmt file.sio                            # format source
souc info                                    # print artifact capabilities
```

---

## 6. Verified Examples

These are gate-backed and known to work:

| File | Description |
|---|---|
| `examples/hello.sio` | Hello World |
| `tests/run-pass/for_in_loops.sio` | For-in loop variants |
| `tests/run-pass/array_mut_ref.sio` | Mutable array references |
| `tests/run-pass/covid_2020_kernel.sio` | Epistemic and temporal acceptance |
| `tests/run-pass/vancomycin_propagation.sio` | Confidence propagation |
| `tests/compile-fail/vancomycin_low_conf.sio` | Compile-time refusal on weak evidence |

Do not assume every file under `examples/` is equally runnable. Some are exploratory or backend-dependent.

---

## 7. What Is Actually Verified

Gate-backed public summary:

- `artifacts/stdlib/stdlib_reliability_status.v1.json`: `81 pass / 0 fail / 1 skip`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`: `pass` for `fmri` and `darwin_pbpk`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`: `pass` for 7 required hyper lanes

For the full conservative contract, read [Minimum Viable Sounio](MINIMUM_VIABLE_SOUNIO.md).

---

## 8. Self-Hosted Compiler (Contributors)

The self-hosted compiler driver is `self-hosted/compiler/main.sio`:

```bash
timeout 240 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --self-test
timeout 180 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --check examples/hello.sio
```

---

## Next Steps

- [Minimum Viable Sounio](MINIMUM_VIABLE_SOUNIO.md) — what actually works today
- [Tutorial](tutorial.md) — step-by-step learning guide
- [Cookbook](../COOKBOOK.md) — task-oriented recipes
- [LLM Programming Guide](LLM_PROGRAMMING_GUIDE.md) — definitive syntax reference
- [Standard Library Reference](../reference/STDLIB_REFERENCE.md) — API docs
- [Gotchas](SOUNIO_GOTCHAS.md) — common mistakes
- [Examples](../../examples/) — real code
