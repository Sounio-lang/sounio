<!-- docs:meta
topic_id: website.docs.getting-started
authority: dual
audience: users
last_validated: 2026-10-15
validated_by: souc-linux Madaros v0.80.0
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.getting-started
-->

# Getting Started with Sounio

> **Other guides**: [Tutorial](tutorial.md) | [Measured tour](../../TOUR.md) | [Known limitations](../compiler/KNOWN_LIMITATIONS.md)

Welcome to **Sounio**, a programming language for scientific code that carries uncertainty, units, and declared effects.

This page teaches the surface measured in October 2026, not the April 22 snapshot. The positive measurements are [`TOUR.md`](../../TOUR.md) (2026-10-04), the open-item ledger in [`KNOWN_LIMITATIONS.md`](../compiler/KNOWN_LIMITATIONS.md) (validated 2026-10-06; KL-11 narrowed 2026-10-13), and [`public-claim-registry.v1.tsv`](../serious-language/public-claim-registry.v1.tsv). [`MINIMUM_VIABLE_SOUNIO.md`](MINIMUM_VIABLE_SOUNIO.md) is an older conservative contract; where it still says April 22, prefer the tour and the ledger.

The committed prebuilt used by the default `bin/souc` route is a Linux x86-64 ELF; it does not run on macOS. Apple Silicon also has a separate current-source native ARM64 build and real Metal route integrated in [#2978](https://github.com/Sounio-lang/sounio/pull/2978). A clean checkout needs the native artifact built and installed before the launcher can use that route. Follow the [macOS ARM64 guide](../compiler/MADAROS_MACOS_ARM64.md) for setup and measured limits. The commands below describe the Linux prebuilt path unless stated otherwise.

## 1. Check before you run

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" check examples/hello.sio
```

`bin/souc` is the default compiler and routes to **Madaros**. Force the legacy bootstrap seed with `SOUNIO_SOUC_ENGINE=lean_single` only when a demo below says so.

On Linux x86-64 you can then compile and run:

```bash
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello.elf
"$SOUC_BIN" run examples/hello.sio
```

Measured on Madaros v0.80.0: `check --show-ast` and `check --show-types` are ignored. The check still exits 0 and prints `check: OK`; it does not print an AST. `compile` and `build` reject both flags with `error: madaros build: unsupported option` (exit 2). `SOUNIO_SOUC_ENGINE=lean_single souc check --show-ast` prints compiler stats, not a source AST.

## 2. Your first program

`examples/hello.sio` is the canonical hello:

```sounio
fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

`let` is immutable. `var` is mutable. Exclusive references are `&!T`, not `&mut T`. Sounio does not use Rust macros (`println!`, `assert!`).

```sounio
let x = 5
var y = 10
y = y + 1

fn increment(x: &!i32) with Mut {
    *x = *x + 1
}
```

## 3. What you can run today

These three intro demos are the measured onboarding path. Run them on Linux x86-64.

```bash
# GUM analytic propagation. Madaros, the default engine.
# Measured in TOUR.md §1 (demos/intro/demo_incerteza.sio).
"$SOUC_BIN" run demos/intro/demo_incerteza.sio

# Derived units. NOT the default engine. Madaros check fails with
# error[E001] "this binding expects a different type" (expected f64, found f64)
# on `let distance: m = 100.0` (#2752). It does not name the unit.
# lean_single rejects `m + s`, but still accepts a multiply into a derived unit (#2751).
SOUNIO_SOUC_ENGINE=lean_single "$SOUC_BIN" run demos/intro/demo_unidades.sio

# Knowledge guards. `souc run` does not enforce `where` by itself (#2753).
# The working path is the source-to-source expansion, then run.
bash scripts/ontology/expand_knowledge_runtime_guards.sh \
  demos/intro/demo_portas_sucesso.sio /tmp/demo_portas_sucesso.sio
"$SOUC_BIN" run /tmp/demo_portas_sucesso.sio

bash scripts/ontology/expand_knowledge_runtime_guards.sh \
  demos/intro/demo_portas_rejeicao.sio /tmp/demo_portas_rejeicao.sio
"$SOUC_BIN" run /tmp/demo_portas_rejeicao.sio
```

### Uncertainty

Do not write `Knowledge { value, epsilon }`. The intro demo uses `measure` and `variance_of`. The checked library surface is `epistemic::knowledge` (`ep_measured`, `ep_val`, `ep_std`, `ep_mul`), anchored by `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio`. First-order variance crosses a *pure helper* (KL-11, narrowed 2026-10-13). It does not cross a general user call. The [tutorial](tutorial.md) is the longer form.

### Units

Units are a fixture-backed prototype (`units.measure` in the claim registry), not a finished type system. Derived annotations such as `f64<m/s>` and `mol/cm3` are closed on the ledger (KL-13) and have named tests. That does not make `let distance: m = 100.0` legal on default Madaros (#2752), and it does not close the lean_single multiply hole (#2751). Copy `demos/intro/demo_unidades.sio`, including its engine comment, rather than inventing a `100.0 m` literal.

### Effects

The checked effect cases are `IO`, `Mut`, `Div`, and `Panic`. Declare them with `with`. Custom `effect` / `handle` blocks are not the checked user surface.

## 4. Commands

```bash
souc check file.sio
souc compile file.sio -o output.elf
souc build file.sio -o output.elf
souc run file.sio          # Linux x86-64 ELF; not a macOS run
souc info
souc --version
```

## 5. Where to look next

- [README.md](../../README.md) — repository map, current platform routes, and evidence boundaries
- [Tutorial](tutorial.md) — syntax, the epistemic API, and the unit engine split
- [TOUR.md](../../TOUR.md) — twenty measured claims, and the table of what does not work
- [Known limitations](../compiler/KNOWN_LIMITATIONS.md) — open ledger, including KL-11 and KL-20
- [Programming guide](programming.md) — longer reference; sections without a cited test are not the contract

Do not treat every file under `examples/` as equally runnable. Enum variants with payloads do not work (KL-20). Closures are not a safe user surface.
