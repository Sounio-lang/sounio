# AGENTS.md

## Project

Sounio is a self-hosted systems + scientific programming language with epistemic types. The compiler is written in Sounio itself, bootstrapped from a C stage0. **This is NOT a Rust project** — no Cargo, no rustc, no Rust toolchain needed.

## Commands

```bash
SOUC=./bin/souc

$SOUC check examples/file.sio          # type-check only
$SOUC run examples/file.sio            # compile to temp ELF, execute, clean up
$SOUC compile file.sio -o output.elf   # compile to named ELF
$SOUC build file.sio --backend gpu -o output.ptx  # compile to PTX (GPU)
$SOUC build file.sio --backend gpu --gpu-precision f32 -o out.ptx  # f32 GPU
make build                             # 3-stage bootstrap (gen1→gen2→gen3) + fixed-point verify
make test                              # full test suite (compile-fail + run-pass + stdlib + ui)
make lint                              # lint .sio files for Rust hallucinations
make check                             # type-check self-hosted compiler + lint gates
```

Run a single test:
```bash
$SOUC run tests/run-pass/specific_test.sio
$SOUC check tests/compile-fail/specific_test.sio
```

Filter the test suite:
```bash
bash scripts/run_sio_test_suite.sh --filter pattern
bash scripts/run_sio_test_suite.sh --verbose --jobs 4
```

Lint a single file:
```bash
python3 scripts/dev/sounio-lint.py path/to/file.sio
python3 scripts/dev/sounio-lint.py --fix path/to/file.sio   # auto-fix
```

## Sounio is NOT Rust — Syntax Gotchas

Agents frequently hallucinate Rust syntax. These will fail at compile time:

| Wrong (Rust) | Right (Sounio) |
|---|---|
| `let x = 5;` | `let x = 5` (no semicolons) |
| `let mut x = 5` | `var x = 5` |
| `&mut T` | `&!T` |
| `assert!(cond)` | `assert(cond)` |
| `println!("hi")` | `println("hi")` |
| `-42` | `0 - 42` (no unary minus) |
| `\|x\| x + 1` | named fn refs only: `let f = square` |
| `#[test]`, `#[derive()]` | no attributes exist |
| `x >> 4` | `x >> 4u8` (bit shifts require u8) |

Effects are **required** on functions with side effects:
```sio
fn greet(name: &str) with IO { println(name) }
fn mutate(x: &!i32) with Mut { *x = 42 }
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }
```
Effects: `IO` (print/file), `Mut` (&! mutation), `Div` (division/modulo), `Panic` (array access, assert, casts)

Helpers must be defined **before** callers (no forward references).

Full syntax reference: `docs/guide/LLM_PROGRAMMING_GUIDE.md`

## Architecture

```
Pipeline: Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR (SSA) → Codegen
```

| Directory | Purpose |
|---|---|
| `self-hosted/compiler/lean_single.sio` | Main compiler source (single-file) |
| `self-hosted/lexer/`, `parser/` | Frontend |
| `self-hosted/check/`, `types/` | Bidirectional type inference + effects |
| `self-hosted/ir/` | IR lowering, optimization, e-graph |
| `self-hosted/native/` | x86-64 ELF emission |
| `self-hosted/gpu/` | GPU backend (PTX, SPIR-V, Metal) |
| `self-hosted/hypercomplex/` | Octonion/sedenion algebra + GPU lowering |
| `self-hosted/gpu/kernels/` | Hypercomplex PTX emitters, O-SSM fused kernels |
| `stdlib/` | Standard library (units, epistemic, stats, linalg…) |
| `bootstrap/` | stage0 (C) → boot2g → self-hosted chain |
| `tests/run-pass/` | Tests that should compile and run |
| `tests/compile-fail/` | Tests that should fail to compile |
| `tests/ui/` | Error message snapshots |
| `tests/stdlib/` | Standard library tests |

## How the Compiler Works

- `bin/souc` is a bash wrapper that delegates to the pre-built native binary at `artifacts/self-hosted/souc-self-hosted-x86_64`
- `scripts/lib/resolve_souc.sh` is the canonical resolution logic; scripts source it to find `souc`
- The default env `SOUNIO_REPO_HARD_NO_RUST=1` — there is no Rust build step
- `SOUNIO_STDLIB_PATH` is set automatically by `bin/souc`; set manually if running the native binary directly
- Linux x86-64 only

## Test Annotations

Tests are discovered by `//@ ` comments in the source file header:

```
//@ run-pass              — expect exit 0
//@ compile-fail          — expect exit != 0
//@ ignore                — skip
//@ check-only            — compile only, do not execute
//@ expect-stdout: TEXT   — stdout must contain TEXT (run-pass)
//@ error-pattern: TEXT   — output must contain TEXT (compile-fail)
//@ known-failure: REASON — documented accepted failure
//@ timeout: SECONDS      — override default 30s timeout
```

Tests without a `//@ run-pass` or `//@ compile-fail` annotation are skipped.

## Commit Format

```
[component] Brief description
```

Components: `lexer`, `parser`, `ast`, `check`, `types`, `effects`, `hir`, `hlir`, `codegen`, `backend`, `gpu`, `cli`, `docs`, `stdlib`, `tests`, `ontology`, `epistemic`, `lsp`, `pkg`, `sir`, `units`, `refinement`

No AI attribution in commits (no "Co-Authored-By").

## Key Constraints

- **No Rust toolchain** — do not run `cargo`, look for `Cargo.toml`, or assume Rust patterns
- **ecosystem/** is distribution/template surface, not the active repo config — do not edit unless explicitly asked
- **Self-hosted bootstrap is safety-critical** — changes to `self-hosted/compiler/lean_single.sio` or `bootstrap/` can break the fixed-point; always verify with `make build`
- **No struct generics yet** — `Knowledge<T>` is monomorphic (f64 only); function-level generics work
- **No closures** — named fn refs only (`let f = square`)
- **No REPL** in native mode
- **GPU backend available** — `souc build --backend gpu` emits PTX; octonion/sedenion lowering via Fano plane; O-SSM fused forward kernel in `self-hosted/gpu/kernels/ossm_forward.sio`

## Lint → Check → Test

The verification order is: lint first (catches Rust hallucinations cheaply), then type-check, then test:

```bash
make lint          # quick: scans for Rust-isms
make check         # type-checks lean_single.sio + runs lint gates
make test          # full test suite
```
