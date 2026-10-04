# CLAUDE.md

This file is the entry point for Claude Code (claude.ai/code) and other AI assistants that write Sounio code or work in the Sounio repository. It covers how to build and run the compiler, the syntax that differs from Rust, the agent tooling that ships with the repository, and the known limitations. `AGENTS.md` carries the same guidance for agents that read that file instead.

Section numbers are stable on purpose: other documents and CI gates cite them. Sections that described how the project itself is developed with agents are not part of this guide.

| Quick reference | |
|---|---|
| Project intent | [`FOUNDER_INTENT.md`](FOUNDER_INTENT.md) |
| Programming guide | [`docs/guide/LLM_PROGRAMMING_GUIDE.md`](docs/guide/LLM_PROGRAMMING_GUIDE.md) |
| LLM cookbook | [`docs/llm-guide/`](docs/llm-guide/) |
| Minimum viable Sounio | [`docs/guide/MINIMUM_VIABLE_SOUNIO.md`](docs/guide/MINIMUM_VIABLE_SOUNIO.md) |
| Style guide | [`docs/guide/SOUNIO_STYLE_GUIDE.md`](docs/guide/SOUNIO_STYLE_GUIDE.md) |
| Gotchas | [`docs/guide/SOUNIO_GOTCHAS.md`](docs/guide/SOUNIO_GOTCHAS.md) |
| Known limitations | [`docs/compiler/KNOWN_LIMITATIONS.md`](docs/compiler/KNOWN_LIMITATIONS.md) |

---

## 2. Project identity

**Sounio** — a self-hosted systems + scientific programming language for epistemic computing, uncertainty propagation, and algebraic effects. Linux x86-64 only. Not a Rust or Julia dialect; own syntax, semantics, philosophy.

Two things are simultaneously true about this repository:

1. **It is a language.** A self-hosted compiler in `self-hosted/`, a bootstrap chain `bootstrap/stage0` (C, ~103 KB) → `boot4` → `gen1` → `gen2` → `gen3` (fixed-point verification: gen2 = gen3 bit-identical).

2. **It is a scientific computing platform.** First-class `Knowledge[T]` with GUM uncertainty propagation, Caputo fractional derivatives, autograd, PINN training, refinement types, algebraic effects (`IO`, `Mut`, `Div`, `Panic`, `Alloc`, `Async`, `GPU`, `Prob`, `Observe`), units, linear types.

---

## 4. Build & run

The compiler is self-hosted (written in Sounio, not Rust). **`bin/souc` is the default compiler entrypoint and now routes to Madaros** — the self-hosted *modular* compiler (`artifacts/self-hosted/madaros`, built via `make build-madaros`). The legacy single-file `lean_single` engine that `bin/souc` used to be is preserved as `bin/souc-lean-single-x86_64`; force it with `SOUNIO_SOUC_ENGINE=lean_single`. lean_single remains the **bootstrap seed** (`make build`, `make build-madaros`) and the canonical fixed-point ELF — it is no longer the default *user-facing* compiler. If Madaros has not been built yet, `bin/souc` falls back to lean_single with a notice on stderr.

> **Naming (canonical): the compiler is spelled `Madaros`** — matching `make build-madaros`, `bin/madaros`, and `docs/MADAROS_STATUS.md`. The source string was fixed on 2026-07-11 (`self-hosted/compiler/main.sio`) **and the shipped ELF `bin/madaros-linux-x86_64` was rebuilt to match**, so `./bin/souc --version` now prints `Madaros v0.80.0`. (A freshly-cloned checkout that has *not* re-run `make build-madaros` locally will still show whatever the committed binary carries; on `main` that is now `Madaros`.) Current version: **v0.80.0**.

```bash
SOUC=./bin/souc
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib   # required when outside repo root

$SOUC --version                           # verify toolchain
$SOUC check file.sio                      # type-check only
$SOUC run file.sio                        # compile + execute + clean up
$SOUC compile file.sio -o output.elf      # emit named ELF binary
$SOUC info                                # compiler status (Madaros only -- SOUNIO_SOUC_ENGINE=lean_single has no `info` subcommand)
```

For lint, harness annotations and the test directory layout, see [`docs/guide/SOUNIO_DEFINITIVE_GUIDE.md`](docs/guide/SOUNIO_DEFINITIVE_GUIDE.md) and [`docs/guide/CHECK_SOUNIO_GUIDE.md`](docs/guide/CHECK_SOUNIO_GUIDE.md).

---

## 5. AI-native tooling

This checkout ships two local agent surfaces:

- [`tools/lsp/README.md`](tools/lsp/README.md) — Sounio LSP: diagnostics, hover, completions, go-to-definition, references, rename over stdio
- [`tools/mcp/README.md`](tools/mcp/README.md) — Sounio MCP server exposing compiler `check`, `compile`, `run`, `test`, stdlib docs, and compiler-error resources over local stdio

Run the MCP server with Claude Code:

```bash
pip install -e tools/mcp
python -m sounio_mcp.server --transport stdio
claude --mcp-server sounio=python:-m:sounio_mcp.server
```

Use `sounio_check` as the first repair-loop step for `.sio` edits. The tool returns the same diagnostic wire family as `souc check --json` and `tools/shared/diagnostic_schema.json`, with MCP-friendly `line`/`column`/`span` fields. For compiler errors, read `sounio://errors/{code}`; for stdlib context, read `sounio://stdlib/{module}`.

See [`tools/mcp/examples/claude_code_usage.md`](tools/mcp/examples/claude_code_usage.md) for the error → fix loop recipe.

---

## 6. Working rules

- **Measure before claiming.** Any quantitative statement about this repository must be backed by a command the operator can re-run. Never write "the codebase is small/incomplete/legacy" based on prior probability.
- **Compilation is the test of existence.** A `.sio` file's status is `./bin/souc check <file>` plus the presence of a caller. Running `./bin/souc run` on a library file and reporting it broken is a category error: most files in `stdlib/` and `examples/` are libraries, not executables.

---

## 7. Sounio syntax (NOT Rust)

Critical differences. **Five of the seven rows below were measured on
2026-08-20 and are style, not enforcement** — the compiler accepts the Rust form.
Write the Sounio form; do not expect a diagnostic if you slip. Rows marked ✓ are
enforced.

| Wrong (Rust) | Correct (Sounio) |
|---|---|
| `let x = 5;` | `let x = 5` — **style, not a compile error.** Measured 2026-08-20: the trailing `;` is accepted and the program runs. Prefer the semicolon-free form; do not expect the compiler to enforce it. |
| `let mut y = 10` | `var y = 10` — ✓ enforced, `error[E040]` |
| `&mut T` | `&!T` — ✓ enforced, `error[E041]` |
| `assert!(cond)` | `assert(cond)` — ✓ enforced as of the ELF this repo ships, `error[E043]` (*Sounio does not use Rust macros*). **The old "DANGEROUS, checks clean and is inert" reading was true of the committed binary, not of the source**: measured 2026-08-29, `assert!(1 == 2)` is refused by a Madaros built from `self-hosted/`, and accepted-then-inert by the ELF that was committed before it. The one-character footgun is closed the moment the shipped ELF is refreshed. |
| `println!("hi")` | `println("hi")` — ✓ enforced, `error[E043]`, same measurement and same caveat as the row above. The *"check clean and SIGSEGV at run time (`rc=139`)"* behaviour recorded in `docs/audit/RUST_MACRO_ACCEPTANCE_2026-08-20.md` is what the **committed** ELF still does; source refuses. |
| `#[test]`, `#[derive()]` | No attributes — ✓ enforced, fails to parse |
| ~~`-42`~~ | **STALE — unary minus works.** Measured 2026-08-20 on both engines: `-3.5`, `f(-7)`, `10 - -3` and `[-1, -2, -3]` all check and compute correctly. `0 - x` is no longer required. |
| `x >> 4` | `x >> 4u8` — **STALE.** Measured 2026-08-20: `x >> 4` checks and computes correctly (`64 >> 4 = 4`). |

Helpers must be defined before callers — no forward references.

Quick reference:

```sounio
let x = 5                              // immutable
var y = 10                             // mutable
var buf: [i64; 8] = [0; 8]             // fixed-size array
&T / &!T                               // shared / exclusive ref
fn f(x: i32) -> i32 with IO { }        // effects declaration
linear struct Handle { fd: i32 }       // linear types
let dose: mg = 500.0                   // units
let arr2 = a ++ b                      // array concatenation
type Pos = { x: i32 | x > 0 }          // refinement type
let m: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
fn observe(x: Unobserved<f64>) -> bool with Observe { x > 0.0 }

// Effects: IO, Mut, Div, Panic, Alloc, Async, GPU, Prob, Observe

impl MyStruct {
    fn get(self: &MyStruct) -> i64 { self.val }
    fn set(self: &!MyStruct, v: i64) with Mut { self.val = v }
}

for i in 0..10 { }      // exclusive range
for i in 0..=10 { }     // inclusive range
if x > 0 { "pos" } else { "neg" }   // if is an expression
```

Full reference: [`docs/guide/LLM_PROGRAMMING_GUIDE.md`](docs/guide/LLM_PROGRAMMING_GUIDE.md).

---

## 8. Architecture

Pipeline: Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR (SSA) → Codegen (x86-64 ELF).

| Directory | Purpose |
|---|---|
| `self-hosted/lexer/`, `parser/` | Frontend (tokenizer, recursive descent) |
| `self-hosted/check/`, `types/` | Bidirectional type inference + algebraic effects |
| `self-hosted/ir/` | IR lowering, e-graph optimization (1000+ rewrite rules) |
| `self-hosted/native/` | x86-64 ELF emission |
| `self-hosted/compiler/` | Codegen drivers (lean, IR, GPU) |
| `self-hosted/gpu/` | PTX/GPU codegen; end-to-end CLI path exists under default Madaros (`souc build --backend gpu`, see §13) -- no GPU CLI surface under `SOUNIO_SOUC_ENGINE=lean_single`, which rejects the invocation outright |
| `stdlib/epistemic/` | `Knowledge<T>`, uncertainty (GUM), provenance |
| `stdlib/units/` | Dimensional analysis |
| `bootstrap/` | stage0 (C) → boot2g → boot3 → boot4 → self-hosted |
| `formal/` | Lean 4 proofs (epistemic type invariants) |

---

## 13. Known limitations

Headline limitations (full list in [`docs/compiler/KNOWN_LIMITATIONS.md`](docs/compiler/KNOWN_LIMITATIONS.md)):

- **Imported-module native path — partial closeout.** Residuals remain: multi-module memory-wall / exclusive-ref fragile chains, named-import/`print_f64` papercuts. Finite-dof `gum_k95` is **trustworthy** under default Madaros.
- `Knowledge<T>` supports struct-level generics (`f64`, `bool`, struct types)
- `--show-ast` / `--show-types` are unavailable under the default Madaros engine (`bin/souc compile ... --show-ast` -> `error: madaros build: unsupported option`); both work under `SOUNIO_SOUC_ENGINE=lean_single` / `bin/souc-lean-single-x86_64`. A REPL does exist (`souc repl` -> `tools/repl.sh`) -- it's a file-based compile-and-run loop over whichever engine `bin/souc` currently resolves to, not a true interactive evaluator.
- `&![T; N]` bare array mutation broken in JIT — use struct wrapper or `(*arr)[i]`
- GPU: end-to-end `kernel fn` → PTX path **exists and is reproducible under default Madaros**. `bin/souc build <file>.sio --backend gpu -o out.ptx` (verified: `examples/kernel_vec_add.sio` → valid PTX). This is Madaros-only: `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc build ... --backend gpu ...` has no GPU CLI surface at all and fails to parse the invocation (verified 2026-08-17). Runtime execution is fixture-bounded (L4-validated profiles).
- **Dual-engine divergence.** `f128` is a working binary128 type on **both** engines, but they cover different surfaces. Both give 113 halvings to `1 + eps == 1` (binary64 gives 53) and print `1/3` as `3.33333333333333333333333333333333317e-0001` via `math::softfloat_f128_fmt::print_f128`.
  - **Madaros** computes `+ - * / %`, unary minus and IEEE compares, and supports params/returns, struct fields, methods, fixed arrays, exact 36-digit printing and exact literals. Measured: `7.5 % 2.0 = 1.5`; `2/3` through a struct field and an array element is correct to the last digit. It refuses, fail-closed:
    - `as` casts between `f128` and `f64` (**E248**, "wide-float casts fail closed");
    - values that flow through an `f128` global or a `type` alias over `f128` (V0-E.4.1 lowering refusal). The declarations alone build: an unused `let G: f128` or `type Wide = f128` compiles;
    - mixing an `f128` with an untyped literal or an `f256` (**E004**). Write typed constants, e.g. `let three: f128 = 3.0; x / three`, not `x / 3.0`.
  - **lean_single** (#2387/#2426) computes binary128 through libgcc for locals, direct-call params/returns, IEEE compares, and `as` casts in both directions. It refuses, fail-closed: struct fields, arrays, reading an `f128` global (an unused one builds; a `type` alias over `f128` works), tuples, methods, fn values, `println` of an `f128` (the stdlib `print_f128` works), and `%` ("modulo requires integer operands").
  - **Literal exactness differs by engine.** Measured 2026-09-26:
    - **Madaros** parses an f128 literal straight to binary128 (V0-E.5.9) and accepts the literals its parser can prove exact in binary128. Decimal digits accumulate in 128 bits (`self-hosted/parser/f128_literal.sio`), so a very long decimal is refused even when its value is exact, such as the full expansion of 2^200. `0.1` fails. `1e23` prints exactly, and so does the hex-float `0x1.0000000000000001p+0` (1 + 2⁻⁶⁴). The hex-float exponent needs an explicit sign: `…p0` is mis-lexed and fails with E012.
    - **lean_single** widens through binary64 and refuses any literal not exact in binary64 (`f128_widen_refuses_inexact_literal` in `lean_single.sio`), so it also rejects `1e23` and that hex-float.
    - Portable constants: small dyadic decimals that are exact in binary64 (`0.25`, `3.0`), `f128_from_limbs(lo, hi)` (works on both), or computation (`one / ten`).
  - **Not implemented anywhere:** `Knowledge<f128>` and GUM over `f128` (KL-15).
  - **`f256`** on Madaros: declarations and some operators typecheck (an unused f256 enum field or global builds), but executable f256 value lowering is refused fail-closed (V0-E.4.1), never lowered as f64. lean_single refuses `f256` outright (KL-9). Only `stdlib/math/softfloat_f256.sio` add/sub over `F256Bits` computes f256 values (KL-15a); there is no f256 mul, div, fields, arrays or printing.
- **Cross-call first-order variance (KL-11) is open.** First-order channels do not cross user calls: `tests/run-pass/gum_fo_across_call.sio` and `tests/run-pass/fo_call_boundary_arity3.sio` still carry `//@ known-failure` (see `docs/compiler/KNOWN_LIMITATIONS.md`). Re-verify `gum_fo_across_call.sio` before relying on general cross-call FO propagation.

---

*This file is the entry point for AI assistants using Sounio. `AGENTS.md` carries the same guidance for agents that read it instead.*
