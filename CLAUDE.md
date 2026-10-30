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

## 3. Canonical branch

`main` is the canonical branch; `integration/sounio-dev-ready-base` is deprecated. Start new work from `main`.

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
- **Fix, don't file.** A defect found during a task is fixed in that task, with a test that fails before the fix and passes after it. Opening an issue instead is allowed only when (1) the fix needs a maintainer decision (language semantics, public API, a choice between legitimate alternatives), (2) the fix clearly exceeds the task's scope, or (3) the agent cannot write to the repository that holds the defect. Such an issue states which of the three applies, localises the cause to file and function, and ships the discriminating test (a minimal program with the expected and actual output). An issue that only describes a symptom, or that reports something the agent could have fixed, is not filed. Before filing, search the open issues for the same defect and add to that one instead.

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
let dose: mg = 500.0                   // NOT on default Madaros: `let d: m = 100.0` is E001 (#2752). lean_single demo only.
let arr2 = a ++ b                      // array concatenation
type Pos = { x: i32 | x > 0 }          // refinement type
let m: Knowledge<f64> = measure(10.0, uncertainty: 0.1)  // demo_incerteza.sio; not Knowledge<mg>
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
- `--show-ast` / `--show-types` on `compile` or `build` are rejected by default Madaros (`error: madaros build: unsupported option`, exit 2). Measured 2026-10-15: `check --show-ast` and `check --show-types` are **ignored** and still print `check: OK`. `SOUNIO_SOUC_ENGINE=lean_single souc check --show-ast` prints compiler stats, not a source AST. A REPL does exist (`souc repl` -> `tools/repl.sh`) -- it's a file-based compile-and-run loop over whichever engine `bin/souc` currently resolves to, not a true interactive evaluator.
- `&![T; N]` bare array mutation broken in JIT — use struct wrapper or `(*arr)[i]`
- GPU: two records, not re-run on 2026-10-15. `TOUR.md` §6 (2026-10-04) measured `souc build --backend gpu` as PTX for empty kernel bodies only. An older note in this section said `examples/kernel_vec_add.sio` emitted valid PTX under Madaros, and that `lean_single` has no GPU CLI (verified 2026-08-17). The root README (2026-10-11) records scoped Spark evidence for GPU/Pireus and says the generic GPU CLI needs its own status. Do not teach a general kernel launch from this bullet.
- **Dual-engine divergence (`f128`/`f256`).** `f128` is a working binary128 type on both engines but they cover different surfaces and refuse different constructs (casts, globals, literals; E248, E004). `Knowledge<f128>`/GUM over `f128` is not implemented (KL-15); `f256` has no executable lowering on Madaros and is refused by lean_single. Full per-engine table: [`docs/compiler/F128_ENGINE_DIVERGENCE.md`](docs/compiler/F128_ENGINE_DIVERGENCE.md).
- **Cross-call first-order variance (KL-11) is narrowed, not closed.** Pure helpers carry first-order channels (ledger, 2026-10-13). A general user call still does not: `tests/run-pass/gum_fo_across_call.sio` and `tests/run-pass/fo_call_boundary_arity3.sio` keep `//@ known-failure`. Do not treat every call as variance-preserving.

---

*This file is the entry point for AI assistants using Sounio. `AGENTS.md` carries the same guidance for agents that read it instead.*
