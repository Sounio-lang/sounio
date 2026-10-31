# CLAUDE.md

Daily instructions for agents working in Sounio. `AGENTS.md` points here for language and compiler guidance. Details and dated measurements: [agent reference](docs/guide/AGENT_REFERENCE_DETAILS.md).

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

Sounio is a self-hosted systems and scientific language for epistemic computing, uncertainty propagation and algebraic effects. It has its own syntax and semantics. Compiler execution targets Linux x86-64. Intent: [`FOUNDER_INTENT.md`](FOUNDER_INTENT.md).

---

## 3. Canonical branch

`main` is the canonical branch; `integration/sounio-dev-ready-base` is deprecated. Start new work from `main`.

---

## 4. Build & run

`./bin/souc` is the public entrypoint; default engine: **Madaros**, the modular self-hosted compiler. `make build-madaros` builds it. The legacy `lean_single` engine remains the bootstrap seed (`make build`) and fallback when Madaros is unavailable; force it with `SOUNIO_SOUC_ENGINE=lean_single`. Use the canonical resolvers in `scripts/lib/resolve_souc.sh` and `scripts/lib/resolve_madaros.sh`. Check [`docs/MADAROS_STATUS.md`](docs/MADAROS_STATUS.md) before interpreting stale binaries.

```bash
./bin/souc --version
./bin/souc check file.sio
./bin/souc run file.sio
./bin/souc compile file.sio -o output.elf
./bin/souc info  # Madaros only
# Outside repo root: export SOUNIO_STDLIB_PATH=/absolute/path/to/stdlib
```

Lint and test harness: [`docs/guide/CHECK_SOUNIO_GUIDE.md`](docs/guide/CHECK_SOUNIO_GUIDE.md).

---

## 5. AI-native tooling

Use `sounio_check` first in the `.sio` repair loop; after every edit run `./bin/souc check <file>`. Compiler errors: `sounio://errors/{code}`; stdlib: `sounio://stdlib/{module}`. Setup and usage: [`tools/mcp/README.md`](tools/mcp/README.md), [`tools/lsp/README.md`](tools/lsp/README.md), [repair-loop example](tools/mcp/examples/claude_code_usage.md).

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

Dated measurements and extended examples: [reference details](docs/guide/AGENT_REFERENCE_DETAILS.md#7-sounio-syntax-not-rust). Full syntax: [`docs/guide/LLM_PROGRAMMING_GUIDE.md`](docs/guide/LLM_PROGRAMMING_GUIDE.md).

---

## 8. Architecture

Pipeline: lexer → parser → AST/check → HIR → SIR → HLIR (SSA) → codegen. Sources: `self-hosted/{lexer,parser,check,types,ir,native,compiler,gpu}/`; scientific libraries: `stdlib/{epistemic,units}/`; bootstrap: `bootstrap/`; Lean proofs: `formal/`. [Detailed directory map](docs/guide/AGENT_REFERENCE_DETAILS.md#8-architecture).

---

## 13. Known limitations

Consult [`docs/compiler/KNOWN_LIMITATIONS.md`](docs/compiler/KNOWN_LIMITATIONS.md) before promising support. Important boundaries:

- Imported-module native compilation still has fragile multi-module/exclusive-reference chains and named-import/`print_f64` issues.
- `Knowledge<T>` supports struct generics; `Knowledge<f128>`/GUM over `f128` remains unsupported (KL-15). `f128` capabilities differ by engine; `f256` lacks executable lowering. See [engine table](docs/compiler/F128_ENGINE_DIVERGENCE.md).
- `--show-ast` / `--show-types` are rejected on Madaros build/compile and may be ignored on check. The REPL is a file-based compile/run loop.
- JIT bare-array mutation via `&![T; N]`: use a struct wrapper or `(*arr)[i]`.
- GPU CLI evidence is scoped and dated; do not infer general kernel launches from PTX emission or another backend's success.
- First-order variance across general user calls remains incomplete (KL-11); preserve known-failure tests.

[Dated evidence and qualifications](docs/guide/AGENT_REFERENCE_DETAILS.md#13-known-limitations).

---
