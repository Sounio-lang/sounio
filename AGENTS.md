# AGENTS.md

**Canonical AI agent guidance for the Sounio repository.** This file is the single source of truth for all AI coding assistants (Claude, Cursor, GPT, Grok, etc.). Other agent entry points (CLAUDE.md, .cursorrules, .cursor/rules/) now redirect here.

Project intent: [`FOUNDER_INTENT.md`](FOUNDER_INTENT.md)

---

## Table of Contents

1. [Project identity](#project-identity)
2. [Build & run](#build--run)
3. [AI-native tooling](#ai-native-tooling)
4. [Working rules](#working-rules)
5. [Sounio syntax (NOT Rust)](#sounio-syntax-not-rust)
6. [Architecture](#architecture)
7. [Compiler resolution](#compiler-resolution)
8. [Known limitations](#known-limitations)
9. [Decision records (ADRs)](#decision-records-adrs)
10. [Context7 integration](#context7-integration)

---

## 1. Project identity

**Sounio** — a self-hosted systems + scientific programming language for epistemic computing, uncertainty propagation, and algebraic effects. Linux x86-64 only. Not a Rust or Julia dialect; own syntax, semantics, philosophy.

Two things are simultaneously true about this repository:

1. **It is a language.** A self-hosted compiler in `self-hosted/`, a bootstrap chain `bootstrap/stage0` (C, ~103 KB) → `boot4` → `gen1` → `gen2` → `gen3` (fixed-point verification: gen2 = gen3 bit-identical).

2. **It is a scientific computing platform.** First-class `Knowledge[T]` with GUM uncertainty propagation, Caputo fractional derivatives, autograd, PINN training, refinement types, algebraic effects (`IO`, `Mut`, `Div`, `Panic`, `Alloc`, `Async`, `GPU`, `Prob`, `Observe`), units, linear types.

### Canonical branch

`main` is the canonical branch; `integration/sounio-dev-ready-base` is deprecated. Start new work from `main`.

### Quick reference

| Resource | Path |
|---|---|
| Programming guide | [`docs/guide/LLM_PROGRAMMING_GUIDE.md`](docs/guide/LLM_PROGRAMMING_GUIDE.md) |
| LLM cookbook | [`docs/llm-guide/`](docs/llm-guide/) |
| Minimum viable Sounio | [`docs/guide/MINIMUM_VIABLE_SOUNIO.md`](docs/guide/MINIMUM_VIABLE_SOUNIO.md) |
| Style guide | [`docs/guide/SOUNIO_STYLE_GUIDE.md`](docs/guide/SOUNIO_STYLE_GUIDE.md) |
| Gotchas | [`docs/guide/SOUNIO_GOTCHAS.md`](docs/guide/SOUNIO_GOTCHAS.md) |
| Known limitations | [`docs/compiler/KNOWN_LIMITATIONS.md`](docs/compiler/KNOWN_LIMITATIONS.md) |

---

## 2. Build & run

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

## 3. AI-native tooling

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

## 4. Working rules

- **Measure before claiming.** Any quantitative statement about this repository must be backed by a command the operator can re-run. Never write "the codebase is small/incomplete/legacy" based on prior probability.
- **Compilation is the test of existence.** A `.sio` file's status is `./bin/souc check <file>` plus the presence of a caller. Running `./bin/souc run` on a library file and reporting it broken is a category error: most files in `stdlib/` and `examples/` are libraries, not executables.
- **Fix, don't file.** A defect found during a task is fixed in that task, with a test that fails before the fix and passes after it. Opening an issue instead is allowed only when (1) the fix needs a maintainer decision (language semantics, public API, a choice between legitimate alternatives), (2) the fix clearly exceeds the task's scope, or (3) the agent cannot write to the repository that holds the defect. Such an issue states which of the three applies, localises the cause to file and function, and ships the discriminating test (a minimal program with the expected and actual output). An issue that only describes a symptom, or that reports something the agent could have fixed, is not filed. Before filing, search the open issues first and extend an existing one rather than opening a duplicate.

### Defects: fix, don't file

A defect you find while working is fixed in the same change, with a test that fails before the fix and passes after it. Open an issue only when one of these holds, and say which:

1. the fix needs a maintainer decision (language semantics, public API, or a choice between legitimate alternatives);
2. the fix clearly exceeds the scope of the task;
3. you cannot write to the repository that holds the defect.

The issue must localise the cause (file and function) and include the discriminating test: a minimal program with expected and actual output. Symptom-only issues, and issues for defects you could have fixed, are not filed. Search the open issues first and extend an existing one rather than opening a duplicate.

### Writing Sounio

Sounio is not Rust. Read section 5 below before writing any `.sio` file, and [`docs/guide/LLM_PROGRAMMING_GUIDE.md`](docs/guide/LLM_PROGRAMMING_GUIDE.md) for the full reference. After every edit, run `./bin/souc check <file>`: compilation is the test of existence, and `souc run` on a library file is a category error.

Repository facts and executable scripts override stale documentation when they disagree. When in doubt, prefer:

1. actual repo files and executable scripts
2. committed docs
3. assumptions

---

## 5. Sounio syntax (NOT Rust)

Critical differences. **Five of the seven rows below were measured on 2026-08-20 and are style, not enforcement** — the compiler accepts the Rust form. Write the Sounio form; do not expect a diagnostic if you slip. Rows marked ✓ are enforced.

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

### LLM-Offload Policy (MANDATORY)

This repo enforces pre-commit LLM review on math, clinical, and external-facing artifacts. See `.claude/AGENT_OFFLOAD_POLICY.md` (canonical) and `.claude/offload-routing.md` (routing).

Required before commit:

- **Math** (PK formulas, GUM, p-box, Lean statements, refinement invariants):
  `bin/llm-offload -t math-review -i <file>` (default fan-out: Grok 4.7 + Kimi K3 via LLM Gateway, plus zai/local)
- **Clinical code** (`stdlib/clinical/*`, vancomycin*, clinical Lean obligations):
  `bin/llm-offload -t review -p deepseek -i <file>`
- **Papers / cover letters / IRB / dissertation** (anything in `docs/papers/`, `docs/dissertation/`, IRB protocols):
  `bin/llm-offload --raw <draft.md> deepseek xai gemini`

Append a log row to `.claude/llm_offload_log.md` for any non-trivial offload. Bug-catching offloads require an `LLM-offload-review:` trailer in the commit message:

```
[component] Brief description

…

LLM-offload-review:
  provider: xai (Grok 4.1 fast reasoning)
  task: math-review
  issue: <one line>
```

Never offload PHI / patient-identifying data. Never skip the policy on math-touching changes — the policy exists because it caught a real sign-error bug in `vp_cmin_point` on 2026-04-30 that had passed Lean theorem statements + 4 green tests.

---

## 6. Architecture

Pipeline: Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR (SSA) → Codegen (x86-64 ELF).

| Directory | Purpose |
|---|---|
| `self-hosted/lexer/`, `parser/` | Frontend (tokenizer, recursive descent) |
| `self-hosted/check/`, `types/` | Bidirectional type inference + algebraic effects |
| `self-hosted/ir/` | IR lowering, e-graph optimization (1000+ rewrite rules) |
| `self-hosted/native/` | x86-64 ELF emission |
| `self-hosted/compiler/` | Codegen drivers (lean, IR, GPU) |
| `self-hosted/gpu/` | PTX/GPU codegen; end-to-end CLI path exists under default Madaros (`souc build --backend gpu`, see §8) -- no GPU CLI surface under `SOUNIO_SOUC_ENGINE=lean_single`, which rejects the invocation outright |
| `stdlib/epistemic/` | `Knowledge<T>`, uncertainty (GUM), provenance |
| `stdlib/units/` | Dimensional analysis |
| `bootstrap/` | stage0 (C) → boot2g → boot3 → boot4 → self-hosted |
| `formal/` | Lean 4 proofs (epistemic type invariants) |

---

## 7. Compiler resolution

Do not hardcode ad hoc compiler routing if a repo resolver already exists.

### Canonical resolution path

Use:
- `scripts/lib/resolve_souc.sh` for the public compiler entrypoint (`bin/souc`).
- `scripts/lib/resolve_madaros.sh` for the Stage1 modular compiler (`bin/madaros`).

as the canonical compiler-resolution logic unless the task explicitly targets another resolver for cleanup or compatibility reasons.

`bin/souc` is currently a compatibility wrapper whose default engine is Madaros. The legacy lean_single engine remains the bootstrap seed and can be forced with `SOUNIO_SOUC_ENGINE=lean_single`. For the engine status and the stale-binary caveat, read `docs/MADAROS_STATUS.md`; stale raw `artifacts/self-hosted/madaros` binaries are not evidence against current `origin/main`.

---

## 8. Known limitations

Headline limitations (full list in [`docs/compiler/KNOWN_LIMITATIONS.md`](docs/compiler/KNOWN_LIMITATIONS.md)):

- **Imported-module native path — partial closeout.** Residuals remain: multi-module memory-wall / exclusive-ref fragile chains, named-import/`print_f64` papercuts. Finite-dof `gum_k95` is **trustworthy** under default Madaros.
- `Knowledge<T>` supports struct-level generics (`f64`, `bool`, struct types)
- `--show-ast` / `--show-types` on `compile` or `build` are rejected by default Madaros (`error: madaros build: unsupported option`, exit 2). Measured 2026-10-15: `check --show-ast` and `check --show-types` are **ignored** and still print `check: OK`. `SOUNIO_SOUC_ENGINE=lean_single souc check --show-ast` prints compiler stats, not a source AST. A REPL does exist (`souc repl` -> `tools/repl.sh`) -- it's a file-based compile-and-run loop over whichever engine `bin/souc` currently resolves to, not a true interactive evaluator.
- `&![T; N]` bare array mutation broken in JIT — use struct wrapper or `(*arr)[i]`
- GPU: two records, not re-run on 2026-10-15. `TOUR.md` §6 (2026-10-04) measured `souc build --backend gpu` as PTX for empty kernel bodies only. An older note in this section said `examples/kernel_vec_add.sio` emitted valid PTX under Madaros, and that `lean_single` has no GPU CLI (verified 2026-08-17). The root README (2026-10-11) records scoped Spark evidence for GPU/Pireus and says the generic GPU CLI needs its own status. Do not teach a general kernel launch from this bullet.
- **Dual-engine divergence.** `f128` is a working binary128 type on **both** engines, but they cover different surfaces. Both give 113 halvings to `1 + eps == 1` (binary64 gives 53) and print `1/3` as `3.33333333333333333333333333333333317e-0001` via `math::softfloat_f128_fmt::print_f128`.
  - **Madaros** computes `+ - * / %`, unary minus and IEEE compares, and supports params/returns, struct fields, methods, fixed arrays, exact 36-digit printing and exact literals. Measured: `7.5 % 2.0 = 1.5`; `2/3` through a struct field and an array element is correct to the last digit. It refuses, fail-closed:
    - `as` casts between `f128` and `f64` (**E248**, "wide-float casts fail closed");
    - values that flow through an `f128` global or a `type` alias over `f128` (V0-E.4.1 lowering refusal). The declarations alone build: an unused `let G: f128` or `type Wide = f128` compiles;
    - mixing an `f128` with an untyped literal or an `f256` (**E004**). Write typed constants, e.g. `let three: f128 = 3.0; x / three`, not `x / 3.0`.
  - **lean_single** (#2387/#2426) computes binary128 through libgcc for locals, direct-call params/returns, IEEE compares, and `as` casts in both directions. It refuses, fail-closed: struct fields, arrays, reading an `f128` global (an unused one builds; a `type` alias over `f128` works), tuples, methods, fn values, `println` of an `f128` (the stdlib `print_f128` works), and `%` ("modulo requires integer operands").
  - **Literal exactness differs by engine.** Measured 2026-09-26:
    - **Madaros** parses an f128 literal straight to binary128 (V0-E.5.9) and accepts the literals its parser can prove exact in binary128. Decimal digits accumulate in 128 bits (`self-hosted/parser/f128_literal.sio`), so a very long decimal is refused even when its value is exact, such as the full expansion of 2^200. `0.1` fails. `1e23` prints exactly, and so does the hex-float `0x1.0000000000000001p+0` (1 + 2⁻⁶⁴). The hex-float exponent sign is optional, as in C99: `…p0` and `…p+0` are the same literal (the unsigned form was refused with E012/E137 until the parser fix in `docs/audit/MADAROS_HEX_FLOAT_UNSIGNED_EXPONENT_DISPATCH_2026-09-26.md`).
    - **lean_single** widens through binary64 and refuses any literal not exact in binary64 (`f128_widen_refuses_inexact_literal` in `lean_single.sio`), so it also rejects `1e23` and that hex-float.
    - Portable constants: small dyadic decimals that are exact in binary64 (`0.25`, `3.0`), `f128_from_limbs(lo, hi)` (works on both), or computation (`one / ten`).
  - **Not implemented anywhere:** `Knowledge<f128>` and GUM over `f128` (KL-15).
  - **`f256`** on Madaros: declarations and some operators typecheck (an unused f256 enum field or global builds), but executable f256 value lowering is refused fail-closed (V0-E.4.1), never lowered as f64. lean_single refuses `f256` outright (KL-9). Only `stdlib/math/softfloat_f256.sio` add/sub over `F256Bits` computes f256 values (KL-15a); there is no f256 mul, div, fields, arrays or printing.
- **Cross-call first-order variance (KL-11) is narrowed, not closed.** Pure helpers carry first-order channels (ledger, 2026-10-13). A general user call still does not: `tests/run-pass/gum_fo_across_call.sio` and `tests/run-pass/fo_call_boundary_arity3.sio` keep `//@ known-failure`. Do not treat every call as variance-preserving.

---

## 9. Decision records (ADRs)

**Architecture decisions must be recorded as ADRs in `docs/decisions/` before implementation.** Check existing ADRs before proposing architecture changes.

The ADR index is at [`docs/decisions/README.md`](docs/decisions/README.md). ADR numbering is append-only; existing ADRs should not be rewritten to change history. When an ADR becomes superseded, update its status and add `Superseded by`.

---

## 10. Context7 integration

Use the `ctx7` CLI to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service -- even well-known ones like React, Next.js, Prisma, Express, Tailwind, Django, or Spring Boot. This includes API syntax, configuration, version migration, library-specific debugging, setup instructions, and CLI tool usage. Use even when you think you know the answer -- your training data may not reflect recent changes. Prefer this over web search for library docs.

Do not use for: refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

### Steps

1. Resolve library: `npx ctx7@latest library <name> "<user's question>"`
2. Pick the best match (ID format: `/org/project`) by: exact name match, description relevance, code snippet count, source reputation (High/Medium preferred), and benchmark score (higher is better). If results don't look right, try alternate names or queries (e.g., "next.js" not "nextjs", or rephrase the question)
3. Fetch docs: `npx ctx7@latest docs <libraryId> "<user's question>"`
4. Answer using the fetched documentation

You MUST call `library` first to get a valid ID unless the user provides one directly in `/org/project` format. Use the user's full question as the query -- specific and detailed queries return better results than vague single words. Do not run more than 3 commands per question. Do not include sensitive information (API keys, passwords, credentials) in queries.

For version-specific docs, use `/org/project/version` from the `library` output (e.g., `/vercel/next.js/v14.3.0`).

If a command fails with a quota error, inform the user and suggest `npx ctx7@latest login` or setting `CONTEXT7_API_KEY` env var for higher limits. Do not silently fall back to training data.
