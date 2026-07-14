# CLAUDE.md

This file is the entry-point for Claude Code (claude.ai/code) and other AI assistants working in the Sounio repository. It is the active source of truth for AI behavior; `AGENTS.md` is the Codex-facing execution contract; together they cover all AI roles in the project.

If you are a human reader: see §11.

| Quick reference | |
|---|---|
| Founder intent and collaboration contract | [`FOUNDER_INTENT.md`](FOUNDER_INTENT.md) |
| Semantic concept registry | [`docs/internal/concepts/README.md`](docs/internal/concepts/README.md) |
| Semantic lane contract | [`docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md`](docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md) |
| Recovery context | [`CLAUDE_HANDOFF.md`](CLAUDE_HANDOFF.md) |
| Codex contract | [`AGENTS.md`](AGENTS.md) |
| Programming guide | [`docs/guide/LLM_PROGRAMMING_GUIDE.md`](docs/guide/LLM_PROGRAMMING_GUIDE.md) |
| LLM cookbook | [`docs/llm-guide/`](docs/llm-guide/) |
| Minimum viable Sounio | [`docs/guide/MINIMUM_VIABLE_SOUNIO.md`](docs/guide/MINIMUM_VIABLE_SOUNIO.md) |
| Style guide | [`docs/guide/SOUNIO_STYLE_GUIDE.md`](docs/guide/SOUNIO_STYLE_GUIDE.md) |
| Gotchas | [`docs/guide/SOUNIO_GOTCHAS.md`](docs/guide/SOUNIO_GOTCHAS.md) |
| Known limitations | [`docs/compiler/KNOWN_LIMITATIONS.md`](docs/compiler/KNOWN_LIMITATIONS.md) |
| Governance | [`docs/governance/`](docs/governance/) |
| LLM offload policy | [`.claude/AGENT_OFFLOAD_POLICY.md`](.claude/AGENT_OFFLOAD_POLICY.md) |

---

## 1. Calibration — read before any analysis

Sounio is not a small experimental repository. Past AI sessions, including those with many hours of context, have consistently underestimated its scope by roughly an order of magnitude. **Calibrate before producing analysis.** The most predictable AI failure mode in this repository is measuring `stdlib/` and treating it as the whole.

Measured 2026-07-11 on `main` via `bash scripts/dev/measure_repo_scale.sh`:

| Versioned `.sio` source | Value |
|---|---:|
| Files | 6,130 |
| Lines (raw) | 2,208,306 |
| Bytes | 76 MB |

| Subsystem | Files | LOC (raw) | What it is |
|---|---:|---:|---|
| `self-hosted/` | 489 | 554,892 | The Sounio compiler (Madaros), written in Sounio |
| `stdlib/` | 1,316 | 478,355 | Math, special functions, statistics, PBPK, epistemic types, autograd, PINN, fractional calculus, RNG, I/O |
| `tests/` | 2,978 | 236,693 | Test suite |
| `examples/` | 483 | 130,370 | Working examples |
| Other | ~864 | ~808,000 | `archive/` (historical evolution), `bootstrap/` (C → Sounio chain), `benchmarks/`, tools, ecosystem |

Re-derive any number above with `bash scripts/dev/measure_repo_scale.sh` — do not quote these from memory.

Verify before disagreeing:

```bash
git ls-files -z '*.sio' | xargs -0 wc -l | tail -1
git ls-files -z '*.sio' | wc -l
```

If your measurement gives ~200k LOC, you measured `stdlib/` alone. Do not proceed under that prior.

---

## 2. Project identity

**Sounio** — a self-hosted systems + scientific programming language for epistemic computing, uncertainty propagation, and algebraic effects. Single-author development since 25 December 2025. Linux x86-64 only. Not a Rust or Julia dialect; own syntax, semantics, philosophy.

Three things are simultaneously true about this repository:

1. **It is a language.** A self-hosted compiler in `self-hosted/`, a bootstrap chain `bootstrap/stage0` (C, ~103 KB) → `boot4` → `gen1` → `gen2` → `gen3` (fixed-point verification: gen2 = gen3 bit-identical).

2. **It is a scientific computing platform.** First-class `Knowledge[T]` with GUM uncertainty propagation, Caputo fractional derivatives, autograd, PINN training, refinement types, algebraic effects (`IO`, `Mut`, `Div`, `Panic`, `Alloc`, `Async`, `GPU`, `Prob`, `Observe`), units, linear types.

3. **It is the platform for a master's dissertation in biomaterials/pharmacology** at PUC-SP (defense Aug–Sep 2026). The dissertation is one application; the language is the broader product.

---

## 3. Session bootstrap

Before non-trivial changes:

1. Read `CLAUDE_HANDOFF.md` — recovery history and workspace context
2. Verify current branch (workspace default: `integration/sounio-dev-ready-base`)
3. Do not start from `main` until reconciliation is completed
4. Do not propose destructive `reset`/`clean`/`rebase` flows on this repo

---

## 4. Build & run

The compiler is self-hosted (written in Sounio, not Rust). **`bin/souc` is the default compiler entrypoint and now routes to Madaros** — the self-hosted *modular* compiler (`artifacts/self-hosted/madaros`, built via `make build-madaros`). The legacy single-file `lean_single` engine that `bin/souc` used to be is preserved as `bin/souc-lean-single-x86_64`; force it with `SOUNIO_SOUC_ENGINE=lean_single`. lean_single remains the **bootstrap seed** (`make build`, `make build-madaros`) and the canonical fixed-point ELF — it is no longer the default *user-facing* compiler. If Madaros has not been built yet, `bin/souc` falls back to lean_single with a notice on stderr.

> **Naming (canonical): the compiler is spelled `Madaros`** — matching `make build-madaros`, `bin/madaros`, and `docs/MADAROS_STATUS.md`. The source string was fixed on 2026-07-11 (`self-hosted/compiler/main.sio`) **and the shipped ELF `bin/madaros-linux-x86_64` was rebuilt to match**, so `./bin/souc --version` now prints `Madaros v0.80.0`. (A freshly-cloned checkout that has *not* re-run `make build-madaros` locally will still show whatever the committed binary carries; on `main` that is now `Madaros`.) Current version: **v0.80.0**.

> **Fixed-point scope:** `make build` verifies the fixed point over `lean_single.sio` (the seed), **not** over `main.sio`/Madaros. Do not describe Madaros itself as fixed-point-verified.

> **CPC 2026 receipts — engine split (verify before quoting):** the two *epistemic* receipts run live under lean_single — `tests/run-pass/order_spread_exact_n4.sio` (exact N=4 spread `2.044226`) and `tests/run-pass/octonion_associator_gum_validation.sio` (GUM variance `0.640000`, abs err ~1.1e-16). The **Python↔Sounio parity delta `2.03e-10` is NOT a lean_single receipt** — it is an `omega 1.0.0-beta.4` cross-language witness (`artifacts/posters/cpc2026-yale/REPRODUCE.md`) requiring the SWOW-EN input from the sibling repo.

> **CPC 2026 Study B artifact location:** the frozen O-SSM reference `results/cpc2026/ossm_statistical_summary.json` (octonion, 10,000 traj × 500 steps, no-training) lives in the **sibling repo `hyperbolic-semantic-networks`**, *not* in this repo. The in-repo `examples/cognitive_ossm/results/ossm_sounio_native_n1000.json` is a historical native re-run that is **excluded from parity claims**: an independent same-subset audit finds up to 21.1% relative metric error. Its repaired source, `run_ossm_native_reference.sio`, passes current Madaros `check` but remains blocked in native-v2 compilation.

> **O-SSM algebra ceiling:** the frozen Study B reference and the canonical `cognitive_ossm/` recurrence are **octonion (8-D, `oct_mul`)**. Do not use separate experimental brain-model sources as evidence for the frozen CPC implementation. The largest non-associative algebra any SSM reaches today is **sedenion (16-D)** in the conversational conflict head `examples/conversational_ossm/o_ssm_conflict.sio` — it lifts the octonion state via `sed_from_pair` and calls `sed_mul` (`stdlib/algebra/sedenion.sio`) to read zero-divisor proximity (`sed_canonical_zd_z/w`); checks clean under lean_single with a live caller in `agent_cli.sio`. Do not conflate array width (`[f64;16]` softmax/sequence buffers) with algebra dimension.

```bash
SOUC=./bin/souc
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib   # required when outside repo root

$SOUC --version                           # verify toolchain
$SOUC check file.sio                      # type-check only
$SOUC run file.sio                        # compile + execute + clean up
$SOUC compile file.sio -o output.elf      # emit named ELF binary
$SOUC info                                # compiler status

# Bootstrap chain
make build    # boot4 → gen1 → gen2 → gen3, verifies gen2 == gen3
make clean    # remove generated stages
make check    # type-check compiler + CI gates
```

Testing:

```bash
bash scripts/run_sio_test_suite.sh                      # full suite
bash scripts/run_sio_test_suite.sh vancomycin --verbose # single test by pattern
bash scripts/stdlib_hyper_execution_gate.sh             # stdlib gates
bash scripts/dev/doctor_workspace.sh                    # workspace health
```

Solo / self-hosted-only workflow (skip Cargo): set `SKIP_BUILD=1` for gate scripts.

For full lint, harness annotations, and test directory layout, see [`docs/guide/SOUNIO_DEFINITIVE_GUIDE.md`](docs/guide/SOUNIO_DEFINITIVE_GUIDE.md) and [`docs/guide/CHECK_SOUNIO_GUIDE.md`](docs/guide/CHECK_SOUNIO_GUIDE.md).

### Concurrency discipline (workspace stability)

The workspace pod is recycled by the k8s liveness probe under **CPU saturation**
(not OOM, not disk). On 2026-05-29 the pod was evicted twice when multiple agents
on the shared checkout each launched a full `souc main.sio` bundle build at once;
the 15-min load hit ~153 on 64 cores. Two hard rules when more than one agent is
active:

1. **Serialize heavy builds.** Any full self-compile / bundle check
   (`souc main.sio`, `lean_single.sio`, `make build`) MUST run through the global
   build lock — never bare:
   ```bash
   scripts/dev/souc-build-lock.sh ./bin/souc self-hosted/compiler/main.sio /tmp/out.elf
   ```
   Cheap `souc check <file>` does not need the lock.
2. **One worktree per agent.** Do not run a second agent directly on
   `/workspace/sounio`. Use a dedicated worktree (see [`.claude/AGENT_HANDOFF.md`](.claude/AGENT_HANDOFF.md)).
   Recommended ceiling: **≤2 agents doing compiler work at once** on this pod.

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

Sprint cross-references:

- `examples/pbpk_rapamycin/` — CC-3 pharmacometrics proof-domain target
- `examples/octonion_nn/` — Cx-3 octonion neural-layer proof-domain target
- [`tools/mcp/examples/claude_code_usage.md`](tools/mcp/examples/claude_code_usage.md) — error → fix loop recipe

---

## 6. Operating principles

The numbered principles below are binding. Each was learned from a measured failure cycle.

1. **Measure before claiming.** Any quantitative statement about this repository must be backed by a command the operator can re-run. Never write "the codebase is small/incomplete/legacy" based on prior probability.

2. **Stubs are not gaps.** Files with low line counts, empty function bodies, or comment-only contents may be intentional structural placeholders (type signatures, design intent, future markers). Do not delete, refactor, or "complete" them without operator confirmation.

3. **Compilation is the test of existence.** A `.sio` file's status is `./bin/souc check <file>` plus the presence of a caller. Running `./bin/souc run` on a library file and reporting it broken is a category error: most files in `stdlib/` and `examples/` are libraries, not executables.

4. **Sounio is the language of this repository.** Science (data generation, statistical analysis, numerical experiments, model comparison) is implemented in Sounio. Introducing Python, JavaScript, or other languages into the science path is drift, even under time pressure. If you find yourself reaching for `import numpy`, stop. Find the Sounio primitive or ask the operator.

5. **Dispatched scope is bounded scope.** Tasks arrive as scoped dispatches. Completing a dispatch does not authorize starting the next one, even if obvious. Halt at the scope boundary and report. See [`.claude/PARALLEL_BLOCKER_CONTRACT.md`](.claude/PARALLEL_BLOCKER_CONTRACT.md).

6. **Numerical values must be derivable, not retrofitted.** When a test fails by a margin, the correct response is to tighten the implementation, broaden the bound with a published derivation, or report `FAIL_HONEST`. Selecting a tolerance because it permits the observed failure to pass is drift.

7. **Auditability over speed.** The operator runs adversarial audits on AI output. Plausible-looking output that does not survive forensic verification is worse than honest partial output. A phase completed much faster than scoped is a flag, not an achievement.

8. **Halt is a deliverable.** Stopping with a clear report of what was done, what was not done, and what blocks the next step is a complete deliverable.

9. **Q1-research first.** Literature review before architecture decisions. Cite sources; acknowledge uncertainty.

10. **Edge of novelty.** Sounio does not copy existing languages. Proposals to match Rust/Julia/Python semantics are rejected unless evidence shows the convergence is correct on first principles.

11. **No drift to mean.** Excellence only. Atomic commits — one logical change per commit. No AI attribution in commit messages.

---

## 7. Sounio syntax (NOT Rust)

Critical differences — these are compile errors:

| Wrong (Rust) | Correct (Sounio) |
|---|---|
| `let x = 5;` | `let x = 5` (no semicolons) |
| `let mut y = 10` | `var y = 10` |
| `&mut T` | `&!T` |
| `assert!(cond)` | `assert(cond)` |
| `println!("hi")` | `println("hi")` |
| `#[test]`, `#[derive()]` | No attributes |
| `-42` | `0 - 42` (no unary minus) |
| `x >> 4` | `x >> 4u8` (bit shifts require `u8`) |

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
| `self-hosted/gpu/` | PTX/GPU codegen (exists; no end-to-end CLI path) |
| `stdlib/epistemic/` | `Knowledge<T>`, uncertainty (GUM), provenance |
| `stdlib/units/` | Dimensional analysis |
| `bootstrap/` | stage0 (C) → boot2g → boot3 → boot4 → self-hosted |
| `formal/` | Lean 4 proofs (epistemic type invariants) |

Bootstrap fixed-point: stage N and N+1 produce bit-identical ELFs. Entrypoint of self-hosted compiler: `self-hosted/compiler/lean_single.sio`.

Compiler bug fixes follow the forensic dispatch protocol documented in `docs/audit/`. Do not patch `self-hosted/` ad hoc; record evidence and proposed fix as a dispatch first.

---

## 9. Documentation style

- EN-UK orthography in new documentation unless preserving quoted source text
- Papers, IRB-facing material, clinical artefacts, and external submissions follow GAIDeT-ICMJE 2025 AI disclosure pattern; update `AI_DISCLOSURE.md` per artefact
- Do not overstate semantic milestones. Report the exact command, path, compiler surface, and evidence used

---

## 10. Mandatory LLM-offload checkpoints

Pre-commit review by orthogonal LLM providers via `bin/llm-offload` is mandatory at the following checkpoints. Full policy: [`.claude/AGENT_OFFLOAD_POLICY.md`](.claude/AGENT_OFFLOAD_POLICY.md).

| Trigger | Command | Required |
|---|---|---|
| Math claims (PK/PD, GUM, p-box, Lean theorem, refinement invariants) | `bin/llm-offload -t math-review -p xai` | Yes |
| Clinical-pathway code (`stdlib/clinical/*`, vancomycin tests, clinical Lean obligations) | `bin/llm-offload -t review -p deepseek` | Yes |
| External-facing artefacts (papers, dissertation, IRB, cover letters) | `bin/llm-offload --raw <draft> deepseek xai gemini` | Yes |

Every non-trivial offload appends to `.claude/llm_offload_log.md`. Bug-catching offloads require an `LLM-offload-review:` trailer in the commit. Codex agents must not skip this step.

Optional but encouraged:

```bash
bin/llm-offload -t expand     -p gemini   -i outline.md   # outline → prose
bin/llm-offload -t scaffold   -p deepseek -i spec.md      # boilerplate
bin/llm-offload -t paraphrase -p qwen     -i letter.md    # tone shifts
bin/llm-offload --status                                  # which keys are loaded
bin/llm-offload --list-tasks                              # available tasks
```

Routing: [`.claude/offload-routing.md`](.claude/offload-routing.md). Task prompts: `.claude/offload-tasks/<task>.md`.

---

## 11. For human readers

This document is written for AI assistants. If you are human:

- **First visit:** start with the project README.
- **Researcher / collaborator:** dissertation context lives under `docs/dissertation/`; language design rationale will live under `docs/design/` (forthcoming); evolution is accessible via commit log and `archive/`.
- **Reviewer (banca, peer review, contribution evaluation):** a tailored overview is planned but not yet available; contact the author directly.

---

## 12. Session persistence

Cross-session context lives in `.claude/`:

- `decisions.md` — architectural choices
- `pending.md` — open questions, work-in-progress
- `session_state.json` — structured state
- `llm_offload_log.md` — offload audit trail

---

## 13. Known limitations

Headline limitations (full list in [`docs/compiler/KNOWN_LIMITATIONS.md`](docs/compiler/KNOWN_LIMITATIONS.md)):

- **Imported-module native path is fragile — real multi-module programs may fail to compile or *silently miscompile*.** On the default native engine (Madaros), four documented defects: `f64→i64` cast inside an imported-module body bitcasts (**silent wrong numbers** — e.g. GUM coverage factors return 1.960 for all dof); `&local_array` passed to a builtin gets a wrong base pointer (breaks `read_file`/`write_file`/`str_from_bytes`); multi-module lowering segfaults / thin-links (importing *any* module with `use` deps — incl. `Knowledge<T>`, `propagate`); named-import + `print_f64` trip E137. **Workarounds:** inline logic into `main()`, keep modules self-contained (no stdlib `use` deps), or run under `lean_single`. Which stdlib results survive native import: [`docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md`](docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md). Consolidated priority (D3→D1→D2→D4) + verification hooks: [`docs/audit/MADAROS_IMPORTED_MODULE_NATIVE_PATH_ESCALATION_2026-07-14.md`](docs/audit/MADAROS_IMPORTED_MODULE_NATIVE_PATH_ESCALATION_2026-07-14.md). Issues: #932 (D1), #933 (D2), #901/#921 (D3), #862 (D4).
- `Knowledge<T>` supports struct-level generics (`f64`, `bool`, struct types)
- No unary minus — write `0 - x`
- No REPL / `--show-ast` / `--show-types` in native mode
- `&![T; N]` bare array mutation broken in JIT — use struct wrapper or `(*arr)[i]`
- GPU: end-to-end `kernel fn` → PTX path **exists and is reproducible**. The default `bin/souc` **does** emit PTX now — `bin/souc build <file>.sio --backend gpu -o out.ptx` (verified: `examples/kernel_vec_add.sio` → valid PTX). Runtime execution is fixture-bounded (L4-validated profiles). See `docs/audit/GPU_PIPELINE_SOTA_ASSESSMENT_2026-05-30.md` for the measured/projected/source-only breakdown

---

## 14. Cluster GPU jobs

The AI/HPC cluster control plane is at `/home/devsounio/beagle/k8s/hpc-sota`. Before GPU work, read:

1. `/home/devsounio/beagle/k8s/hpc-sota/AGENT_BOOTSTRAP.md`
2. `/home/devsounio/beagle/k8s/hpc-sota/DEV_WORKFLOW.md`

Prefer proven wrappers from `ops/lab-ops.sh` over ad hoc `sbatch` or `kubectl`.

---

*This file is the AI-assistant entry-point. For the Codex-facing execution contract, see [`AGENTS.md`](AGENTS.md). For governance authority matrix, see [`docs/governance/DOCS_AUTHORITY_MATRIX.md`](docs/governance/DOCS_AUTHORITY_MATRIX.md). Last revised 17 May 2026; check `git log -1 CLAUDE.md` for current state.*
