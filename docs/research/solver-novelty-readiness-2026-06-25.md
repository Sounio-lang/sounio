<!-- docs:meta
topic_id: repo.docs.research.solver-novelty-readiness-2026-06-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.solver-novelty-readiness-2026-06-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Solver Novelty Readiness

Date: 2026-06-25

This note is a claim boundary for the SAT/SMT/PB/UNSAT solver lane. It records
what Sounio can honestly claim now, what is only a candidate novelty, and what
must be green before a public novelty claim is safe.

Scope: this is a repo-internal working note for the current `/workspace/sounio`
worktree on branch `fix/madaros-tuple-let-desugar` (observed HEAD
`4dd66d8e7` during the 2026-06-25 validation), not a standalone public
artifact. The reproduction commands assume this checkout and its `bin/souc`
wrapper. Before publication, replace live URLs with archived snapshots or
pinned source revisions.

## Current Defensible Claim

Sounio currently has source-level and checked proof-profile plumbing in a
self-hosted language, plus local SMT heuristic experiments that now run green
through the canonical `bin/souc` modular path. The strongest defensible claim is
still architectural and internal:

> Source-level proof-profile gates separate solver production from certificate
> acceptance across SAT-style proofs, PB/VeriPB-shaped proofs, SMT/Farkas
> receipts, and numeric receipts.

This is not a claim of state-of-the-art SAT/SMT/PB solving performance, not a
new Hadwiger-Nelson lower bound, not a public theorem, not a complete
cross-family checker implementation, and not evidence for `χ(R^2) >= 6`.

## External SOTA Anchors

- Live web check on 2026-06-25: Erdős Problems #90 is the planar unit-distance
  counting problem ("how many unit-distance pairs can occur among `n` points"),
  not the Hadwiger-Nelson chromatic-number problem. The Bloom database currently
  marks #90 as "DISPROVED (LEAN)" and points to the May 2026 OpenAI/internal-model
  proof (`https://www.erdosproblems.com/90`,
  `https://openai.com/index/model-disproves-discrete-geometry-conjecture/`).
- Hadwiger-Nelson remains a separate chromatic-number question. The safe public
  boundary is still: de Grey proved the lower bound `χ(R^2) >= 5` in 2018 via a
  finite unit-distance graph that is not 4-colourable, while the classical upper
  bound is 7 (`https://arxiv.org/abs/1804.02385`). This repo has no evidence for
  `χ(R^2) >= 6`.
- Live web check on 2026-06-25: VeriPB is a general proof format for
  proof-logging algorithms over pseudo-Boolean reasoning
  (`https://veripb.org/`); SAT Competition 2026 documentation proposes VeriPB
  with CakePB as a formally verified backend
  (`https://satcompetition.github.io/2026/downloads/checkers/veripb.pdf`).
- SAT proof tooling already includes LRAT/LPR-style verified checking and FRAT
  as an elaboration-friendly SAT proof format
  (`https://satcompetition.github.io/2025/output.html`,
  `https://www.cs.cmu.edu/~mheule/publications/FRAT-TACAS.pdf`).
- Live web check on 2026-06-25: PB Competition 2026 certified tracks require
  UNSAT/OPT certificates in the VeriPB format and rank proof validity
  separately from raw answers
  (`https://www.cril.univ-artois.fr/PB26/`).
- Live web check on 2026-06-25: SMT-COMP emphasizes SMT-LIB common formats and
  common proof/model outputs for skeptical proof assistants
  (`https://smt-comp.github.io/2026/introduction/`).
- Live web check on 2026-06-25: Alethe is SMT-LIB-shaped and intended as a
  uniform SMT proof format; proof reconstruction work includes Isabelle as an
  independent checker for SMT-LIB problems
  (`https://verit.gitlabpages.uliege.be/alethe/specification.pdf`,
  `https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2025.26`);
  cvc5 also documents Alethe proof output and proof-interface work
  (`https://cvc5.github.io/docs/cvc5-1.0.0/proofs/output_alethe.html`,
  `https://cvc5.github.io/blog/2024/03/15/isabelle-reconstruction.html`,
  `https://cvc5.github.io/blog/2024/04/15/interfaces-for-understanding-cvc5.html`).
- Bandit methods in SAT/CDCL are not new: VSIDS, CHB/ERWA, LRB, global
  learning-rate work, restart-mediated VSIDS/CHB switching, and Thompson
  reset-policy work all predate this note. The live matrix
  `docs/research/solver-heuristic-literature-matrix-2026-06-25.md` records the
  current claim boundary. In short, LRB frames branching as a
  multi-armed-bandit-style learning-rate problem, and one recent CDCL
  reset-policy paper uses Thompson Sampling for reset decisions
  (`https://cs.uwaterloo.ca/~ppoupart/publications/sat/learning-rate-branching-heuristic-SAT.pdf`,
  `https://arxiv.org/html/2404.03753v2`). Therefore broad
  "first-in-class Thompson Sampling DPLL" wording is unsafe. The narrow
  candidate wording is: "Thompson-sampling-based variable-score and
  polarity-selection experiments in Sounio's bounded DPLL(T), with
  source-level epistemic/proof-profile instrumentation, pending a fuller
  related-work review."

## Sounio Evidence Present

- `stdlib/theorem/solver_proof_profile.sio` defines gates that reject
  solver-only results unless instance digest, proof/receipt, producer trace,
  independent checker, kernel replay, and scope gate are all present.
  The core predicate is: `solver_result == 0`, accepted domain/format family,
  `instance_digest_ok == 1`, `proof_object_present == 1`,
  `producer_trace_ok == 1`, `independent_checker_ok == 1`,
  `kernel_replay_ok == 1`, `scope_gate_ok == 1`, plus
  `formal_theorem_ready == 1` whenever `public_claim_mask != 0`.
- `stdlib/theorem/sat_rup_microkernel.sio`, `smt_farkas_microkernel` artifacts,
  and `solver_proof_profile` gates give microkernel seeds for proof-checking
  families. These are seeds, not full LRAT/FRAT/VeriPB/Alethe implementations.
- `stdlib/theorem/smt.sio` contains source surfaces for epistemic activity,
  Thompson-style score sampling, Beta-Bernoulli polarity sampling, regime
  labels, and warm restart controls. In the current worktree, both the rebuilt
  isolated raw/native artifact (`/tmp/madaros-novelty-trace-test55`) and the
  canonical modular wrapper (`./bin/souc` resolving to
  `artifacts/self-hosted/madaros`) run all six focused imported
  `tests/stdlib/theorem/test_smt_*` harnesses green through the official suite
  runner. The canonical promotion required native-v2 f64 call-result metadata
  and lifting per-function IR capacity from 512 to 1024 instructions; the
  previous 139s were compiler/runtime substrate failures, not SMT mathematical
  failures. This is focused internal runtime evidence, not external benchmark
  evidence.
- `stdlib/theorem/solver_novelty_readiness.sio` now encodes the difference
  between private readiness and public novelty readiness.

## What Is Missing For Novelty

1. Keep the focused gates green through the canonical wrapper path. The
   current canonical wrapper parity is private Level 2 evidence, not a public
   solver result.
2. Broaden ablations beyond the current local harnesses. A medium local harness
   now covers mean-only, adaptive GUM-UCB, Wilson/LRB, Thompson score,
   Beta-polarity, restart-budget, and epistemic-LRB variants with fixed seeds,
   repeated runs, per-row wall-clock timing, decisions, conflicts, restarts, and
   aggregation. It is still not paper-grade because the slice is generated and
   small rather than a broad benchmark corpus.
3. External benchmarks against real solvers or at least reproducible
   DIMACS/SMT-LIB/OPB benchmark slices with fixed seeds, timeouts, and result
   tables. A real-corpus pilot scaffold now exists, but the first smoke run is
   evidence-incomplete rather than Level 3 evidence: it fetched two SATComp
   2024/GBD `track_main_2024.uri` CNFs, found only `z3` on `PATH`, recorded one
   Z3 `UNSAT` and one Z3 timeout at 10 seconds, and hit a current imported
   native compile/write blocker for the generated Sounio DIMACS harnesses that
   import `theorem::smt`.
4. Real proof format work beyond microkernels: the current RUP, LRAT-hint, and
   Farkas CSV file replays are tiny local subsets; Level 3 still needs full
   LRAT/FRAT parser/checker path or independent checker replay, Alethe subset
   or export/replay path, and VeriPB/CakePB-shaped PB subset.
5. Stable imported-native compiler/runtime path for the test shape used in the
   evidence harnesses: no `/tmp`-only compiler artifact, no trace-dependent
   Heisenbug concern, no large-code truncation, no post-test segfault, and no
   stale known-failure annotation.
6. Literature review narrowing. A first guardrail matrix now exists at
   `docs/research/solver-heuristic-literature-matrix-2026-06-25.md`, but this
   is still not a paper-grade survey. Safe wording is "candidate epistemic
   variable-score and polarity-sampling DPLL(T) experiment with Sounio
   proof-profile instrumentation"; unsafe wording is any broad "first Thompson
   Sampling DPLL/CDCL solver" claim.

## Readiness Levels

- Level 1: idea or architecture only. Interesting, not evidence.
- Level 2: private novelty candidate. Feature exists, focused tests pass,
  proof-profile/scope gates pass, but ablations or external benchmarks are
  missing.
- Level 3: public novelty candidate. Level 2 plus ablation tables, external
  benchmark slice, independent checker path, literature review, and stable
  compiler/runtime gates.

As of this note, the proof-profile discipline is a Level 2 internal architecture
candidate: the module checks and the self-contained runtime gate passes, but
this is still not a public solver result. The epistemic SMT heuristic lane is
now Level 2 private runtime evidence in the canonical wrapper as well as the
rebuilt raw/native path: feature surfaces exist and the focused imported
runtime harnesses pass. It remains below Level 3 until paper-grade ablation
tables, broader external benchmarks, independent checker/proof paths, and
literature review are green. The canonical Beta-TS polarity smoke test is a
characterisation gate,
not evidence of robust dominance: on the current canonical artifact it reports
`mode=1 wins / mode=0 wins / ties = 44 / 47 / 9`.

Update 2026-06-26: the external-benchmark lane now has a real-corpus pilot
package, still below Level 3. The safe wording remains:
"experiment DPLL(T) bounded with epistemic variable-score/polarity-sampling and
proof-profile instrumentation." Do not call this SOTA and do not claim
`χ(R^2) >= 6`.

Pilot smoke command:

```bash
python3 scripts/research/run_solver_external_corpus_pilot.py \
  --sat-count 2 --smt-count 0 --opb-count 0 --timeout-sec 10 \
  --out-dir /tmp/sounio-solver-pilot-smoke
```

Result:

```text
schema=sounio.solver.external_corpus_pilot.run.v1
pilot_sat_acceptance_candidate=0
pilot_evidence_incomplete=1
```

Observed smoke details:

- SAT corpus: 2 real SATComp 2024 CNFs selected via GBD
  `track_main_2024.uri`, both expected `UNSAT`.
- External solvers: `z3` available; `cadical` and `kissat` absent, so the SAT
  multi-solver gate is not satisfied.
- Z3 results: one `UNSAT`, one `TIMEOUT` at 10 seconds.
- Sounio generated DIMACS harnesses: generation succeeded, but compile/run is
  blocked on the current imported native path (`Failed to write native binary
  ... rc=13`) for harnesses importing `theorem::smt`.
- Crosscheck: no definitive Sounio-vs-external SAT rows because Sounio harness
  binaries did not build on this path.
- SMT/OPB: not run in the smoke. The pilot fetcher records explicit blockers
  rather than generated replacements when a small SMT-LIB subset or PB solver is
  unavailable.

Bounded main pilot command:

```bash
python3 scripts/research/run_solver_external_corpus_pilot.py \
  --sat-count 12 --smt-count 12 --opb-count 12 --timeout-sec 30 \
  --max-sat-candidates 80 \
  --out-dir /tmp/sounio-solver-pilot-main-bounded
```

Result:

```text
pilot_sat_acceptance_candidate=0
pilot_evidence_incomplete=1
```

The bounded main run considered 81 GBD rows, found 38 small-download SAT
candidates, and selected 2 harness-compatible real CNFs under the current
`n_vars <= 500`, `n_clauses <= 4096`, `max_clause_width <= 8` Sounio harness
limits. SMT recorded `no_small_direct_smtlib_zenodo_subset_configured_under_512MiB_for_this_pilot`.
OPB recorded `no_pb_solver_available_on_path_or_user_cache`.

## Executable Receipt

- Module: `stdlib/theorem/solver_novelty_readiness.sio`
- Runtime gate: `tests/run-pass/solver_novelty_readiness_tiny.sio`
- Tiny internal ablation emitter:
  `benchmarks/solver/smt_ablation_tiny.sio`
- Tiny internal ablation runner:
  `scripts/research/run_solver_ablation_tiny.sh`
- Medium internal ablation emitter:
  `benchmarks/solver/smt_ablation_medium.sio`
- Medium internal ablation runner:
  `scripts/research/run_solver_ablation_medium.py`
- Tiny DIMACS external-baseline scaffold:
  `scripts/research/run_solver_external_dimacs_tiny.py`
- Tiny Sounio-vs-external DIMACS crosscheck:
  `scripts/research/run_solver_dimacs_crosscheck_tiny.py`
- Medium Sounio-vs-external DIMACS crosscheck:
  `scripts/research/run_solver_dimacs_crosscheck_medium.py`
- Tiny Sounio QF_LIA emitter for SMT-LIB crosscheck:
  `benchmarks/solver/smtlib_qflia_crosscheck_tiny.sio`
- Tiny SMT-LIB external-baseline scaffold:
  `scripts/research/run_solver_external_smtlib_tiny.py`
- Tiny Sounio-vs-external SMT-LIB QF_LIA crosscheck:
  `scripts/research/run_solver_smtlib_crosscheck_tiny.py`
- Tiny OPB external-baseline scaffold:
  `scripts/research/run_solver_external_opb_tiny.py`
- Real external-corpus pilot bootstrap:
  `scripts/research/bootstrap_solver_external_tools.py`
- Real external-corpus pilot fetcher:
  `scripts/research/fetch_solver_external_corpus_pilot.py`
- DIMACS-to-Sounio harness generator:
  `scripts/research/generate_sounio_dimacs_harness.py`
- Real external-corpus pilot runner:
  `scripts/research/run_solver_external_corpus_pilot.py`
- Certificate/proof tool availability check:
  `scripts/research/check_solver_certificate_tools.py`
- Tiny SAT/RUP file-level replay scaffold:
  `scripts/research/run_solver_sat_rup_replay_tiny.py`
- Tiny SAT/LRAT hint replay scaffold:
  `scripts/research/run_solver_sat_lrat_replay_tiny.py`
- Tiny SMT/Farkas file-level replay scaffold:
  `scripts/research/run_solver_smt_farkas_replay_tiny.py`
- Claim-control literature matrix:
  `docs/research/solver-heuristic-literature-matrix-2026-06-25.md`
- Current readiness fingerprint in the tiny gate: `402331456`

Latest current-source validation on 2026-06-25:

```bash
SOUNIO_FORCE_SOURCE_BOOTSTRAP=1 \
  scripts/ci/build_native_souc.sh /tmp/madaros-novelty-trace-test55

SOUNIO_TEST_SOUC_BIN=/tmp/madaros-novelty-trace-test55 \
  ./scripts/run_sio_test_suite.sh test_smt --verbose --jobs 1
```

Result:

```text
PASS test_smt_adaptive_epistemic.sio
PASS test_smt_beta_polarity_ts.sio
PASS test_smt_epistemic_eval.sio
PASS test_smt_regime_getters.sio
PASS test_smt_solver_basic.sio
PASS test_smt_thompson_stats.sio
Pass: 6; Fail: 0; Total: 6
```

The same artifact also builds and runs
`tests/run-pass/solver_novelty_readiness_tiny.sio` with exit code 0.

Canonical modular-wrapper validation on 2026-06-25 before the native-v2 text
mirror repair, after rebuilding
`artifacts/self-hosted/madaros` with:

```bash
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
sha256sum artifacts/self-hosted/madaros
./scripts/run_sio_test_suite.sh test_smt --verbose --jobs 1
./bin/souc compile tests/run-pass/solver_novelty_readiness_tiny.sio \
  -o /tmp/sounio-solver-novelty-tiny-default
/tmp/sounio-solver-novelty-tiny-default
```

Result:

```text
artifacts/self-hosted/madaros:
  ed173301a90166973a2f754e32990da72c28ab5a631b59d82eab03be8088a51e

PASS test_smt_adaptive_epistemic.sio
PASS test_smt_solver_basic.sio
FAIL test_smt_beta_polarity_ts.sio (run exited 139)
FAIL test_smt_epistemic_eval.sio (run exited 139)
FAIL test_smt_regime_getters.sio (run exited 139)
FAIL test_smt_thompson_stats.sio (run exited 139)
Pass: 2; Fail: 4; Total: 6

solver_novelty_readiness_tiny: compile exit 0, runtime exit 0
```

That specific capacity diagnosis was real but incomplete. With
`SOUNIO_NV2_IR_TRACE=1`, the larger `test_smt_epistemic_eval.sio` ELF placed
`smt_add_lia` at text offset `131039` and `smt_get_assignment` at exactly
`131072`, matching the old `CodeBuffer` ceiling. A naive global `CodeBuffer`
widening to 256 KiB created large by-value aggregate pressure and failed the
self-hosted build typecheck.

Follow-up canonical rebuild on 2026-06-25 added a native-v2 direct-to-file text
mirror in `self-hosted/native/codegen_x86_linux.sio` so the emitted `.text`
stream, relocation opcode checks, label patches, and ELF copy path no longer
depend solely on `NativeCompiler.code.bytes[131072]`. Rebuild command:

```bash
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
sha256sum artifacts/self-hosted/madaros
./bin/souc compile tests/run-pass/solver_novelty_readiness_tiny.sio \
  -o /tmp/sounio-solver-novelty-tiny-default-wideglobal-labels
/tmp/sounio-solver-novelty-tiny-default-wideglobal-labels
./scripts/run_sio_test_suite.sh test_smt --verbose --jobs 1
```

Result:

```text
artifacts/self-hosted/madaros:
  912ee55813b25afd5df538e3926ae4f5f86cab7240a83a8e2389b57a758a21af

solver_novelty_readiness_tiny: compile exit 0, runtime exit 0

PASS test_smt_adaptive_epistemic.sio
PASS test_smt_solver_basic.sio
FAIL test_smt_beta_polarity_ts.sio (run exited 139)
FAIL test_smt_epistemic_eval.sio (run exited 139)
FAIL test_smt_regime_getters.sio (run exited 139)
FAIL test_smt_thompson_stats.sio (run exited 139)
Pass: 2; Fail: 4; Total: 6
```

The text-mirror repair changes the failure frontier but does not clear it.
`test_smt_epistemic_eval.sio` now emits `smt_get_assignment` at text offset
`132041` and produces a 143368-byte ELF, proving that this path is no longer
stopping exactly at the 128 KiB cliff. The binary executes through multiple
PHP/K4/random-3SAT solver checks and prints PASS diagnostics before segfaulting.
Under `gdb`, the observed crash PC was `0x406e8a`, inside `smt_add_clause`
with an invalid context-like value (`rax=1`). `test_smt_beta_polarity_ts.sio`
prints a real logical failure before crashing:
`T3 FAIL: posteriors not reset to 1.0/1.0 after smt_reset_activities`, and one
`gdb` run ended at `rip=0`. `test_smt_regime_getters.sio` and
`test_smt_thompson_stats.sio` also execute enough to print diagnostics before
their final `139`.

Therefore the current canonical blocker is now a mixed compiler/runtime and
solver-state issue after large-code materialization, not a mathematical solver
result and not public novelty evidence.

Validated on 2026-06-25 in `/workspace/sounio`:

```bash
./bin/souc check stdlib/theorem/solver_novelty_readiness.sio
./bin/souc run tests/run-pass/solver_novelty_readiness_tiny.sio
./scripts/run_sio_test_suite.sh solver_novelty_readiness --verbose
./bin/souc check stdlib/theorem/smt.sio
./bin/souc check tests/stdlib/theorem/test_smt_beta_polarity_ts.sio
./bin/souc check tests/stdlib/theorem/test_smt_regime_getters.sio
./bin/souc check tests/stdlib/theorem/test_smt_thompson_stats.sio
./bin/souc check tests/stdlib/theorem/test_smt_adaptive_epistemic.sio
./bin/souc check tests/stdlib/theorem/test_smt_epistemic_eval.sio
```

Historical blocker observed earlier on 2026-06-25 before the current-source
writer/effect cleanup:

```bash
bash scripts/ci/madaros_multimodule_witness.sh
./bin/souc check tests/multimodule/thin_single_main.sio
./bin/souc run tests/multimodule/thin_single_main.sio
./bin/souc run tests/stdlib/theorem/test_smt_beta_polarity_ts.sio
./bin/souc run tests/stdlib/theorem/test_smt_regime_getters.sio
./bin/souc run tests/stdlib/theorem/test_smt_thompson_stats.sio
./bin/souc run tests/stdlib/theorem/test_smt_adaptive_epistemic.sio
./bin/souc run tests/stdlib/theorem/test_smt_epistemic_eval.sio
```

The official multimodule witness fails first: `thin_single_main.sio` imports
`thin_single_lib::{add_public}` and should return `7`. On the default checked-in
path at the start of this investigation, current Madaros exited 139 immediately
after `imported_compile: lower_begin` / `lower_array: seed_begin`. The same
failure shape appeared in the five epistemic SMT runtime harnesses.

The current-source rebuilt path supersedes this earlier runtime blocker for the
focused SMT harnesses, but publication claims still require default-wrapper
promotion and external benchmark evidence.

Follow-up diagnostic on 2026-06-25 using rebuilt isolated Madaros artifacts:

```bash
bash scripts/ci/build_modular_madaros.sh /tmp/madaros-novelty-lower-array-test
SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1 \
  SOUNIO_LOWER_BODY_TRACE=1 \
  MADAROS_BIN=/tmp/madaros-novelty-lower-array-test \
  bash scripts/ci/madaros_multimodule_witness.sh
```

A local source experiment in `self-hosted/compiler/module_frontend.sio` and
`self-hosted/ir/lower.sio` avoided copying whole `Program` values in
`module_frontend_lower_programs_array_boxed` and moved body lowering to borrowed
item-list traversal. Rebuilding Madaros from that source advanced the witness
past the previous segfault and through both seed and dependency body lowering.

The current diagnostic frontier is no longer "crash at seed_begin". With
`ir_patch_validated_calls` temporarily bypassed in the new item-list diagnostic
path, the witness reaches:

```text
module_frontend_lower: body_fn_count 2
module_frontend_lower: body_fn_count 1
lower_array: merge_begin 1
lower_array: merge_done 1
Merged IR: 0 functions
IR lowering produced empty module
```

The trace also shows both bodies lower completely:

```text
fn_mut_done name=main
fn_mut_done name=add_public
body_stage skip_patch_for_imported_probe
body_stage before_result
```

Therefore the imported/native blocker is narrowed, not cleared. There are now
two separate compiler/runtime issues blocking public solver evidence:

1. `ir_patch_validated_calls` still needs a real fix; bypassing it removed the
   segfault but is only diagnostic and not production-correct.
2. `ModuleFrontendLowerBoxResult` / `Option<Box<IrModule>>` propagation loses
   the non-empty lowered modules before the final native compile step, yielding
   `Merged IR: 0 functions` even though the seed/dependency body results report
   nonzero function counts.

Later diagnostic update on the same day narrowed this again. A direct
`Box<IrModule>` return path plus an imported-source route through
`module_frontend_compile_imported_to_file` produced an isolated Madaros binary at
`/tmp/madaros-novelty-lower-array-test10`. With that binary, the multimodule
witness no longer fails in front-half lowering:

```text
imported_compile: lower_done
Merged IR: 2 functions
Written to a.out
native_v2_compile: emitted path=a.out
```

The remaining witness failure moved to run semantics:

```text
[madaros-mm-witness] FAIL: thin_single expected_exit=7 actual_exit=0
```

Manual execution of the emitted `a.out` returned the expected program exit
(`7`), showing this was a CLI handoff issue rather than a backend/named-call
semantic failure for the first witness.

The witness script was then narrowed to the artifact property we need for
solver-runtime evidence: for `gate_mode=run`, it now checks the source, builds an
import-aware ELF with `MADAROS_BIN build -o`, executes that ELF directly, and
compares the process exit code. This does not claim that `MADAROS_BIN run`
itself has correct execution semantics.

Additional live check later on 2026-06-25:

```bash
./bin/souc check stdlib/theorem/smt.sio
./bin/souc check tests/stdlib/theorem/test_smt_beta_polarity_ts.sio
./scripts/run_sio_test_suite.sh test_smt --verbose
```

The first two checks pass. The focused `test_smt` runtime suite still fails with
five imported SMT harnesses exiting 139:

```text
FAIL test_smt_adaptive_epistemic.sio (run exited 139)
FAIL test_smt_beta_polarity_ts.sio (run exited 139)
FAIL test_smt_epistemic_eval.sio (run exited 139)
FAIL test_smt_regime_getters.sio (run exited 139)
FAIL test_smt_thompson_stats.sio (run exited 139)
```

One candidate compiler fix was identified and applied locally: imported
cross-module struct preseed now preserves `is_float` metadata for `f64` and
`[f64; N]` fields instead of hardcoding every imported field as integer-like.
This is independently correctness-preserving for imported field lowering, but it
does not by itself clear the SMT runtime segfault. A targeted statement-level
trace also changed the crash location, so statement traces from that build are
treated as Heisenbug diagnostics and must not be cited as stable evidence.

Temporary `/tmp` reductions from earlier in the investigation should be treated
as volatile unless re-run in the same compiler artifact. Re-checking with
rebuilt artifacts showed that several previously noted "pass" reductions no
longer passed, so the stable claim is narrower: imported SMT checks are green;
imported native runtime evidence remains blocked by exit 139.

Validated with `/tmp/madaros-novelty-lower-array-test10`:

```bash
MADAROS_BIN=/tmp/madaros-novelty-lower-array-test10 \
  bash scripts/ci/madaros_multimodule_witness.sh
```

Result:

```text
[madaros-mm-witness] PASS: thin_single (run)
[madaros-mm-witness] PASS: visibility_struct_pub (run)
[madaros-mm-witness] PASS: math_main (run)
[madaros-mm-witness] PASS: thin_chain_compile (compile)
[madaros-mm-witness] PASS: thin_inline_compile (compile)
[madaros-mm-witness] PASS: 5/5 witnesses
```

The current frontier is therefore no longer "can the imported module lower, emit
an ELF, and execute with the expected exit code?" It can, under the diagnostic
route and artifact-execution witness. Remaining compiler debt before using this
as public solver evidence:

1. `MADAROS_BIN run` still compiles and returns the compiler exit rather than
   executing the emitted program.
2. The `ir_patch_validated_calls` bypass in the item-list diagnostic route must
   be replaced with a real fix.
3. The imported epistemic SMT runtime harnesses still need to run green through
   this repaired path; the multimodule witness is necessary compiler substrate
   evidence, not solver-performance evidence by itself.

After type-discipline cleanup in the five imported SMT harnesses, each focused
file passes `check`:

```text
CHECK_OK tests/stdlib/theorem/test_smt_beta_polarity_ts.sio
CHECK_OK tests/stdlib/theorem/test_smt_regime_getters.sio
CHECK_OK tests/stdlib/theorem/test_smt_thompson_stats.sio
CHECK_OK tests/stdlib/theorem/test_smt_adaptive_epistemic.sio
CHECK_OK tests/stdlib/theorem/test_smt_epistemic_eval.sio
```

However, the official suite filter is still red at runtime:

```bash
./scripts/run_sio_test_suite.sh test_smt --verbose
```

On 2026-06-25 this reports `Fail: 5`, `Known failures: 1`; the five failures are
the focused imported SMT experiments exiting 139.

Direct build+execute with the isolated Madaros binary also fails before an ELF is
produced. With trace enabled, the seed test lowers (`body_fn_count 15`), then
the dependency `stdlib/theorem/smt.sio` enters `bodies_begin` and segfaults. The
last function recorded in `/tmp/sounio_lower_body.trace` before the crash is
`smt_adaptive_beta_v2`; by source order the next body is `smt_wilson_score`.
That is a narrowing clue, not yet root-cause proof.

Follow-up instrumentation changed that assessment. With
`/tmp/madaros-novelty-lower-array-test11` the dependency body traversal first
appeared to stop at:

```text
body_mut top idx=19
body_mut kind=0
body_mut before_name
body_mut_fn name=smt_adaptive_beta_v2
```

and then segfault before `fn_mut_begin`. A local source patch copied only the
current `Item` from the list node before reading `fn_def`/`impl_def`, but
`/tmp/madaros-novelty-lower-array-test12` still exited 139.

A cleaner read-only reduction from a second agent found a smaller reproducer:

```sio
//@ run-pass
use theorem::smt::*

fn main() -> i32 with Mut {
    let _ctx = smt_new()
    return 0
}
```

That file passes `check` but `build` exits 139 while lowering the imported
dependency. With the non-intrusive body-function trace, the dependency gets
through `smt_adaptive_beta_v2`, `smt_wilson_score`, and `smt_decay_activity`,
then enters `smt_update_regime` and dies after parameter lowering, before
`fn_mut_after_block`.

Further reduction under `/tmp` indicates that `smt_update_regime`'s final
trust/label block is the first sensitive pattern: branching on `f64`
comparisons such as `regime_conflict_rate > 0.5` and
`regime_explore_trust < 0.3`, then assigning an `i32` regime label. Removing or
extracting that block can move the crash, but that is diagnostic only; it does
not establish a correct solver fix. The blocker is therefore compiler/runtime
lowering of imported SMT bodies with `f64` control flow, not a SAT/SMT
mathematical failure.

A temporary statement-level trace in `lower_block_ref` was intentionally
discarded: it perturbed the crash point earlier to `smt_lrb_blame`/
`smt_decay_activity`, so it was too intrusive for root-cause evidence.

Latest diagnostic update on 2026-06-25:

- Temp reductions of `stdlib/theorem/smt.sio` are only valid when the compiler is
  explicitly pointed at the temp stdlib with `SOUNIO_STDLIB_PATH=/tmp/.../stdlib`.
  Earlier fresh-cut experiments that omitted this environment variable still
  imported the repo's canonical stdlib and are not evidence.
- With `SOUNIO_STDLIB_PATH=/tmp/sounio-smt-freshcut-empty/stdlib` and an empty
  `smt_update_regime` body, `/tmp/madaros-novelty-borrowedcanon-test23` still
  exited 139. That build lowered `smt_update_regime` completely through
  `fn_mut_done name=smt_update_regime`, then crashed immediately after the
  `body_mut before_tail` marker. This implicated the diagnostic
  `&! Lowerer` body traversal, not the SMT algorithm.
- Switching the imported boxed-summary path back to the existing
  `lo2 = lo2.lower_program_bodies_ref(items)` method produced
  `/tmp/madaros-novelty-methodbodies-test24`. That build advanced beyond
  `smt_update_regime`, through `smt_blame_var_phase` and `smt_bump_clause`, and
  then exited 139 inside `smt_conflict_clause`. This is a real narrowing: the
  previous custom mutating traversal was part of the crash surface.
- In the same `test24` artifact, temporarily replacing `smt_conflict_clause`
  with `return` in the `/tmp` stdlib reduction advanced the crash into
  `smt_assign_var`. That shows the blocker is not one solver function or one
  mathematical rule; it follows imported SMT bodies until it hits another rich
  mutable-context body.
- `/tmp/madaros-novelty-blockstmt-test25` added `lower_live:
  block_stmt_kind=...` tracing inside `lower_block_ref`. The extra trace
  perturbed the crash earlier to `smt_decay_activity`, ending after an
  `assign` statement. Treat this as a Heisenbug diagnostic only. Its useful
  signal is the repeated crash class: imported bodies with `&!SmtContext`,
  field/array assignments, nested control flow, and `f64` arithmetic/control
  remain fragile in native lowering.
- `/tmp/madaros-novelty-assignborrow-test26` changed assignment lowering to
  keep assignment targets borrowed (`&s.target`) instead of moving target
  expressions out of borrowed statements. The compiler source checked cleanly
  and the isolated Madaros artifact built, but the minimal imported SMT fixture
  still exited 139 at the same frontier as `test24`: inside
  `smt_conflict_clause`. This is a correctness-oriented lowering cleanup, not a
  solver-readiness clearance.
- A temp-source reduction under `test26` narrowed the first sensitive block in
  `smt_conflict_clause`: removing the whole Beta-Bernoulli nested block lets
  that function lower completely and moves the crash to `smt_assign_var`;
  retaining nested `if` structure without indexed field reads also lets
  `smt_conflict_clause` lower. The crash returns when nested conditions read
  imported struct array fields such as `ctx.assign[var_idx] > 0`. The likely
  compiler surface is therefore lowering/borrowing of imported
  `Index(FieldAccess(...))` expressions in mutable-context bodies, not the
  mathematical validity of the Beta update.
- `/tmp/madaros-novelty-ifborrow-test27` changed `if`/loop/block expression
  lowering to borrow optional blocks and else expressions instead of copying
  `Option<Box<...>>` values out of a borrowed `Expr`. The compiler source
  checked cleanly and the isolated Madaros artifact built. The original minimal
  imported SMT fixture still exited 139. With `SOUNIO_LOWER_LIVE_TRACE=1`, the
  last stable live trace is inside `smt_conflict_clause` parameter processing:
  `param_loop name=clause_idx` after `ctx` was bound. Treat this as another
  crash-frontier perturbation, not as evidence that parameters are the semantic
  root cause.
- `/tmp/madaros-novelty-paramlocal-test28` changed the mutating parameter
  lowering helper to mirror the existing `lower_fn_params_ref` copy-modify-
  writeback style for `current_func` and `locals`. This avoids repeated
  `&! Lowerer` writes while walking imported parameter lists. The source
  checked cleanly and the isolated Madaros artifact built; the multimodule
  witness still passed `5/5` under `MADAROS_BIN=/tmp/madaros-novelty-paramlocal-test28`.
  The minimal imported SMT fixture still exited 139 at `smt_conflict_clause`
  before `fn_mut_after_params`, so this is substrate cleanup/regression evidence
  only, not a solver-runtime clearance.
- Re-running the same minimal imported SMT fixture without
  `SOUNIO_LOWER_BODY_TRACE` still exited 139, so the body trace is not the
  cause of the crash. A short `gdb -batch` run on
  `/tmp/madaros-novelty-paramlocal-test28` caught a raw SIGSEGV with an
  unsymbolized stack (`#0 0x0000000005a44b7c`, followed by a suspicious
  `0xfffffffffffffff4` frame). This is ABI/lowering evidence only; it does not
  identify a solver bug.
- A proposed `/tmp` stdlib reduction was invalidated: replacing
  `/tmp/sounio-smt-freshcut-empty/stdlib/theorem/smt.sio` with a minimal module
  still caused the compiler to lower the repo's real `stdlib/theorem/smt.sio`
  functions. Current source inspection explains why: the imported module
  resolver in `self-hosted/compiler/module_frontend.sio` and
  `self-hosted/compiler/module_loader.sio` constructs `stdlib/...` paths
  directly. Therefore temp-stdlib reductions are not trustworthy unless the
  resolver path is explicitly changed or the canonical workspace stdlib is
  temporarily patched and restored in the same run.
- `/tmp/madaros-novelty-stdlibenv-test29` adds that resolver repair: the
  imported-module resolvers in `self-hosted/compiler/module_frontend.sio` and
  `self-hosted/compiler/module_loader.sio` now try `SOUNIO_STDLIB_PATH` first
  and fall back to the existing literal `stdlib` root when the environment is
  unset. The isolated Madaros artifact built successfully, and the multimodule
  witness stayed green (`PASS: 5/5 witnesses`). With this artifact, running from
  `/workspace/sounio` while setting
  `SOUNIO_STDLIB_PATH=/tmp/sounio-smt-freshcut-empty/stdlib` correctly lowered a
  minimal temp `theorem::smt` module with only four functions. This is
  diagnostic-infrastructure progress only; it does not clear the full SMT
  runtime.
- Re-running the full temp SMT snapshot with the env-aware resolver still exited
  139 in `smt_conflict_clause` after parameter lowering. The better current
  narrowing is: removing the Beta-Bernoulli indexed-field block lets
  `smt_conflict_clause` lower and moves the crash to `smt_assign_var`; reducing
  both `smt_conflict_clause` and `smt_assign_var` to returns moves the crash to
  `smt_assign_literal`. This supports the broad compiler/runtime diagnosis
  (imported mutable SMT bodies remain fragile) and argues against treating any
  single SMT heuristic as the root mathematical issue.

This supersedes the earlier "f64 control flow inside `smt_update_regime`" clue:
that block may still exercise fragile lowering paths, but the stronger current
evidence is that imported SMT runtime is blocked by compiler/runtime lowering of
mutable imported SMT bodies. It is not evidence for or against the solver
algorithm. Public novelty remains blocked until these imported SMT runtime
harnesses execute green without diagnostic bypasses.

Additional env-aware reduction on 2026-06-25:

- With `/tmp/madaros-novelty-stdlibenv-test29`, the env-aware resolver was
  validated from `/workspace/sounio`: setting
  `SOUNIO_STDLIB_PATH=/tmp/sounio-smt-freshcut-empty/stdlib` caused the compiler
  to import a temporary minimal `theorem::smt` module instead of the repo
  stdlib. A four-function temp module (`main`, `smt_new`,
  `smt_conflict_clause`, `smt_assign_var`) built successfully with
  `RC=0`.
- A minimal temp module containing the real `SmtContext` shape and the
  Beta-Bernoulli indexed-field block
  (`ctx.assign[var_idx]`, `ctx.phase_beta[var_idx]`,
  `ctx.phase_alpha[var_idx]`) also built successfully. Therefore the stable
  blocker is not simply "array indexing through imported `&!SmtContext`" or the
  Beta block in isolation.
- A larger temp module containing the full real `smt_conflict_clause` body with
  helper functions stubbed also built successfully. The remaining crash requires
  more accumulated real SMT lowering context than that isolated function body.
- `/tmp/madaros-novelty-stmttrace-test30` added statement begin/done tracing
  inside `lower_block_ref`. The extra trace changed the full-SMT crash frontier
  from `smt_conflict_clause` to earlier float/control helper bodies such as
  `smt_adaptive_beta` and `smt_adaptive_beta_v2`. Treat this as Heisenbug
  evidence only. The useful signal is cumulative fragility while lowering many
  imported SMT helper bodies, not a solver algorithm failure.
- A candidate local restore change in `lower_block_ref` copied the local stack,
  changed `count`, and re-boxed it instead of assigning `lo.locals.count`
  directly. `/tmp/madaros-novelty-localrestore-test31` built, but the full SMT
  fixture regressed and exited 139 already inside `smt_abs_i64`. That candidate
  fix was rejected and reverted.
- `git diff --check` stayed clean, and the multimodule witness stayed green
  with `/tmp/madaros-novelty-stmttrace-test30`:

```bash
MADAROS_BIN=/tmp/madaros-novelty-stmttrace-test30 \
  bash scripts/ci/madaros_multimodule_witness.sh
```

Result:

```text
[madaros-mm-witness] PASS: 5/5 witnesses
```

The latest working classification is unchanged but sharper: Sounio has
interesting solver source surfaces and proof-profile discipline, while imported
SMT runtime evidence is still blocked by compiler/runtime lowering instability.
That is a substrate blocker, not solver novelty evidence.

Follow-up diagnostic after `test31` on 2026-06-25:

- `/tmp/madaros-novelty-ifunit-test32` changed no-`else` `if` lowering to emit
  unit-valued control flow instead of a synthetic else/result register. The
  isolated compiler built and the multimodule witness stayed green
  (`PASS: 5/5 witnesses`), but the full imported SMT fixture still exited 139
  in `smt_adaptive_beta`. This candidate did not clear the blocker.
- `/tmp/madaros-novelty-blockrestoreguard-test33` guarded
  `lo.locals.count = saved_local_count` behind a no-op check. The witness stayed
  green, but the full imported SMT fixture still exited 139. This candidate did
  not clear the blocker.
- `/tmp/madaros-novelty-letmeta-test34` precomputed `let` RHS metadata before
  lowering the RHS to avoid re-reading `s.expr` afterward. The witness stayed
  green, but the full imported SMT fixture still exited 139 at the same
  frontier and introduced a new local warning. This candidate was rejected and
  reverted.
- Scratch reductions with
  `SOUNIO_STDLIB_PATH=/tmp/sounio-smt-freshcut-empty/stdlib` showed that
  removing `if ctx.n_vars <= 0 { return 0.0 }` from `smt_adaptive_beta` moves
  the crash past the loop into the final `f64` return path. Removing both that
  guard and the `let beta_max: f64 = 0.8` statement lets the whole
  `smt_adaptive_beta` body lower before the crash. This points away from solver
  semantics and toward cumulative lowerer/runtime fragility after rich imported
  helper bodies.
- `/tmp/madaros-novelty-blockepi-test35` added lower-block epilogue tracing.
  The extra trace again perturbed the crash earlier, this time into
  `smt_bump_var_epistemic`, ending at the indexed `f64` field assignment:

```sio
ctx.act_var[var_idx] = ctx.act_var[var_idx] +
    ctx.bump_amount * ctx.bump_amount * var_frac * div_mult
```

  This is Heisenbug evidence, but it repeats the stronger crash class:
  imported mutable context bodies with `Index(FieldAccess(...))`, `f64`
  arithmetic, and indexed stores remain fragile after enough real SMT lowering
  context accumulates.
- `/tmp/madaros-novelty-targetsnapshot-test36` then tested that candidate:
  snapshotting `s.target` at the start of `lower_assign_stmt_ref` and using the
  snapshot after RHS lowering. The isolated compiler built and the multimodule
  witness stayed green (`PASS: 5/5 witnesses`), but the full imported SMT
  fixture still exited 139 at the same `smt_bump_var_epistemic` indexed
  `f64`-store frontier. The candidate also introduced a local warning, so it
  was rejected and reverted.

Overclaim cleanup on 2026-06-25:

- `stdlib/theorem/smt.sio` no longer labels score/polarity modes as
  "first-in-class" or "confirmed literature gap"; comments now say
  "candidate experimental mode" and require focused literature review before
  stronger novelty wording.
- `tests/stdlib/theorem/test_smt_beta_polarity_ts.sio` and
  `tests/stdlib/theorem/test_smt_epistemic_eval.sio` no longer print/comment
  public novelty claims when their internal experiments pass.
- `rg` confirms the only remaining `first-in-class` occurrence in this lane is
  this document's warning that such wording is unsafe.

LLM-offload review:

- DeepSeek review attempt failed because the provider reported insufficient
  balance.
- xAI/Grok review flagged overclaim risk in the first draft. This revision
  narrows "working substrate" to checked source-level plumbing, separates source
  surfaces from runtime performance evidence, adds URLs for external anchors,
  and adds reproduction commands.

Canonical native-v2 f64-call update on 2026-06-25:

- Rebuilt canonical `artifacts/self-hosted/madaros` after adding native-v2
  module-level `fn_id -> returns_float` metadata. The narrow bug was that calls
  to imported f64-returning functions could jump to the correct callee, while
  the caller failed to copy `xmm0` back into the IR temp unless the call
  instruction itself carried `IR_FLOAT_REG_MARKER_FLAG`. Local f64 getters were
  marked; imported `theorem::smt` getters were not.
- New canonical artifact:

```text
4a1b8425c5ae60db9cc5db169d691eb04f8f392a00e362369712bba9118afb92  artifacts/self-hosted/madaros
```

- Focused probes now distinguish the repaired surface:

```bash
./bin/souc compile /tmp/smt_local_getter_probe.sio -o /tmp/smt_local_getter_probe_after
/tmp/smt_local_getter_probe_after
./bin/souc compile /tmp/smt_regime_probe.sio -o /tmp/smt_regime_probe_after
/tmp/smt_regime_probe_after
./bin/souc compile /tmp/smt_regime_php_probe.sio -o /tmp/smt_regime_php_probe_after
/tmp/smt_regime_php_probe_after
```

Results:

```text
smt_local_getter_probe_after: 1000100010001000 exit 0
smt_regime_probe_after:      110001000 exit 0
smt_regime_php_probe_after:  0020056061223465527 exit 0
```

The first line means local and imported f64 getters now agree on the same
`SmtContext`: local trust/alpha and imported trust/alpha all print `1000`.
This supersedes the earlier apparent Beta-reset logical failure: the previous
`-9223372036854775808` values came from imported f64 call-result handling, not
from the SMT posterior reset logic.

- At this checkpoint, before the later IR-capacity repair, the focused SMT
  suite was still not green:

```bash
./scripts/run_sio_test_suite.sh test_smt --verbose --jobs 1
```

Result:

```text
PASS test_smt_adaptive_epistemic.sio
PASS test_smt_solver_basic.sio
FAIL test_smt_beta_polarity_ts.sio (run exited 139)
FAIL test_smt_epistemic_eval.sio (run exited 139)
FAIL test_smt_regime_getters.sio (run exited 139)
FAIL test_smt_thompson_stats.sio (run exited 139)
Pass: 2; Fail: 4; Total: 6
```

- Running the four failing tests individually shows material progress before
  the remaining crash:

```text
test_smt_regime_getters: T1-T5 PASS, then exit 139 before T6/summary
test_smt_beta_polarity_ts: T1-T3 PASS, then exit 139 before T4/T5/summary
test_smt_thompson_stats: T1/T2 diagnostics print with real counts, then exit 139
test_smt_epistemic_eval: PHP/K4/random/GUM checks print PASS diagnostics,
  then exit 139 before final clean process exit
```

Two `gdb` runs on the first two binaries caught `rip=0x0` after the last printed
diagnostic. At that checkpoint the blocker was still in the compiler/runtime
substrate: longer imported SMT runs hit a null-control-flow/return crash. It
was not evidence that the solver's Beta reset, regime getters, or epistemic
scoring were mathematically wrong, and it was still not public novelty evidence.

Status at the f64-call checkpoint:

- Source-level SMT/SAT ideas are interesting: GUM-style activity mean/variance,
  Wilson/LRB score mode, Thompson-style scoring, Beta-Bernoulli polarity, and
  regime-gated restart/trust signals.
- Canonical runtime evidence was improved but insufficient: imported f64 getter
  observability was repaired, but the canonical `test_smt` gate remained `2/6`.
- Public novelty remained blocked until the remaining native-v2/runtime crash
  was cleared, benchmark/ablation tables were generated, external solver
  baselines were run, and proof/certificate replay was specified beyond source
  comments. The later IR-capacity update below clears the focused runtime
  crash, but not the benchmark/proof/literature requirements.

Canonical IR-capacity promotion on 2026-06-25:

- The remaining canonical SIGSEGVs were narrowed to silent IR truncation in
  large test `main` functions. With `SOUNIO_NV2_IR_TRACE=1`, the modular path
  reported `test_smt_regime_getters.sio` `main instr_count=512`, exactly the
  old `IR_MAX_INSTRS` ceiling. A marked copy reached T6 and crashed during the
  next solver call; a reduced T1-T5+T6 probe with `main instr_count=272`
  passed. After lifting `IrFunction.instrs` and the helper/test IR arrays from
  512 to 1024, the same full test reports `main instr_count=657` and exits 0.
- Rebuilt canonical artifact:

```text
40f9abc2c697fab697140228b9c9f41a97b677348ae0759b8872a47e515d6a64  artifacts/self-hosted/madaros
```

- Current canonical default gates:

```bash
./scripts/run_sio_test_suite.sh test_smt --verbose --jobs 1
./scripts/run_sio_test_suite.sh solver_novelty_readiness --verbose --jobs 1
./bin/souc check stdlib/theorem/solver_novelty_readiness.sio
./bin/souc compile tests/run-pass/solver_novelty_readiness_tiny.sio \
  -o /tmp/solver_novelty_readiness_tiny_ir1024
/tmp/solver_novelty_readiness_tiny_ir1024
```

Results:

```text
test_smt:
  PASS test_smt_adaptive_epistemic.sio
  PASS test_smt_beta_polarity_ts.sio
  PASS test_smt_epistemic_eval.sio
  PASS test_smt_regime_getters.sio
  PASS test_smt_solver_basic.sio
  PASS test_smt_thompson_stats.sio
  Pass: 6; Fail: 0; Total: 6

solver_novelty_readiness:
  PASS solver_novelty_readiness_tiny.sio
  Pass: 1; Fail: 0; Total: 1

solver_novelty_readiness.sio: check OK
solver_novelty_readiness_tiny_ir1024: exit 0
```

- The Beta-Bernoulli polarity smoke test was corrected to avoid a statistical
  overclaim. It still checks correctness, posterior update, posterior reset,
  fully epistemic PHP(5,4) correctness, and execution of the 100-seed
  comparison. It no longer requires Beta-TS to dominate saved-phase as a unit
  invariant. Current canonical comparison result:

```text
mode=1 wins / mode=0 wins / ties: 44 / 47 / 9
```

This means canonical runtime promotion to Level 2 private evidence is now green,
but the strongest safe public wording is still limited to "candidate epistemic
variable-score and polarity-sampling DPLL(T) experiment". Public novelty remains
blocked on real ablation tables, external benchmarks, proof/certificate replay,
and focused literature review.

Tiny internal ablation scaffold added on 2026-06-25:

```bash
bash scripts/research/run_solver_ablation_tiny.sh
```

This compiles `benchmarks/solver/smt_ablation_tiny.sio`, emits a deterministic
CSV over two repetitions, five fixed mixed-formula seeds, one PHP(5,4) UNSAT
instance, and these configurations:

```text
score_mode=1, phase_mode=0, restart_budget=0  mean-only
score_mode=0, phase_mode=0, restart_budget=0  adaptive GUM-UCB
score_mode=2, phase_mode=0, restart_budget=0  Wilson/LRB
score_mode=3, phase_mode=0, restart_budget=0  Thompson score
score_mode=3, phase_mode=1, restart_budget=0  Thompson score + Beta polarity
score_mode=3, phase_mode=1, restart_budget=3  Thompson score + Beta polarity + warm restarts
score_mode=4, phase_mode=0, restart_budget=0  epistemic LRB
```

The CSV columns are:

```text
instance_id,seed,rep,score_mode,phase_mode,restart_budget,result,decisions,conflicts,restarts,stat_total_d,stat_max_d
```

The runner records whole-program wall time in a `/tmp` manifest as
`elapsed_ms_total`, writes an aggregate `summary.csv` keyed by instance, score
mode, phase mode, and restart budget, and now writes a table-ready
`config_stats.csv` with result values plus mean/min/max decisions, conflicts,
and restarts per configuration. This is intentionally only a tiny internal
ablation scaffold: it starts requirement 2 by making
fixed-seed/repetition/config evidence machine-readable, but it does not yet
satisfy Level 3 because it lacks per-row timing, larger benchmark slices,
broader external solver baselines, and independent proof/certificate checking.

Observed tiny-run receipt from the canonical wrapper:

```text
schema=sounio.solver.smt_ablation_tiny.run.v1
timestamp_utc=20260625T223930Z
souc_bin=/workspace/sounio/bin/souc
madaros_sha256=40f9abc2c697fab697140228b9c9f41a97b677348ae0759b8872a47e515d6a64
run_status=0
elapsed_ms_total=23
rows=84
```

Observed aggregate summary:

```text
instance_id,score_mode,phase_mode,restart_budget,rows,sum_decisions,sum_conflicts,sum_restarts
1,0,0,0,10,21,52,0
1,1,0,0,10,25,60,0
1,2,0,0,10,21,52,0
1,3,0,0,10,18,46,0
1,3,1,0,10,25,60,0
1,3,1,3,10,10,30,4
1,4,0,0,10,27,64,0
2,0,0,0,2,60,122,0
2,1,0,0,2,59,120,0
2,2,0,0,2,76,154,0
2,3,0,0,2,58,118,0
2,3,1,0,2,60,122,0
2,3,1,3,2,58,118,16
2,4,0,0,2,82,166,0
```

Observed table-ready configuration statistics:

```text
instance_id,score_mode,phase_mode,restart_budget,rows,result_values,decisions_mean,decisions_min,decisions_max,conflicts_mean,conflicts_min,conflicts_max,restarts_mean,restarts_min,restarts_max
1,0,0,0,10,0,2.100000,1,4,5.200000,3,9,0.000000,0,0
1,1,0,0,10,0,2.500000,1,5,6.000000,3,11,0.000000,0,0
1,2,0,0,10,0,2.100000,1,4,5.200000,3,9,0.000000,0,0
1,3,0,0,10,0,1.800000,1,4,4.600000,3,9,0.000000,0,0
1,3,1,0,10,0,2.500000,1,6,6.000000,3,13,0.000000,0,0
1,3,1,3,10,0,1.000000,1,1,3.000000,3,3,0.400000,0,1
1,4,0,0,10,0,2.700000,1,6,6.400000,3,13,0.000000,0,0
2,0,0,0,2,0,30.000000,30,30,61.000000,61,61,0.000000,0,0
2,1,0,0,2,0,29.500000,29,30,60.000000,59,61,0.000000,0,0
2,2,0,0,2,0,38.000000,38,38,77.000000,77,77,0.000000,0,0
2,3,0,0,2,0,29.000000,29,29,59.000000,59,59,0.000000,0,0
2,3,1,0,2,0,30.000000,30,30,61.000000,61,61,0.000000,0,0
2,3,1,3,2,0,29.000000,29,29,59.000000,59,59,8.000000,8,8
2,4,0,0,2,0,41.000000,41,41,83.000000,83,83,0.000000,0,0
```

Medium internal ablation scaffold added on 2026-06-25:

```bash
scripts/research/run_solver_ablation_medium.py
```

This compiles `benchmarks/solver/smt_ablation_medium.sio`, emits a deterministic
CSV over four repetitions, twelve fixed seeds for each of two random mixed
families, PHP(5,4), PHP(6,5), and the same seven heuristic configurations used
by the tiny scaffold. The Python runner streams row output with `stdbuf -oL`
when available, records inter-row wall-clock timing, and writes:

```text
smt_ablation_medium.raw.csv
smt_ablation_medium.timed.csv
config_stats.csv
instance_stats.csv
manifest.txt
```

This is the first local ablation table in this note with fixed seeds,
repetitions, per-row timing, decisions, conflicts, restarts, and aggregation in
one reproducible receipt. It is stronger than the tiny smoke, but still not a
public benchmark result: the formulas are generated local fixtures, the timing
is workspace wall-clock timing, and the slice is not SATComp/SMT-COMP-scale.

Observed medium-run receipt from the canonical wrapper:

```text
schema=sounio.solver.smt_ablation_medium.run.v1
timestamp_utc=20260625T232403Z
souc_bin=/workspace/sounio/bin/souc
madaros_sha256=40f9abc2c697fab697140228b9c9f41a97b677348ae0759b8872a47e515d6a64
compile_status=0
run_status=0
elapsed_ms_total=272
rows=728
instances=4
repetitions=4
random_seeds_per_random_family=12
configs=7
```

Observed medium instance statistics:

```text
instance_id,rows,result_values,elapsed_ms_mean,decisions_mean,conflicts_mean,restarts_mean
1,336,0|1,0.086310,2.681548,5.250000,0.050595
2,28,0,0.250000,38.321429,77.642857,1.142857
3,336,0,0.000000,1.178571,3.357143,0.023810
4,28,0,5.285714,214.607143,430.214286,1.142857
```

Example medium timed rows:

```text
instance_id,seed,rep,score_mode,phase_mode,restart_budget,result,decisions,conflicts,restarts,stat_total_d,stat_max_d,row_index,elapsed_ms_since_prev_row,elapsed_ms_cumulative
1,5000,0,1,0,0,0,1,3,0,2,1,1,29,29
1,5000,0,3,1,3,0,1,3,0,2,1,6,0,30
4,0,3,4,0,0,0,226,453,0,328,2,728,5,270
```

Tiny DIMACS external-baseline scaffold added on 2026-06-25:

```bash
scripts/research/run_solver_external_dimacs_tiny.py
```

This generates DIMACS CNFs for the same five mixed 2+5 seeds plus PHP(5,4),
detects installed external solvers, records solver availability/version, and
runs the available solvers with a 15-second per-instance timeout. This starts
requirement 3, but remains a tiny smoke baseline: it is not SATComp-scale, does
not cover OPB, and does not compare against Kissat/CaDiCaL in the current
workspace because those binaries are absent.

Observed external DIMACS receipt:

```text
schema=sounio.solver.external_dimacs_tiny.run.v1
timestamp_utc=20260625T222406Z
instances=6
available_solvers=z3
external_results=/tmp/sounio-solver-external-dimacs-tiny-20260625T222406Z/external_results.csv
```

Observed solver availability:

```text
solver,path,available,version
z3,/workspace/.home/openvscode-server/.local/bin/z3,1,Z3 version 4.16.0 - 64 bit
kissat,,0,
cadical,,0,
minisat,,0,
glucose,,0,
cryptominisat5,,0,
```

Observed `z3` results:

```text
instance_id,name,seed,n_vars,n_clauses,solver,exit_code,result,elapsed_ms
1,mixed_2_5_random,5000,20,60,z3,0,UNSAT,1
1,mixed_2_5_random,5001,20,60,z3,0,UNSAT,1
1,mixed_2_5_random,5002,20,60,z3,0,UNSAT,1
1,mixed_2_5_random,5003,20,60,z3,0,UNSAT,1
1,mixed_2_5_random,5004,20,60,z3,0,UNSAT,1
2,php5_4_unsat,0,20,45,z3,0,UNSAT,1
```

Tiny Sounio-vs-external DIMACS crosscheck added on 2026-06-25:

```bash
scripts/research/run_solver_dimacs_crosscheck_tiny.py
```

This runs the tiny Sounio ablation and the tiny external DIMACS scaffold into a
shared output directory, then joins Sounio result rows against external solver
rows by `(instance_id, seed)`. Sounio result code `0` is mapped to `UNSAT` and
`1` to `SAT`; all other codes are treated as non-matching status labels. In the
current workspace only `z3` is available, so this is a tiny crosscheck against
one real external solver, not a SATComp-scale baseline.

Observed Sounio-vs-z3 DIMACS crosscheck receipt:

```text
schema=sounio.solver.dimacs_crosscheck_tiny.run.v1
timestamp_utc=20260625T225915Z
available_solvers=z3
rows=84
mismatches=0
all_matched=1
crosscheck_results=/tmp/sounio-solver-dimacs-crosscheck-tiny-20260625T225915Z/crosscheck_results.csv
```

Observed crosscheck summary:

```text
metric,value
matches_1,84
rows,84
solver_z3,84
```

Example joined rows:

```text
instance_id,seed,rep,score_mode,phase_mode,restart_budget,sounio_result_code,sounio_result,external_solver,external_result,matches_external
1,5000,0,1,0,0,0,UNSAT,z3,UNSAT,1
1,5000,0,3,1,3,0,UNSAT,z3,UNSAT,1
2,0,1,4,0,0,0,UNSAT,z3,UNSAT,1
```

Medium Sounio-vs-external DIMACS crosscheck added on 2026-06-25:

```bash
scripts/research/run_solver_dimacs_crosscheck_medium.py
```

This runs the medium Sounio ablation, generates matching DIMACS CNFs for all
twenty-six unique medium instances, runs the available external DIMACS solvers,
and joins every Sounio row against external results by `(instance_id, seed)`.
It preserves Sounio row timing in the joined table. In the current workspace
only `z3` is available, so this is a medium one-solver crosscheck, not broad
external solver coverage.

Observed Sounio-vs-z3 DIMACS medium crosscheck receipt:

```text
schema=sounio.solver.dimacs_crosscheck_medium.run.v1
timestamp_utc=20260625T232555Z
available_solvers=z3
instances=26
rows=728
mismatches=0
all_matched=1
crosscheck_results=/tmp/sounio-solver-dimacs-crosscheck-medium-20260625T232555Z/crosscheck_results.csv
```

Observed medium crosscheck summary:

```text
metric,value
matches_1,728
rows,728
solver_z3,728
```

Observed medium solver availability:

```text
solver,path,available,version
z3,/workspace/.home/openvscode-server/.local/bin/z3,1,Z3 version 4.16.0 - 64 bit
kissat,,0,
cadical,,0,
minisat,,0,
glucose,,0,
cryptominisat5,,0,
```

Example medium joined rows:

```text
instance_id,seed,rep,score_mode,phase_mode,restart_budget,sounio_result_code,sounio_result,sounio_elapsed_ms_since_prev_row,external_solver,external_result,matches_external,external_elapsed_ms
1,5000,0,1,0,0,0,UNSAT,1,z3,UNSAT,1,1
3,7000,0,3,0,0,0,UNSAT,0,z3,UNSAT,1,1
4,0,3,4,0,0,0,UNSAT,5,z3,UNSAT,1,2
```

Tiny SMT-LIB external-baseline scaffold added on 2026-06-25:

```bash
scripts/research/run_solver_external_smtlib_tiny.py
```

This generates six deterministic QF_LIA SMT-LIB files, detects installed SMT
solvers, records solver availability/version, and runs available solvers with a
15-second per-instance timeout. It is an SMT smoke baseline only: it is not
SMT-COMP-scale, does not exercise Alethe proof output, and does not cover OPB.

Observed external SMT-LIB receipt:

```text
schema=sounio.solver.external_smtlib_tiny.run.v1
timestamp_utc=20260625T222406Z
instances=6
available_solvers=z3
external_results=/tmp/sounio-solver-external-smtlib-tiny-20260625T222406Z/external_results.csv
```

Observed SMT solver availability:

```text
solver,path,available,version
z3,/workspace/.home/openvscode-server/.local/bin/z3,1,Z3 version 4.16.0 - 64 bit
cvc5,,0,
yices-smt2,,0,
```

Observed `z3` SMT-LIB results:

```text
instance_id,name,seed,expected,solver,exit_code,result,matches_expected,elapsed_ms
1,qflia_box_sat,0,SAT,z3,0,SAT,1,15
2,qflia_box_unsat,0,UNSAT,z3,0,UNSAT,1,10
3,qflia_seeded_bound_unsat,9000,UNSAT,z3,0,UNSAT,1,10
4,qflia_seeded_bound_unsat,9001,UNSAT,z3,0,UNSAT,1,10
5,qflia_seeded_bound_unsat,9002,UNSAT,z3,0,UNSAT,1,10
6,qflia_seeded_bound_unsat,9003,UNSAT,z3,0,UNSAT,1,11
```

Tiny Sounio-vs-external SMT-LIB QF_LIA crosscheck added on 2026-06-25:

```bash
scripts/research/run_solver_smtlib_crosscheck_tiny.py
```

This compiles and runs `benchmarks/solver/smtlib_qflia_crosscheck_tiny.sio`,
which mirrors the six QF_LIA SMT-LIB smoke fixtures in Sounio form, then joins
Sounio rows against the external SMT-LIB runner by `(instance_id, seed)`.
Sounio result code `1` is mapped to `SAT`, `0` to `UNSAT`, and all other codes
to non-matching status labels. In the current workspace only `z3` is available,
so this is a tiny one-solver crosscheck, not SMT-COMP-scale evidence.

Observed Sounio-vs-z3 SMT-LIB crosscheck receipt:

```text
schema=sounio.solver.smtlib_crosscheck_tiny.run.v1
timestamp_utc=20260625T225915Z
source=benchmarks/solver/smtlib_qflia_crosscheck_tiny.sio
available_solvers=z3
rows=6
mismatches=0
all_matched=1
crosscheck_results=/tmp/sounio-solver-smtlib-crosscheck-tiny-20260625T225915Z/crosscheck_results.csv
```

Observed crosscheck summary:

```text
metric,value
matches_1,6
rows,6
solver_z3,6
```

Observed joined results:

```text
instance_id,seed,sounio_expected,sounio_result,external_solver,external_expected,external_result,matches_external
1,0,SAT,SAT,z3,SAT,SAT,1
2,0,UNSAT,UNSAT,z3,UNSAT,UNSAT,1
3,9000,UNSAT,UNSAT,z3,UNSAT,UNSAT,1
4,9001,UNSAT,UNSAT,z3,UNSAT,UNSAT,1
5,9002,UNSAT,UNSAT,z3,UNSAT,UNSAT,1
6,9003,UNSAT,UNSAT,z3,UNSAT,UNSAT,1
```

Tiny OPB external-baseline scaffold added on 2026-06-25:

```bash
scripts/research/run_solver_external_opb_tiny.py
```

This generates four deterministic OPB pseudo-Boolean fixtures and records
availability of OPB/PB solvers. In the current workspace no OPB/PB solver is
installed, so this is blocker evidence, not solver-baseline evidence.

Observed external OPB receipt:

```text
schema=sounio.solver.external_opb_tiny.run.v1
timestamp_utc=20260625T222916Z
instances=4
available_solvers=none
opb_manifest=/tmp/sounio-solver-external-opb-tiny-20260625T222916Z/opb_manifest.csv
external_results=/tmp/sounio-solver-external-opb-tiny-20260625T222916Z/external_results.csv
```

Observed OPB solver availability:

```text
solver,path,available,version
roundingsat,,0,
open-wbo,,0,
open-wbo_static,,0,
scip,,0,
sat4j-pb,,0,
minisat+,,0,
minisatp,,0,
pbsolver,,0,
```

Generated OPB fixtures:

```text
instance_id,name,expected
1,pb_atleast_one_sat,SAT
2,pb_contradictory_bound_unsat,UNSAT
3,pb_cardinality_window_sat,SAT
4,pb_cardinality_window_unsat,UNSAT
```

Certificate/proof tool availability check added on 2026-06-25:

```bash
scripts/research/check_solver_certificate_tools.py
```

This records whether the workspace has external checkers or proof tools for
SAT LRAT/FRAT, SMT Alethe, and PB VeriPB/CakePB-style paths. In the current
workspace all checked tools are absent, so Level 3 certificate claims remain
blocked even though local microkernel seeds exist.

Observed certificate-tool receipt:

```text
schema=sounio.solver.certificate_tools.run.v1
timestamp_utc=20260625T222916Z
pb_veripb_available_count=0
sat_lrat_frat_available_count=0
smt_alethe_available_count=0
availability_csv=/tmp/sounio-solver-cert-tools-20260625T222916Z/certificate_tool_availability.csv
```

Observed certificate/proof tool availability:

```text
group,tool,path,available,version
sat_lrat_frat,drat-trim,,0,
sat_lrat_frat,gratgen,,0,
sat_lrat_frat,lrat-check,,0,
sat_lrat_frat,frat-rs,,0,
smt_alethe,carcara,,0,
smt_alethe,cvc5,,0,
smt_alethe,veriT,,0,
pb_veripb,veripb,,0,
pb_veripb,cakepb,,0,
pb_veripb,roundingsat,,0,
```

Tiny SAT/RUP file-level replay scaffold added on 2026-06-25:

```bash
scripts/research/run_solver_sat_rup_replay_tiny.py
```

This generates DIMACS CNF fixtures and DRAT-style addition proof logs, then
replays a narrow RUP subset from files. The accepted fixture is an XOR-style
UNSAT CNF over two variables; the proof derives unit `1` and then the empty
clause. The rejected fixture is a SAT base clause `(1 or 2)` with a non-RUP
attempt to add unit `1`. This complements the existing source-level
`stdlib/theorem/lrat.sio` and `sat_rup_microkernel` seeds by exercising a
separate parser/proof-log replay path over generated files, but it remains a
tiny local checker: not full DRAT, not LRAT, not FRAT, not an external
independent verified checker, and not sufficient for Level 3 certificate
claims.

Observed SAT/RUP replay receipt:

```text
schema=sounio.solver.sat_rup_replay_tiny.run.v1
timestamp_utc=20260625T225052Z
format=drat_addition_rup_subset
cnf=/tmp/sounio-solver-sat-rup-replay-tiny-20260625T225052Z/fixtures/rup_xor_unsat.cnf
rejected_cnf=/tmp/sounio-solver-sat-rup-replay-tiny-20260625T225052Z/fixtures/rup_sat_base.cnf
accepted_proof=/tmp/sounio-solver-sat-rup-replay-tiny-20260625T225052Z/fixtures/rup_xor_unsat.drat
rejected_proof=/tmp/sounio-solver-sat-rup-replay-tiny-20260625T225052Z/fixtures/rup_sat_wrong.drat
accepted_proof_steps=2
rejected_wrong_proof=1
all_expected=1
```

Observed replay result table:

```text
name,expected_accept,accepted,steps,reason
rup_xor_unsat,1,1,2,accepted
rup_sat_wrong,0,0,0,non_rup_step_1
```

Related source-level SAT/LRAT gates checked in the current canonical wrapper on
2026-06-25:

```bash
./bin/souc check stdlib/theorem/lrat.sio
./bin/souc compile tests/run-pass/lrat_deletion_lifecycle_tiny.sio \
  -o /tmp/sounio-lrat-deletion-lifecycle-tiny
/tmp/sounio-lrat-deletion-lifecycle-tiny
./bin/souc compile tests/run-pass/sat_resolution_rup_bridge_tiny.sio \
  -o /tmp/sounio-sat-resolution-rup-bridge-tiny
/tmp/sounio-sat-resolution-rup-bridge-tiny
./bin/souc compile tests/run-pass/colouring_k3_lrat_tiny.sio \
  -o /tmp/sounio-colouring-k3-lrat-tiny
/tmp/sounio-colouring-k3-lrat-tiny
```

Result: the module check passed, and all three directly executed ELFs returned
exit code 0. The suite filters `lrat`, `sat_resolution`, and
`colouring_k3_lrat_tiny` also returned success, but most Madaros-marked tests
were skipped by the suite harness, so the direct `compile -o` plus ELF
execution above is the stronger current evidence for those tiny gates.

Tiny SAT/LRAT hint replay scaffold added on 2026-06-25:

```bash
scripts/research/run_solver_sat_lrat_replay_tiny.py
```

This generates the same XOR-style UNSAT DIMACS fixture, then replays a tiny
LRAT/RUP-hint subset from text proof lines shaped as:

```text
<new-id> <lemma-lits...> 0 <hint-ids...> 0
d <deleted-ids...> 0
```

Initial DIMACS clauses receive stable IDs in file order. Addition lines must
use fresh IDs, active hints, and a RUP conflict over the hinted clauses. This
moves one step closer to an LRAT-shaped certificate path than the DRAT-addition
RUP smoke above because proof IDs, hint IDs, ID reuse, and deleted antecedents
are checked. It remains a tiny local subset, not full LRAT, not FRAT, not an
external independent verified checker, and not enough for Level 3 certificate
claims.

Observed SAT/LRAT replay receipt:

```text
schema=sounio.solver.sat_lrat_replay_tiny.run.v1
timestamp_utc=20260625T225052Z
format=lrat_rup_hint_subset
cnf=/tmp/sounio-solver-sat-lrat-replay-tiny-20260625T225052Z/fixtures/lrat_xor_unsat.cnf
accepted_proof=/tmp/sounio-solver-sat-lrat-replay-tiny-20260625T225052Z/fixtures/lrat_xor_unsat.lrat
accepted_proof_additions=2
negative_controls=3
all_expected=1
```

Observed SAT/LRAT replay result table:

```text
name,expected_accept,accepted,additions,deletions,reason
lrat_bad_hint,0,0,0,0,non_lrat_rup_step_2:no_conflict_from_hints
lrat_deleted_hint,0,0,0,1,non_lrat_rup_step_2:deleted_hint_3
lrat_reused_id,0,0,0,0,reused_id_1
lrat_xor_unsat,1,1,2,0,accepted
```

Tiny SMT/Farkas file-level replay scaffold added on 2026-06-25:

```bash
scripts/research/run_solver_smt_farkas_replay_tiny.py
```

This generates integer-scaled QF_LIA inequality rows and certificate rows from
files, then replays a narrow Farkas combination rule: all multipliers must be
nonnegative, referenced rows must exist, variable coefficients must cancel to
zero, and the combined bound must be strictly negative. The accepted fixture is
`x <= 0` and `x >= 1`, replayed as `0*x <= -1`. Negative controls cover a
nonzero residual coefficient, a cancelling combination with nonnegative bound,
a negative multiplier, and a missing source row.

This complements the source-level `smt_farkas_microkernel` seed and the
SMT-LIB QF_LIA crosscheck, but it remains a tiny local CSV subset: not Alethe,
not LFSC, not Carcara, not a general SMT proof checker, and not enough for
Level 3 certificate claims.

Observed SMT/Farkas replay receipt:

```text
schema=sounio.solver.smt_farkas_replay_tiny.run.v1
timestamp_utc=20260625T230443Z
format=integer_scaled_farkas_csv_subset
accepted_instance=/tmp/sounio-solver-smt-farkas-replay-tiny-20260625T230443Z/fixtures/qflia_x_le_0_and_x_ge_1.csv
accepted_certificate=/tmp/sounio-solver-smt-farkas-replay-tiny-20260625T230443Z/fixtures/farkas_x_le_0_x_ge_1.csv
accepted_combined_coeff_x=0
accepted_combined_bound=-1
negative_controls=4
all_expected=1
```

Observed SMT/Farkas replay result table:

```text
name,instance_name,expected_accept,accepted,combined_coeff_x,combined_bound,reason
farkas_x_le_0_x_ge_1,qflia_x_le_0_and_x_ge_1,1,1,0,-1,accepted
farkas_nonzero_coeff,qflia_x_le_0_and_x_ge_1,0,0,1,0,nonzero_combined_coeff
farkas_nonnegative_bound,qflia_x_le_0_and_x_ge_0,0,0,0,0,nonnegative_combined_bound
farkas_negative_multiplier,qflia_x_le_0_and_x_ge_1,0,0,1,0,negative_multiplier_r2
farkas_missing_row,qflia_x_le_0_and_x_ge_1,0,0,1,0,missing_row_r3
```

## Correction 2026-06-26: the `ir_patch_validated_calls` skip is load-bearing

Earlier entries in this note (and the compiler-debt list) treated the imported
`ir_patch_validated_calls` skip as removable debt — "the bypass must be replaced
with a real fix" — implying a trivial sibling-mirror restore. That framing is
**wrong** and is hereby superseded.

Controlled rebuild on 2026-06-26: adding
`patch_stats = ir_patch_validated_calls(&! lo2.module)` to both imported-emit
entries (`lower_program_items_bodies_from_summary_with_epistemic_boxed_ref` and
`lower_program_bodies_from_summary_flat_with_epistemic_ref`) regressed **all six
`test_smt` harnesses and the `solver_novelty_readiness` gate from green to
SIGSEGV (139)**. Reverting reproduced the bit-identical green artifact
`40f9abc2c697…`; the broken build was `7aaa058c0568…`.

The skip is a *logical* no-op for `theorem::smt` (no `IR_STRATEGY_INSTRUMENTED`
callees, so the predicate never fires), yet the *act* of calling the post-pass —
two `[i64; 2048]` arrays (32 KB) passed by value per function, deep in the
recursive multimodule lowering stack under the launcher vmem ulimit — corrupts
the imported path where the single-file sites have headroom. The imported
INSTRUMENTED case is also currently unreachable: cross-module `Contest`/`Robust`
fails type-checking first (`E137`, see `tests/multimodule/thin_contest_main.sio`).

Full evidence and a proposed safe fix (heap/`&`-passed strategy tables; gate on
an actual INSTRUMENTED pre-scan) are recorded as a forensic dispatch:
`docs/audit/MADAROS_VALIDATED_CALL_IMPORTED_BYPASS_2026-06-26.md`. Until that
lands, the skip stays in place; it is correct-by-necessity on the current
imported lowering path.
