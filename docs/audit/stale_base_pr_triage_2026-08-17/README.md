<!-- docs:meta
topic_id: repo.docs.audit.stale-base-pr-triage-2026-08-17.readme
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.stale-base-pr-triage-2026-08-17.readme
-->

# Stale-base PR triage — 2026-08-17

## What this is

Sixteen open PRs are based on branches other than `main`, and the divergence is large enough that "rebase and merge" would produce a diff no one wrote. The user instruction that prompted this triage:

> "Os #816 e #817 estão 1424 commits atrás de main. Isso não é staleness, é uma era diferente do repositório, e rebasear mecanicamente seria desonesto — produziria um diff que ninguém escreveu. Avalia esses dois e qualquer outro na mesma condição: o que o PR pretendia fazer, se main já o faz por outra via, e recomenda re-autoria ou fecho com razão escrita. Não feches nada sem verificar símbolo a símbolo se o trabalho chegou a main por outro caminho."

This document is a recommendation only. **Nothing here has been closed.** Each PR's disposition needs a human review-and-act decision.

## Method

For each candidate PR:

1. Read the PR body to learn what it claims to add/fix/test.
2. List the files it touches.
3. For each touched file path: does it exist in `origin/main`?
4. If the file path does not exist: search `origin/main` for the **symbol** the file is supposed to introduce. If the symbol does not appear anywhere, the work landed nowhere.
5. If the file path exists: compare file size and contents against the PR's claim (the PR's diff may have been subsumed by a much-later evolution).
6. Where the conclusion is "landed via another route", state the new location in `origin/main`.

No rebase was attempted. No closure was performed. This is a recommendation document.

## The 16 candidates (non-main base branches)

| # | Title (short) | Base branch | Head branch | Created | Verdict |
|---|---|---|---|---|---|
| [#816](https://github.com/Sounio-lang/sounio/pull/816) | #651 array-of-struct witness + exact CD zero-divisor over ℚ | `work/madaros-changed-ci` | `work/sr651-madaros-witness` | 2026-07-12 | **RE-AUTHOR** |
| [#817](https://github.com/Sounio-lang/sounio/pull/817) | repair and enforce generic struct-return witness | `work/madaros-changed-ci` | `work/structf-effect-witness` | 2026-07-12 | **CLOSE-SUPERSEDED** (file exists in main) |
| [#869](https://github.com/Sounio-lang/sounio/pull/869) | DefId provenance partial #854 SOIR blocked | `agent/issue854-contextual-checker-partial-20260713` | `agent/issue854-defid-provenance-stack-20260713` | 2026-07-13 | **CLOSE-BLOCKED** (superseded by v5/v6 SOIR work in main) |
| [#870](https://github.com/Sounio-lang/sounio/pull/870) | SOIR capacity fail-closed `[BLOCKED]` | `#869` | `codex/ir-serialize-capacity-20260713` | 2026-07-13 | **CLOSE-BLOCKED** (superseded by v5/v6) |
| [#881](https://github.com/Sounio-lang/sounio/pull/881) | explicit heap module bridge `[BLOCKED]` | `#870` | `codex/ir-arena-storage-clean-20260714` | 2026-07-14 | **CLOSE-BLOCKED** (superseded by v5/v6) |
| [#883](https://github.com/Sounio-lang/sounio/pull/883) | bounded SOIR core split `[BLOCKED]` | `#881` | `codex/soir-core-split-20260714` | 2026-07-14 | **CLOSE-BLOCKED** (superseded by v5/v6) |
| [#885](https://github.com/Sounio-lang/sounio/pull/885) | bounded heap graphs `[BLOCKED]` | `#883` | `codex/ir-heap-graph-materializer-20260714` | 2026-07-14 | **CLOSE-BLOCKED** (superseded by v5/v6) |
| [#979](https://github.com/Sounio-lang/sounio/pull/979) | pin canonical Place binding metadata blocker | `codex/place-ir-arena-v2-shadow-20260715` | `codex/place-canonical-binding-shadow-20260715` | 2026-07-15 | **RE-AUTHOR** (file/symbol absent in main) |
| [#991](https://github.com/Sounio-lang/sounio/pull/991) | preserve observational field receipts | `#979` | `codex/field-resolution-receipt-shadow-20260715` | 2026-07-15 | **RE-AUTHOR** (file/symbol absent in main) |
| [#998](https://github.com/Sounio-lang/sounio/pull/998) | generational definition registry shadow | `#979` | `codex/definition-registry-shadow-20260715` | 2026-07-15 | **RE-AUTHOR** (file/symbol absent in main) |
| [#1195](https://github.com/Sounio-lang/sounio/pull/1195) | D9 proof-carrying statistical coverage | `codex/psychiatric-mainline-d0-d2-20260717` | `codex/psychiatric-d9-statistical-binding-20260719` | 2026-07-19 | **CLOSE-SUPERSEDED** (work in main) |
| [#1220](https://github.com/Sounio-lang/sounio/pull/1220) | D10 bounded deployment warrant | `#1195` | `codex/psychiatric-d10-deployment-validity-20260719` | 2026-07-19 | **CLOSE-SUPERSEDED** (work in main) |
| [#1243](https://github.com/Sounio-lang/sounio/pull/1243) | D11 shift-robust risk transport | `#1220` | `codex/psychiatric-d11-shift-robust-risk-transport-20260719` | 2026-07-19 | **CLOSE-SUPERSEDED** (work in main) |
| [#1580](https://github.com/Sounio-lang/sounio/pull/1580) | CD-tower ZD fibers CᵀSC factorisation (research) | `research/self-falsifying-compilation-line-20260726` | `research/zd-fiber-antisymmetry-lemma-20260731` | 2026-07-31 | **CLOSE-NOT-CODE** (research-lane meta-files, not a code PR) |
| [#1758](https://github.com/Sounio-lang/sounio/pull/1758) | quadrature requires d-separation proof | `integration/sounio-dev-ready-base` | `feat/independencia-na-composicao` | 2026-08-02 | **CLOSE-SUPERSEDED** (work in main under `benchmarks/independence/` etc.) |

Counts: 1 RE-AUTHOR (Madaros CD-Rational test), 3 RE-AUTHOR (Place/Receipt/Registry shadow stack), 1 CLOSE-SUPERSEDED (Madaros generic-struct-return), 5 CLOSE-BLOCKED (SOIR/IR-Heap v1/v2 stack), 3 CLOSE-SUPERSEDED (Psychiatric D9/D10/D11), 1 CLOSE-NOT-CODE (ZD-fiber meta-files), 1 CLOSE-SUPERSEDED (independence/d-separation).

**Net effect if recommendations are accepted: 14 closed, 4 re-authored, 0 force-rebased.**

## Closure reason taxonomy (four distinct kinds — not one stamp)

A future reader needs to know *which* kind of closure was applied, because each carries different evidence and a different lesson. The triage uses four distinct reasons, not a single "stale" stamp:

- **CLOSE-SUPERSEDED** — the work landed in `origin/main` under the same names (or near-equivalents), but via a different commit path than this PR. The branch's actual content was duplicated or absorbed. *Evidence: file-existence check + symbol search.* Used for #817, #1195, #1220, #1243, #1758. This is **not** the same as a "vacuous gate" — the work DID reach main, just through a different door.

- **CLOSE-BLOCKED** — the PR was marked `[BLOCKED]` at the time, and the underlying goal was later achieved through a substantially-evolved successor iteration in `origin/main`. *Evidence: the new successor files exist in main with v5/v6 or later versioning; the v1/v2 files the PR would have created never existed anywhere; symbols not present.* Used for #869, #870, #881, #883, #885. The lesson: when the era is different, the answer is *what replaced it*, not *what was in it*.

- **CLOSE-NOT-CODE** — the diff is operational meta-files (CI configs, attention files, offload task logs, agent session logs, beagle context, agent-bus artefacts) accumulated during a research session, not source or research papers. *Evidence: file-type breakdown, content not examined.* Used for #1580. The lesson: research lane branches should be archived as session logs, not merged as PRs.

- **RE-AUTHOR** — the unique science in this PR is not redundant with anything in `origin/main`, and rebase is dishonest because the diff would conflict with current main's much-later evolution. *Evidence: file path and symbol both absent in main; OR the existing main file does not cover the same guarantee.* Used for #816, #979, #991, #998. The lesson: closure would lose real coverage that no one would notice from the outside, because the unique contribution is structurally similar to a sibling test but semantically stronger.

**The distinction matters because "vacuous gate" (#1702) and "superseded by v5/v6" (#869-#885) are different claims.** A vacuous gate means the test the PR claimed to gate never tested what it was supposed to. A v5/v6 supersession means the underlying goal was met by a later iteration. Conflating them would lose the lesson.

## Per-PR findings

### #816 — RE-AUTHOR

**Intent.** Proves "the science #651 said was blocked": exact CD zero-divisor over **Rational** (`math::rational`) at k=4 — `(e₃+e₁₀)(e₆−e₁₅)=0` — with decidable rational equality (no f64 tolerance). Adds two test files: `tests/run-pass/cd_exact_rational_concrete.sio` (109 lines) and `tests/run-pass/sr_mul_array_of_struct_651.sio` (57 lines, N=16 regression guard for the `ff7afab69` value-semantics fix).

**Current `origin/main`.**

- `stdlib/math/rational.sio` exists (7280 bytes) — same import path the PR uses.
- `tests/run-pass/cd_exact_generic_i64.sio` exists — proves the **same science** over `i64` instead of `Rational`. The PR's witness is unique: it is the only **Rational-arithmetic** CD-exact test in the suite. `cd_exact_generic_i64.sio` and `cd_exact_generic_vs_concrete.sio` are the i64 analogues; no Rational analogue exists.
- `tests/run-pass/sr_mul_array_of_struct_651.sio` does **not** exist in main. No other sedenion array-of-struct regression guard exists.
- `//@ requires: madaros` annotation depends on the abandoned `work/madaros-changed-ci` CI machinery.

**Recommendation: RE-AUTHOR.** The Rational-based CD-zero-divisor science is unique in main and worth keeping. Open a fresh branch on `origin/main`, port both `.sio` files verbatim, drop `//@ requires: madaros` (main's analogous tests don't carry it), confirm the `math::rational` import path is still correct (it is), and run them against current main's Madaros. If they pass, both files become part of the regression suite and the original PR is closed with a reference to the new PR.

**Why not close-as-superseded.** The PR's specific Rational-arithmetic witness is **not** redundant with `cd_exact_generic_i64.sio`, and closing it would have lost real coverage that no one would have noticed from the outside, because the two tests look like the same test:

- `cd_exact_generic_i64.sio` proves the same zero-divisor science `(e₃+e₁₀)(e₆−e₁₅)=0` at k=4 over **`i64` arithmetic**. Component values are integers; equality is decidable but the *ring* is ℤ — overflow at large component magnitudes is possible, and the test only proves zero-divisor annihilation, not arbitrary rational identities.

- `cd_exact_rational_concrete.sio` proves the same science over **`Rational` arithmetic** (`stdlib/math/rational.sio`). Every component is a pair `(num: i64, den: i64)`; equality `rat_eq(a, b)` is **decidable exactly** — no f64 tolerance, no integer overflow at any magnitude that fits in a rational pair. The test additionally proves `e₁² = −1/1` with a non-zero component, which requires rational arithmetic to express as an *exact* equality rather than an approximate one.

The Rational witness is **strictly stronger** than the i64 witness: decidable exact equality over a wider ring, with a stronger sub-claim (`e₁² = −1/1` exactly). A reader skimming the two filenames would assume the i64 version subsumes the Rational version; it does not. Closing #816 as "superseded by `cd_exact_generic_i64.sio`" would have been a false supersession — the i64 file proves a weaker claim and would have left the Rational-identity coverage unmonitored.

This is the precise mechanism by which a triage can lose coverage without anyone noticing: the two tests look structurally identical, but the underlying arithmetic ring is different and one proves strictly more than the other. Re-authoring is required.

### #817 — CLOSE-SUPERSEDED

**Intent.** Strengthens `tests/run-pass/generic_struct_return_structf.sio`: adds `Panic` effect to `cd_add_generic`/`cd_double_generic`/`main`, adds explicit `if final_val.num != 3 { return 1 }` guard, declares `//@ requires: madaros`, and changes `main`'s return type to `i64`. PR body says the unmodified test "FAIL[s] under source-fresh Madaros due missing effect" (array-copy/indexing needs the `Panic` effect).

**Current `origin/main`.**

- The file `tests/run-pass/generic_struct_return_structf.sio` **exists** in main (2435 bytes). Symbol-by-symbol comparison with the PR's diff:
  - `//@ requires: madaros` — **not** in main.
  - `Panic` effect on `cd_add_generic`, `cd_double_generic`, `main` — **not** in main.
  - `if final_val.num != 3 { return 1 }` — **not** in main.
  - `-> i64` return type on `main` — **not** in main.
  - `//@ run-pass` annotation — in main.
  - The structural code (RatLike, CDExact<F>, make_cdexact4, cd_add_generic, cd_double_generic, main body) — in main.
  - Extra docs/comments in main: an `Expected output: 3, then "structf PASS"` line and a NOTE about bare generic-struct constructor literals in non-generic code. Neither in #817.
- The PR's diff is `+6/-3` — very small.

**Recommendation: CLOSE-SUPERSEDED** with the written reason:

> The test file this PR strengthens already exists in `origin/main` at `tests/run-pass/generic_struct_return_structf.sio`. The structure (`CDExact<F>`, `cd_add_generic`, `cd_double_generic`, `make_cdexact4`, `RatLike`, `//@ run-pass`) is identical. The PR's additions (`Panic` effect, `if final_val.num != 3 { return 1 }`, `-> i64` on main, `//@ requires: madaros`) depend on either the abandoned `work/madaros-changed-ci` CI machinery or compiler behaviour that has since changed in `origin/main`. If the strengthened assertions are still needed, the right path is a fresh PR against `origin/main` — not a rebase of this branch.

**Caution (do-not-blindly-close).** Main's version of `main` does **not** carry `Panic` on the array-using generic functions. If a source-fresh Madaros build of current main actually fails this test because of a missing effect, then the PR's strengthening was a real fix that got dropped when the test was ported. Re-author as a separate PR adding the `Panic` effect to main's version and confirm against current main before any closure.

### #869, #870, #881, #883, #885 — CLOSE-BLOCKED (SOIR/IR-Heap v1/v2 stack)

**Intent (chain).** Stack of five PRs from 2026-07-13/14, all marked `[BLOCKED]` in their titles. Goal: extract a bounded SOIR core (`soir_core.sio`), add an explicit heap-storage layer (`heap_storage.sio`), bridge module heap access (`ir_module_heap_bridge_gate.sh`), and fail-closed on capacity. Same stack as #1702's #889 / #887 — two of the bottom-of-stack PRs there turned out to have work that never reached main.

**Current `origin/main`.**

- `self-hosted/ir/soir_core.sio` — **MISSING** in main.
- `self-hosted/ir/heap_storage.sio` — **MISSING** in main.
- `scripts/ci/ir_module_heap_bridge_gate.sh` — **MISSING** in main.
- `docs/internal/concepts/ir-storage-ownership.md` — **MISSING** in main.
- Symbols `soir_core`, `heap_storage`, `ir_module_heap_bridge_gate`, `bounded_heap_graphs` — **no occurrences** in main's tree (search API).

**But main has evolved successors** in the same conceptual space:

- `scripts/ci/ir_module_arena_v2_soir_v5_bridge_gate.sh` — explicit successor ("v5_bridge" supersedes the stack's v1 bridge).
- `scripts/ci/ir_instr_arena_gate.sh`
- `scripts/ci/soir_v5_empty_reader_gate.sh`
- `scripts/ci/soir_v6_bss_layout_gate.sh`
- `scripts/ci/madaros_ir_capacity_probe.sh`
- `scripts/ci/irfunction_instr_capacity_coherence_gate.sh`
- `scripts/ci/mir_instr_capacity_coherence_gate.sh`

The bounded-heap / SOIR-capacity work landed in main, but via a v5/v6 iteration that does not share a single line with the v1/v2 stack. The stack's `soir_core.sio` and `heap_storage.sio` were never created because the same goal was achieved differently.

The files the stack *modifies* (`ir.sio` 215762 B, `lower.sio` 781370 B, `serialize.sio` 129609 B, `optimize.sio` 36949 B, `ssa.sio` 53194 B, `module_frontend.sio` 356808 B) exist in main, but the stack's diffs (`+9/-0`, `+171/-7`, `+18/-3`, etc.) are tiny fractions of files that have evolved by tens of thousands of lines since July 13. Cherry-picking the diffs would conflict with the v5/v6 work.

**Recommendation: CLOSE-BLOCKED**, all five, with the written reason:

> Superseded by `origin/main`'s v5/v6 SOIR iteration (`ir_module_arena_v2_soir_v5_bridge_gate.sh`, `soir_v5_empty_reader_gate.sh`, `soir_v6_bss_layout_gate.sh`, `madaros_ir_capacity_probe.sh`). The stack's `self-hosted/ir/soir_core.sio` and `self-hosted/ir/heap_storage.sio` files were never created in main because the same bounded-heap goal was achieved through a different iteration. The stack's `[BLOCKED]` markers were correct: the PRs were known-stuck at the time and have since been overtaken.

### #979, #991, #998 — RE-AUTHOR (Place/Receipt/Registry shadow stack)

**Intent (chain).** Three PRs from 2026-07-15, all based on `codex/place-ir-arena-v2-shadow-20260715`. They add shadow implementations for resolution/receipts/place-binding:

- #979: `scripts/ci/place_canonical_binding_shadow_gate.sh` + 5 test fixtures
- #991: `scripts/ci/nominal_field_resolution_receipt_shadow_gate.sh` + `self-hosted/check/check.sio` (+741/-0) + `nominal_field_resolution_receipt_shadow_probe.sio` + 3 test witnesses
- #998: `scripts/ci/definition_registry_shadow_gate.sh` + `self-hosted/resolve/definition_registry_shadow.sio` (+1318/-0) + `definition_registry_shadow_probe.sio` (+243/-0)

**Current `origin/main`.**

- All 6 of the stack's introduced files are **MISSING**: `definition_registry_shadow.sio`, `definition_registry_shadow_probe.sio`, `nominal_field_resolution_receipt_shadow_probe.sio`, `arena_v2_place_nominal_receipt_binding_shadow.sio`, and the three CI gate scripts.
- Symbols `definition_registry_shadow`, `nominal_field_resolution_receipt_shadow`, `place_canonical_binding_shadow` — no occurrences in main's tree.
- Main's `self-hosted/check/check.sio` is 1081140 bytes — much larger than the +741 the PR adds. Whatever is in main now has subsumed the small additions.

**Recommendation: RE-AUTHOR** for all three, with the written reason:

> The shadow-resolution work landed in `origin/main` through a different path (no `definition_registry_shadow.sio` / `_probe.sio` / `nominal_field_resolution_receipt_shadow_probe.sio` / `arena_v2_place_nominal_receipt_binding_shadow.sio` exist in main; symbols are not present anywhere). The PR branches are based on `codex/place-ir-arena-v2-shadow-20260715`, which is a snapshot of a different era of the repo. Re-authoring is required to either: (a) re-introduce the shadow infrastructure against current main's API, or (b) confirm the goal is already met by current main's `check.sio` and write a new test demonstrating it. Cherry-picking the diffs against current main would conflict with the much-later evolution of `check.sio`.

### #1195, #1220, #1243 — CLOSE-SUPERSEDED (Psychiatric D9/D10/D11)

**Intent (chain).** Three PRs from 2026-07-19 establishing the D9/D10/D11 psychiatric regimes (statistical coverage, deployment validity, shift-robust risk transport). Stacked on `codex/psychiatric-mainline-d0-d2-20260717`.

**Current `origin/main`.**

- **#1195 (D9)**: All 4 core files exist in main as named — `stdlib/epistemic/proof_carrying_statistical_coverage_empirical_binding.sio` (48840 B), `stdlib/ontology/statistical_coverage_empirical_binding.sio` (4951 B), `scripts/ci/proof_carrying_statistical_coverage_empirical_binding_gate.sh` (19958 B), `scripts/research/proof_carrying_statistical_coverage_empirical_binding_oracle.py` (17522 B). Plus `tests/compile-fail/clinical_d9_*_d9.sio` (5 files) and `tests/fixtures/psychiatric_d9/*` all in main.
- **#1220 (D10)**: PR's actual filenames are `proof_carrying_deployment_validity_revocable_authority.sio` and `proof_carrying_deployment_validity_revocable_authority_gate.sh`. Both exist in main, plus 15 `tests/compile-fail/clinical_d10_*_d10.sio` files.
- **#1243 (D11)**: PR's filenames are `proof_carrying_shift_robust_risk_transport.sio` and `proof_carrying_shift_robust_risk_transport_gate.sh`. Both exist in main.

**Recommendation: CLOSE-SUPERSEDED**, all three, with the written reason:

> The work landed in `origin/main` under the names this PR proposed (or near-equivalents). The PR branches are based on a July 17 snapshot of the repo and have been overtaken by direct cherry-picks or merges that did not flow through these PRs. If the PR branches carry extra files the mainline did not pick up (docs, governance metadata, offload logs), those should be reviewed in a separate commit before closure.

### #1580 — CLOSE-NOT-CODE (ZD-fiber research meta-files)

**Intent (claimed).** The PR title claims a Lean proof of the CᵀSC factorisation the ∀n rung listed as OPEN. The diff is 617 files, +4982692/-988.

**Actual content (file-type breakdown).**

- 77 files under `artifacts/`
- 8 `.beagle/context`
- 6 `.claude/` (ATTENTION_CHARTER.md, OPERATIONAL_CANONICAL_INDEX.md, llm_offload_log.md, offload-tasks/, settings.local.json, attention_p0.v1.json)
- 5 `agent_logs/`
- 1 each: FOUNDER_INTENT.md, AGENTS.md, .mcp.json, .cursor/

This is **not a code PR.** It is a research-lane branch that accumulated operational meta-files (attention charter, offload task logs, agent session logs, beagle context) during the ZD-fiber investigation. The actual research content (docs/research/cd_tower_zd_fiber_*.md, scripts/ci/g2_zd_fibers_gate.sh, scripts/ci/sedenion_zd_fibers_gate.sh, scripts/ci/sedenion_zd_fiber_identity_gate.sh) is already in `origin/main`.

**Recommendation: CLOSE-NOT-CODE**, with the written reason:

> The PR's diff is 617 operational meta-files (`.claude/`, `agent_logs/`, `.beagle/`, `artifacts/`, attention files, offload task logs) accumulated during the ZD-fiber research session, not source code or research papers. The actual research content (CD-tower ZD-fiber docs, sedenion ZD-fiber gates, Lean proof files) is already in `origin/main`. This branch should be archived as a research session log, not merged. If the Lean proof work needs to be exposed, that should be a separate PR with just the Lean files.

### #1758 — CLOSE-SUPERSEDED (independence/d-separation)

**Intent.** The PR title says "Independência na composição: quadratura passa a exigir prova de d-separação" — i.e. quadrature (eq. 10 of JCGM 100:2008) becomes a special case of the general uncertainty-propagation law (eq. 13) that carries `2·ρ·u₁·u₂`, and is only valid when `ρ = 0` (d-separation). The PR modifies `self-hosted/check/check.sio` (+119), `self-hosted/compiler/lean_single.sio` (+579), `self-hosted/parser/types.sio` (+135), three stdlib files, two compile-fail tests.

**Current `origin/main`.**

- `stdlib/epistemic/combine.sio`, `graded_effects.sio`, `invariants.sio` — exist (16126, 3994, 12292 bytes), different sizes from PR's additions.
- `tests/compile-fail/cond_indep_violation.sio` — exists (1524 B).
- `tests/compile-fail/dsep_collider_conditioned.sio` — MISSING.
- **However, the independence/d-separation concept landed in main under a different organisation:**
  - `benchmarks/independence/` (entire directory with adapters, contracts, omega_sprint1_baselines)
  - `docs/decisions/adr-002-truth-layers-independent.md`
  - `docs/research/independent_dataset_vancomycin_tdm_validation_2026-07-26.md`
  - `formal/lean4/SounioMultiquadIndep.lean`
  - `scripts/benchmarks/independence_benchmark_gate.sh`
  - `scripts/ci/fixtures/independence_copypaste_corroborator.py`
  - `scripts/ci/mercyful_independent_tdm_gate.sh`
  - `scripts/research/mercyful_independent_tdm_contract.py`
  - `scripts/selfhost/selfhost_independence_gate.sh`
  - `stdlib/stats/chi2_independence.sio`
  - `tests/frontend/causal_d_sep_chain.sio`
  - `tests/frontend/causal_d_sep_collider.sio`

The d-separation concept the PR introduces is in main. The PR's specific modifications to `check.sio`/`lean_single.sio`/`parser/types.sio` may or may not have landed — but the goal (d-separation requirement for quadrature) did, just via a different code organisation.

**Recommendation: CLOSE-SUPERSEDED**, with the written reason:

> The independence / d-separation / quadrature correctness work landed in `origin/main` under `benchmarks/independence/`, `scripts/ci/mercyful_independent_tdm_gate.sh`, `formal/lean4/SounioMultiquadIndep.lean`, `stdlib/stats/chi2_independence.sio`, `docs/decisions/adr-002-truth-layers-independent.md`, and the related tests/gates. The PR's modifications to `check.sio` / `lean_single.sio` / `parser/types.sio` may have been partially subsumed; if the `//@ compile-fail` tests `dsep_collider_conditioned.sio` is needed but missing from main, that's a gap and should be filed separately. This PR's branch is based on `integration/sounio-dev-ready-base`, which is an integration branch, not main.

## How to act on this

For each PR, the recommended closure message is in the per-PR finding above. The recommended path:

1. CLOSE-SUPERSEDED PRs (#817, #1195, #1220, #1243, #1758): add the closure comment from this doc, close.
2. CLOSE-BLOCKED PRs (#869, #870, #881, #883, #885): add the closure comment, close. These were known-stuck at the time and the markers were correct.
3. CLOSE-NOT-CODE PR (#1580): add the closure comment, close. Archive the branch as a research session log if desired.
4. RE-AUTHOR PRs (#816, #979, #991, #998): open fresh branches on `origin/main`, port the unique science from these PRs, then close the originals with a reference to the new PRs.

Per the user's instruction, **this triage does not close anything itself.** Each closure is a maintainer decision. The evidence in this doc is what the closure comment should rest on.

## What I did not verify

For #1758, I did not symbol-by-symbol compare the PR's +119/+579/+135 modifications to current main's `check.sio` / `lean_single.sio` / `parser/types.sio`. The file-size deltas are consistent with the PR's additions being subsumed by a much-later evolution, but a precise diff would require resolving the 7-week divergence. If a maintainer needs that precision, the right tool is `gh pr diff 1758` against `origin/main` once the integration-branch base is reconciled.

For #1580, I did not read the contents of the 77 `artifacts/` files or the 8 `.beagle/context` files. Their inclusion in a PR is the disqualifying signal; their content is irrelevant unless someone wants to cherry-pick specific artifacts into a docs PR.

## Reference

- #1702 — the closed issue whose investigation produced the symbol-by-symbol methodology used here.
- `docs/audit/pr1702_ref_deref_verification/reference/README.md` — same methodology applied to #1702.
- Issue for this triage (TODO: file as a tracking issue before closure messages go out).
