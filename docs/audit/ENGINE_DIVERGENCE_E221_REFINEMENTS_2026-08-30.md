<!-- docs:meta
topic_id: repo.docs.audit.engine-divergence-e221-refinements-2026-08-30
authority: repo_only
audience: users
last_validated: 2026-08-30
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.engine-divergence-e221-refinements-2026-08-30
-->

# Two measured divergences between Madaros and lean_single, in opposite directions

**Date:** 2026-08-30
**Engines:** Madaros built from `origin/main` source on the day of measurement
(`scripts/ci/build_modular_madaros.sh`, 101 746 924-byte ELF) and the committed
`lean_single` bootstrap. **Not** the prebuilt `artifacts/self-hosted/madaros`,
which was 25 days and 201 `self-hosted/` commits stale — see "What a stale
binary cost" below.

## 1. `E221` — Madaros is too permissive

`tests/compile-fail/diagnostic_codes_no_main.sio` is a program with no `main`.

| engine | result |
|---|---|
| `lean_single` | `rc=1`, `error[E221]: no main` |
| **Madaros** | **`rc=0`, writes an 8648-byte ELF, no mention of E221** |

`E221` is emitted only from `self-hosted/compiler/lean_single.sio`. The test was
dormant until #2287 because it stated its intent in prose (`// compile-fail:
E221`) rather than `//@`, so nothing measured this.

This is the direction that matters: the default engine accepts a program the
bootstrap refuses.

## 2. Refinement types — Madaros is too restrictive, and fails closed

`type Pos = { v: i64 | v > 0 }`

| program | `lean_single` | Madaros |
|---|---|---|
| `use_pos(-5)` — violates the predicate | `rc=1`, `error[E208]: refinement type violation — value -5 does not satisfy parameter predicate` | `rc=1`, `error[E008]` return-type mismatch + `error[E009]` argument-type mismatch |
| **`use_pos(5)` — satisfies it** | **`rc=0`** | **`rc=1`, `error[E008]`** |

The second row is the control that settles the interpretation. Madaros rejects a
program that is *correct* under the refinement, so it is not checking predicates
at all — it treats `Pos` as an opaque nominal type distinct from `i64`. The
feature is unimplemented and errs on the safe side.

Both engines refuse the violating program, so there is no soundness hole here.
But the `compile-fail` fixtures in this family pass under Madaros for a reason
unrelated to what they assert.

## Scope: 13 files use `{ x: T | pred }`

Ten under `tests/compile-fail/`, three under `tests/run-pass/`. The three
run-pass ones are **lean_single-only**, and nothing declares it:

    tests/run-pass/ontology_axiom_conjunction.sio      lean_single rc=0, Madaros rc=1
    tests/run-pass/refinement_float_bound.sio          lean_single rc=0, Madaros rc=1
    tests/run-pass/refinement_satisfied.sio            lean_single rc=0, Madaros rc=1

They are green in CI because the suite calibrates against `souc-stage2`
(`lean_single`). If the suite engine ever moves to Madaros, these three go red on
the first run.

## Not established here

Whether the ten `compile-fail` fixtures would fail their own `error-pattern`
under Madaros. A first pass suggested nine of ten did not match, but that
measurement was wrong: the check was case-sensitive while the patterns are
lowercase (`type mismatch`) and the diagnostics are capitalised (`Type
mismatch`). Anyone re-deriving this should match the way
`scripts/dev/run_sio_test_suite_v2.sh` matches, not with `grep -F`.

## What a stale binary cost

The same probe run against the prebuilt `artifacts/self-hosted/madaros` (2026-08-05)
reported six effect kinds — `Approx`, `Deterministic`, `NarrowWidthApproximation`,
`NaturalityG2`, `NonUnitary`, `Perturbative` — whose effect-drop fixtures were
accepted rather than refused, and put `scripts/ci/effect_archaeology_gate.sh` at
seven failures. Rebuilt from current source, every one of those refuses with
`error[E035]`, and the gate is at **one** failure: `Chaotic chaotic_pass.sio`,
exactly the blocker `.github/workflows/ci.yml` already names as its reason for
leaving that gate unwired. That comment is accurate; the seven-failure reading
was an artifact.

Both findings above were re-verified against the fresh build for that reason.
The `E221` divergence survived; the effect-drop one did not exist.
