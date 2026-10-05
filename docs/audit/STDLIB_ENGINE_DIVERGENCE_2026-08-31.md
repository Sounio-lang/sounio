<!-- docs:meta
topic_id: repo.docs.audit.stdlib-engine-divergence-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-08-31
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.stdlib-engine-divergence-2026-08-31
-->

# The two engines disagree about effects in BOTH directions, across 264 stdlib modules

**Date:** 2026-08-31
**Engines:** Madaros rebuilt from `origin/main` (`94adaff03a`); `lean_single`,
byte-identical to the `souc-stage2` CI builds.
**Scope:** `souc check` on all 1611 `.sio` files under `stdlib/`.

## Why the library and not the tests

Earlier sweeps compared engines on test files, where one divergence affects one
test. A stdlib module is different: if the engine CI exercises cannot typecheck a
module, **every test importing it is untestable there**. The effect multiplies
along the dependency graph rather than by file count.

## Raw result, and the 68% that is an artifact

    1611  stdlib modules checked under both engines
     978  verdict differs
     ├─ 896  lean_single refuses, Madaros accepts
     └─  80  Madaros refuses, lean_single accepts
        + 2  other

Splitting the 896 by the diagnostic the seed emits:

    607  error[E221]  no main   <- ARTIFACT: a library module is not a program.
                                   Checking it as one is my measurement error, not
                                   a divergence. Excluded.
    264  error[E035]  effect not declared in function signature
     15  error[E001]  type mismatch
      7  no code emitted
      3  E218 / E200 / E006

So the real population is ~289, not 978, and it is dominated by **one rule**, not
264 independent defects — the same shape as the "1028 failures, one segfault"
reading recorded in `ci.yml` for the corpus gate.

## The finding: E035 is enforced in both directions

`error[E035]` is "effect not declared in function signature". Each engine raises
it on cases the other accepts:

    tests/run-pass/approx_propagation.sio
      madaros      error[E035]        (missing: Approx, Mut, Div, Panic)
      lean_single  accepts

    stdlib/optimize/bfgs.sio
      madaros      check: OK
      lean_single  error[E035] at lines 479, 480, 481

This is not "one engine is stricter than the other". The two effect systems
disagree **symmetrically**: 264 stdlib modules where the seed demands a
declaration Madaros does not, and — from `ENGINE_DIVERGENCE_CORPUS_2026-08-30.md`
— run-pass tests where Madaros demands one the seed does not.

The reverse direction on the library side has its own shape, e.g.

    stdlib/research/erdos90_epistemic.sio
      lean_single  rc=0
      madaros      error[E036] confidence bound is not tight enough

## Consequence

Of the first 60 modules in the E035 population, **50 are referenced by a file
under `tests/`** — `stdlib/optimize/bfgs.sio` by
`tests/stdlib/optimize/test_bfgs_e2e.sio`, for instance. Whatever the correct
verdict is, the disagreement is not confined to unused corners of the library.

## What this does NOT establish

Which engine is right. E035 asks whether a function declares the effects of what
it calls; a difference could be a missing declaration in the library, or a
difference in how each engine propagates effects through imports, or both in
different modules. Adjudicating 264 modules is not attempted here.

## A method error worth recording

I suspected the no-main artifact early, opened one module (`bfgs.sio`) to test the
hypothesis, saw `error[E035]` — a real divergence — and treated the hypothesis as
refuted. `bfgs.sio` is not representative: in a twelve-file sample, five were
`E221`. A single case answers "is there at least one X?"; I used the answer as
though it were "are all of them X?". The two questions differ here by 607 files,
68% of the population.
