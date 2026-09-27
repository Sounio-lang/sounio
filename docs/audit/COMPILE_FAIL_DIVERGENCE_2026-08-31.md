<!-- docs:meta
topic_id: repo.docs.audit.compile-fail-divergence-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-08-31
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.compile-fail-divergence-2026-08-31
-->

# 32 refusals that do not happen under the default engine

**Date:** 2026-08-31
**Engines:** Madaros built from `origin/main` source that day; `lean_single`, the
committed seed, which is byte-identical to the `souc-stage2` CI builds (verified
in `ENGINE_DIVERGENCE_CORPUS_2026-08-30.md`).
**Scope:** all 649 files in `tests/compile-fail/`, compiled under both engines.

## Why this sweep and not another

`tests/run-pass/` has been swept under Madaros before — that is what
`scripts/ci/madaros_corpus_regression_gate.sh` does. `tests/compile-fail/` had
not been. The two failure classes are not equally visible:

- a `run-pass` test failing under Madaros is **broken functionality** — whoever
  runs the program sees it;
- a `compile-fail` test that **compiles** is a **refusal that does not happen** —
  silent by construction, because the program that should have been rejected
  simply runs.

## Result

    649 compile-fail files
     46 compile successfully under Madaros
     ├─ 14  neither engine refuses  -> ALL DECLARED (see below), not a finding
     └─ 32  lean_single refuses, Madaros compiles, NOTHING declares it

The 14 are honest: twelve carry `//@ ignore`, `ir_max_instrs_wall.sio` carries
`//@ requires`, and `door1_too_many_locals_1025.sio` is listed in
`tests/known_failures/hardened_diagnostics_full_suite.txt` — the harness reports it
as `FAIL … expected compile failure but passed` and the known-failures file absorbs
it. Unenforced, but said out loud.

The 32 are not. Every one carries a plain `//@ compile-fail` with **no**
`ignore`, **no** `known-failure`, **no** `requires`. CI is green on them because
CI runs `lean_single`, which refuses them. `bin/souc` — "the compiler people
invoke for normal use", per its own header — compiles them to an ELF.

## The 32

    file                                          lean_single   what the test asserts
    affine_double_use.sio                        —           affine value used more than once
    closure_effect_escape.sio                    error[E035]   
    confidence_gate_reject.sio                   error[E214]   confidence gate violation
    covid_2020_knightian_refusal.sio             —           Knightian uncertainty (ε=⊥) c
    diagnostic_codes_no_main.sio                 error[E221]   error[E221]
    effect_handler_missing_arm.sio               error[E035]   effect not declared in function 
    effect_resume_wrong_type.sio                 error[E035]   effect not declared in function 
    door1_too_many_globals_1025.sio              —           too many globals
    epistemic_path_private_field_access.sio      —           private struct field access
    gtt_body_precision_unused_param_refused.sio  —           requested channel not in express
    gtt_hessian_out_of_topology.sio              —           requested channel not in express
    gtt_let_bound_out_of_topology.sio            —           requested channel not in express
    gtt_indirect_call_out_of_topology.sio        —           requested channel not in express
    gtt_loop_wrong_channel.sio                   —           
    gtt_nary_out_of_topology.sio                 —           requested channel not in express
    gtt_reassignment_wrong_channel.sio           —           
    halo_pgx_gate_refuse.sio                     error[E214]   EpistemicComplete violation
    linear_early_return.sio                      —           not consumed
    linear_field_unconsumed.sio                  —           not consumed
    linear_loop_consume.sio                      —           consumed
    nested_match_outer_non_exhaustive.sio        —           match must be exhaustive
    refinement_guard_and_escape.sio              error[E208]   refinement type violation
    refinement_let_violation.sio                 error[E208]   refinement type violation
    refinement_inline_param_violation.sio        error[E208]   refinement type violation
    refinement_violation_positive.sio            error[E208]   refinement type violation
    refinement_violation_probability.sio         error[E209]   refinement type violation
    reinhart_unit_confusion.sio                  error[E209]   refinement
    study_missing_hypothesis.sio                 error[E211]   study block requires at least on
    tac_compile_gate_refuse.sio                  error[E214]   EpistemicComplete violation
    tuple_destructure_arity_mismatch.sio         error[E213]   arity mismatch: has 2
    unit_f64_unit_expr_unknown_reject.sio        error[E200]   unknown unit in f64<UnitExpr> an
    vancomycin_low_conf.sio                      —           ε

Grouped by what is not refused:

    7  GTT channel/topology violations  (gtt_*)
    5  refinement type violations       (E208/E209)
    4  linear / affine resource discipline
       affine_double_use, linear_early_return,
       linear_field_unconsumed, linear_loop_consume
    4  epistemic gates                  (E214, Knightian, low-confidence)
    3  effect system                    (E035: escape, missing arm, wrong resume type)
    9  assorted: no main, non-exhaustive match, tuple arity, unknown unit,
       study block, private field access, too many globals

Sixteen show no `error[Exxx]` in the lean output because the refusal is printed in
another form; all 32 exited non-zero under lean_single and zero under Madaros,
which is the comparison being made.

Hand-verified outside the sweep, three of them:

    affine_double_use.sio         madaros rc=0 elf=12744   lean rc=1
    closure_effect_escape.sio     madaros rc=0 elf=12744   lean rc=1 error[E035]
    effect_resume_wrong_type.sio  madaros rc=0 elf=8648    lean rc=1 error[E035]

## What this does NOT establish

**Which engine is right, per file.** Five of the 32 are refinement-type
violations, and `ENGINE_DIVERGENCE_E221_REFINEMENTS_2026-08-30.md` already shows
Madaros does not implement refinement predicates at all — so it accepting a
refinement violation is the same unimplemented feature seen from the other side,
not a separate defect. Others may have their own explanations. This audit
establishes that the two engines disagree on 32 refusals and that nothing in the
tree declares it; it does not adjudicate 32 cases.

**That the guarantees are unsound in practice.** A refusal missing from one
engine is not the same as a program doing damage. What is certain is narrower:
these fixtures exist to prove a rejection, the rejection does not occur under the
default compiler, and no annotation says so.

## Method note

Counts come from `bin/souc compile` under both engines and, for the declared/not
question, from the harness and the known-failures file. A "compiled" verdict
requires exit 0 **and** the absence of `typecheck: failed` in the output, because
the wrapper reports that condition with a zero exit.
