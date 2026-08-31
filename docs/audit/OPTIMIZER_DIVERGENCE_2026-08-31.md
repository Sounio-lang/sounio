<!-- docs:meta
topic_id: repo.docs.audit.optimizer-divergence-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-08-31
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.optimizer-divergence-2026-08-31
-->

# `-O` changes the behaviour of 632 of 1854 programs, and nothing in CI runs it

**Date:** 2026-08-31
**Engine:** one Madaros, rebuilt from `origin/main` (`94adaff03a`). Same compiler,
same source, same machine — the only variable is the flag.
**Scope:** every file in `tests/run-pass/`, built twice and run.

## Why this axis differs from the engine-divergence audits

The four engine audits
(`COMPILE_FAIL_DIVERGENCE`, `ENGINE_DIVERGENCE_CORPUS`, `WRONG_ANSWER_DIVERGENCE`,
`STDLIB_ENGINE_DIVERGENCE`) all end at a question this repository has to answer
with judgement: *which engine is right?* Two implementations disagreeing is not
by itself a defect in either.

This one does not have that ambiguity. One compiler, one source file, one build
command differing by a flag. **An optimiser that changes a program's observable
output is wrong by definition, not by opinion.**

## Result

    1854  programs in tests/run-pass/
     632  behave differently with -O
     ├─ 577  same output, exit code goes 0 -> non-zero
     ├─  51  different output
     │       ├─ 35  exit code also changes (0 -> 1, 3, 4, 6, 9, 30, 139, 182)
     │       └─ 16  EXIT CODE STAYS 0 — the silent class
     └─   4  fail to build under -O at all
       3  improve under -O (the only cases going the other way)

The direction is essentially unanimous: 577 of 580 exit-code changes are
pass-becomes-fail. Three go the other way.

**Determinism control:** all 51 output-changing programs and a 60-file sample of
the exit-code-changing ones were built without `-O` and run twice. All 111 gave
identical output and exit code both times, so the differences are caused by the
flag, not by run-to-run variation.

## The silent sixteen

Output changes, exit code stays 0. Nothing observable reports a problem:

    box_all_read_forms.sio
    correlated_eq_identity.sio
    darwin_pop_epistemic_smoke.sio
    dissertation_frontend_parity_alt.sio
    dissertation_frontend_parity_ref.sio
    dissertation_pbpk28_parity_ref_haloperidol.sio
    dissertation_pbpk28_degenerate_parity_ref.sio
    dissertation_pbpk28_parity_ref_venlafaxine.sio
    epistemic_nuclear_decay.sio
    gtt_reassignment_topology.sio
    gum_euler_ode.sio
    gum_iso_budget_ode.sio
    multi_agent.sio
    seq_kaxi_order_independence.sio
    sret_8field.sio
    sprint235_print_f64_e2e.sio

    box_all_read_forms.sio
      without -O   BOXMATRIX OK
      with    -O   BOXMATRIX FAIL
      both exit 0

The program's own self-check flips from OK to FAIL and the process still reports
success. Three of the sixteen are `dissertation_pbpk28_parity_ref_*` —
haloperidol, venlafaxine, and the degenerate reference — the pharmacological
parity path `scripts/ci/madaros_corpus_regression_gate.sh` names in its header
when it explains why silent miscompiles matter.

## Nothing exercises the flag

`-O, --optimize   Enable optimizations` is advertised in the compiler's own
`--help`, so it is a surface users are invited to use.

    workflows           0 uses   (`Mach-O` matches are false positives)
    gates using it      2        exact_bitwise_rebracket_authority_gate.sh
                                 ordered_path_provenance_source_ir_gate.sh
    of those, wired     0
    compiler's own build   does not use it

So the entire optimised path is untested: no gate in CI compiles anything with
`-O`, and the two gates that do are not wired.

## What this does NOT establish

**Where the defect is.** 632 programs is not 632 bugs — the corpus gate's own
history records 1028 failures that were one segfault. A single wrong rewrite in
the e-graph or a single bad peephole could produce all of this. The number
measures blast radius, not defect count.

**That the unoptimised path is correct.** These comparisons are `-O` against no
`-O` on the same engine. Where the two disagree, the flag changed something; that
the default build is right is assumed here, not shown.
