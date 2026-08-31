<!-- docs:meta
topic_id: repo.docs.audit.tolerated-buckets-remeasured-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-08-31
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.tolerated-buckets-remeasured-2026-08-31
-->

# The two tolerated baselines, re-measured — and a correction

**Date:** 2026-08-31
**Engine:** Madaros built from `origin/main` source (`d25b43a4fe`), and
`souc-stage2` built via `selfhost_host_gate.sh` with
`SOUNIO_FORCE_SOURCE_BOOTSTRAP=1`.

## Correction to ENGINE_DIVERGENCE_CORPUS_2026-08-30

That doc ends: *"110 tests ... do not pass under the compiler `bin/souc` invokes
by default, and no gate in this repository notices."*

**The second half is wrong.** `scripts/ci/madaros_corpus_regression_gate.sh`
exists, carries a checked-in baseline, and measures exactly this territory at
larger scope. Its header states the same motivation — the same three silent
miscompiles, the same principle that CI's `full-test-suite` runs the frozen seed.

What is true is narrower and better: **nothing in CI notices**, because the gate
is deliberately unwired, for a reason `.github/workflows/ci.yml` states in full.
That is managed debt, not blindness, and the distinction matters.

## The corpus gate's stated reason no longer holds at its stated size

`ci.yml` records the measurement of 2026-07-27: 1028 corpus programs failing
outside a 314-entry baseline, all traced to **one** segfault at
`module_frontend.sio:5183`, and concludes that wiring it before triage "would
fail Contracts for every open PR".

Re-run 2026-08-31 against a from-source Madaros (29 min, 1842 programs):

| | 2026-07-27 | 2026-08-31 |
|---|---|---|
| failures outside baseline | **1028** | **39** |
| total failures | — | 185 / 1842 |
| baseline entries that now PASS | — | **125** |

The gate now reports 125 baseline entries as fixed and 39 new. Triaging 39 and
reconciling 125 is a different decision from triaging 1028. The parked call is
worth revisiting with these numbers; this audit does not make it.

## The vacuous-annotation baseline mixes two populations

`tests/vacuous_expect_baseline.txt` exists for a real, well-documented reason: the
harness quoted the whole bash `=~` pattern, making it a literal string, so the
capture group never captured and **every** `expect-stdout` assertion matched
vacuously. Fixing the extraction made genuinely wrong annotations fail for the
first time, and the baseline tolerates those pre-existing entries so the fix could
land without turning `full-test-suite` red.

The mechanism is sound. The **label** is not, for part of the contents. Measured
across all 42 entries, compiling and running each under the suite's engine:

| origin | n | what the run shows |
|---|---|---|
| `tests/compile-fail/` | 24 | do not compile — correct for their kind; wrong `error-pattern` is plausible annotation debt |
| `tests/ui/` | 8 | do not compile |
| **`tests/run-pass/`** | **9** | **print their own `_FAIL` token** |
| `tests/stdlib/` | 1 | output matches the annotation |

The nine are not wrong annotations. Take `madaros_gum_fo_mutual.sio`:

    line  2: //@ expect-stdout: MADAROS_GUM_FO_MUTUAL_PASS
    line 70: print("MADAROS_GUM_FO_MUTUAL_PASS\n")
    line 72: print("MADAROS_GUM_FO_MUTUAL_FAIL\n")

The annotation is **correct**. The program takes its own failure branch. The
harness reports `Fail: 0` and files it under
`Vacuous-annotation baseline (tolerated)`.

All nine are GUM uncertainty propagation:

    madaros_gum_fo_eight_param     madaros_gum_fo_mutual
    madaros_gum_fo_if_helper       madaros_gum_fo_mutual_deep
    madaros_gum_fo_let_bytecode    madaros_gum_fo_struct_field
    madaros_gum_fo_let_ctor        madaros_gum_multichannel_fo
    gum_correlated

"The annotation was wrong" and "the program fails" are different debts. The first
is cosmetic; the second is a defect. Splitting the file, or adding a column, would
let the second population be counted.

## Engine disagreement inside the failure

The nine fail under both engines, so they are not an engine divergence — with one
exception, `madaros_gum_fo_sixteen_param.sio`, which passes under the seed and
fails under Madaros. But the engines disagree on the *numbers* they compute.
`madaros_gum_fo_mutual.sio`:

    lean_single   v_even=0.010000  v_peel=0.010000  v_odd=0.010000
    Madaros       v_even=0.000000  v_peel=0.012100  v_odd=0.000000

Madaros returns **zero** variance on two of three terms. Neither engine reaches
the PASS branch, so the test is honest about failing; but the two disagree about
what the wrong answer is, in the first-order uncertainty path.

## Method note

Three entry points to the same compiler bytes behave differently — the `bin/souc`
wrapper, the raw ELF CLI, and the harness. Counts here come from the harness
(`run_sio_test_suite_v2.sh`) and from the gates' own scripts, not from
hand-rolled invocations. See ENGINE_DIVERGENCE_CORPUS_2026-08-30 for what that
trap looks like.
