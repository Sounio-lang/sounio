<!-- docs:meta
topic_id: repo.docs.audit.optimizer-divergence-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-08-31
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.optimizer-divergence-2026-08-31
-->

# `-O` changed the behaviour of 632 of 1724 programs — FIXED, and now gated

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

## FIXED 2026-08-31 — one line, 587 of 632 programs recovered

`ocp_mfi_dse` decided a store was dead by scanning an instruction's SRC1/SRC2
fields. A call reads its arguments from `instr.call_args`, a Box list that peel
does not walk — a limitation the code states in a comment on `ocp_mfi_dce_once`
and reasons is safe "because calls are never removed". That reasoning covers
removing the CALL. It does not cover removing the DEF whose only consumer is a
call argument, which is what happened.

The peel already treats labels and jumps as barriers, for exactly this reason:
after them it cannot know what was read. A call has the same property. Adding
`IrCall` and `IrCallSret` to that barrier restores correctness without walking
the Box.

    before   632 diverged   (577 exit-code, 51 output, 4 build failures)
    after     51 diverged   (36 exit-code, 15 output, 0 build failures)
    fixed    587            zero true regressions

Of the 51 that remain, 48 diverged before as well. The other 3 were missed by the
first sweep, not caused by the fix — they behave identically under both
compilers. `qd_mul` is fixed by the same change, which answers the question left
open in the second reduction: it was ONE fault surfacing in two ways, not two.

Guarded by `scripts/ci/madaros_opt_chained_call_gate.sh`, wired into the
`Contracts` job. The gate builds one multi-module fixture twice with the same
compiler, differing only in `-O`, and refuses a difference. It carries a control:
if the fixture stops passing WITHOUT `-O` the gate fails rather than comparing
two broken builds and calling them equal.

## Result, as originally measured

    1864  files in tests/run-pass/
     140  do not build under Madaros at all, under either flag -> not measurable
    1724  measurable
     632  behave differently with -O
     ├─ 577  same output, exit code goes 0 -> non-zero
     ├─  51  different output
     │       ├─ 35  exit code also changes (0 -> 1, 3, 4, 6, 9, 30, 139, 182)
     │       └─ 16  EXIT CODE STAYS 0 — the silent class
     └─   4  fail to build under -O at all
       3  improve under -O (the only cases going the other way)

The direction is essentially unanimous: 577 of 580 exit-code changes are
pass-becomes-fail. Three go the other way.

**A correction to the denominator.** This document first said "632 of 1854". The
count came from an earlier sweep on a different base and was carried forward
without recounting; the tree measured here holds **1864** files. Worse, the probe
printed nothing when BOTH builds failed, so **140 files were silently dropped** —
they do not build under Madaros with or without the flag, and all 112 files from
`ENGINE_DIVERGENCE_CORPUS_2026-08-30.md` are among them. Absence of output was
being read as absence of divergence. The honest denominator is 1724 measurable
of 1864.

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

## ADDENDUM — one defect isolated, and where the rest actually lives

### A five-line reproducer: `-O` makes quad-double multiplication return zero

    use math::qd128::{qd_from_f64, qd_mul, qd_to_f64}
    fn main() -> i32 with IO, Mut, Panic {
        let v = qd_to_f64(qd_mul(qd_from_f64(2.0), qd_from_f64(3.0)))
        println("v="); print_f64(v)
        0
    }

    without -O   v=6.000000
    with    -O   v=0.000000

Controls isolating the operation — same shape, same import, same build commands:

    qd_add(2,3)   5.000000  /  5.000000    unaffected
    qd_sub(2,3)  -1.000000  / -1.000000    unaffected
    qd_mul(2,3)   6.000000  /  0.000000    BROKEN

And isolating the import rather than the arithmetic: `qd_to_f64(qd_from_f64(2.0))`
round-trips to 2.0 under both, so the defect is in `qd_mul`, not in crossing the
module boundary. Plain `f64` and `i64` multiplication, a local function call, and
a bare `println` are all unaffected under `-O`.

### This explains one case out of 632, not the blast radius

    632  programs affected
      1  imports qd128
    530  are `lorenz*`   (84%)
    246  have `imported` in the name

The `lorenz` family dominates, and its members behave differently from the qd128
case: sampled ones print nothing under either build and only the exit code moves,
0 -> 1. They are certificate-chain smoke tests that return a status rather than
text, which is why 577 of the 580 exit-code changes sit in this population.

So the reproducer above is *a* defect the sweep found, minimal and controlled. It
is not the cause of the 632. If the repository's own precedent holds — `ci.yml`
records 1028 corpus failures that were one segfault — the lorenz family is where
a second reduction should start, and it has not been attempted here.

### Second reduction, on the lorenz family — what it rules out

The `lorenz` family is 84% of the population, so it was reduced next. The result
is negative in a useful way: three plausible causes are excluded.

**Not arity.** `lorenz_i256_step1_taylor2_proof_trace_skeleton_check` takes nine
`i64` arguments and returns the wrong value under `-O`. Read through the process
exit status, which carries the low byte of the returned `i64`:

    expected 174108453 (mod 256 = 37)
      without -O   37     correct
      with    -O   255    = -1

An arity-8 sibling in the same module fails identically, with valid arguments
taken from its own test:

    lorenz_i256_step1_taylor2_replay_preflight_check   expected mod 256 = 187
      without -O   187        with -O   255

A first attempt at this comparison used arbitrary arguments (`1,2,3,4`) for the
low-arity calls. Those agreed across builds — but **vacuously**: both sides hit
the same `-1` validation sentinel, so the agreement said nothing. Only calls with
arguments the function accepts are informative here.

**Not the `Mod` effect.** All 80 `pub fn` in `lorenz_i256_cert_step1.sio` declare
`with Mod`, the held effect, and `stdlib/check/effects.sio` records that 2800
`with Mod` sites "still return -1" — which matches the observed sentinel exactly.
But the qd128 case carries no `Mod` at all and still breaks, so the hold does not
explain it.

**Not the calling convention, in the qd128 case.** `qd_add` and `qd_mul` have
identical signatures — `(Qd128, Qd128) -> Qd128 with Mut, Panic` — same arity,
same types, same effects. `qd_add` is unaffected under `-O`; `qd_mul` returns
zero. Whatever `-O` breaks there is in the multiplication body, not in how the
call is made or how the module boundary is crossed.

The two manifestations also differ in sentinel: the lorenz functions return `-1`,
`qd_mul` returns `0`. That is consistent with one optimiser fault surfacing in two
ways, and equally consistent with two faults. This audit does not decide which.

### ROOT CAUSE — `-O` does not propagate a variable between successive calls

The lorenz reduction reached bottom. Six lines, in a module, called from `main`:

    pub fn rp_add(a: i64, b: i64) -> i64 { return a + b }

    pub fn rp_chain3() -> i64 {
        var x: i64 = 0
        x = rp_add(x, 1)      // 1
        x = rp_add(x, 10)     // 11
        x = rp_add(x, 100)    // 111
        return x
    }

    without -O   111    correct
    with    -O   100

Not "the later calls are dropped". **Every call reads the ORIGINAL `x`, and only
the last assignment survives**: `rp_add(0, 100) = 100`. Two shorter cases confirm
the same reading:

    x=3; x = rp_add(x,5); x = rp_add(x,5)     without -O 13   with -O 8   = add(3,5)
    x=3; x = rp_mul(x,5); x = rp_mul(x,5)     without -O 75   with -O 15  = mul(3,5)
    x=3; x = rp_mul(x,5)                      without -O 15   with -O 15  correct

One call is fine; two or more in sequence lose the accumulation. It is not
arithmetic — `+` and `*` behave identically — and it is not the values.

### Why this explains the family

`lorenz_i256_step1_taylor2_proof_trace_skeleton_instance_fingerprint` chains
**seventeen** calls to `lorenz_i256_cert_mix`, accumulating into `fp`. Under `-O`
only the last one takes effect, so the fingerprint is wrong, and the caller's
`if artifact_fp != <expected>` returns `-1`. Lengthening the chain in a probe
moved the `-O` result by small increments — 191, 192, 194, 198, 202, 207 for
chains of 1, 2, 4, 8, 12, 17 — exactly what a single surviving call produces.

### What the reduction ruled out on the way down

Recorded because each was a plausible cause that cost a measurement:

    arity                 probes at 9, 10, 12, 14 and 17 imported args all agree
    argument marshalling  echo9 and sum9 return correct values under -O
    the `Mod` effect      qd128 carries none and breaks anyway
    the guard chain       removing all 17 early-return guards changes nothing
    nested cross-module   a module-to-module call, one deep, agrees
    the arithmetic        the same mix chain written locally agrees at length 17
    the flag block        the same 60-line flag body, local AND imported, agrees

The last two are the interesting pair: the identical body is correct as a local
function and correct as an imported one — the defect needs the *chained
assignment* specifically.

### The 51 that remain — a SECOND defect, not the same one

The DSE fix removed 587. The remainder splits:

    36  exit code changes    23 lorenz_i256_step*, 6 solver_portfolio*, 7 assorted
    15  output changes       all numeric: complex_arithmetic, polynomial_ops,
                             gum_euler_ode, gum_iso_budget_ode, pbpk_caffeine,
                             test_fem, test_integral_eq, optimization_nelder_mead,
                             epistemic_nuclear_decay, nlme_test, gum_reporting,
                             correlated_eq_identity, gtt_reassignment_topology,
                             clinical_dyadic_non_reduction_{,native_}witness

`complex_arithmetic.sio` fails exactly one of its 18 sub-tests under `-O`
("sqrt negative"), `polynomial_ops.sio` exactly one of 19 ("roots repeated").
Both are single wrong values, not a collapsed program.

**Reproducer** (with that file's own hand-written math functions in scope):

    let r = sqrt_f64(1.0)
    print_f64(atan2_f64(0.0, 0.0 - 1.0))

    without -O   3.141593    correct
    with    -O   3.933087

3.933087 − 3.141593 = 0.791495, which is exactly `atan_f64(1.0)` as that file
computes it. So `atan2_f64` reaches its `atan_f64(y / x) + pi` branch and the
inner call behaves as though it received 1.0 instead of `0.0 / -1.0`.

**The preceding call is the trigger.** Remove the `sqrt_f64` line and the same
`atan2_f64` call is correct. `sqrt_f64` runs a 60-iteration Newton loop;
`atan_f64` runs a 40-iteration Taylor loop.

### Ruled out for this second defect

    a preceding atan_f64 instead of sqrt_f64      correct -- the trigger is sqrt_f64
    atan_f64 called directly, any input           identical under both builds
    atan2_f64 called directly, no preceding call  identical under both builds
    struct fields as call arguments               identical
    an inline division inside a call argument     identical, with and without a
                                                  preceding sqrt_f64
    a stale parameter                             sqrt_f64(K) then atan_f64(0.0)
                                                  returns 0 for K = 1, 4, 0.25 --
                                                  the corrupted value does not
                                                  track the previous argument

The defect needs the branch chain inside `atan2_f64` AND a preceding `sqrt_f64`.
Neither alone reproduces. This is not the call_args/DSE fault fixed above -- that
one is fixed and this survives it.

**Root cause not found.** Recorded at this depth so the next attempt starts from
a two-line reproducer and six dead hypotheses rather than from 51 files.

## What this does NOT establish

**Where the defect is.** 632 programs is not 632 bugs — the corpus gate's own
history records 1028 failures that were one segfault. A single wrong rewrite in
the e-graph or a single bad peephole could produce all of this. The number
measures blast radius, not defect count.

**That the unoptimised path is correct.** These comparisons are `-O` against no
`-O` on the same engine. Where the two disagree, the flag changed something; that
the default build is right is assumed here, not shown.
