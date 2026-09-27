<!-- docs:meta
topic_id: repo.docs.audit.wrong-answer-divergence-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-08-31
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.wrong-answer-divergence-2026-08-31
-->

# Twelve programs where the default engine answers wrong, three of them silently

**Date:** 2026-08-31
**Engines:** Madaros rebuilt from `origin/main` (`94adaff03a`) that day, 99 429 229-byte
ELF; `lean_single`, byte-identical to the `souc-stage2` CI builds.
**Scope:** all 1854 files in `tests/run-pass/`, compiled and **run** under both.

## The class this looks for

Earlier sweeps covered the two visible classes: a refusal that does not happen
(`COMPILE_FAIL_DIVERGENCE_2026-08-31.md`, 32 cases) and functionality that breaks
(`ENGINE_DIVERGENCE_CORPUS_2026-08-30.md`, 112 cases). This one looks for the
class with no signal at all: **the program compiles, runs, exits 0, and prints a
different answer.**

## Result, with the controls that make it readable

    109  output differs between engines, both engines having compiled and produced an ELF
    109  DETERMINISTIC — each Madaros binary run twice gives identical output,
         so the difference is between engines, not run-to-run noise

Then, using each file's own `//@ expect-stdout` as the arbiter rather than my
judgement:

    48  no declared expectation      -> not judgeable by this method
    22  both engines match it        -> they differ elsewhere in the output
    19  only Madaros matches         -> the SEED is the wrong one; nearly all are
                                        `//@ requires: madaros` tests
    12  only lean_single matches     -> the DEFAULT engine answers wrong
     8  neither matches              -> broken under both

The 19 matter as much as the 12. This is not "Madaros is broken": on tests aimed
at Madaros the seed is the one that misses, and the split is close to even.

## The twelve, by failure mode

**Silently wrong — exit 0, wrong numbers. No signal whatsoever.**

    madaros_gum_fo_sixteen_param.sio
      madaros  v10=0.00     v16=0.00      rc=0
      lean     v10=0.0250   v16=0.0400    rc=0

    gum_fo_arity3_boundary.sio
      madaros  ADD2 5.0     ADD3 0.0      rc=0
      lean     ADD2 5.000   ADD3 14.00    rc=0

    print_f64_large_magnitude.sio
      madaros  MAG15 10…                  rc=0
      lean     MAG15 1.00…                rc=0

Variance terms come back **zero** where the test expects 0.0250 and 0.0400, and
the process reports success. This is the shape
`madaros_corpus_regression_gate.sh` was written for — its header cites a silent
miscompile that "corrupted a dissertation-path PBPK variance decomposition into
the wrong pharmacological conclusion".

**Assertion failure — exit 1, no output.** Multi-capture closure escape:

    closure_arity_2.sio   closure_escape.sio   closure_returned.sio
      madaros  rc=1, nothing printed
      lean     rc=0, PASS

`closure_returned.sio` computes `f(4)=19`, `g(10)=80`, `f(0)=7` from closures
returned out of their defining scope, asserts all three, then prints. Under
Madaros an assert fails, so the `print("PASS\n")` on line 20 is never reached.

**Crash.**

    heap_realloc_preserves.sio  rc=132 (SIGILL)
    heap_vec_pattern.sio        rc=132 (SIGILL)
    sparse_matrix.sio           rc=139 (SIGSEGV)   lean: 13/13 tests pass

Ten of the twelve run in CI — they are green there, because CI runs the engine
that gets them right.

## An instrument bug in my own sweep, and what it invalidates

The first pass captured the program's exit status as

    out=$(timeout 60 "$ELF" 2>/dev/null | head -c 4000); rc=$?

`$?` after a pipeline is **`head`'s** status, not the program's, so every row
recorded `rc=0`. That is why the sweep appeared to show 109 programs "both
exiting 0". The exit codes above were re-measured without the pipe; the crashes
and assertion failures were invisible under the original capture.

What survives unchanged: the 109 differ in **output**, and all 109 are
deterministic. Only the exit-status column was wrong, and it is not used above.

## What this does NOT establish

Which engine is right in the 48 with no declared expectation, or in the 22 where
both match the assertion and differ elsewhere. And for the 12: the test's
expectation is the arbiter used here, which is the right arbiter for a test
suite, but it is not a proof that lean_single's number is mathematically correct —
only that it is the one the fixture was written against.
