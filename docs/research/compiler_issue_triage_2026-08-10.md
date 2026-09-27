<!-- docs:meta
topic_id: repo.docs.research.compiler-issue-triage-2026-08-10
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.compiler-issue-triage-2026-08-10
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Compiler issue triage — 82 open issues against `main@6b2198e314`

Date: 2026-08-10
Authority: repo_only
Status: measurement complete for the probed set; desk verdicts labelled as such

Every verdict below is labelled **measured** or **desk-inferred**. No desk inference is
written as if it were measured. Where this document disagrees with
`docs/serious-language/public-claim-registry.v1.tsv`, the registry wins and the
disagreement is flagged rather than silently resolved.


## Summary

**Measured, not inferred.** A compiler built from source at the measured commit; 31 probes across
engine x optimisation x verb; a sweep harness proven to report FAIL on a deliberately wrong
expected value before any verdict was taken from it.

**12 open issues reproduce** on current `main`, **5 do not** and are candidates to close (#643,
#877, #891, #933, #1070). Two issue texts are wrong about their own defect: #1667's headline
`add3(1,2,3)` case is fixed while a different shape still fails, and #852 does not "exit within
seconds" but fails to terminate at all.

**Three things this triage found that were not on anyone's list:**

1. The suite's `expect-stdout` / `error-pattern` assertions have never run. 933 test files carry
   one of the two markers; arming them turns **108 currently-green tests red** and recovers none.
2. A type error in the compiler's own source **builds successfully** and yields a compiler that
   segfaults later — #1494, hit first-hand during this work rather than read from a report.
3. `formal/lean4/lean-toolchain` pins a floating `leanprover/lean4:stable`, so the proofs gate
   breaks on toolchain drift and blames whichever commit merges next.

**Landed:** #1490. Every compiler job is green on `main` before and after it — `Contracts`,
`Sounio Lint`, `Native Self-Host` (Linux and macOS), `Source-Bootstrap Self-Host`, and
`Madaros Current-Source f64 Lowering`. `main` *is* red, on `Lean Proofs` only, from the floating
toolchain pin described below and not from this merge.

**Also landed:** #1501, after porting it onto main's interned-id representation.

**Also landed:** #1493, after repairing the ratchet failure it introduced (closes #1471).

**Repaired and awaiting CI:** #1500, #1508. Each initially failed the self-compilation ratchet —
the compiler built from the tree could no longer typecheck its own entry point (`E175`;
`2xE002`+`E137`). All three causes are fixed and the ratchet now reports
`MADAROS_FIXED_POINT_OK` on both. See "The ratchet failures, resolved".

**Closed as not reproducing:** #643, #877, #891, #933, #1070 — each with its measurement.
**Corrected in place:** #1667 (title symptom fixed, comment-1 shape still fails) and #852
(does not terminate, rather than exiting in seconds).

**#1531 landed** after its delta was measured correctly (16 failures under CI's compiler, not the 108 first reported here) and each one dispositioned — see below.

The two systemic findings explain the backlog's shape better than any individual bug does: type
errors in the compiler do not stop a build, and wrong answers at rc=0 do not fail a test.

## Provenance of the measuring compiler

The checked-in binaries were not used: `artifacts/self-hosted/madaros` predates the newest
`self-hosted/` commit, and its gate receipt attests a source commit from 2026-07-11. A
compiler was built from source at the measured commit.

| field | value |
|---|---|
| repo commit | `6b2198e3147a1a1cddb873c25c5e3145b3b674b9` |
| binary | `madaros-6b2198e314`, 99 106 962 bytes |
| sha256 | `4a2e957456f4da54bd0d622f57568466d443eb1e2d4ce63f6b01221b88fb22a9` |
| `--version` | `Madaros v0.80.0 -- the Sounio self-hosted compiler` |
| `scripts/gates/g6_madaros_identity.sh` | **PASS** |
| drift during the session | `main` advanced to `a89cf97121` — a prebuilt-binary refresh (`[skip ci]`) with no source change, so every verdict below still describes current source |
| independent rebuild | same commit rebuilt on a SLURM node produced **99 106 962 bytes** — identical size |

lean_single probes used the in-tree fixed-point seed `bin/souc-lean-single-x86_64`
(sha256 prefix `483a452db88774a1`).

## The finding that conditions every other number in this repo

`scripts/dev/run_sio_test_suite.sh` is a symlink to `run_sio_test_suite_v2.sh`, which at
lines 312–316 writes:

```bash
if [[ "$line" =~ "//@ expect-stdout:\ "(.*) ]]; then
```

Quoting the right-hand side of `=~` makes bash match it **literally**. Verified directly
under bash 5.2.21: the quoted form does not match at all, while the unquoted control matches
and captures. `expect_stdout` and `error_patterns` therefore stay empty arrays and their
assertion loops iterate zero times.

**Consequence: the run-pass suite asserts only "exit 0" and the compile-fail suite only
"exit != 0".** Counted on this commit, the suite collects **2 951** tests (1 729 `run-pass`,
613 `compile-fail`, plus `ui/`, `stdlib/` and `gpu/` subsets), and **933 of the collected
files carry one of the two dead annotations** — 368 `//@ expect-stdout:` and 565
`//@ error-pattern:`. Every one of those assertions is currently skipped. The file's own
comment concedes the defect and estimates "~305 latent vacuous passes" would flip. Issue
**#444** tracks it; PR **#1531** fixes it.

This matters because the dominant bug class in this backlog is precisely the one that
assertion-free testing cannot see: a wrong answer at rc=0. Accordingly, **no verdict in this
document is taken from a gate result** — each probe asserts an expected value drawn from the
issue text, and the sweep harness was itself proven non-vacuous before use (feeding it a
deliberately wrong expected value makes it report FAIL, not PASS).


## Measured: what arming the assertions costs (#1531 / #444)

Both passes ran on one SLURM node against one from-source build, differing only in the two-line
regex fix (glob + parameter expansion, the robust form the `typecheck-fail` path in the same
file already uses).

| pass | Pass | Fail | Known-failure | Skip | Total |
|---|---|---|---|---|---|
| A — harness as `main` ships it (disarmed) | 1209 | 462 | 64 | 1216 | 2951 |
| B — assertions armed | **1098** | **571** | 66 | 1216 | 2951 |

**108 tests fail only once the assertions are armed. None are recovered.** By prefix: 22
`ontology_*`, 14 `madaros_*`, 8 `epistemic_*`, 5 `knowledge_*`, 4 `observe_*`, 3 each
`refinement_*` / `linear_*` / `call_*`, and a tail of singletons.

A large share are `compile-fail` tests (`ontology_*_reject`, `linear_double_use`,
`ownership_use_after_move`) whose `//@ error-pattern:` now has to match a real diagnostic. Those
have been passing on "exit != 0" alone, so a test could have been failing for entirely the wrong
reason and still counted green.

**This is a repository-wide event, not a triage item.** 108 green tests going red is outside
what a triage pass should absorb, and per the agreed plan this document stops here and hands the
decision back rather than folding it into a merge. The number itself is the deliverable: it is
the cost of finding out what the suite has not been checking, and it is lower than the ~305 the
harness comment predicted.

Recommended sequencing, for whoever picks this up: land #1531 together with a triage of the 108,
not before it. Some will be stale expected-text, some will be genuine defects that the vacuous
assertion has been hiding since the marker was introduced.

## Bucketing of all 82 open issues

| bucket | n | issues |
|---|---|---|
| measured — REPRODUCED | 12 | #639, #852, #858, #888, #1487, #1574, #1610, #1667, #1678, #1682, #1692, #1693 |
| measured — NOT REPRODUCED | 5 | #643, #877, #891, #933, #1070 |
| fix exists in an open PR | 7 | #444, #854, #1021, #1471, #1485, #1487 (also measured), #1507 |
| `stale`-labelled, never re-verified | 5 | #637, #638, #641, #645, #725 |
| out of lane (docs/research/design) | 14 | #834, #839, #874, #896, #1122, #1162, #1240, #1263, #1386, #1423, #1484, #1489, #1557, #1688 |
| desk-inferred, unprobed | 40 | #325, #326, #469, #740, #789, #821, #830, #842, #846, #878, #882, #884, #886, #1156, #1157, #1161, #1415, #1472, #1491, #1494, #1495, #1498, #1499, #1535, #1542, #1567, #1568, #1570, #1577, #1581, #1582, #1584, #1586, #1644, #1646, #1649, #1655, #1658, #1680, #1686 |

The 40 unprobed are a **named cut, not a silent one**: the approved budget was ~30 measured
compiler issues, and the cut followed the ranking in the last section. 56 of the 82 carry no
runnable program in either the body or the comments, which is the main reason the cut fell
where it did.

## Measured verdicts

31 probes across the engine × opt × verb matrix. Full results:
`results.tsv` (one row per probe, with a log path per row).

### REPRODUCED on current main

| issue | config | expected | observed |
|---|---|---|---|
| **#1692** | nv2 `-O`, compile only | compiler exits 0 | **rc=139 segfault**; without `-O` rc=0 |
| **#1693** | nv2 `-O`, compile only | compiler exits 0 | **rc=139 segfault**; without `-O` rc=0 |
| **#1682** | nv2 `-O` | `a0=7 a3=7 \| b0=7 b1=5` | **hangs, rc=124, no output**; without `-O` correct |
| **#1667** | nv2 `-O` (comment-1 repro) | exit 0 | **rc=1**; without `-O` rc=0 |
| **#1678** | lean_single | `hoisted=5 inplace=5` | **`hoisted=5 inplace=15739125760216`** — pointer-shaped garbage |
| **#1574** | lean_single | prints 1.0 … 7.0 | **`1.000000` seven times** — every index reads element 0; nv2 correct |
| **#1610** | lean_single | must not silently compute f128 in f64 | **`s-1 = 0.000000`** — silently wrong; Madaros correctly refuses |
| **#639** | nv2 | `true` | **`false`** — wrong match arm |
| **#639** | lean_single | `true` | **compile error E200** `undefined identifier eps` on the enum payload binding (measured). Whether this is the same defect as #645 is *desk-inferred* from title similarity and was not probed |
| **#852** | both engines | prints `n=…` | **does not terminate within 120 s** at 99 % CPU, no output (see correction below) |
| **#888** | madaros `check` | must reject the shared-ref store | **accepted, rc=0** |
| **#858** | nv2 | exit 0 | **compile error rc=1** |
| **#1487** | nv2 | `s1.tag=1 s1.arr0=7 s2.arr0=99` | **`s1.tag=1 s1.arr0=99 s2.arr0=99`** — the array-typed field aliases while the scalar field copies correctly; lean_single is correct. This is the direct evidence that PR #1501 is still needed |

### NOT REPRODUCED — candidates to close

| issue | evidence |
|---|---|
| **#643** copy-then-mutate struct aliasing | in-tree repro `docs/handoff/repros/d6_struct_copy_aliases.sio` prints the correct `p.flag=false q.flag=true` on **both** engines |
| **#891** `print_int` garbled after an f64 print | prints the expected `5.700000 (42)` on both engines |
| **#933** `&local_array` to a builtin | prints `hi`; PR #1280 landed the `&buf` auto-unwrap |
| **#1070** `-O` crashes on reduced scalar programs | all three of the issue's reproducers correct with and without `-O` |
| **#877** E039 on linear consume-and-return | `madaros check` exits 0 |

### Two corrections to the issues' own text

**#1667 — the headline symptom is fixed; the class is not.** The issue's last comment states
"This issue's ORIGINAL symptom is still open — `add3(1,2,3)` still returns 3". Measured on
`main@6b2198e314`, `add3(1,2,3)` prints **6**, correctly, both with and without `-O`. What
still fails is the 7-line repro from comment 1 (three chained `if/else` increments), which
returns 1 under `-O` where it must return 0. The issue should be retitled to the surviving
shape rather than closed or left as-is.

**#852 — the symptom changed.** The issue says "no output, exits within seconds". Measured:
the program compiles fine (42 047-byte ELF) and then **does not terminate within 120 s**,
burning 99 % CPU with no output, on both engines. Whether that is a loop or a capacity wall
was not determined; what is measured is non-termination, which is a different failure mode
from the "exits within seconds" the issue records.


## #1494 demonstrated itself during this triage

While restacking PR #1501 onto `main`, one accessor in `self-hosted/ir/lower.sio` was left
reading `StructFieldEntry.name` after `main` had renamed that member to `name_id: i64`. The
enclosing function is declared `-> Name`.

**The compiler built anyway — `build_modular_madaros.sh` returned rc=0 after 232 s** — and the
resulting compiler then **segfaulted (rc=139) on the first program that exercised the deep-copy
path**. A type error in the compiler's own source produced a working build and a broken binary.

That is exactly **#1494** ("typecheck errors in imported modules are non-fatal and still emit
code"), observed here as a first-party consequence rather than a report. It also raises #1494's
practical severity: it does not merely let bad user code through, it lets the compiler
mis-build itself, and the failure surfaces far from its cause.

Combined with the harness finding above, the two together explain the shape of this backlog:
type errors in the compiler do not stop a build, and wrong answers at rc=0 do not fail a test.


## #1507 is still live on `main`, and the disarmed harness is why nobody noticed

PR #1508 ships its own acceptance test, `tests/run-pass/tuple_f64_slot_arithmetic.sio`,
annotated `//@ expect-stdout: ALL PASS`. Run against **plain `main@6b2198e314`**, that test
prints **`SOME FAIL`** and exits **rc=0** (measured).

Two things follow.

First, `main` did independently implement a fix for #1507/#1502 —
`lower_type_expr_is_tuple_of_all_f64`, which marks an all-f64 tuple with the same
`returns_float=2` code already used for f64 arrays. But it does **not** cover a **mixed**
tuple such as `(i64, f64)`, and that is precisely the shape the issue is about: the
`covariance()` accessor in `stdlib/epistemic/correlation.sio` returns `(i64, f64)`. So the
defect survives in the exact case it was filed for.

Second, this test **passes CI today**. Its only real assertion is the `expect-stdout` marker,
and that marker is skipped by the disarmed harness; `rc=0` is all the suite checks. This is the
#444 finding with a concrete, current victim rather than a projection.

PR #1508's design (a per-slot `1024+mask`) subsumes `main`'s all-f64 special case, so the
restack keeps the mask and drops the superseded branch. That is a **design choice, not a
textual merge**, and it is recorded here because a future reader will otherwise see main's
helper disappear and assume the merge lost it.

## Blocker records

```text
Blocker-ID: BLK-20260810-optimizer-O-segv-global
Status: reproduced
Severity: B1
Class: compiler-semantics
Lane: optimizer / opt_cleanup
Repro: madaros --native-v2-compile <one-line file containing only `pub let G: i64 = 5`> out.elf -O
Observed: compiler exits 139 (SIGSEGV)
Expected: compiler exits 0, as it does without -O
Acceptance-Gate: the same command exits 0
Evidence-Level: E2
Evidence: results.tsv rows i1692_nv2_O / i1692_nv2_noO
Next-Action: bisect opt_cleanup against the no--O control
```

```text
Blocker-ID: BLK-20260810-imported-typeerror-nonfatal-selfbuild
Status: reproduced
Severity: B1
Class: compiler-semantics
Lane: checker / imports (issue #1494)
Repro: leave `StructFieldEntry.name` (renamed to `name_id` on main) in a `-> Name` accessor in
  self-hosted/ir/lower.sio, then run scripts/ci/build_modular_madaros.sh
Observed: build rc=0; the produced compiler segfaults (139) on the first deep-copy program
Expected: the build fails with a type error naming the accessor
Acceptance-Gate: the same tree fails to build
Evidence-Level: E2
Evidence: logs/verify-1501c.log (build rc=0 elapsed=232s, probe rc=139)
Next-Action: fold this first-party witness into #1494; it is stronger than the original report
```

```text
Blocker-ID: BLK-20260810-lean-global-array-index
Status: reproduced
Severity: B1
Class: compiler-semantics
Lane: lean_single seed
Repro: bin/souc-lean-single-x86_64 <literal-initialised global [f64;7], printed in a loop> out.elf
Observed: 1.000000 printed seven times
Expected: 1.000000 .. 7.000000
Acceptance-Gate: the seven values differ
Evidence-Level: E2
Evidence: results.tsv row i1574_lean (nv2 control passes)
Next-Action: the global literal-initialiser path, not the indexing path -- the nv2 control is correct
```



## CI baseline before any merge

`main` is **not branch-protected**. The `Release Gate` workflow has failed on `main` every day
from 2026-08-03 through 2026-08-10 — eight consecutive runs. The failing job is **`Apple
Self-Host`**; `Full Self-Host (5 generations)`, `Source-Bootstrap Self-Host`, `Rust-Free Proof`,
`Seed Policy`, `LSP Smoke` and `Open Release Blockers` all pass.

That is the macOS lane, which is open issue **#821**. Recording it here because the Merge
Contract's B0 stop-condition is "CI red on `main` after merge": red-on-Apple is the **baseline**,
not a consequence of anything landed from this triage, and anyone landing work needs that
distinction to avoid either a false alarm or a real regression hiding behind a known red.

## Restacking the fix PRs onto `main`

GitHub's `mergeStateStatus` is measured against each PR's **declared base**, not `main`. #1501
reports `CLEAN` because its base is #1490's branch; against `main` it conflicts. Real merge
attempts, not the API, give this:

| PR | closes | vs `main` | resolution | verified |
|---|---|---|---|---|
| #1490 | unblocks #1487 | **clean** | none needed | constant bump, `2^20 -> 2^22` |
| #1501 | #1487 | 10 hunks (`ir.sio`, `lower.sio`) | ported `StructFieldEntry` onto main's interned-id representation | **yes** — see below |
| #1500 | #1485 | 2 hunks (`module_frontend.sio`) | took main: its structured `parse_failed_path` supersedes the PR's `print()` diagnostic | in progress |
| #1493 | #1471 | 5 hunks (`lower.sio`) | took main: its `(*reused)` in-place writes supersede the PR's `let preserved` copy of a ~504 KB `IrFunction`. The PR's actual fix is in `check/check.sio` and is untouched | **yes** — `LINEAR_NESTED_BRANCHES_OK`, and the compile-fail case rejects with `error[E040]` |
| #1508 | #1507 | 3 hunks (`lower.sio`) | **design choice** — kept the PR's `1024+mask`; see the #1507 section | **yes** — `ALL PASS` (plain `main` gives `SOME FAIL`) |

**#1501 verification.** After the port, a build from the merged tree passes all three checks:

| check | result |
|---|---|
| authored #1487 probe | `s1.tag=1 s1.arr0=7 s2.arr0=99` — aliasing gone |
| `tests/run-pass/aggregate_nested_field_deep_copy.sio` | `AGGREGATE_NESTED_DEEP_COPY_OK` |
| `tests/run-pass/perturb_struct_array_field_no_alias.sio` | `PERTURB_NO_ALIAS_OK` |

Two resolution traps worth recording, both of which a textual merge would have taken silently:

- `main` had changed `confidence` from `is_float: 1` to `is_float: 3` after the PR forked.
  Taking the PR's side of that hunk reverts a later fix.
- `main` renamed `StructFieldEntry.name` to `name_id`. One accessor was left on the old name;
  the compiler **built anyway** and segfaulted at the first deep copy — see the #1494 section.

Nothing has been merged to `main`. Every result above comes from a local restack built and
tested on an isolated node.


### #1508 took two attempts, and the first failure is instructive

Taking the PR's side of all three hunks produced a compiler that **segfaulted (139)** on the
PR's own test. One of the three was not a competing fix at all but a `main` **representation**
change: the locals stack became a pointer written in place (`(*stk).field[i]`, `lo.locals = stk`)
instead of a value re-boxed (`stk.field[i]`, `lo.locals = Box::new(stk)`). Taking the PR's side
there reintroduced a value copy of a struct holding several `[i64; 4096]` arrays.

The working resolution treats the three hunks separately:

1. classifier — PR's `1024+mask` (subsumes main's all-f64-tuple special case)
2. locals stack — **main's** pointer form, plus the PR's new `tuple_float_mask` field
   initialised in that form
3. float-classification of a field access — **union**: consult the PR's per-slot mask first,
   then fall through to main's two existing lookups

The lesson generalises to the rest of this backlog: in a two-week-stale PR against this
compiler, "same file, same function" hunks are a mix of *competing fixes* and *representation
changes*, and the two need opposite resolutions.


### Joint verification before landing

The four remaining PRs all touch `self-hosted/ir/lower.sio`, so verifying them one at a time
against `main` does not establish that they coexist. An integration branch off
`main@6d84b8d19b` (i.e. after #1490 landed) merged all four resolved trees **cleanly**, and one
build from it gives:

| check | result |
|---|---|
| `probe_1487.sio` | PASS |
| `aggregate_nested_field_deep_copy.sio` | PASS |
| `perturb_struct_array_field_no_alias.sio` | PASS |
| `linear_nested_branches.sio` | PASS |
| `tuple_f64_slot_arithmetic.sio` | PASS (`ALL PASS`) |
| `linear_consumed_in_one_branch_only.sio` | correctly rejected, `error[E040]` |
| `stdlib/darwin_pbpk/epistemic_pbpk28.sio` | **compiles (rc=0)** and runs 8/9 — plain `main` cannot compile it at all (that is #1485) |
| `scripts/ci/madaros_full_gate.sh` | **11 PASS**, exit 0 |

acceptance: **5 pass, 0 fail**.

One caveat on the PBPK number: 8 passed / 1 failed is not a clean sheet. What #1500 changes is
that the file goes from *not compiling* to *compiling and running*; the remaining sub-test
failure is a separate residual and is not attributable to the conflict resolution, which touched
only diagnostics.

**The local gate set used above was incomplete — see "What landing actually found".** It ran
`madaros_full_gate.sh` and `madaros_current_source_f64_lowering_gate.sh`, both of which pass on
trees that CI then rejects. The gate that decides is `scripts/ci/madaros_fixed_point_gate.sh`, a
*separate script* that happens to run inside the same CI job as the f64 lowering gate. Any local
pre-merge check on this compiler must run it explicitly.

Note on PR CI: the compiler gates on a pull request are **gated behind the `Impact` job** and do
not appear until it passes. Once it does, `Contracts`, `Sounio Lint`, `Native Self-Host (Linux
x86_64)`, `Source-Bootstrap Self-Host (Linux x86_64)` and `Madaros Current-Source f64 Lowering`
all run. So a PR does get a real safety net — but only after a delay, and a reader checking
`gh pr checks` in the first minute will see just `Impact`/`PR Triage` and could wrongly conclude
there is no coverage.



## What landing actually found

### #1490 landed; `main` then went red, and it was not #1490

#1490 merged as `6d84b8d19b`. The next `CI` run on `main` failed — the first `CI` failure on
`main` after several green runs — which is textbook B0 ("CI red on `main` after merge"). It is
not attributable to the merge:

- the failing job is **`Lean Proofs`**, on `formal/lean4/SounioEpistemic.lean:272:56`:
  `Tactic 'split' failed: Could not split an 'if' or 'match' expression in the goal`
- #1490's merge changed exactly three paths: `self-hosted/native/gc.sio` (one constant plus
  comments), `bin/madaros-linux-x86_64`, `artifacts/self-hosted/madaros.gate-receipt`. No Lean
  file, and no path a Lean proof can observe.
- **`formal/lean4/lean-toolchain` pins `leanprover/lean4:stable` — a floating pin.** The CI log
  shows it resolving to `lean-4.33.0`. The proof `bayesianUpdate_zero_evidence` ends
  `simp only [...] at *; split <;> omega`, and that is what broke.

So a proofs gate is pinned to a moving toolchain, and the failure lands on whatever commit
happens to be merged next. Worth its own issue: pin `lean-toolchain` to an exact version.

### #1500 is disqualified by a gate my local verification did not run

#1500 passes `Contracts`, `Sounio Lint`, `Full Test Suite`, `Native Self-Host (Linux x86_64)`,
`Native Self-Host (macOS arm64)` and `Source-Bootstrap Self-Host` — and fails
**`Madaros Current-Source f64 Lowering`**:

```
MADAROS_FIXED_POINT_FAIL: stopped at rung 'none' but this tree is recorded as reaching 'check'.
gen1 cannot typecheck the compiler's own entry point: 1 errors, rc=1
        1 error[E175]
```

`E175` is the visibility-preflight diagnostic that #1500 is *about*. With #1500 applied, the
compiler can no longer typecheck its own entry point — the self-compilation ratchet regresses
from `check` to `none`.

**#1500 was not merged.** Its PBPK result stands (plain `main` cannot compile
`epistemic_pbpk28.sio` at all; with #1500 it compiles and runs 8/9), so the fix is addressing a
real defect — but it cannot land in this shape.

The methodological point is worth keeping: the local acceptance suite ran the PR's own tests and
`madaros_full_gate`, all green, and still missed this. The ratchet gate
(`scripts/ci/madaros_current_source_f64_lowering_gate.sh`) is the one that asks whether the
compiler can still build itself, and it belongs in any local pre-merge check on this compiler.
It has been added to the verification used for the remaining PRs.



### The ratchet rejects three of the four, and only CI could tell

Every one of the four PRs passes `Contracts`, `Sounio Lint`, `Full Test Suite`, `Native
Self-Host` (Linux **and** macOS) and `Source-Bootstrap Self-Host`. What separates them is one
gate — `scripts/ci/madaros_fixed_point_gate.sh`, which asks whether the compiler built from the
tree can still typecheck its own entry point:

| PR | fixed-point ratchet | outcome |
|---|---|---|
| #1501 | **passes** | **merged** (`d9d56436ee`) |
| #1500 | `1 error[E175]` | blocked |
| #1493 | `1 error[E020]`, `1 error[E016]` | blocked |
| #1508 | `2 error[E002]`, `1 error[E137]` | blocked |

For #1493 the likely reading is that the PR does its job: it makes the linear/borrow checker
stricter, and the compiler's own source violates the newly-enforced rule in two places. If so
the remedy is to repair those two sites, not to weaken the checker — but that grows the PR
beyond what a triage should decide.

All three blocked PRs keep their restacked branches: the conflict resolution is pushed (additive
merge of `main`, never a force-push) and the evidence is posted as a comment on each, so the next
attempt starts from the ratchet output rather than rediscovering it.

**Final landing state:** #1490 and #1501 merged; #1500, #1493, #1508 restacked, verified against
their own acceptance tests, and blocked on the ratchet; #1531 held pending a decision on its
108-test delta.

`main` CI at `d9d56436e` (both merges applied) closes exactly on the pre-existing baseline:

| job | result |
|---|---|
| Contracts, Sounio Lint, Website, Impact | success |
| Native Self-Host (Linux x86_64) | success |
| Native Self-Host (macOS arm64) | success |
| Source-Bootstrap Self-Host (Linux x86_64) | success |
| **Madaros Current-Source f64 Lowering** (the ratchet) | **success** |
| Full Test Suite | success |
| `Lean Proofs` | **failure** — the floating `leanprover/lean4:stable` pin, unchanged from before either merge |
| `CI Decision` | failure, solely because `Lean Proofs` did |

Every compiler job is green. The one red is the toolchain drift documented above, and it predates
this work.



## The ratchet failures, resolved

All three blocked PRs were repaired. In every case the ratchet was right and the fix was small
— but none of the three would have been found without running that specific gate.

**#1500 — one missing keyword.** `ir_name_is_knowledge_ctor` was added to `self-hosted/ir/ir.sio`
without `pub`, while its only caller lives in `ir/lower.sio`. Every sibling `ir_name_is_*` helper
in that file is `pub`. One word; the ratchet then reports `MADAROS_FIXED_POINT_OK`, rung `check`
with `rc=0 errors=0`.

**#1493 — a named const in a repeat-literal length.** `BorrowEnv.branch_snap` is declared
`[bool; 4096]` but `borrow_env_new` initialised it `[false; BRANCH_SNAP_PLANE]`, producing
`E020` + `E016`. The const holds exactly 4096, so this is not an arithmetic error — the checker
does not resolve a named const in that position. A comment above the field asserted the opposite
("a const in a repeat-literal LENGTH ... is fine; only the type position is affected"); that
claim is false and has been corrected in place.

**#1508 — three causes behind three errors, and one of them was mine.**
- `2x E002`: `1 << (bit as u8)` will not assign to an `i64`. Every other shift in `self-hosted`
  uses `as i32`; these were the only `as u8` shifts in the tree.
- `E137`: a `let` bound in the tail position of an `else` block reads back as undeclared.
- `E137` (second round): **the conflict resolution recorded above dropped main's
  `lower_opt_type_is_tuple_of_all_f64` and `lower_type_expr_is_tuple_of_all_f64` while keeping
  the call to the first one.** Taking one side of a hunk wholesale deleted two definitions whose
  caller survived elsewhere in the file. Restored — all-f64 tuples keep main's `returns_float=2`
  path, mixed tuples take the PR's `1024+mask`.

That last one is worth stating plainly: the same class of mistake this report warns about in
other people's merges appeared in its own. It was caught by the gate, not by review.

### What this says about the ratchet

Two of the three failures (#1508's `as u8` shifts and its `let`-in-tail-position, #1493's const)
are cases where **new code used constructs the compiler's own checker rejects**. They compile
fine as user programs. They only fail when the compiler must typecheck itself, which is exactly
what `madaros_fixed_point_gate.sh` measures and what nothing else does — `madaros_full_gate.sh`
and `madaros_current_source_f64_lowering_gate.sh` are green on all three broken trees.

Practical consequence: **a PR touching `self-hosted/` is not verified until the fixed-point gate
has run on it.**

## `Lean Proofs`: fixed upstream, and de-fragilised

While this triage was running, `7cd35ba73c` (#1701) took `main` green by pinning
`lean-toolchain` back to `v4.32.2`. That is the correct immediate fix and is left in place.

The underlying proof was also repaired (#1703), so the pin can move forward later without
reopening the failure. Traced goal at the failure point:

```
hv : prior.confidence ≤ bound
⊢ (if prior.confidence + 0 ≤ bound then prior.confidence + 0 else bound) = prior.confidence
```

The `if` is present — `split` simply stopped being able to case on this `ite` between 4.32.2 and
4.33.0. No case analysis is needed: `Nat.add_zero` reduces the guard to exactly `hv`, so
`exact if_pos hv` closes it. Verified under **both** toolchains, and under 4.33.0 a full
`lake build` completes (215 jobs).



## #1531 landed, and the numbers in this report were wrong twice

The harness fix is merged. The suite now evaluates its `//@ expect-stdout:` and
`//@ error-pattern:` annotations for the first time, with pre-existing debt recorded in
`tests/vacuous_expect_baseline.txt` rather than left invisible.

### Correcting this document's own measurements

Two numbers reported earlier here were measured against the wrong subject:

- **"108 tests go red"** — measured by applying a hand-written regex fix and running the suite
  against a from-source **Madaros**.
- **"Fail: 542"** — same mistake, with #1531's own harness.

`.github/workflows/ci.yml` runs the suite with `SOUNIO_TEST_SOUC_BIN=/tmp/souc-stage2` and
`--format junit`. **stage2 and Madaros do not emit the same diagnostics**, so neither number
predicted CI. The true figures, from the CI job itself:

| | |
|---|---|
| Pass | **1545** |
| Fail | **16** (beyond the author's 36-entry baseline) |
| Known failures | 137 |
| Vacuous baseline tolerated | 35 of 36 |
| Total | 2956 |

A shrink of the baseline from 36 to 8 was also attempted and **reverted**: it was derived from
the Madaros run, and CI immediately failed on entries it had removed. The lesson is the one this
report already makes about the ratchet, in a second costume — *which binary produced the
measurement is part of the measurement.*

### What arming the assertions actually exposed

Of the 16 CI failures, all were run again under Madaros to tell a feature gap from a defect:

- **14 `madaros_gum_fo_*` fail under BOTH engines**, and `madaros_gum_fo_knowledge_ops`
  segfaults. Their `MADAROS_GUM_FO_*_PASS` markers had never been evaluated, so an entire
  feature's regression family read green while the feature did not work. They are named
  `madaros_*` yet carry no `//@ requires: madaros` — and adding one would be false, since
  Madaros fails them too. Filed as **#1706**.
- **2 are a real engine divergence**: `ontology_property_weakening` and `gum_correlated` pass
  under Madaros and fail under stage2.

That is the concrete payoff of #444: not a projection, but fourteen tests that asserted a
feature works when it does not, plus one that crashes.



## Following the thread to its end: #1706 defect 1, fixed

Arming the assertions (#1531) exposed 14 `madaros_gum_fo_*` tests asserting a PASS marker
neither engine produces (#1706). Investigating *why* found two independent defects, not one.

**Defect 1 — variance was never tracked per struct field.** It was keyed by local NAME and by
base REG only, so `let a = A { x: k.value }` followed by `variance_of(a.x)` had nowhere to look
and emitted a silent `0.0`. **One hop was enough**; this was not a deep-nesting corner case. The
compiler's own trace (`SOUNIO_LOWER_LIVE_TRACE=1`, printing `fo_xfer_miss name=…`) was what
separated this from the transfer-table story.

Fixed and merged as **#1711** (`9c6a04d865`):

| expression | before | after |
|---|---|---|
| `a.x` (one hop) | `0.000000` | **`0.090000`** |
| `d.c.b.a.x` (four hops) | `0.000000` | **`0.090000`** |
| `d.v0` | `0.000000` | **`4.000000`** |
| `pk2.cl0` (aggregate alias) | `0.000000` | **`0.090000`** |

152 lines, insertions only. No new tables — `Lowerer` is passed by value and already carries
`[i64; 4096]` plus two `[i64; 1024]` inline, so a field slot encodes `(base_reg, field_idx)` into
two disjoint key bands of the existing base-reg table.

**Defect 2 — the FO transfer classifier's coverage — remains open.** `fo_classify_expr_transfer`
accepts only identity, `lit*x`, `a+b`, `a*b`, and a bare forwarded call; `ExprIf` bodies, calls
nested inside a binary, and >2-param bodies fall through unregistered. So `v_call` and `v_meth`
still read zero, all 14 entries stay in `tests/vacuous_expect_baseline.txt`, and #1706 stays open.

### Two method notes worth keeping

**A regression test must be proven to discriminate.** `gum_fo_field_chain_variance.sio` was run
against unmodified `main` (FAIL) *and* the fix (OK) before being committed. In a report whose
subject is assertions that never ran, shipping a test that passes either way would have been
self-defeating.

**`//@ requires: madaros` is truthful here and would be a lie on the other 14.** This fix lives
in `self-hosted/ir/lower.sio`, which only Madaros runs — Madaros passes, stage2 does not, so the
annotation states a real engine boundary. On the `madaros_gum_fo_*` family the same annotation
would be false, because Madaros fails those too.

### The suite is flaky, and that is now visible

`main` fails roughly one run in four, on a *different* test each time
(`global_scalar_init_still_works`, `observe_io_boundary`) — always `missing stdout` or
`missing error`. A re-run of an unchanged commit went from red to 10/10 green, which is what
identified it as flakiness rather than regression.

Those are precisely the assertions #1531 armed. The instability is not new; the ability to see it
is. Whoever picks this up should expect to re-run before believing a single red.


## Ranking

Silent-wrong ranks above crash, because a crash announces itself.

1. **harness blindness** — #444 / #1531. It is why classes 2 and 3 went unnoticed.
2. **blocks self-compile / bootstrap** — #1692, #1693 (both measured), #1686, #1680, #1678, #1649
3. **silent miscompile, no diagnostic** — #1682, #1667, #1574, #1610, #639 (measured); #1586, #1570, #1655, #1499, #1507
4. **blocks a named deliverable** — #1485 (dissertation PBPK), #1577 / #1584 (chemistry)
5. **capacity walls** — #1646, #1658, #1491
6. **checker / ergonomics** — #888 (measured), #1581, #1567, #1568
7. **out of lane** — the 14 above

## Reproducibility

```bash
# build the measuring compiler (~214 s on a SLURM node, ~4 min on the pod)
bash scripts/ci/build_modular_madaros.sh /tmp/madaros-6b2198e314
# prove the harness is not vacuous, then sweep
SWEEP_ROOT=<worktree> SWEEP_BIN=/tmp/madaros-6b2198e314 \
SWEEP_OUT=<out> SWEEP_MANIFEST=<manifest.tsv> bash sweep.sh
```

Note for future sweeps: `./bin/souc foo.sio out.elf` silently routes to **lean_single**, because
`bin/souc` intercepts the raw positional `SRC OUT` form before the engine switch. Probes must
use verbs (`check` / `run` / `compile`) or `--native-v2-compile` directly.
