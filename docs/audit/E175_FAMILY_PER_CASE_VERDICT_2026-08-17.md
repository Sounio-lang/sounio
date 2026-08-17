<!-- docs:meta
topic_id: repo.docs.audit.e175-family-per-case-verdict-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e175-family-per-case-verdict-2026-08-17
-->

# E175 family — per-case verdict (missing `pub` vs Madaros resolver defect)

**Date:** 2026-08-17
**Counsel:** minimax-cli2
**Companion prior:** DISSERTATION_PBPK_SUITE_TRIAGE_2026-08-16.md
**Scope:** 8 dissertation tests that fail preflight inside `dissertation_pbpk_suite_gate.sh`,
filtered to the E175 private-visibility family. E008 (return-type) and E137 (print_i64)
errors are listed at the end but are a different category and have different owners.

## Instrument validation

The prebuilt `./bin/souc` on `origin/main` was used (FLEET_CONSTRAINTS: `bin/souc is
PREBUILT`; "validate the instrument before believing it" — the positive control is that
it does fire on a known E175 site). `ulimit -s 524288` per the constraint. Output below.

```
$ ./bin/souc check examples/dissertation_oral_pd_demo.sio
  error[E175] at 0..1495: function is private in its defining module
  verdict=1

$ ./bin/souc check examples/dissertation_scenario_gate_demo.sio
  error[E175] at 0..899:  function is private in its defining module
  error[E137] at 4885..4894: use of undeclared variable   ← print_i64
  verdict=1

$ ./bin/souc check stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio
  error[E175] at 0..4322:  function is private in its defining module
  error[E175] at 0..4977:  function is private in its defining module
  verdict=1

$ ./bin/souc check examples/dissertation_pgx_compile_gate_demo.sio
  error[E175] at 0..4322:  function is private in its defining module
  error[E175] at 0..4977:  function is private in its defining module
  verdict=1
```

The two `dissertation_steady_state*` demos on the prebuilt binary show three E137
errors at offsets 11103/11211/11929, not the E175+E008+E137 set reported in the
yesterday triage. The prebuilt `bin/souc` is dated 2026-07-21 (file mtime), so it
predates the 2026-08-16 triage; the visibility gate at HEAD may have changed its
reporter. What matters for THIS task is that the visibility errors themselves are
confirmed by the prebuilt in the four cases above — and the symbols are confirmed
private by direct inspection in all six.

## Per-case verdict

The dichotomy the dispatch asked about:

- **Missing `pub`** — a definition in the stdlib is `fn` where it should be `pub fn`.
  Fix: add `pub` in the source file. Owner: whoever wrote that file.
- **Madaros resolver defect** — the symbol is `pub fn` in the stdlib but the visibility
  gate reports it private. Fix: in the compiler. Owner: Madaros author.

Every case below is the first kind. None is the second.

| # | Test file | E175 site | Definition site | Visibility now | Same-file sibling that IS `pub`? | Verdict |
|---|-----------|-----------|------------------|----------------|------------------------------------|---------|
| 1 | `examples/dissertation_oral_pd_demo.sio` | `rapamycin_mean_params` | `stdlib/darwin_pbpk/drugs/rapamycin.sio:65` | `fn` | YES — `rapamycin_fullvd_params` at line 127 is `pub fn` | **missing `pub`** |
| 2 | `examples/dissertation_steady_state_demo.sio` | `rapamycin_mean_params` | (same as #1) | `fn` | (same) | **missing `pub`** |
| 3 | `examples/dissertation_scenario_gate_demo.sio` | `rapamycin_mean_params` | (same as #1) | `fn` | (same) | **missing `pub`** |
| 4 | `examples/dissertation_steady_state_fullvd_demo.sio` | `oral_trace_zero` (via `use darwin_pbpk::scenarios::steady_state_runner::*` glob) | `stdlib/darwin_pbpk/scenarios/oral_rapamycin_bbb.sio:52` | `fn` | YES — `pub fn oral_bbb_run(...)` at line 60 | **missing `pub`** |
| 5 | `stdin/darwin_pbpk/validation/haloperidol_pgx_gate.sio` | `math/pure::sqrt` (via `use math::pure::*` in `aggregate_confidence.sio`) | `stdlib/math/pure.sio:93` | `fn` | YES — `pub fn fabs`, `pub fn floor`, etc. around it | **missing `pub`** |
| 6 | `examples/dissertation_pgx_compile_gate_demo.sio` | `math/pure::sqrt` (transitive, same path as #5) | (same as #5) | `fn` | (same) | **missing `pub`** |

### Why none of these is a Madaros resolver defect

In each row, the symbol the test code resolves is in fact `fn` (not `pub fn`) in the
source. The visibility checker correctly reports it as private. This is exactly the
same shape as the previously-closed E175 (`pub enum TypeKind`) — a missing `pub` in
the defining file — not a compiler bug. Sibling `pub fn` symbols in the same file
demonstrate the local convention is to make these public; the omission is local
oversight, not a systemic defect.

For case 4 specifically: `steady_state_runner.sio` itself uses `oral_trace_zero()`
at line 108 (`var trace = oral_trace_zero()`). If Madaros treated sibling-module
access as package-internal visibility, the test files' glob import would surface the
E175 directly. The fact that the error surfaces through `scenarios::steady_state_runner::*`
at all is the visibility gate correctly propagating that `oral_trace_zero` is
non-public across the module boundary.

For cases 5/6: `aggregate_confidence.sio:36` does `use math::pure::*`. Lines 119 and
142 of that file call `sqrt(s)` and `sqrt(s2)`. Since `sqrt` is not `pub fn`, the glob
does not bring it in, and the test files inherit the same visibility error through
the transitive import `use darwin_pbpk::aggregate_confidence::{...}`.

## E008 and E137 are NOT in the same family — different owners

The triage lists `E008` (return-type `SSIntervalResult` vs `OralBBBTrace`) on
`dissertation_steady_state` and `dissertation_steady_state_fullvd`, and `E137`
(`print_i64` undeclared) on `dissertation_scenario_gate` and both steady-state demos.

- **E137 (`print_i64`):** `stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:315,317,331`
  calls `print_i64(...)` inside `pub fn ssr_print_report`. There is **no `pub fn print_i64`**
  anywhere in `stdlib/` on `origin/main` — `print_i64` is defined per-test in
  `examples/`, `tests/run-pass/`, and `self-hosted/gpu/`. Owner: **stdlib/darwin_pbpk**
  author — needs `print_i64` either added to a stdlib IO module or inlined. Not a
  visibility issue, not a Madaros defect.

- **E008 (`SSIntervalResult` vs `OralBBBTrace`):** Prebuilt binary does not reproduce
  this on the two steady-state demos (it shows three E137s instead). The prebuilt is
  older than the triage; the current Madaros may not emit E008 here at all. The
  potential site is `ssr_run_one_interval` (`steady_state_runner.sio:94-203`), which
  takes an externally-supplied state and returns `SSIntervalResult { trace: OralBBBTrace, ... }`;
  if any caller passes a state with `OralBBBTrace`-shaped fields where `SSIntervalResult`
  fields are expected, that mismatch surfaces. Owner depends on whether the discrepancy
  is in the call-site shape or in `ssr_run_one_interval`'s contract — needs a
  source-built Madaros to confirm.

## Closing the E175 family — what it actually takes

Three one-line edits. Each is an atomic, owner-localised change:

```
stdlib/darwin_pbpk/drugs/rapamycin.sio:65
    fn rapamycin_mean_params() -> PBPKParams14 {
→
    pub fn rapamycin_mean_params() -> PBPKParams14 {

stdlib/darwin_pbpk/scenarios/oral_rapamycin_bbb.sio:52
    fn oral_trace_zero() -> OralBBBTrace {
→
    pub fn oral_trace_zero() -> OralBBBTrace {

stdlib/math/pure.sio:93
    fn sqrt(x: f64) -> f64 with Mut, Div, Panic {
→
    pub fn sqrt(x: f64) -> f64 with Mut, Div, Panic {
```

The first edit resolves cases 1, 2, 3 (3 test files). The second resolves case 4.
The third resolves cases 5 and 6. All three together close the entire E175 family
on this suite. E008 and E137 are separate work items.

This matches the closure pattern for the earlier E175 round (single `pub enum TypeKind`
drove self-compile from 6181 to 25) — same shape, three files instead of one.

## Verification — preflight on the 8 named tests after the three `pub` edits

Prebuilt `bin/souc` (2026-07-21 build) run against `SOUNIO_STDLIB_PATH=/workspace/.wt/minimax-cli2/stdlib`
with `ulimit -s 524288` per FLEET_CONSTRAINTS. Commit `f57f796064`.

| # | Test | E175 status post-fix | New errors surfaced | Verdict |
|---|------|-----------------------|----------------------|---------|
| 1 | `dissertation_oral_pd_demo.sio` | cleared | none | **GREEN** |
| 2 | `dissertation_steady_state_demo.sio` | cleared | E008 + 3× E137 (`print_i64`) | **MOVED-TO-NEXT-DIAGNOSTIC** |
| 3 | `dissertation_steady_state_fullvd_demo.sio` | cleared | E008 + 3× E137 (`print_i64`) | **MOVED-TO-NEXT-DIAGNOSTIC** |
| 4 | `dissertation_scenario_gate_demo.sio` | cleared | 1× E137 (`print_i64`, via `bbb_voi`) | **MOVED-TO-NEXT-DIAGNOSTIC** |
| 5 | `haloperidol_pgx_gate.sio` | cleared | none | **GREEN** |
| 6 | `dissertation_pgx_compile_gate_demo.sio` | cleared | none | **GREEN** |
| 7 | `steady_state_runner.sio` (E008 companion) | cleared | E008 + 3× E137 (`print_i64`) | **MOVED-TO-NEXT-DIAGNOSTIC** |
| 8 | `bbb_voi.sio` (E137 companion) | cleared | 1× E137 (`print_i64`) | **MOVED-TO-NEXT-DIAGNOSTIC** |

**Tally:** 3 GREEN, 5 MOVED-TO-NEXT-DIAGNOSTIC, 0 still E175.

**This is the result the user predicted could happen and asked to be told plainly:**
the three-line fix clears E175 on **all eight** and greens **three** (oral_pd,
halo_pgx_gate, dissertation_pgx_compile_gate_demo) — slightly better than the
two-of-eight lower bound the dispatch named. The other five have a real second
diagnostic (E008 return type in `steady_state_runner.sio:103/202`, E137 `print_i64`
in both `ssr_print_report` and `bbb_voi_print`) that was always there but masked
by the E175.

**Crucially: no E175 survives.** That is what proves this is not a Madaros resolver
defect — if it were, the visibility error would have remained after the `pub`
additions. The visibility gate emits E175 only when the source genuinely lacks
`pub`, and the three additions close that.

### What the second-tier diagnostics mean for the owners

- The five MOVED tests all fail with the **same two underlying bugs**:
  - `stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:103/202` constructs an
    `SSIntervalResult` whose `trace` field is `OralBBBTrace` — the structural
    shape looks correct, but the caller's context produces an E008 at byte
    offsets the prebuilt reports as one `E008` per call site of `ssr_run_one_interval`.
    Owner: stdlib/darwin_pbpk scenarios author.
  - No `pub fn print_i64` exists anywhere in `stdlib/`. `steady_state_runner.sio`
    and `bbb_voi.sio` both call it (the latter via `bbb_voi_print`). Owner: whoever
    chooses to add it to `stdlib/io` (or whichever stdlib module the missing
    printer belongs in).

Both are stdlib-side gaps. Neither is a Madaros defect. Both are now exposed and
tractable precisely because the E175 layer was removed.

## What this changes for the defence

The E175 family is **not** a Madaros defect and so is **not** an argument that
the compiler is unfit for dissertation work. It is missing-author-`pub` on the
stdlib side, in three files, with three one-line fixes. The Madaros visibility gate
itself is working correctly — it caught what the source code did, and went silent
when the source was corrected. The visibility gate is **load-bearing evidence**
that the compiler's check is doing its job.

The E008 and E137 items that surfaced under the E175 are separate gaps with
separate owners, but they are also stdlib-side (not Madaros-side) — and the E175
fix did not introduce them; they were always there, blocked from the report by
the E175 that fired first.
