<!-- docs:meta
topic_id: repo.docs.audit.e008-e137-family-per-case-verdict-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e008-e137-family-per-case-verdict-2026-08-17
-->

# E008 + E137 family — per-case verdict (stdlib-side hygiene, opposite owners from a Madaros defect)

**Date:** 2026-08-17
**Counsel:** minimax-cli2
**Companion prior:** [E175_FAMILY_PER_CASE_VERDICT_2026-08-17.md](E175_FAMILY_PER_CASE_VERDICT_2026-08-17.md)
**Scope:** Two stdlib-source defects that surfaced on `dissertation_pbpk_suite_gate.sh`
after the E175 layer was removed by commit `fb7c61d573`. E008 (return-type mismatch) and
E137 (use of undeclared variable) — both are author-side hygiene gaps, opposite owners
from a Madaros resolver defect.

## Instrument validation

Same prebuilt `./bin/souc` as the E175 audit doc. After the three `pub` additions,
the prebuilt reports E008 and E137 on the previously-MOVED tests — confirming the
visibility gate correctly stopped firing once the source was corrected (E175 family
closed), and the next-layer checks (type, identifier) now report the next set of
real source-code problems. `ulimit -s 524288` per FLEET_CONSTRAINTS.

## Per-case verdict — the dichotomy

For each defect below, the same dichotomy as the E175 doc:

- **Missing-author-fix in stdlib** — the source genuinely has the wrong shape. Fix is
  in the source file. Owner: whoever wrote that file.
- **Madaros resolver/checker defect** — the source is correct, but the checker reports
  it wrong. Fix is in the compiler. Owner: Madaros author.

Every case below is the first kind. None is the second.

### E008 — return-type mismatch (`SSIntervalResult` vs `OralBBBTrace`)

**Site:** `stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:159`

```sio
if nreject > 10000 { return trace }       // <-- trace is OralBBBTrace
                                          // <-- enclosing fn returns SSIntervalResult
```

The enclosing function `ssr_run_one_interval` is declared:

```sio
fn ssr_run_one_interval(...) -> SSIntervalResult with Mut, Div, Panic {
    ...
    var trace = oral_trace_zero()           // trace: OralBBBTrace
    ...
    if nreject > 10000 { return trace }     // E008 fires here
    ...
    SSIntervalResult { trace: trace, sys_st: sys_st, bbb_st: bbb_st, abs_st: abs_st }
}
```

The normal return at L202 constructs an `SSIntervalResult` from four locals
(`trace`, `sys_st`, `bbb_st`, `abs_st`). The bail-out branch at L159 returned only
`trace` — a different type. Same shape as E175: source genuinely has the wrong shape;
fix is one line; owner is the stdlib author.

**Verdict:** **Missing-author-fix in stdlib.** Not a Madaros defect.

**Fix (commit `0a7edf7bf9`):** mirror the normal return at L202:

```sio
if nreject > 10000 {
    return SSIntervalResult { trace: trace, sys_st: sys_st, bbb_st: bbb_st, abs_st: abs_st }
}
```

### E137 — use of undeclared variable (`print_i64`)

**Sites (4 callsites across 2 files):**

- `stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:315` — `print_i64(r.n_doses_run as i64)`
- `stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:317` — `print_i64(r.dose_of_ss as i64)`
- `stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:331` — `print_i64((i + 1) as i64)`
- `stdlib/darwin_pbpk/bbb/bbb_voi.sio:146` — `print_i64(rank_k)`

**What the symbol is:** `print_i64` is **not** a builtin (unlike `print`, `print_f64`,
`println`, which are compiler intrinsics). It is defined as `fn print_i64(x: i64)` at:

- `stdlib/metrology/calibration.sio:322` — `fn` (not `pub fn`; private)
- `stdlib/plot/bar.sio:301` — `fn` (not `pub fn`; private)

Neither is publicly importable. The **correct** public symbol is
`pub fn sci_print_i64(n: i64) with IO, Mut, Panic, Div` at
`stdlib/io/scientific.sio:47`, exported via `stdlib/io/mod.sio:106`. It is already
imported elsewhere in stdlib as:

```sio
use io::scientific::{sci_print_i64}
```

(Confirmed by 4 existing call sites in `stdlib/darwin_pbpk/export/`.)

**Verdict:** **Missing-author-fix in stdlib.** Author assumed `print_i64` was a
builtin like `print_f64`; it isn't — it's `sci_print_i64` and needs the import.
Same shape as E175: source genuinely has the wrong shape; fix is one line per
file; owner is the stdlib author.

**Fix (commit `0a7edf7bf9`):** in both files, add `use io::scientific::{sci_print_i64}`
and rename `print_i64(...)` → `sci_print_i64(...)` at the callsites.

## Why none of these is a Madaros resolver/checker defect

| Site | Checker | What the source says | What the checker reports | Truth? |
|------|---------|------------------------|----------------------------|--------|
| `steady_state_runner.sio:159` | E008 | `return trace` (`OralBBBTrace`) inside fn returning `SSIntervalResult` | "return value does not match function's declared return type" | YES — checker is correct |
| `steady_state_runner.sio:315,317,331` | E137 | `print_i64(...)` | "use of undeclared variable" | YES — checker is correct |
| `bbb_voi.sio:146` | E137 | `print_i64(...)` | "use of undeclared variable" | YES — checker is correct |

The checkers (type checker, identifier resolver) report what the source genuinely
does. After the source is corrected, the checkers go silent — exactly the same
shape as the E175 closure. A Madaros defect would persist after the source fix;
these do not.

## Verification — preflight on the 8 named tests after E008 + E137 fixes

Prebuilt `bin/souc` run against edited stdlib via `SOUNIO_STDLIB_PATH`, `ulimit -s 524288`.
Commit `0a7edf7bf9`.

| # | Test | After E175 closure | After E008 + E137 closure | Verdict |
|---|------|---------------------|------------------------------|---------|
| 1 | `dissertation_oral_pd_demo.sio` | GREEN | **GREEN** | unchanged |
| 2 | `dissertation_steady_state_demo.sio` | MOVED-NEXT (E008 + 3× E137) | **GREEN** | E008 + 3× E137 closed |
| 3 | `dissertation_steady_state_fullvd_demo.sio` | MOVED-NEXT (E008 + 3× E137) | **GREEN** | E008 + 3× E137 closed |
| 4 | `dissertation_scenario_gate_demo.sio` | MOVED-NEXT (1× E137 via bbb_voi) | **MOVED-NEXT** — now to Madaros `handles full` | E137 closed; new diagnostic is **Madaros-side**, different owner |
| 5 | `haloperidol_pgx_gate.sio` | GREEN | **GREEN** | unchanged |
| 6 | `dissertation_pgx_compile_gate_demo.sio` | GREEN | **GREEN** | unchanged |
| 7 | `steady_state_runner.sio` (companion, `souc check`) | FAIL (E008 + 3× E137) | **CHECK OK** | E008 + 3× E137 closed |
| 8 | `bbb_voi.sio` (companion, `souc check`) | FAIL (1× E137) | **CHECK OK** | E137 closed |

**Tally:** 6 GREEN, 1 MOVED-NEXT, 0 still-E008/E137.

## Test 4 — what `madaros: handles full` means

After the E137 layer on test 4 is removed, the source compiles past type checking
and reaches Madaros's native code generation. The codegen then fails with:

```
madaros: handles full
```

(reported at rc=182, after a successful compile-and-link phase — the ELF is written,
but Madaros refuses to execute it because the handle table is exhausted for this
particular IR shape). This is **not** a missing-author-fix in stdlib. It is a
**Madaros-internal code-generation capacity limit** — different class, different
owner. The stdlib hygiene pattern (E175, E008, E137) is fully closed; the one
remaining red test is a Madaros-side issue that would require a Madaros-side
fix (or a source-side reduction in handle usage) to clear.

**Concretely:** the pre-fix baseline (with the E137 unfixed) failed at parse/check
time, never reaching codegen. The post-fix state passes parse/check but trips
codegen. The two failures are not the same layer.

## Closing the E008 + E137 family — what it actually takes

Five one-line edits in two files (commit `0a7edf7bf9`):

```
stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:26
    +use io::scientific::{sci_print_i64}

stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:159
    -if nreject > 10000 { return trace }
    +if nreject > 10000 {
    +    return SSIntervalResult { trace: trace, sys_st: sys_st, bbb_st: bbb_st, abs_st: abs_st }
    +}

stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:315,317,331
    -print_i64(...)  →  sci_print_i64(...)  (3 callsites)

stdlib/darwin_pbpk/bbb/bbb_voi.sio:37
    +use io::scientific::{sci_print_i64}

stdlib/darwin_pbpk/bbb/bbb_voi.sio:146
    -print_i64(rank_k)  →  sci_print_i64(rank_k)
```

Combined diff: 9 insertions, 5 deletions across 2 files. Same shape as the E175
closure: atomic, owner-localised, no compiler involvement.

## What this changes for the defence

E008 and E137 are **not** Madaros defects and so are **not** an argument that
the compiler is unfit for dissertation work. They are missing-author-fix items
in stdlib — one struct-literal typo (wrong return type) and one wrong-symbol
assumption (`print_i64` is private; the public surface is `sci_print_i64`).
Both checkers (type, identifier) are doing their job; they emit only when the
source genuinely has the wrong shape, and go silent once the source is correct.

The one remaining red test on the dissertation suite (test 4,
`dissertation_scenario_gate_demo.sio`) is a Madaros `handles full` codegen
capacity issue. It is a separate work item, a separate owner, and would require
either a Madaros-side capacity increase or a source-side reduction in handle
usage (e.g. by splitting the import graph into smaller modules). It is not
within scope of this PR's hygiene fixes.
