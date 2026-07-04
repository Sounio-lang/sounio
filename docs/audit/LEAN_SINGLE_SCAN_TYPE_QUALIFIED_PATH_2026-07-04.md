<!-- docs:meta
topic_id: repo.docs.audit.lean-single-scan-type-qualified-path-2026-07-04
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-scan-type-qualified-path-2026-07-04
-->

# lean_single forensic dispatch — `scan_type` drops trailing `::segment`s of a qualified type path, corrupting the enclosing arity scan

Date: 2026-07-04
Branch: `main` (post-PR #624, Bug A)
Class: **checker gap** (a qualified type name in a parameter/field annotation is
silently truncated to its first path segment, and the unconsumed remainder is
misread as extra parameters by whatever scan called it) — root-causes and
closes issue #601's "Bug B"
Status: root-caused, fixed, verified (full test suite 1311 pass / 0 fail / 127
known failures / 689 skip — back to the pre-existing baseline, zero deltas)

## Symptom

A function parameter (or any type position) annotated with a `::`-qualified
type path produces a false `arity mismatch` at every call site, even when the
call passes the exact correct number of arguments:

```sio
// pkg/addr.sio
module pkg::addr
pub struct Thing { x: i64 }
pub fn thing_new(x: i64) -> Thing { Thing { x: x } }

// pkg/sub/wrapper.sio
module pkg::sub::wrapper
pub fn take_thing(target: &pkg::addr::Thing) -> i32 { 0 }

// main.sio
use pkg::addr::*
use pkg::sub::wrapper::*
fn main() -> i32 {
    let t = thing_new(5)
    take_thing(&t)   // error: arity mismatch
}
```

Matches issue #601's original Bug B repro exactly. Real-world instance:
`stdlib/database/pure/engine.sio`'s `engine_create_table(db: &!database::pure::types::InMemoryDB, name: string)`
— every call to any function in `engine.sio` failed with "arity mismatch"
regardless of argument count, forcing `tests/stdlib/database/test_database_core.sio`
into a `//@ known-failure` marker as of PR #624 (the Bug A fix), since fixing
Bug A let these calls reach arity-checking for the first time and exposed
this independent, pre-existing defect.

## Root cause

`self-hosted/compiler/lean_single.sio`'s `scan_type()` (line 5094, shared by
both the x86-64 and aarch64 backends — it does no codegen, only token
bookkeeping) resolves a plain-identifier type name by reading exactly **one**
token as the type name:

```sio
if TK[p as usize] == 3 {
    let ns = TS[p as usize]
    let ne = TE[p as usize]
    ...  // ~300 lines of primitive-name / Option<T> / Box<T> / Knowledge<T> / struct+enum lookup
    p = p + 1
    if TK[p as usize] == 23 { ... }  // generic <T> args
    ...
}
SCAN_TY_NEXT = p
```

For `pkg::addr::Thing`, `ns`/`ne` capture only `"pkg"`. None of the primitive
names match; `st_find(gl_name_hash("pkg"))` and `en_find(...)` both fail
(there is no struct or enum literally named `pkg`), so the type resolves to
`SCAN_TY = 0` (unknown) with `SCAN_TY_HASH = hash("pkg")`. `p` advances past
`pkg` only, and since the next token is `::` (not `<`), the generic-args
check is skipped. `SCAN_TY_NEXT` is set to `p`, pointing at the *first* `::`
— the rest of the path (`::addr::Thing`) is left completely unconsumed in the
token stream.

The caller that exposes this — the function-signature arity pre-pass
(`self-hosted/compiler/lean_single.sio` ~line 26368) — advances its own
scan cursor `q = SCAN_TY_NEXT` after each parameter's type and then resumes
its `while TK[q] != ')'` loop looking for the next parameter. With
`SCAN_TY_NEXT` stranded mid-path, the loop walks the leftover tokens one at a
time: `::` is not an identifier (falls through, `q += 1`), then `addr` **is**
an identifier — the loop's `if TK[q] == 3 { FN_ARITY += 1; ... }` branch fires,
counting it as a *second* parameter. The same happens again for `Thing`. A
function with one real qualified-type parameter ends up with `FN_ARITY == 3`
(the real parameter plus two phantom ones peeled off its own type
annotation), so every correctly-arity'd call site fails arity-checking.

This is structurally the same class of defect as Bug A (PR #624) — a `::`
chain silently truncated to its first segment, corrupting whatever comes
after — but in type-annotation position (`scan_type`) rather than call
position (`compile_primary`).

## Fix

Flatten a `::`-separated identifier chain down to its terminal segment
*before* any of `scan_type`'s existing type-name resolution runs, exactly
mirroring the Bug A fix's approach:

```sio
if TK[p as usize] == 3 {
    while TK[p as usize] == 3 && TK[(p + 1) as usize] == 51 && TK[(p + 2) as usize] == 3 {
        p = p + 2  // skip segment + ::
    }
    let ns = TS[p as usize]
    let ne = TE[p as usize]
    ...  // unchanged below — now resolves "Thing", not "pkg"
```

The loop condition requires the *next* token after the `::` to also be an
identifier, so it only fires on a genuine qualified chain and naturally stops
at the terminal segment (nothing follows `Thing` with another `::ident`
pair). All of the existing struct/enum/primitive/generic resolution below
runs unchanged on the now-correct terminal name — `st_find(hash("Thing"))`
succeeds, `SCAN_TY`/`SCAN_TY_HASH` are set correctly, and `SCAN_TY_NEXT`
lands past the whole qualified path rather than mid-chain. Single shared
function; no separate aarch64 twin exists for `scan_type` (unlike
`compile_primary`), so one edit covers both backends.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1311  Fail: 0  Known failures: 127  Skip: 689  Total: 2127
```

Exactly the pre-existing baseline (the same numbers as before PR #624's Bug A
fix) — zero regressions, and `tests/stdlib/database/test_database_core.sio`'s
`//@ known-failure` marker (added in PR #624 to document this exact bug) is
removed since it now genuinely passes.

Also confirmed directly with instrumented `println` output against the real
`stdlib/database/pure/engine.sio` API (not just "compiles without error"):

```
engine_create_table (1st)  → 1   (expected: created)
engine_create_table (2nd)  → 0   (expected: already exists)
engine_insert_row x2       → 1, 1
engine_table_row_count     → 2
engine_get_cell            → 42
engine_drop_table          → 1
engine_table_row_count     → -1  (expected: table gone)
```

All eight values match `test_database_core.sio`'s own hand-written
assertions exactly.

## Cross-references

- `docs/audit/LEAN_SINGLE_MULTISEGMENT_QUALIFIED_CALL_2026-07-04.md` — Bug A
  (PR #624), the call-site sibling of this fix; this dispatch's "Symptom"
  section documents how fixing Bug A unmasked Bug B.
- GitHub issue #601 — tracks Bug B (closed by this fix). Bugs C–G remain
  open.
