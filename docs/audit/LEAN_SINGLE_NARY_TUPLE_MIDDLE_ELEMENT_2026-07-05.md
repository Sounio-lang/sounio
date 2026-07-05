<!-- docs:meta
topic_id: repo.docs.audit.lean-single-nary-tuple-middle-element-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-nary-tuple-middle-element-2026-07-05
-->

# lean_single forensic dispatch — N-ary (3+) tuples silently drop/mistype middle elements

Date: 2026-07-05
Branch: `main` (post-PR #629, Bug F)
Class: **checker type-tracking bug + a co-located codegen data-loss bug**, both
rooted in a "tuples have exactly 2 elements" assumption baked into the tuple
type-hash encoding — root-causes and fixes issue #601's "Bug G", and is
substantially more severe than the issue's own description
Status: root-caused, fixed, verified (full test suite 1314 pass / 0 fail /
124 known failures / 689 skip / 2127 total — exact match to the current
baseline, zero regressions)

## Summary

Issue #601 described Bug G as "a struct type in the LAST position of a
3-tuple corrupts the MIDDLE element's inferred type" — a checker-only,
type-labeling bug. Investigation confirmed that framing but found the actual
defect is deeper and worse: **every** tuple literal with 3 or more elements
silently produced a wrong buffer size and dropped every "middle" element's
value at construction time, regardless of whether the elements' types
differ. The checker error reported in issue #601 only occurs when the
mislabeling happens to change the checked *type* (e.g. last element is a
struct); for same-typed elements (`(f64, f64, f64)`) the identical mechanism
silently duplicated the *last* element's value into every middle binding,
with no diagnostic at all.

## Reproduction

Issue #601's own repro (checker-visible symptom):

```sio
struct Foo { id: i64 }
fn f(val: f64, u: f64) -> (f64, f64, Foo) { (val, u, Foo { id: 1 }) }
fn main() -> bool {
    let (v, u, iri) = f(0.18, 0.02)
    u > 0.0   // error: ordered comparison requires matching numeric operands
}
```

The deeper, silent-corruption variant (compiles and runs with **no error at
all**, pre-fix):

```sio
fn triple() -> (f64, f64, f64) { (1.0, 2.0, 3.0) }
fn main() -> i64 {
    let (a, b, c) = triple()
    println(a); println(b); println(c)
    0
}
// pre-fix output: a=1.000000 b=3.000000 c=3.000000  (b should be 2.0 — its
// real value was never stored anywhere; c's value got read twice)
```

## Root cause: two co-located "exactly 2 elements" assumptions

The tuple type-hash format (`tcount*100M + first_f64*10M + last_f64*1M +
first_nslots*1000 + total_nslots`, used throughout `self-hosted/compiler/
lean_single.sio`) and its `TUP_CACHE` backing store were designed to track
only the FIRST and LAST element of a tuple — there was no storage for
anything in between. Two call sites built directly on that assumption:

**1. `scan_type()`'s tuple-type branch** (the pass that scans a `(T, U, ...)`
type — e.g. a function's declared return type): computed
`total_nslots = first_nslots + last_nslots`, silently omitting every middle
element's contribution to the total. For `(f64, f64, Foo)` this produced
`total_nslots = 2` (1 + 1), when the true total is 3 — undercounting the
SRET buffer size for any 3+-element tuple return type.

**2. Tuple-literal construction** (`compile_or()`'s parenthesized-expression
branch, x86 only — see aarch64 scope note below): compiled each element in
source order, but only ever preserved the FIRST element's value (saved to a
temp slot before the next `compile_or()` could clobber `rax`) and the value
left in `rax` after the loop ends (implicitly "whichever element is
textually last"). For a 3+-element literal, every element strictly between
the first and the last was compiled — its value briefly sat in `rax` — and
was then **silently overwritten by the next element's `compile_or()` before
ever being stored anywhere**. The buffer was then allocated at
`first_nslots + last_nslots` (matching defect #1) and only the first and
last elements were copied in; the middle element's slot in memory was never
written by this code at all.

`tuple_destructure_from_ptr_x86()` (the `let (a, b, c) = expr` binder) then
reads element 0 from the "first" bucket and **every other index from the
"last" bucket, at the same byte offset for every index ≥ 1**  — so for a
3-tuple, both index 1 and index 2 are read as "the last element type," at
the identical address. That is what produced `b=3.0, c=3.0` in the
homogeneous case above: there was never a real offset for a genuine
"index 2"; both index 1 and index 2 aliased onto whatever the last element's
slot held. The `.1` field-access path (a separate, smaller code path) had
the identical first/last-only aliasing for its own type lookup.

## Fix: generalize to N-ary tuples, don't special-case N=3

Per repo principle 10 ("edge of novelty" / no band-aids) and this session's
own Bug F lesson (a narrow special case is a trap when the underlying
mechanism is what's wrong), the fix generalizes the tuple-type cache to
carry **every** element, not just first/last, and updates every consumer to
use it:

- `TUP_CACHE` gained a per-row array (`TUP_CACHE_ELEM_TY`/`_HASH`, capped at
  16 elements/row — the widest tuple measured across the entire codebase via
  `-> (T, T, ...)` census is 8, see Verification) plus `TUP_CACHE_TCOUNT`.
  Populated via a scratch channel (`TUP_SCRATCH_ELEM_TY/HASH/COUNT`) so the
  existing `tup_cache_register(hash, first_ty, first_hash, last_ty,
  last_hash)` call sites that only ever dealt with 2 elements are
  byte-identical in behavior (scratch left at `COUNT=0`).
- `scan_type()`'s tuple branch now finds every top-level element's token
  span first (pure token scan), then scans each one's type, summing **all**
  elements' slot counts into `total_nslots` and populating the scratch
  channel before registering — first/last are still recorded for the
  existing 2-element decoders, kept for backward compatibility.
- Tuple-literal construction now saves **every** element's `rax` (a scalar
  value or an aggregate pointer — either way one register-width word) to its
  own temp slot immediately after compiling it, so no element is clobbered
  by compiling the next one. Once all N are known, one buffer sized to the
  true sum of every element's slots is allocated and each element is copied
  to its correct cumulative offset (`tup_base + total_slots - 1 -
  prefix_sum(i)`, the same addressing scheme the original 2-element code
  used, generalized to N terms). Bounded at 16 elements; a literal with more
  produces a hard compiler error rather than silently truncating.
- `tuple_destructure_from_ptr_x86()` and the `.1` field-access path now
  prefer the full per-element cache row (when `TUP_CACHE_TCOUNT > 2`) over
  the first/last split, addressing every element by its own type and a
  running byte-offset accumulator.
- `tup_total_slots_true()` (used for `.0` copy sizing and return-copy sizing
  elsewhere) sums the full per-element array when available instead of
  first+last only.
- Deleted a since-redundant (and, after the `scan_type` fix, actively wrong)
  block in the function-signature scanner that used to re-derive a
  2-element-only return-type hash from scratch whenever `scan_type`'s own
  hash looked under-populated — `scan_type` now always computes the correct
  N-ary hash directly, so this rescan is unneeded and would have overwritten
  the correct value with the old truncated one.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127
```

Exact match to the current baseline — zero regressions from a fix that
touches a mechanism used by every tuple in the codebase.

Directly confirmed by runtime value (not just "no compile error"):
- Issue #601's original repro: compiles clean, `u > 0.0` → `true`, `v=0.18`,
  `u=0.02` (both correct, independently printed).
- The homogeneous silent-corruption case: `(a, b, c) = (1.0, 2.0, 3.0)` now
  gives `a=1.0, b=2.0, c=3.0` (was `a=1.0, b=3.0, c=3.0` pre-fix — `b`'s true
  value was never stored anywhere and `c`'s value was read twice).
- `.1` field access on a 3-tuple with a struct last element: `t.0=1.0`,
  `t.1=2.0` (was mistyped as the struct's type pre-fix).
- A 4-element tuple destructure: `(10.0, 20.0, 30.0, 40.0)` → all four values
  distinct and correct (not previously reachable via any passing test).
- Widest tuple measured across the repo via `grep -rnE -- '-> *\([^()]+,
  [^()]+,[^()]+\)'`: 8 elements (`stdlib/epistemic/pce.sio`'s
  `gauss_hermite_nodes`/`gauss_legendre_nodes`, `stdlib/compiler/ast/
  sedenion_encoding.sio`/`sedenion_ops.sio`) — well inside the 16-element cap.

## Discovered but explicitly out of scope

- **aarch64**: `compile_primary_a64()`'s parenthesized-expression branch has
  no tuple-literal construction support at all (compiles only the first
  element and drops any `, expr2, ...` that follows) — a separate,
  pre-existing, and considerably larger a64 gap, not something this fix
  extends or regresses. `tuple_destructure_from_ptr_a64()` (the destructure
  side) is left unchanged since there is no way to construct an a64 3+-tuple
  for it to correctly consume yet.
- **`.2`+ field access**: `expr.2` and beyond remain a hard compiler error
  (`tup_idx > 1` → "tuple index out of bounds"), unchanged. This is a
  separate, narrower, pre-existing restriction that issue #601 did not
  report and this fix does not lift.
- **Tagged/Option-element tuple `match` patterns**: the `is_tuple_pattern`
  branch of `match` compilation has the identical first/last-only addressing
  for tuples of `Option`-tagged elements specifically — a different,
  narrower feature than the `let`/`.N` paths fixed here, not exercised by
  issue #601's repro, not touched.
- **1-slot struct elements inside a tuple get mistyped as a plain scalar on
  destructure/field-access** (e.g. `(f64, Foo)` where `Foo` has one `i64`
  field binds the `Foo` position as `i64`, so a later `.id` access reads
  garbage). Confirmed via a controlled test that this reproduces
  *identically* on the unmodified, pre-this-dispatch compiler for a plain
  2-element tuple — genuinely independent of Bug G and this fix. Not
  issue-tracked yet.
- **Tuples with more than 16 elements** now produce a hard compiler error at
  the literal-construction site rather than a prior undefined/silently-wrong
  behavior. No tuple in the codebase currently exceeds 8 elements (see
  Verification), so this is a defensive bound, not an active restriction.

## Cross-references

- GitHub issue #601 — Bug G closed by this fix (all of A–H now resolved:
  A/B/C/E/G fixed at source, D verified-already-fixed, F root-caused with
  fix explicitly rejected — see `docs/audit/
  LEAN_SINGLE_BUGF_ROOTCAUSED_NOT_FIXED_2026-07-05.md`).
- `docs/audit/LEAN_SINGLE_BUGF_ROOTCAUSED_NOT_FIXED_2026-07-05.md` — the
  immediately preceding dispatch in this campaign; its "generalize, don't
  special-case" lesson from a rejected fix directly informed this fix's
  design.
