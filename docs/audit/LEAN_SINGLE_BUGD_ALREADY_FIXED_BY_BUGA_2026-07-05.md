<!-- docs:meta
topic_id: repo.docs.audit.lean-single-bugd-already-fixed-by-buga-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-bugd-already-fixed-by-buga-2026-07-05
-->

# lean_single forensic verification — issue #601 Bug D no longer reproduces (fixed as a side effect of the Bug A fix, PR #624)

Date: 2026-07-05
Branch: `main` (post-PR #626, Bug C)
Class: **verification, no source change** — closes issue #601's "Bug D"
without a code fix
Status: verified fixed, zero regressions, no compiler change in this dispatch

## Summary

Issue #601's "Bug D" ("qualified call nested as an argument miscounts the
OUTER call's arity") was tested against a freshly-rebuilt `lean_single` on
top of `main` @ `22a0263f2` (post-Bug-A/#624, Bug B/#625, Bug C/#626) and
**does not reproduce**. No source change was needed or made in this
dispatch — this document records the verification so the finding is not
re-investigated from scratch.

## Original repro (issue #601)

```sio
// pkg/sub/inner.sio
module pkg::sub::inner
pub fn two_args(a: i64, b: i64) -> i64 { a + b }

// main.sio
fn outer(x: i64, y: i64) -> bool { x == y }
fn main() -> i64 {
    let r1 = outer(pkg::sub::inner::two_args(1, 2), 3)   // originally: error: arity mismatch (on outer!)
    let r2 = pkg::sub::inner::two_args(1, 2)              // originally: fine, standalone
    0
}
```

Tested against current `main` (structurally identical repro, instrumented
with `println` to check runtime values, not just absence of a compile
error):

```
compile: fns=24
1   // outer(two_args(1,2), 3) == outer(3,3) == true
3   // two_args(1,2) == 3
```

No error, and both values are correct. Also verified with additional
variations not in the original repro — multiple qualified calls as
arguments to a 3-parameter outer call, mixed arities, and a qualified call's
result used with subsequent field access — all compile and run correctly:

```sio
let r1 = outer3(bugDpkg2::two::add(1, 2), bugDpkg2::two::add(3, 4), 5)       // 15, correct
let r2 = outer3(bugDpkg2::two::triple(1, 2, 3), 10, bugDpkg2::two::add(1, 1)) // 18, correct
println(bugDpkg2::vec::make_vertex(7, 9).x)                                   // 7, correct
```

## Why this is fixed: same root cause as Bug A

Bug D's mechanism, as originally hypothesised in issue #601 ("likely the
argument-list parser flattens a nested qualified call's own arguments into
the outer call's argument count") and confirmed by this verification, was a
*consequence* of Bug A's defect (`docs/audit/LEAN_SINGLE_MULTISEGMENT_QUALIFIED_CALL_2026-07-04.md`,
PR #624), not an independent bug:

Before PR #624, a 2+-`::` qualified call resolved only its *first* `::segment`
(via the enum-variant catch-all), emitted a stub `0`, and abandoned parsing —
leaving the call's own `(args)` tokens (and the remaining `::segment`s)
**unconsumed in the token stream**. When that qualified call appeared as a
*standalone statement*, the leftover tokens were merely garbage that the next
statement's parser had to (incidentally) recover from. When it appeared
*nested inside an outer call's argument list*, those same leftover tokens —
including the inner call's own `(1, 2)` — were still sitting where the outer
call's argument-scanning loop was looking for the next comma or closing
paren, inflating the outer call's perceived argument count. Standalone vs.
nested was never a separate code path; it was the same truncated-parse defect
observed through two different lenses.

PR #624's fix (flatten a 2+-`::` chain to its terminal segment, then fall
through to ordinary `fn_find`/bare-call resolution) makes the qualified call
consume its own tokens completely and correctly in *every* context, so the
outer call's argument scan never sees anything left over. No additional
change was needed for the nested-argument case specifically.

## Related note already resolved elsewhere

Issue #601's Bug D entry also flagged, as a possibly-related but unconfirmed
concern, that qualified/3+-segment calls "fail to propagate their return type
correctly for later indexing/field-access" (citing issue #570). That was
fixed independently by issue #570's own fix, prior to this session, and is
confirmed still working here (`make_vertex(7, 9).x` above).

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_verify.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127
```

Matches the current baseline exactly (unchanged from post-Bug-C) — no
regressions, because no source was changed.

## Cross-references

- `docs/audit/LEAN_SINGLE_MULTISEGMENT_QUALIFIED_CALL_2026-07-04.md` — Bug A
  (PR #624), the fix that resolved this as a side effect.
- GitHub issue #601 — Bug D closed by this verification (no code PR).
  Bugs E–G remain open, plus the `use ... as alias` variant noted in
  `docs/audit/LEAN_SINGLE_NAMED_USE_IMPORT_2026-07-05.md`.
