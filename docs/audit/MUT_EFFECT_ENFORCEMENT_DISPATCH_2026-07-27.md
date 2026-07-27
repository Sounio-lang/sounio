<!-- docs:meta
topic_id: repo.docs.audit.mut-effect-enforcement-dispatch-2026-07-27
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.mut-effect-enforcement-dispatch-2026-07-27
-->

# Dispatch — implement escape-based `Mut` enforcement in the modular Madaros checker

**Filed:** 2026-07-27 · **Status:** OPEN (dispatch, not yet implemented) · **Protocol:** CLAUDE.md §8 (`self-hosted/` changes require a forensic dispatch before code).

## Summary

The intended semantics for the `Mut` effect — now specified in
`docs/spec/LANGUAGE_SPECIFICATION.md` §7.2.1 — is **escape-based**: mutation the
caller can observe requires `Mut`; mutating a function-local `var` does not,
because the binding dies with the frame. The checker is intended to *infer* this
from the body rather than make the author reason about it.

**Neither shipped engine implements that rule.** They fail in opposite
directions, and the default compiler is the permissive one:

| Case | Madaros (default, `self-hosted/check/`) | `lean_single` (frozen seed) | Intended |
|---|---|---|---|
| local `var` mutation | accepts without `Mut` | **rejects** without `Mut` | accept |
| write through `&![T; N]` | **accepts** without `Mut` | rejects without `Mut` | require `Mut` |

So implementing the specified rule is **not a relaxation of existing
behaviour** — in the default compiler it is *new enforcement* for the escaping
case, where today nothing is enforced at all. That is the material finding of
this dispatch, and it is why the work is filed rather than done inline.

## Evidence

**Madaros enforces effects only at call sites, never at an assignment.**
`report_effect_error` (E035, `self-hosted/check/check.sio:11666`) has exactly one
caller path: `check_callee_effects` (`:11688`) / its in-place twin
`checker_check_callee_effects_inplace` (`:6920`), invoked from ~17 sites, all of
which pass a *callee's* declared effect set (`sig.effects`) to check it is a
subset of the current function's. No site derives a `Mut` requirement from a
mutation in the body. Confirmed behaviourally:

```
$ cat probe.sio
fn bump() -> i64 { var x = 0  x = x + 1  x }
$ ./bin/souc check probe.sio                       # Madaros, default
check: OK
$ cat probe2.sio
fn write_through(buf: &![i64; 4]) { (*buf)[0] = 99 }
$ ./bin/souc check probe2.sio                      # Madaros, default
check: OK
```

**`lean_single` requires `Mut` for both, and the trigger is `Mut`, not `Div`.**

```
$ SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check probe.sio
error: effect not declared in function signature at line 3   # the local mutation
typecheck: failed
$ SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check probe2.sio
error: effect not declared in function signature at line 2   # the write-through
typecheck: failed
```

Isolated to rule out `Div` as the real cause: a local `var` mutation containing
**no** division is still rejected by the seed, and a bare `a / b` with **no**
mutation is accepted by it. (Note in passing: the seed therefore does not
enforce `Div` either. Out of scope here — do not bundle it.)

**Live consequence in the compiler's own source.** `self-hosted/ir/egraph.sio`
fails `SOUNIO_SOUC_ENGINE=lean_single souc check` at line 1549, inside
`eg_isqrt` (`egraph.sio:1511`) on `y = (y + x / y) / 2` — a local `var` in a
pure integer helper. This is one of the two pre-existing reasons that file
cannot currently be exercised standalone under either engine (the other is a
Madaros lowering segfault, filed separately as
`docs/audit/MODULE_FRONTEND_LOWER_ARRAY_SEED_CRASH_DISPATCH_2026-07-27.md`).

## Blast radius — measured, syntactic

A deterministic scan of all versioned `.sio` sources (excluding `archive/`,
`bootstrap/`, `.claude/`), matching functions that take an exclusive-ref
parameter (`&!` or `*mut`), omit `Mut` from the signature, and assign through
that parameter (`(*p)… =`, `p.field =`, `p[i] =`, `*p =`):

```
fns_with_exclusive_ref_param     = 6894
  ...of which lack Mut in sig    =  847
  ...AND write through it        =  254   <-- would newly need `with Mut`
```

**254 functions** is the migration cost of turning enforcement on. Examples:
`examples/algo/sorting_demo.sio::partition`, `::merge`, `::heapify`,
`examples/algo/graph_demo.sio::dfs_visit`,
`examples/collections/bitset_demo.sio::bitset_set`.

Caveats on this number, stated rather than buried: it is **syntactic**, so it
(a) misses closure-capture escapes entirely, (b) misses mutation that escapes
through a helper call rather than a direct assignment, and (c) may over-count if
any matched assignment is to a local shadowing a parameter name. Treat 254 as
the right order of magnitude and the starting worklist, not a certified total.

**Do not baseline this work with a naive per-file `souc check` sweep.** That
instrument is unreliable here for a reason this repository already documented in
`docs/audit/CHECKER_GUARD_WIRING_DISPATCH_2026-07-11.md`: files that depend on
imports fail when checked standalone because imports are unresolved. Measured
again on 2026-07-27 while scoping this dispatch — a 60-file `stdlib/` sample
produced 52 "failures" that were dominated by that artefact (e.g.
`stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio` reports
*"declare the variable before use, or import it from another module"*), not by
effect divergence. Use the real harness (`bash scripts/run_sio_test_suite.sh`),
which supplies module context.

## Relationship to prior art in this repository

`docs/audit/CHECKER_GUARD_WIRING_DISPATCH_2026-07-11.md` triaged E035 and
recorded: **"NO — do not wire … The permissiveness is almost certainly
intentional gradual/optional effects. Enforcing would false-reject a large
fraction of the ~6,000-file corpus. Requires an explicit design decision from
the maintainer, not a bug fix."**

Two things have changed since, and both should be weighed before acting:

1. A maintainer design decision now exists — escape-based inference, recorded in
   §7.2.1 of the specification.
2. The measured blast radius for the *escaping* case specifically is **254
   functions**, not "a large fraction of ~6,000 files". The earlier estimate was
   for E035 as a whole (all effects, including `IO`/`Div` and the non-escaping
   local case, which the decision explicitly *excludes*).

That materially narrows the risk versus the 2026-07-11 assessment — but it does
not eliminate the need for the regression discipline that dispatch prescribed.

## Implementation notes

1. **This is an additive checker guard**, so the 2026-07-11 landing rule
   applies verbatim: *do not land unless the change produces zero new failures
   across everything that passes today*, baselined with the real harness, not a
   per-file sweep.
2. **Inference, not annotation-checking.** The specified semantics is that the
   checker derives the requirement. The insertion point is wherever assignment
   places are resolved in the modular checker — start from the assignment/place
   handling in `self-hosted/check/check.sio` and the exclusive-reference
   machinery, and thread an "escapes" bit out of place resolution rather than
   pattern-matching syntax at the call site.
3. **Migrate before enforcing, in that order.** Land the 254 signature
   additions first (mechanical, individually reviewable, green under both
   engines since `lean_single` already demands `Mut` there), *then* turn the
   guard on. Enforcing first makes the corpus red for the duration of the
   migration.
4. **Do not touch `lean_single.sio`.** The seed's broad reading is a known,
   deliberately frozen property (`docs/compiler/KNOWN_LIMITATIONS.md`); relaxing
   it risks the bootstrap fixed point for no gain, since the correctness
   guarantee lives in Madaros. The consequence — that `egraph.sio` and similar
   pure helpers stay un-checkable under the seed — is a documented limitation,
   not a target of this dispatch.
5. **Scope discipline.** `Div` is also unenforced by the seed (noted above).
   Do not bundle it. One effect per dispatch.

## Recommendation

Land the **254-function signature migration** first as its own PR — it is
mechanical, it is already required by one of the two engines, and it is a
prerequisite for enforcement being landable at all. File the checker-side
inference as a follow-up dispatch once that migration is green, and keep the
guard off until then.
