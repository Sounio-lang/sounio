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

**254 functions** would need `with Mut` added *directly*. Examples:
`examples/algo/sorting_demo.sio::partition`, `::merge`, `::heapify`,
`examples/algo/graph_demo.sio::dfs_visit`,
`examples/collections/bitset_demo.sio::bitset_set`.

### Correction (same day): the real cost is the transitive closure, ~1673

**254 is not the migration cost. It is the seed of it.** An earlier revision of
this dispatch recommended landing the 254 signature additions first as a
"mechanical" prerequisite. That recommendation was wrong and is retracted
below; this section records why, because the mechanism is the interesting part.

Madaros *does* enforce effect **propagation** at call sites — that is exactly
what `check_callee_effects` does. It simply never **originates** a `Mut`
requirement from a mutation. So the effect system is consistent but
incomplete: declared effects flow correctly from callee to caller, and no
effect is ever born. Adding origination — which is what the specified rule
requires — forces the entire propagation closure above each origin to be
declared too. Verified directly:

```
fn writes(buf: &![i64; 4]) with Mut { (*buf)[0] = 1 }
fn caller(buf: &![i64; 4]) { writes(buf) }        # no Mut

$ ./bin/souc check cascade.sio                    # Madaros, default
error[E035] ... : effect not declared in function signature (missing: Mut)
```

Computing the closure over the 254 origins (iterate: any function lacking
`Mut` that calls a function in the set joins the set; repeat to fixpoint):

```
round 1: +430  -> 677
round 2: +719  -> 1396
round 3: +208  -> 1604
round 4:  +58  -> 1662
round 5:  +10  -> 1672
round 6:   +1  -> 1673
round 7:   +0  -> converged

TRANSITIVE CLOSURE = 1673 functions
  stdlib       980
  examples     334
  self-hosted  244   <-- the compiler's own source
  tests        104
  tools          7
  benchmarks     4
```

Spot-checked for soundness rather than trusted: `examples/algo/graph_demo.sio::dfs`
genuinely calls `dfs_visit`; `sorting_demo.sio::quicksort_range` calls
`partition`; `::mergesort_recursive` calls `merge`. All three lack `Mut`.

**Over-count caveat, stated plainly.** Call resolution here is by *name*
(`\bname\s*\(`), not by module-aware resolution, so the closure over-counts
where an unrelated function or method shares a name across modules. 1673 is an
upper bound. It is not a tight one — but it would have to be wrong by more than
6× to restore the "mechanical" framing, and the direction of the error is
knowable only with a real call graph, which the modular checker could provide
and this scan cannot.

**This vindicates the 2026-07-11 assessment rather than narrowing it.** That
dispatch's "would false-reject a large fraction of the ~6,000-file corpus" was
closer to right than this dispatch's first estimate. The earlier revision here
claimed 254 "materially narrows the risk"; it does not, because 254 counts
origins and the compiler enforces the closure. **1673 of roughly 6,900 functions
that take an exclusive reference is a fraction of the corpus, and 244 of them
are in `self-hosted/`** — meaning the compiler would have to be re-annotated to
compile itself under its own new rule, with all the bootstrap and fixed-point
exposure that implies.

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
   §7.2.1 of the specification. That removes the "requires a maintainer design
   decision" blocker, and only that one.
2. The blast radius is now measured rather than estimated, and it **confirms**
   the 2026-07-11 concern: 254 origins, **1673 functions in the transitive
   closure**, 244 of them inside `self-hosted/`.

An earlier revision of this section argued the measurement *narrowed* the risk.
It does not — that argument compared 254 origins against an estimate of the
whole closure. Corrected above. The regression discipline that dispatch
prescribed applies in full, and its scepticism about wiring E035-class
enforcement is better supported now than when it was written.

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
3. **Migration order is still migrate-then-enforce, but the migration is not
   mechanical.** It is 1673 functions including 244 in the compiler, and it must
   be driven by the closure, not by the 254 origins — annotating an origin
   without its callers makes the callers red immediately (see the Correction
   section). Get a real call graph out of the modular checker before planning
   the sweep; a name-based scan is not a safe migration driver even though it is
   adequate for sizing. Consider whether the checker should *infer and
   propagate* silently, requiring the annotation only at a module boundary,
   which would collapse most of the 1673.
4. **Do not touch `lean_single.sio`.** The seed's broad reading is a known,
   deliberately frozen property (`docs/compiler/KNOWN_LIMITATIONS.md`); relaxing
   it risks the bootstrap fixed point for no gain, since the correctness
   guarantee lives in Madaros. The consequence — that `egraph.sio` and similar
   pure helpers stay un-checkable under the seed — is a documented limitation,
   not a target of this dispatch.
5. **Scope discipline.** `Div` is also unenforced by the seed (noted above).
   Do not bundle it. One effect per dispatch.

## Live evidence (reviewer-reported, PR #1531, same day)

Two data points surfaced in review of #1531, reported by the reviewer and
recorded here as reported — **not independently reproduced.** An attempt to
reproduce the second one directly (`souc check self-hosted/compiler/main.sio`
under Madaros) hit the same AST-closure preflight abort this PR's own body
already documents as advisory/inconclusive, so the reviewer's methodology
(evidently a full self-compile, not a single-file `check`) was not repeated
here.

1. **The 880-file `requires: madaros` corpus is closer to landable than its
   raw size implies.** Run against a from-source Madaros (merged `main`,
   includes #1522) via `SOUNIO_TEST_SOUC_BIN` + `SOUNIO_MADAROS_AVAILABLE=1`:
   `Pass: 579, Fail: 13, Known failures: 286, Skip: 2` (sums to 880). The 13
   failures cluster into two named groups (4 misc; 9 in one `lorenz_i128_*`/
   `lorenz_i256_*` interval/Taylor-solver family) rather than scattering — a
   triage-sized problem, not a lane. This is a different corpus from the
   1673-function closure above (it is the existing `requires: madaros` test
   *annotations*, not functions requiring `Mut`), reported here because it
   bears on how expensive "measure before deciding" turns out to be in
   practice for Madaros-only-guard work generally. Two caveats the reviewer
   attached and worth preserving: enabling this in CI means moving the engine
   (`SOUNIO_TEST_SOUC_BIN`) together with the flag, not setting the flag alone
   against `souc-stage2` — these tests are annotated precisely because
   `lean_single` cannot compile them; and the marginal build cost is near zero
   because `scripts/ci/madaros_current_source_f64_lowering_gate.sh` already
   produces a shared from-source Madaros ELF for five other gates.

2. **A measured argument for the "infer and propagate" reading, not the
   "annotate everywhere" one.** E035 count on a Madaros self-compile reportedly
   moved 226 → 230 after #1522 (`fix(check): generate Mut at the store site`)
   landed. Read together with this dispatch's mechanism section: #1522 added
   an *origination* point (the store site now originates `Mut` where nothing
   did before), and because Madaros has no automatic propagation up the call
   graph, four functions that write through an exclusive reference without
   declaring it started failing honestly — the count rose, not fell, exactly
   as this dispatch's closure argument predicts for *any* newly-added origin
   under the current call-site-only enforcement. That is live, not
   hypothetical, evidence that "annotate everywhere" scales badly one origin
   at a time, and it is the concrete case for scoping the "inferred and
   propagated" option in the Recommendation below before more origins land.

## Recommendation

**Do not start the signature migration, and do not wire the guard.** Both were
recommended in the first revision of this dispatch; the closure measurement
retracts both.

What to decide first, because it changes the size of the work by an order of
magnitude: **should the annotation be required at every function in the closure,
or inferred-and-propagated silently with the annotation required only where the
effect crosses a module boundary?**

- *Annotation required everywhere* (what the current call-site enforcement
  implies): ~1673 functions, 244 in the compiler itself, with bootstrap and
  fixed-point exposure. This is a multi-week corpus migration, not a
  prerequisite step.
- *Inferred and propagated* (what §7.2.1's "the checker is intended to infer
  this" already says): the checker derives `Mut` through the call graph and only
  demands it be written where a caller cannot see the callee's body. Most of the
  1673 collapse. This is more checker work and far less corpus churn, and it is
  the reading the specification already commits to.

The second option is almost certainly the intended design and is what §7.2.1
says. It should be scoped and costed before any annotation lands, because
annotating under option one and then implementing option two would leave ~1400
signatures carrying an annotation the compiler no longer needs.

Until that is decided, the specification (§7.2.1) and the guide correctly
describe the intended rule and honestly record that neither engine implements
it. That is a stable, non-misleading state to sit in.
