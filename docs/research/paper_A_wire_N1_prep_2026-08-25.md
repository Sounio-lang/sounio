<!-- docs:meta
topic_id: repo.docs.research.paper-a-wire-n1-prep-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-wire-n1-prep-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NS anti-garbling wire — N1 prep pack (2026-08-25)

Free, pre-code preparation for the NS→checker wire (synthesis §26), assembled so the
hot-file sprint can start the moment the coordination handshake clears. **Nothing here
touches `self-hosted/` or `stdlib/`** — these are staging artifacts in the research tree.

## ⛔ The blocker is a handshake, not code

§26 (codex grant, msg-1787455026) authorizes the wire under a strict serialization
protocol. Before the **first** shared-file edit, ALL of the following must happen, and none
can be done unilaterally from this FFI worktree:

1. Create the dedicated worktree `fable/ns-wire-20260823` from base `06e85a6ada`
   (**not** this `lane/fable-1/p0f-ffi-takeover` branch — §26 forbids the FFI branch).
2. Claim the write-set: `noise_sets.sio` (new), `types.sio`, `check.sio`, `epistemic.sio`,
   `scripts/bootstrap/bootstrap_concat.sh`, `run_knowledge_bootstrap_tests.sh`,
   `test_parse_all.sio`, the new tests, one NS gate path. Do **not** touch `check/mod.sio`
   without asking.
3. File the Semantic-Lane declaration (below).
4. Send codex `branch + claim_id + E230 + test/gate names` on the bus **before** editing.
5. xai math-review before commit.

So this pack does everything up to step 4's payload. Landing N1 is gated on the above and
is **not** attempted here.

## Grounding confirmed today (2026-08-25)

- **E230 is free** — `grep -rn E230 self-hosted/ stdlib/` returns nothing; highest used
  `E2xx` is `E218`. E230 is distinct from E222 (R-ORIGIN) and E224.
- **Base `06e85a6ada` exists** — `git cat-file -t` = commit.
- **Field host** — `TypeEntry` at `self-hosted/check/types.sio:139`; the ~6 sibling
  `ontology_id: -1` init sites are the pattern to mirror.
- **Join template** — `compat.sio:250` `TyModelFamily` arm (`a.tag == b.tag`) is the shape;
  the `kadd`/`kmul` site is `check.sio:~18862`.

## Semantic-Lane declaration (ready to file)

```
Semantic-Lane-ID: ns-antigarbling-wire-20260823
Concept-IDs:      SOUNIO-NOISE-SYMBOL, SOUNIO-ANTIGARBLING   (sibling to SOUNIO-PROVENANCE)
Owner:            Fable-1
Base:             06e85a6ada
Diagnostic:       E230 — "anti-garbling: independence-assuming op over non-disjoint/unknown
                  noise-symbol sets"  (distinct from E222 R-ORIGIN, E224)
Write-set:        self-hosted/check/noise_sets.sio (new), types.sio, check.sio, epistemic.sio,
                  scripts/bootstrap/bootstrap_concat.sh, run_knowledge_bootstrap_tests.sh,
                  self-hosted/test_parse_all.sio, tests/{compile-fail,run-pass}/ns_*, scripts/ci/ns_antigarbling_gate.sh
Do-not-touch:     self-hosted/check/mod.sio (without asking); E222/E224 (keep causally separable)
```

## N1 — representation only (behavior-neutral) — the diff spec

**Goal:** add the field and the module; consult it nowhere. Bootstrap + source build
byte-identical behaviour.

1. **`types.sio` — `TypeEntry`:** add trailing `noise_set_id: i64` **after** `provenance_id`.
   Default `-1` (⊤ / unknown) at every construction site (mirror the `ontology_id: -1`
   inits). Do **not** reuse `ontology_id` (ChEBI/domain — collision) nor `knowledge_epsilon`
   (overloaded Transport/Diagram/Fairness/Grade).
2. **`noise_sets.sio` (new module):** the interned-set table + the pure API
   `ns_intern(set) -> i64`, `ns_union(i64,i64) -> i64` (⊤ absorbs), `ns_disjoint(i64,i64) -> bool`
   (both ≥ 0 and bit-disjoint via the table; either ⊤ ⇒ false). Handles are identities;
   union/disjoint dereference through the table, no bitwise ops on the handle. (Semantics
   validated by the souc-green prototype `ns_contract.sio`.)
3. **No rule reads `noise_set_id` yet.** N1 acceptance = the existing full suite is green and
   the bootstrap is behaviorally unchanged (the large, safe diff).

N2 (transfer: seed/union/ident + parametric call-summary) → N3 (E230 gate at kadd/kmul +
same-source-built sabotage) → N4 (named gate + regression) follow, per §26.

## Acceptance fixtures (ready to drop into `tests/` at N3/N4)

> Staged here; they encode the target post-wire behaviour. `//@ compile-fail` fixtures expect
> **E230**; `//@ run-pass` fixtures must type-check and run.

**`tests/compile-fail/ns_add_shared_source_rejected.sio`**
```sio
//@ compile-fail
//! x + x: operands share a measured source ⇒ independence-assuming add is E230.
fn main() -> i64 with Epistemic, Mut, Div, Panic {
    let x = measure(10.0, uncertainty: 1.0)
    let s = x + x            // E230: non-disjoint noise-symbol sets
    return 0
}
```

**`tests/compile-fail/ns_add_unknown_conservative.sio`**
```sio
//@ compile-fail
//! One operand has unknown (⊤) source-set ⇒ never disjoint ⇒ E230 (conservative).
fn main() -> i64 with Epistemic, Mut, Div, Panic {
    let x = measure(10.0, uncertainty: 1.0)
    let u = opaque_knowledge()   // returns Knowledge<f64> with noise_set_id = -1 (⊤)
    let s = x + u                // E230: unknown is not provably disjoint
    return 0
}
```

**`tests/run-pass/ns_add_disjoint_ok.sio`**
```sio
//@ run-pass
//! x + y: two independent measurements (disjoint sources) ⇒ admitted.
fn main() -> i64 with Epistemic, Mut, Div, Panic {
    let x = measure(10.0, uncertainty: 1.0)
    let y = measure(20.0, uncertainty: 2.0)
    let s = x + y            // OK: {s_x} ∩ {s_y} = ∅
    return 0
}
```

**`tests/run-pass/ns_ident_preserves_source.sio`**
```sio
//@ run-pass
//! Copy/ident preserves the source-set: ident(x) + y stays disjoint from y only if x is.
fn main() -> i64 with Epistemic, Mut, Div, Panic {
    let x = measure(10.0, uncertainty: 1.0)
    let y = measure(20.0, uncertainty: 2.0)
    let ix = x               // copy: inherits {s_x}
    let s = ix + y           // OK: disjoint; and ix + x would be E230 (identity survives)
    return 0
}
```

## Gate script (ready to drop at `scripts/ci/ns_antigarbling_gate.sh`)

```bash
#!/usr/bin/env bash
# NS anti-garbling gate: the same-source-built sabotage witness.
# PASS iff: (a) the compile-fail fixtures raise E230, (b) the run-pass fixtures build,
# and (c) with ONLY the NS rule disabled on the SAME source build, the E230 on x+x
# vanishes while an unrelated E222 (R-ORIGIN) fixture still fails.
set -euo pipefail
SOUC=./bin/souc
fail=0
expect_e230() { $SOUC check "$1" 2>&1 | grep -q 'E230' || { echo "MISS E230: $1"; fail=1; }; }
expect_ok()   { $SOUC check "$1" >/dev/null 2>&1 || { echo "MISS OK: $1"; fail=1; }; }

expect_e230 tests/compile-fail/ns_add_shared_source_rejected.sio
expect_e230 tests/compile-fail/ns_add_unknown_conservative.sio
expect_ok   tests/run-pass/ns_add_disjoint_ok.sio
expect_ok   tests/run-pass/ns_ident_preserves_source.sio

# Sabotage: rebuild with SOUNIO_NS_DISABLE=1 (NS rule only); E230 must vanish, E222 must remain.
SOUNIO_NS_DISABLE=1 $SOUC check tests/compile-fail/ns_add_shared_source_rejected.sio 2>&1 \
  | grep -q 'E230' && { echo "SABOTAGE FAIL: E230 survived NS-disable"; fail=1; }
SOUNIO_NS_DISABLE=1 $SOUC check tests/compile-fail/r_origin_launder.sio 2>&1 \
  | grep -q 'E222' || { echo "SABOTAGE FAIL: E222 vanished with NS-disable"; fail=1; }

[ $fail -eq 0 ] && echo "NS anti-garbling gate: PASS" || { echo "NS anti-garbling gate: FAIL"; exit 1; }
```

## What is done vs. blocked

| Item | State |
|---|---|
| E230-free confirmed; base commit confirmed; field/join sites located | ✅ done |
| Semantic-Lane declaration drafted | ✅ ready to file |
| N1 diff spec (field + module API, behavior-neutral) | ✅ specified |
| Four acceptance fixtures + gate script | ✅ drafted (staged) |
| Create worktree `fable/ns-wire-20260823`, claim files, notify codex, xai review | ⛔ **handshake — needs codex/user** |
| Land N1 in `self-hosted/` + Madaros build (build-lock, ~4min) | ⛔ blocked on the handshake |
