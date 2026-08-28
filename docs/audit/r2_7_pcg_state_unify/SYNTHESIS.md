<!-- docs:meta
topic_id: repo.docs.audit.r2-7-pcg-state-unify.synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-7-pcg-state-unify.synthesis
-->

# R.2.7 — Unify PcgState + step in pcg64_core — RESOLVED (via R.3.1)

**Status:** RESOLVED (2026-05-18, via R.3.1 `d5b43742…`). Previously
RESOLVED-PARTIAL via R.2.8; rng.sio third-caller Phase D landed once
R.3.1 fixed the TUP_CACHE_KEY hash collision. HALTED IN PHASE A
on 2026-05-17; subsequently unblocked by the compiler patch landed in
`ce9810ee9` and completed by R.2.8 caller wiring.
**Wall-clock:** ~20 min Phase A diagnosis (2026-05-17) + R.2.8 Phase D
(~2h, 2026-05-18).
**Outcome:** distributions.sio + sampling.sio fully wired via
`type X = PcgState`; rng.sio reverted (a separate var-init/tuple-`.0`
fragility surfaced under Pass 1c.5 when `type Pcg64 = PcgState` is
added alongside the splitmix64/xoshiro state machine). See
[../r2_8_alias_deep_resolve/SYNTHESIS.md](../r2_8_alias_deep_resolve/SYNTHESIS.md).

## What happened

Phase A smoke-probed the Path A design (type alias `type DstPcg64 =
PcgState` over stdlib-imported `PcgState`) and found it fails to
compile under souc HEAD when the alias appears in a tuple return type
crossed by a stdlib import.

## Detailed finding

Pattern that fails:

```sounio
use stdlib::random::pcg64_core::{PcgState, pcg_step}
type DstPcg64 = PcgState

fn dst_pcg64_next_i64(rng: DstPcg64) -> (DstPcg64, i64) {
    return pcg_step(rng)   // pcg_step : PcgState -> (PcgState, i64)
}
// souc: error: return type does not match function signature
```

The same pattern in isolation (both modules in `/tmp/`, imported by short
name without the `stdlib::` prefix) **compiles and runs correctly**:

```sounio
// /tmp/alias_cross_mod.sio
struct Foo { x: i64 }
pub fn step(f: Foo) -> (Foo, i64) { (Foo { x: f.x + 1 }, f.x) }

// /tmp/alias_cross_main.sio
use alias_cross_mod::{Foo, step}
type Bar = Foo
fn wrap(b: Bar) -> (Bar, i64) { return step(b) }   // ✓ compiles, prints 11/10
```

So the failure is **stdlib-specific** — the typechecker discriminates
stdlib-imported struct identity from same-dir-imported struct identity
when the alias sits inside a tuple return.

## Variants attempted (all fatal)

| Variant | Result |
|---|---|
| `return pcg_step(rng)` | `error: return type does not match function signature` |
| `pcg_step(rng)` (tail-expr) | `error: tail type mismatch` |
| `let r = pcg_step(rng); return (r.0, r.1)` | `error: return type does not match function signature` |
| `let state: DstPcg64 = r.0; return (state, r.1)` | `error: return type does not match function signature` |

Also verified with a fresh struct (`MyU128 = PcgU128`) — same fatal
behavior. So this is not specific to `PcgState`; it's general to all
aliased stdlib-imported structs in tuple-return position.

## Why this halts Path A

Without alias-tuple-unification across stdlib boundaries, every wrapper
in the three caller modules would have to either:

(a) **Construct a new struct literal** of the locally-named type from
    the core return — but then the local "alias" isn't really an alias,
    it has to be a separate struct definition, defeating the dedup.
(b) **Not be a wrapper at all** — i.e. rename the user-facing API to
    `pcg_step` everywhere, which is Path B (hard switch).

Either way, the Path A "zero downstream signature change" promise from
DISPATCH §2.1 cannot be kept.

## Reproduction artefacts (preserved for forensics)

- `reference/core_alias_smoke.sio` — the failing case (does not compile)
- `reference/alias_isolated_works.sio` — documentation of the
  /tmp/-local control case that does compile

## Decision matrix

| Option | Effort | Blast radius | Recommendation |
|---|---|---|---|
| (i) Close R.2.7 as NOT_DONE; keep R.2.6 end state | 0 | 0 | **Recommended.** R.2.6 already harvested the big win (−149 LOC). R.2.7 was tech-debt-only and would only have saved ~40 more LOC. The marginal value isn't worth the type-system friction. |
| (ii) Path B (hard switch): rename all sampler signatures to `PcgState` | ~1-2h | ~50-100 signature edits across `distributions.sio`, `rng.sio`, `sampling.sio`, plus possibly `RngPcgWrapper` callers | Possible if operator wants to push through. The R.2.4/5 oracles still catch any algorithmic regression. |
| (iii) File a souc typechecker bug for stdlib-alias-tuple unification, then revisit R.2.7 | ~30 min to write up | 0 stdlib changes today | Reasonable parallel track if (i) is taken. The behavior looks like a typechecker limitation rather than intentional design — same-dir imports treat aliases consistently, stdlib imports don't. |

## Current state

- `stdlib/random/pcg64_core.sio` — **untouched** from its R.2.6 state.
  The speculative additions (PcgState, pcg_step, pcg_next_f64,
  pcg_next_f64_nonzero, pcg_bounded) were reverted because they'd be
  unused without a successful Path A.
- `stdlib/random/{distributions,rng,sampling}.sio` — **untouched**.
- Branch state is functionally identical to post-R.2.6 (`ac36becb8`).

## Recommendation

**Take option (i): close R.2.7 as NOT_DONE.** Pair with option (iii)
as a follow-up dispatch to file the typechecker observation — even if
nothing else changes, knowing this limitation exists is valuable for
future refactor planning.

Operator authorization required to:
- proceed with Path B (option ii), or
- close R.2.7 here.
