<!-- docs:meta
topic_id: repo.docs.audit.r2-7-pcg-state-unify.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-7-pcg-state-unify.dispatch
-->

# DISPATCH R.2.7 — Unify caller-side state structs + step function

**Opened.** 2026-05-17
**Predecessors.** R.2.4 / R.2.5 (algorithmic canonicalisation); R.2.6 (helper unification → `pcg64_core.sio`).
**Class.** Pure stdlib refactor. No compiler involvement. No algorithmic change.
**Priority.** P4 — tech-debt only. Marginal win vs R.2.6 (~3 redundant struct decls + ~30 LOC of duplicated step body). Defer freely.
**Branch.** Continue on `sounio-pure/r2-1-park-miller`.
**Time budget.** 1–2h single session.

---

## §0 — Sounio-Pure constraint

May read/write `stdlib/random/pcg64_core.sio`, `distributions.sio`,
`rng.sio`, `sampling.sio`, `lib.sio`. Probes under
`docs/audit/r2_7_pcg_state_unify/reference/`. No external tools.
Inherits oracles from R.2.4 (1024) + R.2.5 sampling (256 vs R.2.4) +
R.2.5 rng self-oracle (1024) = **2304/2304 bit-exact** is the
acceptance witness.

**HALT and report** if any algorithmic edit becomes necessary, or if
the type-alias path (Phase A) fails to round-trip.

---

## §1 — Scope

After R.2.6, three caller modules each ship an identical struct +
near-identical step body:

```
// distributions.sio                rng.sio                       sampling.sio
struct DstPcg64 {                   struct Pcg64 {                struct SmpPcg64 {
    state_hi: i64,                      state_hi: i64,                state_hi: i64,
    state_lo: i64,                      state_lo: i64,                state_lo: i64,
    inc_hi: i64,                        inc_hi: i64,                  inc_hi: i64,
    inc_lo: i64,                        inc_lo: i64,                  inc_lo: i64,
}                                   }                             }

fn dst_pcg64_next_i64(...)          fn pcg64_next_i64(...)        fn smp_pcg64_next_i64(...)
    { /* 11-line body */ }              { /* 11-line body */ }        { /* 11-line body */ }
```

Three duplicated struct definitions (≈4 LOC × 3 = 12) and three nearly-
identical step bodies (≈11 LOC × 3 = 33). Total ≈45 LOC of duplication.

The user-facing API names *also* differ across the three (`dst_pcg64_*`
vs `pcg64_*` vs `smp_pcg64_*`); R.2.7 does **not** rename them.

---

## §2 — Design fork

### §2.1 — Path A (type alias, RECOMMENDED — verified workable)

Add to `pcg64_core.sio`:

```
pub struct PcgState {
    state_hi: i64, state_lo: i64,
    inc_hi: i64,   inc_lo: i64,
}

pub fn pcg_step(rng: PcgState) -> (PcgState, i64) { /* canonical body */ }
```

Then each caller module **deletes its struct and `use`s a type alias**:

```
use stdlib::random::pcg64_core::{PcgState, pcg_step, ...}
type DstPcg64 = PcgState                              // in distributions.sio
```

(and `type Pcg64 = PcgState` in rng.sio, `type SmpPcg64 = PcgState` in
sampling.sio.)

**Verified 2026-05-17 in workspace probe:** `type Bar = Foo` where
`Foo` is a struct compiles and `fn take_bar(b: Bar)` accepts `Foo`
values bit-exact. Downstream API signatures (`uniform_sample(dist,
rng: DstPcg64)`, `sample_normal(mean, std, rng: SmpPcg64)`, etc.)
remain unchanged because `DstPcg64` and `SmpPcg64` are still valid
type names.

**Pro.** Zero downstream signature changes. PBPK code, tests, examples,
the dissertation pipeline — all untouched.
**Con.** Two type names (the alias and the underlying) refer to the
same thing; readers must follow the alias to learn the shape. Mitigated
by the alias living in a single one-line `type X = PcgState` per module
right under the `use`.

### §2.2 — Path B (hard switch, NOT recommended)

Replace `DstPcg64` / `Pcg64` / `SmpPcg64` with `PcgState` in every
function signature across the three modules and all downstream
callers (samplers, `RngPcgWrapper`, etc.). Larger blast radius;
touches dozens of distribution-sampler signatures in distributions.sio
+ sampling.sio.

**Pro.** Single canonical name `PcgState` everywhere.
**Con.** ~50-100 downstream signature edits per module; risk of typo
regression. The R.2.4/5 oracles still catch any algorithmic break,
but they won't catch e.g. a misplaced `rng: Pcg64` left over in some
helper signature.

**Recommended: Path A.**

### §2.3 — Step function unification

In both paths, hoist the step body into `pcg_step(state: PcgState)` in
core. Each caller's `*_next_i64` becomes a one-line wrapper that calls
`pcg_step`. Saves ≈30 LOC.

```
// e.g. in distributions.sio:
fn dst_pcg64_next_i64(rng: DstPcg64) -> (DstPcg64, i64) {
    return pcg_step(rng)
}
```

(Type-aliased so `DstPcg64 == PcgState`, signature compiles.)

### §2.4 — `*_next_f64` and `*_bounded` also unify

Both have identical bodies modulo state-type names. Move to core as
`pcg_next_f64`, `pcg_bounded`. Each caller's `*_next_f64` /
`*_bounded` becomes a wrapper one-liner.

`*_next_f64_nonzero` is identical across all three callers too —
hoist as `pcg_next_f64_nonzero`.

`*_next_bool` exists only in `rng.sio` (`pcg64_next_bool`); keep
there.

### §2.5 — Stays in callers

- `pcg64_new(seed)` / `dst_pcg64_new(seed)` / `smp_pcg64_new(seed)` —
  seed functions differ between modules (rng.sio uses splitmix64,
  distributions/sampling use pcg-cpp canonical). **Stay in callers.**
- All distribution samplers (`uniform_sample`, `normal_sample`, etc.)
  in distributions.sio and sampling.sio. **Stay.** The state-type
  alias means their signatures don't have to change.
- `RngPcgWrapper` and its shims in rng.sio. **Stay.**

---

## §3 — Attack plan

### Phase A — Verify alias round-trip (10 min)

(Already done in dispatch authoring: `/tmp/alias_test.sio` confirms
`type Bar = Foo` + `fn take_bar(b: Bar)` accepts `Foo` values. If a
deeper test in the actual `stdlib/random/` import path reveals an
issue, **HALT** and fall back to Path B with explicit operator
authorization.)

Smoke probe: write a minimal `core_alias_smoke.sio` that imports
`PcgState` from a draft `pcg64_core.sio`, aliases it as `type DstPcg64
= PcgState`, and verifies `dst_pcg64_next_i64(...)` still returns the
expected first sample for seed=31415 (= `-2825318976064776997` from
the R.2.4 oracle).

### Phase B — Author core additions (15 min)

Add to `pcg64_core.sio` (extending the R.2.6 module):
- `pub struct PcgState`
- `pub fn pcg_step(rng: PcgState) -> (PcgState, i64)`
- `pub fn pcg_next_f64(rng: PcgState) -> (PcgState, f64)`
- `pub fn pcg_next_f64_nonzero(rng: PcgState) -> (PcgState, f64)`
- `pub fn pcg_bounded(rng: PcgState, n: i64) -> (PcgState, i64)`

Bodies byte-equivalent to the R.2.5/2.6 caller copies.

### Phase C — Wire callers (20 min)

For each of distributions.sio, rng.sio, sampling.sio:
1. Add `PcgState, pcg_step, pcg_next_f64, pcg_next_f64_nonzero` (+ `pcg_bounded` where applicable) to the existing `use stdlib::random::pcg64_core::{...}`.
2. Delete the local struct (`DstPcg64` / `Pcg64` / `SmpPcg64`).
3. Add `type X = PcgState` alias one-liner.
4. Rewrite `*_next_i64` / `*_next_f64` / `*_next_f64_nonzero` /
   `*_bounded` as one-line wrappers calling the core functions.
5. **Do not touch** the seed function bodies.
6. **Do not touch** any distribution sampler (`uniform_sample` etc.).

### Phase D — Replay all oracles (5 min)

Same 4-gate replay as R.2.6:
1. R.2.4 distributions oracle: 1024/1024 bit-exact
2. R.2.5 sampling vs R.2.4 oracle: 256/256
3. R.2.5 rng self-oracle: 1024/1024
4. R.2.4 stat sanity: 6/6 PASS

### Phase E — Closing synthesis (10 min)

`SYNTHESIS.md` against §7. Commit. HALT for operator review before
push.

---

## §4 — Out of scope

- Renaming user-facing API (`dst_pcg64_*` / `pcg64_*` / `smp_pcg64_*`).
- Renaming caller-side struct names from a *user* perspective — the
  aliases `DstPcg64` / `Pcg64` / `SmpPcg64` stay valid and usable.
- Touching distribution samplers, RngPcgWrapper, or any downstream
  caller (PBPK, tests, examples). The whole point of Path A is that
  these don't move.
- Park-Miller, xoshiro, splitmix64. Untouched.
- Any algorithmic change. R.2.7 is pure code motion.

---

## §5 — Halt conditions

- **Alias round-trip fails** in the actual stdlib import path (despite
  the Phase A smoke proving it works in isolation). Means some
  downstream code does runtime type discrimination that aliases break.
  HALT and fall back to Path B with explicit auth.
- **Any oracle replay fails one sample.** Refactor introduced a bug.
- **Stat sanity drifts.** Same.
- **LOC delta is non-negative.** Refactor didn't consolidate; reconsider.
- **Temptation to also unify the seed functions or rename the user-facing API.** HALT — those are larger downstream-touching changes, separate dispatch.

---

## §6 — Deliverables on close

1. `stdlib/random/pcg64_core.sio` — extended with `PcgState` + 4 step/sampler functions.
2. `stdlib/random/distributions.sio` — struct deleted, alias + wrappers.
3. `stdlib/random/rng.sio` — same.
4. `stdlib/random/sampling.sio` — same.
5. `docs/audit/r2_7_pcg_state_unify/reference/core_alias_smoke.sio` — Phase A alias verification.
6. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — closing writeup.

---

## §7 — Acceptance

R.2.7 is **VALIDATED** iff:

1. ✓ R.2.4 distributions oracle replay: 1024/1024 bit-exact.
2. ✓ R.2.5 sampling oracle vs R.2.4: 256/256 bit-exact.
3. ✓ R.2.5 rng self-oracle: 1024/1024 bit-exact.
4. ✓ R.2.4 stat sanity: 6/6 PASS.
5. ✓ Net LOC delta in `stdlib/random/` ≤ −40 (consolidation actually happened; smaller floor than R.2.6's −100 because R.2.6 already harvested the big wins).
6. ✓ No user-facing API renamed (`dst_pcg64_*` / `pcg64_*` / `smp_pcg64_*` / `DstPcg64` / `Pcg64` / `SmpPcg64` all still usable).
7. ✓ Distribution samplers (`uniform_sample`, `normal_sample`,
     `sample_weighted_index`, etc.) and `RngPcgWrapper` shims —
     **zero** signature changes.
8. ✓ Algorithm unchanged (implied by 1–3).

If 1, 2, or 3 fails: FAIL.
If 7 fails (any sampler signature changed): HALT — scope creep; revert
the offending edit.

---

## §8 — Notes

- The marginal LOC win over R.2.6 is small (~40 vs R.2.6's 149) because
  R.2.6 already harvested the helper deduplication. R.2.7 is mostly
  about reaching the "obvious" end state — single state struct, single
  step function — rather than about lines of code.
- Path A (type alias) is what makes this dispatch cheap. Without
  alias support, the cost-benefit would have flipped against doing
  this at all.
- After R.2.7, `pcg64_core.sio` will hold: the U128 helpers, the
  `PcgState` struct, and the canonical step/sampler functions. Caller
  modules retain: their seed functions, their distribution samplers,
  and their type alias one-liner. That's the natural separation: core
  owns "what PCG64 is"; callers own "how this codebase uses PCG64".

**END OF DISPATCH.**
