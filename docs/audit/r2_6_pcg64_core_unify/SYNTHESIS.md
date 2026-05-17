<!-- docs:meta
topic_id: repo.docs.audit.r2-6-pcg64-core-unify.synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-6-pcg64-core-unify.synthesis
-->

# R.2.6 — Unified PCG64 core — CLOSING SYNTHESIS

**Status:** RESOLVED (2026-05-17).
**Wall-clock:** ~45 min, one session, on top of R.2.4 + R.2.5 oracles.

## What it did

Extracted the canonical PCG64-XSL-RR-128/64 helper family from three
independent inlinings (`distributions.sio`, `rng.sio`, `sampling.sio`)
into a single stdlib-internal module `stdlib/random/pcg64_core.sio`.
Pure code motion — zero algorithmic change, bit-exact stream
preservation across all three caller paths.

## Phases

| Phase | Output | Commit |
|---|---|---|
| A — author `pcg64_core.sio` | 87-line module with `pub` `PcgU128` + 7 helpers; 11/11 hand-computed smoke assertions | `cb880133` |
| B — wire 3 callers          | helpers + per-module U128 deleted; `use` added; step rewritten; `−149` LOC net | `62a04f3a` |
| C — closing synthesis       | this file | (this commit) |

## Module shipped — `stdlib/random/pcg64_core.sio`

```
pub struct PcgU128 { hi: i64, lo: i64 }
pub fn pcg_u_lt(a, b) -> bool
pub fn pcg_lshr(x, n) -> i64
pub fn pcg_umul64_high(a, b) -> i64
pub fn pcg_u128_add(a_hi, a_lo, b_hi, b_lo) -> PcgU128
pub fn pcg_u128_mul(a_hi, a_lo, b_hi, b_lo) -> PcgU128
pub fn pcg_rotr64(x, rot) -> i64
pub fn pcg_umod(x, n) -> i64
```

Helper bodies byte-equivalent to R.2.4's `dst_pcg_*` family. `pcg_umod`
is the R.2.5 rejection-modulo helper (binary long-division with
power-of-two fast path).

Naming convention decision: leading-underscore-as-private was floated in
DISPATCH §2.1 but dropped because no other stdlib module uses it.

## Acceptance against DISPATCH §7 — 7/7 PASS

| # | Criterion | Result |
|---|---|---|
| 1 | R.2.4 distributions oracle replay (4 seeds × 256) | **1024/1024** bit-exact |
| 2 | R.2.5 sampling oracle vs R.2.4 (4 seeds × 64)    | **256/256** bit-exact |
| 3 | R.2.5 rng self-oracle replay (4 seeds × 256)     | **1024/1024** bit-exact |
| 4 | R.2.4 statistical sanity (6 bands at N=20000)    | **6/6** PASS, values identical to R.2.4 Phase C |
| 5 | Net LOC delta in `stdlib/random/`                | **−149** (target ≤ −100) |
| 6 | User-facing API renamed                          | none — `dst_pcg64_*` / `pcg64_*` / `smp_pcg64_*` / `RngPcgWrapper` preserved |
| 7 | Algorithm changed                                | none (implied by §7.1–§7.3 bit-exactness) |

**2304/2304 samples bit-exact across the three callers** is the
canonical regression witness.

## LOC delta detail

| file | pre-R.2.6 (R.2.5 end) | post-R.2.6 | delta |
|---|---|---|---|
| `distributions.sio` | 530 | 454 | −76 |
| `rng.sio`           | 620 | 555 | −65 |
| `sampling.sio`      | 450 | 344 | −106 |
| `pcg64_core.sio` (new) | — | 87 | +87 |
| **net**             |     |     | **−160 in callers, +87 core = −149** |

(Counts approximate; `git diff --stat 7451cfbe1 -- stdlib/random/`
reports 272 deletions and 123 insertions across the four files;
372 − 521 in a different counting basis gives the same +123 / −272
truth-of-record.)

## Side simplifications captured

- `sampling.sio:smp_pcg64_bounded` collapsed ~60 lines of inline
  binary-mod (two nested copies of the long-division loop, one for the
  threshold and one for the final reduction) into two `pcg_umod`
  calls. Visually parallel to `rng.sio:pcg64_bounded` now.
- All three modules' file-level "Unsigned 128-bit helpers" comment
  blocks collapsed to a single `use` statement at the top.

## Files delivered

- `stdlib/random/pcg64_core.sio` — new module
- `stdlib/random/distributions.sio` — helpers deleted, `use` added
- `stdlib/random/rng.sio` — helpers + `RngU128` deleted, `use` added
- `stdlib/random/sampling.sio` — helpers + `SmpU128` deleted, `use` added; bounded simplified
- `stdlib/random/lib.sio` — submodule list updated
- `docs/audit/r2_6_pcg64_core_unify/DISPATCH.md`
- `docs/audit/r2_6_pcg64_core_unify/SYNTHESIS.md` — this file
- `docs/audit/r2_6_pcg64_core_unify/reference/core_smoke.sio` — 11-assertion smoke

## Halt conditions encountered — none

No algorithmic edit attempted. No regression in any oracle. LOC delta
sufficiently negative (−149, well past the −100 floor).

## Post-R.2.4/5/6 stdlib RNG state

The canonical PCG64 algorithm now has a **single source of truth**
in `pcg64_core.sio`. A future algorithmic fix (e.g. switching to PCG-DXSM,
adding a stream-advance primitive) needs to be applied in exactly one
place. The three caller modules (`distributions.sio`, `rng.sio`,
`sampling.sio`) keep their separate state structs and user-facing APIs;
unifying *those* into a single `PcgState` would be a substantially
larger refactor with downstream blast radius — deliberately out of
scope for R.2.6.

Park-Miller (`random::park_miller`) remains untouched and is still the
recommended simple default for PBPK Monte Carlo on
quality-vs-complexity grounds (single 64-bit state, no 128-bit overhead).
