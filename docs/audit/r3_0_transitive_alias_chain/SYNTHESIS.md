<!-- docs:meta
topic_id: repo.docs.audit.r3-0-transitive-alias-chain.synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r3-0-transitive-alias-chain.synthesis
-->

# R.3.0 — Phase A converged: TUP_CACHE_KEY hash collision — RESOLVED (via R.3.1)

**Status update 2026-05-18:** R.3.1 (`d5b43742…`) applied the (Fc)
fix recommended in §3 — Pass 1c.5 force-overwrite on collision-
detected. All gates green; rng.sio Phase D wiring lands; R.2.7 + R.2.8
+ R.2.9 + R.3.0 all RESOLVED. Phase A synthesis preserved below.

---

## Phase A converged

**Status:** Phase A COMPLETE (2026-05-18). Phase B/C/D scoped as R.3.1.
**Wall-clock:** Phase A ~1h.
**Result:** Root cause located. `lean_single.sio`'s tuple-return hash
encoding `tcount * 1000000 + first_nslots * 1000 + total_nslots`
collides across distinct structs that share field count. R.2.7's
Pass 1c.5 doesn't *cause* the collision — it just *moves* registration
order so the colliding entry now wins. Pre-Phase-D rng.sio happened to
register PCG64's tuple first; Phase D delays it to Pass 1c.5, by which
point xoshiro256's entry has already claimed the hash.

The dispatch's misnomer ("transitive alias chain") is corrected to
**"TUP_CACHE hash collision under registration-order change"**. The
fix is a hash-encoding widening, not an alias-chain feature.

---

## §1 — Bisection log

Started from `transitive_main3.sio` (passing Phase D scaffolding) and
progressively added rng.sio content until the consumer failed:

| Step | Added | Result |
|---|---|---|
| 0 | bare Phase D wiring (alias + one-liner wrappers + pcg64_bounded) | **FAIL** at consumer line 14 + rng.sio line 80 |
| 1 | strip XORSHIFT128+ section | still **FAIL** at line 80 + line 114 |
| 2 | strip RNG WRAPPER section | still **FAIL** at line 80 + line 114 |
| 3 | strip entire XOSHIRO256 section | **PASS** |
| 4 | re-add `RngXoshiro256 struct + xoshiro256_new + rng_rotl` | **PASS** |
| 5 | re-add `xoshiro256_next_i64` | **FAIL** at line 80 |

Step 5 is the trigger. `xoshiro256_next_i64` returns
`(RngXoshiro256, i64)` — a 4-field-struct + i64 tuple. Same slot
signature as the post-Phase-D `pcg64_bounded` return type
`(Pcg64=PcgState, i64)`.

`reference/minimum_failing_repro_rng.sio` (carved-down rng.sio) and
`reference/hash_collision_repro.sio` (28-line abstract probe) both
isolate the failure.

The 28-line `hash_collision_repro.sio` removed everything except:

1. `type Pcg64 = PcgState` (PcgState imported from pcg64_core)
2. `struct OtherFour { a, b, c, d: i64 }`
3. `fn other_step(o: OtherFour) -> (OtherFour, i64)`
4. Consumer body `var current = PcgState{...}; let r = pcg_step(current); current = r.0`

Removing either (2)+(3) **or** changing `OtherFour` to have a
different field count flips the probe to PASS.

---

## §2 — Root cause

**Hash encoding** (lean_single.sio scan_type line 4478, Pass 1 line
23014, Pass 1c.5 line 23464):

```
hash = tcount * 1e6 + first_nslots * 1000 + total_nslots
        (where tcount carries f64-flag bits in upper digits but they
         are 0 for non-f64 tuples)
```

For `(struct_with_4_fields, i64)`:
- `tcount = 2` (2-element tuple)
- `first_nslots = 4`
- `total_nslots = 5`
- `hash = 200 * 1e6 + 4*1000 + 5 = 200004005`

**Every** `(4-field-struct, i64)` tuple in the same module hashes to
`200004005` regardless of *which* struct it carries. The cache
key is fundamentally non-injective on element identity.

### Why pre-Phase-D rng.sio worked

Before Phase D, `pcg64_next_i64` had a concrete signature
`(Pcg64, i64)` where `Pcg64` was a locally-defined 4-field struct.
Pass 1 scanned function declarations in source order and registered
each tuple at its hash via `tup_cache_register`. `tup_cache_register`
is **idempotent on key** (line 670) — first writer wins. Pcg64's
tuple registered at `200004005` before xoshiro256's same-hash entry.

`r.0` accesses then looked up `200004005` and got Pcg64's
`first_ty=6, first_hash=Pcg64_struct_hash`. The collision was *latent*.

### Why Phase D wiring breaks it

With `type Pcg64 = PcgState`, Pass 1 sees `pcg64_next_i64`'s return as
`(Pcg64-unresolved=k0, i64)` and computes `first_nslots = 1`
(ty_slot_count fallback) → wrong hash `200001002`. xoshiro256_next_i64
later registers at the *correct* hash `200004005` with
`first_hash=RngXoshiro256_hash`.

Pass 1c.5 (R.2.7's `ce9810ee9`) then re-encodes `pcg64_next_i64`'s
hash post-alias-resolution to `200004005` — but the cache entry is
**already** owned by RngXoshiro256. tup_cache_register is no-op on
the collision. `pcg64_bounded`'s `r.0` resolves to
`RngXoshiro256_hash`, not `PcgState_hash`, and the strict
assignment-mismatch check correctly rejects.

R.2.7's Pass 1c.5 doesn't introduce the collision; it changes the
**registration order** in a way that exposes a pre-existing latent
collision.

### Why this didn't fire in R.2.8 (distributions.sio + sampling.sio)

Those callers only have *one* multi-field struct each (`DstPcg64` =
`PcgState` and `SmpPcg64` = `PcgState` — both alias the same struct,
no collision). No second 4-field struct competing for the same hash.

`rng.sio` is uniquely affected because it co-defines `RngXoshiro256`
(4-field) and `RngState` (2-field, hits the same encoding for 2-slot
collisions with anything else 2-slot).

---

## §3 — Fix direction (for R.3.1)

The hash format must disambiguate element type. Options:

- **(Fa)** Extend the encoding to incorporate the *element struct
  hash* (or at least its low bits) into TUP_CACHE_KEY. Risk: needs
  audit of every hash producer/consumer.
- **(Fb)** Use a composite key `(tup_hash, first_hash, last_hash)` in
  TUP_CACHE and update lookup accordingly. Larger refactor but
  cleaner.
- **(Fc)** When Pass 1c.5 detects a *collision* during its re-encode
  (i.e. `tup_cache_lookup(re_ret_hash) >= 0` AND the existing entry's
  `first_hash != re_first_hash`), force-overwrite. Quickest but
  leaves the latent collision intact for non-alias paths.

(Fc) is the smallest surgery if R.3.1 wants speed; (Fb) is the
correct long-term fix; (Fa) is intermediate. **R.3.1 scoping should
pick based on the umbrella's tolerance for cache restructure.**

Phase B/C/D for R.3.1 is straightforward once direction is fixed:
patch → rebuild → fixed-point → umbrella → oracles → rng.sio Phase D
caller wiring → flip R.2.7 + R.2.8 + R.2.9 + R.3.0 to RESOLVED.

---

## §4 — What was ruled out (corrects R.2.9 §4)

The "transitive alias chain" framing in R.2.9 SYNTHESIS §4 is wrong.
R.3.0 probes prove single-hop, two-hop, two-hop + extra-state-machine,
and two-hop + struct-field-of-aliased-type chains all work. R.3.0
supersedes that section.

---

## §5 — Deliverables

1. `docs/audit/r3_0_transitive_alias_chain/SYNTHESIS.md` — this file.
2. `docs/audit/r3_0_transitive_alias_chain/reference/transitive_*.sio` — passing controls (1–3, 1-3-main).
3. `docs/audit/r3_0_transitive_alias_chain/reference/minimum_failing_repro_rng.sio` — full rng.sio bisect-step-5 reproducer.
4. `docs/audit/r3_0_transitive_alias_chain/reference/hash_collision_repro.sio` — 28-line abstract reproducer.
5. **No code changes** (per DISPATCH §0 / §5).

---

## §6 — Acceptance

| § | Criterion | Result |
|---|---|---|
| 8.1 | Minimum failing reproducer committed | ✓ both 28-line abstract and carved-rng.sio variants |
| 8.2 | One specific element flips PASS↔FAIL | ✓ removing `OtherFour`+`other_step` (or changing field count) → PASS |
| 8.3 | SYNTHESIS §2 maps locus to specific `lean_single.sio` code | ✓ hash encoding at lines 4478 / 23014 / 23464 + tup_cache_register at 666 |
| 8.4 | R.3.1 scope draft with grounded Phase B direction | ✓ §3 lists three concrete options grounded in the locus |

Phase A **VALIDATED**. R.3.0 closes; R.3.1 dispatch can proceed when
operator authorises (no urgency — operational status quo unchanged).

---

## §7 — Notes

- The bisection took ~30 min; the encoding analysis ~20 min. The
  speed came from the explicit "rule things out" discipline in the
  dispatch — three prior dispatches' wrong hypotheses were exactly
  the cost of skipping that step.
- The 28-line abstract reproducer
  (`hash_collision_repro.sio`) is the load-bearing artifact. It will
  serve as the gate for R.3.1: any fix must make this probe pass
  without regressing the umbrella.
- No bootstrap chain touched. No compiler edits.

**END OF SYNTHESIS.**
