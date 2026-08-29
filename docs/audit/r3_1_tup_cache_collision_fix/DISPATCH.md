<!-- docs:meta
topic_id: repo.docs.audit.r3-1-tup-cache-collision-fix.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r3-1-tup-cache-collision-fix.dispatch
-->

# DISPATCH R.3.1 — Fix TUP_CACHE_KEY hash collision

**Opened.** 2026-05-18 (R.3.0 Phase A converged at `3509828a3`).
**Predecessor.** R.3.0 (`docs/audit/r3_0_transitive_alias_chain/SYNTHESIS.md`)
located the root cause: `(tcount * 1e6 + first_nslots * 1000 +
total_nslots)` is non-injective on element struct identity; every
`(N-field-struct, i64)` tuple in a module hashes to the same key.
**Class.** Compiler edit on `self-hosted/compiler/lean_single.sio`.
Self-hosted; requires `lean_single_fixed_point_gate.sh` +
`native_v2_cpu_compiler_umbrella_gate.sh` PASS.
**Priority.** P4 — purely unblocks rng.sio Phase D wiring (~−45 LOC
cosmetic refactor). No correctness issue in current operational
consumers; latent only.
**Branch.** `sounio-pure/r2-1-park-miller`.
**Time budget.** 2–4h single session.

---

## §0 — Sounio-Pure constraint

May read/write `self-hosted/compiler/lean_single.sio`, rebuild
`bin/souc-linux-x86_64`, run scripts/ci gates, write probes under
`docs/audit/r3_1_tup_cache_collision_fix/reference/`. No Python /
R / external tools. No bootstrap chain edits (boot3/boot4/native-v2).

If a fix requires touching the stage1 bootstrap chain — HALT.

---

## §1 — What R.3.0 located

Three relevant code sites in `self-hosted/compiler/lean_single.sio`:

| Line | Site | Role |
|---|---|---|
| 4478 | `scan_type` tuple branch | Computes `SCAN_TY_HASH = tcount * 1e8 + first_f64 * 1e7 + last_f64 * 1e6 + first_nslots * 1000 + total_nslots` (100M format) |
| 23014 | Pass 1 fn-ret signature scan | Computes `scan_ret_hash = tcount_enc * 1e6 + first_nslots * 1000 + total_nslots` (1M format, derived from scan_type's by `/1e6`) |
| 23464 | Pass 1c.5 fn-ret re-encode | Re-runs the same `tcount_enc * 1e6 + ...` formula post-alias-resolution |
| 666–680 | `tup_cache_register` | Idempotent on key — `if TUP_CACHE_KEY[i] == tup_hash { return }` |
| 683–690 | `tup_cache_lookup` | Returns first-registered entry's index for the colliding key |

The 1M-format hash drops the `tcount` value (always 2 for our case) +
the `first_f64`/`last_f64` flags into the high digits — but does **not**
encode element-struct identity. Two distinct 4-field structs A and B
both produce `(A, i64)` and `(B, i64)` hashes equal to `200004005`.

---

## §2 — Three fix directions (from R.3.0 §3)

| Option | Surgery | Risk | Notes |
|---|---|---|---|
| **(Fa)** Encode low bits of `first_hash` (and `last_hash`) into TUP_CACHE_KEY | Medium (~40 lines, audit all hash producers/consumers) | Hash format change touches scan_type, Pass 1, Pass 1c.5, tuple-literal expr at line 9336, every consumer that decodes via `/ 1000000` math. Risk: subtle decoding mismatches across paths. | Cleanest single-key fix. |
| **(Fb)** Composite cache key `(tup_hash, first_hash, last_hash)` in TUP_CACHE | Large (~80 lines, add two more parallel arrays, change register + lookup + every callsite) | Larger refactor; touches every tup_cache callsite. Risk: cache size pressure if Sounio's struct uniqueness grows. | Correct long-term shape. |
| **(Fc)** Pass 1c.5 force-overwrite on collision-detected | Small (~15 lines, only in Pass 1c.5 re-encode loop) | Leaves collision latent for non-alias paths. Risk: only addresses Phase D Path A; a future caller defining two 4-field structs *both* with concrete signatures (no alias on either) would still hit the original collision. | Quickest. |

**Recommended primary direction: (Fc).** Rationale:
- The only path R.2.7 / R.2.8 / R.2.9 / R.3.0 have surfaced needs is
  the alias-resolution path. Concrete-struct callers in the umbrella
  haven't exposed the latent collision in practice (it's been latent
  since at least R.2.4).
- Surgery is bounded: extend Pass 1c.5's idempotency check so when
  the existing entry's `first_hash` differs from the re-encoded
  `re_first_hash`, the new (alias-resolved) entry wins. Other passes
  remain untouched.
- If the surgery widens the scope of regressions, fall back to (Fa).
  Do **not** attempt (Fb) in R.3.1.

**Fallback: (Fa).** If (Fc) breaks fixed-point or umbrella, pivot to
hash-format widening. Allocates low 4–8 bits of TUP_CACHE_KEY to
`first_hash & 0xff` (or similar) so equal-slot-count distinct structs
get distinct keys without breaking the decoding math used by lookups.

**Out of bounds:** (Fb). R.3.1 budget doesn't support a parallel-arrays
refactor of the cache. If both (Fc) and (Fa) fail, ship R.3.1 as a
DEFERRED with notes for a multi-session R.3.x.

---

## §3 — Attack plan

### Phase A — Verify the locus (15 min)

1. Run `reference/hash_collision_gate.sio` under current souc
   (`c7ea6a4d…`). Expect: `error: assignment type mismatch at line 37`.
2. Add a temporary `print_int(EXPR_TY_HASH); print_int(TUP_CACHE_FIRST_HASH[tup_ci])`
   debug at the failing assignment site (one of the 10 sites at lines
   17015 / 17194 / etc.). Rebuild souc, re-run gate. Expected output:
   identical hash, divergent first_hash (RngXoshiro256_hash vs PcgState_hash).
3. If observed: locus confirmed. Remove debug prints; proceed to Phase B.
4. If unexpected (different bug shape): HALT and re-scope.

### Phase B — Apply (Fc) (30–45 min)

In `self-hosted/compiler/lean_single.sio` Pass 1c.5 re-encode loop
(around line 23465):

```
tup_cache_register(re_ret_hash, re_first_ty, re_first_hash, re_last_ty, re_last_hash)
```

Replace with a force-overwrite-on-mismatch variant:

```
let existing_ci = tup_cache_lookup(re_ret_hash)
if existing_ci >= 0 {
    // Collision: check whether the existing entry's element types match the alias-resolved ones.
    if TUP_CACHE_FIRST_HASH[existing_ci as usize] != re_first_hash || TUP_CACHE_LAST_HASH[existing_ci as usize] != re_last_hash {
        // Force-overwrite: this function's alias-resolved entry wins.
        TUP_CACHE_FIRST_TY[existing_ci as usize] = re_first_ty
        TUP_CACHE_FIRST_HASH[existing_ci as usize] = re_first_hash
        TUP_CACHE_LAST_TY[existing_ci as usize] = re_last_ty
        TUP_CACHE_LAST_HASH[existing_ci as usize] = re_last_hash
    }
} else {
    tup_cache_register(re_ret_hash, re_first_ty, re_first_hash, re_last_ty, re_last_hash)
}
```

This adds an explicit collision-resolver in Pass 1c.5 only. Non-alias
paths still use the idempotent registration.

### Phase C — Validation (45 min)

1. `lean_single_fixed_point_gate.sh`: PASS required (stage1 == stage2
   == stage3). This is the cheap fast-fail signal.
2. `reference/hash_collision_gate.sio`: PASS required (was FAIL).
3. `native_v2_cpu_compiler_umbrella_gate.sh`: 12/12 PASS required.
4. R.2.4 distributions oracle: 1024/1024 bit-exact.
5. R.2.5 rng self-oracle: 1024/1024 bit-exact via reverted rng.sio.

### Phase D — rng.sio Phase D wiring (15 min)

With the collision fix in, retry R.2.8's deferred rng.sio Phase D:

- `type Pcg64 = PcgState`
- Collapse `pcg64_next_*` family to one-line forwards into
  `pcg_step` / `pcg_next_f64` / `pcg_next_f64_nonzero`.

Validate: rng_oracle_gen.sio compiles + 1024/1024 bit-exact against
R.2.5 oracle.

### Phase E — Commit + push (10 min)

Three commits:

1. `self-hosted/compiler/lean_single.sio` + `bin/souc-linux-x86_64` —
   Pass 1c.5 collision-resolver patch.
2. `stdlib/random/rng.sio` — R.2.8's deferred Phase D wiring.
3. `docs/audit/r3_1_tup_cache_collision_fix/` — DISPATCH + SYNTHESIS +
   reference/.
4. `docs/audit/r3_0_transitive_alias_chain/SYNTHESIS.md` — flip
   Phase A COMPLETE → R.3.0 RESOLVED (close with pointer to R.3.1).
5. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/SYNTHESIS.md` — flip
   RESOLVED-PARTIAL → RESOLVED.
6. `docs/audit/r2_8_alias_deep_resolve/SYNTHESIS.md` — flip
   RESOLVED-PARTIAL → RESOLVED.
7. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — flip
   RESOLVED-PARTIAL → RESOLVED.

HALT for operator review before push.

---

## §4 — Out of scope

- Bootstrap chain (boot3/boot4/native-v2).
- (Fb) composite-cache-key refactor.
- Any other stdlib refactor outside R.2.8's deferred rng.sio.
- Refinement-type predicates in Pass 1c.5.

---

## §5 — Halt conditions

- **Phase A locus verification fails.** R.3.0 diagnosis was wrong;
  re-scope.
- **Phase B (Fc) breaks lean_single fixed-point.** Revert. Either
  pivot to (Fa) within budget or close R.3.1 as DEFERRED.
- **Phase B (Fc) causes any non-R.2.x umbrella sub-gate regression.**
  Revert; the collision-resolver had unintended scope.
- **Phase C oracle replay fails one sample.** Stream drift — revert
  Phase D wiring, keep compiler patch, mark R.3.1 PARTIAL.
- **Diagnosis reveals (Fc) isn't sufficient (e.g. collision fires on
  non-alias path during fixed-point).** Pivot to (Fa). If (Fa) also
  fails within 90 min from pivot, HALT and ship R.3.1 PARTIAL.

---

## §6 — Deliverables on close

1. `self-hosted/compiler/lean_single.sio` — Pass 1c.5 collision-resolver
   patch (or (Fa) hash widening if pivoted).
2. `bin/souc-linux-x86_64` — rebuilt; new md5 + size delta documented.
3. `stdlib/random/rng.sio` — Phase D-wired (`type Pcg64 = PcgState` +
   one-liner wrappers).
4. `docs/audit/r3_1_tup_cache_collision_fix/DISPATCH.md` — this file.
5. `docs/audit/r3_1_tup_cache_collision_fix/SYNTHESIS.md` — closing
   writeup with diagnosis confirmation + patch diff + LOC delta.
6. `docs/audit/r3_1_tup_cache_collision_fix/reference/hash_collision_gate.sio` — Phase A/C gate probe (already staged).
7. R.2.7 / R.2.8 / R.2.9 / R.3.0 SYNTHESIS flipped to RESOLVED.

---

## §7 — Acceptance

R.3.1 is **VALIDATED** iff:

1. ✓ `lean_single_fixed_point_gate.sh`: PASS (stage1 == stage2 == stage3).
2. ✓ `native_v2_cpu_compiler_umbrella_gate.sh`: 12/12 PASS.
3. ✓ `hash_collision_gate.sio`: typechecks (was FAIL).
4. ✓ R.2.4 distributions oracle: 1024/1024 bit-exact.
5. ✓ R.2.5 rng self-oracle: 1024/1024 bit-exact (now via wired rng.sio).
6. ✓ R.2.8-deferred rng.sio Phase D lands; LOC delta ≤ −40.
7. ✓ R.2.7 + R.2.8 + R.2.9 + R.3.0 SYNTHESIS all flipped to RESOLVED.

If 1 or 2 fails: FAIL. Revert; reassess.
If 3 fails: FAIL. Phase B fix didn't close the bug.
If 4 or 5 fails: FAIL. Stream drift; debug wiring.
If 6 PARTIAL (compiler patch lands but rng.sio wiring fails for a
*different* reason): ship the compiler patch alone; R.3.1 PARTIAL;
predecessor synthesises stay RESOLVED-PARTIAL.

---

## §8 — Notes

- R.3.0's diagnosis is the load-bearing artefact. Don't re-derive it
  during Phase A — verify and proceed.
- The four-dispatch chain (R.2.7 → R.2.8 → R.2.9 → R.3.0 → R.3.1)
  has converged because R.3.0 explicitly refused to pre-commit to a
  fix direction. R.3.1 can pre-commit because the diagnosis is now
  evidence-backed, not hypothesised.
- After R.3.1 lands, the four-cycle sequence retires with:
  - one compiler bugfix (R.2.9: 1-field-struct scalar branch)
  - one compiler bugfix (R.3.1: TUP_CACHE collision in Pass 1c.5)
  - one stdlib refactor (R.2.7 Path A across all three callers)
  - cumulative −189 LOC stdlib reduction (R.2.6 −149 + R.2.8 −28 +
    R.3.1 rng.sio −45 ≈ −222; better than the original R.2.7 target).

**END OF DISPATCH.**
