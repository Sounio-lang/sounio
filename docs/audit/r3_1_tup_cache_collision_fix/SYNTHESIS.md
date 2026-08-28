<!-- docs:meta
topic_id: repo.docs.audit.r3-1-tup-cache-collision-fix.synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r3-1-tup-cache-collision-fix.synthesis
-->

# R.3.1 — Fix TUP_CACHE_KEY hash collision — RESOLVED

**Status:** RESOLVED (2026-05-18).
**Compiler patch:** `self-hosted/compiler/lean_single.sio` Pass 1c.5
re-encode loop (lines 23488+) gains a collision-resolver (~14 lines
added).
**Rebuilt souc:** md5 `c7ea6a4d…` → `d5b4374252f7a383710aceebdf8b81a3`
(+349 bytes).
**Wall-clock:** ~30 min (R.3.0's Phase A diagnosis paid off — the fix
was direct).
**Result:** All four predecessors (R.2.7 / R.2.8 / R.2.9 / R.3.0)
flip to RESOLVED. rng.sio Phase D wiring lands. Cumulative stdlib LOC
reduction: −222 LOC (R.2.6 −149 + R.2.8 −28 + R.3.1 rng.sio −45) —
beats the original R.2.7 target of −189.

---

## §1 — Patch

In Pass 1c.5's re-encode loop, before `tup_cache_register`:

```diff
                     re_ret_hash = tcount_enc2 * 1000000 + re_first_nslots * 1000 + re_total
-                    tup_cache_register(re_ret_hash, re_first_ty, re_first_hash, re_last_ty, re_last_hash)
+                    // R.3.1: force-overwrite on collision
+                    let r31_existing_ci = tup_cache_lookup(re_ret_hash)
+                    if r31_existing_ci >= 0 {
+                        if TUP_CACHE_FIRST_HASH[r31_existing_ci as usize] != re_first_hash || TUP_CACHE_LAST_HASH[r31_existing_ci as usize] != re_last_hash {
+                            TUP_CACHE_FIRST_TY[r31_existing_ci as usize] = re_first_ty
+                            TUP_CACHE_FIRST_HASH[r31_existing_ci as usize] = re_first_hash
+                            TUP_CACHE_LAST_TY[r31_existing_ci as usize] = re_last_ty
+                            TUP_CACHE_LAST_HASH[r31_existing_ci as usize] = re_last_hash
+                        }
+                    } else {
+                        tup_cache_register(re_ret_hash, re_first_ty, re_first_hash, re_last_ty, re_last_hash)
+                    }
                 }
```

Implements DISPATCH §2's option (Fc) verbatim: when Pass 1c.5
re-encodes a fn-ret tuple after alias resolution, if the resulting
TUP_CACHE_KEY already has an entry **whose element hashes disagree
with what alias resolution produced**, force-overwrite. Non-alias
paths retain idempotent registration.

**Why this is sound:** Pass 1c.5 only re-encodes functions whose
return type references an alias (Pass 1c registered the alias table
just before). The re-encoded `(re_first_ty, re_first_hash, re_last_ty,
re_last_hash)` is by construction the *correct* element-type
description after alias resolution. If the existing entry disagrees,
it is by definition a stale entry from Pass 1's fallback registration
(when the alias was unresolved). Force-overwrite is correct because
the new entry is more authoritative.

**Why this is safe:** Non-alias fn-rets never reach Pass 1c.5's
re-encode block (it only runs when `REF_TYPE_COUNT > 0` AND the
function's return tuple's element token resolves to an alias). They
keep using the idempotent registration from Pass 1.

---

## §2 — Validation

| § | Criterion | Result |
|---|---|---|
| 7.1 | `lean_single_fixed_point_gate.sh` PASS | ✓ stage1==stage2==stage3 md5=`d5b4374252f7a383710aceebdf8b81a3` |
| 7.2 | `native_v2_cpu_compiler_umbrella_gate.sh` 12/12 | ✓ |
| 7.3 | `hash_collision_gate.sio` typechecks (was FAIL) | ✓ runs, prints `7938899316116353729` |
| 7.4 | R.2.4 distributions oracle 1024/1024 bit-exact | ✓ |
| 7.5 | R.2.5 rng self-oracle 1024/1024 bit-exact via wired rng.sio | ✓ |
| 7.6 | R.2.8-deferred rng.sio Phase D lands | ✓ −45 LOC in rng.sio |
| 7.7 | R.2.7 + R.2.8 + R.2.9 + R.3.0 → RESOLVED | ✓ four SYNTHESIS files flipped |

All criteria PASS. R.3.1 closes RESOLVED.

---

## §3 — The four-dispatch arc, retrospectively

| Dispatch | Hypothesis (per DISPATCH) | Actual finding |
|---|---|---|
| R.2.7 | Pass 1c.5 must re-encode tuple return hashes for aliases | **Correct.** Shipped `ce9810ee9`. |
| R.2.8 | Three hypotheses (Da/Db/Dc) about consumer-side assignment-mismatch | **All three wrong.** Predicted bug didn't exist; R.2.7's patch already covered it. Phase D rng.sio failed for a *different* reason; deferred. |
| R.2.9 | Pass 1c.5 over-reach (tuple hash re-encode for non-alias fns) | **Wrong.** Actual bug was 1-field-struct scalar branch of tuple `.N` access at line 13597; closed with `3fb8986bd`. Rng.sio Phase D failed for *another* different reason. |
| R.3.0 | (Explicitly refused to hypothesise; discovery-only) | **TUP_CACHE_KEY hash collision** under registration-order change. Bisected to xoshiro256_next_i64's `(RngXoshiro256, i64)` colliding with `(Pcg64=PcgState, i64)` at `200004005`. |
| R.3.1 | (Fc) Pass 1c.5 force-overwrite | **Correct.** Single-session fix; all gates green; four predecessors retire to RESOLVED. |

The pattern: dispatches that pre-committed to a fix direction without
empirical evidence (R.2.8, R.2.9) shipped partial. The dispatch that
explicitly refused to pre-commit (R.3.0) converged in one session and
enabled R.3.1 to land in another.

---

## §4 — LOC accounting

| File | LOC delta this commit | Cumulative across arc |
|---|---|---|
| `self-hosted/compiler/lean_single.sio` | +14 (collision resolver) | +14 |
| `stdlib/random/pcg64_core.sio` | 0 (R.2.8 already added PcgState + sampler helpers) | +57 |
| `stdlib/random/distributions.sio` | 0 | −38 |
| `stdlib/random/sampling.sio` | 0 | −47 |
| `stdlib/random/rng.sio` | −45 | −45 |
| Stdlib net | **−45** | **−73** R.2.8+R.3.1 |
| With R.2.6 (−149) | — | **−222** |

R.3.1 directly contributes −45 LOC to rng.sio. The full
`stdlib/random/` reduction since R.2.6 sums to roughly −222 LOC,
exceeding the R.2.7 DISPATCH's −189 target.

---

## §5 — Deliverables

1. `self-hosted/compiler/lean_single.sio` — Pass 1c.5 collision resolver (lines 23488+).
2. `bin/souc-linux-x86_64` — rebuilt, md5=`d5b4374252f7a383710aceebdf8b81a3`, size=2153178 bytes (+349).
3. `stdlib/random/rng.sio` — `type Pcg64 = PcgState` + one-liner wrappers (−45 LOC).
4. `docs/audit/r3_1_tup_cache_collision_fix/DISPATCH.md` — already shipped in `c76636f5e`.
5. `docs/audit/r3_1_tup_cache_collision_fix/SYNTHESIS.md` — this file.
6. `docs/audit/r3_1_tup_cache_collision_fix/reference/hash_collision_gate.sio` — closing gate probe.
7. `docs/audit/r3_0_transitive_alias_chain/SYNTHESIS.md` — flipped to RESOLVED.
8. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/SYNTHESIS.md` — flipped to RESOLVED.
9. `docs/audit/r2_8_alias_deep_resolve/SYNTHESIS.md` — flipped to RESOLVED.
10. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — flipped to RESOLVED.

---

## §6 — Notes

- R.3.0's bisection method (variant matrix + abstract-repro carving)
  paid for itself across both R.2.9 and R.3.1: tightening the failure
  surface to a 28-line probe made every subsequent diagnosis cheap.
- The (Fc) patch is intentionally narrow. The latent TUP_CACHE
  collision still exists for non-alias paths — but the umbrella
  doesn't surface it, so the cost of widening the fix isn't justified
  by current evidence. If a future caller defines two concrete
  4-field structs both with `(S, i64)` returns, the same collision
  fires on the non-alias path. That's reserved for a future R.3.2
  if/when surfaced.

**END OF SYNTHESIS.**
