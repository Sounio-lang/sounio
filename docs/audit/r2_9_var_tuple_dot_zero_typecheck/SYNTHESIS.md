<!-- docs:meta
topic_id: repo.docs.audit.r2-9-var-tuple-dot-zero-typecheck.synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-9-var-tuple-dot-zero-typecheck.synthesis
-->

# R.2.9 — `var x = fn(seed); x = tuple.0` typecheck fragility — RESOLVED (via R.3.1)

**Status update 2026-05-18:** Originally shipped RESOLVED-PARTIAL
because rng.sio Phase D was deferred. R.2.9 §4's "transitive alias"
hypothesis was wrong (corrected in R.3.0 SYNTHESIS). The actual
blocker was TUP_CACHE_KEY hash collision, fixed in R.3.1
(`d5b43742…`). rng.sio Phase D now landed; all R.2.9 acceptance
criteria now PASS.

**Original status:** RESOLVED-PARTIAL (2026-05-18).
**Compiler patch:** `self-hosted/compiler/lean_single.sio` lines 13597–
13630 (scalar branch of tuple `.N` field access). Rebuilt
`bin/souc-linux-x86_64` md5 `ffdc0fd5…` → `c7ea6a4d…` (+415 bytes).
**Wall-clock:** ~2h single session.
**Result:** All four classes of failing repros now typecheck; umbrella
12/12 PASS; fixed-point PASS; R.2.4 distributions oracle 1024/1024
bit-exact (regression check). R.2.7 rng.sio caller wiring re-attempted
in Phase D — fails for a **different** reason (transitive alias
`pcg64_core::PcgState → rng::Pcg64 → consumer`), so rng.sio stays
inlined; R.2.7 + R.2.8 remain RESOLVED-PARTIAL.

---

## §1 — Phase A diagnosis

DISPATCH §3 Phase A predicted three hypotheses (Da/Db/Dc) centred on
Pass 1c.5 over-reach. Diagnostic matrix (10 variants A–J + 3 follow-ups
K/L/M) rejected all three and isolated the root cause sharply:

| Property | Variants | Result |
|---|---|---|
| **1-field struct** in tuple slot (any init form) | A,B,C,D,G,I,J,K,M | **FAIL** |
| 2-field struct | E | PASS |
| 4-field struct (standalone) | F | PASS |
| 1-field struct returned directly (not via tuple) | L | PASS |
| `let sm = r.0` (no var-assign, no explicit type) | H | PASS (runtime correct too — see §1.1) |
| `let val: i64 = r.0` (explicit i64 ascription) | H3 | PASS — proves RHS *is* recorded as i64 |

**Root cause:** `self-hosted/compiler/lean_single.sio:13597–13610`
(scalar branch of tuple `.N` field access). For `elem_slots == 1`, the
code unconditionally sets `EXPR_TY = 1` (i64) — losing the nominal
struct type when the element is a 1-field struct (which is also 1 slot).
Downstream field-access on the recorded i64 is liberal enough to keep
working (which masks the bug in `let sm = r.0; sm.state` flows like H),
but the strict assignment-type-mismatch check at one of the 10 sites
correctly rejects `var sm: SplitMix64; sm = i64_value`.

This is **not** Pass 1c.5 over-reach — the bug fires even without any
alias (standalone repro confirms). R.2.7's patch merely *exposed* the
fragility in compositions where Pass 1c.5 had been masking it via
some hash-coincidence path that I didn't fully characterize.

### §1.1 — The H-passes-runtime puzzle

H (`let sm = r.0; return sm.state`) typechecks AND runs correctly (8).
That's because `sm` is recorded as i64 internally (EXPR_TY=1), but the
field-access path for a 1-slot value-of-i64 happens to load offset 0
correctly for both i64 (the value itself) and 1-field struct (the
single field at offset 0). Field-access typechecking is lenient enough
to accept `.state` on what it thinks is i64; codegen does the right
thing. Only `var = ...` strict assignment-comparison hits the failure
path.

The H3 probe (`let val: i64 = r1.0`) confirms this: explicit i64
ascription **passes**, proving the RHS is genuinely typed as i64.

---

## §2 — Patch

```diff
                 if elem_slots == 1 {
                     // Scalar element: dereference the pointer so rax holds the value.
                     let tup_first_f64 = (EXPR_TY_HASH / 10000000) % 10
                     let tup_last_f64 = (EXPR_TY_HASH / 1000000) % 10
                     let elem_is_f64 = if tup_idx == 0 { tup_first_f64 } else { tup_last_f64 }
                     em(0x48); em(0x8b); em(0x00)  // mov rax, [rax]
+                    // R.2.9: preserve nominal struct type for 1-field-struct elements.
+                    let r29_tup_ci = tup_cache_lookup(EXPR_TY_HASH)
+                    var r29_elem_ty: i64 = -1
+                    var r29_elem_hash: i64 = 0
+                    if r29_tup_ci >= 0 {
+                        if tup_idx == 0 {
+                            r29_elem_ty = TUP_CACHE_FIRST_TY[r29_tup_ci as usize]
+                            r29_elem_hash = TUP_CACHE_FIRST_HASH[r29_tup_ci as usize]
+                        } else {
+                            r29_elem_ty = TUP_CACHE_LAST_TY[r29_tup_ci as usize]
+                            r29_elem_hash = TUP_CACHE_LAST_HASH[r29_tup_ci as usize]
+                        }
+                    }
                     if elem_is_f64 == 1 {
                         EXPR_IS_F64 = 1
                         EXPR_TY = 2
+                        EXPR_TY_HASH = 0
+                    } else if r29_elem_ty == 6 {
+                        // 1-field struct in this tuple slot — preserve nominal type.
+                        EXPR_IS_F64 = 0
+                        EXPR_TY = 6
+                        EXPR_TY_HASH = r29_elem_hash
                     } else {
                         EXPR_IS_F64 = 0
                         EXPR_TY = 1
+                        EXPR_TY_HASH = 0
                     }
-                    EXPR_TY_HASH = 0
                 } else {
```

**Surgery scope:** 1 function, 1 branch, +19 lines net. No change to
codegen (the `mov rax, [rax]` deref is correct for both i64 and 1-field
struct — they share the same in-memory representation). Only the
EXPR_TY/EXPR_TY_HASH propagation gains a tup_cache consultation.

**Why narrow gating (`r29_elem_ty == 6` only):**
- Refinement types (k=11) and Knowledge<T> shouldn't appear here as 1-slot
  tuple elements in practice (refinement of i64 is k=11 with hash carrying
  predicate, not a struct).
- ABI/struct enums (k=4) for which 1-slot enum element appears in tuple
  return would also benefit, but the umbrella didn't exercise it; widening
  to `(r29_elem_ty == 6 || r29_elem_ty == 4)` is reserved for a future
  audit if a real case surfaces.

---

## §3 — Acceptance

| § | Criterion | Result |
|---|---|---|
| 7.1 | `lean_single_fixed_point_gate.sh` PASS | ✓ stage1==stage2==stage3 md5=`c7ea6a4d…` (was `ffdc0fd5…`) |
| 7.2 | `native_v2_cpu_compiler_umbrella_gate.sh` 12/12 | ✓ |
| 7.3 | `standalone_repro.sio` typechecks (was FAIL) | ✓ also runs, prints 44 |
| 7.4 | `imported_repro_main.sio` typechecks (was FAIL) | ✓ also runs, prints 44 |
| 7.5 | R.2.4 distributions oracle 1024/1024 bit-exact | ✓ via `dst_pcg64_next_i64` |
| 7.6 | R.2.5 rng self-oracle 1024/1024 bit-exact | ✓ via reverted rng.sio |
| 7.7 | rng.sio Phase D wiring lands; R.2.7+R.2.8 → RESOLVED | **PARTIAL** (see §4) |
| 7.8 | Net LOC delta in `stdlib/random/rng.sio`: ≤ −40 | **0** (rng.sio reverted again) |

§7.7 and §7.8 PARTIAL → R.2.9 closes **RESOLVED-PARTIAL** per
DISPATCH §7 fallback. The compiler patch is fully validated and
load-bearing; the rng.sio aspirational refactor remains blocked on a
separate, deeper compiler issue.

---

## §4 — rng.sio Phase D re-attempt: blocked on transitive alias

After applying R.2.9's patch and rebuilding souc, I re-tried R.2.8's
deferred rng.sio Phase D wiring (`type Pcg64 = PcgState` + one-liner
wrappers). Now the failure is **different**:

```
error: assignment type mismatch at line 14    (in rng_oracle_gen.sio: `rng = r.0`)
error: assignment type mismatch at line 105   (in rng.sio's pcg64_bounded: `current = r.0`)
error: Type mismatch in call argument at line 314  (in rng.sio's rng_new: passes step1.0)
error: assignment type mismatch at line 114   (in rng.sio's xoshiro256_new)
```

Key control: an isolated `/tmp/r29_4field.sio` that does
`use stdlib::random::pcg64_core::{PcgState, pcg_step}; type Pcg64 = PcgState; var current = PcgState{...}; current = pcg_step(current).0`
**passes**. The only structural difference from the failing
oracle-gen path is the transitive alias chain:

- **Passes:** consumer imports `PcgState` directly from `pcg64_core` and
  aliases it locally.
- **Fails:** consumer imports `Pcg64` from `rng`, which re-aliases
  `PcgState` from `pcg64_core` — a two-hop alias chain crossing module
  boundaries.

This is a separate compiler limitation: alias resolution doesn't fully
propagate through a re-aliased import. R.2.7 / R.2.8's Pass 1c.5 fix
handled single-hop aliases inside one module; transitive
`A_module::A_alias = B_module::B_alias = struct C` chains are
unhandled. Out of R.2.9's narrow var/tuple-assign scope.

**Suggested next-audit scope (R.3.x?):** decouple transitive alias
resolution from module-import order; either (a) flatten alias chains
at REF_TYPE_DATA registration time, or (b) make field-access /
assignment-mismatch sites recursively unwrap aliases via
REF_TYPE_DATA chain.

---

## §5 — Deliverables

1. `self-hosted/compiler/lean_single.sio` — patch at lines 13597–13630.
2. `bin/souc-linux-x86_64` — rebuilt, md5=`c7ea6a4d0f5d25d88d161bec6a2b6c9a`, size=2152829 bytes (+415).
3. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/DISPATCH.md` — already shipped in `84278435c`.
4. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/SYNTHESIS.md` — this file.
5. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/reference/standalone_repro.sio` — Phase A primary repro.
6. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/reference/imported_repro_{main,lib}.sio` — Phase A imported variant (alias path fixed during validation).
7. `stdlib/random/rng.sio` — **unchanged** (revert of attempted Phase D).
8. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — unchanged (RESOLVED-PARTIAL stands).
9. `docs/audit/r2_8_alias_deep_resolve/SYNTHESIS.md` — unchanged (RESOLVED-PARTIAL stands).

---

## §6 — Notes

- **The compiler patch is unconditionally a fix.** Independent of
  whether R.2.7 Path A ever fully lands for rng.sio, the
  var/tuple-`.0`-assignment-on-1-field-struct pattern is now correct
  for every consumer. Future stdlib refactors that surface a 1-field
  state struct (e.g. SplitMix64, simple counters, monad-like wrappers)
  will not trip this fragility again.
- **No bootstrap chain touched.** R.2.9 scope (lean_single.sio +
  bin/souc rebuild) held cleanly.
- **Diagnosis cost was the win.** The variant matrix (10 + 3 probes)
  converged on the exact code site within ~30 min. The patch itself
  was ~10 lines, mostly mechanical tup_cache lookup.

**END OF SYNTHESIS.**
