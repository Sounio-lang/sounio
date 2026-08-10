<!-- docs:meta
topic_id: repo.docs.research.l9-zd-census-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.l9-zd-census-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Level-9 ZD census — exact census, histogram-law confirmation, exact verification (512 dimensions)

**Date:** 2026-07-26
**Status:** `EXECUTABLE` (gate green 2026-07-26)
**Parents:** `docs/research/l8_zd_census_benchmark_spec_2026-07-26.md` (L8 census + fast harness, Z(9) = 249084 prediction), `docs/research/nullity_histogram_law_spec_2026-07-26.md` (solved multiplicity law; level 9 was its live falsification target), `scripts/research/routon_zd_contract.py` (sign-table reference)
**Harness:** `scripts/research/l9_zd_census_fast.c`
**Gate:** `scripts/ci/l9_zd_census_gate.sh`
**Falsifiers:** `docs/research/l9_zd_census_falsifiers_2026-07-26.md`

---

## 1. What this is

The level-9 Cayley–Dickson algebra (512 dimensions) canonical zero-divisor census, computed exactly and verified exactly. Two laws survive out-of-sample tests at this level:

1. **Growth law, third out-of-sample confirmation.** `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4`, previously exact at `b = 4..8`, predicts `Z(9) = 249084` canonical zero divisors (index pairs `1 ≤ i < j < 512`, ×2 signs). The level-9 scan confirms the prediction **exactly**: 124542 index pairs × 2 signs = 249084.
2. **Nullity-histogram counting law, second out-of-sample confirmation.** The solved multiplicity law `μ_s(b) = 2^(b−s+1)·c⁰(s−1)`, `c⁰(b) = 3·(2b−3)·2^(b−2) + 3` (each `μ_s` attained by exactly `2^(b−s)` distinct nullity values, `s = 4..b`) predicts the level-9 multiplicity multiset

   ```
   1344×32, 2016×16, 2736×8, 3480×4, 4236×2, 4998×1
   ```

   (63 distinct nullity values, total mass 124542). The scan reproduces it **exactly**, closing the live falsification target of `docs/research/nullity_histogram_law_falsifiers_2026-07-26.md`.

The census is audited by an independent exact verifier: dense GF(65521) Gaussian elimination of `M(sgn) = I + sgn·Q` for **all 260610 pair-signs** (130305 candidate pairs × 2 signs), requiring `512 − rank = census nullity` in every case. Result: **0 mismatches**.

---

## 2. Methods

Identical to the L8 harness (§2 of the L8 spec), with `BITS = 9`. Both methods share only the Cayley–Dickson sign table `S[i, j]`; the C table is bit-identical to the NumPy reference: FNV-1a-64 over the 262144 int8 entries is `342532a0b57edf83` in both implementations (checked by the gate).

- **Method 1 — census** (audited 2-cycle criterion): for `l = i ⊕ j`, `p(k) = S[i,k]·S[j,k]·S[i,k⊕l]·S[j,k⊕l]`; `(i, j)` is a canonical ZD pair iff some `p(k) = +1`, and `nullity(L_a) = #{k : p(k) = +1}/2` exactly, for both signs simultaneously.
- **Method 2 — exact verification** (generic linear algebra): GF(65521) rank of `M(sgn)` by Gaussian elimination with partial pivoting. Exactness argument unchanged from L8: `M(sgn)` decomposes into `2×2` blocks `[[1, sgn·q'], [sgn·q, 1]]` whose rank (1 iff `q·q' = +1`, else 2) is the same over any field of characteristic ≠ 2, so the modular rank equals the Q rank; Method 2 does not use the decomposition — it is a dense exact computation and hence an independent audit.

New at this level: the harness additionally checks the observed multiplicity multiset of the nullity histogram against the counting law's level-9 prediction, computed from the formula (not hardcoded): for each `s = 4..9`, `μ_s(9) = 2^(9−s+1)·c⁰(s−1)` must be attained by exactly `2^(9−s)` distinct nullity values, and no other multiplicity may occur.

### Contract assertions (all exact, all green)

| Clause | Statement | Result |
|---|---|---|
| census | 124542 index pairs, 249084 triples `= Z(9)` law | PASS |
| histogram law | multiplicity multiset exactly `{1344×32, 2016×16, 2736×8, 3480×4, 4236×2, 4998×1}`; 63 distinct nullities; mass 124542 | PASS |
| fibers | 498 `= F(9) = 2^9 − 9 − 5` fibers; labels exactly `{l ∈ [8,512) : l not a power of 2}`; every `m`-born fiber has `2^9 − 2^(9−m+2)` triples | PASS |
| nullity law | every `m`-born pair has nullity `2^(9−m+2)·t`, `t` odd, `1 ≤ t ≤ 2^(m−3)−1`; every allowed value occurs in every birth class `m = 4..9` | PASS |
| max nullity | `252 = 2^8 − 4` (native-erasure maximum) | PASS |
| verification | GF(65521) rank = census nullity on all 260610 pair-signs, 0 mismatches | PASS |

---

## 3. Cost

Task: full level-9 census **with exact verification** (build sign table, enumerate all canonical ZD pairs with exact nullities, check all laws, independently verify every pair-sign). Hardware: this dev container (single thread, x86_64); compiler `cc -O2` (GCC 13.3).

| phase | L8 (dim 256) | L9 (dim 512) | scaling |
|---|---|---|---|
| sign table | 0.003 s | 0.013 s | ~4× |
| exact census scan | 0.010 s | 0.109 s | ~11× |
| exact verification | 6.1 s (64770 pair-signs) | 153.9 s (260610 pair-signs) | ~25× |
| **total** | **6.1 s** | **154.0 s** | ~25× |

The L8 spec predicted ~16× for verification (~100 s); the measured 25× comes from the per-matrix cost growing super-quadratically in the dimension (94 µs ≈ 1.4·N² ns at N=256 → 590 µs ≈ 2.25·N² ns at N=512): the 512 KB elimination buffer no longer fits L2. The census itself remains essentially free (0.1 s). Projection for level 10 on this hardware: ~4× matrices × ~4–6× per-matrix cost ≈ 40–100 min, still feasible but worth a sparse elimination first.

---

## 4. Level-9 data

- Census: 124542 index pairs, 249084 triples. Density `249084 / (511·510) = 0.9558` (L8: 0.9228 — still increasing).
- Fibers: 498; stratification (triples): `m=4`: 7 labels × 384; `m=5`: 15 × 448; `m=6`: 31 × 480; `m=7`: 63 × 496; `m=8`: 127 × 504; `m=9`: 255 × 508. Sum = 249084. ✓
- Nullity histogram (index pairs), 63 distinct values `4..252` step 4:

```
  4:4236   8:3480  12:3480  16:2736  20:2736  24:2736  28:2736
 32:2016  36:2016  40:2016  44:2016  48:2016  52:2016  56:2016  60:2016
 64..188 step 4: 1344 each (32 values)
192:2016 196:2016 200:2016 204:2016 208:2016 212:2016 216:2016 220:2016
224:2736 228:2736 232:2736 236:2736 240:3480 244:3480 248:4236 252:4998
```

Sum check: `4236 + 2·3480 + 4·2736 + 8·2016 + 32·1344 + 8·2016 + 4·2736 + 2·3480 + 4236 + 4998 = 124542`. ✓

Multiplicity structure, as predicted by the counting law:

| terminal `s` | `μ_s(9)` | # values | nullities |
|---|---|---|---|
| 4 | `2^6·c⁰(3) = 64·21 = 1344` | 32 | 64, 68, …, 188 |
| 5 | `2^5·c⁰(4) = 32·63 = 2016` | 16 | 32..60 and 192..220 |
| 6 | `2^4·c⁰(5) = 16·171 = 2736` | 8 | 16..28 and 224..236 |
| 7 | `2^3·c⁰(6) = 8·435 = 3480` | 4 | 8, 12, 240, 244 |
| 8 | `2^2·c⁰(7) = 4·1059 = 4236` | 2 | 4, 248 |
| 9 | `2·c⁰(8) = 2·2499 = 4998` | 1 | 252 |

Observations (the L8 doubling pattern extends, with the same characterized exception):

- The interior plateau `64..188` sits at `1344 = 2·672`, exactly twice the L8 plateau; the edge multiplicities also double: `2016 = 2·1008`, `2736 = 2·1368`, `3480 = 2·1740`. The doubling holds for every terminal level `s ≤ 7`: `μ_s(9) = 2·μ_s(8)`.
- The extreme multiplicities again outgrow doubling: `μ_8(9) = 4236 ≠ 2·2118 = 2·μ_8(8)` and `μ_9(9) = 4998` is a new terminal level with no L8 counterpart. As explained in the nullity-law spec, the extremes have terminal levels `s = b−1` and `s = b` and track `c⁰(b−2)`, `c⁰(b−1)` (1059→2499) rather than doubling.
- The histogram is symmetric on the doubled-L8 part (`8:3480 ↔ 244:3480`, `32:2016 ↔ 220:2016`, etc.) and asymmetric at the extremes (`hist[4] = 4236 ≠ 4998 = hist[252]`).

---

## 5. What this is NOT

- **Not a proof.** Both laws remain inductive, now exact at six levels (`b = 4..9`); the L9 census is a passed falsification test, not a theorem. Level 10 (1024-dim, prediction `Z(10) = 1019388`, multiplicity multiset `2688×64, 4032×32, 5472×16, 6960×8, 8472×4, 9996×2, 11526×1`) is the next target — see the falsifiers doc. The same code path does it by changing `BITS`, but the verification phase needs a sparse elimination to stay practical (§3).
- **Not the full ZD locus.** Only canonical 2-unit sums `e_i ± e_j`.
- **Not a clinical claim.**

---

## 6. Reproduce

```bash
# full contract (compile + run + sign-table cross-hash vs NumPy; ~3 min)
bash scripts/ci/l9_zd_census_gate.sh
# expect: L9_ZD_FAST_VERDICT PASS, L9_ZD_CENSUS_GATE_OK

# fast census-only mode (seconds; skips the GF(65521) verification phase)
L9_GATE_FAST=1 bash scripts/ci/l9_zd_census_gate.sh
# expect: L9_ZD_FAST_VERDICT PASS (census-only; ...), L9_ZD_CENSUS_GATE_OK
```

---

## 7. AI disclosure

Spec and harness drafted under human direction (2026-07-26). No clinical content. GAIDeT-ICMJE 2025.
