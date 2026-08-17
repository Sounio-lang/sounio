<!-- docs:meta
topic_id: repo.docs.research.l9-nullity-histogram-law-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.l9-nullity-histogram-law-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Level-9 nullity-histogram law verification — out-of-sample confirmation at 512 dimensions

**Date:** 2026-07-26
**Status:** `EXECUTABLE` (gate green 2026-07-26)
**Parents:** `docs/research/nullity_histogram_law_spec_2026-07-26.md` (the counting law; level 9 named there as the next falsification target), `docs/research/l8_zd_census_benchmark_spec_2026-07-26.md` (implementation template)
**Harness:** `scripts/research/l9_nullity_histogram_law_contract.c`
**Gate:** `scripts/ci/l9_nullity_histogram_law_gate.sh`

---

## 1. What this is

The nullity-histogram counting law of the parent spec — multiplicity `μ_s(b) = 2^(b−s+1)·c⁰(s−1)` attained by exactly `2^(b−s)` distinct nullity values per terminal level `s ∈ {4,…,b}`, with `c⁰(b) = 3·(2b−3)·2^(b−2) + 3` — was derived at levels 4–7 and confirmed out of sample at level 8 against a census tabulated *before* the law was stated. This contract runs the **second out-of-sample test**, at level 9 (512-dimensional Cayley–Dickson algebra, 130305 candidate index pairs), against the prediction recorded in the parent spec:

```
multiplicities  1344  2016  2736  3480  4236  4998   (= 2^(10-s)·c⁰(s−1), s = 4..9)
distinct values   32    16     8     4     2     1    (= 2^(9−s))
total index pairs 124542 = Z(9)/2,   Z(9) = 249084
```

**Result: the law matches the exact level-9 scan with zero deviations — full histogram, per birth class, and terminal multiplicity structure.** No new structure appeared; the histogram has exactly the predicted `b−3 = 6` distinct multiplicities over 63 distinct nullity values (`4..252` step 4), and the scan total equals the census-law prediction `Z(9) = 4^9 − 26·2^9 + 2^8 − 4 = 249084` exactly. The level-9 histogram (index pairs):

```
  4:4236   8:3480  12:3480  16:2736  20:2736  24:2736  28:2736
 32:2016  36:2016  40:2016  44:2016  48:2016  52:2016  56:2016  60:2016
 64..188 step 4: 1344 each (32 values)
192:2016 196:2016 200:2016 204:2016 208:2016 212:2016 216:2016 220:2016
224:2736 228:2736 232:2736 236:2736 240:3480 244:3480 248:4236 252:4998
```

Sum check: `4236 + 2·3480 + 4·2736 + 8·2016 + 32·1344 + 8·2016 + 4·2736 + 2·3480 + 4236 + 4998 = 124542`. ✓

The lemmas underpinning the derivation (L1 ε-identity, L2 left = right nullity, L3 native recursion `nullity = 2^(m−1) − 2ν − 4`, L4 doubling) were previously verified exhaustively only at `b ≤ 7`. This contract verifies all four **at level 9** (L3/L4 against an independent exact level-8 census): 0 violations over 130305 pairs (L1, L2), 64770 native pairs (L3), and 32385 embedded + 32385 high pairs (L4).

## 2. Methods

Single C translation unit (`scripts/research/l9_nullity_histogram_law_contract.c`), extending the audited L8 fast-census template (`scripts/research/l8_zd_census_fast.c`) with `BITS = 9`:

1. **Exact census (2-cycle criterion).** For `1 ≤ i < j < 512`, `ℓ = i ⊕ j`, `p(k) = S[i,k]·S[j,k]·S[i,k⊕ℓ]·S[j,k⊕ℓ]`; `nullity = #{k : p(k) = +1}/2`, exact integer arithmetic — the audited Method 1 of all prior contracts. An independent exact level-8 census (`k ∈ [0,256)`) is computed for the L3/L4 checks.
2. **Law evaluation.** The descent law `N(m, 9, t) = 2^(9−m+V+1)·c⁰(m_s−1)` (2-adic descent of the odd part `t`) is evaluated in C and compared against the scan **per birth class** (`m = 4..9`, every odd `t`) and in total; the terminal multiplicity structure `{μ_s : 2^(9−s)}` is compared against the scan's multiplicity-of-multiplicity histogram.
3. **Independent GF(65521)-rank audit.** Dense Gaussian elimination of `M(sgn) = I + sgn·Q` over `GF(65521)` — generic exact linear algebra, an exact Q-rank by the 2×2-block argument of the L8 contract (characteristic ≠ 2) — on a deterministic stride-32 subsample of the candidate pairs (4073 pairs × 2 signs = 8146 pair-signs, both signs, 0 mismatches, ~5 s). `L9_FULL_VERIFY=1` runs the complete audit of all 260610 pair-signs (0 mismatches, 144 s — run once for this spec).
4. **Cross-implementation audit.** FNV-1a-64 of the C sign table equals the NumPy reference (`routon_zd_contract.get_sign_matrix(9)`): `342532a0b57edf83`, checked by the gate.

### Contract clauses (all exact, all green)

| Clause | Statement | Result |
|---|---|---|
| census | 124542 index pairs, 249084 triples `= Z(9)` law | PASS |
| lemmas | L1, L2 (130305 pairs), L3 (64770 native), L4 (32385 embedded + 32385 high) at level 9: 0 violations | PASS |
| fibers | 498 `= F(9) = 2^9 − 9 − 5` fibers, size law, odd-part law with completeness (every allowed `t` occurs in every class `m = 4..9`), max nullity `252 = 2^8 − 4` | PASS |
| law histogram | descent law `N(m,9,t)` reproduces the scan per class and in total (63 distinct values) | PASS |
| terminal structure | exactly 6 distinct multiplicities `{1344:32, 2016:16, 2736:8, 3480:4, 4236:2, 4998:1}` | PASS |
| GF(65521) audit | rank = census nullity on 8146 pair-signs (subsample) / 260610 (full mode), 0 mismatches | PASS |

## 3. What this is NOT

- **Not a proof.** The law remains an inductive law with a first-principles derivation (Lemmas L1–L4 of the parent spec, themselves proved from the block recursion) now confirmed out of sample at two successive levels (8 and 9). Level 10 (1024-dim, predicted `Z(10) = 4^10 − 29·2^10 + 2^9 − 4 = 1019388`, histogram multiplicities `μ_s(10) = 2^(11−s)·c⁰(s−1)` = 2688, 4032, 5472, 6960, 8472, 9996, 11526 attained by 64, 32, 16, 8, 4, 2, 1 values) is the next falsification target; the same code path does it by changing `BITS`, with the GF audit subsample keeping runtime near-linear in the pair count.
- **Not the full zero-divisor locus.** Only canonical 2-unit sums `e_i ± e_j`.
- **Not a G₂ or automorphism statement. Not a clinical claim.**

## 4. Reproduce

```bash
bash scripts/ci/l9_nullity_histogram_law_gate.sh
# expect: L9_NULLITY_LAW_VERDICT PASS, L9_NULLITY_LAW_GATE_OK   (~7 s)

L9_FULL_VERIFY=1 bash scripts/ci/l9_nullity_histogram_law_gate.sh
# full GF(65521) audit of all 260610 pair-signs (~2.5 min)
```

## 5. AI disclosure

Spec and harness drafted under human direction (2026-07-26). No clinical content. GAIDeT-ICMJE 2025.
