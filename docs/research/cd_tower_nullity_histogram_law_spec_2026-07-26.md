<!-- docs:meta
topic_id: repo.docs.research.cd-tower-nullity-histogram-law-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-nullity-histogram-law-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The Cayley–Dickson nullity histogram law — a closed form for the zero-divisor kernel spectrum

**Date:** 2026-07-26
**Status:** `HYPOTHESIS` → `EXECUTABLE` (C_GREEN reached 2026-07-26, 7/7 clauses)
**Parents:** `docs/research/routon_zd_spec_2026-07-26.md` (L7 nullity spectrum, open multiplicity question), `scripts/research/routon_zd_contract.py` (exact 2-cycle criterion)
**Harness:** `scripts/research/cd_tower_nullity_histogram_law_contract.py`
**Gate:** `scripts/ci/cd_tower_nullity_histogram_law_gate.sh`

---

## 1. What this is

The routon contract tabulated the exact nullity of every canonical zero divisor `a = e_i ± e_j` at levels 4–7 of the Cayley–Dickson tower and reported the level-7 histogram

```
{4:684, 8:504, 12:504, 16:336, 20:336, 24:336, 28:336, 32:336,
 36:336, 40:336, 44:336, 48:504, 52:504, 56:684, 60:870}
```

explicitly leaving the multiplicities `684, 504, 336, …, 870` as "data without a claimed counting law" (routon spec §4, "Not an explanation of the odd-part distribution").

This document states the law that explains them, and verifies it **one level beyond all previously computed data** (level 8, 256 dimensions). The law has four parts:

1. a new invariant of zero-divisor fibers (the **fiber type** `τ`);
2. an exact **label count law** `A(m, τ) = 7 + v₂(τ)`;
3. an exact **recursion** generating every fiber's nullity histogram;
4. the resulting **aggregate generation law**: at level `b` the histogram consists of exactly `2^(b−k)` distinct nullity values of multiplicity `6·2^(b−k)·f(k)` each, `k = 4..b`, where `f(k) = (2k−5)·2^(k−3) + 1`.

The L7 multiplicities are thereby explained:

```
336 = 6·2³·f(4) = 48·7      (8 values)
504 = 6·2²·f(5) = 24·21     (4 values)
684 = 6·2·f(6)  = 12·57     (2 values)
870 = 6·f(7)    = 6·145     (1 value, the maximum nullity 60)
```

To our knowledge the nullity (kernel-dimension) spectrum of canonical Cayley–Dickson zero divisors, its fiber structure, and this counting law have not appeared in the literature; the level-8 verification (`Z(8) = 59772`, full histogram, all 243 fibers) is a new computation that was a **prediction**, not a fit.

---

## 2. Setup and notation

Recall from the routon spec: a canonical zero-divisor pair is `(i, j)`, `1 ≤ i < j < 2^b`, with `L_{e_i ± e_j}` singular; the xor-label `ℓ = i ⊕ j` groups pairs into fibers; a label supports zero divisors iff `ℓ ≥ 8` and `ℓ` is not a power of two; its **birth level** is `m = ⌊log2 ℓ⌋ + 1`, and we write

```
ℓ = 2^(m−1) + r,   1 ≤ r ≤ 2^(m−1) − 1.
```

The nullity of `L_a` for an `m`-born pair at level `b` is `2^(b−m+2)·t` with `t` odd, `1 ≤ t ≤ 2^(m−3) − 1` (routon spec, C9). We reparametrize the odd part by

```
u = (t − 1)/2 ∈ [0, 2^(m−4) − 1],   nullity at birth level = 8u + 4.
```

`lowbit(r) = r & (−r) = 2^(v₂(r))` is the lowest set bit of `r`; `v₂` is the 2-adic valuation.

---

## 3. The theorem

**Scope.** Theorems 1–4 below are stated as exact laws of the Cayley–Dickson tower at every level; their *evidence* is exhaustive computational verification (exact integer arithmetic, no floating point) over all fibers and all pairs at levels 4–8, together with the L8 out-of-sample prediction test (Theorem 5). The induction proving them for all `b` is sketched in §5 and remains open; where we rely on unproven extension beyond level 8 we say so explicitly.

### Theorem 1 (Fiber type invariant)

Define the **fiber type** of a label `ℓ = 2^(m−1) + r` by

```
τ(ℓ) = (r − lowbit(r)) >> 3  ∈  [0, 2^(m−4) − 1].
```

The nullity histogram of the fiber of `ℓ` at every level `b ≥ m` depends only on `(m, τ(ℓ), b)`. Moreover lifting is exact: the histogram at level `b` equals the birth-level histogram with nullities **and** multiplicities scaled by `2^(b−m)`.

*Verification:* exhaustive over all `7+22+53+116+243 = 441` fibers at levels `b = 4..8` (contract C1, C3).

Note what `τ` says: the histogram is invariant under (a) moving the lowest set bit of `r`, and (b) changing the three bits of `r` immediately above its lowest set bit. Only the remaining high bits, shifted down, matter.

### Theorem 2 (Label count law)

The number of labels born at level `m` of fiber type `τ` is

```
A(m, τ) = m + 3           (τ = 0)
A(m, τ) = 7 + v₂(τ)       (1 ≤ τ ≤ 2^(m−4) − 1)
```

— in particular **independent of `m`** for `τ ≥ 1`. Consistency check (exact, see contract C7 context):
`Σ_τ A(m, τ) = (m+3) + Σ_{τ=1}^{2^(m−4)−1} (7 + v₂(τ)) = (m+3) + 7·(2^(m−4)−1) + (2^(m−4) − (m−4) − 1) = 2^(m−1) − 1`, the number of labels born at `m`; here we used the classical identity `Σ_{τ=1}^{2^p−1} v₂(τ) = 2^p − p − 1`.

*Verification:* exhaustive at `m = 4..8` (contract C2).

### Theorem 3 (Fiber histogram recursion)

Let `c_m(τ, u)` be the number of index pairs in a type-`τ` fiber at birth level `m` whose nullity is `8u + 4`, and write `M = 2^(m−4) − 1`, `M′ = 2^(m−5) − 1`. Then `c` is generated by

```
base (m = 4):                      c₄(0, 0) = 6
old type (τ ≤ M′):                 c_m(τ, 2u+1) = 2·c_{m−1}(τ, u)   for u < M′
                                   c_m(τ, M)    = 2·c_{m−1}(τ, M′) + 2
new type (τ = M′+1 + τ′):          c_m(τ, (M−1) − 2u) = 2·c_{m−1}(τ′, u)
                                   c_m(τ, M)    = 2
```

Every step preserves the fiber size `Σ_u c_m(τ, u) = 2^(m−1) − 2` exactly. Combined with Theorem 1's lifting law, this generates the nullity histogram of **every** canonical zero-divisor fiber at **every** level.

*Verification:* exhaustive equality with the exact 2-cycle scan for all 441 fibers at `b = 4..8` (contract C3).

*Proof status:* the recursion is a computationally verified exact law at levels `4..8`; a fully hand-written induction for all `b` is sketched in §5 but is not yet complete. We state Theorems 1–4 as theorems of the scan (exact integer arithmetic, no floating point) at the verified levels, and as the natural conjectural extension for all `b`.

### Theorem 4 (Aggregate generation law — the headline)

At level `b ≥ 4`, the multiset of nullities of canonical zero-divisor index pairs consists of exactly `2^(b−k)` distinct nullity values, each occurring with multiplicity

```
mult_b(k) = 6 · 2^(b−k) · f(k),      f(k) = (2k − 5)·2^(k−3) + 1,
```

for each **generation** `k = 4..b`. Since a nullity's 2-adic valuation determines its birth class (`m = b + 2 − v₂(ν)`), generations never collide.

Consequences:

- **Distinct nullities at level b:** `Σ_k 2^(b−k) = 2^(b−3) − 1`.
- **Extremal multiplicities:** the maximum nullity `2^(b−1) − 4` has multiplicity `6·f(b)`; the minimum nullity `4` has multiplicity `12·f(b−1)` (for `b ≥ 5`; at `b = 4` min = max).
- **Census identity (Corollary).** The histogram law sums to the growth law of the routon contract:
  `Z(b) = 12 · Σ_{k=4}^{b} 4^(b−k) · f(k)`, since `Z(b) − 4·Z(b−1) = 12·f(b)` and `Z(4) = 84 = 12·f(4)` (exact induction, contract C7). The multiplicity law is therefore a strict refinement of the census law `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4` — it *contains* the census as its total mass (not an independent derivation: both sides descend from the same fiber recursion).

*Verification:* value-by-value equality of the law-generated histogram with the exact scan at `b = 4..8`, plus the multiset-of-multiplicities form (contract C4, C6, C7).

The generation sequence `f(k) = 7, 21, 57, 145, 353, 833, 1921, …` (`k = 4, 5, 6, …`) is the new numerical invariant of the tower; `mult_b(k) = 6·2^(b−k)·f(k)` explains every multiplicity at every computed level.

### Theorem 5 (Level-8 falsification test — survived)

The law's level-8 predictions, made before the L8 scan was run:

```
Z(8) = 59772 triples (29886 index pairs)     F(8) = 243 fibers
31 distinct nullities, max = 124 = 2^7 − 4
multiplicity multiset {672:16, 1008:8, 1368:4, 1740:2, 2118:1}
```

The exact scan at `b = 8` confirms all of them, and the per-fiber recursion (Theorem 3) holds for all 243 L8 fibers, including the 127 native ones spanning the 16 fiber types `τ = 0..15` (contract C5, C1–C4). The growth law `Z(b)` thereby survives its second consecutive out-of-sample test, and the histogram law its first. The same L8 census was independently confirmed by a parallel lane with a separate C implementation and an exact GF(65521) rank audit of all 64770 pair-sign matrices (`docs/research/l8_zd_census_benchmark_spec_2026-07-26.md`, 0 mismatches) — which explicitly left the multiplicity counting law as the open question this document answers.

---

## 4. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **C1_TYPE_INVARIANT** | Fiber histograms depend only on `(m, τ, b)`; lifting scales nullities and multiplicities by exactly `2^(b−m)`. | All 441 fibers at b=4..8. |
| **C2_LABEL_COUNT_LAW** | `A(m, τ) = m+3` (τ=0), `7 + v₂(τ)` (τ≥1). | Exhaustive at m=4..8. |
| **C3_FIBER_HISTOGRAM_RECURSION** | The recursion of Theorem 3 reproduces every fiber histogram exactly. | All 441 fibers at b=4..8. |
| **C4_AGGREGATE_GENERATION_LAW** | Full histogram equality with the law, value-by-value and as multiplicity multiset `{6·2^(b−k)f(k) : 2^(b−k)}`. | b=4..8. |
| **C5_L8_FALSIFICATION** | `Z(8)=59772`, `F(8)=243`, 31 distinct nullities, max 124. | Exact L8 scan. |
| **C6_EXTREMAL_COROLLARIES** | max-nullity mult `6·f(b)`; min-nullity mult `12·f(b−1)` (b≥5); distinct count `2^(b−3)−1`. | b=4..8. |
| **C7_CENSUS_IDENTITY** | `12·Σ 4^(b−k)f(k) = Z(b)` as an integer identity. | b=4..16 (closed forms). |

---

## 5. Proof sketch and what remains open

The exact 2-cycle criterion (routon spec §2) reduces everything to the sign cocycle: for label `ℓ` and pair `(a, a⊕ℓ)`, `nullity = ½·#{k : p(k) = +1}` with `p(k) = S[a,k]·S[a⊕ℓ,k]·S[a,k⊕ℓ]·S[a⊕ℓ,k⊕ℓ]`. The Cayley–Dickson sign table is nested (`S_b` restricted to indices `< 2^(b−1)` is `S_{b−1}`), so a pair lifted from level `b` to `b+1` has its `p`-pattern duplicated, which gives the exact kernel doubling (already exploited in the routon contract) and the `2^(b−m)` lifting law of Theorem 1. The two extra pairs each fiber gains per level (`2^(m−1)−2 → 2^m−2` is `2x+2`) appear always to land in the maximum-nullity class — this is the `+2` in the recursion and the source of the `2·c + 2` top-class rule (we have not yet isolated a coordinate description of these two pairs; that identification is part of the open induction).

A complete induction for all `b` requires controlling how the `p`-pattern of the *new* native labels at level `m` reflects that of level `m−1` (the `(M−1) − 2u` reflection in Theorem 3). We have verified this reflection exhaustively through level 8 but do not yet have the closed cocycle computation that proves it in general. This is the main remaining open problem, together with:

- an intrinsic (coordinate-free) meaning of the fiber type `τ(r) = (r − lowbit(r)) >> 3` — why do the lowest set bit and the three bits above it drop out?
- the assignment of *individual* nullity values to generation `k` (the multiset law of Theorem 4 is exact; the per-value generation map follows the recursion but has no closed form yet);
- extension from canonical 2-unit sums to the full zero-divisor locus.

## 6. What this is NOT

- **Not a hand proof for all levels.** Theorems 1–4 are exact laws verified exhaustively at levels 4–8 (all fibers, all pairs, integer arithmetic only); the all-`b` induction is sketched, not complete. Level 9 (512-dim, `Z(9) = 249,084` predicted) is the next falsification target.
- **Not the full ZD variety.** Only canonical pairs `e_i ± e_j` are counted.
- **Not a G₂ or automorphism statement.**
- **Not a clinical claim.**

## 7. Reproduce

```bash
python3 scripts/research/cd_tower_nullity_histogram_law_contract.py
# expect: C1..C7 PASS, CD_HISTOGRAM_LAW_VERDICT C_GREEN   (~1 s)

bash scripts/ci/cd_tower_nullity_histogram_law_gate.sh
# expect: CD_HISTOGRAM_LAW_GATE_OK
```

Pure Python + NumPy, self-contained; exact integer arithmetic throughout (the 2-cycle criterion); no SVD, no floating point. The full b=4..8 scan runs in well under a second.

## 8. AI disclosure

Spec and harness drafted under human direction (2026-07-26). No clinical content. GAIDeT-ICMJE 2025.
