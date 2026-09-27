<!-- docs:meta
topic_id: repo.docs.research.routon-zd-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.routon-zd-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Routon zero-divisor structure — level-7 annihilation geometry (128 dimensions)

**Date:** 2026-07-26  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (C_GREEN reached 2026-07-26)  
**Parents:** `scripts/research/chingon_zd_contract.py` (L6 fibers), `docs/research/chingon_zd_spec_2026-07-25.md`, `docs/research/trigintaduonion_zd_spec_2026-07-25.md`, `docs/research/g2_zd_fibers_spec_2026-07-25.md`, `examples/routon_projective_measurement.sio` (L7 projective measurement)  
**Harness:** `scripts/research/routon_zd_contract.py`  
**Gate:** `scripts/ci/routon_zd_gate.sh`

---

## 1. What this is

The Cayley–Dickson tower continues past the chingons to the routons (level 7, dimension 128; repo naming per `examples/routon_projective_measurement.sio`). This contract extends the catastrophe scan to level 7 and analyzes the **fiber structure** of the level-7 zero-divisor locus: the canonical zero-divisor census, the decomposition by xor-label, fiber sizes, the defect diagonal, and — novel at this level — the **exact nullity spectrum** of the canonical zero divisors.

Headline result: the growth law conjectured at levels 4–6 predicts `Z(7) = 13884`. The level-7 scan confirms it **exactly**, together with the fiber-count law (`F(7) = 116`), the fiber-size law, and the defect diagonal. The scan additionally produces a structure not visible at levels 4–6 in tabulated form: the nullity of `L_a` for an `m`-born canonical zero divisor at level `b` is exactly `2^(b−m+2)·t` with `t` odd, `1 ≤ t ≤ 2^(m−3) − 1`, and every allowed value occurs in every birth class.

The computation is **novel** (the level-7 fiber structure and nullity spectrum have not been explicitly tabulated in the literature we are aware of).

---

## 2. Mathematical setup

### The doubling tower

```
𝕆 (8) → 𝕊 (16) → 𝕋 (32) → 𝕀 (64) → routons (128)
L3        L4        L5        L6         L7
```

Zero divisors are born at `𝕊`. A **canonical zero divisor** is a 2-unit sum `a = e_i ± e_j` (`1 ≤ i < j < 2^b`, `e_0 = 1` the real unit) whose left multiplication matrix `L_a` is singular. Each canonical pair carries an **xor-label** `ℓ = i ⊕ j`; pairs sharing a label form a **fiber**.

### The exact 2-cycle criterion (method, new at this level)

For `a = e_i + sgn·e_j`, the left-multiplication matrix is `L_a = A + sgn·B` with `A`, `B` signed permutation matrices (row `k` of `A` has its single nonzero at column `k⊕i`, value `S[i, k⊕i]`, where `S[i, j]` is the Cayley–Dickson sign of `e_i·e_j`; likewise `B` with `j`). Since `A` is orthogonal,

```
det(L_a) = det(A) · det(I + sgn·Q),     Q = A^T B,
```

and `Q` is again a signed permutation matrix whose underlying permutation is the fixed-point-free involution `k ↦ k ⊕ ℓ` with `ℓ = i ⊕ j`. `Q` therefore decomposes into `2^(b−1)` signed 2-cycles, and

```
det(I + sgn·Q) = ∏_cycles (1 − q_k · q_{k⊕ℓ}),
   where q_k · q_{k⊕ℓ} = S[i,k]·S[j,k]·S[i,k⊕ℓ]·S[j,k⊕ℓ] =: p(k) ∈ {+1, −1}.
```

Each factor is `0` or `2`, and `sgn` cancels. Consequences, all exact integer arithmetic with no floating point:

- `(i, j)` is a canonical zero-divisor pair iff `p(k) = +1` for some `k` — for **both** signs simultaneously. Sign duality (observed empirically at levels 4–6) is thereby a theorem of the scan.
- `nullity(L_a) = #{bad cycles} = ½·#{k : p(k) = +1}`, exactly.

Clause C8 cross-checks this criterion against the SVD scan used by the L4/L5/L6 contracts at **every** level `b = 4..7` (full census equality: 42, 294, 1518, 6942 index pairs). The criterion is used as the primary scan because it is exact and `O(2^b)` per pair instead of `O(8^b)` per SVD; the SVD scan remains the audited reference.

### Census and growth law

Computed censuses (contract C1/C2; triples `(i, sgn, j)` per the L4–L6 convention):

| level b | algebra | dim | candidates `(n−1)(n−2)` | canonical ZDs | density |
|---|---|---|---|---|---|
| 4 | `𝕊` | 16 | 210 | 84 | 0.4000 |
| 5 | `𝕋` | 32 | 930 | 588 | 0.6323 |
| 6 | `𝕀` | 64 | 3906 | 3036 | 0.7773 |
| 7 | routons | 128 | 16002 | **13884** | 0.8676 |

All four censuses are reproduced exactly by

```
Z(b) = 4^b − (3b − 1)·2^b + 2^(b−1) − 4        (b ≥ 4)
```

The level-7 value `Z(7) = 13884` was a **prediction** of the law (stated as the falsification target in `chingon_zd_spec_2026-07-25.md` §4 and `chingon_zd_falsifiers_2026-07-25.md` C2), not a fit: the scan confirms it, so the law survives its first out-of-sample test. The competing quadratic-in-`2^b` interpolant `15·4^(b−2) − 27·2^(b−1) + 60`, which also agrees at `b = 4, 5, 6`, predicts `13692` at `b = 7` and is hereby **falsified**. The law remains inductive (now a four-point law derived from the fiber laws below), not a theorem; its level-8 prediction is `Z(8) = 59772`, the next falsification target.

### Fiber structure

A label `ℓ` supports zero divisors at level `b` iff `ℓ ≥ 8` and `ℓ` is not a power of two. Hence

```
F(b) = 2^b − b − 5        →  F(4) = 7,  F(5) = 22,  F(6) = 53,  F(7) = 116
```

Every label `ℓ` has a **birth level** `m = ⌊log2 ℓ⌋ + 1`. At level 7 the 116 fibers stratify by birth level (sizes in triples):

| birth level m | labels | count | size at L7 | total |
|---|---|---|---|---|
| 4 (`𝕊`-born) | 9..15 | 7 | 96 | 672 |
| 5 (`𝕋`-born) | 17..31 | 15 | 112 | 1680 |
| 6 (`𝕀`-born) | 33..63 | 31 | 120 | 3720 |
| 7 (routon-native) | 65..127 | 63 | 124 | 7812 |
| | | 116 | | **13884** |

Fiber sizes obey `size(m, b) = 2^b − 2^(b−m+2)`, verified for every fiber at `b = 4, 5, 6, 7` (contract C4). Level 7 is the first level at which **four** birth generations coexist.

### The defect diagonal

Each fiber is *almost full*. Of the `2^(b−1) − 1` candidate index pairs per label (×2 signs), exactly `2^(b−m+1) − 1` index pairs are missing, and the missing pairs form the explicit diagonal (contract C7, verified for all 116 L7 fibers). Writing `ℓ = 2^(m−1) + r` with `1 ≤ r ≤ 2^(m−1) − 1`:

```
missing(ℓ, b) = { {a, a ⊕ ℓ} : a ∈ span_F2{r, 2^m, 2^(m+1), …, 2^(b−1)} ∖ {0} }
```

The fundamental missing pair is `{r, 2^(m−1)}`: `e_r ± e_{2^(m−1)}` is **invertible**, and doubling propagates the defect upward. Every routon-native fiber misses exactly one pair, `{ℓ−64, 64}`. Sign duality holds for every present index pair — now exact by the 2-cycle criterion, not merely observed.

### The nullity spectrum (novel at this level)

The 2-cycle criterion yields the exact nullity of every canonical zero divisor at no extra cost. The observed law (contract C9, verified exhaustively at `b = 4, 5, 6, 7`):

```
nullity of an m-born canonical ZD at level b = 2^(b−m+2) · t,
    t odd,  1 ≤ t ≤ 2^(m−3) − 1,
```

and at level 7 **every** allowed value occurs in **every** birth class:

| birth level m | base `2^(7−m+2)` | odd parts `t` | nullities at L7 |
|---|---|---|---|
| 4 | 32 | {1} | {32} |
| 5 | 16 | {1, 3} | {16, 48} |
| 6 | 8 | {1, 3, 5, 7} | {8, 24, 40, 56} |
| 7 | 4 | {1, 3, …, 15} | {4, 12, 20, 28, 36, 44, 52, 60} |

The full L7 histogram (index pairs): `{4: 684, 8: 504, 12: 504, 16: 336, 20: 336, 24: 336, 28: 336, 32: 336, 36: 336, 40: 336, 44: 336, 48: 504, 52: 504, 56: 684, 60: 870}`.

Corollaries:

- The maximum nullity at level `b` is `2^(b−1) − 4`, attained only by routon-native pairs; at L7 the maximum is **60**, attained e.g. by `e_3 + e_66`. This confirms the native-erasure curve `ker_native(L_n) = 2^(n−1) − 4` conjectured in `examples/routon_projective_measurement.sio` — as a **maximum**, not a uniform value: the native spectrum is the 8-value set `{4, 12, …, 60}`, so the conjecture is refined, not merely confirmed.
- The lifted-kernel doubling of the same example (`e_3+e_10`: ker `4→8→16→32` across L4→L7) is the `m = 4`, `t = 1` case of the law and is exact.
- The 2-adic valuation of the nullity of an `m`-born pair at level `b` is exactly `b − m + 2`; doubling a pair up one level exactly doubles its kernel.
- Exact nullities agree with SVD numerical rank on a spot-check spanning all 12 distinct L7 values (contract C9).

### Tower embedding

The lower-level zero divisors embed exactly: the `𝕀` census equals the routon census restricted to indices `< 64`, and labels are nested `labels(𝕊) ⊂ labels(𝕋) ⊂ labels(𝕀) ⊂ labels(routons)` (contract C5; the lower inclusions are gated by the L6 contract).

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **C1_ZD_CENSUS** | The 13884 canonical routon zero divisors are correctly enumerated. | 13884 triples found (6942 index pairs × 2 signs). |
| **C2_GROWTH_LAW** | The census law `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4` reproduces 84, 588, 3036, 13884. | Exact match at b = 4, 5, 6, 7. |
| **C3_FIBER_DECOMPOSITION** | The zero divisors decompose into `F(b) = 2^b − b − 5` fibers; L7 labels are exactly `{9..15} ∪ {17..31} ∪ {33..63} ∪ {65..127}`. | 7/22/53/116 fibers; label sets match. |
| **C4_FIBER_SIZE_LAW** | Every fiber born at level `m` has size `2^b − 2^(b−m+2)` at level `b`. | All fibers at b = 4, 5, 6, 7 match. |
| **C5_TOWER_EMBEDDING** | `𝕀` ZDs ⊂ routon ZDs by index restriction; labels nested across all four levels. | Exact restricted-census equality. |
| **C6_DENSITY_GROWTH** | ZD density strictly increases: 0.4000 < 0.6323 < 0.7773 < 0.8676. | Monotonic, values as tabulated. |
| **C7_NATIVE_DEFECT** | Each fiber born at `m` misses exactly `2^(b−m+1) − 1` index pairs at level `b`, equal to the defect diagonal; sign duality holds; the 63 routon-native fibers each miss exactly `{ℓ−64, 64}`. | All 116 L7 fibers match. |
| **C8_EXACT_SVD_CROSSCHECK** | The exact 2-cycle criterion reproduces the SVD-based census at every level b = 4..7. | Full census equality: 42/294/1518/6942 index pairs. |
| **C9_NULLITY_LAW** | Every `m`-born canonical ZD at level `b` has nullity `2^(b−m+2)·t`, `t` odd, `1 ≤ t ≤ 2^(m−3)−1`; every allowed value occurs in every L7 birth class; max at L7 is 60 = `2^6 − 4`, attained by `e_3 + e_66`; SVD spot-check of exact nullities across all 12 distinct L7 values. | Exhaustive at b = 4..7; spot-check passes. |

---

## 4. What this is NOT

- **Not a proof.** The growth law, size law, defect diagonal, and nullity law are exact at levels 4–7 and derived from doubling heuristics; level 8 (256-dim, candidates `(2^8−1)(2^8−2) = 64770` triples, prediction `Z(8) = 59772`) is the natural next falsification target. The exact 2-cycle criterion makes the L8 scan cheap (no SVD needed for the census; SVD cross-check still advised).
- **Not a G₂ statement.** `G₂ = Aut(𝕆)` acts on the `𝕊` fibers; the L7 computation is enumerative and makes no automorphism claim at level 7. The `𝕊`-born fibers inside the routons are the double lifts of those in `g2_zd_fibers_spec_2026-07-25.md`.
- **Not the full zero-divisor locus.** Only canonical 2-unit sums `e_i ± e_j` are scanned; the full ZD variety of the routons is larger.
- **Not an explanation of the odd-part distribution.** The nullity law pins down the *set* of nullities per birth class; the histogram multiplicities (e.g. `684, 504, 336, …, 870` at L7) are reported as data without a claimed counting law.
- **Not a clinical claim.**

---

## 5. Reproduce

```bash
python3 scripts/research/routon_zd_contract.py
# expect: C1..C9 PASS, ROUTON_ZD_VERDICT C_GREEN   (~12 s; the L7 SVD
# cross-check in C8 is the bulk of the runtime)

bash scripts/ci/routon_zd_gate.sh
# expect: ROUTON_ZD_GATE_OK
```

Pure Python + NumPy, self-contained. The sign table `S[i, j] = cds(i, j)` is precomputed once per level (128×128 at L7). The primary scan evaluates `p(k) = S[i,k]·S[j,k]·S[i,k⊕ℓ]·S[j,k⊕ℓ]` vectorized over `k` — exact integer arithmetic, no floating point; the SVD scan of the L4/L5/L6 contracts is retained as the C8 reference and as the C9 nullity spot-check oracle.

---

## 6. AI disclosure

Spec and harness drafted under human direction (2026-07-26). No clinical content. GAIDeT-ICMJE 2025.
