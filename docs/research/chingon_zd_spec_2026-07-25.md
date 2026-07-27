<!-- docs:meta
topic_id: repo.docs.research.chingon-zd-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.chingon-zd-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Chingon zero-divisor structure — level-6 annihilation geometry (64 dimensions)

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (C_GREEN reached 2026-07-25)  
**Parents:** `scripts/research/trigintaduonion_zd_contract.py` (L5 fibers), `docs/research/trigintaduonion_zd_spec_2026-07-25.md`, `docs/research/g2_zd_fibers_spec_2026-07-25.md`, `examples/chingon_projective_measurement.sio` (L6 projective measurement)  
**Harness:** `scripts/research/chingon_zd_contract.py`  
**Gate:** `scripts/ci/chingon_zd_gate.sh`

---

## 1. What this is

The Cayley–Dickson tower continues past the trigintaduonions to the chingons `𝕀` (level 6, dimension 64; repo naming per `examples/chingon_projective_measurement.sio`). This contract extends the catastrophe scan to level 6 and analyzes the **fiber structure** of the `𝕀` zero-divisor locus: the canonical zero-divisor census, the decomposition by xor-label, fiber sizes, and the comparison with levels 4 (sedenions `𝕊`) and 5 (trigintaduonions `𝕋`).

The computation is **novel** (the level-6 fiber structure has not been explicitly tabulated in the literature we are aware of) and it **closes the combinatorics**: the censuses at levels 4, 5, 6 are reproduced exactly by a single closed-form growth law, and every fiber size is reproduced exactly by a birth-level size law with an explicit defect diagonal.

---

## 2. Mathematical setup

### The doubling tower

```
𝕆 (8) → 𝕊 (16) → 𝕋 (32) → 𝕀 (64)
L3        L4        L5        L6
```

Zero divisors are born at `𝕊`. A **canonical zero divisor** is a 2-unit sum `a = e_i ± e_j` (`1 ≤ i < j < 2^b`, `e_0 = 1` the real unit) whose left multiplication matrix `L_a` is singular. Each canonical pair carries an **xor-label** `ℓ = i ⊕ j`; pairs sharing a label form a **fiber**.

### Census and growth law

Computed censuses (contract C1/C2):

| level b | algebra | dim | candidates `(n−1)(n−2)` | canonical ZDs | density |
|---|---|---|---|---|---|
| 4 | `𝕊` | 16 | 210 | 84 | 0.4000 |
| 5 | `𝕋` | 32 | 930 | 588 | 0.6323 |
| 6 | `𝕀` | 64 | 3906 | **3036** | 0.7773 |

All three censuses are reproduced exactly by

```
Z(b) = 4^b − (3b − 1)·2^b + 2^(b−1) − 4        (b ≥ 4)
```

This is not merely a three-point fit: it is the algebraic sum of the fiber-count law and fiber-size law below,

```
Z(b) = Σ_{m=4}^{b} (F(m) − F(m−1)) · size(m, b)
     = Σ_{m=4}^{b} (2^(m−1) − 1)(2^b − 2^(b−m+2)),
```

both of which are verified **fiber by fiber** (not just in aggregate) at b = 4, 5, 6. It remains an inductive law rather than a theorem: its level-7 prediction is `Z(7) = 13884`, and the falsifiers doc states what would break it. (A plain quadratic-in-`2^b` interpolant `15·4^(b−2) − 27·2^(b−1) + 60` also agrees at b = 4, 5, 6 but is a *different* function — it predicts 13692 at b = 7 — and is not the law claimed here.)

### Fiber structure

A label `ℓ` supports zero divisors at level `b` iff `ℓ ≥ 8` and `ℓ` is not a power of two. Hence the fiber count is

```
F(b) = 2^b − b − 5        →  F(4) = 7,  F(5) = 22,  F(6) = 53
```

Every label `ℓ` has a **birth level** `m = ⌊log2 ℓ⌋ + 1` (the unique `m` with `2^(m−1) < ℓ < 2^m`). At level 6 the 53 fibers stratify by birth level:

| birth level m | labels | count | size at L6 | total |
|---|---|---|---|---|
| 4 (`𝕊`-born) | 9..15 | 7 | 48 | 336 |
| 5 (`𝕋`-born) | 17..31 | 15 | 56 | 840 |
| 6 (`𝕀`-born) | 33..63 | 31 | 60 | 1860 |
| | | 53 | | **3036** |

Fiber sizes obey

```
size(m, b) = 2^b − 2^(b−m+2)
```

verified for every fiber at b = 4, 5, 6 (contract C4). Schafer doubling doubles each annihilator per level, so older fibers grow; native fibers are born at size `2^b − 4` (64 − 4 = 60 at L6).

### The defect diagonal

Each fiber is *almost full*. Of the `2^(b−1) − 1` candidate index pairs per label (×2 signs), exactly `2^(b−m+1) − 1` index pairs are missing, and the missing pairs form an explicit diagonal (contract C7). Writing `ℓ = 2^(m−1) + r` with `1 ≤ r ≤ 2^(m−1) − 1`:

```
missing(ℓ, b) = { {a, a ⊕ ℓ} : a ∈ span_F2{r, 2^m, 2^(m+1), …, 2^(b−1)} ∖ {0} }
```

The fundamental missing pair is `{r, 2^(m−1)}`: `e_r ± e_{2^(m−1)}` is **invertible** — the new generator adjoined at the birth level never annihilates its lower-half partners — and doubling propagates this defect upward (1 missing pair at birth, 3 after one lift, 7 after two). In particular every chingon-native fiber misses exactly one pair, `{ℓ−32, 32}`. Additionally, every present index pair carries **both** signs (sign duality, verified C7).

### Tower embedding

The lower-level zero divisors embed exactly: the `𝕊` census equals the `𝕋` census restricted to indices `< 16`, and the `𝕋` census equals the `𝕀` census restricted to indices `< 32`; labels are nested `labels(𝕊) ⊂ labels(𝕋) ⊂ labels(𝕀)` (contract C5).

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **C1_ZD_CENSUS** | The 3036 canonical `𝕀` zero divisors are correctly enumerated. | 3036 pairs found. |
| **C2_GROWTH_LAW** | The census law `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4` reproduces 84, 588, 3036. | Exact match at b = 4, 5, 6. |
| **C3_FIBER_DECOMPOSITION** | The zero divisors decompose into `F(b) = 2^b − b − 5` fibers; L6 labels are exactly `{9..15} ∪ {17..31} ∪ {33..63}`. | 7/22/53 fibers; label sets match. |
| **C4_FIBER_SIZE_LAW** | Every fiber born at level `m` has size `2^b − 2^(b−m+2)` at level `b`. | All fibers at b = 4, 5, 6 match. |
| **C5_TOWER_EMBEDDING** | `𝕊` ZDs ⊂ `𝕋` ZDs ⊂ `𝕀` ZDs by index restriction; labels nested. | Exact restricted-census equality. |
| **C6_DENSITY_GROWTH** | ZD density strictly increases: 0.4000 < 0.6323 < 0.7773. | Monotonic, values as tabulated. |
| **C7_NATIVE_DEFECT** | Each fiber born at `m` misses exactly `2^(b−m+1) − 1` index pairs at level `b`, equal to the defect diagonal; sign duality holds; the 31 `𝕀`-native fibers each miss exactly `{ℓ−32, 32}`. | All 53 L6 fibers match. |

---

## 4. What this is NOT

- **Not a proof.** The growth law, size law, and defect diagonal are exact at levels 4–6 and derived from a doubling heuristic; level 7 (128-dim routons, ~125 s extrapolated runtime) is the natural next falsification target.
- **Not a G₂ statement.** `G₂ = Aut(𝕆)` acts on the `𝕊` fibers; the L6 computation is enumerative and makes no automorphism claim at level 6. The `𝕊`-born fibers inside `𝕀` are the lifts studied in `g2_zd_fibers_spec_2026-07-25.md`.
- **Not the full zero-divisor locus.** Only canonical 2-unit sums `e_i ± e_j` are scanned; the full ZD variety of `𝕀` is larger.
- **Not a clinical claim.**

---

## 5. Reproduce

```bash
python3 scripts/research/chingon_zd_contract.py
# expect: C1..C7 PASS, CHINGON_ZD_VERDICT C_GREEN   (~4 s)

bash scripts/ci/chingon_zd_gate.sh
# expect: CHINGON_ZD_GATE_OK
```

Pure Python + NumPy, self-contained. The sign table `S[i, j] = cds(i, j)` is precomputed once per level and `L_a` is assembled by fancy indexing (row `k` is nonzero only in columns `k⊕i`, `k⊕j`); this is the identical matrix to the naive assembly in the L4/L5 contracts, only faster.

---

## 6. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical content. GAIDeT-ICMJE 2025.
