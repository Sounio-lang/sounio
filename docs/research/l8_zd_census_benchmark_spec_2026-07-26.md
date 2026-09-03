<!-- docs:meta
topic_id: repo.docs.research.l8-zd-census-benchmark-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.l8-zd-census-benchmark-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Level-8 ZD census benchmark — fast exact implementation with exact verification (256 dimensions)

**Date:** 2026-07-26
**Status:** `EXECUTABLE` (gate green 2026-07-26)
**Parents:** `docs/research/routon_zd_spec_2026-07-26.md` (L7 structure, Z(8) = 59772 prediction), `scripts/research/routon_zd_contract.py` (baseline implementation)
**Harness:** `scripts/research/l8_zd_census_fast.c`, `scripts/research/l8_zd_baseline_benchmark.py`
**Gate:** `scripts/ci/l8_zd_census_gate.sh`

---

## 1. What this is

Two results in one artifact:

1. **Falsification test.** The growth law `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4`, exact at `b = 4, 5, 6, 7`, predicts `Z(8) = 59772` canonical zero divisors (index pairs `1 ≤ i < j < 256`, ×2 signs) at level 8 (256-dim Cayley–Dickson algebra). The level-8 scan confirms the prediction **exactly**: 29886 index pairs × 2 signs = 59772. The law survives its **second** out-of-sample test (the first was Z(7) = 13884).
2. **Benchmark.** The exact-verification phase of the previous Python implementation (`svd_zd_index_pairs`, the audited SVD oracle of the L4–L7 contracts) costs **275.6 s** at level 8. The new C implementation performs the census **and** a strictly more rigorous exact verification in **6.1 s** — a **45× speedup** for the same task (census + exact verification), with no approximation anywhere.

The L8 nullity histogram is tabulated below (§4); as at L7, the multiplicities are reported as data — the counting law behind them remains the programme's open combinatorial question, now with two levels of data.

---

## 2. Methods

Both methods share only the Cayley–Dickson sign table `S[i, j]` (sign of `e_i·e_j`), built by the same recursion as `routon_zd_contract.py:cds`. The C table is bit-identical to the NumPy one: FNV-1a-64 over the 65536 int8 entries is `a24bd1a4e58b9f83` in both implementations (checked by the gate).

### Method 1 — census (the audited 2-cycle criterion)

Unchanged from the L7 contract: for `l = i ⊕ j`, `p(k) = S[i,k]·S[j,k]·S[i,k⊕l]·S[j,k⊕l] ∈ {+1,−1}`; `(i, j)` is a canonical ZD pair iff some `p(k) = +1`, and `nullity(L_a) = #{k : p(k) = +1}/2` exactly, for both signs simultaneously. Cost: `O(2^b)` integer ops per candidate pair.

### Method 2 — exact verification by GF(65521) rank (replaces the SVD oracle)

For each candidate pair `(i, j)` and **each** sign `sgn ∈ {+1, −1}` the verifier builds the `256×256` matrix

```
M(sgn) = I + sgn·Q,   Q[k][k⊕l] = S[i,k]·S[j,k⊕l],
```

over `GF(65521)` (65521 is the largest 16-bit prime) and computes its rank by Gaussian elimination with partial pivoting — generic exact linear algebra, no closed-form nullity formula, no sparsity assumption beyond the standard zero-entry skip. Since `L_a = A·(I + sgn·Q)` with `A` a signed permutation matrix, `rank(L_a) = rank(M)`.

**Exactness (why a modular computation is an exact Q-rank).** `Q` is a signed permutation matrix whose underlying permutation is the fixed-point-free involution `k ↦ k⊕l`. Hence `M(sgn)` decomposes (after a simultaneous row/column permutation) into `2×2` diagonal blocks `[[1, sgn·q'], [sgn·q, 1]]` with `q·q' = p(k) ∈ {+1, −1}`. A block has rank 1 if `p(k) = +1` and rank 2 if `p(k) = −1`, over **any** field of characteristic ≠ 2 — including Q and GF(65521). Therefore `rank_GF(65521)(M) = rank_Q(M)` for every pair and sign; the modular rank is not a probabilistic certificate but an exact one. (Method 2 does not *use* the block decomposition — it runs dense GE on the full matrix; the decomposition is only the proof that GF(65521) rank equals Q rank.)

The verifier audits Method 1 on the **complete** candidate set: all `C(255, 2) = 32385` pairs × 2 signs = 64770 rank computations, requiring `256 − rank = nullity_tab[i][j]` (0 for non-ZD pairs) in every case. Result: **0 mismatches**. This is strictly stronger than the L7 SVD cross-check, which was numerical (tolerance `1e-9`) and covered only the +1 sign.

### Contract assertions (all exact, all green)

| Clause | Statement | Result |
|---|---|---|
| census | 29886 index pairs, 59772 triples `= Z(8)` law | PASS |
| fibers | 243 `= F(8) = 2^8 − 8 − 5` fibers; labels exactly `{l ∈ [8,256) : l not a power of 2}`; every `m`-born fiber has `2^8 − 2^(8−m+2)` triples | PASS |
| nullity law | every `m`-born pair has nullity `2^(8−m+2)·t`, `t` odd, `1 ≤ t ≤ 2^(m−3)−1`; every allowed value occurs in every birth class `m = 4..8` | PASS |
| max nullity | `124 = 2^7 − 4` (native-erasure maximum) | PASS |
| verification | GF(65521) rank = census nullity on all 64770 pair-signs, 0 mismatches | PASS |

---

## 3. Benchmark

Task: full level-8 census **with exact verification** (build sign table, enumerate all canonical ZD pairs with exact nullities, independently verify every pair-sign). Hardware: this dev container (single thread, x86_64); compiler `cc -O2` (GCC 13.3). Baseline: `scripts/research/l8_zd_baseline_benchmark.py` (NumPy 2.5.1, `.venv` Python 3.12), which reuses `routon_zd_contract.py` unchanged.

| phase | Python baseline | C fast | speedup |
|---|---|---|---|
| sign table (256×256) | 0.059 s | 0.003 s | ~20× |
| exact census scan | 0.166 s | 0.010 s | ~17× |
| exact verification | 275.6 s (SVD, +1 sign only, numerical) | 6.1 s (GF(65521) rank, both signs, exact) | **~45×** |
| **total** | **275.9 s** | **6.1 s** | **~45×** |

The verification phase is the entire gap: the NumPy census was already vectorized, but the SVD oracle does 32385 dense `256×256` singular-value decompositions in a Python loop. The C verifier replaces each SVD with one exact GF(p) Gaussian elimination of a matrix whose fill-in is provably zero, and audits strictly more (both signs, exact arithmetic) in 45× less time. Wall-clock numbers above are stable across 3 runs (6.10–6.16 s).

---

## 4. Level-8 data

- Census: 29886 index pairs, 59772 triples. Density `59772 / (255·254) = 0.9228` (L7: 0.8676 — still increasing).
- Fibers: 243; stratification (triples): `m=4`: 7 labels × 192; `m=5`: 15 × 224; `m=6`: 31 × 240; `m=7`: 63 × 248; `m=8`: 127 × 252. Sum = 59772. ✓
- Nullity histogram (index pairs), 31 distinct values `4..124` step 4:

```
  4:1740   8:1368  12:1368  16:1008  20:1008  24:1008  28:1008
 32:672   36:672   40:672   44:672   48:672   52:672   56:672   60:672
 64:672   68:672   72:672   76:672   80:672   84:672   88:672   92:672
 96:1008 100:1008 104:1008 108:1008 112:1368 116:1368 120:1740 124:2118
```

Sum check: `1740+2·1368+4·1008+16·672+4·1008+2·1368+1740+2118 = 29886`. ✓

Observations on the open multiplicity question (data only, no claimed law):

- The L8 multiplicities are all **even**; the interior plateau `32..92` sits at `672 = 2·336`, exactly twice the L7 interior plateau (`16..44` at 336).
- The L8 edge values `1368 = 2·684`, `1008 = 2·504` are exactly twice the L7 edge values (684, 504) — but the extreme multiplicities do **not** double: L7 had `{4: 684, 60: 870}` while L8 has `{4: 1740, 124: 2118}`, and `1740 ≠ 2·684`, `2118 ≠ 2·870`. The new extremes outgrow the doubling pattern.
- The histogram is not symmetric (`hist[4] = 1740 ≠ 2118 = hist[124]`), but is symmetric on the doubled-L7 part (`8:1368 ↔ 116:1368`, `16:1008 ↔ 108:1008`).

---

## 5. What this is NOT

- **Not a proof.** `Z(b)` remains an inductive law, now exact at five levels (`b = 4..8`); the L8 census is a passed falsification test, not a theorem. Level 9 (512-dim, prediction `Z(9) = 249084`) is the next target; the same code path does it by changing `BITS`. Cost scaling to L9: the verifier's Gaussian elimination skips zero entries (standard practice), and the fill-in of `M(sgn)` is provably zero (§2), so each elimination step touches only `O(2^b)` entries and one rank computation costs `O(4^b)`, not the dense `O(8^b)`; measured per-matrix cost at L8 is ~94 µs ≈ `1.4·(2^8)^2` ns, confirming the quadratic behavior. With ~4× more candidates and 4× per-matrix cost, L9 verification should cost ~16× L8 (~100 s).
- **Not an explanation of the histogram multiplicities.** §4 extends the data to a second level; the counting law for multiplicities (1740, 1368, 1008, 672, …, 2118) is still open.
- **Not the full ZD locus.** Only canonical 2-unit sums `e_i ± e_j`.
- **Not a clinical claim.**

---

## 6. Reproduce

```bash
# fast contract (compile + run + sign-table cross-hash vs NumPy)
bash scripts/ci/l8_zd_census_gate.sh
# expect: L8_ZD_FAST_VERDICT PASS, L8_ZD_CENSUS_GATE_OK   (~6 s)

# baseline to beat (NumPy exact scan + SVD verification)
./.venv/bin/python3 scripts/research/l8_zd_baseline_benchmark.py
# expect: BASELINE_TOTAL seconds≈276, census_equal=True
```

---

## 7. AI disclosure

Spec and harness drafted under human direction (2026-07-26). No clinical content. GAIDeT-ICMJE 2025.
