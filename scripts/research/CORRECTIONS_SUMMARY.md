# Audit Corrections — Cayley-Dickson Hierarchy Paper

**Date:** 2026-08-15
**Auditor:** Claude (external-style audit, verified by Kimi Code)
**Status:** All 6 defects fixed and re-verified

## Bugs Fixed

### 1. Dyck-1 Generator (CRITICAL)
**Bug:** `gen_dyck1` forced `invalid[:, 0] = 2`, making `label == (token[0] == ')')` at 100% accuracy. Also `must_open = (depth == 0) | (remaining <= depth)` forced opening when it should force closing.

**Fix:** Swap `( at position i ↔ ) at position j` in valid words. Preserves count (count classifier at chance = 0.500), breaks order. Exclude position 0 from opens to keep `P(token0='(') = 1.000` in both classes.

**Files fixed:** `ossm_dyck_scaling.py`, `cayley_dickson_paper_reproduction.py`, new: `dyck1_fast.py`

**Corrected result (3 seeds, L=32, 50 epochs):**
| Model | Accuracy |
|-------|----------|
| OctTree-8 | 0.691 ± 0.009 |
| RealTree-8 | 0.500 ± 0.000 |
| **Gap** | **+0.191** (was +0.46 with bug) |

### 2. Sedenion Multiplication Table (CRITICAL)
**Bug:** `_build_sed_sign` used `c = a ^ b` (XOR) for index assignment and copied `sign(a,b)` to both cross-blocks. Result: 42/210 anticommutativity violations.

**Fix:** Cayley-Dickson doubling: `(a,b)(c,d) = (ac - conj(d)b, da + b conj(c))`
- lo-hi: `sign(i,8+j) = OCT_SIGN[j,i]`, `idx(i,8+j) = 8+OCT_IDX[j,i]`
- hi-lo: `sign(8+i,j) = -OCT_SIGN[i,j]`, `idx(8+i,j) = 8+OCT_IDX[i,j]`  
- hi-hi: `sign(8+i,8+j) = OCT_SIGN[j,i]`, `idx(8+i,8+j) = OCT_IDX[j,i]`

**Files fixed:** `cayley_dickson_paper_reproduction.py`, `pseudoknot_experiment.py`, `hook21_bracket.py`, `nback_sedenion_experiment.py`

**Verification:** 0/120 violations, flexible, non-alternative, octonion subalgebra closed.

### 3. MatrixTree Wl=Wr (MATERIAL)
**Bug:** `Wl = Wr = U[level] @ V[level]` — same matrix for left and right, making the node symmetric (commutative). Guaranteed the baseline fails.

**Fix:** Independent `U_left/V_left` and `U_right/V_right` parameters.

**Corrected result (1 seed, L=32):**
| Model | Params | Accuracy |
|-------|--------|----------|
| OctTree-8 (Fano) | 182 | 0.729 |
| MatrixTree-Full | 1946 | **0.771** (was 0.500 with bug) |
| RealTree-8 | 182 | 0.500 |

### 4. [2,1]-hook Verification Formula (MATERIAL)
**Bug:** `verify_hook_21()` used ad-hoc `A(a,b,c)+A(b,a,c)−A(a,c,b)−A(c,a,b)` instead of the correct projector. Also `hook_aab = assoc_aab + assoc_aba - assoc_aab - assoc_baa` self-cancels.

**Fix:** Use `hook_21_bracket()` which computes `T - Λ³(T) - Sym³(T)` with all 6 permutations.

**Corrected result:**
- Octonion: ‖[2,1]‖ = 1.5–3.2 × 10⁻¹⁵ (machine zero — identity)
- Sedenion: ‖[2,1]‖ = 21–44 (nonzero — non-alternativity)

### 5. Pseudoknot Generator (MATERIAL)
**Bug:** `gen_pk` produced exactly 2 constant sequences — memorization, not classification.

**Fix:** Each configuration instantiated twice (nested + crossed) with identical token positions. Only signal is closing order.

### 6. Pseudoknot Dimensional Control (NEW)
Added RealTree-16 (element-wise, 16-dim) as dimensional control for SedenTree-16.

**Key result:** RealTree-16 = 0.500 (chance) proves sedenion advantage is algebraic, not parametric.

## Corrected Headline Results

### Pseudoknot: nested vs crossed (strongest result)
| Model | Dim | Product | Params | Accuracy |
|-------|-----|---------|--------|----------|
| RealTree-8 | 8 | element-wise | 214 | 0.500 |
| OctTree-8 | 8 | octonion ⊗ | 214 | 0.727 |
| RealTree-16 | 16 | element-wise | 398 | **0.500** |
| **SedenTree-16** | **16** | **sedenion ⊗** | **398** | **0.992** |

### Dyck-1 (corrected, 3 seeds)
| L | OctTree | RealTree | Gap |
|---|---------|----------|-----|
| 32 | 0.691 ± 0.009 | 0.500 | +0.191 |
| 64 | 0.514 ± 0.016 | 0.500 | +0.014 |

### Decisive test (corrected MatrixTree)
| Model | Params | Accuracy |
|-------|--------|----------|
| OctTree-8 | 182 | 0.729 |
| MatrixTree-Full | 1946 | 0.771 |
| RealTree-8 | 182 | 0.500 |

## What Changed in the Paper's Claims

| Original claim | Corrected status |
|---|---|
| OctTree beats RealTree by +42-46% on Dyck-1 | **+19% at L=32, fading by L=64. Real but modest.** |
| Fano plane is necessary (MatrixTree fails) | **Fano is parameter-efficient, not necessary. MatrixTree works with Wl≠Wr.** |
| SedenTree beats OctTree on crossing | **CONFIRMED with CD-correct sedenion + dimensional control.** |
| [2,1]-hook vanishes on O, nonzero on S | **CONFIRMED with correct projector.** |

## Files Modified

1. `scripts/research/cayley_dickson_paper_reproduction.py` — Dyck gen, sedenion table, MatrixTree, pseudoknot gen
2. `scripts/research/ossm_dyck_scaling.py` — Dyck gen
3. `scripts/research/pseudoknot_experiment.py` — sedenion table
4. `scripts/research/hook21_bracket.py` — sedenion table, hook-21 verification
5. `scripts/research/nback_sedenion_experiment.py` — sedenion table
6. `scripts/research/dyck1_fast.py` — new file, vectorized corrected generator
7. `docs/papers/main/cayley_dickson_hierarchy_paper_2026-08-13.md` — §9.4 added
8. `docs/papers/main/cayley_dickson_restructured_2026-08-13.md` — tables and abstract updated
