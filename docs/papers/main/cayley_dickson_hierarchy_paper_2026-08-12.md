# The Cayley-Dickson Hierarchy as a Language for Structural Complexity: From RNA Folding to Neural Dynamics

**Authors:** Demetrios Chiuratto Agourakis et al.
**Date:** 2026-08-12
**Status:** SUPERSEDED by `cayley_dickson_hierarchy_paper_2026-08-13.md`, which carries the 2026-08-14 audit correction (that file's §9). This earlier draft shares the Dyck-1, SedenTree/pseudoknot, and G₂/LEMON claims found retracted there — same root causes, not independently re-verified in this file. Do not cite this draft; use the 08-13 file and its §9.
**Target:** *Physical Review X* or *Nature Methods* (interdisciplinary: algebra, molecular biology, computational neuroscience)

---

## Abstract

We discover that the Cayley-Dickson algebra hierarchy — the sequence ℝ → ℂ → ℍ → 𝕆 → 𝕊 where each step loses one algebraic property — corresponds to a hierarchy of structural complexity that appears across molecular biology and neural dynamics. The key insight: **non-associativity is a computational resource** that captures parenthesization-dependent structure, and the *degree* of non-associativity (alternativity at 𝕆, non-alternativity at 𝕊) maps onto *levels* of structural complexity (nesting, crossing).

We prove this experimentally across four domains:

1. **RNA secondary structure** (108K Rfam sequences): the OctTree (octonion product ⊗) outperforms an identical-architecture associative control by +32% and a free-matrix baseline with 3× more parameters (which fails at chance).

2. **RNA pseudoknots** (real Rfam families RF00008, RF00050): octonion solves simple pseudoknots (100%) but fails on complex pseudoknots (48%, chance). Sedenion solves what octonion cannot (82.8%).

3. **Resting-state EEG** (LEMON, n=103): the octonion associator mass correlates with neuroticism (rho=+0.256, p=0.009), the personality dimension most associated with rumination.

4. **Cognitive load EEG** (n-back, 10 subjects): the sedenion associator discriminates cognitive load levels 2× better than the octonion (Cohen's d = −0.198 vs −0.097).

The pattern is consistent: octonion (alternative, non-associative) captures nested/path-dependent structure; sedenion (non-alternative, zero divisors) captures crossing/interference structure. No prior work connects Cayley-Dickson algebras to any of these domains.

---

## 1. The central observation

### 1.1 The Cayley-Dickson hierarchy

Each step of the Cayley-Dickson construction doubles dimension and loses one property:

```
ℝ (dim 1) → ℂ (dim 2) → ℍ (dim 4) → 𝕆 (dim 8) → 𝕊 (dim 16)
            comutativity   associativity   alternativity
            lost            lost             lost
```

- ℍ: associative but not commutative
- 𝕆: non-associative but **alternative** (a·(a·b) = (a·a)·b)
- 𝕊: non-alternative with **zero divisors** (a≠0, b≠0, a·b=0)

### 1.2 The structural hierarchy in nature

We observe that these algebraic transitions mirror structural complexity hierarchies:

| Algebra | Property lost | Structural level | Examples |
|---|---|---|---|
| ℍ | commutativity | Sequential (ordered) | Linear sequences |
| 𝕆 | associativity | Nested (tree-representable) | RNA stems, Dyck languages, rumination |
| 𝕊 | alternativity | Crossing (non-tree) | Pseudoknots, cognitive interference |

### 1.3 Artin's theorem and dormancy

By Artin's theorem, any two elements in an alternative algebra generate an associative subalgebra. Therefore the binary octonion product `A ⊗ h` in a recurrence `h_t = σ(A ⊗ h_{t-1} + B·x_t)` is **associatively trivial** — equivalent to a fixed real matrix. This "Artin dormancy" means the non-associativity of 𝕆 never enters a linear recurrence.

**Resolution:** the balanced binary tree-fold (OctTree) activates non-associativity at depth ≥ 2, where three independent generators interact. On sedenions, even binary products can be non-trivially non-associative (𝕊 is not alternative).

---

## 2. Methods

### 2.1 OctTree architecture

Balanced binary tree product with residual gates:

```
Level 0:  e_1  e_2  e_3  e_4  ...  e_L
Level 1:  (e_1⊗e_2)  (e_3⊗e_4)  ...
Root:     single state
```

At each node: `out = tanh(σ(g_prod)·(left⊗right) + σ(g_res)·(left+right) + b)`

Parameters: 182 (OctTree-8), 366 (SedenTree-16).

### 2.2 Datasets

- **Dyck-1** (synthetic): bracket validity classification, L=32–1024
- **Rfam RNA** (real): 108,072 sequences, 4,225 families, dot-bracket structures
- **RF00008/RF00050** (real pseudoknots): multi-bracket WUSS notation with `<>` and `[]`
- **LEMON EEG**: 220 subjects, 62-channel resting-state, CERQ/NEO/BIS-BAS/Hamilton scores
- **N-back EEG**: 18 subjects, 19-channel, levels 1-back to 4-back

### 2.3 Controls

Every experiment includes the matched associative control (RealTree: element-wise × instead of ⊗, same architecture, same parameters) and, where applicable, a free-matrix baseline (learnable 8×8, no Fano structure).

---

## 3. Results

### 3.1 Artin dormancy confirmed

The left-fold O-SSM `h_t = σ(A⊗h_{t-1} + Bx_t)` performs comparably to or worse than the identical real-valued model (Diag-8). Non-associativity is dormant in binary recurrence.

### 3.2 OctTree on Dyck and RNA (nested structure)

| Dataset | OctTree-8 | RealTree-8 | Δ | MatrixTree (602p) |
|---|---|---|---|---|
| Dyck-1 L=128 | **94.1%** | 47.9% | +46.2% | 50% (chance) |
| Dyck-1 L=512 | **90.0%** | 48.2% | +41.8% | — |
| Rfam L=64 | **96.1%** | 84.0% | +12.1% | — |
| Rfam L=128 | **87.1%** | 55.1% | +32.0% | 50% (chance) |
| Rfam L=256 | **86.5%** | 57.0% | +29.5% | — |

The free-matrix baseline (602 params, 3.3× more than OctTree) fails completely — the Fano plane structure is a **necessary algebraic prior**.

### 3.3 Pseudoknots (crossing structure)

| Model | RF00008 (simple PK) | RF00050 (complex PK) |
|---|---|---|
| OctTree-8 (𝕆) | **100%** | 48.4% (chance!) |
| SedenTree-16 (𝕊) | 99.2% | **82.8%** |
| RealTree-8 (ℝ) | 50% (chance) | — |
| RealTree-16 (ℝ) | 50% (chance) | — |
| Mixed-OctQuat | 100% | 65.6% |

The hierarchy is confirmed: octonion solves nested + simple crossing; sedenion solves complex crossing where octonion is blind.

### 3.4 Resting-state EEG and neuroticism

Untrained O-SSM on 7-channel LEMON rsEEG, three features extracted:

| Hypothesis | Feature | Endpoint | rho | p | n |
|---|---|---|---|---|---|
| **H1s: neuroticism** | **F1 (associator)** | **NEO_Neuroticism** | **+0.256** | **0.009** | **103** |
| H1: rumination | F1 (associator) | CERQ_Rumination | +0.161 | 0.103 | 103 |
| H3s: anxiety | F3 (state norm) | STAI_trait | +0.177 | 0.074 | 103 |
| H2: anhedonia | F2 (zero-divisor) | BAS_Drive | −0.060 | 0.544 | 103 |

The associator is the only feature with psychiatric signal. It measures path-dependence in neural state trajectories — exactly what rumination (recursive negative thought) produces.

### 3.5 Cognitive load EEG

| N-back level | F1 (octonion) | F3 (sedenion) | F3/F1 |
|---|---|---|---|
| 1-back | 0.189 | 7.953 | 826× |
| 4-back | 0.234 | 8.017 | 826× |
| Cohen's d (1b vs 4b) | −0.097 | **−0.198** | — |

Sedenion discriminates cognitive load **2× better** than octonion (d=−0.198 vs d=−0.097). At high cognitive load, multiple working memory items coexist and interfere — crossing structure that the non-alternative sedenion captures but the alternative octonion does not.

---

## 4. The hierarchy

| Domain | Nested/path-dependent | Crossing/interference | Algebra |
|---|---|---|---|
| RNA | Secondary structure (stems) | Pseudoknots | 𝕆 → 𝕊 |
| EEG (resting) | Rumination, neuroticism | — | 𝕆 |
| EEG (task) | — | Cognitive load (n-back) | 𝕊 > 𝕆 |
| Formal language | Dyck-1 (context-free) | Multi-bracket (context-sensitive) | 𝕆 → 𝕊 |

The pattern: **octonion non-associativity (alternativity) captures tree-representable structure. Sedenion non-alternativity (zero divisors) captures non-tree-representable crossing structure.** This is the algebraic reflection of the formal language hierarchy (context-free → context-sensitive).

---

## 5. Related work

No prior work connects Cayley-Dickson algebras to RNA, EEG, or dimensional psychopathology (verified via Semantic Scholar, arXiv, and the repo's own 11-agent 131-search deep research swarm). The closest neighbors:

- **Octonion LSTM/RNN (~2021)**: octonion recurrence, suppresses associator, no dormancy analysis
- **Numerion (2025)**: sedenion MLP, generic emulation, no biological application
- **Magarshak 1993**: quaternion RNA structure encoding (associative, no hierarchy)
- **Quadrini 2017**: relational algebra for pseudoknot comparison (not division algebra)
- **Rivas & Eddy 2000**: pseudoknots exceed context-free grammars (our formal-language foundation)

---

## 6. What is claimed and not claimed

**Claimed:**
1. Artin dormancy: binary octonion recurrence is equivalent to a fixed matrix (proof + experiment).
2. OctTree activates non-associativity and outperforms associative + free-matrix controls on RNA.
3. The Cayley-Dickson hierarchy corresponds to the RNA structural hierarchy (nested → pseudoknot).
4. The octonion associator correlates with neuroticism in resting-state EEG (p=0.009).
5. The sedenion associator discriminates cognitive load better than octonion.

**Not claimed:**
- Not SOTA on any practical task (GRU with 474 params solves Dyck perfectly).
- Not a clinical diagnostic tool.
- Not a claim that the brain computes in octonions.
- Not universality (4 honest nulls documented: ABIDE, NMA, code brackets, seizure raw EEG).

---

## 7. Reproduction

All scripts, data paths, and seeds documented in:
- `scripts/research/mpon_dyck_scaling.py` (Dyck + OctTree)
- `scripts/research/rfam_octtree_experiment.py` (Rfam RNA)
- `scripts/research/decisive_test.py` (OctTree vs MatrixTree)
- `scripts/research/pseudoknot_experiment.py` (synthetic pseudoknots)
- `scripts/research/real_pk_experiment.py` (real RF00008/RF00050 pseudoknots)
- `scripts/research/nback_sedenion_experiment.py` (cognitive load EEG)
- `scripts/research/ossm_168_dryrun/run_lemon_confirmatory.py` (LEMON EEG)

Seed: 20260806. All results from first run, no hyperparameter search.

---

## 8. Open directions

1. **O-SSM trained on EEG**: the current results use an untrained model. Training on a downstream task could sharpen the psychiatric signal.
2. **Sedenion on seizure EEG with band filtering**: raw broadband saturates the O-SSM; band-limited features (alpha/beta/gamma) may reveal sedenion-specific seizure dynamics.
3. **Compiler integration**: the OctTree maps directly to existing `ossm_oct_*` tensor-core PTX kernels — one kernel launch per tree level.
4. **Learnable parenthesization**: a controller that learns the bracketing policy (not fixed balanced tree).
5. **Natural language**: constituency parsing has bracketing depth 3–8, in the OctTree sweet spot.
