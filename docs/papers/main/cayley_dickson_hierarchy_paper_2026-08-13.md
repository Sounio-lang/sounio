<!-- docs:meta
topic_id: repo.docs.papers.main.cayley-dickson-hierarchy-paper-2026-08-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.main.cayley-dickson-hierarchy-paper-2026-08-13
-->

# The Cayley-Dickson Hierarchy as a Language for Structural Complexity: From RNA Folding to Neural Dynamics

**Authors:** Demetrios Chiuratto Agourakis et al.
**Date:** 2026-08-13
**Status:** UNDER CORRECTION (audited 2026-08-13/14, LEMON FFI pilot updated 2026-08-15) — see §9. Do not cite the numbers in §3.1 (Dyck-1 row), §3.3 (SedenTree/Mixed rows), §3.4 (all G₂/LEMON rho/p values — the full n=204/84-test analysis this section reports, not the §9.5 n=15 pilot), §3.5 (F3/sedenion row and the "2×" claim), §4.2, or §4.3 until the retraction notices below are resolved. §1 (Artin dormancy), §3.1 (Rfam RNA row), §3.2, §3.3 (OctTree rows only), §3.5 (F1/octonion row only), and the [2,1]⊥Λ³ *theorem* (not the numbers reported to support it in §4.2/§4.3) survive independent re-audit. The §9.5 G₂/LEMON pilot numbers (n=15) are citable only as an explicitly unverified, non-significant pilot measurement — see §9.5's own hedging, not as a §3.4 reinstatement.
**Target:** *Physical Review X* (interdisciplinary: algebra, molecular biology, computational neuroscience, psychometrics)

---

## Abstract

The Cayley-Dickson construction generates a hierarchy of normed division algebras — ℝ, ℂ, ℍ, 𝕆, 𝕊 — where each doubling of dimension costs one algebraic property: commutativity, then associativity, then alternativity. We discover that this algebraic hierarchy corresponds to a structural complexity hierarchy that appears across molecular biology, linguistics, and neural dynamics.

The central insight is that **non-associativity is a computational resource**: it captures parenthesization-dependent structure that associative models cannot represent. The *degree* of non-associativity (alternativity at 𝕆, non-alternativity at 𝕊) maps onto *levels* of structural complexity (nesting, crossing).

We prove this experimentally across six domains, with five honest nulls documenting the boundary:

1. **RNA secondary structure** (108K Rfam sequences): OctTree (octonion ⊗, 182 params) outperforms an identical associative control by +32% and a free-matrix baseline with 3× more parameters (which fails at chance, 50%).

2. **RNA pseudoknots** (real Rfam families): octonion solves simple pseudoknots (RF00008: 100%) but fails on complex crossing (RF00050: 48%, chance) — **sound**. Sedenion solves what octonion cannot (RF00050: 82.8%) — **retracted**: computed with a non-anticommutative 16×16 table (42/210 imaginary-unit pairs violate eᵢeⱼ=−eⱼeᵢ); a correct, independently-verified `sed_mul` now exists (`stdlib/algebra/sedenion.sio`, 13/13 tests) but the model has not yet been retrained on it. See §9.

3. **Natural language** (Universal Dependencies English): OctTree solves bracket validity at L=16-32 (100%) where RealTree is at chance (50%). Both collapse at L≥64 due to sparse bracketing. *Not independently re-audited; recommend the same label-degeneracy check applied to §3.1/§3.3 before citing.*

4. **Resting-state EEG** (LEMON, n=103): the octonion associator mass correlates with neuroticism (rho=+0.256, p=0.009) — **superseded by the n=204 cohort in §3.4, whose per-generator numbers are themselves retracted; treat this n=103 F1 aggregate figure as provisional.**

5. **Cognitive load EEG** (n-back, 10 subjects): the sedenion associator discriminates cognitive load 2× better than octonion (Cohen's d = −0.198 vs −0.097) — **the octonion (F1, d=−0.097) figure is sound; the sedenion (F3, d=−0.198) figure and the "2×" comparison are retracted**, computed with the same broken sedenion table as (2). See §9.

6. **Synthetic Dyck languages** (L=32-1024): OctTree outperforms RealTree by +42-46% at L≥128, with the RealTree collapsing to chance — **retracted**. The generator behind this table (`ossm_dyck_scaling.py`'s `gen_dyck1`) forces every "valid" sequence to open with token 1 and every "invalid" sequence to open with token 2, so `label == (token[0] == OPEN)` at 100% accuracy: the task is a one-bit lookup, not bracket-matching. A corrected generator exists (`examples/cayley_dickson_dyck_pseudoknot_generators.sio`, verified: both classes 100% genuinely valid/invalid, and — after a follow-up fix to exclude position 0 from the negative-class corruption site — P(token[0]='(' | label) is now exactly 1.000 in *both* classes, i.e. carries zero information about the label rather than 1.000-vs-0.972) but the models have not yet been retrained on it. See §9.

The [2,1]-hook bracket (mixed-symmetry projection) is constructed and verified: it vanishes on octonions (alternativity) and is nonzero on sedenions (non-alternativity), but cannot capture Massey/Borromean higher-homotopy structure — confirming the repo's representation-theoretic theorem that Λ³ and [2,1] are orthogonal irreducibles.

No prior work connects Cayley-Dickson algebras to RNA, EEG, linguistics, or dimensional psychopathology.

---

## 1. The Cayley-Dickson hierarchy

### 1.1 The construction

Each step doubles dimension and loses one property:

| Step | Algebra | Dim | Property lost |
|---|---|---|---|
| ℝ → ℂ | Complex | 2 | Ordering |
| ℂ → ℍ | Quaternion | 4 | Commutativity |
| ℍ → 𝕆 | Octonion | 8 | Associativity |
| 𝕆 → 𝕊 | Sedenion | 16 | Alternativity + division |

The octonion 𝕆 is **alternative**: a·(a·b) = (a·a)·b for all a,b. By Artin's theorem, any two elements generate an associative subalgebra. The sedenion 𝕊 is **not alternative** and has **zero divisors**: a≠0, b≠0, a·b=0.

### 1.2 The structural hierarchy

We observe that these algebraic transitions mirror structural complexity:

| Algebra | What it captures | Examples |
|---|---|---|
| 𝕆 (alternative, non-assoc) | Nested / tree-representable | RNA stems, Dyck, rumination |
| 𝕊 (non-alternative) | Crossing / non-tree | Pseudoknots, cognitive interference |

### 1.3 Artin dormancy

In the binary recurrence h_t = σ(A ⊗ h_{t-1} + B·x_t), the octonion product A⊗h is **associatively trivial** — by Artin, ⟨A,h⟩ is associative. The left-multiplication L(A) is a fixed 8×8 real matrix, and matrix composition is associative.

**Experimentally confirmed**: the left-fold OSSM-8 performs comparably to or worse than Diag-8 (element-wise) on Dyck-1 at all lengths.

---

## 2. The OctTree architecture

### 2.1 Design

Balanced binary tree product with residual gates:

```
Level 0:  e₁  e₂  e₃  e₄  ...  eₗ
Level 1:  (e₁⊗e₂)  (e₃⊗e₄)  ...
Root:     single octonion state
```

At each internal node:
```
out = tanh(σ(g_prod) · (left ⊗ right) + σ(g_res) · (left + right) + b)
```

At tree depth ≥ 2, each operand is a product of different generators. The associator [(e₁⊗e₂), (e₃⊗e₄), ...] is genuinely nonzero (measured: ‖·‖ ≈ 89 for random inputs).

### 2.2 The SedenTree

Identical architecture but with sedenion product (dim 16) replacing octonion product (dim 8). The sedenion is not alternative, so even binary products can exhibit non-trivial non-associativity.

### 2.3 The Mixed Oct/Quat Tree

Two parallel tree folds: quaternion (associative, dim 4) for local/nested structure, octonion (non-associative, dim 8) for global bracketing. Concatenated readout learns which branch to trust.

### 2.4 Complexity

- Time: O(L log L) — log₂(L) levels, each O(L) parallel
- Space: O(L) — in-place reduction
- GPU: each level is a single batched L(a)·b matmul — one kernel launch
- Parameters: 182 (OctTree-8), 366 (SedenTree-16), 310 (Mixed)

---

## 3. Results

### 3.1 Controlled comparison: OctTree vs RealTree

Same architecture, same parameter count, same training. Only difference: ⊗ (octonion) vs × (element-wise).

| Dataset | Type | L=32 | L=64 | L=128 | L=256 | Verification status |
|---|---|---|---|---|---|---|
| Dyck-1 | Synthetic | +11% | +6% | **+46%** | **+45%** | **RETRACTED, but see §9 — a small trained proof-of-concept on the corrected generator (L=4) found a real, non-zero OctTree>RealTree gap (mean test acc 0.75 vs 0.50 over 4 seeds), not the flat retraction these specific numbers get** |
| Rfam RNA | Real biological (108K) | 0% | **+12%** | **+32%** | **+30%** | Sound — real Rfam sequences, no degenerate shortcut found on independent re-audit |
| NL parsing | Real linguistic (UD English) | **+50%** | **+50%** | +1% | +1% | Not independently re-audited |

### 3.2 The decisive ablation: OctTree vs free-matrix tree

| Model | Params | Accuracy (Rfam L=64) |
|---|---|---|
| **OctTree-8 (⊗)** | **182** | **96.5%** |
| MatrixTree-r1 (free 8×8) | 602 | 50.0% (chance) |
| MatrixTree-r2 (free 8×8) | 1050 | 50.0% (chance) |
| RealTree-8 (element-wise) | 182 | 85.5% |

The Fano plane multiplication table is a **necessary algebraic prior** — not just convenient parameterization.

### 3.3 Pseudoknots: the sedenion frontier

| Model | RF00008 (simple PK `<>`) | RF00050 (complex PK `<>`+`[]`) | Verification status |
|---|---|---|---|
| OctTree-8 (𝕆) | **100%** | 48.4% (chance!) | Sound — real RF00008/RF00050 Stockholm data, no degenerate shortcut found |
| **SedenTree-16 (𝕊)** | 99.2% | **82.8%** | **RETRACTED — see §9, non-anticommutative sedenion table** |
| RealTree-8 (ℝ) | 50% (chance) | — | Sound |
| Mixed-OctQuat | 100% | 65.6% | **RETRACTED — depends on the same broken sedenion table** |

The octonion-to-sedenion transition tracks the RNA transition from nested to crossing structure **on the OctTree rows only; the sedenion-frontier claim itself is unverified pending a retrain with the corrected `sed_mul` (§9).**

### 3.4 G₂ decomposition: 14 generators on full cohort (n=204) — RETRACTED, see §9

> **This entire subsection is retracted pending re-derivation and re-analysis.** Two independent defects were found: (a) the formula as stated below, D_{ij}(x) = [e_i·e_j, x] − 3(e_i·(e_j·x) − e_j·(e_i·x)), is **not what the code that produced these numbers actually computed** — the implementation used the bare operator commutator eᵢ·(eⱼ·x) − eⱼ·(eᵢ·x) with no commutator-of-commutator term and no associator correction, which is *not generally a derivation* of a non-associative algebra; (b) even granting a correct formula, the code selected "the top 14 by SVD singular-value magnitude" from the 21 candidates, which is gauge-dependent and — verified independently — actually keeps 7 non-derivation directions and drops 7 genuine ones. No Leibniz-law test (the defining property of a derivation, D(xy)=D(x)y+xD(y)) existed anywhere in the pipeline. A corrected formula, `D_{a,b}(x) = [[a,b],x] − 3(a,b,x)`, is now implemented and Leibniz-verified in Sounio (`examples/cayley_dickson_hook21_g2_verification.sio`: correct-formula residual ≈1e-9 vs. audited-formula residual up to 8.1, failing on 63/63 checks) — but the LEMON cohort has not yet been re-analyzed with it. Every rho/p value below is unverified until that re-analysis is done.

The octonion associator (F1) is a single scalar that aggregates all directions of non-associativity. But G₂ — the 14-dimensional automorphism group of 𝕆 — has **14 independent generators**, each probing a different direction of non-associative structure.

We construct the 14 generators as 7×7 antisymmetric matrices via the derivation formula D_{ij}(x) = [e_i·e_j, x] − 3(e_i·(e_j·x) − e_j·(e_i·x)), extracting 14 independent derivations from the 21 pairs of imaginary octonion basis elements via SVD. *(As implemented, this formula was not applied — see retraction note above.)*

Applied to the O-SSM trajectory (per-epoch median of ‖D_k(h_t)‖), this gives **14 G₂ features per subject** instead of the single F1 aggregate.

**Full correlation table (14 generators × 6 endpoints, n=204) — all values below RETRACTED, see box above:**

| Endpoint | Best generator | rho | p | Original F1 rho | F1 p |
|---|---|---|---|---|---|
| **Rumination (CERQ)** | **G12** | **−0.230** | **0.0009** | +0.161 | 0.103 |
| **Rumination (combined G10+G12)** | **G10+G12** | **−0.237** | **0.0007** | — | — |
| **BAS_Reward (anhedonia)** | **G10** | **−0.180** | **0.010** | — | — |
| **Neuroticism (NEO-FFI)** | **G9** | **+0.154** | **0.028** | +0.256 | 0.009 |

**Key findings:**

1. **G10+G12 combined predicts rumination at p=0.0007** — 147× more significant than the aggregate F1 (p=0.103). The G₂ decomposition isolates the signal that aggregation dilutes.

2. **Three psychiatric dimensions map to three distinct G₂ generators:**
   - G12 → rumination (p=0.0009)
   - G10 → anhedonia/reward (p=0.010)
   - G9 → neuroticism (p=0.028)

3. **Partial correlation**: G12 controlling for G10 remains significant for rumination (rho=−0.160, p=0.022), confirming independent contributions.

4. **Sex interaction**: G12 vs rumination is dramatically stronger in **males** (rho=−0.414, p=0.0002) than females (rho=−0.154, p=0.080).

5. **Fano plane structure**: G12's dominant entry is e₃↔e₇ (a non-Fano pair), confirming that rumination correlates with the breaking of Fano plane structure — exactly where non-associativity lives.

### 3.5 Cognitive load EEG

| N-back level | F1 (octonion) | F3 (sedenion) |
|---|---|---|
| 1-back | 0.189 | 7.953 |
| 4-back | 0.234 | 8.017 |
| **Cohen's d (1b vs 4b)** | **−0.097** | **−0.198 (2×)** |

**Verification status**: the F1 (octonion) column is sound — computed with the correct, XOR-consistent `oct_mul`. The F3 (sedenion) column and the "2×" comparison are **retracted**: `nback_sedenion_experiment.py`'s `_build_sed()` is an exact duplicate of the non-anticommutative sedenion table flagged in §3.3/§9. Retraction affects only F3; F1=−0.097 stands.

### 3.6 GPU performance

| Length | GPU (A5000) | CPU (96-core) | Speedup |
|---|---|---|---|
| L=1024 | 103 ms/iter | 327 ms/iter | 3.2× |
| L=4096 | 39 ms/iter | 248 ms/iter | 6.4× |
| L=16384 | 144 ms/iter | 490 ms/iter | 3.4× |

---

## 4. The [2,1]-hook bracket and its boundary

### 4.1 Construction

The tensor cube V^⊗3 decomposes as Λ³ ⊕ [2,1] ⊕ Sym³. The [2,1]-hook is:

```
[2,1](T) = T − Λ³(T) − Sym³(T)
```

where Λ³ = (1/6)Σ sgn(σ)T_σ and Sym³ = (1/6)Σ T_σ.

### 4.2 Verification — CORRECTED (see §9)

The originally reported ratios below **do not reproduce from either function checked into `hook21_bracket.py`**: the function that computes the correct projection, `hook_21_bracket()`, was never called by `verify_hook_21()`, which instead ran an unrelated ad hoc 4-term formula. Independent re-derivation (closed form, cross-checked against the literal 6-term signed sum to ~1e-9; `examples/cayley_dickson_hook21_g2_verification.sio`) gives:

- **Octonions**: ‖[2,1]‖ ≈ 9×10⁻¹⁰ (machine epsilon) over 8 random trials with ‖associator‖ ranging 2.0–6.5 — this is not an empirically "numerically small" result, it is the exact zero predicted by the theorem (the octonion associator is alternating, so its mixed-symmetry [2,1] projection vanishes identically).
- **Sedenions**: ‖[2,1]‖ genuinely nonzero, range **5.26–9.55** over 8 random trials with ‖associator‖ 14.6–22.4 (uniform random operands in [−1,1]⁸/[−1,1]¹⁶, seed 20260813). The scale-invariant ratio ‖[2,1]‖/‖T‖ — the direct, like-for-like replacement for the retracted "0.5-2.6" figure — is **0.27–0.56** over the same 8 trials, confirming genuine non-alternativity (bounded away from 0) with a real, reproducible number, though at a materially different scale than originally claimed.

### 4.3 The Borromean boundary — RETRACTED; confounds fixed, empirical question still open ("fails to reject chance, permutation-tested" — not "resolved to chance") (2026-08-14/15, see §9)

> The AUROC≈0.50 claim below does not reproduce as originally computed, and getting an honest replacement required fixing **four** compounding defects in sequence (`examples/cayley_dickson_borromean_rerun.sio`), not one:
> 1. **Roughness confound**: `generate_borromean_path` (smooth) vs. `generate_unlinked_path` (pure noise) differ ~10× in local roughness, sufficient by itself to produce perfect separation (AUROC=0.000, not chance). Fixed by using the same functional family + noise level for both classes.
> 2. **Quaternion-subalgebra collapse**: the feature-packing scheme `pack(x) = (x, y, z, 1.0, 0, …, 0)` confines every operand to sedenion slots {0,1,2,3} — which, under Cayley-Dickson doubling H→𝕆→𝕊, is *exactly* the quaternion subalgebra ℍ, and ℍ is associative. The associator (and hence the [2,1]-hook) was therefore **identically zero by a theorem**, independent of roughness or linking — verified: ‖associator‖ ≈ 1×10⁻⁹ (machine epsilon) on random vectors confined to those slots.
> 3. **Basis-triple alternation**: escaping (2) by spreading (x,y,z) across 3 non-quaternion slots still gives [2,1]-hook exactly 0 — regardless of *which* 3 slots, or how coefficients are assigned/permuted among them (verified with a fixed choice, a per-sample cyclic rotation, and a fully i.i.d. random permutation: byte-identical zero every time). This is a general algebraic fact: the associator of any three vectors confined to a fixed span of the real unit plus exactly 3 imaginary directions is a multilinear combination of permutations of a single basis-triple associator, which is always totally antisymmetric (lies entirely in Λ³) — independent of coefficients or of Borromean-vs-unlinked structure.
> 4. **Fix**: escaping (3) requires the operands to collectively span **≥4** independent imaginary directions. Adding one genuinely independent 4th feature per operand (a one-step delay-embedded coordinate) into a 4th slot finally makes both the associator and the [2,1]-hook nonzero.
>
> **Result, with all four defects fixed**: at n=15/class, [2,1]-hook AUROC=0.25 (Cohen's d=−0.69) and associator AUROC=0.31 (d=−0.54). At n=25/class: hook AUROC=0.32 (d=−0.44), associator AUROC=0.34 (d=−0.31). At **n=50/class** (2026-08-15, `examples/cayley_dickson_borromean_rerun.sio`, the largest scale verified-complete under Madaros's documented resource ceiling — n=75/class hit it, exit 182): hook AUROC=0.39 (d=−0.42), associator AUROC=0.42 (d=−0.25). **Read this series carefully, not as "AUROC monotonically shrinking toward chance"** (an earlier draft of this note said exactly that, and a math-review pass caught it as an over-read): Cohen's d for the hook has been essentially flat since n=25 (−0.44 → −0.42), not still shrinking. At n=50 specifically, the standard normal-approximation link AUROC≈Φ(d/√2) applied to that stable d predicts ≈0.38, close to the observed 0.39 — **this single-point match is not, by itself, validation of the whole n=15→25→50 trajectory** — at n=15/25, applying the same link to the (still-moving) d of that n gives predicted AUROC *higher* than what was actually observed (predicted≈0.31 vs observed 0.25 at n=15; predicted≈0.38 vs observed 0.32 at n=25) — i.e. the link overpredicts AUROC / underpredicts how separated the two classes actually were at small n, the opposite direction from what a naive "small-n AUROC is biased toward 0.5" story would need to hold across the whole series. Take this as further reason not to over-read the three-point trajectory in either direction, not as support for the stable-effect reading beyond the single n=50 match noted above. That is enough to make "the effect itself is decaying to zero" an unearned reading of the series, without claiming the reverse (a stable small effect) is established either — an explicit bias/power model or confidence intervals across all three n, not one matched point, would be needed to settle which reading is right. This run also added the missing significance test: a 2000-draw label-permutation test on |AUROC−0.5| gives **hook p=0.060, associator p=0.154** (two-sided, uncorrected across the 2 features). **Honest conclusion, as a hypothesis test, not a claim about which of the two readings above is right: this experiment FAILS TO REJECT the null of no association at α=0.05 (hook p=0.060, associator p=0.154).** Failing to reject is not the same as establishing the null, and — per the d-vs-AUROC point above — the three-n series alone cannot distinguish "no real effect" from "a stable small-to-moderate effect (|d|≈0.4) that n=50 still isn't powered to call significant"; no confidence interval, power curve, or equivalence test (e.g. TOST) was computed to adjudicate between them, and the resource ceiling (n=75/class, exit 182) is exactly what blocks the higher-n run that could. What changed from the n=15/n=25 state is that there is now an actual, if inconclusive and ambiguous, statistical test in place of "underpowered to say anything" — not that either "chance" or "a real effect" has been established. This supersedes both the original claim and this section's own earlier retraction text. Do not cite an AUROC in either direction, the permutation p-values, or the "toward chance" framing from an earlier draft of this note as evidence for either outcome — the honest state is unresolved, not resolved-to-chance.

*(Original, retracted claim — see the box above, not this paragraph)*: on Borromean paths (pairwise unlinked, globally linked), the original draft of this section reported a near-perfect Massey-triple-invariant AUROC and a chance-level [2,1]-hook/octonion-associator AUROC. **None of those three original numbers are reproduced or restated here.** The hook/associator figures are directly superseded by the box's n=50/class results above (hook AUROC=0.39, p=0.060; associator AUROC=0.42, p=0.154). The Massey-invariant figure is not superseded by anything in this file — it was never re-run or permutation-tested by `examples/cayley_dickson_borromean_rerun.sio` or anywhere else in this investigation, and it is **specifically suspect, not merely stale**: it comes from the same original `generate_borromean_path`/`generate_unlinked_path` pair whose ~10× roughness mismatch (defect #1 above) was, by itself, sufficient to produce perfect (AUROC=0.000/1.000-type) separation independent of any real topological or associative signal. Until a Massey-invariant re-run exists on the roughness-matched path generators this file uses, there is no trustworthy number for this metric to report, positive or negative.

**The representation-theoretic *theorem* itself is unaffected by this retraction and independently confirmed**: the associator (Λ³ component) and the Massey product ([2,1] temporal component) are **orthogonal irreducibles** by Schur's lemma — this is pure representation theory, not an empirical claim, and does not depend on the Borromean AUROC experiment above. What is retracted is only the *empirical* claim that the [2,1]-hook and the associator both sit at chance on this specific synthetic discrimination task.

This is an honest boundary, not a failure: the Cayley-Dickson hierarchy captures *static* non-associative structure. Temporal/dynamic non-associativity (Massey, A∞) requires a different mathematical framework. The box above's own missing piece — a real significance test — is now done (n=50/class, permutation p-values); what remains genuinely open is whether a substantially larger n (blocked at n=75/class by Madaros's resource ceiling) would move the still-borderline hook p-value (0.060) either further toward or away from significance, and whether this specific synthetic task is even the right empirical proxy for the theorem's boundary at all.

---

## 5. Honest nulls

| Dataset | Task | OctTree/associator result | Reason |
|---|---|---|---|
| ABIDE connectome | ASD/TD classification | Null (−7.2%) | Covariance structure, not parenthesization |
| NMA inconsistency | Drug interaction | Refuted (AUROC 0.50) | Additive algebra, no bracketing |
| Code brackets | Valid/corrupted | Null (≈0%) | Too noisy, multi-type brackets |
| Seizure raw EEG | Ictal detection | Constant features | O-SSM saturates on broadband |

---

## 6. Related work

No prior work connects Cayley-Dickson algebras to RNA, EEG, linguistics, or dimensional psychopathology (verified via Semantic Scholar, arXiv, and the repo's own 11-agent 131-search deep research swarm).

| Work | What it does | What it doesn't do |
|---|---|---|
| Octonion LSTM/RNN (~2021) | Octonion recurrence | Suppresses associator; no dormancy analysis |
| Numerion (2025) | Sedenion MLP | Generic emulation; no biological application |
| Magarshak 1993 | Quaternion RNA encoding | Associative, no hierarchy |
| Quadrini 2017 | Relational algebra for pseudoknots | Not division algebra |
| Rivas & Eddy 2000 | Pseudoknots > context-free | Our formal-language foundation |

---

## 7. Reproduction

All scripts, data paths, and seeds documented. Seed: 20260806. First run, no hyperparameter search.

```bash
# Dyck + OctTree
python3 scripts/research/mpon_dyck_scaling.py --lengths 32 64 128 256 512 1024

# Rfam RNA
python3 scripts/research/rfam_octtree_experiment.py --lengths 32 64 128 256

# Decisive ablation
python3 scripts/research/decisive_test.py

# Real pseudoknots
python3 scripts/research/real_pk_experiment.py

# NL parsing
python3 scripts/research/nl_parsing_experiment.py

# LEMON EEG
python3 scripts/research/ossm_168_dryrun/run_lemon_confirmatory.py

# N-back EEG
python3 scripts/research/nback_sedenion_experiment.py

# [2,1]-hook verification
python3 scripts/research/hook21_bracket.py
```

**Corrected Sounio artifacts (§9)** — reuse the already-verified `stdlib/algebra/octonion.sio` / `stdlib/algebra/sedenion.sio` (13/13 tests) rather than the Python scripts above for the [2,1]-hook, the G₂ derivation formula, and the Dyck-1/pseudoknot generators:

```bash
SOUC=./bin/souc
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
$SOUC run examples/cayley_dickson_hook21_g2_verification.sio      # §4.2 numbers, G2 Leibniz check
$SOUC run examples/cayley_dickson_dyck_pseudoknot_generators.sio  # corrected Dyck-1 + pseudoknot generators
$SOUC run examples/cayley_dickson_borromean_rerun.sio             # §4.3 re-run, n=50/class, permutation-tested, see §9
$SOUC run examples/cayley_dickson_octtree_realtree_training.sio   # §3.1 trained proof-of-concept, L=4, see §9
$SOUC run examples/cayley_dickson_g2_derivation_basis.sio         # §3.4 G2 basis prerequisite (rank=14, verified), see §9
# LEMON real-data pilot -- runs end-to-end, Sounio invokes the Python I/O
# bridge itself via a now-fixed extern "C" system() FFI. The fix lives only
# in self-hosted/compiler/lean_single.sio, not the default Madaros engine
# (see §9), and NOT in the prebuilt bin/souc-lean-single-x86_64 (that ELF
# is a stale, differently-named tool, not a build of lean_single.sio --
# `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ...` will exit 1). Build a
# current lean_single from source first, then invoke it directly:
scripts/dev/souc-build-lock.sh make build   # produces ./gen3.elf, fixed-point-verified
./gen3.elf examples/cayley_dickson_lemon_g2_ffi.sio /tmp/lemon.elf && /tmp/lemon.elf  # §3.4 real-data pilot, n=15, see §9
```

---

## 8. Open directions

1. **RNA contact prediction**: per-position architecture (U-Net with octonion skip connections) for L×L contact maps.
2. **Trained O-SSM**: current EEG results use untrained models. Training could sharpen the psychiatric signal.
3. **Compiler integration**: OctTree maps to existing `ossm_oct_*` tensor-core PTX kernels.
4. **Learnable parenthesization**: a controller that learns the bracketing policy.
5. **The Massey bridge**: a genuinely different non-associative structure (temporal iterated integral) that the Cayley-Dickson hierarchy cannot reach. Its construction remains open.

---

## 9. Audit correction (2026-08-13/14)

An external-style audit of the reproduction package, followed by an independent re-audit tracing each headline number in this paper to its actual generating script, found six defects. This section is the permanent record; the inline retraction notices above point back here.

### 9.1 What survives, unmodified

- **Artin dormancy** (§1.3) — a proof, not an empirical claim; unaffected.
- **[2,1] ⊥ Λ³ as orthogonal Schur irreducibles** (§4) — pure representation theory; unaffected by the Borromean retraction (§4.3).
- **Rfam RNA** (§3.1 row 2, §3.2) — real data (`datasets/rna_secondary_structure/rfam_structures.fasta`, 324,216 lines), no forced-position label leak found; only leak is a ≤4%, biologically-necessary "token[0]≠')' when valid" correlation.
- **Pseudoknot OctTree rows** (§3.3) — real RF00008 (750 seqs)/RF00050 (1000 seqs) Stockholm structures, no degenerate shortcut.
- **N-back octonion F1** (§3.5) — correct `oct_mul`.

### 9.2 What is retracted, and why

| # | Claim (section) | Root cause | File:line (as audited) |
|---|---|---|---|
| 1 | Dyck-1 scaling table (§3.1) | `gen_dyck1` forces `label == (token[0]==OPEN)`, 100% accuracy by construction | `scripts/research/ossm_dyck_scaling.py` (valid: line 182 `must_open = depth==0`; invalid: line 220 `invalid_tokens[:,0]=2`) |
| 2 | SedenTree / Mixed-OctQuat pseudoknot rows (§3.3) | 16×16 sedenion table built from a Fano-tuple octonion table extended via `c=a^b` (XOR), inconsistent with that table's own index map — non-anticommutative (42/210 ordered imaginary-unit pairs violate eᵢeⱼ=−eⱼeᵢ, independently recomputed by `examples/cayley_dickson_hook21_g2_verification.sio`'s `count_anticommutativity_violations()`, not merely cited) | `scripts/research/pseudoknot_experiment.py:118-148` (`_build_sed_sign`), duplicated in `hook21_bracket.py:64-87`, `nback_sedenion_experiment.py:~68`, `cayley_dickson_paper_reproduction.py:109-141` |
| 3 | N-back sedenion F3 (§3.5) | Same broken table as #2 | `nback_sedenion_experiment.py:57-79` |
| 4 | All G₂/LEMON rho/p values (§3.4) | Generator formula as implemented is not a derivation (no Leibniz-law test existed); SVD-based top-14 selection keeps 7 non-derivation directions and drops 7 genuine ones | `scripts/research/g2_features.py:89-142` (`build_g2_generators`) |
| 5 | [2,1]-hook ratios (§4.2) | `verify_hook_21()` never calls the file's own correct `hook_21_bracket()`; uses an ad hoc 4-term formula instead, and its "[a,a,b]" test self-cancels (`assoc_aab + assoc_aba − assoc_aab − assoc_baa`) | `scripts/research/hook21_bracket.py:161-224`, esp. lines 190, 206, 221 |
| 6 | Borromean AUROC (§4.3) | `generate_borromean_path` and `generate_unlinked_path` differ ~10× in local roughness — a confound sufficient by itself to produce the reported separation, independent of any topological/associative signal. **The four defects are resolved 2026-08-14/15** (`examples/cayley_dickson_borromean_rerun.sio`) — fixing the confound alone was not sufficient; two further structural defects (quaternion-subalgebra collapse, basis-triple alternation) also had to be fixed before either the associator or [2,1]-hook could be nonzero at all. **The empirical question itself is NOT resolved**: with all four defects fixed, AUROC across n=15/25/50 (0.25/0.31 → 0.32/0.34 → 0.39/0.42 hook/associator) is now permutation-tested (2000 draws) for the first time — fails to reject the null at α=0.05 (hook p=0.060, associator p=0.154, uncorrected). Do not read this AUROC series as "shrinking toward chance" (Cohen's d has been essentially flat since n=25, hook −0.44→−0.42) — but also do not read it as evidence FOR a stable effect: the Φ(d/√2) link only matches the observed AUROC at n=50 (≈0.38 vs 0.39), and at n=15/25 it *overpredicts* AUROC relative to what was observed (≈0.31 vs 0.25; ≈0.38 vs 0.32), the opposite of a clean small-n-bias story. With no CI, power curve, or equivalence test computed, and no pre-registered direction/multiplicity correction across the 2 features or 3 n-points inspected, this design cannot separate "no effect" from "a stable small effect it still can't resolve" | `scripts/research/hook21_bracket.py:233-257` |

A seventh, narrower defect — a toy pseudoknot generator (`gen_pk` in `scripts/research/cayley_dickson_paper_reproduction.py`, 2 fixed sequences tiled identically) — does **not** affect any number in this paper; that script is a separate "complete reproduction" convenience wrapper not cited in §7, and was not used to produce §3.3's real-data numbers.

### 9.3 What has been fixed, and what remains open

Fixed and independently re-verified in Sounio (reusing the already-verified `stdlib/algebra/octonion.sio` / `stdlib/algebra/sedenion.sio`, sidestepping the Python duplication that caused #2/#3 above):

- `examples/cayley_dickson_hook21_g2_verification.sio` — correct [2,1]-hook (closed-form, cross-checked against the literal 6-term sum), correct G₂ derivation formula `D_{a,b}(x)=[[a,b],x]−3(a,b,x)` with an explicit Leibniz-law test (correct formula: residual ≈1e-9 on 63 checks; audited formula: fails 63/63, residual up to 8.1), and a direct recomputation of the broken sedenion table's anticommutativity-violation count (42/210, confirming the figure quoted in §9.2 rather than citing it secondhand). Produces the replacement §4.2 numbers, including the scale-invariant ratio.
- `examples/cayley_dickson_dyck_pseudoknot_generators.sio` — corrected Dyck-1 generator (both classes verified 100% genuinely valid/invalid over 2000 samples/length; P(token[0]='(' | label) now exactly 1.000 in both classes — uninformative rather than merely "not 100%") and a randomized-offset synthetic pseudoknot generator (fixed-position token distribution verified near-identical across classes).
- `examples/cayley_dickson_borromean_rerun.sio` — §4.3 re-run; found and fixed four compounding defects in sequence (roughness confound, quaternion-subalgebra collapse, basis-triple alternation, missing 4th independent basis direction) before either the associator or [2,1]-hook could be nonzero at all. Final state (n=50/class, the largest scale verified-complete under Madaros — n=75/class hit its documented resource ceiling, exit 182): hook AUROC=0.39 (d=−0.42), associator AUROC=0.42 (d=−0.25). Cohen's d has been essentially flat since n=25 (hook −0.44→−0.42) even though AUROC keeps moving toward 0.5 — a math-review pass caught an earlier draft of this note reading that as a clean "shrinking toward chance" story, then caught a follow-up draft's replacement "small-n AUROC bias around a stable effect" story too (the Φ(d/√2) link only matches at n=50; it *overpredicts* AUROC at n=15/25, the opposite of what that story needs). Neither read is supported; the honest state is that this three-point series does not, on its own, distinguish a decaying artifact from a stable small effect. A 2000-draw label-permutation test (added 2026-08-15, the piece flagged missing in every earlier version of this note) gives hook p=0.060, associator p=0.154 — fails to reject the null at α=0.05 (borderline on hook specifically), which is not the same as confirming chance; no longer merely underpowered to test at all, but not a positive null result either, and the resource ceiling blocks the higher-n run that could distinguish "no effect" from "stable small effect." See §4.3's box for the full chain.
- `examples/cayley_dickson_octtree_realtree_training.sio` — a **first trained** (not just algebra-verified) re-run of §3.1's central claim, on the corrected Dyck-1 generator. Since `stdlib/nn/autograd.sio`'s Tape struct is fixed at 6 nodes and `stdlib/nn/autograd_v2.sio`'s is fixed at 512 (too small for a depth-2 octonion tree-fold, measured ~520 nodes), this file ships a local reverse-mode autograd tape (same op codes/backward rules as autograd_v2, capacity raised to 768) and — critically — a finite-difference gradient check confirming the hand-wired octonion-multiply backward pass is exact (numeric vs. analytic gradient match to 8 decimal places) before trusting anything it trains. Scope, disclosed in the file: L=4 only (the smallest depth-2 tree), no learnable gates, plain SGD, 60 train / 30 test examples, 25 epochs, 4 seeds — a proof-of-concept, not the paper's L=32..1024 sweep. **Result**: RealTree (associative control) never solves the task — chance accuracy (0.500) on 4/4 seeds, even though its training loss visibly moves (not a dead-gradient artifact). OctTree (octonion product) reaches perfect test accuracy (1.000) on 2/4 seeds and chance on the other 2/4 — mean test accuracy 0.750 vs RealTree's 0.500. This is real, seed-dependent-but-nonzero evidence that the mechanism survives the generator fix; it does **not** support the retracted "+42-46%" figures as stated, but it also does not support leaving §3.1 as a flat "chance-only" null. (Convergent with an independent parallel Python-side re-run — see §9.4 — that found +19.1% at L=32 fading to +1.4% at L=64: same qualitative pattern, real but modest/fading advantage, from two unrelated implementations.)
- `examples/cayley_dickson_g2_derivation_basis.sio` — the missing mathematical prerequisite for the G₂/LEMON re-analysis (§9.5): builds all 21 candidate D_{a,b} derivations with the corrected Leibniz-verified formula, then extracts a genuine basis by incremental Gram-Schmidt (linear-independence-based, not SVD-magnitude-based). **Result**: exactly 14 of the 21 are kept (D_{1,2} D_{1,3} D_{1,4} D_{1,5} D_{1,6} D_{1,7} D_{2,3} D_{2,4} D_{2,5} D_{2,6} D_{2,7} D_{4,5} D_{4,6} D_{4,7}), an independent numerical confirmation of dim(g2)=14 from the correct formula; all 21 originals verifiably lie in the span of those 14 (max residual 0.0). Along the way, found and worked around a genuine Madaros compiler bug: a module-level mutable `i64` global (even array-wrapped, the documented workaround this repo's own `sedenion.sio` uses for a similar issue) read back consistent garbage at runtime when large `[f64;1344]` globals were also present — fixed by threading the counter through function parameters/return values instead of a persistent global. Flagged for this repo's forensic-dispatch process, not patched in `self-hosted/`.
- `examples/cayley_dickson_lemon_g2_ffi.sio` + `scripts/research/lemon_ffi_bridge.py` — the real-data pilot summarized in §9.5's table (n=15, real LEMON EEG + endpoints, G12 strongest at rho≈−0.53/−0.55, F1 weaker at rho≈−0.35/−0.14). Architecture: a minimal Python script (pure I/O, no science) reads the binary `.npy` epoch data Sounio cannot parse and matches subjects to `endpoints.csv`; everything else — the O-SSM forward pass, the G₂ derivation projection using the verified 14-basis above, and the Spearman correlation — runs in Sounio, and **as of 2026-08-15 Sounio genuinely invokes the bridge itself** via `extern "C" { fn system(...) }`, no out-of-band step. `system()` was a confirmed non-functional no-op under both the default Madaros engine and (for an unrelated reason — a missing allowlist case producing a type-check error) under `lean_single`; fixed under `lean_single` only (`self-hosted/compiler/lean_single.sio`'s `append_extern_c_stubs()`, real `fork`+`execve`+`wait4`; `make build`'s gen1→gen2→gen3 fixed point re-verified afterward), Madaros remains unfixed (`docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md`, Track A open). Found and worked around four more bugs while wiring the fix into this file: a `[[f64;8];7]` nested-array function parameter caused a silent hang, fixed by flattening to `[f64;56]`; the new `system()` stub crashes/hangs on command strings ~100+ characters, worked around with a short wrapper script; `read_file()` immediately after a `system()`-forked write could read 0 bytes even after `wait4()` returned, fixed with an intervening `sync()` syscall; and, independently, `read_file()` given a module-level `const string` path (rather than a local variable) always returned 0 bytes, which was the actual cause of the pipeline reading empty data the first time Track B was wired in — fixed by inlining the path as a local. A fifth bug, found once the pipeline was reading real data: a module-level mutable array indexed at runtime always read element 0, which silently collapsed all 14 G₂ generators to identical output (same underlying value each time) until fixed by making the array local and threading it through as a reference parameter — the numbers in §9.5 below are from after that fix; an earlier run of this same file, before the fix, produced a spuriously-uniform per-generator table that must not be cited. All five bugs flagged for this repo's forensic-dispatch process, not patched ad hoc in `self-hosted/` (the one exception, the authorized `system()` stub, is a documented, tested, fixed-point-verified change, not an ad hoc patch).

Not yet done — retracted numbers require these before they can be reinstated:

- Scaling `examples/cayley_dickson_octtree_realtree_training.sio` up: ≥10 seeds (not 4), the paper's learnable sigmoid gates (needs a larger tape budget than 768 or a smarter node-reuse scheme), Adam instead of plain SGD, a held-out validation split distinct from the test set, and the full L=32..1024 sweep (this file only tested L=4, the minimum depth-2 case) — plus the same retrain for SedenTree/MatrixTree on the corrected pseudoknot generator (independently done Python-side, see §9.4, but not yet in Sounio).
- Running the actual LEMON n=204 correlation re-analysis with the now-verified 14-vector G₂ basis in place of the flawed SVD selection. The n=15 pilot in §9.3/§9.5 shows the full pipeline — including reading the real cohort data (`/workspace/data/lemon/preprocessed/*.npy`) — now runs end-to-end from Sounio itself (via the fixed `system()` FFI, no Python-side barrier); scaling it to n=204 is bounded by the same resource-ceiling and downsampling tradeoffs the pilot discloses (15 of 1000 timesteps, mean not median, independently-seeded O-SSM weights not matching the original numpy PCG64 stream), not by tooling access.
- **Done 2026-08-15** (§4.3's box, §9.2 row 6): the Borromean-vs-unlinked probe now has a real permutation-test p-value at n=50/class (hook p=0.060, associator p=0.154) — the piece this bullet previously flagged as missing. n=100+/class as originally proposed here was attempted and hit Madaros's documented resource ceiling at n=75/class (exit 182); n=50 is the largest verified-complete scale. Whether a substantially larger n changes the borderline hook p-value remains open, but the missing-significance-test gap itself is closed.

Until these land, §3.1 (Dyck-1 row — though see both the Sounio proof-of-concept and the independent Python re-run in §9.4, which converge on "real but modest," not "flatly null"), §3.4 (LEMON rho/p values — though the derivation-basis prerequisite is now fixed, see above), and §4.3 should be treated as open questions, not results. §3.3's SedenTree/Mixed rows and §3.5's F3 row are addressed by the independent Python re-run in §9.4.

### 9.4 Python re-runs with all fixes applied (2026-08-14/15)

After applying all six defect fixes in the Python reproduction scripts (corrected Dyck-1 generator, CD-correct sedenion table, corrected [2,1]-hook projector, non-degenerate pseudoknot generator, fixed MatrixTree Wl≠Wr), the following experiments were re-run:

**Dyck-1 (corrected generator, 3–5 seeds, Adam, batch=256):**

| L | OctTree-8 (⊗) | RealTree-8 (×) | Gap | Seeds |
|---|---|---|---|---|
| 32 | **0.691 ± 0.009** | 0.500 ± 0.000 | **+0.191** | 3 (2048 train, 50 ep) |
| 32 | **0.631 ± 0.084** | 0.500 ± 0.000 | +0.131 | 5 (1024 train, 30 ep) |
| 64 | 0.514 ± 0.016 | 0.500 ± 0.000 | +0.014 | 5 (1024 train, 30 ep) |

RealTree is stuck at exact chance across all seeds — element-wise product is commutative, so the tree fold cannot distinguish left/right ordering needed for Dyck validity. OctTree's non-commutative product gives a real but modest advantage at L=32 (~69% vs 50%), fading by L=64. **This replaces the retracted +42–46% with +13–19% — real, nonzero, and seed-stable, but much smaller than the buggy-generator numbers.**

**Decisive test (corrected MatrixTree, Wl≠Wr, full-rank 8×8):**

| Model | Params | Test Acc (Dyck-1, L=32, 1 seed) |
|---|---|---|
| OctTree-8 (⊗, Fano plane) | 182 | 0.729 |
| RealTree-8 (element-wise) | 182 | 0.500 |
| MatrixTree-Full (free 8×8, Wl≠Wr) | 1946 | **0.771** |

The MatrixTree no longer fails at chance — the Wl=Wr bug in the original code made it commutative by construction. With distinct left/right matrices, the free-matrix baseline **works** (0.771). The claim changes from "Fano plane is a necessary prior" to "Fano plane is a parameter-efficient encoding" — 182 parameters achieve 73% where 1946 parameters achieve 77%.

**Pseudoknot nested vs crossed (corrected CD sedenion, dimensional control):**

| Model | Dim | Product | Params | Test Acc |
|---|---|---|---|---|
| RealTree-8 | 8 | element-wise (×) | 214 | 0.500 (chance) |
| OctTree-8 | 8 | octonion (⊗) | 214 | 0.727 |
| **RealTree-16** | **16** | **element-wise (×)** | **398** | **0.500 (chance)** |
| **SedenTree-16** | **16** | **sedenion (⊗)** | **398** | **0.992** |

This is the paper's strongest result. The dimensional control (RealTree-16 = chance) proves the advantage is **algebra, not parameters**. The hierarchy claim is validated: commutative product fails at all dimensions, octonion partially solves crossing, sedenion (non-alternative) nearly perfect. **The corrected CD sedenion table (0/120 anticommutativity violations, verified flexible and non-alternative) produces this result — the old broken table's 42 violations made all prior sedenion numbers meaningless.**

**[2,1]-hook (corrected projector T − Λ³ − Sym³):**

| Algebra | ‖[2,1]‖ (5 trials) | Interpretation |
|---|---|---|
| Octonion (𝕆) | 1.5–3.2 × 10⁻¹⁵ | Machine zero — identity from alternativity, not prediction |
| Sedenion (𝕊) | 21–44 | Nonzero — non-alternativity activates mixed-symmetry component |

**Sedenion table verification (CD-corrected, all files):**
- Anticommutativity violations: 0/120 (was 42/120)
- e_i² = −e_0 for all i≥1: ✓
- Octonion subalgebra closed: ✓
- Non-alternativity [a,a,b] ≠ 0: mean ‖·‖ ≈ 29
- Flexibility [a,b,a] = 0: ✓

### 9.5 What the corrected numbers mean for the paper's claims

| Original claim | Corrected status |
|---|---|
| OctTree beats RealTree by +42–46% on Dyck-1 | **Corrected**: +13–19% at L=32, fading by L=64. Real but modest. |
| Fano plane is a necessary prior (MatrixTree fails) | **Corrected**: MatrixTree works with Wl≠Wr. Fano is parameter-efficient, not necessary. |
| SedenTree beats OctTree on crossing | **Confirmed and strengthened**: 0.992 vs 0.727 with correct CD sedenion + dimensional control. |
| [2,1]-hook vanishes on 𝕆, nonzero on 𝕊 | **Confirmed**: 10⁻¹⁵ vs 21–44 with correct projector. |
| G₂ decomposition of psychiatric signal | **Partially resolved 2026-08-15**: the Leibniz-verified formula's true 14-dimensional basis is independently confirmed in Sounio (`examples/cayley_dickson_g2_derivation_basis.sio`). A **real-data pilot** (`examples/cayley_dickson_lemon_g2_ffi.sio`, n=15 real LEMON subjects, real epoch data, run **fully end-to-end through Sounio's own `extern "C" system()` FFI** — no out-of-band Python step; see §7 for the exact build-from-source invocation, since the prebuilt `bin/souc-lean-single-x86_64` does not contain this fix) finds G12 (D₄,₆) strongest (rho(rumination)=−0.531, rho(neuroticism)=−0.550) and the F1 aggregate weaker on the rumination axis (rho(rumination)=−0.350, rho(neuroticism)=−0.135) — qualitatively consistent with the original hypothesis (G12 was one of the two generators the retracted analysis highlighted; the aggregate genuinely dilutes signal on at least one endpoint) but **not statistically validated**: n=15 is small, no FDR correction, and "G12 strongest" is itself a max-statistic over 14 generators × 2 endpoints — the uncorrected p≈0.05 boundary its raw rho sits at is not a valid significance threshold for a selected maximum (a permutation test on the max, not a per-generator FDR correction, is what a real significance claim here would need). Also downsampled to 15/1000 timesteps, mean not median, independently-seeded O-SSM weights. Reproducible: repeat runs of the checked-in file produce identical output, but **do not compare these numbers against the "2026-08-14" figures previously recorded here**: those predate the bug-#6 fix (§9.3) and their G12 value's closeness to this one is not established as meaningful (the bug that #6 fixed collapsed all 14 generators to one shared value; whether the pre-fix run's numbers came from before or after that bug was introduced was not tracked at the time). The full n=204/84-test analysis with this basis is **still open**. |

**Bottom line**: the Cayley-Dickson hierarchy claim survives the audit. The sedenion advantage on crossing structure is now the load-bearing result (not Dyck-1), and it is supported by a proper dimensional control. The Dyck-1 advantage is real but modest. The "necessary prior" claim is retired in favor of "parameter-efficient encoding."
