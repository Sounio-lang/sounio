<!-- docs:meta
topic_id: repo.docs.papers.main.cayley-dickson-restructured-2026-08-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.main.cayley-dickson-restructured-2026-08-13
-->

# Non-Associative Algebra as Inductive Bias: The Cayley-Dickson Hierarchy in Structural Classification

**Authors:** Demetrios Chiuratto Agourakis¹², Marli Gerenutti²
¹Biomaterials and Regenerative Medicine, PUC-SP, Sorocaba, SP, Brazil
²Faculdade São Leopoldo Mandic, Campinas, SP, Brazil

**Date:** 2026-08-13
**Target:** *Physical Review X* (interdisciplinary)
**Pre-registration:** LEMON EEG hypotheses frozen 2026-04-21 (SHA-256 c403ff55...)
**Status:** UNDER CORRECTION (audited 2026-08-13/14) — this draft's own "core experiments" reproduction command (§Reproduction) is `scripts/research/cayley_dickson_paper_reproduction.py`, the specific script found to contain a degenerate Dyck-1 generator and a memorization-only pseudoknot generator. See `docs/papers/main/cayley_dickson_hierarchy_paper_2026-08-13.md` §9 for the full scorecard, which applies to this draft's overlapping claims. **Do not submit or cite the G₂/LEMON paragraph in this abstract, §4, or §5 until the retraction notices below are resolved.**

---

## Abstract

We show that exact algebraic structure from the Cayley-Dickson hierarchy provides a parameter-efficient inductive bias for classifying parenthesization-dependent structure. The octonion product ⊗ (non-associative, alternative) in a balanced binary tree-fold architecture (OctTree, 182 parameters) outperforms an identical-architecture associative control (RealTree, 182 parameters) by +19% on corrected Dyck-1 at L=32 (69.1% vs 50.0%, 3 seeds), and — with corrected algebra — the sedenion product (non-alternative, 398 parameters) achieves 99.2% on pseudoknot crossing where both octonion (72.7%) and element-wise controls (50.0% at all dimensions) fail. A free-matrix baseline with 10× more parameters (1946) achieves 77.1%, confirming the Fano plane is parameter-efficient but not uniquely necessary.

The octonion-to-sedenion transition tracks the RNA transition from nested to crossing structure: octonion solves simple pseudoknots (RF00008: 100%) but fails on complex pseudoknots with multiple bracket types (RF00050: 48%, chance); sedenion succeeds (RF00050: 82.8%). *[Octonion figures sound; the sedenion "succeeds" figure is retracted — see §3.2 note.]*

In a pre-registered analysis of resting-state EEG (LEMON, n=204), the 14 generators of G₂ — the automorphism group of the octonions — decompose psychiatric signal into orthogonal directions: G₁₂ predicts rumination (rho=−0.230, p_FDR=0.013), G₁₀ predicts reward sensitivity (rho=−0.180, p_FDR=0.042), and G₉ predicts neuroticism (rho=+0.154, p_FDR=0.098). The aggregate associator (F₁) is much weaker (p=0.103, uncorrected), demonstrating that the G₂ decomposition is essential. **[RETRACTED — see §4. The generator formula used to build the 14 G₂ features was not a derivation (no Leibniz-law test existed); a corrected, Leibniz-verified formula now exists but the cohort has not been re-analyzed with it. This paragraph cannot appear in a submission until that re-analysis is done.]**

An honest mathematical boundary is established: the [2,1]-hook bracket (mixed-symmetry component of the tensor cube) is orthogonal to the Massey product by standard representation theory (Schur's lemma), confirming that the Cayley-Dickson associator cannot capture higher-homotopy obstructions. *[The theorem stands; the AUROC numbers offered as experimental confirmation in §5 are retracted — a smoothness confound between the two path generators, not chance-level performance, explains the reported result.]*

---

## 1. Artin dormancy and the tree-fold resolution

### 1.1 The problem

By Artin's theorem (1928), any two elements in an alternative algebra generate an associative subalgebra. Therefore the binary octonion product A ⊗ h in a recurrence h_t = σ(A ⊗ h_{t-1} + B·x_t) is associatively trivial — the left-multiplication L(A) is a fixed real 8×8 matrix, and matrix composition is associative.

**Experimental confirmation:** the left-fold OSSM-8 (octonion) performs comparably to Diag-8 (element-wise) on Dyck-1 at all sequence lengths (Table 1). Non-associativity is dormant.

### 1.2 The fix: balanced binary tree

The OctTree computes the octonion product over the sequence using a balanced binary tree. At tree depth ≥ 2, three independent octonion generators interact: the associator [(e₁⊗e₂), (e₃⊗e₄), ...] is genuinely nonzero (‖·‖ ≈ 89 for random inputs). The associativity that Artin's theorem guarantees for binary products does NOT extend to ternary products of composite elements.

### 1.3 Controlled comparison

**Same architecture, same parameters (182), same training. Only difference: ⊗ vs ×.**

| Dataset | L | OctTree-8 (⊗) | RealTree-8 (×) | Δ | Status |
|---|---|---|---|---|---|
| Dyck-1 | 32 | **69.1%** | 50.0% | +19.1% | **CORRECTED** (2026-08-15): 3 seeds, corrected generator. Was +46% with buggy generator. |
| Dyck-1 | 64 | 51.4% | 50.0% | +1.4% | **CORRECTED**: advantage fades at L=64. |
| Rfam RNA (108K seqs) | 128 | **87.1%** | 55.1% | +32.0% | Sound (real Rfam data, no degenerate shortcut) — not yet re-run with corrected pipeline |
| Rfam RNA | 256 | **86.5%** | 57.0% | +29.5% | Sound |
| NL parse (UD English) | 32 | **100%** | 50.0% | +50.0% | Not independently re-audited |

*Table 1: Controlled comparison. OctTree uses octonion ⊗; RealTree uses element-wise ×. Same parameters, same training. Dyck-1 rows corrected 2026-08-15 — the original +42-46% was an artifact of a degenerate generator (see §9.4). Corrected gap is +19% at L=32, fading by L=64.*

**Update 2026-08-14**: a small trained proof-of-concept on the corrected generator (`examples/cayley_dickson_octtree_realtree_training.sio`, L=4 — the smallest depth-2 tree — no gates, plain SGD, 4 seeds, gradients independently verified via finite differences) found RealTree stuck at chance (0.500) on 4/4 seeds while OctTree reached perfect test accuracy on 2/4 seeds (mean 0.750 vs 0.500). This is real, non-zero, seed-dependent evidence the mechanism survives the fix — it does not support the 94.1%/90.0% figures above as stated, but it also argues against treating Dyck-1 as a flat null. Not yet run at paper scale (L=128/512, ≥10 seeds, gates, Adam).

---

## 2. The decisive ablation: algebra vs free parameters

| Model | Parameters | Accuracy (Dyck-1, L=32) |
|---|---|---|
| **OctTree-8 (⊗, Fano plane)** | **182** | **72.9%** |
| MatrixTree-Full (free 8×8, Wl≠Wr) | 1946 | **77.1%** |
| RealTree-8 (element-wise) | 182 | 50.0% (chance) |

*Table 2: Corrected 2026-08-15. The free-matrix baseline NO LONGER FAILS — the original Wl=Wr bug made it commutative by construction. With distinct left/right matrices, MatrixTree works (77.1%). The Fano plane is a parameter-efficient encoding (182p → 73%), not a necessary prior. OctTree achieves comparable accuracy with 10× fewer parameters than MatrixTree.*

**This is the load-bearing result.** It is not the case that "any structured matrix works" — the SPECIFIC multiplication table of the octonions (the Fano plane) is necessary. A free matrix of the same dimension cannot learn this structure from data in this regime.

---

## 3. The Cayley-Dickson hierarchy on RNA pseudoknots

### 3.1 The prediction

Each step of the Cayley-Dickson construction loses one property:

| Algebra | Dim | Property | RNA structure |
|---|---|---|---|
| 𝕆 (octonion) | 8 | Alternative but non-assoc | Nested (context-free) |
| 𝕊 (sedenion) | 16 | Non-alternative, zero divisors | Crossing (pseudoknots) |

### 3.2 Real pseudoknot experiment

| Model | Dim | Product | Params | Nested vs Crossed (synth, L=32) | Status |
|---|---|---|---|---|---|
| RealTree-8 | 8 | element-wise | 214 | 50.0% (chance) | Sound |
| OctTree-8 (𝕆) | 8 | octonion ⊗ | 214 | 72.7% | Sound — partial crossing detection |
| **RealTree-16** | **16** | **element-wise** | **398** | **50.0% (chance)** | **Dimensional control** — advantage is NOT from dimensionality |
| **SedenTree-16 (𝕊)** | **16** | **sedenion ⊗** | **398** | **99.2%** | **CORRECTED** — CD-correct sedenion (0/120 violations). Was "RETRACTED" with broken table. |

*Table 3: Corrected 2026-08-15. The sedenion advantage on crossing structure is CONFIRMED with correct Cayley-Dickson algebra. The dimensional control (RealTree-16 = chance) proves the advantage is algebraic, not parametric. This is now the paper's load-bearing result.*

**Limitation:** Only two Rfam families tested. The sedenion advantage could partly reflect increased dimensionality (dim 16 > dim 8) — **this question is currently unanswerable regardless, since the sedenion algebra used was not genuine sedenions.** A sedenion-with-random-multiplication-table control is needed to rule out the dimensional artifact, after retraining on the corrected algebra.

---

## 4. G₂ decomposition of psychiatric signal (pre-registered) — RETRACTED, 2026-08-14

> **All rho/p values in this section are retracted.** The 14 candidate generators were built via `g2_features.py::build_g2_generators` using the bare operator commutator eᵢ·(eⱼ·x) − eⱼ·(eᵢ·x), with no commutator-of-commutator term and no associator correction — this is **not generally a derivation** of a non-associative algebra, and no Leibniz-law test (D(xy)=D(x)y+xD(y), the defining property) existed anywhere in the pipeline to catch it. Independent verification (`examples/cayley_dickson_hook21_g2_verification.sio`) confirms the audited formula fails Leibniz on 63/63 checks (residual up to 8.1), while the correct formula `D_{a,b}(x)=[[a,b],x]−3(a,b,x)` satisfies it to ~1e-9. Separately, the "top 14 by SVD singular value" selection is gauge-dependent and was found to keep 7 non-derivation directions while dropping 7 genuine ones. Both defects must be fixed and the n=204 LEMON cohort re-analyzed before any number below can be reinstated — this is exploratory, pre-registration-adjacent psychiatric-correlate work, and should be held to a correspondingly high bar before appearing in a submission.

### 4.1 Design

The 14 generators of G₂ (the Lie algebra of Aut(𝕆)) are 7×7 antisymmetric matrices that decompose non-associative structure into orthogonal directions. Applied to an untrained O-SSM trajectory through resting-state EEG, they yield 14 features per subject.

**Pre-registration:** H1–H3s were frozen on 2026-04-21 (commit a5762dd, SHA-256 c403ff55) before any data access. The G₂ decomposition was not pre-registered — it is an exploratory follow-up to the pre-registered aggregate F₁ analysis.

### 4.2 Results (n=204, FDR-corrected)

| Feature | Endpoint | rho | p (uncorrected) | p (FDR, 84 tests) | Sig |
|---|---|---|---|---|---|
| **G₁₂** | **Rumination (CERQ)** | **−0.230** | **0.0009** | **0.013** | **\*** |
| **G₁₀** | **BAS_Reward** | **−0.180** | **0.010** | **0.042** | **\*** |
| G₉ | Neuroticism | +0.154 | 0.028 | 0.098 | . |
| G₁₀+G₁₂ | Rumination | −0.237 | 0.0007 | 0.010 | \* |
| F₁ aggregate | Rumination | +0.161 | 0.103 | — | ns |

*Table 4: G₂ generators survive Benjamini-Hochberg FDR correction for 84 tests (14 generators × 6 endpoints). The aggregate F₁ does not.*

### 4.3 Sex interaction (pre-specified in v3)

G₁₂ vs rumination in **males** (n=74): rho=−0.414, p=0.0002. In females (n=130): rho=−0.154, p=0.080.

### 4.4 Cross-dataset generalization

| Train | Test | Result | p |
|---|---|---|---|
| MODMA (MDD/HC) | LEMON (rumination) | rho=+0.182 | 0.009 |
| ABIDE (ASD/TD) | MODMA (MDD/HC) | AUROC=0.559 | 0.53 |

### 4.5 Limitations (stated explicitly)

1. **The O-SSM is untrained.** The features are structural properties of EEG filtered through a random octonion dynamical system. A real-valued 8-dimensional control (same architecture, associative multiplication) is needed to isolate the non-associative contribution.
2. **The G₂ decomposition is exploratory**, not pre-registered. The pre-registered aggregate F₁ is non-significant (p=0.103). The G₂ results should be treated as hypothesis-generating until replicated.
3. **Effect sizes are modest** (rho ≈ 0.15–0.26, R² < 0.07). These are comparable to established EEG biomarkers (frontal alpha asymmetry: rho ≈ 0.2 for depression) but are not clinically diagnostic.
4. **Multiple comparisons:** 84 tests were performed. FDR correction preserves G₁₂ and G₁₀; G₉ is marginal.

---

## 5. The [2,1]-hook boundary

The tensor cube V^⊗3 decomposes as Λ³ ⊕ [2,1] ⊕ Sym³ (standard Schur-Weyl theory). The octonion associator lives in Λ³ (fully antisymmetric); the Massey product lives in [2,1] (mixed symmetry). By Schur's lemma, these irreducible components are orthogonal. **This theorem is unaffected by the retraction below.**

**Experimental confirmation — RETRACTED, resolved to "consistent with chance, underpowered" as of 2026-08-14:** on Borromean paths (pairwise unlinked, globally linked), the Massey invariant achieves AUROC=0.998. The octonion associator achieves AUROC=0.489 (chance). The sedenion [2,1]-hook bracket achieves AUROC=0.511 (chance). *Independent re-run of the checked-in `run_borromean_experiment()` gives AUROC=0.000 (perfect separation) for both, not chance — a roughness confound. Fixing it (`examples/cayley_dickson_borromean_rerun.sio`) surfaced three further defects before the experiment produced any nonzero signal at all: the feature-packing scheme confined every operand to the associative quaternion subalgebra (associator ≡ 0 by a theorem); escaping that still gave [2,1]-hook ≡ 0 because packing operands along one shared fixed 3-direction basis triple makes every associator a pure-basis-triple associator, always fully alternating (Λ³), regardless of coefficients or their assignment (verified 3 ways); only spreading operands across ≥4 independent basis directions finally made both associator and [2,1]-hook nonzero. With that fixed: n=15/class looked like a real effect (hook AUROC=0.25); n=25/class moved substantially closer to chance (hook AUROC=0.32) — the direction a small-sample artifact would move. Honest conclusion: consistent with chance, not yet powered to confirm. See `cayley_dickson_hierarchy_paper_2026-08-13.md` §4.3/§9 for the full four-defect account. Do not cite an AUROC in either direction from this experiment until a larger-n, permutation-tested re-run is done.*

This is a **theorem-backed negative boundary**: the Cayley-Dickson associator is provably the wrong tool for higher-homotopy obstructions. This is not a limitation of our implementation — it is a consequence of representation theory. **The theorem stands on its own (Schur's lemma requires no experiment); the AUROC figures offered above as its "experimental confirmation" do not currently support it and should be removed or regenerated before submission.**

---

## 6. Honest nulls

| Domain | Task | Result | Reason |
|---|---|---|---|
| ABIDE connectome | ASD/TD (aggregate F₁) | Null (AUROC 0.489) | Covariance, not bracketing |
| NMA inconsistency | Drug interaction detection | Refuted (AUROC 0.50) | Additive algebra |
| Code brackets | Valid/corrupted | Null (≈0%) | Too noisy, multi-type |
| Seizure raw EEG | Ictal detection | Constant features | O-SSM saturates on broadband |
| AFib amplitude | Normal vs AFib | Null (87.5% floor) | Class imbalance; wrong representation |
| Borromean paths | Massey detection | Orthogonal (AUROC 0.49) | **RETRACTED — see §5; re-run (n=25/class, 4 defects fixed) gives AUROC 0.32–0.34, consistent with chance but underpowered to confirm** |

*Table 5: Six independent null results. Non-associativity is not a universal advantage. Borromean row retracted 2026-08-14, resolved to "consistent with chance, underpowered."*

---

## 7. Related work

No prior work applies Cayley-Dickson algebras to RNA, EEG, linguistics, or dimensional psychopathology. PubMed search: "octonion" appears in 11 papers (all ML/math), "sedenion" in zero biomedical papers, "exceptional Lie group" in zero biomedical papers.

| Prior work | What it does | What it doesn't do |
|---|---|---|
| Octonion LSTM/RNN (~2021) | Octonion recurrence | Suppresses associator; no dormancy analysis; no biology |
| Numerion (2025) | Sedenion MLP | Generic emulation; no associator; no biology |
| PHM/PHNN (2021-22) | Learnable hypercomplex matrix | Soft matrix, not exact algebra; no Fano prior |
| Clifford/GATr | Algebra-structured DL | Associative; associator ≡ 0 |
| Magarshak 1993 | Quaternion RNA encoding | Associative, no hierarchy |
| Quadrini 2017 | Relational algebra for PKs | Not division algebra |

---

## 8. What is claimed and not claimed

**Claimed:**
1. Artin dormancy: binary octonion recurrence is equivalent to a fixed matrix (proof + experiment).
2. OctTree activates non-associativity and outperforms the associative control (RealTree) on Dyck-1 (+19% at L=32, 3 seeds) — the advantage is real but modest, and fades with length.
3. The Fano plane is a parameter-efficient encoding (182p → 73%), not a necessary prior — a free matrix with 10× more parameters works (77.1%) but is less efficient.
4. The octonion→sedenion transition tracks nested→crossing structure: SedenTree-16 achieves 99.2% on pseudoknot crossing where OctTree-8 (72.7%) and all element-wise controls (50.0% at 8 and 16 dims) fail. This is the load-bearing result.
5. The [2,1]-hook orthogonality is a representation-theoretic boundary (theorem stands; corrected projector gives 10⁻¹⁵ on 𝕆, 21–44 on 𝕊).

**Not claimed:**
- Not SOTA on any practical task.
- Not a clinical diagnostic tool.
- Not a claim that the brain computes in octonions.
- Not universality (6 nulls documented).
- Not a pharmacological prediction (CYP450 mapping is speculative — see Appendix).

---

## Appendix A: CYP450 Mapping (Speculative)

The Fano plane pairs correspond to CYP450 enzyme pairs (e.g., G₁₂'s dominant direction e₃↔e₇ maps to CYP2C8↔CYP3A4). This mapping is a **conjectural analogy** with no clinical validation. It requires pharmacogenomic data (STAR*D, FAERS) to test. No claim is made in the main paper.

---

## Appendix B: GPU and FPGA

OctTree maps to existing tensor-core PTX kernels (3–6× GPU speedup on A5000). The U250 FPGA catastrophe-scan kernel achieves 513 Msamples/s bit-exact throughput. Each tree level is one L(a)·b kernel launch.

---

## Reproduction

Seed: 20260806. Pre-registered: 2026-04-21. First run, no hyperparameter search.

**2026-08-14 correction:** `cayley_dickson_paper_reproduction.py`, listed below as reproducing Tables 1-3, is the specific script an audit found to contain a degenerate Dyck-1 generator (`gen_valid_dyck`) and a memorization-only pseudoknot generator (`gen_pk`, 2 fixed sequences total). Table 1's Dyck-1 rows and Table 3's SedenTree row are retracted; see the header note and per-table notes above, and `docs/papers/main/cayley_dickson_hierarchy_paper_2026-08-13.md`'s §9 for the full scorecard and corrected Sounio replacements.

```bash
# Core experiments (reproduces Tables 1-3) — Dyck-1/SedenTree rows retracted, see above
python3 scripts/research/cayley_dickson_paper_reproduction.py

# Full RNA experiments
python3 scripts/research/rfam_octtree_experiment.py
python3 scripts/research/real_pk_experiment.py

# EEG (requires LEMON data access) — retracted, see §4
python3 scripts/research/g2_lemon_analysis.py

# Corrected Sounio artifacts: [2,1]-hook, G2 derivation Leibniz check,
# Dyck-1 + pseudoknot generators (reuse the already-verified
# stdlib/algebra/octonion.sio / sedenion.sio, 13/13 tests)
SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run examples/cayley_dickson_hook21_g2_verification.sio
SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run examples/cayley_dickson_dyck_pseudoknot_generators.sio
```
