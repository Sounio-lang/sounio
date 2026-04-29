# Non-Associative Epistemic Connectomics — Phase 1 Preregistered Protocol

**Version**: 1 (2026-04-12)
**Scope**: Phase 1 only — de-risking on synthetic ground truth + n=10 pilot on ABIDE-I.
**Frozen before data touched**: YES (this file committed before any ABIDE subject is loaded).

## Motivation

Functional connectivity in ASD has been studied extensively under associative algebras (ℝ, ℂ, quaternion weights). The non-associative structure of the octonions provides a natural algebraic object — the **associator** `[a,b,c] = (ab)c − a(bc)` — that is identically zero in any associative labeling and becomes non-trivial only when edge labels draw from the full 𝕆 structure rather than an associative subalgebra. If connectomic graph structure in ASD carries information that is genuinely "octonion-native" — i.e. not reducible to a ℝ/ℂ/ℍ labeling — the subject-level statistic on the associator field should separate ASD from TD.

A parallel secondary hypothesis treats the 16D sedenion algebra 𝕊, whose zero-divisor set is a nontrivial algebraic variety: subjects whose feature point lies closer to the zero-divisor support should exhibit a specific form of algebraic singularity. Experiment C reuses the precomputed 168-class geometry from `artifacts/research/sedenion_zero_divisor_geometry.v1.json`.

## Hypotheses

- **H1 (primary, Experiment B)**: ASD subjects exhibit a higher subject-level associator-field statistic (95th percentile of per-triple `‖[a,b,c]‖²`) than TD subjects, computed on 𝕆-labeled derivations of resting-state functional connectivity.
- **H2 (secondary, Experiment C)**: ASD subjects exhibit a smaller minimum distance from their sedenion feature point to the union of the 168 zero-divisor supports than TD subjects.
- **H0 for each**: No group-level difference; observed effect-size is indistinguishable from a within-subject label-permutation null.

## Design

### Octonion labeling (Experiment B)
For each subject, build an 8-component octonion-valued label per ROI node `i`:
```
L_i = (1.0, v1[i], v2[i], v3[i], v4[i], v5[i], v6[i], v7[i])
```
where `v1..v7` are the first seven non-trivial Laplacian eigenvectors of the ROI correlation graph (CC200 atlas), as already computed by `scripts/research/abide_preprocess.py`. This is the minimal octonion embedding that carries all seven Fano-triple generators and preserves the existing G₂ pipeline's eigenbasis.

### Subject-level statistic (Experiment B)
For each triple `(i,j,k)` of distinct nodes from a fixed 30-node working subset (deterministic across subjects — the first 30 ROIs by CC200 ordering):
```
A_{ijk} = ‖ (L_i · L_j) · L_k  −  L_i · (L_j · L_k) ‖²
```
where `·` is octonion multiplication under the Fano-plane convention from `tests/run-pass/knowledge_octonion_structure.sio`. This yields n_triples = C(30, 3) = 4,060 per subject.

**Reductions** (both reported, primary marked):
- `mean(A)` — central tendency
- `p95(A)` — 95th percentile (**primary**; heavier-tail sensitive; non-associativity is expected to concentrate in extremes, not the mean)

### Null model (Experiment B)
**Within-subject octonion label permutation**: for each subject, randomly permute the 8 octonion channels (σ ∈ S_8, excluding identity) applied uniformly across nodes, preserving the Fano structure shell but shuffling which eigenvector sits in which octonion slot. 1,000 permutations per subject. Produces a per-subject null distribution of `p95(A)` values against which the observed value is z-scored.

Rationale: this null preserves (a) graph topology, (b) the marginal distribution of each eigenvector, (c) the number of triples, and destroys only the specific octonion-slot alignment. Whitening against this null isolates alignment-specific signal from generic algebraic activity.

### Test statistic (Experiment B)
- **Effect size**: Cohen's d on subject-level z-scored `p95(A)`, ASD vs TD.
- **Confidence interval**: 10,000-bootstrap 95% CI.
- **Distribution test**: two-sample Kolmogorov–Smirnov on z-scored `p95(A)` distributions.

### Experiment C (sedenion zero-divisor proximity)
Per-subject 16D feature vector: `F = (mean(v1), std(v1), mean(v2), std(v2), ..., mean(v8), std(v8))` where `v8` is the 8th Laplacian eigenvector. Distance `d_ZD` = minimum Euclidean distance from `F` to any primitive projective zero-divisor pair support listed in `artifacts/research/sedenion_zero_divisor_geometry.v1.json`. Test: Cohen's d on `d_ZD`, ASD vs TD, with 10k-bootstrap 95% CI. (Phase 1: stub only — no subjects processed.)

### Multiple-comparison correction
- Family: {H1, H2}, size 2. Holm-Bonferroni at family α = 0.05.
- H1 is primary and receives the first rejection slot if its p-value is smallest.

### Inclusion criteria
- ABIDE-I subject with ≥ 100 usable TRs after scrubbing (matches `scripts/research/abide_preprocess.py` cutoff).
- Subject-level frame present in `artifacts/research/abide/frames.bin` and passes `extract_frame` without exception.
- Subject pair-matching by site and age is **not required for Phase 1 pilot** (n=10). Full-cohort Phase 2 will add site/age regression covariates.

### Pilot parameters
- **n = 10**, balanced: 5 ASD (lowest `DX_GROUP == 1` `FILE_ID` sort) + 5 TD (lowest `DX_GROUP == 2` `FILE_ID` sort). Deterministic selection; no randomness at the subject-sampling step.
- Working subset: first 30 ROIs by CC200 ordering.
- 4,060 triples per subject, 1,000 null permutations per subject.

### Stopping / power
Pilot produces a point estimate of Cohen's d. Phase 2 full-cohort target (n ≈ 1,034) has power > 0.99 at α = 0.05 for |d| ≥ 0.2, so Phase 2 proceeds **only if pilot |d| > 0.15 and 95% CI excludes zero**. Otherwise the design is revised before Phase 2 is committed.

## Changes after freeze

Any post-freeze change to the octonion labeling, reduction (mean/p95), null model, or statistic must be logged in this file with a timestamped section and justified. The pre-freeze design is definitive.
