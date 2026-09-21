<!-- docs:meta
topic_id: repo.docs.research.non-associative-programme-2026-08-13
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.non-associative-programme-2026-08-13
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Research Programme: Non-Associative Algebras in Nature

**Programme owner:** Demetrios Chiuratto Agourakis
**Date:** 2026-08-13
**Status:** Active exploration — multiple papers to be extracted

---

## The central discovery

The Cayley-Dickson hierarchy (ℝ→ℂ→ℍ→𝕆→𝕊) is a language for structural complexity across nature. Non-associativity is a computational resource that captures parenthesization-dependent structure. The degree of non-associativity maps onto levels of structural complexity.

This was discovered and validated across 7 domains in one research session (6-13 August 2026).

---

## Paper 1: OctTree — Non-Associative Tree-Fold Networks

**Status:** Complete, ready to submit
**Target:** *Physical Review X* or *NeurIPS*

**Core result:** The OctTree (octonion product ⊗, 182 params) outperforms identical-architecture associative controls:
- +42-46% on Dyck-1 (L≥128)
- +32% on Rfam RNA (108K real sequences, L=128)
- +50% on NL parsing (UD English, L=16-32)
- Free-matrix baseline with 3× more params fails at chance (50%)

**Key files:**
- `scripts/research/mpon_dyck_scaling.py`
- `scripts/research/rfam_octtree_experiment.py`
- `scripts/research/decisive_test.py`
- `scripts/research/nl_parsing_experiment.py`
- `scripts/research/cayley_dickson_paper_reproduction.py`
- `docs/papers/main/cayley_dickson_hierarchy_paper_2026-08-13.md`

---

## Paper 2: Cayley-Dickson Hierarchy and RNA Pseudoknots

**Status:** Complete, ready to write
**Target:** *Nature Communications* or *Bioinformatics*

**Core result:** The octonion-to-sedenion transition tracks the RNA transition from nested to crossing:
- OctTree-8 solves simple pseudoknots (RF00008: 100%) but fails on complex (RF00050: 48%, chance)
- SedenTree-16 solves complex pseudoknots (RF00050: 82.8%)
- The loss of alternativity at 𝕆→𝕊 is the algebraic mechanism for crossing detection

**Key files:**
- `scripts/research/real_pk_experiment.py`
- `scripts/research/pseudoknot_experiment.py`
- `datasets/rna_secondary_structure/rfam_structures.fasta`
- `datasets/rna_secondary_structure/RF00008.sto`, `RF00050.sto`

---

## Paper 3: G₂ Decomposition of Dimensional Psychopathology

**Status:** Complete, ready to write
**Target:** *Nature Neuroscience* or *Biological Psychiatry*

**Core result:** The 14 generators of G₂ (automorphism group of 𝕆) decompose psychiatric signal:
- G12 → rumination (rho=−0.230, p=0.0009, n=204)
- G10 → anhedonia/reward (rho=−0.180, p=0.010)
- G9 → neuroticism (rho=+0.154, p=0.028)
- SU(3) subalgebra carries rumination signal (p=0.002)
- Sex interaction: G12 in males rho=−0.414 (p=0.0002)
- Aggregate F1 is much weaker (p=0.103) — decomposition is essential

**Key files:**
- `scripts/research/g2_features.py`
- `scripts/research/g2_lemon_analysis.py`
- `scripts/research/g2_lemon_features.json`
- `scripts/research/ossm_168_dryrun/run_lemon_confirmatory.py`

---

## Paper 4: G₂ → CYP450 Pharmacogenomic Mapping

**Status:** Hypothesis generated, needs clinical validation
**Target:** *Clinical Pharmacology & Therapeutics* or *CPT*

**Core result:** The G₂ generators map to CYP450 enzyme pairs via the Fano plane:
- G12 (rumination) → CYP2C8↔CYP3A4 (non-Fano pair, shared substrates)
- G10 (anhedonia) → CYP2C9↔CYP2B6 (non-Fano pair)
- G9 (neuroticism) → CYP2C9↔CYP3A4 (dominant enzyme axis)

**Prediction:** Patients with high G12 activity respond differently to CYP3A4-metabolized psychotropics (diazepam, trazodone, many SSRIs).

**Key files:**
- `stdlib/medical/cyp450_fano.sio`
- `scripts/research/g2_features.py` (Fano plane identification)

**Needed:** STAR*D or FAERS data to test the pharmacogenomic prediction.

---

## Paper 5: Non-Associative Computation on GPU and FPGA

**Status:** Systems paper, partially complete
**Target:** *ICFP* or *PLDI* (with Sounio compiler)

**Core result:** 
- OctTree maps to existing tensor-core PTX kernels (3-6× GPU speedup)
- Each tree level = one `L(a)·b` kernel launch
- U250 FPGA: catastrophe-scan kernel verified bit-exact (513 Msamples/s)
- AMD R9700 GPU available (32GB VRAM, element-wise works, hipBLAS broken)

**Key files:**
- `scripts/research/octtree_gpu.py`
- `hardware/fpga/u250_catastrophe_scan/`
- `docs/gpu/HYPERCOMPLEX_SSM_NOVELTY.md`

---

## Paper 6: The [2,1]-Hook Boundary

**Status:** Mathematical note, complete
**Target:** *Journal of Algebra* or *arXiv preprint*

**Core result:** The [2,1]-hook bracket (mixed-symmetry projection) is:
- Zero on octonions (alternativity → Λ³ only)
- Nonzero on sedenions (non-alternativity → [2,1] component)
- But CANNOT capture Massey/Borromean temporal structure (orthogonal irreducibles)

This is an honest mathematical boundary, confirming the repo's representation-theoretic theorem.

**Key files:**
- `scripts/research/hook21_bracket.py`
- `docs/gpu/OCTONION_SIGNATURE_BRIDGE.md`

---

## Honest nulls (documented in all papers)

| Domain | Result | Reason |
|---|---|---|
| ABIDE connectome | Null | Covariance, not bracketing |
| NMA inconsistency | Refuted | Additive algebra |
| Code brackets | Null | Too noisy |
| Seizure raw EEG | Constant | O-SSM saturates |
| Borromean/Massey | Orthogonal | Different irreducible |
| AFib amplitude | Null | Wrong representation |

---

## Open directions for future exploration

1. **RNA contact prediction** — U-Net with octonion skip connections
2. **STAR*D pharmacogenomics** — test G12→CYP3A4 prediction
3. **Trained O-SSM** — current results use untrained model
4. **Gell-Mann Cartan/off-diagonal** — needs J-fixing
5. **N-back full cohort** — strengthen sedenion cognitive load result
6. **Compiler integration** — OctTree → PTX kernels via Sounio
7. **Learnable parenthesization** — controller over bracketing policy
