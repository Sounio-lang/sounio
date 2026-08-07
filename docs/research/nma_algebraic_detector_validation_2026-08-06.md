<!-- docs:meta
topic_id: repo.docs.research.nma-algebraic-detector-validation-2026-08-06
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.nma-algebraic-detector-validation-2026-08-06
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NMA Algebraic Inconsistency Detector — Validation Result

**Date:** 2026-08-06
**Status:** `EXECUTABLE` — Route C (octonion) REFUTED. Route A (multivariate) has a narrow aggregation advantage only. Design-masking is a genuine loop-level detection impossibility.
**Seed:** `docs/internal/garden/seeds/2026-05-09-novelty-weather-map.md` Constellation 3
**Algebra audit:** `docs/proposals/nma_nonassociative_algebra_note.md`
**Simulation:** `scripts/research/nma_algebraic_detector_validation.py`

---

## What was tested

The garden seed (Constellation 3) proposed that octonion associators could detect design-sensitive inconsistency in network meta-analysis (NMA) that standard tests miss. The hostile reviewer said: *"You renamed loop inconsistency with octonions."*

The algebra audit (`nma_nonassociative_algebra_note.md`) already concluded that NMA consistency is natively an abelian group cohomology object (H¹ defect on the treatment graph) — associative, with no natural associator. It identified three routes to non-associativity and called for a validation simulation as gatekeeper.

This is that simulation.

## Experiment design

**Four detectors** tested on **five scenarios** (2000 synthetic 3-treatment networks each):

| Detector | Description | Route |
|---|---|---|
| D0 — Bucher z-score | Scalar additive cocycle defect, design-agnostic | Baseline |
| D1 — Hotelling T² | Multivariate vector cocycle, two correlated outcomes | Route A |
| D2 — Design-weighted | Bucher z-score × design-spread factor | Heuristic |
| D3 — Octonion associator | Embed effect + design into 𝕆, measure ‖[a,b,c]‖ | Route C |

| Scenario | What it models | Ground truth |
|---|---|---|
| S0 | Fully consistent | No inconsistency |
| S1 | Simple scalar inconsistency (one edge perturbed) | Always inconsistent |
| S2 | Design-confounded (design features bias effects) | Inconsistent when designs differ |
| S3 | Design-masked (biases cancel on the scalar loop) | Edges biased, loop closes |
| S4 | Multivariate (opposite-sign inconsistency across outcomes) | Always inconsistent |

**Falsification criterion:** if D3 never beats D0, Route C is refuted. If D1 beats D0 on S4, Route A is supported.

## Results

### AUROC table

| Task | D0 Bucher | D1 Hotelling | D2 Design | D3 Octonion |
|---|---|---|---|---|
| S0 vs S1 (simple inconsistency) | **0.706** | 0.763 | 0.703 | 0.501 |
| S0 vs S2 (design-confounded) | 0.592 | **0.607** | 0.588 | 0.497 |
| S0 vs S3 (design-masked) | 0.498 | 0.495 | 0.499 | **0.501** |
| S0 vs S4 (multivariate) | **0.787** | 0.606 | 0.785 | 0.508 |
| S0 vs ALL | **0.656** | 0.623 | 0.654 | 0.504 |

### Score distributions (median [IQR])

| Detector | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| D0 Bucher | 0.68 [0.33–1.16] | 1.32 [0.69–2.03] | 0.90 [0.45–1.53] | 0.66 [0.31–1.18] | 2.02 [0.93–3.31] |
| D3 Octonion | 0.54 [0.40–0.72] | 0.55 [0.40–0.71] | 0.54 [0.40–0.72] | 0.54 [0.40–0.71] | 0.56 [0.40–0.73] |

### Diagnostic correlations (S1 networks, n=500)

- D3 vs D0: **rho = −0.03** (no correlation with the standard test)
- D3 vs total effect magnitude: **rho = −0.05** (does not respond to effect estimates at all)
- D3 vs design-feature spread: **rho = 0.25** (weakly driven by random design cross-products)
- D3 on S0 vs S1: Mann-Whitney **p = 0.29** (cannot distinguish consistent from inconsistent)

### Sensitivity sweep

The octonion design scaling σ was swept over {0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0}. The AUROC of D3 on S0-vs-S2 and S0-vs-S3 was constant (0.486, 0.514) across all values. The associator's blindness is not a tuning problem.

## Findings

### 1. Route C (octonion embedding) is REFUTED

The octonion associator is completely inert to NMA inconsistency. Its score distribution is identical across all five scenarios (median ≈ 0.54, IQR [0.40–0.72]). It has zero correlation with the Bucher z-score (rho = −0.03) and cannot distinguish consistent from inconsistent networks (p = 0.29).

**Why it fails.** The standard NMA consistency condition is additive: Δ = c_AB + c_BC + c_CA = 0. The associator measures non-associativity of octonion multiplication: [a,b,c] = (a·b)·c − a·(b·c). These are fundamentally different operations. The associator's response is dominated by the imaginary cross-products of the design features through the Fano plane structure, not by the real-part effect. There is no principled mapping from additive evidence composition to octonion multiplication that preserves the inconsistency signal.

**This confirms the algebra audit's verdict** (`nma_nonassociative_algebra_note.md` §3): 𝕆 is not the natural algebra for NMA consistency. The imposed structure does not become useful as a detector.

### 2. Route A (multivariate) has a narrow aggregation advantage only

D1 (Hotelling T²) beats D0 on S1 (0.763 vs 0.706) — the multivariate detector accumulates evidence across two outcome dimensions, giving a power gain. This is the well-known Hotelling vs. separate-t-test advantage and has nothing to do with non-associativity.

However, D1 LOSES to D0 on S4 (0.606 vs 0.787). When inconsistency has opposite signs across outcomes (efficacy shift +0.2, safety shift −0.2), the multivariate aggregation partially cancels the signal. The multivariate detector only helps when inconsistencies are correlated across outcomes — and it hurts when they are anti-correlated.

### 3. Design-masking is a genuine loop-level detection impossibility

S3 (design-masked inconsistency) is undetectable by ALL methods (AUROC ≈ 0.50 for everyone). When design biases cancel on the loop (bias_AB + bias_BC − bias_AC = 0), the scalar cocycle defect is zero by construction, and no loop-level statistic can recover the hidden inconsistency.

This is not a failure of the detectors — it is a mathematical impossibility. The information has been genuinely destroyed at the loop level. Detecting design-masking requires information BEYOND the loop: either network-level patterns (multiple loops sharing biased edges), external design metadata, or model-based priors on design effects.

### 4. Design-weighted detection (D2) adds nothing

Scaling the Bucher z-score by a design-spread factor (D2) does not change the AUROC meaningfully. The design spread is roughly constant across scenarios, so the rescaling does not help discriminate.

## What this means for the research programme

1. **The NMA associator spin-out should be dropped for the octonion framing.** The algebra audit was right; the simulation confirms it. No sunk-cost loyalty.

2. **The multivariate H¹ framing has modest value.** Hotelling T² on the vector cocycle gives a power gain when outcome inconsistencies are correlated. This is a known multivariate testing result, not a novel algebraic contribution. It does not justify a standalone paper.

3. **The design-masking impossibility is the interesting result.** It identifies a structural limit of loop-level NMA inconsistency detection and points to where additional information (network structure, external metadata, design-effect modelling) is necessary. This is a genuine insight, but it is a negative result about detection limits, not a new detector.

4. **The cohomological framing (H¹ of the treatment graph) is the correct algebra.** The Bucher z-score IS the H¹ cocycle defect test. There is no richer algebraic structure to exploit at the loop level.

## Status of the garden seed

**Constellation 3 is closed as a butterfly.** The falsification path ran; the octonion associator lost. The seed's own instruction was followed: *"Build a Python or Sounio synthetic NMA generator with known ground truth, standard inconsistency metrics, and an associator score. The Garden should forbid paper claims until this benchmark exists."*

The benchmark now exists. The associator score does not detect what the standard test misses. No paper claim is licensed.

The multivariate NMA thread (Route A) survives as a modest methodological observation but does not carry the algebraic novelty weight the seed hoped for. The connectomics/G₂/𝕆 program, correctly decoupled in the algebra note, continues independently.

## Open questions that survive

- **Q-alg-1.** Can design-masking be detected at the NETWORK level (not loop level) by examining patterns across multiple loops? This is a graph-theoretic question, not an algebraic one.
- **Q-alg-2.** Does crossover/sequential NMA data exist in sufficient density to power a Route-B (path-dependent) study? Route B was not tested here because it requires 4+ treatment loops with sequential treatment ordering, which is a different experimental design.
- **Q-alg-3.** Is the cohomological framing itself novel in the NMA literature, or has H¹ of the treatment graph been used before? This is a literature question, not a simulation question.

## Reproduction

```bash
python3 scripts/research/nma_algebraic_detector_validation.py
```

Runtime: ~30 seconds. Results: `scripts/research/nma_detector_results.json`.
