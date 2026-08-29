# PPCR preregistration — does affect-network curvature predict depression BEYOND density? (multi-cohort)

**Frozen:** 2026-05-27, BEFORE computing any curvature↔depression association on the openESM data.
**Author:** Demetrios (Sounio). **Status:** confirmatory, density-controlled, multi-cohort replication.

## Background & motivation
On the single Kossakowski MDD case, the affect-network's discrete curvature (Forman-Ricci, exact-OT
Ollivier-Ricci) co-varied with classical CSD (|r|≈0.5) but the relationship **reduced to network
density** (mean|r| tracked CSD at +0.70; partialing density out collapsed/destabilized both curvatures).
n=1 cannot separate curvature from density because density does not vary across a single subject. This
preregistered study uses **between-subject** variation in three openESM cohorts to test whether
per-subject affect-network curvature predicts depression **after controlling for network density**.

## Hypotheses
- **H1 (primary):** Across subjects, per-subject mean **Forman-Ricci** curvature of the affect network
  is associated with depression severity *after* controlling for network density (mean|r|).
  Direction: **negative** (lower curvature ↔ higher depression — the fragility prediction).
- **H2:** Same for exact-OT **Ollivier-Ricci** (direction left two-sided; Forman & ORC were
  anti-correlated at −0.89 in n=1, so ORC's predicted sign is positive if H1 holds).
- **H0 / null-of-interest:** the partial association is ~0 — curvature adds nothing beyond density
  (the n=1 outcome generalizes; depression is a density phenomenon, not a curvature one).

## Datasets (3 cohorts, fixed roles)
| role | dataset | N | network items (curvature computed on) | depression outcome (mean over beeps) |
|---|---|---|---|---|
| DISCOVERY | 0033_fisher | 40 | all affect items EXCEPT the 3 outcome items (~19 nodes) | mean{down, hopeless, anhedonia} |
| REPLICATION-1 (high N) | 0039_kuczynski | 515 | {loneliness, left_out, social_interaction, vulnerability, perceived_responsiveness, covid_anxiety} | mean{depressed, anhedonia} (PHQ-2) |
| REPLICATION-2 | 0010_geschwind | 130 | {cheerful, relaxed, worried, fearful} | mean{sad} |

**No-circularity rule:** the depression-outcome items are NEVER included in the network on which
curvature is computed.

## Fixed analysis pipeline (identical across cohorts)
1. **Inclusion:** subjects with ≥30 completed beeps on the network items (Fisher/Geschwind all qualify;
   Kuczynski filters low-compliance subjects). Report n excluded.
2. **Per-subject network:** Pearson correlation among network items over that subject's beeps
   (mean-imputed for sparse missingness); edge if **|r| ≥ 0.25**, weight = |r| (same threshold the n=1
   work used; robustness already shown stable over [0.15,0.40]).
3. **Per-subject metrics (native Sounio, GUM-typed):**
   - **Forman-Ricci** (weighted, node weight 1), mean over edges.
   - **Ollivier-Ricci** (exact-OT: weighted random-walk measure, hop-distance, log-Sinkhorn ε=0.05),
     mean over edges.
   - **density** = mean|r| over all item pairs (threshold-free); also n_edges.
   - Each curvature carried as `Epistemic{val, variance, confidence}` (variance from per-edge spread;
     confidence 0–1000); subjects with degenerate/empty graphs flagged.
4. **Outcome:** per-subject depression = mean of the outcome items over beeps.
5. **Primary test:** first-order **partial Spearman** ρ(Forman, depression | density), per cohort.
   Report raw ρ, partial ρ, and the premise corr(density, depression).
6. **Confirmatory decision rule:** H1 supported if, in the DISCOVERY cohort, partial ρ(Forman,
   depression | density) is negative AND |partial ρ| ≥ 0.20, AND the SIGN replicates in ≥1 replication
   cohort. Density-reduction concluded if partial ρ collapses (|·|<0.10) or sign-flips across cohorts.
7. **Epistemic gate (Sounio differentiator):** report the cross-subject association with GUM-propagated
   confidence; only claim an effect when confidence ≥ 800/1000 AND the partial survives.

## Researcher degrees of freedom locked here
Embedding/graph: Pearson, |r|≥0.25, weight=|r|. Curvature: Forman (primary) + exact-OT ORC (secondary).
Density control: mean|r| (primary), n_edges (sensitivity). Outcome: mean of named depression items.
Test: partial Spearman. Direction: H1 negative. No item reselection after seeing outcomes.

## What would change the conclusion
- Partial survives (|ρ|≥0.20, sign-consistent, ≥1 replication): curvature is a real density-independent
  correlate of depression — the n=1 density-reduction does NOT generalize.
- Partial collapses/sign-unstable: the n=1 finding generalizes — affect-network *density* is the
  EWS-relevant quantity; curvature is a reparametrization. (A clean, publishable null either way.)

Data: openESM Zenodo harmonized `_ts.tsv` (CC-BY); records 17347474/17348039/17347658.
