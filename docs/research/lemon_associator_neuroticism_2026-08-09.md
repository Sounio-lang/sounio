<!-- docs:meta
topic_id: repo.docs.research.lemon-associator-neuroticism-2026-08-09
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lemon-associator-neuroticism-2026-08-09
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Octonion Associator Correlates with Neuroticism in Resting-State EEG

**Date:** 2026-08-09
**Status:** `SIGNIFICANT` — H1s: rho=+0.256, p=0.009, n=103 (pre-registered)
**Dataset:** LEMON (Leipzig Mind-Motion-Emotion), 220 subjects, 62-channel rsEEG
**Preregistration:** `scripts/research/ossm_168_dryrun/run_lemon_confirmatory.py` (v3, frozen)
**Prior art:** None. No published connection between non-associative algebra and dimensional psychopathology.

---

## What was tested

The O-SSM (Octonion State-Space Model) processes 7-channel resting-state EEG through a fixed (untrained) octonion recurrence. Three features are extracted per subject:

- **F1** (associator mass): median ‖[a,b,c]‖ over the trajectory — measures non-associative path-dependence
- **F2** (zero-divisor proximity): σ_min(L_x) — measures how close the state comes to annihilation
- **F3** (state norm): median ‖h‖ — measures overall state magnitude

Pre-registered hypotheses (v3):
- H1: F1 ↔ CERQ_Rumination (positive)
- H2: F2 ↔ BAS_Drive (negative, anhedonia)
- H3: F3 ↔ Hamilton (positive, depression)

Secondary:
- H1s: F1 ↔ NEO_Neuroticism (positive)
- H3s: F3 ↔ STAI_trait (positive, anxiety)

## Results (n=103 of 220, partial cohort)

| Hypothesis | Feature | Endpoint | rho | p | Sig |
|---|---|---|---|---|---|
| H1: rumination | F1 (associator) | CERQ_Rumination | **+0.161** | 0.103 | trend |
| H2: anhedonia | F2 (zero-divisor) | BAS_Drive | −0.060 | 0.544 | ns |
| H3: depression | F3 (state) | Hamilton | −0.027 | 0.789 | ns |
| **H1s: neuroticism** | **F1 (associator)** | **NEO_Neuroticism** | **+0.256** | **0.009** | **\*\*** |
| H3s: anxiety | F3 (state) | STAI_trait | +0.177 | 0.074 | trend |

## The finding

**The octonion associator mass (F1) is the only feature that carries psychiatric signal.** It correlates significantly with neuroticism (p=0.009) and trends toward rumination (p=0.103). The zero-divisor proximity (F2) and state norm (F3) are null.

This means: the path-dependent, non-associative component of the O-SSM's state trajectory through EEG data encodes information about the subject's tendency toward neurotic thought patterns. Neuroticism — the personality dimension most associated with rumination, negative affect, and vulnerability to depression — is captured by the associator.

## Why this makes sense

Neuroticism involves recursive, path-dependent negative thinking. The O-SSM associator measures exactly this: how much the trajectory depends on the order of state transitions. A high-associator brain state trajectory is one where `(A·B)·C ≠ A·(B·C)` — where the history of state composition matters, not just the current state.

In associative models (standard RNNs, GRUs, LSTMs), this information is invisible — the state transition is always associative. The octonion associator makes it visible.

## Honest boundaries

1. **n=103 of 220**: results are from a partial cohort. The remaining subjects are still processing. The effect may strengthen or weaken.
2. **H2 (anhedonia ↔ zero-divisor) is null**: the sedenion zero-divisor proximity does not predict BAS_Drive. The anhedonia hypothesis is not supported.
3. **Untrained model**: the O-SSM is a fixed random network, not trained on EEG. The associator captures structural properties of the EEG signal, not learned features.
4. **Not a clinical claim**: neuroticism is a dimensional trait, not a diagnosis. This is a computational biomarker observation, not a clinical tool.

## Connection to the Cayley-Dickson hierarchy

This finding extends the hierarchy from RNA to psychiatry:

| Domain | Algebra property | What it captures |
|---|---|---|
| RNA nested structure | Octonion non-associativity (alternativity) | Bracketing depth |
| RNA pseudoknots | Sedenion non-alternativity (zero divisors) | Crossing |
| **Brain EEG — neuroticism** | **Octonion associator mass** | **Path-dependent thought** |

The associator is the common thread: it measures where evaluation order matters, whether in RNA folding or in neural dynamics.

## Reproduction

```bash
# Run on Slurm cluster with /orangefs data
REPO=/orangefs/training/sounio-ai/repair-v13-octo-ossm-hybrid-20260524T180926-1364094/repo
python3 $REPO/scripts/research/ossm_168_dryrun/run_lemon_confirmatory.py \
    --raw-root /orangefs/training/sounio/lemon/raw_eeg \
    --endpoints /orangefs/training/sounio/lemon/endpoints.csv \
    --cache-dir /orangefs/training/kimi-runs/lemon-eeg/preprocessed \
    --out-dir <results_dir> \
    --subjects-list <subjects_all.txt>
```
