<!-- docs:meta
topic_id: repo.docs.research.paper-a-rq4-mc-adequacy-2026-08-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-rq4-mc-adequacy-2026-08-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# RQ4 (a) — probabilistic adequacy of the first-order truth: Monte Carlo on the same cohort (2026-08-31)

**Closes** the one item every external reviewer of the NS mechanization left open (Grok 4.5/4.6
×2, Kimi K3: "probabilistic adequacy — `trueVar` is algebraic; no sampling semantics"). The
affine forms are *first-order* truth. Are they the truth that matters for the clinical decision?

**Artifact:** `docs/research/sounio/rq4_vanco_mc_adequacy.sio` — Sounio, no `knowledge.sio`
import (the naive `ep_*` rules are inlined verbatim), runs under **Madaros and lean_single with a
byte-identical `RQ4_MC` line**; a second Madaros run is byte-identical (deterministic LCG).

```
RQ4_MC n=5000 k=1000 warn_t=909 warn_n=2803 warn_mcsd=911 warn_mcq=877
       t_mcq_both=867 t_only=42 mcq_only=10  n_mcq_both=877 n_only=1926 mcq_only_n=0
       var_mc_over_t_permille=999 var_n_over_mc_permille=300917
```

## Design

Same 5 000 nominal patients as the RQ4 cohort (LCG seed 20260831). Per patient, K = 1 000 draws of
(weight, SCr, Q, Vp) ~ Normal(nominal, u) with the RQ4 uncertainties (u(weight) = 1 kg, u(SCr) =
10 %, u(Q) = u(Vp) = 20 %; Irwin–Hall 12-sum normals from an independent LCG stream), the
two-compartment AUC₀₋₂₄ = A/α + B/β evaluated in plain `f64` per draw; then `mean_MC`, `Var_MC` and
`P_MC[AUC < 400]`. The point estimate is the nominal chain value in every rule.

| rule | WARN iff point ≥ 400 and … | WARNs |
|---|---|---|
| **T** affine (first order) | point − 2·sd_T < 400 | 909 |
| **N** naive `ep_*` chain | point − 2·sd_N < 400 | 2 803 |
| **MC-sd** sampling variance, same rule | point − 2·sd_MC < 400 | 911 |
| **MC-q** sampling quantile, no normality | P_MC[AUC < 400] > 0.025 | 877 |

## Results

**1. The first-order variance is the sampling variance.** Mean `Var_MC / Var_T` = **0.999** over the
cohort (min 0.857, max 1.158 — the extremes are the low-CrCl tail, where 1/CL is most curved).
The delta-method linearisation through Cockcroft–Gault, Matzke, the two-compartment roots and the
phase quotients loses nothing that matters at these uncertainties.

**2. The decision agrees.** Under the same ±2·sd rule, T and MC-sd disagree on **28 of 5 000
patients (0.6 %)**: 896 common WARNs, 13 T-only, 15 MC-only — noise of K = 1 000 draws, symmetric.
Against the true quantile (MC-q, 877 WARNs) T has 42 extra and misses 10: the ±2σ rule is slightly
*conservative* for the right-skewed 1/CL, which is the safe direction; the affine truth and the
sampling truth pick the same patients at 94.8 % / 98.5 %.

**3. The 300× is real.** `Var_N / Var_MC` = **300.9** — the naive chain's over-statement measured
against sampling truth is the same as against first-order truth (300.7). Against MC-q the naive
chain produces **1 926 spurious WARNs and silences 0** (877 common). The garbling finding of the
phase decomposition is not an artefact of linearisation.

## What this changes

- Paper A §6.4 residual (v) "algebraic, no distributional adequacy claimed" becomes *"algebraic;
  adequacy measured: Var_MC/Var_T = 0.999 (0.857–1.158), decision agreement 99.4 % under the same
  rule, on the RQ4 cohort"*. It is still not a theorem — it is a measurement on one model at one
  set of uncertainties — and the review record marks the reviewers' item as **measured, not
  proved**.
- Dissertation Contribution 1 (GUM-through-ODE vs Monte Carlo) gets its clinical number: the
  O(N_p) first-order propagation reproduces a 5 000 × 1 000-draw Monte Carlo's variance to 0.1 %
  on average and its WARN decision on 99.4 % of patients.
- The quantile rule is the more honest clinical rule; the ±2σ rule over-warns by ~4 % on this
  chain. That is a *rule* choice, orthogonal to the propagation, and the paper's WARN counts are
  stated under the ±2σ rule throughout.

## Reproduce

```bash
bin/souc run docs/research/sounio/rq4_vanco_mc_adequacy.sio            # ~1 min, Madaros
SOUNIO_SOUC_ENGINE=lean_single bin/souc run docs/research/sounio/rq4_vanco_mc_adequacy.sio   # byte-identical
```

Engine note: a struct field named `var` is rejected by lean_single (keyword) and accepted by
Madaros — the field is `vr` for that reason.
