# Preregistered predictions — SAN-v7 threshold sweep (2026-08-06, ~20:30 UTC)

**Status: PREREGISTERED.** Written and committed BEFORE jobs 8732/8733/8734
(thresholds 0.85 / 0.90 / 0.975) and 8738/8739 (ViT/GPT legs) produced any
ledger. The git timestamp of this commit is the precedence proof. If the
measurements falsify these predictions, the falsification will be reported
in the paper with the same prominence as a confirmation.

## Basis (measured, jobs 8717/8720, full CIFAR-10, τ = 0.85, seed 17)

| threshold | t* | final acc | S_m (TMAC) | latency speedup | post-τ exit_frac |
|---|---|---|---|---|---|
| 0.80 (v7)  | 11 | 0.7998 | 9 799  | 1.41× | 0.52–0.98 |
| 0.95 (v7b) | 14 | 0.8594 | 12 757 | 1.12× | 0.68–0.86 |

Mechanism model: the gate estimates P(stage-k head correct); the threshold θ
admits exits with P > θ. Raising θ → fewer exits → higher final accuracy,
higher training burden, lower inference speedup. Local linearisation between
the two measured points gives slopes ≈ +0.40 acc / Δθ, +19.7k TMAC / Δθ,
−1.9 speedup / Δθ; we predict with intervals, not points, because there is
no reason to believe the frontier is exactly linear.

## Predictions (full CIFAR-10, ResNet-50, τ = 0.85, seed 17)

**Monotone constraints (all-or-nothing for the sweep):**
- final_acc(0.80) ≤ final_acc(0.85) ≤ final_acc(0.90) ≤ final_acc(0.95) ≤ final_acc(0.975)
- S_m is monotone non-decreasing in θ over the five points
- latency speedup is monotone non-increasing in θ over the five points

**P1 — threshold 0.85 (job 8732):**
- final acc ∈ [0.805, 0.835]
- S_m ∈ [10 200, 11 400] TMAC
- latency speedup ∈ [1.25×, 1.37×]

**P2 — threshold 0.90 (job 8733):**
- final acc ∈ [0.825, 0.855]
- S_m ∈ [11 200, 12 400] TMAC
- latency speedup ∈ [1.15×, 1.28×]

**P3 — threshold 0.975 (job 8734):**
- final acc ∈ [0.855, 0.880] (≥ τ, contract satisfied at end of training)
- S_m ∈ [12 800, 14 200] TMAC
- latency speedup ∈ [1.02×, 1.12×]

**P4 — invariants for all three:**
- L1 conservation PASS with exits > 0
- L2 PASS (val acc at t* ≥ 0.85)
- exit_frac at t* = 0.000 and L6 FAIL (by construction, as declared in §4.9)
- S_m below the within-run EarlyStop in every case (the exit discount pays
  for post-τ training)

**P5 — ViT/GPT legs (jobs 8738/8739, confidence-threshold exits, post-τ
training active):** both reach their declared τ within budget; post-τ exit
fraction for ViT rises above the 0.03 reported for the pre-v7 line; final
accuracy at end of post-τ training stays ≥ τ for both families.

## Scoring rule

A prediction interval counts as CONFIRMED if the measured value lands inside;
the monotone constraints count as a single all-or-nothing prediction. P4/P5
are point claims. Any falsified item is reported as falsified — that is the
point of preregistering.
