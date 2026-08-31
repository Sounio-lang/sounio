# Preregistered predictions — SAN-v7 threshold sweep (2026-08-06, ~20:30 UTC)

**Status: PREREGISTERED.** Written and committed BEFORE jobs 8732/8733/8734
(thresholds 0.85 / 0.90 / 0.975) and 8738/8739 (ViT/GPT legs) produced any
ledger. The git timestamp of this commit is the precedence proof. If the
measurements falsify these predictions, the falsification will be reported
in the paper with the same prominence as a confirmation.

**Timing disclosure (added 2026-08-06 21:45 UTC, before inspecting any sweep
output).** Job 8732 (threshold 0.85, prediction P1) reached a terminal state
at 21:19:29 UTC — 22 minutes *before* the commit below (21:41:54 UTC). The
author had not inspected its output: the last cluster inspection before
writing this file was ~20:20 UTC, at which point all three sweep jobs were
RUNNING, and the first read of 8732's log happens after this disclosure is
committed. The git-timestamp precedence therefore provably holds for P2, P3,
P4 (8733/8734 still RUNNING at commit time) and P5 (8738/8739 PENDING), but
**not** for P1, which rests on the author's word alone. We report this
rather than retro-fit the precedence claim.

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

---

## RESULTS — scorecard (added 2026-08-07, all five jobs terminal)

Measured values (jobs 8732/8733/8734/8738/8739; v7/v7b from 8717/8720):

| θ | final acc | S_m (TMAC) | latency | exit_frac late post-τ | t* |
|---|---|---|---|---|---|
| 0.80  | 0.7998 | 9 799  | 1.41× | 0.71–0.98 | 11 |
| 0.85  | 0.8293 | 12 163 | 1.27× | 0.89–0.96 | 15 |
| 0.90  | 0.8379 | 10 357 | 1.18× | 0.85–0.92 | 10 |
| 0.95  | 0.8594 | 12 757 | 1.12× | 0.68–0.86 | 14 |
| 0.975 | 0.8688 | 13 795 | 1.04× | 0.74–0.79 | 15 |

**P1 (0.85):** acc 0.8293 ∈ [0.805, 0.835] CONFIRMED; S_m 12 163 ∉
[10 200, 11 400] FALSIFIED (high); latency 1.27× ∈ [1.25, 1.37] CONFIRMED.

**P2 (0.90):** acc 0.8379 ∈ [0.825, 0.855] CONFIRMED; S_m 10 357 ∉
[11 200, 12 400] FALSIFIED (low); latency 1.18× ∈ [1.15, 1.28] CONFIRMED.

**P3 (0.975):** acc 0.8688 ∈ [0.855, 0.880] CONFIRMED; S_m 13 795 ∈
[12 800, 14 200] CONFIRMED; latency 1.04× ∈ [1.02, 1.12] CONFIRMED.

**Monotone constraints:** acc 0.7998 ≤ 0.8293 ≤ 0.8379 ≤ 0.8594 ≤ 0.8688
CONFIRMED (perfectly monotone). Latency 1.41 ≥ 1.27 ≥ 1.18 ≥ 1.12 ≥ 1.04
CONFIRMED (perfectly monotone). S_m 9 799 ≤ 12 163 ≰ 10 357 FALSIFIED —
the 0.90 point dips below the 0.85 point.

**P4 invariants:** L1 conservation PASS with exits > 0 in all three (exits
1961/1750/1585 per 10 000, exact) CONFIRMED; L2 PASS in all three
CONFIRMED; exit_frac(t*) = 0.000 with L6 FAIL by construction in all three
CONFIRMED; "S_m below within-run EarlyStop in every case" FALSIFIED —
EarlyStop legs were 9 967 (t*=23), 10 798 (t*=25), 7 891 (t*=18) TMAC, so
the SAN won only the 0.90 pairing (10 357 < 10 798).

**P5 (ViT/GPT legs):** ViT — τ reached (t*=4, 0.3121 ≥ 0.251) CONFIRMED;
post-τ exit fraction 0.31–0.38, ten times the pre-v7 line's 0.03 CONFIRMED;
final acc 0.3763 ≥ τ CONFIRMED; L6 PASS at t* (0.342) — first L6 pass on
full CIFAR-10. GPT — τ reached (t*=4, 0.1670 ≥ 0.165) CONFIRMED; L6 PASS
(0.120); but final acc 0.1309 < τ at end of post-τ training FALSIFIED;
inference speedup 3.33× (not predicted; recorded).

**Totals:** accuracy predictions 5/5 (3 intervals + monotone + all L2);
latency predictions 5/5; S_m predictions 1/3 intervals + monotone falsified;
baseline-invariant falsified; P5 5/6 sub-claims.

## What the falsifications teach (mechanism)

1. The accuracy and latency frontiers in θ are smooth, monotone, and
   predictable within ±0.015 acc / ±0.08× — the threshold is a reliable
   deployment dial.
2. The training-burden frontier is NOT monotone in θ because S_m is
   dominated by t* jitter: the necessary part nec(t*) swings with the epoch
   at which τ happens to be crossed (t* ∈ {10, 11, 14, 15} across identical
   configs), and that swing (~2 000 TMAC) exceeds the inter-threshold
   effect. S_m(θ) = nec(t*) + discounted post-τ epochs; only the second
   term is θ-controlled.
3. The EarlyStop baseline is itself high-variance (7 891–13 705 TMAC across
   identical configs), so any blanket "SAN beats EarlyStop" claim is
   unstable; within-run pairing is necessary and still noisy. The stable,
   reportable claim is the SAN's own frontier, not the pairwise win.
