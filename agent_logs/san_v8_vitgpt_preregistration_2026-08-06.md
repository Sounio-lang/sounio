# Preregistered predictions — SAN-v8 (cost-aware gate) + ViT/GPT frontier sweeps (2026-08-06, ~23:00 UTC)

**Status: PREREGISTERED.** Committed BEFORE any of the six jobs below was
submitted to the cluster. Unlike the first preregistration
(`san_v7_frontier_preregistration_2026-08-06.md`), this one strictly precedes
all job starts — the submission happens after this commit lands, and the
commit timestamp is verifiable against the Slurm SubmitTime of every job.

## E — SAN-v8: cost-aware gate (ResNet-50, full CIFAR-10, τ = 0.85, seed 17,
adaptive exit threshold 0.95, `SAN_LARGE_GATE_COST_AWARE=1`)

Mechanism: the v7 gate BCE estimates P(stage-k head correct). v8 keeps the
target and adds pos_weight = (n_stages − k) per stage, so "correct AND cheap"
is worth more gradient than "correct AND expensive"; the learned equilibrium
shifts early-gate firing propensity up (unit test: mean sigmoid at gate 0
moves 0.093 → 0.256 while deep gates stay, smoke scale). Reference point at
the same threshold (v7b, job 8720): final acc 0.8594, S_m 12 757 TMAC,
latency 1.12×, late post-τ exit_frac 0.68–0.86.

- **E1:** late post-τ exit_frac ∈ [0.75, 0.97] (above the v7b band).
- **E2:** final acc ∈ [0.820, 0.870]. We explicitly allow a dip below τ —
  pushing early exits harder spends accuracy; the question is how much.
- **E3:** latency speedup ∈ [1.10, 1.40] (at least v7b's 1.12×).
- **E4:** S_m ∈ [10 000, 13 000] TMAC. Wide on purpose: the first
  preregistration falsified S_m monotonicity because nec(t*) jitter
  dominates; we now predict with the jitter priced in.
- **E5:** L1 conservation PASS with exits > 0; L2 PASS; L6 FAIL at t* by
  construction (declared in §4.9).

## G — ViT frontier sweep (confidence threshold Δ; reference Δ = 0.45, job
8738: final acc 0.3763, exit@t* 0.342, late exits 0.31–0.38, 1.17×)

- **G1:** late post-τ exit_frac monotone decreasing across Δ ∈ {0.35, 0.45,
  0.55, 0.65}.
- **G2:** final acc monotone non-decreasing in Δ.
- **G3:** every run reaches τ = 0.251 (L2 PASS) and ends ≥ τ.
- **G4:** latency speedup monotone decreasing in Δ.
- Intervals: Δ=0.35 → exit ∈ [0.40, 0.65], acc ∈ [0.330, 0.380];
  Δ=0.55 → exit ∈ [0.15, 0.32], acc ∈ [0.370, 0.430];
  Δ=0.65 → exit ∈ [0.04, 0.20], acc ∈ [0.390, 0.470].

## G — GPT frontier sweep (reference Δ = 0.31, job 8739: final acc 0.1309
< τ = 0.165, late exits ~0.95, 3.33× speedup)

- **G5:** late exit_frac monotone decreasing: Δ=0.40 ∈ [0.75, 0.93];
  Δ=0.50 ∈ [0.45, 0.85].
- **G6:** final acc monotone increasing: Δ=0.40 ∈ [0.135, 0.175];
  Δ=0.50 ∈ [0.150, 0.200].
- **G7:** the open question P5 left: at least one of the two new Δ values
  ends ≥ τ = 0.165.
- **G8:** latency speedup monotone decreasing: Δ=0.40 ∈ [1.8, 2.8];
  Δ=0.50 ∈ [1.3, 2.2].

## Scoring rule

As before: intervals CONFIRMED if the measured value lands inside; monotone
constraints are all-or-nothing; every falsification is reported in the paper
with the same prominence as a confirmation.
