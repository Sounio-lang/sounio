# Results — defensible-conjunction ablation (NULL, valid)

Date: 2026-06-29
Branch: `research/novelty-ablation` (base `d7bf33814`, madaros `cb2f7685…`)
Pre-registration: `docs/research/conjunction-ablation-prereg-2026-06-29.md`
Apparatus: `benchmarks/solver/iso_conjunction_template.sio` + `scripts/research/run_iso_conjunction.sh`
+ `scripts/research/analyze_conjunction.py`

## Result (process-isolated, z3-gated, 40/40 UNSAT, none excluded)

2×2 over (score_mode ∈ {1 mean, 3 Thompson}) × (phase_mode ∈ {0 saved, 1 Beta-TS}),
overconstrained random-3-SAT UNSAT (α=6.0, n=50). Δ = conflicts(BASE) − conflicts(config),
positive ⇒ config helps.

| config | score/phase | mean conflicts | Δ vs BASE | 95% CI | better/worse/tie |
|---|---|---|---|---|---|
| BASE   | 1 / 0 | 82.6 | — | — | — |
| BANDIT | 3 / 0 | 85.3 | −2.8 | [−7.3, +1.4] | 19/20/1 |
| POL    | 1 / 1 | 79.9 | +2.7 | [−3.1, +8.3] | 22/16/2 |
| **CONJ** | 3 / 1 | 82.5 | **+0.1** | [−7.2, +6.5] | 18/19/3 |

**Every cell's CI includes 0 → NULL.** The defensible conjunction (CONJ) is statistically
indistinguishable from plain mean-only branching here (Δ=+0.1). The decomposition: the
Thompson-cadence axis trends slightly harmful (BANDIT −2.8, n.s.), the Beta-polarity axis
trends slightly helpful (POL +2.7, n.s.); they roughly cancel in CONJ.

## Two things this settles

1. **No positive evidence for the narrow conjunction in this regime** — but, unlike the
   variance *additive bonus* (which was −146, worse 35/40), the conjunction does **no harm**.
   The narrow claim is undamaged here; it is simply unsupported here.
2. **The *sampling* use of `act_var` is benign, the *additive-bonus* use is toxic.** BANDIT
   (mode 3) samples from N(act_mean, act_var) and lands at Δ=−2.8 (≈neutral); the additive
   `mean + β·sqrt(act_var)` bonus (mode 5/0) was Δ=−146. Same quantity, opposite consequence —
   this vindicates the pre-registration's caveat and sharpens what is actually wrong with the
   variance angle (it is the additive ranking distortion, not touching act_var per se).

## Why "null here" is expected and what it implies

This regime (overconstrained UNSAT, α=6.0) is one where **plain greedy mean already solves in
~83 conflicts** — there is little headroom, and few hard branch decisions for a smarter
heuristic to win. A wash here does not condemn the conjunction; it says **this is the wrong
regime to demonstrate its value.** To find positive evidence (if it exists), the next test
needs a regime where mean-only struggles — e.g. structured/industrial-like instances with
heavy backtracking, or harder/larger instances — and ideally a non-toggleable-confound
handle on regime-gating (not available in this build).

## Honest standing of the narrow novelty claim, after both experiments

- Variance *additive bonus* (`score_mode=5`): **disconfirmed** (harmful, mechanism not
  calibration).
- Cadence + joint var/polarity conjunction (regime intrinsic): **null** in overconstrained
  UNSAT — no benefit, no harm. No positive support yet; undamaged.
- Net: the claim is **not yet supported by any positive result**, but the surviving conjuncts
  are benign. The path to support is a regime where greedy mean is weak — untested.

## Boundaries

Synthetic, pilot scale, single regime, regime-gating not isolatable. No external benchmark,
no certificate, no SOTA/public-novelty claim. Null is the reported, valid outcome.
