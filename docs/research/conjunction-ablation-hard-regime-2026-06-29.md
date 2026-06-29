# Results — conjunction ablation in the HARD regime (the dissection + first positive signal)

Date: 2026-06-29
Branch: `research/novelty-ablation` (base `d7bf33814`, madaros `cb2f7685…`)
Apparatus: `benchmarks/solver/iso_conj_a43_template.sio` + `scripts/research/run_iso_conjunction.sh`
+ `analyze_conjunction.py`; hardness pilot `hardness_pilot_template.sio`.
Follows the NULL result at α=6.0 (`conjunction-ablation-results-2026-06-29.md`).

## Why a harder regime

α=6.0 overconstrained UNSAT gives mean-only BASE only ~83 conflicts — little headroom, so
all axes were null. A hardness pilot (uniform 3-SAT, n=50, vary α) found that **near the
transition (α≈4.30, m=215)** UNSAT instances cost BASE ~186 conflicts and still terminate
(no decision-cap censoring). That is the "mean struggles but solver still finishes" window.

## Result (process-isolated, z3-gated; 80 seeds → 38 UNSAT kept, 42 excluded as SAT/disagree)

n=50, α=4.30. Δ = conflicts(BASE) − conflicts(config); positive ⇒ config helps. BASE = 192.9.

| config | score/phase | mean conflicts | Δ vs BASE | 95% CI | better/worse |
|---|---|---|---|---|---|
| BASE   | 1 / 0 | 192.9 | — | — | — |
| BANDIT | 3 / 0 | 230.2 | **−37.3** | [−51.7, −24.2] | 9/29 |
| POL    | 1 / 1 | 173.3 | **+19.6** | [+5.3, +35.1] | 25/12 |
| CONJ   | 3 / 1 | 212.8 | −19.9 | [−38.8, −0.4] | 14/22 |

All three CIs exclude 0. This **dissects** the conjunction:

- **POL (Beta-Bernoulli polarity Thompson sampling) significantly HELPS** — ~10% fewer
  conflicts-to-UNSAT (25/12 instances). **First positive signal in the whole arc.**
- **BANDIT (Thompson variable selection, samples N(act_mean, act_var)) significantly HURTS**
  (−37.3, 29/9). The variable-side use of the disconfirmed `act_var` is toxic even as
  sampling spread once instances are hard enough — consistent with the variance-bonus result.
- **CONJ (both) is net harmful** only because it includes the harmful BANDIT axis:
  BANDIT(−37) + POL(+20) ≈ CONJ(−20). The good polarity axis cannot rescue the bad var axis.

## Coherence with α=6.0 (same directions, amplified)

| config | Δ @ α=6.0 (null) | Δ @ α=4.30 (hard) |
|---|---|---|
| BANDIT | −2.8 (n.s.) | −37.3 (sig) |
| POL    | +2.7 (n.s.) | +19.6 (sig) |
| CONJ   | +0.1 (n.s.) | −19.9 (sig) |

The hard regime did not change the signs — it made the already-present trends significant.
This is strong internal consistency: the easy regime simply lacked the headroom to resolve them.

## Honest reframe of the narrow novelty claim

The verified boundary's "per-decision bandit applied JOINTLY to variable AND polarity" is
**not supported** — jointness is exactly what sinks it, because the variable half is harmful.
But a *narrower* claim now has its **first positive evidence**:

> **Per-decision Beta-Bernoulli polarity (phase) Thompson sampling significantly reduces
> conflicts-to-UNSAT vs saved-phase, on hard near-transition random-3-SAT UNSAT (≈10%,
> 25/12, CI [+5.3,+35.1]).**

Standing of the three conjuncts after all experiments:
- variance additive bonus (score_mode=5): **disconfirmed** (toxic, mechanism).
- variable-side Thompson/act_var (score_mode=3): **disconfirmed** (toxic in hard regime).
- **polarity-side Beta-Bernoulli sampling (phase_mode=1): SUPPORTED (positive) in this regime.**
- regime-gating: not isolatable here.

## Boundaries / next

Synthetic, pilot scale (38 instances), single regime (hard random-3-SAT UNSAT), one solver
(bounded DPLL, no clause learning), no certificate. NOT SOTA / not public-novelty. The honest
next steps to firm up the *polarity* positive: replicate across more seeds and a second hard
ratio; confirm it survives on a stronger (CDCL) solver; and ship verified LRAT receipts. The
variable-side bandit and the "joint" framing should be dropped.
