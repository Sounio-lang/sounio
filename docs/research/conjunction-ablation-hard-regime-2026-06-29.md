# Results — conjunction ablation in the HARD regime (dissection: joint claim disconfirmed; polarity = unconfirmed trend)

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

## OUT-OF-SAMPLE CONFIRMATION (α=4.4, fresh seeds 4000–4079) — POL does NOT robustly replicate

The α=4.3 result was found after α=6.0 (null) and community structure (failed), so α=4.3 is a
post-hoc regime pick. Two hardening checks were run:

- **Split-half of the α=4.3 batch:** first 40 seeds Δ_POL=+22.8 (15/2, sig); second 40
  Δ_POL=+16.8 (10/10, CI includes 0). The pooled significance was partly carried by the first half.
- **Fresh batch at α=4.4** (m=220, seeds 4000–4079, 46 UNSAT kept, 0 censored, all z3-agreed):

| config | Δ vs BASE @ α=4.4 | 95% CI | better/worse | vs α=4.3 |
|---|---|---|---|---|
| BANDIT | −22.5 | [−34.7, −10.3] | 15/29 | replicates (harmful) |
| CONJ   | −38.8 | [−55.5, −22.2] | 11/33 | replicates (harmful) |
| **POL** | **+4.4** | **[−5.1, +13.9]** | 27/17 | was +19.6 → **n.s.** |

**Verdict:** the *harm* of the variable-side bandit (BANDIT) and the joint conjunction (CONJ)
is **robust and replicated** across α=4.3 and α=4.4. The *help* of polarity (POL) is **fragile**:
significant at α=4.3, not significant at α=4.4, second-half-null in its own split. POL is a
**consistently-positive-direction trend** (Δ = +2.7 @α6, +19.6 @α4.3, +4.4 @α4.4; majority
better in every batch) but does **NOT clear the bar for a confirmed positive result.** The
α=4.3 significance was partly favorable noise. Reported as a suggestive, unconfirmed trend —
not a result. (Also clears check #1: zero censoring at either ratio, so the kept set is not a
hardness-biased easy tail; and POL wins on decisions too, not just conflicts.)

## Honest reframe of the narrow novelty claim

The verified boundary's "per-decision bandit applied JOINTLY to variable AND polarity" is
**disconfirmed** — jointness is exactly what sinks it, because the variable half is harmful,
robustly and replicably (BANDIT and CONJ significantly worse at BOTH α=4.3 and α=4.4).

The polarity half is a **suggestive but UNCONFIRMED** positive: consistently positive in
direction across all three batches (Δ = +2.7 @α6 n.s., +19.6 @α4.3 sig, +4.4 @α4.4 n.s.;
majority-better every time) but it fails out-of-sample significance. Not a result.

> **Suggestive, unconfirmed:** per-decision Beta-Bernoulli polarity Thompson sampling *may*
> reduce conflicts-to-UNSAT vs saved-phase on hard random-3-SAT UNSAT — positive direction in
> 3/3 batches, but significant in only 1/2 hard-ratio batches. Needs more data / a stronger
> solver to confirm or kill.

Standing of the conjuncts after all experiments:
- variance additive bonus (score_mode=5): **disconfirmed** (toxic, mechanism; replicated via sweep).
- variable-side Thompson/act_var (score_mode=3): **disconfirmed** (toxic in hard regime; replicated α4.3+α4.4).
- joint var+polarity conjunction (CONJ): **disconfirmed** (net harmful, replicated).
- polarity-side Beta-Bernoulli sampling (phase_mode=1): **suggestive positive, UNCONFIRMED** (fails out-of-sample sig).
- regime-gating: not isolatable here.

## Boundaries / next

Synthetic, pilot scale (38 instances), single regime (hard random-3-SAT UNSAT), one solver
(bounded DPLL, no clause learning), no certificate. NOT SOTA / not public-novelty. The honest
next steps to firm up the *polarity* positive: replicate across more seeds and a second hard
ratio; confirm it survives on a stronger (CDCL) solver; and ship verified LRAT receipts. The
variable-side bandit and the "joint" framing should be dropped.
