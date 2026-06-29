# Pre-registration — the defensible-conjunction ablation (cadence × joint polarity)

Date: 2026-06-29
Branch: `research/novelty-ablation` (base `d7bf33814`, madaros `cb2f7685…`)
Status: **PRE-REGISTERED — written before any conflict numbers were observed.**
Follows: the variance-bonus result (`act-var-ablation-results-v2-isolated-2026-06-29.md`,
which disconfirmed the `score_mode=5` second-moment *additive bonus* as a mechanism).

## Why this experiment

The verified claim boundary says the narrow novelty rests on a **conjunction**:
per-decision bandit cadence + JOINT variable+polarity sampling + regime-gated exploration.
The variance *additive-bonus* conjunct is dead (prior result). This tests what remains.

## Toggle surface (and an honest constraint)

Cleanly ablatable from the harness:
- **score_mode**: `1` = pure `act_mean` (deterministic, VSIDS-like baseline) vs `3` =
  Thompson sampling on variable selection (per-decision bandit cadence; samples from
  N(act_mean, act_var) with sign noise, spread scaled by regime trust).
- **phase_mode**: `0` = saved-phase polarity (baseline) vs `1` = Beta-Bernoulli polarity
  Thompson sampling (the joint second axis).

NOT ablatable (stated limitation): **regime-gating cannot be toggled** — `regime_explore_trust`
is intrinsic, reset every `smt_solve()`, "cannot be pinned externally" (smt.sio:229). For
mode 3 it scales the TS spread; for mode 1 (no exploration term) it is inert. So the
conjunction is tested **with regime-gating intrinsically on**; its isolated contribution is
not measurable here.

**Sharp caveat carried forward:** mode 3's Thompson spread is `sqrt(act_var)·regime_trust` —
it still depends on the same growing `act_var` whose *additive* use was just disconfirmed.
Whether the *sampling* use of act_var also harms is genuinely open; this experiment tests it.

## Design — 2×2, process-isolated, z3-gated

Same instances as v2: overconstrained random-3-SAT UNSAT, α=6.0, n=50, 40 instances,
seeds 3000–3039. Each instance solved in its OWN process (one instance per process — the
validated workaround for the cross-`smt_new()` state leak). Four configs per instance:

| label | score_mode | phase_mode | what it adds |
|---|---|---|---|
| BASE | 1 | 0 | none (mean var-pick, saved phase) |
| BANDIT | 3 | 0 | + per-decision Thompson on variable |
| POL | 1 | 1 | + Beta-polarity Thompson only |
| CONJ | 3 | 1 | + both (the defensible conjunction, regime on) |

`restart_budget=0`, `no_ple=1`, `disc_beta_scale` irrelevant (modes 1/3 don't read it).
Deterministic per instance (hash is a pure fn of var-index and n_decisions) → one run per cell.

## Metric & pre-committed outcomes

- Metric: **conflicts-to-completion** (UNSAT), paired per instance vs BASE.
- Per config: mean Δ = conflicts(BASE) − conflicts(config) (positive ⇒ config *helps*),
  95% bootstrap CI, paired sign test.
- z3-gate: every instance must be definitive UNSAT and z3-agree; censored/disagreeing excluded.

**Pre-committed reads:**
- CONJ Δ > 0 significant ⇒ the defensible conjunction reduces conflicts here — the first
  *positive* evidence for the narrow claim (bounded to this regime).
- CONJ Δ ≤ 0 (null or harmful) ⇒ **report it.** Combined with the decomposition (BANDIT, POL),
  this tells us whether the cadence axis or the polarity axis is the problem. If BANDIT harms,
  the act_var dependence sinks the Thompson-cadence conjunct too, and the narrow claim is in
  serious trouble regardless of framing.
- The decomposition is the point: a CONJ win built on a harmful BANDIT axis would be suspect;
  a CONJ win with both axes individually neutral-or-helpful is the clean story.

## Boundaries

Synthetic mechanism test, pilot scale, single regime (overconstrained UNSAT). Regime-gating
not isolated. No external benchmark, no CDCL parity, no verified certificate. No SOTA / no
public-novelty claim. Null is shippable and informative.
