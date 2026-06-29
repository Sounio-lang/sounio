# Results v2 (PROCESS-ISOLATED, VALID) — β=0 vs β>0 variance-aware branching

Date: 2026-06-29
Branch: `research/novelty-ablation` (base `d7bf33814`, madaros `cb2f7685…`)
Supersedes: `act-var-ablation-results-v1-2026-06-29.md` (RETRACTED — confounded by a
cross-`smt_new()` allocation-history state leak).
Apparatus: `benchmarks/solver/iso_cell_template.sio` + `scripts/research/run_iso_ablation.sh`
+ `scripts/research/analyze_iso.py`

## Method fix

v1's within-process design was confounded: a cross-`smt_new()` allocation-history state
leak (localized — NOT smt_new zeroing, NOT locals, NOT globals; a stale read in the search
path triggered by *varied* prior solves) made a later solve cell depend on earlier ones.
Fix: **process isolation** — each instance is solved in its OWN process that touches only
that instance (validated clean: a single-instance process's β=0 is stable, and β>0 is
order-independent; isolated values match the clean first-solve baselines). 40 instances ×
(β=0, β>0) across 40 independent processes.

## Result (VALID, pre-registered, z3-gated)

Overconstrained UNSAT random 3-SAT, α=6.0, n=50, 40 instances — all returned definitive
UNSAT, **all agreed with z3**, none censored, none excluded.

| metric | β=0 | β>0 (`disc_beta_scale=1.0`) |
|---|---|---|
| mean conflicts-to-UNSAT | 166.9 | 312.9 |

- mean Δ (β=0 − β>0) = **−146** ; 95% bootstrap CI **[−186, −108]** (excludes 0).
- paired sign: β>0 **better on 5 / worse on 35** of 40.

The variance bonus turned ON nearly **doubles** conflicts-to-UNSAT (+87%) and is worse on
**88% of instances**. The confounded v1 reported the same direction (worse 33/40); removing
the leak made the effect *larger and tighter*, not smaller — the contamination had inflated
several β=0 baselines (e.g. seed 3028: 1621 confounded → 191 clean), muddying rather than
manufacturing the signal.

## Interpretation (honest — the variance conjunct is disconfirmed in its favorable regime)

Uniform random 3-SAT has **statistically exchangeable variables by construction** → a-priori
a **small-gap** regime (variables hard to tell apart), which is exactly where UCB-V theory
predicts an empirical-variance exploration bonus **should help**. It significantly **hurt**.
This is a clean **negative signal for the variance / UCB-V conjunct specifically** — the
conjunct the verified claim boundary already flagged as the riskiest, prior-art-adjacent
"open-frontier bet." (No post-hoc "large-gap rescue": α controls hardness, not cross-variable
symmetry.)

## Mechanism vs calibration — settled by an isolated scale sweep

A 0-vs-1.0 test alone cannot separate "variance-awareness is the wrong mechanism" from
"this particular bonus is miscalibrated / too strong" (the formula's `3ζ·ln(S0)/S0` term is
large at small S0). The discriminator: sweep `disc_beta_scale` — flat damage ⇒ mechanism,
monotone dose-response ⇒ calibration. Run process-isolated (`iso_sweep_template.sio`):

| scale | mean conflicts | Δ vs β=0 | worse/40 |
|---|---|---|---|
| 0.00 | 166.9 | — | — |
| 0.05 | 297.0 | −130 | 35 |
| 0.10 | 300.2 | −133 | 35 |
| 0.25 | 307.2 | −140 | 35 |
| 0.50 | 309.5 | −143 | 36 |
| 1.00 | 312.9 | −146 | 35 |

The curve is **essentially flat**: at scale 0.05 (1/20th strength) the damage is already
89% of full, and a **20× change in scale moves the effect ~12%**. This is the **mechanism**
signature, not calibration — were the bonus merely too strong, shrinking it 20× would
largely recover β=0. It does not. The exploration *bias* (the variance term systematically
favours under-explored / small-S0 variables) is **directionally wrong** for this regime, so
even a tiny nudge flips enough argmax decisions to do near-maximal harm.

## What this licenses

- **Drop the second-moment variable score** (`act_var` / discounted-Bernstein, `score_mode=5`'s
  bonus) from the narrow novelty claim, *for this regime*. The scale sweep shows the harm is the
  mechanism (flat across a 20× range), not a tunable magnitude, so this is a directional verdict
  on the variance-exploration idea — not merely "this constant is too big." The claim is bounded
  to the tested regime (overconstrained random-3-SAT UNSAT), not proven universal.
- **Keep the conjuncts not under test and not damaged:** per-decision bandit cadence + JOINT
  variable/polarity sampling + regime-gated exploration. The defensible conjunction does not
  depend on the variance term; this result leaves it intact (and untested here).

## Boundaries (unchanged)

Synthetic mechanism test, pilot scale (n=50, 40 instances), single regime. NOT an external
benchmark, NOT CDCL parity, NO verified UNSAT certificate (solver emits no DRAT; receipt half
deferred). No SOTA, no public-novelty claim. Safe wording: "a pre-registered, process-isolated,
z3-gated ablation finds the candidate variance-aware (UCB-V) variable bonus significantly
increases conflicts-to-UNSAT (worse 35/40, +87%) in an a-priori-symmetric random-3-SAT UNSAT
regime, and a scale sweep (0.05–1.0, flat damage) attributes the harm to the mechanism rather
than miscalibration — evidence against the variance conjunct in this regime; the cadence +
joint-axis + regime-gating conjunction is untested and undamaged."

## Substrate debt surfaced (for the compiler lane)

Two bugs found en route, both documented: (1) `ae8befbad` regresses `test_smt` 6/6→0/6
(imported-SMT lowering); (2) a cross-`smt_new()` allocation-history stale read in the solver
search path (worked around here by process isolation; root cause needs a gdb-level hunt).
