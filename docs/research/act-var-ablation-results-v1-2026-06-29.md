# Results v1 — β=0 vs β>0 variance-aware branching ablation

Date: 2026-06-29
Branch: `research/novelty-ablation` (base `d7bf33814`, madaros `cb2f7685…`)
Pre-registration: `docs/research/act-var-ablation-prereg-2026-06-29.md` (incl. pilot-revision addendum)
Apparatus: `benchmarks/solver/beta_ablation_v1.sio` + `scripts/research/analyze_beta_ablation.py`

## Headline (honest, negative)

In the only conflict-bearing regime this bounded DPLL(T) reaches via random 3-SAT —
**overconstrained UNSAT, α=6.0, n=50** — turning the discounted empirical-Bernstein
variance bonus (`score_mode=5`) **ON (β>0, `disc_beta_scale=1.0`) makes the solver
WORSE** than OFF (β=0, mean-only):

| metric | β=0 | β>0 |
|---|---|---|
| mean conflicts-to-UNSAT | 210.4 | 313.5 |

- mean Δ (β=0 − β>0) = **−103** ; 95% bootstrap CI **[−171, −11]** (excludes 0).
- paired sign: β>0 **better on 7 / worse on 33** of 40 (the robust headline; one
  outlier seed 3028 where β=0=1621/β>0=327 inflates the mean, but the 33/40 sign
  split is outlier-insensitive).
- Gate: 40/40 instances kept — all returned definitive UNSAT, **all agreed with z3**,
  none censored by the 100000-decision cap, none excluded.

So H1 ("β>0 reduces conflicts-to-UNSAT") is **rejected in this regime**; the measured
effect is a significant *increase* in conflicts.

## Interpretation — a disconfirmation in the FAVORABLE regime (no rescue)

An earlier draft of this note framed the negative as "consistent with theory — α=6.0 is a
large-gap regime where variance exploration is expected to waste effort." That framing was
**motivated reasoning and is retracted.** The "large-gap" label was assigned *post-hoc*
because it rescued the result — exactly the a-priori-vs-measured circularity the
pre-registration forbids.

The a-priori truth points the other way. **Uniform random 3-SAT has statistically
exchangeable variables by construction** — in expectation no variable is more constrained
than any other, at *any* clause/variable ratio (α controls hardness/UNSAT, not
cross-variable asymmetry). So a-priori this is a **small-gap** regime: variables are hard
to tell apart — precisely the setting where UCB-V predicts the variance bonus **should
help**. The conflict spread (45–1621, median 141; genuine search, not a few-variable
blowout) is consistent with symmetric, not dominated-by-few.

**Honest reading:** the variance bonus significantly **hurt (33/40) in its own
theoretically-favorable (small-gap) regime.** This is a **negative signal for the
variance / UCB-V conjunct specifically** — which the verified claim boundary already flags
as the riskiest conjunct and the "open-frontier bet." It is *not* evidence for some other
arm, and it does not license "mechanism still alive, just need the small-gap test": the
a-priori-symmetric regime *was* the small-gap test, and it did not show benefit.

Whether *any* regime shows benefit remains open (speculative, one line): a genuinely
large-gap / planted-asymmetry generator is untested. But pursuing it requires *measuring*
cross-variable gap structure, which converts the label from a-priori to empirical — and if
that comes back small-gap, this result gets *more* damning, not less.

## What is solid (and what is not)

Solid:
- End-to-end apparatus works: deterministic generation → solve at both β → byte-identical
  DIMACS → z3 correctness gate → paired stats with CI + sign test.
- `score_mode=5` runs bounded/correct; solver verdicts agree with z3 (3/3 spot-checks +
  40/40 gate).
- A real, pre-registered, z3-verified negative finding in the large-gap regime.

NOT established:
- Any *benefit* of variance-aware branching (the small-gap arm is untested).
- Anything at external-benchmark scale, CDCL parity, or with verified UNSAT certificates
  (solver emits no DRAT today). No SOTA claim. No public-novelty claim.

Safe wording: "a pre-registered, z3-gated ablation finds the candidate variance-aware
(UCB-V) variable bonus *increases* conflicts-to-UNSAT (worse on 33/40) in an
a-priori-symmetric random-3-SAT UNSAT regime — a negative signal for the variance conjunct
in the regime where UCB-V theory predicts it should help. Apparatus, z3 gate, and
determinism verified."

## Next (decision point — see session checkpoint)

The honest decision is **not** "how do we find a regime where the variance score helps"
(that is where the motivated reasoning lives). It is:

> **Does the variance / UCB-V conjunct survive, or do we pivot the narrow claim to the
> conjuncts NOT taking empirical damage** — per-decision bandit cadence + JOINT
> variable/polarity sampling + regime-gated exploration — **dropping the second-moment
> (`act_var` / discounted-Bernstein) variable score entirely?**

The verified boundary always rated the variance score the riskiest, prior-art-adjacent
conjunct; this run is the first empirical evidence against it. The joint-axis + cadence +
regime-gating conjunction (the actually-defensible claim) does not depend on the variance
term and is untouched by this result.

Subordinate options if the variance angle is kept alive: a planted-asymmetry (genuinely
large-gap) generator with *measured* gap structure; or a descriptive `disc_beta_scale`
sweep (never treated as confirmation). Receipt half (DRAT + cake_lpr) stays deferred.
