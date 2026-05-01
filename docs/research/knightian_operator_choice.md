# Knightian Operator: Decision Document

**Status**: M0 deliverable. Locks the formal substrate for M2 (`stdlib/epistemic/knightian.sio`) and downstream M3-M6.

**Decision date**: 2026-04-30
**PI**: Demetrios Chiuratto Agourakis

## Question

How does Sounio represent **Knightian uncertainty** — uncertainty about the probability distribution itself, not just about the value drawn from it — at the type level, in a way that:

1. Composes with the existing `Knowledge<T>` (GUM variance) machinery
2. Has a formalizable Lean 4 soundness proof
3. Is computationally tractable (no infinite credal sets at runtime)
4. Has clinical operational meaning (defensible to a regulator)

## Three Candidates

### A. Walley imprecise probability (1991)

A *credal set* `M ⊆ Δ(Ω)` is a convex set of probability measures. Lower expectation `E_*(X) = inf_{P ∈ M} E_P(X)`, upper expectation `E^*(X) = sup`.

- **Pro**: most general; subsumes p-boxes, Dempster-Shafer, sets-of-priors.
- **Pro**: rich decision theory (Γ-maximin, E-admissibility).
- **Con**: credal sets can be infinite-dimensional; tractable only for restricted families (linear-vacuous, neighborhood models).
- **Con**: Lean formalization heavy — Mathlib has measure theory but not credal sets.

### B. Ferson probability box (p-box)

Bounded CDF: `F̲(x) ≤ F(x) ≤ F̄(x)` for all x. Operations preserve containment via interval extensions of distributional ops.

- **Pro**: simple representation (two CDFs); tractable arithmetic via CDF convolution bounds.
- **Pro**: established in **engineering UQ practice** (NASA, Sandia, NRC). Operational acceptance is the multiplier.
- **Pro**: composes naturally with GUM (mean ± variance) — a Gaussian p-box recovers GUM as a special case.
- **Pro**: Lean 4 formalization is straightforward (CDF as monotone function `ℝ → [0,1]`, containment as pointwise inequality).
- **Con**: discards information not encoded in marginal CDF (no joint dependence).
- **Con**: less expressive than Walley.

### C. Klibanoff smooth ambiguity (2005)

Second-order subjective distribution `μ` over priors `π`, with ambiguity-averse utility `V = ∫ φ(E_π[u]) dμ(π)` for concave `φ`.

- **Pro**: smooth, differentiable — friendly to optimization and AD.
- **Pro**: Knightian aversion parameterized by curvature of `φ`.
- **Con**: requires choice of `μ` (the prior over priors) — how do you elicit it?
- **Con**: clinical defensibility weak — a regulator asks "where did `μ` come from?" and the answer is "prior over priors", which is subjective^2.
- **Con**: Lean formalization moderate (second-order integration).

## Decision: **Ferson p-box**, with Walley as theoretical back-stop

Rationale, in order of weight:

1. **Operational acceptance (decisive)**. Engineering UQ standards (NASA-STD-7009A, NRC NUREG-1855, EPA Risk Assessment Guidance) reference p-boxes by name. A clinical regulator (FDA SaMD, EMA AI Act) trying to evaluate "did the verification cover the right uncertainty class?" will recognize p-boxes; will not recognize Walley credal sets without effort. **For a first-of-kind submission, recognizable matters more than maximally general.**

2. **GUM compatibility (decisive)**. Sounio's existing `Knowledge<T>` carries `(value, variance, confidence)`. A Gaussian p-box `[F̲, F̄]` where `F̲(x) = Φ((x - μ + δ)/σ)` and `F̄(x) = Φ((x - μ - δ)/σ)` recovers GUM-with-bias for `δ = 0` and adds a Knightian band for `δ > 0`. The lift `Knowledge<f64> → Knightian<f64>` is a one-line constructor.

3. **Lean tractability (decisive for 6-month horizon)**. P-box soundness in Lean 4: define `PBox := { lo, hi : ℝ → [0,1] // monotone lo ∧ monotone hi ∧ lo ≤ hi }`. Operations are `(lo₁ ⊕ lo₂, hi₁ ⊕ hi₂)` for some convolution `⊕`. Containment preservation is pointwise ≤ on the CDF. No measure theory, no credal sets, no Mathlib-heavy machinery. Estimable in 4-6 weeks of Lean work.

4. **Clinical defensibility (operational)**. A vancomycin trough's Knightian uncertainty has natural p-box form: assay bias `[lo, hi]` produces a CDF band `[F̲, F̄]` around the measured value. The doctor and the regulator both understand "the trough is between A and B with 95% confidence, accounting for assay calibration uncertainty". They do not understand "the credal set has these vertices".

5. **Walley as theoretical back-stop**. The PL paper (M5) cites Walley as the more general framework, positions p-box as the operational specialization, and notes that Sounio's substrate could be extended to Walley if needed. This protects publication from "you should have used Walley" referee comments.

## Operational definition (locked)

In Sounio (M2):

```sio
struct PBox {
    lo_mean: f64,    // lower CDF mean
    hi_mean: f64,    // upper CDF mean
    variance: f64,   // shared variance (Gaussian p-box)
    confidence: i64, // 0-1000 like Knowledge<T>
}
```

`Knightian<f64>` is implemented as `PBox`. A `Knowledge<f64>` lifts to `PBox` via `lo_mean = hi_mean = mean`. Operations `⊕` are interval extensions of the underlying GUM operations.

In Lean (M2):

```lean
structure PBox where
  lo : ℝ → ℝ        -- lower CDF
  hi : ℝ → ℝ        -- upper CDF
  monotone_lo : Monotone lo
  monotone_hi : Monotone hi
  bounded_lo : ∀ x, 0 ≤ lo x ∧ lo x ≤ 1
  bounded_hi : ∀ x, 0 ≤ hi x ∧ hi x ≤ 1
  containment : ∀ x, lo x ≤ hi x
```

Soundness theorem (target): if `f : ℝ → ℝ` is monotone, then the Sounio operation `pbox_apply f p` produces a p-box `q` such that for any true CDF `F` with `p.lo ≤ F ≤ p.hi`, the CDF of `f(X)` (where `X ~ F`) is contained in `[q.lo, q.hi]`.

## Pivot triggers

If during M2 we discover:
- p-box convolution arithmetic blows up variance super-linearly (operational failure) → switch to **Walley with neighborhood priors** (more general, slower).
- Lean p-box formalization stalls past 8 weeks → publish PL paper with operational p-box only, defer Lean to follow-up.

## References

- Walley (1991) *Statistical Reasoning with Imprecise Probabilities*. Chapman & Hall.
- Ferson, Kreinovich, Ginzburg, Myers, Sentz (2003) *Constructing Probability Boxes and Dempster-Shafer Structures*. Sandia SAND2002-4015.
- Klibanoff, Marinacci, Mukerji (2005) "A smooth model of decision making under ambiguity". *Econometrica* 73(6).
- NASA-STD-7009A (2016) *Standard for Models and Simulations*.
- JCGM 100:2008 (GUM) §F.2.4 — explicit acknowledgement that GUM does not handle distributional uncertainty.

## §6 — M2.5 update: joint-dependence pushback resolved by Fréchet enclosure (2026-04-30)

**Trigger.** Multi-provider consensus fan-out
(`bin/llm-offload --raw <prompt> deepseek xai gemini qwen`) of this
decision document on 2026-04-30 surfaced a substantive 2-way pushback
(DeepSeek + Grok 4.1 convergent; gemini/qwen blocked on OpenRouter
credits). See `docs/research/knightian_operator_consensus_2026-04-30.md`
for the consolidated review and `.claude/llm_offload_log.md` for the
audit row. Headline pushback:

> Joint (Vc, CL) dependence is unaddressed. Univariate p-boxes bound
> marginal CDFs only. Vancomycin Cmin is non-monotone in (Vc, CL);
> the two parameters are correlated (popPK r ≈ 0.3–0.7) and the
> correlation structure itself is Knightian. Naive p-box-on-marginals
> assumes independence and can over-refuse safe doses (positive
> correlation) or accept toxic ones (negative correlation).

**Resolution.** Math-review-first second-round
(`bin/llm-offload -t math-review -p xai`, validated 7/7 by Grok 4.1)
established the following theorem and applied it to the vancomycin
implementation already shipped under `claude/approx-effect`:

> **Theorem (Fréchet outer enclosure for monotone-in-each-arg).**
> Let `f : ℝ × ℝ → ℝ` be C¹ on a closed rectangle `R = [a,b] × [c,d]`
> with strict per-argument monotonicity `∂f/∂x > 0`, `∂f/∂y < 0` on `R`.
> Then for every `(x, y) ∈ R`, `f(a, d) ≤ f(x, y) ≤ f(b, c)`.
> Hence for any random variables `(X, Y)` supported on `R` with ANY
> joint distribution (any copula, any correlation, even unknown
> Knightian), the deterministic interval `[f(a,d), f(b,c)]` contains
> `f(X, Y)` almost surely.

Vancomycin Cmin satisfies the hypotheses for all physiological
`(Vc, CL, D, τ) > 0` (math-review 2026-04-30, Q2: `∂Cmin/∂Vc > 0`,
`∂Cmin/∂CL < 0`). The corner enumeration in
`stdlib/clinical/vancomycin_pbpk.sio`'s `predict_cmin_knightian`
**is already** the Fréchet outer enclosure — the joint-dependence
omission, real in general, **does not bite** on this functional shape.

**Implementation evidence (M2.5).**

- `stdlib/epistemic/knightian.sio` adds `pb_apply2_monotone_inc_dec`,
  `pb_apply2_monotone_inc_inc`, `pb_apply2_monotone_dec_dec` —
  generic Fréchet enclosure wrappers for any 2-arg monotone-in-each-
  arg PBPK / engineering computation.
- `stdlib/clinical/vancomycin_pbpk.sio`'s `predict_cmin_knightian`
  header now states the soundness theorem explicitly and links to
  the wrapper.
- `tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio`
  empirically verifies the enclosure: 250 samples (50 each at
  `r ∈ {-0.7, -0.3, 0, +0.3, +0.7}`) with deterministic LCG, all
  enclosed by the predicted Knightian Cmin band. PASS.
- `formal/lean4/SounioFrechet.lean` provides three Mathlib-free,
  fully-proven Lean theorems (`Nat`-shadow): `_inc_dec`, `_inc_inc`,
  `_dec_dec`, plus a `vancomycin_cmin_frechet_enclosure` instantiation.
  Build green; Grok math-review approved 5/5 statements.

**Lean budget revision.**

DeepSeek's consensus estimate (`docs/research/knightian_operator_consensus_2026-04-30.md`)
of 800–1200 lines for sound univariate p-box arithmetic and ~3000
lines for the safety theorem is the right order-of-magnitude target
for a Mathlib-grounded discharge. Grok's 250-line estimate covers
the structural skeleton at the `Nat` level (achieved here in
`SounioFrechet.lean`). Full Float-Real lift remains a 4-6-week
follow-up.

**Where this DOES NOT work.** The Fréchet enclosure relies on
strict per-argument monotonicity over the rectangle. It breaks if
the function has a saddle point or non-monotone region in any
sub-rectangle reachable from the marginals. For vancomycin
specifically Grok confirmed (Q5) no physiological regime violates
the hypothesis. For other PBPK / engineering computations,
applicability must be verified case by case — any future model
introduction MUST run the same math-review (`-t math-review`) check
before relying on `pb_apply2_monotone_*`.

## Status

**M2 LOCKED. M2.5 LANDED.** M3.5 (Walley neighborhood model for
elicitation surface; lift to p-box at propagation surface via
Fréchet) deferred as the next operator-level milestone. Re-open
the operator decision only if a future PBPK model violates
per-argument monotonicity.
