<!-- docs:meta
topic_id: repo.docs.research.knightian-operator-choice
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.knightian-operator-choice
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

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

## §7 — M3.5 update: Walley elicitation surface added (2026-05-01)

The deferred M3.5 milestone (Walley neighborhood model for the
elicitation surface) has now been mechanised. The elicitation /
propagation split announced in §6 is fully realised:

- **Elicitation surface** (`stdlib/epistemic/walley.sio`):
  `CredalSet` parametrises ε-contamination credal sets
  `M_ε(P0, [s_lo, s_hi])` (Berger 1990; Walley 1991 §2.9). Two
  distinct lifts to p-box are exposed:

  - `credal_to_pbox` — bounds the **mean** `E_P(X)` over `P ∈ M_ε`.
    Sharp on `lo_mean` / `hi_mean`; variance is a sound upper bound
    `(1 − ε)·σ²_0 + ε·R²` where `R = max(|μ_0 − s_lo|, |s_hi − μ_0|)`.

  - `credal_to_support_pbox` — bounds the **support** of `P` (vacuous
    Popoviciu). This is the lift that must be fed into Fréchet
    propagation (`pb_apply2_monotone_*`) for nonlinear monotone f, so
    the enclosure dominates realised values of `f(X, Y)` rather than
    just `E[f(X, Y)]`.

- **Propagation surface** (M2.5 — `pb_apply2_monotone_*`):
  unchanged. The composition `credal_to_support_pbox ∘
  pb_apply2_monotone_inc_dec` provides a copula-free outer enclosure
  of `f(X, Y)` for any joint distribution with marginals in
  `M_ε(C_X) × M_ε(C_Y)`.

### Math-review record

Two math-reviews via `bin/llm-offload -t math-review -p xai`:

1. **M3.5.0 thesis review** (2026-04-30) — caught two real bugs in
   the proposed design:
   - Variance upper bound was missing the cross-term
     `ε(1 − ε)(μ_0 − μ_Q)²`; my originally proposed formula
     `(1 − ε)·σ²_0 + ε·((s_hi − s_lo)/2)²` was UNSOUND
     (counter-example: μ=0, s_lo=0, s_hi=2, σ²_0=0, Q = δ_2, ε = ½
     gives Var_P = 1 > claimed 0.5).
   - Fréchet on the **mean rectangle** under-encloses `f(X, Y)`
     for nonlinear monotone f, because realised values lie outside
     `[lo_mean, hi_mean]`. The fix is to feed the **support
     rectangle** (`credal_to_support_pbox`) into Fréchet.

   The implementation was revised to use `Var_P(X) ≤ E_P[(X − μ_0)²]`
   (sound, no missing cross-term) and to expose two distinct lifts.

2. **M3.5 implementation review** (2026-05-01) — 11/11 findings
   `[OK]`. No bugs flagged. The corrected variance bound is sound,
   the support-vs-mean dichotomy resolves Bug B, the Lean theorems
   are complete, and the composition with M2.5 Fréchet is correct.

### Lean discharge

Five structural theorems mechanised in `formal/lean4/SounioWalley.lean`,
all proven in core Lean 4 (Nat-shadow, no `axiom`, no `sorry`, no
Mathlib):

- `walley_collapse_at_zero_nat` — at ε = 0, the lifted band
  collapses to a point (precise / Knowledge-recovery).
- `walley_collapse_gap_zero_nat` — gap is zero at ε = 0.
- `walley_vacuous_lo_at_one_nat` / `_hi_at_one_nat` /
  `_gap_at_one_nat` — at ε = 1, the band fills the support.
- `walley_gap_monotone_in_epsilon_nat` — the gap is non-decreasing
  in ε. Sound elicitation invariant: increasing Knightian doubt
  cannot tighten the posterior.
- `walley_frechet_composition_holds` — direct reduction of the
  Walley → Fréchet composition to
  `Sounio.Frechet.frechet_enclosure_monotone_inc_dec_nat`.

### Test evidence

Five `tests/stdlib/epistemic/test_walley_*.sio` round-trip tests,
all green via `bash scripts/run_sio_test_suite.sh walley`:

- `test_walley_collapse.sio` (ε = 0 → precise p-box).
- `test_walley_vacuous.sio` (ε = 1 → full-support band, R²
  variance).
- `test_walley_width_monotone.sio` (11-point ε sweep, monotone +
  endpoint check + linear midpoint).
- `test_walley_support_lift.sio` (support lift independent of
  μ_0/ε/σ²_0 — three different `CredalSet`s share the same
  `credal_to_support_pbox` output).
- `test_walley_frechet_compose.sio` (composition with
  `pb_apply2_monotone_inc_dec` for `f(x, y) = x/y`; 16-point grid
  sample verifies enclosure soundness).

### Operator surface (final)

| Surface | Module | Bounds | Use |
|---|---|---|---|
| Elicitation (mean band) | `walley.sio :: credal_to_pbox` | `E_P(X)` | linear-in-X aggregation, decision of "is the mean in range?" |
| Elicitation (support band) | `walley.sio :: credal_to_support_pbox` | `supp(P)` | feeder for nonlinear monotone propagation |
| Propagation | `knightian.sio :: pb_apply2_monotone_*` | `f(X, Y)` realisation | Fréchet enclosure, no copula assumption |
| Decision gate | `knightian.sio :: pb_within / pb_strictly_*` | yes/no / refuse | safety gate, conservative refusal |

The three operator surfaces compose without information loss, each
bound is provably sound, and each soundness claim is mechanised in
core Lean 4 (Nat-shadow). This realises the §1 design goal of
"epistemic honesty + operational tractability + verified soundness."

## Status

**M2 LOCKED. M2.5 LANDED. M3.5 LANDED.** The Walley elicitation
surface and the Fréchet propagation surface compose. The remaining
operator-level work is:

- Ferson-Klibanoff bridge: smooth-ambiguity wrapper over the same
  `CredalSet` to give a Bayesian decision-theoretic surface for
  cost-of-Knightian-uncertainty calculations (deferred as M5+).
- Float-Real lift: replace the `Nat`-shadow theorems with
  Mathlib-free Float ordered-field theorems for end-to-end
  mechanised soundness (deferred 4-6 weeks).

Re-open the operator decision only if a future PBPK model violates
per-argument monotonicity.
