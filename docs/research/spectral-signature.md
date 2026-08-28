<!-- docs:meta
topic_id: repo.docs.research.spectral-signature
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.spectral-signature
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The object that composes by product — and its spectral signature (the first positive)

*The generative turn, with the filter attached. Annihilation is binary and composition must be
multiplicative; additive composition has no zero divisors (retroactively explaining three of the four
training negatives, all additive) — and the same constraint is Part II's conatus (norm multiplicativity),
so philosophy and engineering coincide before the data. The one place multiplicative composition already
lives in real training is the gradient. This is the first construction that passes the filter and produces
a measurable, distinctive prediction. Implements OPUS-4.8-EXTRA's generative turn.*

## The necessary-condition filter
Under **additive** composition (`θ + g₁ + g₂`) there is no product, order is irrelevant, and `g₁+g₂=0` only
by **opposition** (`g₂=−g₁`) — opposition is a *negative* product, not zero. A vector space under addition
is, for annihilation, a division algebra: nothing nonzero vanishes except the negative of the other. So the
missing object **must compose by product (or at least non-additively)**, or no sedenion structure can appear
by construction. This eliminates plain SGD, summed regularizers, and sum-aggregated losses — three of the
four earlier negatives tested additive objects and could not have shown annihilation. It is the same
condition Part II imposes: conatus is norm-multiplicativity, vacuous in an additive system.

## Where multiplicative composition already lives: the gradient
`J = J_L J_{L-1} ⋯ J_1`. **Vanishing gradient is annihilation by composition:**
`σ_min(∏J) ≪ ∏σ_min(J)`. The entire residual-connection / normalization industry exists to fight it; none
of it was ever formulated as zero-divisor structure. Same for the O/S-SSM: the state transition is a
product, and `σ_min` of the product over time is how much of the past survives — memory is preserved
composition, forgetting is composition failing.

## The distinct prediction, and its confirmation (`spectral_signature.py`, no training)
Classical theory reads vanishing gradient as **magnitude**: the whole spectrum slides down, and dynamical
isometry (Saxe, Pennington) keeps it concentrated near 1. The algebra predicts something else: in 𝕊 the
`L_xᵀL_x` spectrum is `{D₁−2q ×4, D₁ ×8, D₁+2q ×4}`, so a **low-multiplicity subspace collapses while the
bulk stays healthy — a gap opens.**

- **Test A** (single `L_x` → a zero divisor): exactly **4** singular values collapse (`0.136→0.003` as
  `δ→0`), the median stays `1.0`, and the gap `σ4/σ5` grows `7×→18×→68×→340×`. Low-multiplicity collapse
  with a widening gap — not a uniform slide.
- **Test B** (product of 12 Jacobians, each scaled to top-σ=1 to isolate *shape*): the **sedenion** product
  keeps a discrete three-tier spectrum `[0 ×4 | −1.8 ×8 | −14.3 ×4]` — **4 dead modes, gap `3.2×10¹²` at
  rank 12→13**: the `4/8/4` structure *survives composition*. The **real Gaussian** control slides smoothly
  (`−0.3, −0.6, … −10.6`), gap only `30×` inside a continuous decay. Distinct signatures.

## The discriminant (and the prior art to cite/differentiate)
| signature | reading |
|---|---|
| whole spectrum slides uniformly | magnitude — classical vanishing gradient; residuals fix it |
| **few σ collapse, gap, bulk intact** | **structural annihilation (𝕊)** — residuals preserve norm and do nothing; invisible to dynamical isometry (which asks if the *whole* spectrum is near 1) |
| whole representation → rank 1 | rank collapse (Dong et al., "attention loses rank doubly exponentially") — the nearest neighbor, but *different*: whole rep degenerates, not a small subspace dying with a healthy bulk |

## Honest scope — what is shown, and the cheap next test
Shown: the signature **exists and is distinctive** for sedenion-structured multiplicative composition, and
it **survives depth**, differing measurably from both classical vanishing gradient (uniform) and rank
collapse (whole→rank-1). It passes the multiplicative-composition filter. **Not yet shown:** that any *real
trained network* exhibits it. That is the critique's actual test, and it is the cheapest here —
**no training required**: take existing checkpoints, compute the spectrum of the Jacobian product along a
few input paths, and look for a low-multiplicity collapse *with a gap* versus a uniform slide. If a gap is
there, it is a failure mode the dynamical-isometry literature does not measure; if it is a uniform slide,
the structure is not there and one afternoon of rented GPU says so. Harness `spectral_signature.py`.
