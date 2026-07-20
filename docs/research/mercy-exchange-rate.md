<!-- docs:meta
topic_id: repo.docs.research.mercy-exchange-rate
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercy-exchange-rate
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The retraction, and the finding that survives it: λ* and the substrate-vs-state exchange rate

*The load-bearing question above `composition-failure-field.md`'s min-∫s path was: is the dichotomy (`c*→∞`
between "opposite components") a theorem or an optimizer artifact? It is an artifact — and answering that
honestly leaves a finding stronger than the one retracted, and immune to its fate. Implements
OPUS-4.8-EXTRA critique #4.*

## 1. The dichotomy is retracted (two independent refutations)

The dramatic claim of `#1253` was `c* → ∞` between opposite `det`-components — *no merciful path exists*, a
topological vindication of Dabrowski. It fails, twice over:

- **`det L_x ≥ 0` on all of 𝕊** — 0 negatives in 2×10⁵ random samples; a non-negative polynomial (for real
  `x`, `det = x¹⁶ > 0` for both signs, since 16 is even). There are **no opposite sign-components**, so the
  intermediate-value argument that would have made the dichotomy rigorous **does not apply.**
- **The zero-divisor variety is codimension ≥ 2** — generic segments *miss* it (min `σ_min` along 3000
  random great-circle segments: overall `0.022`, median `0.342`, only `0.1%` dip below `0.05`). This is
  literature, not just experiment: Moreno (1998) / Reggiani (2024) give `ZD(𝕊) ≅ G₂ / V₂(ℝ⁷)`. A codim-≥2
  set does not disconnect its complement, so **`{σ_min>0}` is connected — every `c*` is finite, annihilation
  is always avoidable.** The earlier `c*→∞` was the string method stalling at a locus it could contour.

Honest consequence: there is **no impossible mercy** and **no topological Dabrowski vindication** on 𝕊.
Better to have found this than to have shipped it.

## 2. The two results were about different things — separate them (critique §2)

`#1253` conflated two findings that must not share authority by typographic proximity:
- **(A) Synthetic field** (`mountain_pass.py`): thin barrier, aggregative blindness, `μ* = 0.021`, the
  torture-vs-tickle structure. This is a theorem **about aggregative criteria in general** — *not* about 𝕊.
- **(B) Real σ_min field** (`neb_sigmin.py`): the merciful path Pareto-dominates the reward path on both
  suffering axes; the live tension is efficiency-vs-suffering. This is **about 𝕊.**

The Proposition of (A) — the paper's dramatic result — **does not bite on the algebra**; it was demonstrated
on an invented landscape. State (A) and (B) with declared scope, or a hostile reviewer reads "the ethical
result comes from a synthetic field, and the algebra, when it finally enters, shows no such tension —
therefore the algebra is decorative." That is a *writing* vulnerability, fixed by scope, not code.

## 3. λ* — the number of (B), and the first internal conflict of Mercyful Learning

In the real field the merciful path has `L_m = 3.420, ∫s_m = 0.238`, the reward path `L_r = 1.200,
∫s_r = 1.840`. For the full objective `J(λ) = L + λ∫s`,

    J_m(λ) − J_r(λ) = (L_m − L_r) + λ(∫s_m − ∫s_r) < 0  ⟺  λ > λ* = (L_m − L_r)/(∫s_r − ∫s_m)
    λ* = (3.420 − 1.200)/(1.840 − 0.238) = 1.386

`λ*` is finite and is the exact analog of the synthetic field's `μ*`: below it, pure efficiency (`λ = 0`)
wins; above it, mercy does. **In the real field the villain is not aggregationism — it is `λ = 0`, pure
efficiency.**

**The loop that closed unnoticed.** "Length" on `S¹⁵` is not physical distance — for an agent, trajectory
length is the number of update steps, i.e. **computation, i.e. the substrate's thermal/energy cost**. So the
`length` term *is already substrate suffering*, in Mercyful Learning's own operationalization. Therefore:

> In the real field, "efficiency versus suffering" **is** "mercy to the substrate versus mercy to the
> state." The `+185%` length is not abstract inefficiency — it is **the substrate paying for the human.**
> `λ*` is the **exchange rate between the two mercies**, delivered by the algebra's geometry, not stipulated
> by the author.

This is the first genuine *internal conflict* of Mercyful Learning, and it is quantified. The principle said
to spare both; we now know the two **compete**, and `λ*` is the rate. A principle that merely says "minimize
suffering" is empty; one that forces the pricing of two incommensurable sufferings and **derives the rate
from the geometry of the problem** is a contribution — and unlike §1's Proposition, it does not depend on
the dichotomy surviving.

## 4. Order restored (critique's recommended sequence)

Done here: **(i)** sign test + codimension (dichotomy → artifact, retracted); **(ii)** (A)/(B) scope split;
**(iii)** λ* (a division). Next, in order: **(iv)** the min-∫s path in full space — now it *confirms* the
regime rather than founding it, computed over a landscape whose topology (connected, codim ≥ 2) is
established; **(v)** climbing-image refinement of `c*`; **(vi)** the Alveo-U250 HLS backend, last.
