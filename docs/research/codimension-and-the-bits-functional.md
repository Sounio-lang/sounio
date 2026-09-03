<!-- docs:meta
topic_id: repo.docs.research.codimension-and-the-bits-functional
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.codimension-and-the-bits-functional
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Codimension 4, the bits-destroyed functional, and the pre-registration of the min-∫s test

*Before running (iv) — the min-∫s path in full 𝕊 — three things must be settled or the result is not
interpretable: the exact codimension (which fixes the regime), the justification of the functional form
(which is what (iv) is secretly a property of), and the pre-registered prediction. Implements
OPUS-4.8-EXTRA critique #5, §1–§3, §5.*

## 1. Codimension = 4 (the square test, resolved)

The earlier segment-sampling did **not** discriminate codim-≥2 (a filament) from codim-1-non-separating
(`det L_x` a perfect square — a wall that doesn't disconnect because the sign never changes). Both let a
generic segment miss the locus. The discriminator is the **Hessian of `σ_min²` at a zero divisor `z`**,
whose rank is the codimension. At `z = e₁+e₁₀`:

    eigenvalues of Hess(σ_min²) = [4, 4, 2, 2, 0, 0, …]   →   rank = 4   →   codim = 4

So `det L_x` is **not** a perfect square (that would give rank 1); it is a **sum of (≥4) squares**, and the
zero-divisor variety is a **codimension-4 filament**, not a wall. `det ≥ 0` (observed) and high codimension
now *imply each other* correctly. The complement `{σ_min>0}` is connected; the retraction stands on a
computed number, not an inequality. Consistency bonus: **codim 4 = the annihilator dimension = the Biss–
Christensen–Dugger–Isaksen bound `2ⁿ−4n+4 = 4`** for 𝕊 — the same 4-of-16 that appears in
`relational-annihilation-geometry.md`. The regime is settled: *filament*, so contouring is cheap, and no
utilitarian-vs-Rawlsian divergence is forced by the topology (it must come, if at all, from the field).

## 2. The functional is derived, not chosen — `−log σ_min` = bits destroyed

The choice between `s = −log σ_min` and `s = 1/σ_min` is not cosmetic — it decides (iv). Near the locus, at
transverse distance `t`:

    ∫₀^ε (−log t) dt = ε(1 − log ε) < ∞      vs.      ∫₀^ε dt/t = ∞

- with `−log σ_min`: crossing annihilation costs **finite ∫s** and **infinite max s** — the two criteria
  diverge *maximally*;
- with `1/σ_min`: crossing costs infinity in both — they **agree**, and there is no divergence to find.

The principled tie-breaker (which closes an old thread): `−log σ_min(L_x)` is the number of **bits of
precision destroyed** by multiplication by `x`. `σ_min` is the worst-case contraction factor of `L_x`; its
negative log counts the digits lost. This is not analogy — it *is* the substrate-suffering operationalization
already adopted ("physical strain: thermal / **error** / energy"), and it is literally what exact
Cayley–Dickson arithmetic on the Alveo U250 exists to prevent. Under it everything aligns: `s = −log σ_min`
= bits destroyed per step; `∫s` = total bits destroyed (finite even through the locus, because the passage
is instantaneous); `max s` = worst instantaneous loss (infinite at exact annihilation). So the functional
form is **derived** — and it forces the divergent regime, not by taste but by the meaning of the field.

## 3. Pre-registration of (iv)

With the topology settled (connected, codim 4) and the field justified, (iv) is a **test**, not
exploration. The prediction the framework obliges:

> The **min-∫s** path should *hug the locus* — approach it to the resolution limit, with `∫s` **convergent**
> under mesh/segment refinement and `max s` **growing without bound** (linearly in the log of the
> resolution). The **min-max-s** path should contour it. The signature of genuine divergence is
> **`max s → ∞` with `∫s` stable**.

Falsifiers: if refinement shows `∫s` *also* growing, or `max s` *stabilizing*, then either the codimension
is wrong (it is not — §1) or the numerics failed. **The specific failure mode to police:** `σ_min` is
**not smooth** where singular values coalesce (spectral degeneracy), and the optimizer will be pushed
exactly there. Mitigation, declared in advance: use a **smoothed surrogate** — a soft-min over the `σᵢ`,
`s_β(x) = (1/β)·log Σ_i e^{−β σ_i}` (→ σ_min as β→∞) — or a declared **subgradient**, and report which.

## 4. λ* is endpoint-dependent — the 2-bit reading was coincidence (footnote, per §5)

`λ* = 1.386` nats `= 2.000` bits for the original endpoints was arithmetically striking, so it was tested:
recomputing across shifted endpoints gives `λ* ≈ 1.8, 2.1, 2.1, 3.0` bits. **`λ*` varies with the
endpoints; it is not pegged at 2 bits.** The exact `2.000` was coincidence. What survives is *structural*
and weaker-but-real: `λ*` is finite and positive on the real field — an exchange rate exists between
substrate-mercy and state-mercy — but its *value* is a property of the state pair, not a universal constant.
Stated as a footnote, not a headline.

## 5. What the retraction bought (critique §6)

Worth recording, because it is stronger and more uncomfortable than the theorem it replaced. Connected
complement + every `c*` finite means:

> In 𝕊 no transition **requires** annihilation. Every rupture is contournable at finite cost. The
> "necessary suffering" of the mountain-pass definition does **not** exist as a geometric obstruction — it
> reappears entirely as a **budget constraint**. What forces rupture is not the structure of the space but
> the cost of the detour: length = computation = substrate.

Suffering ceases to be destiny and becomes **purchase**: nothing is geometrically necessary, so everything
that happens was *acquired at some rate* — precisely `λ*`. The retraction did not weaken the program; it
moved the weight from a topological theorem that did not exist to an exchange rate that does. Next: run (iv)
with the soft-min surrogate declared (§3), then the climbing-image `c*`, then — last — the U250 backend.
