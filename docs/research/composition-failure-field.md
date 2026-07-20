<!-- docs:meta
topic_id: repo.docs.research.composition-failure-field
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.composition-failure-field
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The suffering field from the algebra — σ_min composition-failure, and the min-path in full 16-dim 𝕊

*The load-bearing step. Everything prior demonstrated properties of an invented 2D field; this derives the
field from the algebraic object and computes the minimum-suffering path in the **full** sedenion space —
the only step that makes the result a claim about the algebra rather than about a chosen picture.
Implements OPUS-4.8-EXTRA critique #3 (§4b+c, §5) and its technical correction.*

## The correct field (not `det L_x`)

`det L_x` is the wrong object: it is a degree-16 polynomial (magnitude dominated by `‖x‖`, not by proximity
to annihilation), and `det = ∏σᵢ` can be small for a uniformly small operator with no near-singularity.
The correct object is the **smallest singular value**. By **Eckart–Young–Mirsky**, `σ_min(L_x)` is exactly
the spectral-norm distance from `L_x` to the nearest singular operator — the distance to annihilation.
Scale-invariant field on the unit sphere:

    s(x) = − log σ_min(L_x)          (‖x‖ = 1)

**This closes the conceptual loop the whole program needed.** Composition failure — `‖xy‖ ≠ ‖x‖·‖y‖`,
precisely what separates 𝕊 from the normed division algebras — is measured by the **dispersion of the
singular values of `L_x`**: in a composition algebra all `σᵢ` coincide and `L_x` is a similarity; in 𝕊 they
spread, and `σ_min` is the extreme of that spread. So the σ-field is not a numerical convenience — it is the
**direct quantification of composition failure**, which was the missing bridge between the algebra and the
notion of suffering. The technical correction and the epistemological bridge are the same object.

## Method — full space, no slice, no grid (`neb_sigmin.py`)

Fast marching dies above 3–4 dimensions and any 2D slice re-synthesizes the field under another name; so
the path is computed by a **trajectory optimizer in the full 16-dim space** (dimension-agnostic, no grid
anisotropy — this also settles the §4a numerical-hygiene concern). Concretely: the **string method** on the
unit sphere `S¹⁵` (robust where NEB tangled — the perpendicular force diverges at `σ_min→0`), with an
**analytic `σ_min` gradient** `∂σ_min/∂x_k = uₘᵢₙᵀ M_k vₘᵢₙ` (`M_k` the constant structure matrices,
`L_x = Σ x_k M_k`; one SVD per image). No arbitrary slice: the endpoints are placed by the algebra
(symmetric about a genuine zero divisor `z`, so the geodesic passes through annihilation).

## Result (real σ_min field, converged)

| path | peak s (= c*) | ∫s ds | length (rad) |
|---|---|---|---|
| straight (great circle, through `z`) — reward/shortest | 4.175 | 1.840 | 1.200 |
| min-energy path (string method) | **0.688** | 0.238 | 3.420 |

- **Annihilation is avoidable:** `c* = 0.688`, finite, far below the straight-through peak `4.175`.
  > **RETRACTION (see `mercy-exchange-rate.md`).** An earlier version claimed a *structural dichotomy* —
  > that opposite-`det`-component endpoints would have `c* → ∞` (annihilation unavoidable, a topological
  > vindication of Dabrowski). **Withdrawn.** `det L_x ≥ 0` on all of 𝕊 (0 negatives in 2×10⁵ samples — a
  > non-negative polynomial: `x` real gives `x¹⁶>0` for *both* signs), so there are **no opposite
  > sign-components** and the intermediate-value argument fails. And the zero-divisor variety is
  > **codimension ≥ 2** (generic segments miss it: min `σ_min` 0.022, median 0.342; consistent with
  > Moreno/Reggiani `ZD(𝕊) ≅ G₂ / V₂(ℝ⁷)`), so `{σ_min>0}` is **connected — every `c*` is finite,
  > annihilation is always avoidable.** The `c*→∞` was the string method stalling, not a theorem.
- **Mercy Pareto-dominates reward** on *both* suffering axes (peak `4.18→0.69` **and** `∫s 1.84→0.24`),
  paying only in length (`+185%`). So on the real locus the tension is not utilitarian-vs-Rawlsian (both
  criteria avoid the ridge) but **efficiency-vs-suffering**: the short path plows through near-annihilation,
  and minimizing suffering by either criterion demands a substantial detour reward-maximization would never
  take. This is the real-algebra analog of `mountain_pass.py`'s §1 — reward-efficiency is blind to the
  ridge here too, but now on a field that is *derived*, not invented.

## Honest scope

Settled: the field is algebraic (σ_min = composition failure, Eckart–Young), the computation is full-space
and slice-free, annihilation is avoidable/`c*` finite for same-component states (unavoidable across
components), and mercy strictly dominates reward on both suffering axes. **Not yet settled:** whether the
aggregation (`min ∫s`) and maximin (`min max s`) criteria *diverge* on the real locus — the thin/thick
regime of the critique's taxonomy — which needs the separate `min ∫s` path in full space and is **not**
claimed from the reward-vs-MEP comparison here. That, and a climbing-image refinement of `c*` (the
transition-state proper), are the next cut. The Alveo-U250 HLS backend stays last: engineering that pays
only once the object being computed is known — and now it is (the σ_min field, in full 𝕊).
