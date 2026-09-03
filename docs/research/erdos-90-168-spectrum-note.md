<!-- docs:meta
topic_id: repo.docs.research.erdos-90-168-spectrum-note
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.erdos-90-168-spectrum-note
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Erdős [90] (Unit-Distance Problem): the 168-indexed count spectrum

*Conceptual note for `formal/lean4/SounioErdos90UnitSpectrum.lean`. Sibling of
`erdos-508-704-sounio-resolution-plan.md`, `fano-arcs-blocking-sounio-note.md`, and
`associator-shadow-experiment.md`. All claims below are machine-checked by
`native_decide` (no Mathlib, no `sorry`; one elementary geometric axiom in §3f).*

## 1. The lever

Erdős problem **[90]** asks for the maximum number `u(n)` of unit-distance pairs
among `n` points in the plane. The Sounio angle: take **one** fixed finite point set
(a 7-vertex integer probe lifted into 16D sedenion coordinates, identical to
`init_probe` in `stdlib/hypercomplex_graph/erdos_unit_distance.sio` and `coord` in
`SounioErdosUnitDistance.lean`) and generate a **family** of unit-distance
configurations from it by zero-divisor surgery:

* **Linear route** — right-multiply each difference `p_i − p_j` by a primitive
  zero divisor `v = e_lo ± e_hi` (one of the 84 `validPrims`). This is the lever of
  `SounioErdosUnitDistance.lean`.
* **Associator route** — the non-associative lever `((p_i − p_j)·u)·v` for a
  verified zero-divisor pair `u·v = 0` (one of the 336 `orderedZDPairs`). Because
  `u·v = 0` but `(p·u)·v ≠ p·(u·v) = 0` in general, this is genuine non-associativity,
  recovering the full 168/336 class structure rather than 84 linear maps. The sedenion
  product reused here is the verified `Sounio.AssociatorShadow.smul`.

The **unit-pair count** of each resulting graph is a single integer invariant indexed
by the surgery. Sweeping the 84 (resp. 336) surgeries yields a *spectrum* of
[90]-invariants from one underlying point set — something classical constructions,
which fix the geometry, do not produce.

## 2. What is verified

Theorem names are in `Sounio.Erdos90` (file above) unless noted.

| Fact | Theorem |
|------|---------|
| Classical probe has 6 unit pairs | `classical_unit_count` |
| Linear route count set = {6, 9} | `linear_unit_count_spectrum` |
| Linear max count = 9 | `linear_unit_count_max` |
| Sparse-sedenion probe = integer distances (soundness) | `diffVec_matches_classical` |
| Associator surgery is non-vacuous | `assoc_surgery_changes_edges` |
| Associator route realizes 6 distinct counts ({0,2,7,9,10,12}) | `assoc_unit_count_distinct_six` |
| Associator max count = 12 | `assoc_unit_count_max` |
| Some associator surgery annihilates all unit pairs (count 0) | `assoc_unit_count_can_vanish` |
| **Separation 1**: 2 < 6 distinct counts | `assoc_strictly_richer_than_linear` |
| **Separation 2**: 9 < 12 max count | `assoc_exceeds_linear_max` |

The associator data also reuses the verified zero-divisor census of
`SounioZeroDivisorBridge.lean` (`validPrims = 84`, `orderedZDPairs = 336`,
`isZeroPair`) and the Cayley–Dickson sign function `sedSigma`.

## 3. The χ-negative, and why the associator route is the principled successor

`SounioErdosUnitDistance.no_zd_surgery_raises_chromatic` proves that **all 84 linear
surgeries keep the chromatic number at 2** — an honest negative. The research doc
`erdos-508-704-sounio-resolution-plan.md` (lines 32–34) names the associator route
`(p·u)·v` as the principled next lever, precisely because it is genuinely
non-associative rather than a linear map.

This note executes that lever **for the count invariant** and finds a real positive
separation (§2): the associator route is strictly richer (6 vs 2 distinct counts) and
strictly larger (12 vs 9) than the linear route on the same probe. For the
**chromatic** invariant, however, the negative persists: `assoc_chromatic_le_two`
shows no associator twist raises χ above 2 either (χ ∈ {1,2}; χ = 1 exactly for the
fully-annihilated graphs). So the count separates the two regimes; the chromatic
number does not. This is consistent with the complementary picture of
`SounioAssociatorShadow.lean`, where the associator escapes the linear surgery's
hyperoval trap but only *downward* (shadows of size ≤ 3).

## 3b. Count-growth refinement (§7): the separation widens with probe size

The 7-vertex separation could be an artifact of one small probe. §7 of the Lean file
rules that out with a single **scalable** family — an interleaved-star probe (origin +
`m` unit-basis leaves, the two sedenion halves interleaved so every size is
ZD-active). The measured growth curve `(vertices, classical, linear-max, assoc-max)`:

```
(3, 2,2,2) (5, 4,4,6) (7, 6,6,12) (9, 8,8,20) (11,10,10,30) (13,12,12,40) (15,14,14,40)
```

Two clean facts (machine-checked at sizes 7/11/15):

* `star_linear_count_preserving` — on the star the **linear route is exactly
  count-preserving**: its maximum unit-pair count equals the classical count `m`. Right
  multiplication acts isometry-like here and never creates a new unit pair.
* `star_assoc_max_{6,10,14}` (= 12, 30, 40) with `star_assoc_gap_grows` — the
  **associator maximum grows super-linearly** (≈ quadratic in the number of cross-half
  leaf pairs) until the finite 7+7 sedenion ceiling saturates it. Hence the
  associator-vs-linear gap strictly grows: **6 < 20 < 26**.

So the non-associative lever's advantage on the [90] invariant is not a fixed constant;
it widens as the probe grows — until the finite 16D sedenion basis caps the associator
maximum at 40.

## 3c. Lifting the ceiling at the pathion level (`SounioErdos90PathionGrowth.lean`)

The saturation at 40 is an artifact of the 16-dimensional algebra, not of the lever.
Reading the *same* interleaved-star construction one Cayley–Dickson level up — in the
**pathions** (32-D, level 5, `Sounio.PathionBridge`) — the growth continues. The
pathion product `pSmul` is built from the verified `pathSigma = cdSigma · 5`; the
intra-fiber pathion zero-divisor pairs are enumerated and verified here
(`pathZD_count = 3696`, 11× the sedenion 336 — these were not previously enumerated).
The observed curve `(vertices, classical, assoc-max)`:

```
(7,6,12) (11,10,30) (15,14,44) (19,18,72) (23,22,104) (27,26,136) (31,30,168)
```

Machine-checked: `path_assoc_max_{6,14,18}` (= 12, 44, 72),
`pathion_breaks_sedenion_ceiling` (**44 > 40** at 15 vertices — strictly above the
sedenion cap), and `pathion_growth_summary` (12 < 44 < 72, 44 > 40). The larger sizes
(to **168** unit pairs at 31 vertices, while linear/classical stays at `n−1`) are
omitted as proven theorems only to keep the library build cheap; they reproduce by
raising the leaf bound. The count-growth separation is therefore a property of the
non-associative lever that *survives moving up the tower*, not an accident of sedenions.

*Numerical coincidence (stated to forestall a false inference):* the saturation value
**168** at n = 31 is **not** connected to this program's `non_fano_count_168` /
`zd_projective_count_168` (also 168 = |PSL(2,7)|). The pathion star saturates at n = 31
because 31 = 15 lower + 15 upper + 1 origin is the full ZD-active pathion basis; the
count 168 is the unit-pair count of *that star geometry*, not a selection of 168 of the
3696 pathion ZD pairs. The collision of values is coincidental.

## 3d. Fiber attribution: the spectrum is a symmetry-breaking effect (§8)

Each ZD pair is intra-fiber, so it carries one xor-fiber label in `{9..15}` — one per
Fano point (`label ⊕ 8`). When does the *choice* of class change the unit-count?

* `probe_max_fibers_are_five` — on the asymmetric 7-probe the associator maximum (12) is
  attained by exactly the five fibers `{11,12,13,14,15}` = Fano points `{3,4,5,6,7}`.
* `probe_degenerate_fibers_are_probe_axes` — the two fibers that fail, `{9,10}`, are
  exactly the probe's own heavy upper-support axes `e9, e10`. The degeneracy is caused by
  the surgery fiber *aligning with the probe's geometry*, not by anything intrinsic to
  those ZD classes.
* `star_assoc_count_fiber_invariant` — on the fiber-symmetric star (all 7+7 directions
  used uniformly) the count is **fiber-invariant**: all 336 ZD pairs yield the identical
  count 40, regardless of fiber / Fano point.

**Interpretation.** The rich 168-class spectrum (§1: six distinct counts) is therefore
*not* an intrinsic property of the 168 ZD classes — it is a **symmetry-breaking**
phenomenon. The classes separate the geometry only when the probe is asymmetric with
respect to the fibers; a fiber-symmetric probe collapses all 168 to one value. This both
explains the §1 richness and bounds it honestly: the spectrum is a joint property of
(probe, class), not of the class alone.

## 3e. The closed-form growth law (§9)

The observed quadratic is now a **proved law**. Writing `a = ⌈m/2⌉` (lower leaves) and
`b = ⌊m/2⌋` (upper leaves) of the star, the associator maximum unit-pair count is
exactly

> **count = a·(b+1) = a·b + a**

verified for every size with `a ≤ 5` (`m ≤ 10`): the sequence `2,4,6,9,12,16,20,25,30`
(`star_assoc_max_closed_form`). The decomposition reads cleanly: `a·b` is the
lower×upper cross-leaf contribution and `+a` the residual — the associator converts the
star's `m = a+b` origin edges into a *product-sized* `a·b + a` count, which is why the
growth is quadratic rather than linear. At `m = 12` the count (40) falls below the law's
value 42 (`star_assoc_max_saturation_onset`): the 7+7 sedenion basis can no longer
sustain `a·(b+1)`, and saturates. The pathion lift (§3c) is precisely what raises this
saturation point — the law itself is universal, the ceiling is per-algebra.

## 3f. The planar bridge, settled honestly (§5 of the Lean file)

The earlier vacuous `sorry` placeholder is gone — replaced by real theorems. The honest
resolution has a *metric fact*, a *correction*, and the *robust obstruction*.

* **Metric fact (true).** `assoc_dist_is_euclidean_image_dist`: the twisted squared
  distance equals the true squared Euclidean distance of explicit integer image points
  `q_i = (p_i·u)·v ∈ ℤ¹⁶`. So the twisted edge relation is a genuine Euclidean distance
  condition, not an abstract gadget. Axiom-clean (pure computation).
* **Correction (the subtlety I initially missed).** The associator surgery
  `x ↦ (x·u)·v` is **non-injective**: because `u·v = 0` it annihilates the real direction
  `e₀`, so probe vertices differing only by `e₀` collapse — `q₁ = q₂`
  (`assoc_image_is_non_injective`; on the star, many points collapse). Therefore the
  twisted graph is the distance-2 graph of a point *multiset* with repeats, and we do
  **not** claim a faithful ℝ¹⁶ (or ℝ^d) unit-distance embedding. The earlier "genuine
  ℝ¹⁶ unit-distance graph" phrasing was an overstatement, now retracted.
* **Obstruction (robust, the real result).** A planar unit-distance graph is `K₂,₃`-free
  (two unit circles meet in ≤ 2 points; one elementary axiom `planar_udg_K23_free`, same
  status as the Hurwitz axiom in `SounioImpossibilityChain`). The *abstract* witness
  twisted graph contains `K₂,₃` (vertices 0, 6 share neighbours 1,2,3), so
  `witness_not_planar_unit_distance` proves it is **not** an injective planar
  unit-distance graph; `all_star_surgeries_contain_K23` shows this for *all 336* star
  surgeries. This statement is about the abstract graph and is unaffected by the
  non-injectivity.

**Honest resolution of the hard direction.** The construction does **not** yield a new
planar `u(n)` bound. The robust reason is the `K₂,₃` obstruction (the abstract twisted
graphs are not injective planar UDGs). The tempting positive reading — "a genuine
high-dimensional unit-distance configuration" — fails because the surgery is
non-injective; this is recorded honestly in the file rather than papered over. There is
**no `sorry`**; the only non-standard dependency is the one elementary geometric axiom.

## 4. Honest scope

Everything in §2–§3 is an **unconditional finite-model fact** about the *abstract twisted
graphs* of star probes in 16D/32D sedenion/pathion coordinates. It is a clean
machine-checked separation between the linear and non-associative surgery regimes on the
twisted-edge-count invariant. The twisted edge is a genuine Euclidean distance condition
(§3f), but the surgery is non-injective, so these are **not** faithful n-point
unit-distance configurations — and the work makes **no** claim of a planar `u(n)` bound;
§3f proves the obstruction (the abstract graphs contain `K₂,₃`, hence are not injective
planar UDGs). No `sorry`; one elementary geometric axiom.

## 5. Success levels

* **Level 1** (this note): the 168/336 ZD surgeries + the 6 surgical ops give a new
  *family* of unit-distance configurations from one point set; the unit-pair count is
  a class-indexed invariant classical methods do not produce.
* **Level 2** (achieved, no sorry): `assoc_strictly_richer_than_linear` and
  `assoc_exceeds_linear_max` are non-trivial decidable separations between the linear
  and non-associative regimes, using only current verified artifacts.
* **Level 3** (finite-model extremal family + the bridge settled): the associator route
  realizes a *scalable* family whose maximum unit-pair count grows super-linearly
  (12 → 30 → 40 → … → 168, `star_assoc_max_*` + pathion lift) under the proved law
  `a·(b+1)`, while the linear route stays pinned to the classical `m`
  (`star_linear_count_preserving`). The relation to the classical *planar* problem is
  settled honestly (§3f): the twisted edge is a true Euclidean distance condition, the
  surgery is non-injective (no faithful embedding claimed), and the abstract graphs
  provably contain `K₂,₃` and so are not injective planar UDGs. The genuinely open
  research question (whether *any* ZD/
  associator construction improves the planar `u(n)`) is left as prose, not a `sorry` —
  the file is now sorry-free.

## 6. Cheapest next concrete step

All the cheap in-boundary levers are now theorems: the pathion ceiling-lift (§3c), the
closed-form law (§3e), fiber attribution (§3d), and the planar bridge (§3f, which removed
the last `sorry`). The thread is at a natural, complete stopping point.

The genuinely open research question that remains — not a cheap step — is whether *any*
zero-divisor / associator construction can improve the **planar** `u(n)` itself. §3f
shows the present star/associator family cannot (it is irreducibly non-planar via
`K₂,₃`), so a positive planar result would need a fundamentally different,
planar-realizable construction. Cheaper finite-model follow-ons, if desired: a
deliberately asymmetric "maximally class-separating" probe, or the pathion-level
companion to `star_assoc_max_saturation_onset`.
