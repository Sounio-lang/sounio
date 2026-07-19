<!-- docs:meta
topic_id: repo.docs.research.petitot-semantic-potential
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.petitot-semantic-potential
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Petitot's semantic potential, and its octonion counterpart — where the two formalizations of rupture meet and diverge

*Follow-on to `rupture-as-singularity.md`. Pulls the next flagged thread: build Petitot's semantic
potential explicitly and test the correspondence to the Cayley–Dickson / Fano structure. The honest
result is a **divergence** that is more useful than a forced match — two formalizations of one intuition
that locate the obstruction in different places.*

## 1. Petitot's potential, reproduced (`petitot_potential.py`)

Petitot realizes the Greimas semiotic square as the critical-point structure of a potential: semantic
positions = wells (attractors); oppositions = the bifurcation set; the semantic jump = crossing it.

- **Cusp** `V = x⁴/4 + a·x²/2 + b·x` → **binary contrariety**: two wells (the contraries A, B) over a
  bistable region (~19% of the control plane), bounded by the fold curve `4a³+27b²=0` (the bifurcation
  set — the semantic jump).
- **Butterfly** `V = x⁶/6 + t·x⁴/4 + v·x²/2 + w·x` → the **mediating "complex/neutral term"**: up to
  **three** coexisting wells; the 3-well "pocket" (~16% of the scanned slice) is Petitot's complex term,
  and it is bounded by butterfly cusp lines, **not** a Boolean corner.

**The impossibility theorem, illustrated.** The square's two opposition-types are topologically distinct
moves: *contrariety* A|B = two wells that can **both vanish** (merge over the cusp, `2→1` well — "both
false" is possible); *contradiction* A|¬A = **antipodal** (one appears exactly as the other vanishes —
"neither" is impossible). A Boolean lattice `2²` has a **single** complement, so it cannot host two
distinct opposition-types: the square is **not Booleanizable**, and the distinction is carried by the
*topology of the strata*, not by logic.

## 2. The octonion / Fano model of a *system* of oppositions

Model each opposition as a sign-structure that closes into a subalgebra, and ask how oppositions combine:

- The **7 Fano lines** (associative triples `{i,j,i⊕j}`) are the **7 quaternion subalgebras** of 𝕆 — each
  a Booleanizable "square" (a ℤ/2×ℤ/2 opposition that closes associatively).
- **Within** one line the associator is **0** (associative ⇒ Booleanizable). **Across** lines the
  associator is **2.0** (`‖[e_i,e_j,e_k]‖²=4`, the non-Fano value) — the system of squares is **not
  globally Booleanizable**, and the obstruction *is the associator*. Two Fano lines meet in exactly one
  unit (the Fano incidence) — oppositions share terms but do not fuse.

## 3. The divergence (the honest finding)

The two models locate the obstruction in **different places**:

| | single opposition (one square) | system of oppositions (the field) |
|---|---|---|
| **Petitot (catastrophe)** | already **not** Booleanizable (needs the strata topology) | — |
| **Octonion (algebraic)** | a **quaternion**: associative, **Booleanizable** | the **octonion**: non-associative, **not** Booleanizable — obstruction = associator |

In the octonion model the single square *is* Booleanizable — which **contradicts Petitot's claim about the
lone square**. The algebraic non-Booleanizability appears only at the level of the **field** of
oppositions. We do not paper over this. It suggests a genuine **reconciliation hypothesis**: the single
semantic square is *local* (a quaternion — meaning that composes fine in isolation), and the topology
Petitot needs for the lone square is the **shadow of the ambient non-associativity** — no real semantic
square is isolated; it sits in a field (𝕆) whose other squares will not combine associatively with it, and
that ambient obstruction is what forces the extra topological structure back onto the single square. That
is a testable claim: *Petitot's non-Booleanizability of the isolated square should be derivable as the
restriction of the octonion associator to a neighborhood of one Fano line.* Unproven here; stated
precisely.

## 4. The deep conjecture (interpretive — flagged)

The synthesis noted Wildgen's four-actant semantics already **requires the exceptional singularities
E₆, E₇, E₈** (and X₉), and that the octonions generate the exceptional Lie groups (the Freudenthal–Tits
magic square G₂→F₄→E₆→E₇→E₈). These share the **E₆/E₇/E₈ Dynkin/ADE labels** — Arnold's exceptional
*singularities* and the exceptional *Lie algebras* are linked by the McKay correspondence. So the
conjecture the whole thread points at: **the organizing center of rich (four-actant) semantic morphology
is the octonionic exceptional structure** — Petitot/Wildgen's singularity-theoretic semantics and the
Cayley–Dickson algebra are two faces of one exceptional geometry, the same G₂/E-series that governs the
associator (semantic rupture) and the zero-divisor locus (epistemic annihilation, `rupture-as-singularity`).
This is the interpretive frontier, not a result: the ADE label-match is real, the semantic realization is
conjectural, and the landmine (analogy mistaken for theorem — the "catastrophe controversy") is live.

## 5. Where this leaves the functor F

The functor F (𝕆-associator → homology detecting semantic rupture) now has a concrete substrate on both
sides: rupture as the **singular locus / bifurcation set** (`rupture-as-singularity.md`) and the semantic
square as **potential strata** (here). The next move for F is to make the "restriction of the associator to
a Fano-line neighborhood = Petitot's strata topology" claim precise — i.e. exhibit F on a single worked
opposition and check that its value reproduces the cusp/butterfly stratification. Harness
`petitot_potential.py`.
