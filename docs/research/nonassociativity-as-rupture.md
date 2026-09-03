<!-- docs:meta
topic_id: repo.docs.research.nonassociativity-as-rupture
authority: historical
audience: researchers
last_validated: 2026-08-17
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.nonassociativity-as-rupture
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

> **Re-author note (2026-08-17):** restored onto current `main` from PR #1237 tip `e2f66da28b36` so the parent literature map cited by `rupture-abcd-claims_2026-07-24.md` and `rupture-programme-synthesis_2026-07-25.md` exists in-tree again. Content is the original synthesis; no claim expansion.

> **Corrections, 2026-08-23.** Four mathematical errors survived the #1813 restoration and are corrected *in place* below. The lineage restoration itself is kept: current documents cite this parent, and a silent rewrite or a correction living elsewhere would not travel with that citation. (1) The associator is not the G₂ 3-form. (2) V₂(ℝ⁷) ≅ G₂/SU(2); the quotient G₂/V₂(ℝ⁷) is not a homogeneous space. (3) The space of zero-divisor *pairs* and the locus of *elements* are different manifolds. (4) “Statistical invisibility” is a named open hypothesis, not a result. Tensor-core numerical quotes in §9 read “within measured tolerance”, never “exact”.

# Non-associativity as the algebra of rupture — a research synthesis

*A literature map and program synthesis grounded in three parallel deep-research sweeps (categorical
semantics; morphodynamics & epistemic rupture; sedenion zero-divisors & annihilation), ~180 primary
sources. Its purpose: locate where the non-associative program (𝕆 as the algebra of meaning/rupture,
via the homology functor F; sedenion zero-divisors as the *open* Hypothesis H-annihilation, not a
measured statistical invisibility) sits in the real
literature, and what is genuinely unclaimed.*

---

## 0. The reframe

The program was never about winning ML benchmarks. Non-associativity is the **object**, not a feature,
and the object has a name the two research vectors share: **rupture** — the failure of composition to be
context-free. The associator `[a,b,c] = (ab)c − a(bc) ≠ 0` says precisely: *how you group changes the
result*. That is the rupture of **meaning** (composition that won't stay context-free) and the rupture of
**knowledge** (the break that won't compose linearly). The empirical experiments were never meant to
"beat" anything; they were meant to **instrument the structure**.

## 1. The spine: G₂ governs both faces

The unifying fact, hidden inside results that were mis-filed as "nulls":

- The octonion associator `[a,b,c] = (ab)c − a(bc)` is **alternating and Im 𝕆-valued** (verified:
  `[a,b,c] = −[b,a,c] = −[a,c,b]`, lands in Im 𝕆; Baez, *The Octonions*, 2002, arXiv math/0105155).
  **Correction, 2026-08-23:** it is *not* the G₂-invariant 3-form. That 3-form is the scalar
  `φ(a,b,c) = ⟨ab, c⟩` on Im 𝕆 ≅ ℝ⁷. They are different objects: a vector-valued map versus a
  3-form. The relation is a pairing, not an identity — `⟨[a,b,c], d⟩` reconstructs (a multiple of)
  the G₂-invariant 4-form; `φ` is recovered as `⟨ab, c⟩`.
- **Correction, 2026-08-23 — two manifolds, not one.** Moreno 1998 (arXiv q-alg/9710013) identifies
  the space of zero-divisor *pairs* with G₂. Reggiani 2024 (arXiv 2411.18881) identifies the
  *single-element* locus ZD(𝕊) with the Stiefel manifold V₂(ℝ⁷), which carries a G₂-invariant
  metric and **curvature**. Pair space and element locus have different dimensions and different
  geometry; they must not be collapsed. V₂(ℝ⁷) is a Stiefel manifold, not a subgroup of G₂, so the
  quotient “G₂/V₂(ℝ⁷)” is not a homogeneous space. The correct statement is **V₂(ℝ⁷) ≅ G₂/SU(2)**.

**The same exceptional geometry governs the two faces of rupture** — the associator (semantic) and the
zero divisors (epistemic). This is the hinge of the program, not a coincidence to report.

## 2. Face I — semantic rupture (the functor F / Tapestry)

**The orthodoxy is associative.** Categorical compositional distributional semantics (DisCoCat:
Coecke–Sadrzadeh–Clark 2010, arXiv 1003.4394) is built on compact-closed **monoidal** structure; its
associativity is load-bearing (Mac Lane coherence). Frobenius/DisCoCirc extensions add copying and flow,
never non-associativity.

**The failure of compositionality is real but was never made an algebra.** It appears as: non-associative
Lambek calculus (Lambek 1961 — syntax as binary trees, associativity deliberately false); the Generative
Lexicon (Pustejovsky 1995 — coercion, meaning-shift-in-context); idioms as *scalar* violation. Each
removes a rule or adds an operator; none builds a non-associative *semantic* algebra.

**Meaning-rupture IS formalized — as catastrophe (morphodynamics).** Thom (*Structural Stability and
Morphogenesis*, 1972) → Petitot (*Morphogenesis of Meaning*, 1985): actant = attractor, semantic
opposition = the bifurcation set, the semantic jump = crossing it; the semiotic square as a
threshold-bounded multi-well potential, with Petitot's **impossibility theorem** (the square cannot be
faithfully Booleanized — contrariety/contradiction are positional-topological, not logical). The decisive
precedent: **Wildgen (*Catastrophe Theoretic Semantics*, 1982) already pushes four-actant semantics into
the exceptional singularities E₆, E₇, E₈, X₉** — exceptional/non-classical structure already carrying
semantic load, the nearest ancestor to this thesis.

**Rupture as a provable obstruction.** Abramsky–Brandenburger (2011, arXiv 1102.0264) and
Abramsky–Sadrzadeh (2014, arXiv 1403.3351): contextuality / "meaning that won't glue" = a **non-vanishing
Čech cohomology class (H¹)** — consistent locally, impossible globally.

**The unclaimed move:** the **non-associative, homological completion of the Abramsky obstruction
program** — carry "meaning that won't glue" from a presheaf global-section obstruction (associative,
linear) to an **algebraic** one borne by the *associator* and detected by the homology functor **F**. The
four ingredients — **octonion / associator / meaning / homology** — have never been assembled. Sole
precedent to differentiate: Goertzel, *On the Algebraic Structure of Consciousness* (1996) — octonion
algebra of *consciousness*, not meaning, no homology; heterodox venue.

## 3. Face II — epistemic rupture (the Annihilation)

**The whole French tradition theorizes the break qualitatively — and the absence of a formalism is the
finding.** Bachelard (*rupture épistémologique*, 1938), Canguilhem, Althusser (*coupure*, 1965), Foucault
(*épistémè*, discontinuity) — no algebra, topology or dynamics of the break itself, by their own method.
Schroeder (arXiv 2402.16924) uses Bachelard's rupture in philosophy of information but never touches
hypercomplex algebra — so the link from *rupture épistémologique* to zero divisors is original here.

**Every existing formalization of the break is a NEGATIVE object:**
- **Missing morphism:** structuralist incommensurability = failure of a reduction relation
  ρ ⊆ Mp(T)×Mp(T′) to exist (Sneed 1971; Balzer–Moulines–Sneed 1987); categorical version = absence of an
  equivalence functor (Halvorson–Tsementzis; Barrett–Halvorson 2016).
- **Ungluable section:** Abramsky's H¹ obstruction (§2).
- **Metric blow-up:** information geometry hands rupture for free — **KL-divergence → ∞ under support
  mismatch, Fisher-metric singularities** at model boundaries (Amari; Watanabe's singular learning
  theory, with genuine free-energy phase transitions) — yet no philosophy-of-science work has claimed it.

**Non-associativity as the ceiling of knowledge composition (rigorous precedents):**
Günaydin–Piron–Ruegg (1978, Commun. Math. Phys. 61) — octonionic QM on H₃(𝕆) has **no tensor product, no
multi-particle states**: non-associativity *caps* the gluing of knowledge into a larger system. Jackiw
(1985) — the 3-cocycle: non-associativity = a cohomological obstruction to a globally consistent operator
frame. Szabo (*Nonassociative Physics*, arXiv 1903.05673) — non-geometric flux backgrounds where
coordinates fail to associate ⇒ no global position frame.

**Annihilation — the sharpest opening.** ℂ, ℍ, 𝕆 are **division algebras with no zero divisors**, so all
existing hypercomplex statistics is annihilation-free. Genuine "statistical invisibility by annihilation"
(`E[XY]=0` with X,Y a nonzero zero-divisor pair) is **Hypothesis H-annihilation (open)**, not a
result: the algebraic fact `XY = 0` does not imply a statistical claim. The hypothesis would need
a probability law, a measure, and an estimator before it can be a theorem; none of the three is
defined here. The construction is conceivable **only from 𝕊 upward — exactly where no probability
theory has been built**. The *pair* space and the *element* locus remain the two manifolds of §1
(pairs with G₂; elements ≅ V₂(ℝ⁷) ≅ G₂/SU(2), Reggiani); distributions *supported on or
transported across* either are well-posed as programmes and remain unstudied. This is the Annihilation-paper thesis with a rigorous geometric anchor. Combinatorial
vocabulary: de Marrais's box-kites (42 assessors / 84 pairs / 168 primitive units = |PSL(2,7)|,
XOR-indexed) — counts reliable (agree with Cawagas 2004), the PSL(2,7) organizing role asserted not
proved; load-bearing geometry from Moreno/Reggiani/Biss–Christensen–Dugger–Isaksen (annihilator
dim ≤ 2ⁿ−4n+4).

## 4. The decisive insight — the associator is the *positive, graded* rupture object

Across all three sweeps, every neighboring formalism models rupture with a **negative** object: *no* map
(Sneed/Halvorson), *no* global section (Abramsky), the metric *diverges* (Amari/Watanabe), the attractor
*vanishes* (Thom). None proposes a **positive, computable, graded** one.

**The associator is exactly that:** a single algebraic quantity that is **zero under associativity**
(the associative regime) and, at rupture, **nonzero — measuring the magnitude *and direction* of
the break.** **Correction, 2026-08-23:** the vanishing is associativity, not continuity; a continuous
non-associative product still has a nonzero associator. And the historical seam is exact: Hamilton **coined the word "associative" at the very moment
he saw the octonions fail it** (letter to Graves: "A·BC = AB·C … but not so, generally, with your
octaves"). Non-associativity was named at the rupture point.

So the program is the **positivization** of the rupture-as-obstruction tradition — replacing "the map that
isn't there" with "the associator that is," a graded invariant that can unify:
- the **semantic** side (Petitot/Wildgen morphodynamics, whose four-actant limit already demands
  exceptional singularities), via the homology functor F, and
- the **epistemic** side (incommensurability-as-failed-reassociation; Hypothesis H-annihilation),
under **one exceptional geometry, G₂**.

## 5. Re-reading the experiments (they were instrumentation, not contests)

- **#1230** (CD associator related to the G₂ 3-form by the pairing of §1, alternating, ⊥ Massey): not a
  failed bridge — the **hinge**. **Correction, 2026-08-23:** it does *not* identify the associator with
  the 3-form. It locates a pairing between the associator and the G₂ geometry that also governs the
  zero-divisor manifolds of §1, and distinguishes 𝕆 from higher-homotopy (Massey) as *distinct* modes
  of rupture (a fine, publishable distinction).
- **#1225** (Borromean/Massey): the topological face of "structure that appears exactly where primary
  products annihilate" — invisible pairwise, bound at third order. The cleanest metaphor for
  *invisibility that carries structure* (the annihilation thesis in topology).
- **ABIDE**: the question "does it classify ASD/TD?" was the wrong one. The structural question —
  Ollivier-Ricci **curvature as the geometry of connectome fragility/rupture** (Sandhu–Tannenbaum:
  Δcurvature × Δrobustness ≥ 0; Farooq–Tannenbaum ASD curvature in pathways invisible to scalar DTI) —
  was never asked.

## 6. The genuinely unclaimed (the jewels)

1. **Associator = semantic rupture + a homology functor F** (the four ingredients never assembled).
2. **Hypothesis H-annihilation (open):** sedenion zero divisors as a *candidate* for statistical
   invisibility / epistemic annihilation (no probability theory below 𝕊; the two annihilation
   manifolds of §1 known but unused; law, measure and estimator still undefined).
3. **The unification:** both faces are **G₂** — a single exceptional-geometric theory of rupture with a
   semantic face (the associator) and an epistemic face (annihilation).
4. **Positivization:** the associator as the first *positive, graded, computable* rupture invariant, where
   every prior formalism (Sneed, Abramsky, Amari, Thom) offers only a negative/obstruction object.

## 7. Canonical reading list (must engage)

*Semantic / morphodynamic:* Thom, *Structural Stability and Morphogenesis* (1972) · **Petitot,
*Morphogenesis of Meaning* (1985) — the highest formal bar to clear** · **Wildgen, *Catastrophe Theoretic
Semantics* (1982) — nearest ancestor (E₆–E₈, X₉)** · Coecke–Sadrzadeh–Clark (2010, arXiv 1003.4394) ·
Lambek (1961) + Moortgat multimodal · Marcolli–Chomsky–Berwick, *Mathematical Structure of Syntactic
Merge* (2023, arXiv 2305.18278) · Stasheff associahedra / A∞ (Loday–Vallette 2012).
*Obstruction / epistemic:* **Abramsky–Brandenburger (2011, arXiv 1102.0264) — the rigor model** ·
Abramsky–Sadrzadeh (2014, arXiv 1403.3351) · Balzer–Moulines–Sneed, *An Architectonic for Science* (1987)
+ Kuhn (1962) + Bachelard (1938) · Amari information geometry + Watanabe singular learning theory.
*Algebra / annihilation:* **Baez, *The Octonions* (2002, arXiv math/0105155) — the spine** · Moreno (1998,
q-alg/9710013) · **Reggiani (2024, arXiv 2411.18881) — element locus ZD(𝕊) ≅ V₂(ℝ⁷) ≅ G₂/SU(2), curved** · Cawagas (2004) ·
Biss–Christensen–Dugger–Isaksen (2007/2008) · Günaydin–Piron–Ruegg (1978) · Jackiw (1985) + Szabo (2019,
arXiv 1903.05673).
*Philosophy of structure / gesture:* **Châtelet, *Figuring Space* (1993/2000) — the grammar of
non-associativity as a figure of thought; Hamilton-as-gesture** · Zalamea, *Synthetic Philosophy of
Contemporary Mathematics* (2009).
*Cognition / affect (dynamical realization):* Kelso, *Dynamic Patterns* (1995) + HKB · Rabinovich et al.,
stable heteroclinic channels (2008).
*Precedent to differentiate:* Goertzel (1996) — octonion-of-consciousness, not meaning, no homology.

## 8. Landmines

- **Reputation gradient:** octonion-consciousness is crank-adjacent (Goertzel's venue + fringe
  octonion-mysticism). Foreground the **functor F** and a **falsifiable prediction**; anchor the epistemic
  side in Günaydin-ceiling / obstruction-cohomology, not consciousness talk.
- **Measurement vs structure:** "homology of meaning" risks being another TDA persistence-diagram study.
  The defensible, unclaimed contribution is the **functor**, not a measurement.
- **Coherence debt:** weakening DisCoCat's associativity forfeits monoidal coherence — you owe the
  replacement, which is exactly where **A∞/operadic higher coherence** earns its place.
- **Catastrophe-controversy wound:** Thom/Wildgen verb↔catastrophe assignments are schematism, not
  theorems; cite the classification theorem, not the analogies. de Marrais's PSL(2,7) claim is conjectural
  (use his counts/XOR arithmetic, cite Moreno/Reggiani for proofs).

## 9. Where the compiler fits

Not an ML play. It is the **infrastructure to compute in the algebra of rupture** — the associator, its
VJP, the zero-divisor locus, discrete curvature — in hardware, **within measured tolerance**
(octonion/sedenion associator and its full VJP on Blackwell tensor cores, verified on GB10). The
experiments instrument the structure; they were never contests. The scaffold was mistaken for the
cathedral once already — this document is the correction.

## Warrant of equations (2026-08-23)

A green CI run on this file is StructuralCI: the registry synced, the links resolve, no reference is
broken. It is not a ComputationalOracle, a FormalProof, or a LiteratureValidatedClaim. A document
that current work cites as parent, and that carries equations, should name which of those four backs
each displayed identity. The four corrections above are LiteratureValidatedClaim against Baez /
Moreno / Reggiani; they are not FormalProof in this repository, and #1813's watcher correctly
refused to decide scientific semantics.

---

*Status: all three sweeps folded in. Open verification before verbatim citation (flagged by the sub-agents,
from search-cache rather than full text): Petitot's 1977 "Topologie du carré sémiotique" potential
functions; Wildgen's exact E₆–E₈/X₉ derivation; de Marrais's PSL(2,7) action; the Sandhu market-curvature
sign (reported as increase-during-crisis in the primary source). The morphodynamic layer (Thom/Petitot as
the singularity-theoretic bridge between the two faces) is the next thread to develop.*
