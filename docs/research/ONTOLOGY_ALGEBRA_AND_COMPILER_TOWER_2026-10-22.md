<!-- docs:meta
topic_id: repo.docs.research.ontology-algebra-and-compiler-tower-2026-10-22
authority: repo_only
audience: researchers
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ontology-algebra-and-compiler-tower-2026-10-22
-->

# Ontology as algebra, typed zeros, a semiring reasoner, and the compiler as a tower

**Date:** 2026-10-22 · **Author:** Demetrios Chiuratto Agourakis

**Status: research record — conjectures with named ancestors and stated tests.
Nothing here is implemented beyond the anchors cited as existing.** The
companion record for reasoning, metacognition and the interoceptive organ is
[EPISTEMIC_REASONING_AND_METACOGNITION_2026-10-22.md](EPISTEMIC_REASONING_AND_METACOGNITION_2026-10-22.md).
Each section ends with what would make it false.

---

## 1. Ontology as body, not label

### 1.1 Teleosemantics, and what hardware has carried so far

The proposal: when a value descends the compiler, its *function* — what it is
for, not only what shape it has — descends with it. In philosophy of mind this
is Millikan's teleosemantics (a representation's content is its proper
function). Hardware has carried partial versions: tagged architectures (Burroughs
B5000, Lisp machines) carried type bits per word; CHERI (in silicon on Arm
Morello) carries bounds and permissions per pointer; typed assembly language
(Morrisett et al., 1999) carries types into machine code. All carry **type or
permission, never function**: none knows that a number is a vancomycin dose in
mg/kg whose function is to decide an infusion.

The type system already carries more than that. Subsumption of `Knowledge` over
ontology classes refuses combinations the ontology forbids: **E261** for
disjoint classes and **E262** for unrelated ones
(`self-hosted/check/check.sio:8054–8057`; catalogue rows in
[`error-catalog.md`](../llm-guide/error-catalog.md)).

Two ways function can descend:

1. **as annotation** — extra bits beside the datum, separable from it (tags);
2. **as body** — the datum's form *is* its function, as a folded protein's
   structure is what it does. A dual number is already of this kind: its
   uncertainty is not a label but part of the arithmetic.

### 1.2 The algebra

**Degree zero exists: physical units.** A quantity is value × dimension;
multiplication adds dimension exponents, addition is defined only within one
dimension. That is a **graded ring**: the grade is part of the object (Kennedy's
units-of-measure types, implemented in F#, the oracle language of ADR-009).

**Generalise the grade from dimensions to ontological roles.** In EL+, roles
compose by chains (`r ∘ s ⊑ t`); a composition can grade an algebra, so the
product of two values carries the composition of their roles.

**In a path algebra** (the algebra of a graph of roles), the product of two
elements that do not compose is defined to be **zero** — not an error, not an
exception: annihilation inside the algebra. **The ontology's incompatibilities
become the algebra's zero divisors.** "Vancomycin dose × heart rate" does not
raise an alert; it is zero, because the path does not exist.

This is the sedenion programme read from the other side. There, the zero divisor
was read as relational annihilation and the empirical line died because ordinary
training did not leave that imprint
([program registry](PROGRAM-REGISTRY-mercyful-learning.md)). Here the zero
divisor has meaning **by construction**: it is designed, not discovered in data.

Honesty about "body" in silicon: in the end the grade is bits too. The real
difference between label and body is **inseparability** — in a graded algebra no
operation is defined on the value without the grade, so there is no check to
bypass, only the product rule.

**False if:** a role-graded product cannot be made associative where the
ontology's role chains are (the path algebra is associative by construction;
the risk is in the grading map from EL+ chains, which must be checked per
TBox), or if the grading forces a combinatorial blow-up in the number of grades
that the checker cannot afford.

## 2. Typed zeros

The exact-core work fixed that a zero carries its provenance and that coercion
between kinds is refused: the analytic layer's verdict is `MeasuredF64(eps)`, the
exact layer decides annihilation by integer equality
([`docs/EXACT_CORE.md`](../EXACT_CORE.md): *"exactness is a property of the
contract, not of the number"*). This record adds species, one of which is
dangerous untyped.

| species | example | meaning | status |
|---|---|---|---|
| rounding zero | magnitude below what the format represents | measured | exists (`MeasuredF64`) |
| cancellation zero | `x − x`; `ab = 0` with `a, b ≠ 0` (exact zero divisor) | proved | exists (exact layer) |
| **incompatibility zero `0_⊥`** | the path-algebra product of incompatible roles | **this quantity does not exist**, not "it is zero" | new; today a refusal (E261/E262), not a value |
| absence of evidence | a class with no known instances | **unknown, not zero** (description logic is open-world) | must never be represented as zero |

The danger is confusing `0_⊥` with the first two: if the incompatibility zero
becomes a numerical zero, someone reads "dose × heart rate = 0 mg·bpm", a lie with
the face of a result. In a graded algebra the separation is natural: each grade
has its own zero `0_g` ("this quantity exists and equals zero"); the product of
incompatibles falls in no grade and goes to `0_⊥` ("this quantity does not
exist"). Same bits, opposite meaning — **the grade carries the meaning of the
zero.** In the reasoner this is exactly the distinction description logic
requires: an **unsatisfiable** class (`0_⊥`, empty in every model) versus a
class **without known instances** (unknown).

The hardware already concedes the point once: IEEE 754's `+0` and `−0` keep the
direction a value came from.

**Consequence for the compiler today:** keep E261/E262 as refusals. If `0_⊥`
ever becomes a value (to let a reasoner compute with it), it must be a distinct
type with no coercion to any numeric zero — the same rule the exact core
already enforces between `MeasuredF64` and the exact verdict.

## 3. A reasoner over exchangeable semirings

`stdlib/ontology/elplus.sio` computes role-aware EL+ closure as a fixpoint of
eight completion rules over flattened **Boolean** arrays (`S`, `R`, `rclos`,
`exid`), dense for small TBoxes and sparse at OAEI-Anatomy scale. Three
generalisations, each deeper than the last:

1. **Saturation as linear algebra.** The completion rules are sparse matrix
   products over the Boolean semiring, iterated to a fixpoint; classification
   becomes matrix multiplication, which runs on the kernels the compiler
   already emits. Precedent: GraphBLAS, TensorLog.
2. **Change the semiring, change the question.**
   - Boolean: *is* A ⊑ B?
   - provenance polynomials (Green, Karvounarakis & Tannen, PODS 2007): *why*
     A ⊑ B? Minimal justifications fall out of the algebra — axiom pinpointing
     (Baader & Peñaloza) without a separate search;
   - confidence: *how sure* is A ⊑ B, when axioms carry uncertainty;
   - **`Knowledge`**: axioms carry `u`, `e` and Q, and "Q does not rise without
     an oracle" holds *inside* the reasoning.
3. **Inconsistency as annihilation.** Disjointness axioms (`A ⊓ B ⊑ ⊥`) become
   zero divisors; an unsatisfiable class is one whose product annihilates;
   detecting inconsistency becomes detecting zero.

Technical condition, stated once: with cycles, the fixpoint exists only in
well-behaved semirings (ω-continuous; idempotent for termination). The full
provenance semiring must be replaced by an absorptive one (e.g. `Sorp`/`Why`
variants) to guarantee termination.

**False if:** the provenance-semiring fixpoint over the sparse variant is not
tractable at the scale `elplus.sio` already handles in Boolean form; that is the
first measurement.

## 4. The compiler as a tower

### 4.1 Self-similarity and the Galois structure

Lexer, parser, AST, check, IR, native: each level receives a representation,
analyses it, and hands down one that is **less ambiguous and less free**; inside
each level the pass pipeline repeats the cycle in miniature. Each lowering is
many-to-one: it **loses information and gains commitment**.

Descending is a **function**; ascending is a **relation** (one machine program
is compatible with many sources). What a low level "knows" about the level above
is a **set** — every upper state compatible with what it sees. That is a Galois
connection (abstraction/concretisation), and Cousot showed that a system's
semantics form a **hierarchy** of Galois connections. The rigorous enclosure
`e` of the [interval proposal](../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md)
is the numerical instance: the shadow, below, of everything that could be true
above.

What abstract interpretation leaves outside is the subject: the analyst computes
the abstractions from outside. Here each level would compute the concretisation
**about itself** — "given what I see, what could be true above?". DWARF is the
embryo: the map from each instruction back to a source line, used only to debug.
Provenance in `Knowledge` is an epistemic DWARF.

Practical gain of the self-similarity: if every stage preserves the same
epistemic invariant (Q does not rise without an oracle; `e` still encloses the
value), **one parametric proof** serves every level — the way CompCert composes
per-pass simulations.

### 4.2 Entropy and value of information as a conjugate pair

Writing machine code is expensive because it **chooses a microstate**: the
source is a macrostate, the correct machine programs implementing it are its
microstates, and `S = log(number of microstates)`. Every bit the upper levels
left free (which register, which instruction, which order) is a decision to be
paid — which is why register allocation and optimal instruction selection are
NP-hard: the choice space *is* that entropy. The same S appears both ways:
descending, one **pays** S bits of decision; ascending, one **loses** S bits of
knowledge (concretisation returns a set of size 2^S). *Cost of deciding downward
equals ignorance upward.*

Pair it with `Φ_n` — here **value of information**, not Tononi's Φ. Under the
logarithmic scoring rule, the value of a signal is the mutual information
between signal and outcome: `Φ_n = I(signal_n ; correct)`, and always
`Φ_n ≤ S_n`. The pair is exactly the **information bottleneck** (Tishby, Pereira
& Bialek, 1999): compress (lower S) while preserving relevance (keep Φ), with a
multiplier β; the slope `dΦ/dS = 1/β` makes β the conjugate variable, as
temperature is between energy and entropy. Each level of the tower is a point on
the bottleneck curve and each lowering a step along it.

**Conjecture:** the tower walks the curve with β varying self-similarly; a
constant ratio between consecutive levels is the fractal signature, and
`log Φ_n = −D · log S_n + c` defines a dimension D.

**Ancestor and warning:** the information plane applied to deep-network layers
(Shwartz-Ziv & Tishby) was contested by Saxe et al. (2018), because estimating
mutual information in deterministic networks is treacherous. The compiler has an
advantage: S_n can be **counted** (enumerate legal choices per level), not
estimated. **False if** the counted points do not fall on a line in log-log.

The physical reading, kept as reading: below the ISA the tower continues (µops,
out-of-order scheduling choosing an order at run time, a bit as a thresholded
voltage with an error probability, ECC). The central dogma is the same shape —
DNA → RNA → protein is lowering; the degenerate genetic code (64 codons, 20 amino
acids) is many-to-one instruction selection; protein → DNA is a relation — and
replication shows the price of fidelity: polymerase alone errs ~1 in 10⁴–10⁵,
proofreading improves that ~100×, mismatch repair reaches ~1 in 10⁹–10¹⁰, and
Hopfield's kinetic proofreading (1974) shows fidelity beyond equilibrium costs
dissipated energy. Landauer is an analogy here: deciding in a compiler is not
erasing physical bits.

### 4.3 Memory allocation by semantic distance

The memory hierarchy is a metric space, and a hierarchical one: the distance
between two addresses is the first cache level they share — a tree metric
(ultrametric). Program values live in a space of semantic distance. **Allocation
is embedding one metric into the other with least distortion.** Fakcharoenphol,
Rao & Talwar (2004): every metric embeds into tree metrics with expected
distortion O(log n); the hierarchy *is* a tree, so the problem has a known
guarantee.

Semantic distance is exactly the information lost on the way down the tower;
today's allocator works blind, from liveness and interference. The proposal is
to carry it down and use it to decide. Immediate use in Sounio: `Correlated`
values must update their covariance together, so keeping them in one cache line
makes epistemic honesty cheaper.

**Which distance:** the provenance graph of `Knowledge` gives the structure; its
**Ollivier–Ricci curvature** gives the geometry (the tool of the author's
computational psychiatry, here a compiler pass). Positive curvature: overlapping
neighbourhoods, values born and used together → same cache line. Negative
curvature: a bridge between groups → the place to cut between pages or NUMA
nodes. **Ricci flow** makes it hierarchical: evolving weights by curvature
shrinks positive edges and stretches negative ones, and communities separate
coarse-to-fine (Ni et al., 2019). The flow's hierarchy is a tree, and so is the
memory hierarchy: each cache level corresponds to a scale of the flow.

Practical constraint: Ollivier–Ricci needs an optimal-transport solve per edge,
too expensive per compilation. In the compiler, **Forman** curvature
(combinatorial, cheap) or a Jost–Liu lower bound; Ollivier for offline
validation. Ancestor: locality-driven data layout (Chilimbi et al., 1999), by
access profile, never by meaning.

**Hypotheses that make it science:** (1) semantic proximity predicts co-access
beyond what liveness predicts; (2) mean curvature of the provenance graph
predicts cache-miss rate (tree-like, negatively curved graphs embed with little
distortion; dense positive regions either fit a line or pay). **False if** (1)
adds nothing over liveness on a measured corpus.

## Claims Forbidden

- Do not describe role-graded algebras, typed zeros beyond the exact core, the
  semiring reasoner, or curvature-guided allocation as implemented.
- Do not use "fractal dimension" for the compiler before §4.2's counted
  measurement exists; until then it is an image.
- Do not read Φ here as integrated information or as anything about
  consciousness.

## References

- Baader, F., & Peñaloza, R. (2010). Axiom pinpointing in general tableaux. *Journal of Logic and Computation*, 20(1), 5–34.
- Chilimbi, T. M., Hill, M. D., & Larus, J. R. (1999). Cache-conscious structure layout. *PLDI 1999*.
- Cousot, P. (2002). Constructive design of a hierarchy of semantics of a transition system by abstract interpretation. *Theoretical Computer Science*, 277(1–2), 47–103.
- Fakcharoenphol, J., Rao, S., & Talwar, K. (2004). A tight bound on approximating arbitrary metrics by tree metrics. *JCSS*, 69(3), 485–497.
- Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings. *PODS 2007*.
- Hopfield, J. J. (1974). Kinetic proofreading. *PNAS*, 71(10), 4135–4139.
- Kennedy, A. (1997). Relational parametricity and units of measure. *POPL 1997*.
- Millikan, R. G. (1984). *Language, Thought, and Other Biological Categories*. MIT Press.
- Morrisett, G., Walker, D., Crary, K., & Glew, N. (1999). From System F to typed assembly language. *TOPLAS*, 21(3), 527–568.
- Ni, C.-C., Lin, Y.-Y., Luo, F., & Gao, J. (2019). Community detection on networks with Ricci flow. *Scientific Reports*, 9, 9984.
- Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. *Journal of Functional Analysis*, 256(3), 810–864.
- Saxe, A. M., et al. (2018). On the information bottleneck theory of deep learning. *ICLR 2018*.
- Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *Allerton Conference*.
- Watson, R. N. M., et al. (2015). CHERI: a hybrid capability-system architecture. *IEEE S&P 2015*.
