<!-- docs:meta
topic_id: repo.docs.research.multiquad-faithfulness-note
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.multiquad-faithfulness-note
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Multiquadratic-field faithfulness and the QF ring groundwork

**Status:** early research note (Level 1–2). Machine-checked additive/multiplicative
commutativity for the `QF` kernel; associativity, distributivity, units, and the
ℝ-embedding wall explicitly open.

**Lean artifact:** `formal/lean4/SounioMultiquadRing.lean` — standalone core Lean (no imports,
no Mathlib). Compile: `lean formal/lean4/SounioMultiquadRing.lean`.

**Upstream context:** `SounioDeGreyUnitDistance.lean` (2670 edge unit-distance certificate over
`QF`), `SounioDeGreyChi5Concrete.lean` (geometry leg discharged on `QF × QF`).

---

## 1. What “faithfulness” means here

The de Grey / Heule pipeline shows that G₅₂₉ has squared Euclidean distance **exactly 1**
when coordinates live in the multiquadratic field

\[
\mathbb{Q}(\sqrt{3},\sqrt{5},\sqrt{7},\sqrt{11}),
\]

represented in Sounio/Lean as `QF := List Int × Int`: sixteen integer numerators over the
basis `{ \sqrt{\prod S} : S \subseteq \{3,5,7,11\} \}` plus a common denominator. Addition and
multiplication (`qadd`, `qmul`) are the standard “clear denominators + XOR on basis masks”
operations copied verbatim from `SounioDeGreyUnitDistance.lean`.

Saying that “dist² = 1 in `QF` implies a **real** unit distance in ℝ²” is stronger than the
symbolic certificate. It requires two logically separate facts:

1. **Ring structure:** `QF` (modulo the usual identification of scaled representatives) is the
   commutative ring presented by those sixteen basis radicals with the XOR/`bcoeff` product
   rules — i.e. the laws of a commutative ring hold for `qadd`/`qmul`.

2. **Field + independence over ℝ:** the sixteen radicals are ℚ-linearly independent in ℝ, and
   the evaluation map sending each basis element to the corresponding real radical is a
   **ring homomorphism** `QF → ℝ`. Only then does symbolic distance-1 imply Euclidean
   distance-1 after embedding.

Item (2) is textbook multiquadratic field theory over ℝ. It cannot even be **stated** in core
Lean without constructing ℝ (Dedekind cuts, Cauchy reals, or Mathlib’s `Real`). That is
deliberately out of scope for this Mathlib-free project.

Item (1) is the tractable, ℝ-free part. It is what `SounioMultiquadRing.lean` begins to
mechanise.

---

## 2. What is proved in core Lean (today)

`SounioMultiquadRing.lean` copies the kernel locally and proves:

| Theorem | Statement | `#print axioms` |
|---|---|---|
| `qadd_comm` | `∀ x y, qadd x y = qadd y x` | `propext`, `Quot.sound` only |
| `qadd_zero_left` | `∀ x, qfLen16 x → qadd qzero x = x` | `propext`, `Quot.sound` only |
| `qadd_zero_right` | `∀ x, qfLen16 x → qadd x qzero = x` | via `qadd_comm` |
| `qmul_comm` | `∀ x y, qmul x y = qmul y x` | `propext`, `Quot.sound`, `Classical.choice`, plus **16 finite permutation certificates** from `native_decide` on `i ↦ i XOR idx` for `idx < 16` |

No `sorry`, no unlabelled `axiom`, no `sorryAx` on any of the above.

**Normalisation caveat:** `qadd`/`qmul` always emit a length-16 coefficient list. Zero laws are
therefore stated under `qfLen16 x` (the de Grey coordinates in `SounioDeGreyUnitDistance.lean`
satisfy this). Shorter lists are padded implicitly by `gi`/`getD` for arithmetic but are not
definitionally equal to their padded form — a separate normal-form story, not hidden.

**`qmul_comm` audit note:** commutativity of the XOR product uses a bijection argument on
`Fin 16`, discharged by case-split + `native_decide` on each XOR permutation. This is honest
finite proof, but it introduces the usual `native_decide` trust boundary (same family as the
2670-edge geometry certificate).

---

## 3. What remains open (named obligations, not axiomatised)

The file ends with explicit `Prop`-valued obligations — **not** assumed:

| Obligation | Why open |
|---|---|
| `QmulAssocObligation` | `qmul_assoc` needs a triple sum over 16×16×16 basis crossings with rational `bcoeff` factors; without `ring` or `BigOperators`, the reindexing is large |
| `QmulLeftDistribObligation`, `QmulRightDistribObligation` | `qmul` distributes over `qadd` only after clearing denominators; proof is foldl/foldl algebra over nested sums |
| `QaddNegObligation` | naive `qsub qzero x` scales numerators by `x.2`; additive inverse to `qzero` needs a dedicated normalised negation, not raw `qsub` |
| `QmulOneObligation` | existence of a multiplicative unit compatible with denominator scaling |

These are the remaining **symbolic** ring laws. Proving them in core Lean is feasible in
principle (likely more `native_decide` on finite indices, or a custom `BigOperators`-free
sum library), but not done here.

---

## 4. The honest wall: ℚ-linear independence and `QF ↪ ℝ`

Even with every commutative-ring law proved, faithfulness to **physical** unit distance in ℝ²
still requires:

- a constructed ordered field ℝ;
- a homomorphism `eval : QF → ℝ` with `eval(b_mask) = √(∏ S)` on basis masks;
- ℚ-linear independence of the sixteen radicals (equivalently: the 16×16 basis matrix has full
  rank over ℚ).

That is exactly the standard “multiquadratic field embeds into ℝ via radicals” lemma. Mathlib
mechanises it via `Real.sqrt`, field towers, and `ring`. None of that is novel for the de Grey
result — and none of it is available in this core-Lean toolchain.

**What *is* the honest self-hosted summit today:** the exact **symbolic field-plane**
`QF × QF` with unit relation `unitFP` (algebraic dist² = 1), discharged for all 2670 edges of
G₅₂₉ in `SounioDeGreyChi5Concrete.lean`. Combined with the SAT leg (G₅₂₉ not 4-colourable,
verified externally), this yields chromatic number ≥ 5 for the **symbolic** unit-distance graph
on that plane — not yet for measurable subsets of ℝ² until the ℝ lift is added.

The ℝ lift is **deferred standard analysis**, not a gap in the Sounio-specific or de
Grey-specific novelty. This note records the ring-law groundwork that makes the lift
*meaningful* once ℝ exists, without overstating how far we are without it.

---

## 5. Maturity

| Layer | Status |
|---|---|
| Edge geometry in `QF` (2670 edges) | machine-checked (`native_decide`) |
| Geometry leg on `QF × QF` | machine-checked (`SounioDeGreyChi5Concrete.lean`) |
| Commutativity of `qadd`/`qmul` | machine-checked (`SounioMultiquadRing.lean`) |
| Full commutative-ring axioms for `QF` | partial; obligations listed |
| `QF ↪ ℝ` faithfulness | not statable in core Lean; textbook / Mathlib path |

**Bottom line:** we have a clean, auditably honest split — symbolic field-plane chromatic lower
bound (proved modulo SAT), plus early ring-law lemmas, with the ℝ embedding wall named precisely
rather than hand-waved.
