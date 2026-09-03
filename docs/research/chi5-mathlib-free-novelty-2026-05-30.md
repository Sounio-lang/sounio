<!-- docs:meta
topic_id: repo.docs.research.chi5-mathlib-free-novelty-2026-05-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.chi5-mathlib-free-novelty-2026-05-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Novelty assessment: a Mathlib-free, from-scratch-ℝ machine-checked proof of χ(ℝ²) ≥ 5

*SOTA literature review — 2026-05-30. Search tools: WebSearch / WebFetch. EN-UK orthography.*
*Scope: assess the novelty of a formalisation **artefact**, not of the underlying mathematics.*

## The artefact under assessment

A machine-checked proof, in **Lean 4 without Mathlib** (core Lean plus `native_decide`), that the
chromatic number of the plane satisfies **χ(ℝ²) ≥ 5** (Hadwiger–Nelson lower bound, de Grey 2018).
Its distinguishing engineering features are: (1) the reals are built from scratch as a quotient of
Cauchy sequences of rationals (`RealEq`), wrapped in an abstract **`RootedField`** — an ordered field
equipped only with √3, √5, √7, √11, with *no* total `sqrt` and *no* completeness axiom (no Mathlib
`Real`, `Real.sqrt`, or `ring`); (2) the non-4-colourability of a G₅₂₉ unit-distance fragment is a
SAT (cake_lpr / LRAT) certificate re-checked inside Lean via `native_decide`; (3) an abstract algebraic
**transfer theorem** reduces χ(F²) ≥ 5 for any such `RootedField F` to the SAT leg, after which ℝ is
plugged in; (4) the whole stack sits inside a self-hosted, single-author language (Sounio).

The mathematics is **established and peer-reviewed** (de Grey 2018, *Geombinatorics*; arXiv:1804.02385).
The question here is purely whether *this formalisation engineering* is new.

## Q1 — Prior formalisations of Hadwiger–Nelson / χ(ℝ²) ≥ 5

- **`vasnesterov/HadwigerNelson` (Lean 4)** is the closest prior art and the strongest competitor. Its
  README states it verifies the 510-vertex Heule graph is non-4-colourable, *and* "reduce[s] the
  colorability of the plane to the colorability of finite unit distance graphs and complete[s] the
  proof" — i.e. it targets the full χ(ℝ²) ≥ 5 lower bound. Crucially it is **built on Mathlib** (uses
  `ring_nf`, Mathlib `Real`/`dist`) and on **LeanSAT** + an external SAT solver (CaDiCaL).
  <https://github.com/vasnesterov/HadwigerNelson>
- **`google-deepmind/formal-conjectures` (Lean 4 / Mathlib)** contains `HadwigerNelsonAtLeastFive : 5 ≤ χ(ℝ²)`
  but the body is `sorry` — a stated, *unproved* target over Mathlib's `ℝ`. (Also `…AtLeast4`,
  `…AtMostSeven`, `…AtLeastThree`.)
  <https://github.com/google-deepmind/formal-conjectures/blob/main/FormalConjectures/ErdosProblems/508.lean>
- **Heule and collaborators** produced the SAT side: DRAT/LRAT certificates of non-4-colourability for
  the 1581→529→510-vertex graphs, checkable by **formally-verified checkers in ACL2, Coq and Isabelle**
  (Cruz-Filipe et al., "Efficient Certified RAT Verification", CADE-26, arXiv:1612.02353; Heule,
  arXiv:1805.12181; "Trimming Graphs…", arXiv:1907.00929). This verifies only the **finite
  non-4-colourability leg**, not the geometric reduction to χ(ℝ²), and not over a hand-built ℝ.
- **No complete de Grey χ ≥ 5 formalisation found in Coq / Isabelle / HOL.** Those systems appear only
  as LRAT *certificate checkers* for the finite graph, never as a full plane-level result.

**Bottom line for Q1:** a *Mathlib-based* Lean formalisation of χ(ℝ²) ≥ 5 already exists (vasnesterov).
**No Mathlib-free formalisation of χ(ℝ²) ≥ 5 was found in any system** in the queries run
("formalization chromatic number plane Lean", "de Grey … formal proof Coq/Isabelle/HOL",
"native_decide LRAT … chromatic number unit distance"). Absence of evidence ≠ evidence of absence.

## Q2 — Mathlib-free / from-scratch constructive ℝ in an *applied* theorem

- Building ℝ as a quotient of rational Cauchy sequences (rational-ε to avoid circularity) is a
  **well-trodden pedagogical exercise**: the *Logic and Proof* course
  (<https://leanprover-community.github.io/logic_and_proof/the_real_numbers.html>) and Tao's *Analysis I*
  Lean port (`teorth/analysis`, `Chapter5.Real := Quotient CauchySequence.instSetoid`,
  <https://github.com/teorth/analysis/blob/.../Analysis/Section_5_3.lean>) both do exactly this — but as
  *foundations*, typically followed by switching to Mathlib's optimised `ℝ`.
- I found **no precedent** for proving a *named, open-problem-adjacent* result over a **hand-built
  constructive ℝ that deliberately avoids any standard library**. The applied-formalisation norm is the
  opposite: lean *on* Mathlib's `Real` (vasnesterov, formal-conjectures).
- The specific abstraction here — an **ordered field carrying only {√3, √5, √7, √11} and no total
  sqrt/completeness**, then a transfer theorem parametric over such fields — is unusual. I found **no
  prior "RootedField"-style abstraction** for unit-distance/chromatic arguments. (de Grey's coordinates
  live in a real field generated by a handful of surds, so isolating exactly the needed roots is
  mathematically natural, but I could not locate a published formalisation that packages it this way.)
  *Not verified — searches were inconclusive rather than negative.*

## Novelty verdict (calibrated)

- **The mathematics is not novel.** χ(ℝ²) ≥ 5 is de Grey 2018, peer-reviewed and independently SAT-verified. *(High confidence.)*
- **A χ(ℝ²) ≥ 5 Lean formalisation is not novel in the absolute.** `vasnesterov/HadwigerNelson` already targets it — **over Mathlib + LeanSAT**. *(Medium-high confidence; I could not directly open the source tree to confirm it is fully `sorry`-free, only the README's claim of completion.)*
- **The specific *engineering* combination — Mathlib-free + from-scratch Cauchy-sequence ℝ + abstract `RootedField` transfer + `native_decide`-rechecked LRAT + inside a self-hosted single-author language — is plausibly novel as an artefact.** No prior work matching this profile surfaced. *(Medium confidence: this rests on absence of search hits, and on not having inspected competing repos line-by-line.)*
- **What I could NOT verify:** (a) that vasnesterov is genuinely complete and `sorry`-free (raw README fetch 404'd; relying on the GitHub description); (b) that no unpublished / non-indexed Mathlib-free attempt exists; (c) any prior `RootedField`-style abstraction in the literature; (d) the precise dependency footprint (does the Sounio artefact's `native_decide` route through a trusted-compiler axiom — the known `native_decide` soundness caveat re: `@implemented_by` exploits, cf. `GasStationManager/ReplaceNativeDecide`).

**Net:** frame any claim as *"a novel Mathlib-free formalisation engineering of a known result"*, **not**
*"first formalisation of χ(ℝ²) ≥ 5"* (that would be false — vasnesterov; and the statement exists in
formal-conjectures). The defensible novelty is the *dependency-minimal, abstract-field* approach and its
home in a self-hosted language.

## What to check before any external claim

1. **Open vasnesterov's source** and confirm whether `theorem … : 5 ≤ χ(ℝ²)` is fully proved
   (`#print axioms`, no `sorry`) and *exactly* which Mathlib pieces it uses — this calibrates the
   "Mathlib-free" differentiator.
2. **Confirm the Sounio artefact's own axiom footprint**: run `#print axioms` on the top theorem; state
   precisely whether it depends on `Classical.choice`, propositional extensionality, and the
   `native_decide` / trusted-compiler axiom (and whether the LRAT leg can also be `decide`-checked).
3. **Verify the abstract transfer theorem instantiates ℝ with no hidden completeness use** — the selling
   point is "only √3,√5,√7,√11, no completeness". Make sure no lemma silently assumes a complete field.
4. **Cross-check the surd set** against the G₅₂₉ / de Grey coordinate field (de Grey's graph uses surds
   such as √3, √11, √33); confirm {√3,√5,√7,√11} is sufficient *and* that 529 vs 510/509 vertex counts
   and provenance are stated accurately.
5. **Pin the SAT certificate provenance** (cake_lpr / LRAT, which solver, which graph) and note it is
   re-checked *inside* Lean, distinct from external ACL2/Coq/Isabelle LRAT checkers.
6. **Search the formal-methods venues** (ITP/CPP/CICM/JAR) and the Lean Zulip for any 2025–2026
   Mathlib-free Hadwiger–Nelson work before publishing a "first/novel" claim; this review used web
   search only and did not query those archives directly.
7. **Run the mandatory `bin/llm-offload -t math-review -p xai`** on the Lean theorem statements and the
   transfer lemma before any external/math claim, per repo policy.

### Sources
- de Grey 2018 — arXiv:1804.02385 (*Geombinatorics*).
- `vasnesterov/HadwigerNelson` — <https://github.com/vasnesterov/HadwigerNelson>.
- `google-deepmind/formal-conjectures` — `FormalConjectures/ErdosProblems/508.lean`, `Wikipedia/HadwigerNelson.lean`.
- Cruz-Filipe, Heule, Hunt, Kaufmann, Schneider-Kamp, "Efficient Certified RAT Verification", CADE-26 — arXiv:1612.02353.
- Heule, "Computing Small Unit-Distance Graphs with Chromatic Number 5" — arXiv:1805.12181; "Trimming Graphs Using Clausal Proof Optimization" — arXiv:1907.00929.
- *Logic and Proof*, §21 (Cauchy-sequence ℝ); `teorth/analysis` `Section_5_3.lean`.
- `GasStationManager/ReplaceNativeDecide` — `native_decide` soundness caveat.
