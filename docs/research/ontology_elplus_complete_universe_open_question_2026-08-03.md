<!-- docs:meta
topic_id: repo.docs.research.ontology-elplus-complete-universe-open-question-2026-08-03
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ontology-elplus-complete-universe-open-question-2026-08-03
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Open question: complete concept universe for the EL+ role-aware closure

- **Date:** 2026-08-03
- **Status:** Open research problem (documented, not pursued)
- **Related:** `formal/OntologyELPlusClosureComplete.lean`, `formal/OntologyELPlusClosureVerified.lean`

## Current result

`OntologyELPlusClosureComplete.lean` proves `subBPlusC_iff` / `conflictBPlusC_iff` over `conceptUniv t`:

- `baseConcepts t = {⊤} ∪ atoms ∪ subconcepts-of-axioms` (closed under subconcepts)
- `conceptUniv t = baseConcepts t ∪ {∃s.C | C ∈ baseConcepts t, s : Fin m}` (one existential layer)

This is the correct semantic domain for the repair-oracle application over normalized SNOMED-fragment TBoxes. The completeness theorem is *exactly* the strength the application consumes: subsumption and conflict between any two concepts that a repair tool asserts (after normalization at the boundary).

## Why "complete over all concepts" is not the next step

The classical EL+ result (Baader–Brandt–Lutz, "Pushing the EL Envelope") is that saturation over **subconcepts of the TBox** decides subsumption between TBox subconcepts. Arbitrary-concept subsumption `C ⊑ D` is reduced to that by **normalization**: introduce fresh names `A_C`, `A_D`, add structure-sharing definitions, extend the TBox, saturate, and check `A_C ⊑ A_D`.

Enlarging `conceptUniv` with deeper existential layers (`conceptUnivK t (k+1) = conceptUnivK t k ∪ {∃s.C | C ∈ conceptUnivK t k}`) is computable but buys **no new theorem**: it is exponential cost for zero logical gain, because completeness would still be "over universe members" and derivable pairs with nested endpoints of depth `k+1` still escape.

## The tractable next frontier

If all-concepts completeness is ever needed, the correct target is the **normalization-reduction theorem**:

1. Define `normConcept : Concept α ρ → List (AxiomP α ρ) × α` (fresh-name flattening).
2. Prove conservativity of definitional extensions for `Der`.
3. Compose with `subBPlusC_iff` over the extended TBox.

This yields `subBPlusC (t ++ normC ++ normD) A_C A_D = true ↔ Der t C D` for all `C, D`, preserving PTIME.

## Recommendation

- Do **not** pursue a "bigger `conceptUniv`" (deeper ∃ layers) — exponential cost, same expressivity ceiling.
- Document "direct closure over arbitrary concepts" as an open research problem.
- Treat post-coordinated concepts as normalized at the boundary, which is what the application already does.
