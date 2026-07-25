<!-- docs:meta
topic_id: repo.docs.internal.concepts.nonassociative-order
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.nonassociative-order
-->

# Nonassociative Order

Concept-ID: `SOUNIO-NONASSOCIATIVE-ORDER`

## Founder Intent

When grouping or interaction history changes a result, the language must retain
that order instead of normalizing it under associative assumptions.

## Mathematical Core

```text
[a,b,c] = (a*b)*c - a*(b*c)
```

The associator and its norm are executable mathematical objects.

## Current Surfaces

- `stdlib/algebra/associator_field.sio`
- `stdlib/epistemic/uncertain_octonion.sio`
- `stdlib/epistemic/perturbation_graph.sio`
- `stdlib/epistemic/parenthesization_receipts.sio` (nominal research boundary)
- `stdlib/epistemic/state_aliasing_receipts.sio` (nominal state-sufficiency boundary)
- `self-hosted/ir/ir.sio` (`IrAssociator`)
- `self-hosted/native/lower_ir.sio`
- K-AXI/GPU associator kernels

## Required Invariants

- `NonAssoc` is an effect obligation, not evidence of a physical ontology.
- Parenthesization remains explicit where the algebra is nonassociative.
- Correlation and order-induced variance are not double counted.
- Mathematical identities, structural interpretations, and physical claims
  remain separately labeled.

## Claims Forbidden

- A system is physically octonionic merely because the associator models it.
- `kappa * norm_sq(associator)` is a physical variance term without a binding,
  units, and discriminating experiment.

## Ordered Path Receipt Boundary

`parenthesization_receipts.sio` keeps a narrower distinction executable before
any algebraic carrier is selected:

```text
OrderedTransformationPathI64
!= PipelineOrderSensitivityI64
!= AggregationBoundaryI64
!= BracketingDesignI64
!= AlgebraicAssociatorProbeI64
!= BasisTripleAssociatorProbeI64
!= BasisTripleAssociatorAbstentionI64
!= RefinedStateRepresentationContestPlanI64
!= ParenthesizationSensitivityI64
!= clinical authority
```

An `OrderedTransformationPathI64` is explicitly still an ordered,
function-composition path. A parenthesization-sensitive result requires a
declared aggregation boundary, two distinct predeclared groupings, a matched
candidate-model contest, and a declared synthetic discriminating feature. The
library records those categories with private constructors; it does not make an
algebraic, physical, psychiatric, causal, or clinical conclusion.

`BasisTripleAssociatorProbeI64` is a separate, explicitly mathematical bridge:
it delegates to `stdlib/algebra/associator_field.sio`, the canonical octonion
basis carrier, and requires a nonzero norm before recording the selected basis
tags. `tests/run-pass/associator_field_octonion.sio` fixes the two controls used
here: Fano triple `(1,2,3)` has `norm_sq = 0`, while non-Fano triple `(1,2,4)`
has `norm_sq = 4`. The receipt may establish a nonzero result for that declared
carrier only. It does not bind the carrier to a biological process, an empirical
representation, a psychiatric mechanism, or the parenthesization-sensitive
model contest.

`BasisTripleAssociatorAbstentionI64` preserves the complementary zero result
for one selected basis triple. That zero does not certify that a wider carrier,
an empirical system, or a representation contest is associative; it only
refuses to fabricate a nonzero basis-triple probe where the declared arithmetic
returned zero.

`RefinedStateRepresentationContestPlanI64` is a stronger research-plan
boundary, not an empirical result. It is constructible only from a prior
`StateRefinementRequestI64` and a bracketing design with the same declared
source system and target feature. It carries distinct tags for a refined
ordered-state baseline, a capacity-matched associative control, a
bracket-sensitive candidate, and declared capacity/held-out-evaluation plans.
It also requires plans for a triple-composition probe, an associative-projection
control, and a bracket-null control. Those tags preserve what a future
preregistration must compare and audit; they do not say that capacity was
matched, external validation occurred, the associator was activated, a model
was fit, or any model won.

`PipelineOrderSensitivityI64` records the narrower collision in which two
declared transformations produce different synthetic outputs in opposite
orders. It remains distinct from a parenthesization result: a difference
between `T2(T1(history))` and `T1(T2(history))` is not a statement that
`(a diamond b) diamond c` differs from `a diamond (b diamond c)`.

### Semantic Lane Declaration

```text
Semantic-Lane-ID: EPISTEMIC-PARENTHESIZATION-RECEIPTS-V0-20260721
Owner: Codex
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER; SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: preserve an operational distinction between sequence order, declared grouping boundaries, algebraic probes, and research conclusions
Transformation: add private-constructor nominal receipts that require a declared aggregation boundary and contest before a synthetic parenthesization-sensitivity receipt is constructible, then bind a stronger preregistered representation-contest plan to a prior state-refinement obligation
Types-Changed: added OrderedTransformationPathI64, PipelineOrderSensitivityI64, AggregationBoundaryI64, BracketingDesignI64, AlgebraicAssociatorProbeI64, BasisTripleAssociatorProbeI64, BasisTripleAssociatorAbstentionI64, AssociativityContestI64, RefinedStateRepresentationContestPlanI64, BracketDiscriminatingFeatureI64, ParenthesizationSensitivityI64, and ParenthesizationAbstentionI64
Effects-Changed: the BasisTripleAssociatorProbeI64 and BasisTripleAssociatorAbstentionI64 constructors declare Mut, Div, Panic, and NonAssoc for the selected basis-associator computation; the RefinedStateRepresentationContestPlanI64 constructor declares Panic for cross-receipt and candidate-distinctness refusal
IR-Changed: none
Claims-Introduced: import callers cannot silently substitute an ordered path, a pipeline-order collision, an aggregation boundary, a nominal algebraic probe, a nonzero basis-triple associator probe, a zero basis-triple abstention, or a preregistered refined-state contest plan for a parenthesization-sensitivity receipt
Claims-Forbidden: physical nonassociativity, biological or psychiatric mechanism, empirical model fit, causal effect, treatment effect, clinical utility, or clinical authority
Assumptions: supplied tags describe a synthetic fixture or declared protocol mapping; they do not validate that mapping
Write-Set: stdlib/epistemic/parenthesization_receipts.sio; stdlib/epistemic/state_aliasing_receipts.sio; tests/run-pass/epistemic_parenthesization_receipts_import_smoke.sio; tests/run-pass/epistemic_state_aliasing_receipts_import_smoke.sio; tests/compile-fail/epistemic_*parenthesization*.sio; tests/compile-fail/epistemic_exact_associator_probe_cannot_replace_parenthesization_sensitivity.sio; tests/compile-fail/epistemic_basis_associator_abstention_cannot_replace_parenthesization_sensitivity.sio; tests/compile-fail/epistemic_refined_state_contest_plan_cannot_replace_parenthesization_sensitivity.sio; tests/compile-fail/epistemic_refined_state_contest_plan_direct_construction.sio; tests/fixtures/epistemic/basis_triple_associator_probe_rejects_fano.sio; tests/fixtures/epistemic/basis_triple_associator_abstention_rejects_nonfano.sio; scripts/ci/epistemic_receipt_source_fresh_gate.sh; docs/internal/concepts/nonassociative-order.md
Read-Set: stdlib/algebra/associator_field.sio; stdlib/algebra/octonion.sio; tests/run-pass/associator_field_octonion.sio; stdlib/epistemic/observation_provenance.sio; stdlib/epistemic/state_aliasing_receipts.sio; docs/internal/concepts/ordered-path-provenance.md; docs/research/psychiatric_informative_observation_contract_2026-07-21.md
Positive-Witness: the declared carrier control binds `(1,2,3)` to norm_sq=0 and `(1,2,4)` to norm_sq=4; imported synthetic receipt smoke carries history source identity through ordered path, boundary, design, exact probe, zero abstention, a source/target-matched state-refinement obligation, distinct ordered/associative/bracket-sensitive plan tags, and declared triple-composition, associative-projection, and bracket-null control plans before contest, feature, sensitivity, and abstention
Negative-Witness: ordered path != aggregation boundary; pipeline-order sensitivity != parenthesization sensitivity; aggregation boundary != bracketing design; nominal algebraic probe != parenthesization sensitivity; nonzero basis-triple associator probe != parenthesization sensitivity; zero basis-triple abstention != parenthesization sensitivity; refined-state contest plan != parenthesization sensitivity; imported callers cannot fabricate a refined-state contest plan or sensitivity; sensitivity != clinical authority
Acceptance-Gate: bin/souc check stdlib/epistemic/parenthesization_receipts.sio; bin/souc run tests/run-pass/epistemic_parenthesization_receipts_import_smoke.sio; scripts/run_sio_test_suite.sh --filter-prefix epistemic_ --jobs 1 --verbose; scripts/ci/epistemic_receipt_source_fresh_gate.sh
Integration-Target: #901-gated source-fresh imported psychiatric collision suite
Authoritative-Only-If: these receipts prove only the selected nominal program boundaries; no semantic or empirical authority follows from a default-wrapper run
```

### Integration Receipt

```text
Semantic-Outcome: the language now preserves the difference between ordered processing, pipeline-order sensitivity, and a separately declared parenthesization-sensitive research scaffold
Concept-Status-Before: ordered-path provenance and nonassociative algebra were independently represented, with no generic nominal bridge requiring a boundary and contest before sensitivity
Concept-Status-After: imported callers can construct the synthetic sensitivity receipt only after the declared typed chain, while a separate source/target-matched refined-state representation contest plan preserves the baseline obligations without becoming a result
Distinctions-Added: ordered path != pipeline-order sensitivity; pipeline-order sensitivity != parenthesization sensitivity; ordered path != aggregation boundary; aggregation boundary != bracketing design; nominal algebraic probe != nonzero basis-triple probe != zero basis-triple abstention != refined-state contest plan != sensitivity receipt; sensitivity receipt != clinical authority
Distinctions-Preserved: ordinary function composition remains associative; declared nonassociative algebra remains separate from a protocol boundary; observation provenance remains non-clinical
Distinctions-Erased: none
Evidence-Run: basis `(1,2,4)` is checked against norm_sq=4 and basis `(1,2,3)` against norm_sq=0; imported smokes carry both source identity and the refined-state contest plan; the source-fresh gate also requires raw runtime refusal when a nonzero probe is called on `(1,2,3)` or a zero abstention is called on `(1,2,4)`; focused compile-fail substitutions and current-source gate remain required
Fallback-Path: default wrapper reported target-resolution fallback=unresolved_default_x86_64_linux; results are nominal API evidence only, not source-fresh imported-native proof
Legacy-Kept: associator field, ordered-path compiler IR, observation-provenance receipts, and existing research contracts remain unchanged; state-aliasing receipts retain their prior semantics and expose only their existing source-system tag
Conflicting-Lanes: #901 retains ownership of source-fresh imported-native acceptance and target-resolution repair
Next-Semantic-Interface: an import-bearing synthetic collision may proceed only after #901 supplies a current-source no-target-fallback acceptance gate
```

## Observed State Alias Boundary

`state_aliasing_receipts.sio` adds an antecedent question to the associativity
contest: before asking whether a representation needs a special aggregation
operator, has the declared observed proxy actually retained enough state to
make two histories interchangeable for the future feature under study?

```text
ObservedProxyI64
!= ObservedProxyCollisionI64
!= PredictiveChallengeI64
!= PredictiveSeparationI64
!= StateRefinementRequestI64
!= latent-state equivalence
!= causal equivalence
!= clinical authority
```

An `ObservedProxyCollisionI64` records two distinct source-aligned histories
that share one supplied proxy tag. It is not evidence of a hidden biological
state. A `PredictiveSeparationI64` requires a predeclared horizon, target
feature, comparator, and two different synthetic output tags. It records a
modeling collision only; it does not establish predictive accuracy,
generalization, causal effect, treatment effect, or a clinical decision.

This boundary reflects a disciplined version of a powerful research question.
PK/PD hysteresis and drug-target kinetics can make the same systemic exposure
correspond to different observed effects because timing, target-site kinetics,
feedback, metabolites, or regulatory state matter. That motivates retaining
history or state; it does not by itself select a nonassociative carrier. The
IUPHAR ligand-bias guidance likewise treats time, assay, pathway, cell state,
and system context as interpretation-critical. Computational-mechanics causal
states give the abstract comparator: histories may be grouped only when their
conditional futures agree under the declared predictive task.

- [PK/PD hysteresis mechanisms](https://pmc.ncbi.nlm.nih.gov/articles/PMC4332569/)
- [Drug-target kinetic PK/PD models](https://pmc.ncbi.nlm.nih.gov/articles/PMC7050630/)
- [IUPHAR guidance on GPCR ligand bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC7612872/)
- [Computational mechanics and causal-state equivalence](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.html)

The consequence is deliberately conservative: a collision produces either a
typed request to refine the declared state or an explicit equivalence
abstention. It does not manufacture a causal-state certificate from matching
proxies, and it does not make an octonionic or clinical assertion.

### Semantic Lane Declaration

```text
Semantic-Lane-ID: EPISTEMIC-STATE-ALIASING-RECEIPTS-V0-20260724
Owner: Codex
Concept-IDs: SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-NONASSOCIATIVE-ORDER; SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: an observed scalar, occupancy proxy, or summary must not silently erase distinct declared histories before their task-relative interchangeability is tested
Transformation: add private-constructor nominal receipts from source-aligned observed-proxy collision through predeclared synthetic predictive challenge, separation, state-refinement request, or explicit equivalence abstention
Types-Changed: added ObservedProxyI64, ObservedProxyCollisionI64, PredictiveChallengeI64, PredictiveSeparationI64, StateRefinementRequestI64, and StateEquivalenceAbstentionI64
Effects-Changed: none
IR-Changed: none
Claims-Introduced: imported callers cannot silently substitute a matching observed proxy or a no-separation abstention for a declared predictive separation; a separation can request state refinement without becoming state, causal, or clinical equivalence
Claims-Forbidden: latent-state identity, predictive accuracy, empirical state sufficiency, biological mechanism, treatment effect, causal effect, transportability, clinical utility, or clinical authority
Assumptions: tags describe a synthetic fixture and a declared comparator protocol; they do not validate its measurement map, output semantics, or generalization
Write-Set: stdlib/epistemic/state_aliasing_receipts.sio; tests/run-pass/epistemic_state_aliasing_receipts_import_smoke.sio; tests/compile-fail/epistemic_state_aliasing_*.sio; scripts/ci/epistemic_receipt_source_fresh_gate.sh; docs/internal/concepts/nonassociative-order.md; docs/internal/concepts/bindings.tsv
Read-Set: stdlib/epistemic/observation_provenance.sio; stdlib/epistemic/parenthesization_receipts.sio; docs/internal/concepts/ordered-path-provenance.md; docs/research/psychiatric_nonassociativity_representation_contract_2026-07-21.md
Positive-Witness: imported synthetic smoke constructs two source-aligned histories with one proxy tag, a predeclared challenge, unequal synthetic outputs, refinement request, and a separate abstention
Negative-Witness: imported callers cannot fabricate an observed proxy; collision != separation; abstention != separation; refinement request != clinical authority
Acceptance-Gate: bin/souc check stdlib/epistemic/state_aliasing_receipts.sio; scripts/run_sio_test_suite.sh --test-list /tmp/sounio-state-aliasing-receipts-20260724.list --jobs 1 --verbose; bash scripts/ci/epistemic_receipt_source_fresh_gate.sh
Integration-Target: source-fresh imported psychiatric collision suite after #901 acceptance
Authoritative-Only-If: these receipts prove only the selected nominal program boundaries; a source-fresh raw receipt proves the selected imports ran without wrapper fallback, never a scientific or clinical claim
```

### Integration Receipt

```text
Semantic-Outcome: the library now keeps observed-proxy equality distinct from task-relative predictive separation and state-refinement pressure
Concept-Status-Before: provenance receipts could retain distinct histories, but no nominal bridge prevented a shared synthetic proxy from being treated as a future-state conclusion
Concept-Status-After: imported callers can make the collision explicit and construct a separation only through a predeclared challenge; no equivalence certificate is constructible from the proxy
Distinctions-Added: observed proxy equality != history equivalence; collision != predictive separation; predictive separation != latent-state identity; refinement request != causal or clinical authority; abstention != equality proof
Distinctions-Preserved: ordered history, source-system provenance, parenthesization contest, nonassociative carrier hypothesis, and explicit abstention remain separate
Distinctions-Erased: none
Evidence-Run: focused library check, imported positive smoke, compile-fail substitutions, documentation consistency, and structural source-bound gate; wrapper results remain qualified until Foundry receipt
Fallback-Path: default wrapper may execute the imports with source=fallback; it is retained only as qualified API evidence, not current-source runtime evidence
Legacy-Kept: observation provenance and parenthesization receipts remain unchanged; no latent-state, causal, or clinical type is introduced
Conflicting-Lanes: #901 owns current-source imported-native repair and generated governance metadata; this lane modifies neither compiler source nor generated governance files
Next-Semantic-Interface: after source-fresh receipt, preregister an empirical representation contest in which bag, refined ordered-state, capacity-matched associative, and bracket-sensitive models share observation provenance and external validation; require a triple-composition probe, associative-projection control, and bracket-null control so a dormant or associatively collapsed carrier cannot be promoted; a bracket-sensitive carrier remains only a candidate if it beats the refined ordered-state and associative controls
```
