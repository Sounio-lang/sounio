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
it evaluates the existing octonion basis-triple associator and requires a
nonzero norm before recording the selected basis tags. It may establish a
nonzero result for that declared carrier only. It does not bind the carrier to
a biological process, an empirical representation, a psychiatric mechanism,
or the parenthesization-sensitive model contest.

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
Transformation: add private-constructor nominal receipts that require a declared aggregation boundary and contest before a synthetic parenthesization-sensitivity receipt is constructible
Types-Changed: added OrderedTransformationPathI64, PipelineOrderSensitivityI64, AggregationBoundaryI64, BracketingDesignI64, AlgebraicAssociatorProbeI64, BasisTripleAssociatorProbeI64, AssociativityContestI64, BracketDiscriminatingFeatureI64, ParenthesizationSensitivityI64, and ParenthesizationAbstentionI64
Effects-Changed: none
IR-Changed: none
Claims-Introduced: import callers cannot silently substitute an ordered path, a pipeline-order collision, an aggregation boundary, a nominal algebraic probe, or a nonzero basis-triple associator probe for a parenthesization-sensitivity receipt
Claims-Forbidden: physical nonassociativity, biological or psychiatric mechanism, empirical model fit, causal effect, treatment effect, clinical utility, or clinical authority
Assumptions: supplied tags describe a synthetic fixture or declared protocol mapping; they do not validate that mapping
Write-Set: stdlib/epistemic/parenthesization_receipts.sio; tests/run-pass/epistemic_parenthesization_receipts_import_smoke.sio; tests/compile-fail/epistemic_*parenthesization*.sio; docs/internal/concepts/nonassociative-order.md; docs/internal/concepts/bindings.tsv
Read-Set: stdlib/epistemic/observation_provenance.sio; docs/internal/concepts/ordered-path-provenance.md; docs/research/psychiatric_nonassociativity_representation_contract_2026-07-21.md
Positive-Witness: imported synthetic receipt smoke carries history source identity through ordered path, boundary, design, probe, contest, feature, sensitivity, and abstention, while separately checking each shared contest component
Negative-Witness: ordered path != aggregation boundary; pipeline-order sensitivity != parenthesization sensitivity; aggregation boundary != bracketing design; algebraic probe != parenthesization sensitivity; nonzero basis-triple associator probe != parenthesization sensitivity; contest != discriminating feature; imported callers cannot fabricate sensitivity; sensitivity != clinical authority
Acceptance-Gate: bin/souc check stdlib/epistemic/parenthesization_receipts.sio; scripts/run_sio_test_suite.sh --test-list /tmp/sounio-parenthesization-receipts-20260721.list --jobs 1 --verbose
Integration-Target: #901-gated source-fresh imported psychiatric collision suite
Authoritative-Only-If: these receipts prove only the selected nominal program boundaries; no semantic or empirical authority follows from a default-wrapper run
```

### Integration Receipt

```text
Semantic-Outcome: the language now preserves the difference between ordered processing, pipeline-order sensitivity, and a separately declared parenthesization-sensitive research scaffold
Concept-Status-Before: ordered-path provenance and nonassociative algebra were independently represented, with no generic nominal bridge requiring a boundary and contest before sensitivity
Concept-Status-After: imported callers can construct the synthetic sensitivity receipt only after the declared typed chain, while incompatible receipts refuse substitution
Distinctions-Added: ordered path != pipeline-order sensitivity; pipeline-order sensitivity != parenthesization sensitivity; ordered path != aggregation boundary; aggregation boundary != bracketing design; nominal algebraic probe != basis-triple associator result != sensitivity receipt; sensitivity receipt != clinical authority
Distinctions-Preserved: ordinary function composition remains associative; declared nonassociative algebra remains separate from a protocol boundary; observation provenance remains non-clinical
Distinctions-Erased: none
Evidence-Run: library check passed; one imported positive smoke and four compile-fail substitutions passed under the default Madaros wrapper
Fallback-Path: default wrapper reported target-resolution fallback=unresolved_default_x86_64_linux; results are nominal API evidence only, not source-fresh imported-native proof
Legacy-Kept: associator field, ordered-path compiler IR, observation-provenance receipts, and existing research contracts remain unchanged
Conflicting-Lanes: #901 retains ownership of source-fresh imported-native acceptance and target-resolution repair
Next-Semantic-Interface: an import-bearing synthetic collision may proceed only after #901 supplies a current-source no-target-fallback acceptance gate
```
