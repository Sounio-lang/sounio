<!-- docs:meta
topic_id: repo.docs.internal.concepts.second-order-compilation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.second-order-compilation
-->

# Second-Order Compilation

Concept-ID: `SOUNIO-SECOND-ORDER-COMPILATION`

## Founder Intent

The compiler is a scientific instrument and therefore part of the experiment.
It must expose when its own transformations, representations, observations, or
fallbacks preserve, alter, or erase information relevant to the scientific
question.

## Canonical Distinctions

```text
source intent             != compiler intervention
requested semantics       != realised semantics
artifact                  != observation
no observed divergence    != semantic equivalence
unaligned traces          != equal traces
compiler-caused residual  != physical effect
instrumentation           != neutral observation by default
blocked observation       != observed zero
```

## Current Surfaces

- `docs/architecture/second-order-compilation.md`: normative C2-v0 contract.
- `docs/decisions/adr-007-second-order-compilation.md`: decision record.
- existing EISA dd64/qd128 execution profiles: first planned witness surface.
- existing epistemic, precision, zero-provenance, and non-associative concept
  contracts: composed semantic dependencies.

## Required Invariants

- Every C2 comparison pins or declares all intervention dimensions.
- Requested and realised semantic profiles are recorded separately.
- First divergence requires aligned operation identities.
- Blind spots and fallback paths are explicit receipt fields.
- Observation instrumentation is declared when non-interference is unproven.
- Receipt status cannot collapse blocked, unaligned, incomparable, equivalent
  within projection, and divergent outcomes into one Boolean.
- Numerical divergence cannot be promoted directly to a physical or clinical
  claim.

## Evidence Status

Status: `hypothesis`

The semantic architecture and first acceptance boundary are specified. No C2
first-divergence receipt has yet passed the required positive, negative,
tamper, and blind-spot cases.

## Pending Interface

`first-divergence-receipt`

## Permitted Claims

- Sounio has adopted an experimental semantic contract for observing
  compiler interventions.
- The first executable C2 witness has a bounded acceptance rule.

## Forbidden Claims

- C2 is already integrated through every Madaros stage.
- Current receipts prove physical or clinical causality.
- Current dd64/qd128 surfaces are native IEEE `f128`/`f256`.
- Lack of observed divergence proves semantic equivalence.
- Madaros is conscious or performs human self-observation.

## Founding Semantic Lane

```text
Semantic-Lane-ID: C2-V0-CONSTITUTION-20260712
Owner: Codex root
Concept-IDs: SOUNIO-SECOND-ORDER-COMPILATION
Intent-Preserved: compiler-caused scientific information loss must be observable and must not be reported as physical absence or semantic equivalence
Transformation: introduce the experimental C2 semantic contract and bounded first-witness decision rule
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: an experimental second-order compilation contract exists with a named first acceptance boundary
Claims-Forbidden: compiler-wide integration, native f128/f256/f512, physical mechanism, clinical effect, or novelty superiority
Assumptions: existing concept contracts remain authoritative and are composed without redefinition
Write-Set: docs/architecture/second-order-compilation.md; docs/decisions/adr-007-second-order-compilation.md; docs/decisions/README.md; docs/internal/concepts/README.md; docs/internal/concepts/second-order-compilation.md; docs/internal/concepts/registry.tsv; docs/internal/concepts/bindings.tsv; generated docs governance metadata
Read-Set: FOUNDER_INTENT.md; AGENTS.md; CLAUDE.md; docs/internal/concepts/*; docs/decisions/*; docs/architecture/truth-layers.md
Positive-Witness: documentation and semantic registry gates accept the C2 contract
Negative-Witness: the concept remains hypothesis and forbidden claims remain explicit until an executable receipt exists
Acceptance-Gate: bash scripts/dev/check_docs_registry.sh; bash scripts/dev/check_docs_consistency.sh; bash scripts/ci/founder_intent_contract_gate.sh
Integration-Target: origin/main after review
Authoritative-Only-If: all acceptance gates pass and an orthogonal semantic review finds no silent widening of claims
```

## Integration Receipt

```text
Semantic-Outcome: experimental C2-v0 contract registered; executable first-divergence receipt remains absent
Concept-Status-Before: none
Concept-Status-After: hypothesis
Distinctions-Added: source intent versus compiler intervention; requested versus realised semantics; bounded observation versus equivalence; unaligned versus equal; instrumentation versus neutral observation
Distinctions-Preserved: value versus arithmetic error; arithmetic error versus uncertainty; computational provenance versus physical causality; compile success versus runtime parity
Distinctions-Erased: none
Evidence-Run: docs registry PASS; docs registry selftest PASS; docs consistency PASS; founder intent contract PASS; xAI/Grok 4.3 adversarial review findings disposition recorded
Fallback-Path: none
Legacy-Kept: all compiler, EISA, precision, provenance, and receipt paths remain unchanged
Conflicting-Lanes: none with recent semantic writers; repository scanner reports historical overlaps outside this documentation-only write-set
Next-Semantic-Interface: first-divergence-receipt
```
