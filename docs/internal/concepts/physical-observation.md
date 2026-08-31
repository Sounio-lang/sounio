<!-- docs:meta
topic_id: repo.docs.internal.concepts.physical-observation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.physical-observation
-->

# Physical Observation

Concept-ID: `SOUNIO-PHYSICAL-OBSERVATION`

Status: hypothesis.

## Founder Intent

An observed value belongs to an encounter among system, environment,
instrument, protocol, projection, and scale. It must not be represented as an
isolated system property when those contexts affect interpretation.

## Proposed Relation

```text
PhysicalState x Environment -> Interaction
Interaction x Dynamics x BoundaryConditions -> Evolution
Evolution x Projection x Instrument x Protocol -> PhysicalObservationReceipt
```

## Candidate Receipt Axes

- predicted and observed epistemic values;
- residual and declared resolution;
- frame and coordinate convention;
- environment and boundary-condition identity;
- instrument, calibration, and protocol identity;
- projection and discarded components;
- approximation and conservation receipts.

## Existing Inputs

- `stdlib/physics/`
- `stdlib/particle_physics/`
- `stdlib/quantum/`
- `stdlib/metrology/`
- `SOUNIO-ZERO-PROVENANCE`
- `SOUNIO-EPISTEMIC-NUMERIC-VALUE`

## Non-Goals

- No universal runtime type has been selected.
- Similar structure across domains does not establish a shared causal mechanism.
- No particular quantum interpretation is assumed.

## Promotion Rule

Remain a hypothesis until a narrow physical witness shows that a typed receipt
distinguishes experimentally meaningful cases collapsed by a scalar interface.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
