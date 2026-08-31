<!-- docs:meta
topic_id: repo.docs.internal.concepts.explicit-discharge
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.explicit-discharge
-->

# Explicit Discharge


Status: **executable**

Concept-ID: `SOUNIO-EXPLICIT-DISCHARGE`

## Founder Intent

Loss of tracked meaning must be an explicit program event. Evidence, precision,
provenance, or status may be forgotten, but not silently at a semantic boundary.

## Canonical Example

`ZeroReceiptF64 -> ErasedZeroF64 -> f64` in
`stdlib/epistemic/zero_event.sio`.

## Required Invariants

- Receipt and discharged representations are distinct types.
- Discharge records a reason or policy tag.
- Extraction before discharge is a compile failure.
- Discharge is an auditable declaration, not proof that forgetting was
  scientifically justified.

## Generalization Boundary

Precision reduction, projection, coarse-graining, measurement, and claim
promotion require separately reviewed discharge protocols. They are not
inferred automatically from the zero receipt.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
