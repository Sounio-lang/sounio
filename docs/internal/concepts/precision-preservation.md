<!-- docs:meta
topic_id: repo.docs.internal.concepts.precision-preservation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.precision-preservation
-->

# Precision Preservation


Status: **executable**

Concept-ID: `SOUNIO-PRECISION-PRESERVATION`

## Founder Intent

Small, residual, or cancellation-sensitive effects must not disappear because
an implementation silently narrows the representation. `f128`, `f256`,
double-double, and quad-double paths are scientific surfaces.

## Current Surfaces

- `stdlib/math/dd64.sio`
- `stdlib/math/qd128.sio`
- EISA error lanes and qd bridges
- ENIR error kinds `dd64` and `qd128`
- native-v2 aggregate and arithmetic lowering

## Required Invariants

- Narrowing is explicit or proven lossless for the stated contract.
- Equality after narrowing does not imply equal correction histories.
- Tests include adversarial cancellation and residual witnesses.
- Backend failure is never silent fallback to `f64`.
- Higher precision alone does not establish physical significance.

## Current Frontier

`dd64` has a native passing control. The qd128/EISA graph lowers without the
former segmentation fault but native emission still fails closed on classified
paths. This is a compiler frontier, not permission to demote precision.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
