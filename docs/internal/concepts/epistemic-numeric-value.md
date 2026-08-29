<!-- docs:meta
topic_id: repo.docs.internal.concepts.epistemic-numeric-value
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.epistemic-numeric-value
-->

# Epistemic Numeric Value


Status: **executable**

Concept-ID: `SOUNIO-EPISTEMIC-NUMERIC-VALUE`

## Founder Intent

A computed number is not adequately represented by its value lane alone. The
language must preserve quantified error, uncertainty, admissibility status,
provenance, and policy wherever those distinctions affect interpretation.

## Canonical Axes

```text
value       numerical estimate or representation
error       arithmetic correction or bound
uncertainty metrological or statistical uncertainty model
status      clean, poisoned, trapped, or unsupported
provenance  computational origin and transformation history
policy      explicit gate or decision rule
```

These axes are related but not interchangeable.

## Current Surfaces

- `stdlib/epistemic/`
- `stdlib/eisa/`
- `self-hosted/enir/`
- source type family `Knowledge[T]`

## Required Invariants

- Optimizations must not silently discard a tracked axis.
- ENIR-to-MIR lowering preserves source identity or emits explicit discharge or
  unsupported status.
- Computational provenance is not physical causality.
- An exact arithmetic correction is not a confidence interval.

## Pending Interface

Bind a physical observation to frame, instrument, projection, protocol, and
resolution without collapsing these axes.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
