<!-- docs:meta
topic_id: repo.docs.internal.concepts.zero-provenance
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.zero-provenance
-->

# Zero Provenance


Status: **executable**

Concept-ID: `SOUNIO-ZERO-PROVENANCE`

## Founder Intent

Equal surface zeros must not imply equal events. A scalar zero must not erase
the computational evidence that produced it before an explicit decision to
forget that evidence.

## Canonical Distinctions

- absent;
- cancelled;
- annihilated by nonzero factors;
- below declared resolution;
- rounded with a nonzero correction trail;
- gated by policy; and
- unknown.

## Distinct From

- IEEE floating-point class;
- missing data;
- statistical non-significance;
- physical or clinical causal explanation; and
- poisoned or trapped execution status.

## Authoritative Surface

- `stdlib/epistemic/zero_event.sio`
- `scripts/ci/zero_event_native_compile_privacy_gate.sh`
- `scripts/ci/zero_event_gate.sh`
- `scripts/ci/zero_event_native_v2_matrix.sh`

## Required Invariants

- Constructors remain opaque outside the defining module.
- Erasure to `f64` requires explicit typed discharge.
- EISA flags are derived views and do not replace `val`, `err`, or `u`.
- ENIR `fp_class` must not substitute for zero evidence.

## Pending Interface

Map evidence into ENIR provenance/status without claiming causal provenance or
silently changing the taxonomy.

## Claims Forbidden

- A zero receipt establishes a biological, clinical, or physical mechanism.
- Every scalar zero requires this runtime representation.
- `InfinityEvent` is implied by this concept.
