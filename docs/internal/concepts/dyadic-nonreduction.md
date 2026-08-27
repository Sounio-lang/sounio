<!-- docs:meta
topic_id: repo.docs.internal.concepts.dyadic-nonreduction
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.dyadic-nonreduction
-->

# Dyadic Non-Reduction


Status: **executable**

Concept-ID: `SOUNIO-DYADIC-NONREDUCTION`

## Founder Intent

A declared participant pair must not silently replace relational state when
retained relational history changes bounded predictions under a common probe.

## Executable Core

The D0 fixture constructs two synthetic candidates with equal declared current
participant, relational, and context projections but distinct retained-history
predictive modes. A common input produces an exact nonzero separation.

## Current Surfaces

- `stdlib/epistemic/dyadic_non_reduction.sio`
- `tests/run-pass/clinical_dyadic_non_reduction_native_witness.sio`
- `scripts/research/dyadic_non_reduction_oracle.py`
- `scripts/ci/dyadic_non_reduction_gate.sh`

## Required Invariants

- Candidate and history identifiers are not observable features.
- Every comparison uses a declared common input and exact arithmetic.
- A promoted predictive mode may restore finite Markov factorability.
- The result is participant-product non-reduction, not unbounded-history
  irreducibility.

## Claims Forbidden

- A bounded synthetic collision identifies a real relationship.
- The witness establishes subjective experience, suffering, consent, a causal
  mechanism, or clinical authority.
