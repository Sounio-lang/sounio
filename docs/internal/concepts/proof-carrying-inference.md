<!-- docs:meta
topic_id: repo.docs.internal.concepts.proof-carrying-inference
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.proof-carrying-inference
-->

# Proof-Carrying Inference


Status: **executable**

Concept-ID: `SOUNIO-PROOF-CARRYING-INFERENCE`

## Founder Intent

Scientific software must preserve what is known, how it became admissible,
which alternatives remain, and which stronger conclusions are still forbidden.
An inference result is therefore a typed evidence path, not only a scalar or a
selected model.

## Executable Core

For a declared finite hypothesis family, an admissible observation may remove
exactly those surviving hypotheses whose frozen predictions disagree with it.
The transition retains the observation, schema, provenance, probe burden, and
elimination set. Missing or unaudited observations cannot enter that update.

The D2 fixture additionally provides:

- an exact four-hypothesis version space;
- a bounded minimax adaptive probe policy;
- typed abstention when evidence is unavailable or inadmissible;
- declared-family refutation when no hypothesis survives;
- finite-family identification when exactly one hypothesis survives.

## Ontology Binding

`stdlib/ontology/proof_carrying_inference.sio` distinguishes observations,
provenance, protocol inputs, inference receipts, declared-family conclusions,
causal claims, suffering claims, and clinical authority.

This binding is currently a parallel nominal boundary. The ontology module and
its negative witnesses independently re-express the kernel's distinctions, but
a runtime D2 receipt is not yet transported as an ontology-typed result. Such a
result-identity bridge requires its own source-to-IR evidence and is not implied
by the focused ontology checks.

## Required Invariants

- Missing or provenance-disconnected observation is not negative evidence.
- Provenance is checked before model elimination, including an exact link to
  the last accepted provenance identifier.
- Every surviving or eliminated hypothesis is recomputed from the same
  observation and prediction table.
- Each transition carries the exact provenance edge. Its bounded base-31
  fingerprint is an audit convenience, not a collision-free proof.
- State and transition consumers replay all receipt invariants because current
  Madaros cross-module field visibility is not a sealing boundary.
- Empty survival refutes the declared family; it does not refute reality.
- Singleton survival identifies only within the declared family.
- Synthetic probe burden is not subjective suffering or measured harm.
- Predictive separation is not an intervention or causal mechanism.
- Abstention is a successful typed outcome, not a compiler failure.

## Claims Forbidden

- The declared hypothesis family is globally complete.
- A surviving model is true, causal, diagnostic, or clinically actionable.
- Missingness is ignorable without an explicit missingness assumption.
- The bounded probe policy minimizes real patient burden or suffering.
- This concept is the first epistemic programming or provenance system.
- Receipts are cryptographically authentic or unforgeable under hostile code.

## Current Surfaces

- `stdlib/epistemic/proof_carrying_model_contest.sio`
- `stdlib/ontology/proof_carrying_inference.sio`
- `scripts/research/proof_carrying_model_contest_oracle.py`
- `scripts/ci/proof_carrying_model_contest_gate.sh`
