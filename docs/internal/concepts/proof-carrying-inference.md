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

## SYNC-004 Worldline Merge Advice Boundary

```text
Semantic-Lane-ID: sync004-worldline-epistemic-abi
Owner: codex-root
Concept-IDs: SOUNIO-PROOF-CARRYING-INFERENCE
Intent-Preserved: preserve contradictions and missing evidence without giving a model authority over personal continuity
Transformation: verified conflict metadata becomes a digest-only epistemic analysis artifact
Types-Changed: none; a new library receipt and stable C ABI are additive
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio can classify six evidence categories and emit byte-exact EpistemicMergeAnalysis v1 CBOR
Claims-Forbidden: Sounio selected a branch, resolved truth, authorized a merge, observed protected plaintext, or mutated a ledger
Assumptions: conflict frames, membership, signatures, domains, and causal completeness were already verified by the UNEQSELF runtime
Write-Set: the new worldline advice module, focused witness, versioned FFI crate, and selftest
Read-Set: UNEQSELF worldline-merge-v1 protocol and public analysis vector
Positive-Witness: the FFI emits the exact 420-byte public analysis vector and the Sounio receipt keeps all six categories distinct
Negative-Witness: malformed pointers fail closed; exported ABI contains no decide, commit, select, mutate, or authorization symbol
Acceptance-Gate: scripts/ci/sounio_worldline_merge_epistemic_selftest.sh
Integration-Target: UNEQSELF Swift runtime EpistemicMergeAnalysis decoder
Authoritative-Only-If: never; a current authorize_merge member must sign a separate MergeDecision
```

The additive surfaces are:

- `stdlib/epistemic/worldline_merge_advice.sio`
- `tools/uneqself-worldline-merge-ffi/`
- `tests/run-pass/worldline_merge_advice.sio`
- `scripts/ci/sounio_worldline_merge_epistemic_selftest.sh`

This is an instance of proof-carrying inference, not a new truth predicate.
The six lists preserve observations, inferences, contradictions, missing
evidence, open obligations, and alternatives as different categories. An empty
or unavailable category is not silently promoted into negative evidence.

### Semantic Outcome Receipt

- Status before: the UNEQSELF protocol had a public, digest-only merge-analysis
  vector, but Sounio had no executable receipt type or ABI for producing it.
- Status after: an additive Sounio receipt and one-function C ABI reproduce the
  canonical 420-byte `EpistemicMergeAnalysis` artifact without reading protected
  plaintext or acquiring decision authority.
- Distinctions added: the receipt keeps observations, inferences,
  contradictions, missing evidence, open obligations, and alternatives in six
  independent digest categories.
- Distinctions preserved: missing evidence is not negative evidence; analysis
  is not selection; synthesis is not authorization; a model output is not a
  ledger mutation.
- Distinctions erased: none.
- Positive evidence: the focused Sounio run-pass witness checks complete,
  deferred, and invalid inventories; the Rust unit test reproduces the public
  vector byte for byte and verifies its artifact digest.
- Negative evidence: null or malformed ABI inputs fail closed, and the release
  library exports exactly `uneqself_worldline_merge_analyze_v1` with no decide,
  commit, select, mutate, or authorization symbol.
- Validation route: `bin/souc check` and the focused `.sio` suite ran in the
  canonical remote worktree. Rust `fmt`, `clippy -D warnings`, tests, release
  build, and symbol inspection ran from an isolated copy on `t560-proxmox`
  because the workspace pod has no active Rust toolchain.
- Legacy behavior: preserved; all surfaces are additive.
- Known semantic conflicts: none within this lane's declared write set.
- Next interface: Swift loads the versioned ABI output and still requires a
  separately authenticated `authorize_merge` signer for `MergeDecision`.
