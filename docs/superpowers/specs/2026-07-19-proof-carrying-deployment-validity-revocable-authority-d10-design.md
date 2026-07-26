<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-19-proof-carrying-deployment-validity-revocable-authority-d10-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-19-proof-carrying-deployment-validity-revocable-authority-d10-design
-->

> Metadata note: `last_validated` is generated from the repository governance
> baseline. D10 evidence and review receipts are dated 2026-07-19.

# D10 Proof-Carrying Deployment Validity And Revocable Authority Design

## Semantic Lane Declaration

```text
Semantic-Lane-ID: PSYCHIATRIC-D10-DEPLOYMENT-VALIDITY-20260719
Owner: codex-3
Concept-IDs: SOUNIO-PROOF-CARRYING-DEPLOYMENT-VALIDITY-REVOCABLE-AUTHORITY
Intent-Preserved: evidence, order, context, provenance, time, and authority may not be silently erased
Transformation: add a bounded warrant-carrying deployment typestate above unchanged D9
Types-Changed: new D10-only nominal observation, refusal, private typestate, lease, and reserved authority types
Effects-Changed: none outside effects declared by new D10 functions
IR-Changed: none
Claims-Introduced: exact frozen collision arithmetic and nominal non-promotion for a synthetic canary lease
Claims-Forbidden: external validation, production permission, clinical authority, affine consumption, live revocation, general anytime validity, novelty
Assumptions: frozen finite fixtures, exact integer arithmetic, canonical Madaros, private constructor enforcement
Write-Set: D10 kernel, ontology, witnesses, oracle, negatives, gate, concept/docs/registry bindings, offload log
Read-Set: D9-D0 kernels and gates, semantic and blocker contracts, current literature and regulatory sources
Positive-Witness: standalone exact runtime plus imported private-token flow check
Negative-Witness: clinical and ontology compile-fail matrices plus private-constructor refusals
Acceptance-Gate: scripts/ci/proof_carrying_deployment_validity_revocable_authority_gate.sh
Integration-Target: codex/psychiatric-d9-statistical-binding-20260719
Authoritative-Only-If: canonical Madaros, independent oracle agreement, exact negatives, dual ontology paths, recursive D9-D0 green, mandatory reviews
```

## Architecture

The module has three layers:

1. Public observations and refusal receipts. These are inspectable but
   non-authoritative.
2. Private typestate tokens. Imported callers can pipe them by inference but
   cannot construct their literals.
3. Reserved private authorities with acceptors and no producers.

The maximum producer issues a `D10FixtureCanaryLeaseToken`. A lease observation
can be marked live or revoked within the finite epoch simulation. No function
returns production or clinical authority.

This is classic nominal typestate, not indexed, dependent, affine, or linear
typing. Any caller can invoke the frozen fixture producers, so the private wall
prevents literal construction but does not authenticate the caller or the
external origin of evidence. Spending and revocation are immutable local
transition traces, not a shared ledger or live reference monitor.

## Required Gates

- exact Brier, two-look, e-process, checked deferral, direct change-conformance,
  local spend-transition, and epoch arithmetic in both Sounio and Python;
- private token flow through an imported check-only witness;
- exact expected/found diagnostics for every nominal wall;
- E176 for direct construction of reserved private authorities;
- default and rebuilt current-source ontology validation;
- recursive D9 gate and therefore D8-D0;
- xAI and Z.AI math review plus hostile clinical-authority review.

## Compiler Boundary

D10 changes no compiler, resolver, IR, or earlier semantic file. Imported
execution remains check-only under `BLK-20260718-D6-MULTIMODULE-RUNTIME`.
Static one-use and live revocation require a future compiler/formal lane owned
outside D10.
