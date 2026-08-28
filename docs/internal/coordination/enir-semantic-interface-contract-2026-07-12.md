<!-- docs:meta
topic_id: repo.docs.internal.coordination.enir-semantic-interface-contract-2026-07-12
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.enir-semantic-interface-contract-2026-07-12
-->

# ENIR Semantic Interface Contract: E3D

This is the semantic review input for
`work/madaros-v2-e3d-multipred-scalar-memory-ssa-full-codex` at
`ce2f94407`. It does not certify the implementation or its integration base.

```text
Semantic-Lane-ID: ENIR-E3D-MULTIPRED-SCALAR-MEMORY-SSA
Owner: work/madaros-v2-e3d-multipred-scalar-memory-ssa-full-codex
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE, SOUNIO-ZERO-PROVENANCE,
  SOUNIO-EXPLICIT-DISCHARGE, SOUNIO-PRECISION-PRESERVATION
Intent-Preserved: control-flow and memory transformations must not silently
  erase numeric uncertainty, zero evidence, discharge state, or precision
Transformation: represent and lower multipredecessor scalar and memory SSA
  while retaining enough identity to verify semantic joins
Types-Changed: inspect commit; no semantic type promotion authorized here
Effects-Changed: none authorized
IR-Changed: yes; CFG/SSA/MIR surfaces in the E3 lineage
Claims-Introduced: only the exact CFG/SSA behaviors demonstrated by positive
  and negative witnesses
Claims-Forbidden: physical causality, clinical mechanism, automatic
  preservation of epistemic metadata not explicitly represented and tested
Assumptions: predecessor identity is stable; join semantics are explicit;
  memory versions are not treated as observational truth
Write-Set: inspect commit ce2f94407 before integration
Read-Set: self-hosted/enir, self-hosted/mir, concept registry
Positive-Witness: required from lane receipt
Negative-Witness: malformed or semantically ambiguous join must fail closed
Acceptance-Gate: lane gate plus current integration-base compiler gate
Integration-Target: successor of current origin/main selected after topology review
Authoritative-Only-If: roundtrip/verifier/lowering witnesses pass without
  silent fallback and the resulting commit is contained in the selected target
```

## Required Review Questions

1. Does a multipredecessor join preserve the distinction among value,
   arithmetic error, uncertainty, and ignorance?
2. Is computational provenance retained as computational provenance, without
   being relabeled as physical causality?
3. Can an IEEE zero class erase or stand in for zero provenance? It must not.
4. Is any `f128`/`f256` or epistemic value narrowed during lowering? Any such
   conversion requires an explicit, inspectable discharge.
5. Do parser, printer, verifier, hash/equality, and lowering agree on every new
   semantic field?
6. Does an ambiguous join fail closed, and is the negative witness executable?

## Schema Extension Rule

Do not add speculative provenance or observation fields merely to make the IR
look future-ready. A new field becomes authoritative only with:

- a registered concept and exact meaning;
- parser/printer or builder coverage as applicable;
- verifier invariants;
- equality/hash/roundtrip behavior;
- positive and negative witnesses; and
- a named discharge when information can be erased.

`SOUNIO-PHYSICAL-OBSERVATION` is read-only context for this lane. E3D may leave
an interface for future observation receipts, but it must not claim that an SSA
edge, memory version, or provenance tag is a physical observation.

## Review Disposition

Current disposition: `IMPLEMENTATION_VERIFIED_IN_LANE_TOPOLOGY_UNRESOLVED`.

The branch is clean and remote-synchronized. Its focused gate passed at
`ce2f94407` with two paths, 12 source negatives, 72 artifact tampers, 60 runtime
tampers, independent replay, and receipt SHA-256
`78e5ffbbb2c302f75bff271a0757c9e6e06a3ffa518068b5b7ba0fd77530b6e0`.

The implementation preserves both possible histories in the static scalar and
memory phi descriptors. At runtime it records the selected edge and incoming
value/version, then copies the selected runtime value into the join result. The
structural encounter is therefore explicit and receipted, while the resulting
numeric value carries the realized history rather than a new provenance object
representing the encounter itself.

This is not classified as a defect. It is the exact boundary for a successor
witness: a `JoinReceipt` may distinguish the realized input, the declared
alternatives, and the selection rule without pretending that an unexecuted path
was physically observed. Alias analysis remains deferred, machine IR is unused,
and codegen is unchanged.

That successor witness is integrated by PR `#804` at merge commit
`571f3bfce` on `canon/madaros-v2-sota`. Two paths produce the bit-identical
observable value `202.0` while their selected edge, scalar incoming value,
Memory SSA incoming versions, and execution receipt SHA-256 remain distinct.
The witness claims computational event distinction only; it does not claim
physical causality or observation of the unexecuted alternative.

The E3D and E3E tips are intentionally contained in
`origin/canon/madaros-v2-sota`, not `origin/main`. Their integration topology is
resolved for the Madaros v2 canon; promotion from canon to `main` remains a
separate release decision.
