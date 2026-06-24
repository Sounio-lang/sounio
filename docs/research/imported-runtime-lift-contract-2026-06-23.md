<!-- docs:meta
topic_id: repo.docs.research.imported-runtime-lift-contract-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.imported-runtime-lift-contract-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Imported Runtime Lift Contract

Date: 2026-06-23

This note records an imported-runtime lift contract over the private evidence
envelope, runtime ABI blocker, and kernel replay evidence router. The contract
does not lift the blocker. It records the current blocked state and names the
future evidence required before solver/proof-checker receipts can be promoted
from private/local bookkeeping to imported/native runtime evidence.

This contract contains no solver output and no Lorenz runtime evidence. Numeric
IDs named here are non-substantive traceability tokens only. They are not
outputs of this contract and are not reasserted as solver proof, replay
execution, replay verification, imported runtime evidence, finite-cover
evidence, boundary-gluing evidence, or theorem evidence.

## Gate Record

- Module: `stdlib/safety/imported_runtime_lift_contract.sio`
- Tiny runtime test: `tests/run-pass/imported_runtime_lift_contract_tiny.sio`
- Imported smoke test: `tests/run-pass/imported_runtime_lift_contract_imported.sio`
- Artifact fingerprint: `642810357`
- Audit fingerprint: `508176294`
- Instance fingerprint: `276304915`
- Certificate fingerprint: `819273604`
- Status code: `95`

## Non-Substantive Traceability Tokens

- Private evidence envelope token pair: `834620917` / `276591483`
- Runtime ABI blocker token pair: `584291376` / `936740152`
- Kernel replay router token pair: `709284613` / `418905276`
- Private evidence envelope status: `94`
- Runtime ABI blocker status: `92`
- Kernel replay router status: `93`

## Current Blocked Observations

- `multimodule_witness_pass_mask = 0`
- `wide_call_runtime_pass_mask = 0`
- `imported_solver_runtime_pass_mask = 0`
- `imported_kernel_replay_pass_mask = 0`
- `known_blocker_mask = 7`
- `future_lift_requirement_mask = 31`
- `current_missing_lift_mask = 31`
- `ready_to_lift_mask = 0`
- `imported_runtime_promotion_mask = 0`
- `portable_runtime_evidence_mask = 0`
- `lift_contract_next_action_mask = 31`
- `lift_anchor_mask = 63`
- `ok_mask = 1023`

Here, `future_lift_requirement_mask = 31` names five future requirements:
multimodule witness pass, wide-call runtime pass, imported solver runtime pass,
imported kernel replay pass, and a cleared known-blocker mask. The current
contract requires `current_missing_lift_mask = 31` and `ready_to_lift_mask = 0`
because none of those requirements is satisfied here.

## Lifting Boundary

This contract can be replaced by a positive lift only after current-state
evidence under the selected compiler artifact shows all of the following:

- `multimodule_witness_pass_mask >= 1`
- `wide_call_runtime_pass_mask >= 1`
- `imported_solver_runtime_pass_mask >= 1`
- `imported_kernel_replay_pass_mask >= 1`
- `known_blocker_mask = 0`
- `ready_to_lift_mask >= 1`
- an imported solver or replay receipt executes at native runtime, not merely
  in frontend/typecheck mode

Until then, the imported runtime promotion and portable runtime evidence masks
must remain zero.

## Claim Boundary

This contract preserves these nonclaims:

- `public_claim_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `finite_cover_certificate_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `formal_theorem_ready = 0`

This is not a SAT, SMT, PB, or Lorenz proof checker result; not imported runtime
evidence; not replay execution; not replay verification; not a Hadwiger-Nelson
result; not a public theorem promotion; not boundary gluing; not a finite-cover
certificate; and not a global Lorenz flowpipe theorem. Its job is to keep the
path from private solver/proof-checker bookkeeping to imported/native runtime
evidence explicit and blocked until the runtime evidence actually exists.
