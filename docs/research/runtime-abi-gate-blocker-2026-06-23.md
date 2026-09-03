<!-- docs:meta
topic_id: repo.docs.research.runtime-abi-gate-blocker-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.runtime-abi-gate-blocker-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Runtime ABI Blocker Gate

Date: 2026-06-23

This note records a runtime ABI evidence gate for solver and proof-checker
receipts. The gate exists because the current compiler can typecheck imported
solver and Lorenz receipt APIs, while imported native runtime evidence remains
blocked by the Madaros multimodule/native ABI failure.

This gate contains no solver output and no Lorenz runtime evidence. Numeric IDs
named here are non-substantive traceability tokens only. They are not outputs of
this gate and are not reasserted as solver proof, replay execution, replay
verification, or runtime proof-checker evidence.

## Gate Record

- Module: `stdlib/safety/runtime_abi_gate_blocker.sio`
- Tiny runtime test: `tests/run-pass/runtime_abi_gate_blocker_tiny.sio`
- Imported smoke test: `tests/run-pass/runtime_abi_gate_blocker_imported.sio`
- Artifact fingerprint: `584291376`
- Audit fingerprint: `936740152`
- Instance fingerprint: `265813904`
- Certificate fingerprint: `718406529`
- Status code: `92`

## Non-Substantive Traceability Tokens

- Prior profile token pair: `964210753` / `526184309`
- Prior preflight token pair: `391742608` / `650219347`
- Solver domain: `4` (Sounio numeric receipt)
- Proof format: `6` (Lorenz i128/i256 numeric receipt)
- Profile status: `88`
- Verifier preflight status: `91`

## Local Non-Portable Gate Observations

- `frontend_typecheck_pass_mask = 3` (local environment state, not a portable claim)
- `self_contained_runtime_pass_mask = 1` (local environment state, not a portable claim)
- `multimodule_witness_pass_mask = 0`
- `wide_call_runtime_pass_mask = 0`
- `imported_solver_runtime_pass_mask = 0`
- `known_blocker_mask = 7`
- `runtime_evidence_level = 2`
- `imported_runtime_promotion_mask = 0`
- `runtime_abi_anchor_mask = 63`
- `frontend_contract_mask = 3`
- `self_contained_evidence_mask = 1`
- `imported_runtime_missing_mask = 7`
- `known_madaros_runtime_blocker_mask = 7`
- `runtime_evidence_private_level = 2`
- `imported_promotion_blocked_mask = 0`
- `public_theorem_ready_mask = 0`
- `global_promotion_ready_mask = 0`
- `runtime_abi_next_action_mask = 31`

Here, `known_blocker_mask = 7` means three independent blockers are active; it
does not encode positive evidence. `runtime_evidence_level = 2` means the ABI
gate is present but unsatisfied; it is not a proof-strength tier.

## Observed Gate Behavior

Current local observations still block imported runtime promotion. These are
expected blocker observations, not evidence of solver or Lorenz verification:

- `bash scripts/ci/madaros_multimodule_witness.sh` fails on
  `thin_single expected_exit=7 actual_exit=139`.
- `bash scripts/ci/native_v2_imported_core_abi_gate.sh` did not reach runtime
  acceptance in this worktree; the imported IR summary path reported frontend
  errors against the current compiler/stdlib surface.
- `bash scripts/ci/native_v2_imported_body_lowering_gate.sh` likewise did not
  reach runtime acceptance in this worktree.

The gate therefore treats imported solver tests as frontend/typecheck contracts
only until the multimodule and imported ABI gates pass on the selected compiler
artifact.

## Lifting Criteria

This blocker can be lifted only after current-state evidence shows all of the
following under the selected compiler artifact:

- `multimodule_witness_pass_mask >= 1`
- `wide_call_runtime_pass_mask >= 1`
- `imported_solver_runtime_pass_mask >= 1`
- `known_blocker_mask = 0`
- an imported solver or Lorenz receipt runs at native runtime, not merely in
  frontend/typecheck mode

No solver evidence or Lorenz runtime evidence is present in this gate.

Until then, the gate must remain in the blocking/no-acceptance state.

## Claim Boundary

This gate preserves these nonclaims:

- `formal_theorem_ready = 0`
- `public_claim_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`

This is not a SAT, SMT, PB, or Lorenz proof checker result; not imported runtime
evidence; not replay execution; not replay verification; not a Hadwiger-Nelson
result; not a public theorem promotion; not boundary gluing; not a finite-cover
certificate; and not a global Lorenz flowpipe theorem. Its job is to prevent
imported solver API typechecks from being over-read as runtime proof-checker
evidence while the ABI blocker is still present.
