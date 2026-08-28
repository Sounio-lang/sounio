<!-- docs:meta
topic_id: repo.docs.research.kernel-replay-evidence-router-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.kernel-replay-evidence-router-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Kernel Replay Evidence Router

Date: 2026-06-23

This note records a safety-layer router for local kernel-replay evidence. It
keeps self-contained replay/checker/scope observations separate from imported
native runtime evidence while the runtime ABI blocker is active.

This router contains no solver output and no Lorenz runtime evidence. Numeric
IDs named here are non-substantive traceability tokens only. They are not
outputs of this router and are not reasserted as solver proof, replay
verification, imported runtime evidence, or theorem evidence.

## Gate Record

- Module: `stdlib/safety/kernel_replay_evidence_router.sio`
- Tiny runtime test: `tests/run-pass/kernel_replay_evidence_router_tiny.sio`
- Imported smoke test: `tests/run-pass/kernel_replay_evidence_router_imported.sio`
- Artifact fingerprint: `709284613`
- Audit fingerprint: `418905276`
- Instance fingerprint: `627341890`
- Certificate fingerprint: `149286735`
- Status code: `93`

## Non-Substantive Traceability Tokens

- Runtime ABI blocker token pair: `584291376` / `936740152`
- Prior profile token pair: `964210753` / `526184309`
- Prior preflight token pair: `391742608` / `650219347`
- Runtime ABI blocker status: `92`
- Profile status: `88`
- Verifier preflight status: `91`

## Local Non-Portable Router Observations

- `local_kernel_replay_mask = 1` (local only)
- `local_checker_trace_mask = 1` (local only)
- `local_scope_gate_mask = 1` (local only)
- `imported_kernel_replay_mask = 0`
- `portable_runtime_evidence_mask = 0`
- `imported_runtime_promotion_mask = 0`
- `abi_blocker_active_mask = 1`
- `private_acceptance_mask = 1`
- `router_anchor_mask = 63`
- `local_replay_available_mask = 7`
- `imported_replay_missing_mask = 7`
- `abi_blocker_required_mask = 1`
- `private_only_acceptance_level = 2`
- `public_promotion_blocked_mask = 0`
- `global_promotion_blocked_mask = 0`
- `finite_cover_promotion_blocked_mask = 0`
- `boundary_gluing_promotion_blocked_mask = 0`
- `router_next_action_mask = 31`

Here, `local_replay_available_mask = 7` means only local self-contained replay,
checker trace, and scope-gate observations are present. It does not encode
portable imported runtime evidence. The imported replay and promotion masks
must remain zero while the runtime ABI blocker is active.

## Claim Boundary

This router preserves these nonclaims:

- `public_claim_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `finite_cover_certificate_mask = 0`
- `boundary_gluing_proof_mask = 0`

This is not a SAT, SMT, PB, or Lorenz proof checker result; not imported runtime
evidence; not replay execution; not replay verification; not a Hadwiger-Nelson
result; not a public theorem promotion; not boundary gluing; not a finite-cover
certificate; and not a global Lorenz flowpipe theorem. Its job is to let local
kernel-replay bookkeeping proceed without letting that local bookkeeping become
portable runtime or theorem evidence.
