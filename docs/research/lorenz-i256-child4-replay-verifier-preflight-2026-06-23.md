<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-replay-verifier-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-replay-verifier-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Replay Verifier Preflight

Date: 2026-06-23

This note records a verifier-preflight receipt for the Lorenz i256 child4
replay manifest. It checks that the manifest and its solver-profile,
proof-profile, input-guard, and boundary-face bundle anchors are present, and
it records which verifier phases are locally available before replay execution.

It is intentionally not a replay execution record and not a replay verification
record.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_replay_verifier_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_replay_verifier_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_replay_verifier_preflight_imported.sio`
- Artifact fingerprint: `391742608`
- Audit fingerprint: `650219347`
- Instance fingerprint: `184903726`
- Certificate fingerprint: `709561238`
- Status code: `91`

## Inputs

- Replay manifest: `916482305` / `274650918`
- Solver-profile bridge: `784215609` / `639104872`
- Solver proof profile: `964210753` / `526184309`
- Child4 boundary-gluing input guard: `508913246` / `672184905`
- Child4 boundary-face preflight bundle: `812640573` / `294736811`
- Solver domain: `4` (Sounio numeric receipt)
- Proof format: `6` (Lorenz i128/i256 numeric receipt)
- Replay manifest status: `90`
- Verifier preflight status: `91`

## Verifier Surface

- `manifest_focused_gate_mask = 15`
- `manifest_offload_review_mask = 3`
- `verifier_phase_mask = 31`
- `verifier_dependency_mask = 31`
- `verifier_input_mask = 31`
- `verifier_reject_mask = 63`
- `known_failure_mask = 1`
- `full_portfolio_green_mask = 0`
- `heavy_validation_mask = 0`
- `replay_execution_mask = 0`
- `replay_verified_mask = 0`
- `verifier_anchor_mask = 255`
- `verifier_status_mask = 3`
- `verifier_available_phase_mask = 31`
- `verifier_required_input_mask = 31`
- `verifier_required_reject_mask = 63`
- `known_imported_native_lowering_failure_mask = 1`
- `no_full_portfolio_evidence_mask = 0`
- `no_heavy_validation_evidence_mask = 0`
- `replay_not_executed_mask = 0`
- `replay_not_verified_mask = 0`
- `public_theorem_ready_mask = 0`
- `global_promotion_ready_mask = 0`
- `verifier_next_action_mask = 31`

## Claim Boundary

This verifier preflight preserves these nonclaims:

- `formal_theorem_ready = 0`
- `public_claim_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `child4_discharge_mask = 0`
- `finite_cover_certificate_mask = 0`

This is not replay execution, not replay verification, not public theorem
promotion, not a boundary-gluing proof, not child4 discharge, not a finite-cover
certificate, not a global cover certificate, not an invariant or shadowing
proof, not an unbounded-time proof, and not a global Lorenz flowpipe theorem.
The imported smoke remains a known runtime failure for the current
imported/native lowering path; this preflight records that limitation rather
than hiding it.
