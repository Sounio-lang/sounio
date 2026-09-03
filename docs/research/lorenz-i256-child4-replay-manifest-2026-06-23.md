<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-replay-manifest-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-replay-manifest-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Replay Manifest

Date: 2026-06-23

This note records a replay manifest for the Lorenz i256 child4 solver-profile
bridge. The manifest ties together the private solver-profile bridge, the
shared solver proof profile, the child4 boundary-gluing input guard, and the
child4 boundary-face preflight bundle. It is deliberately a replay receipt, not
a theorem promotion.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_replay_manifest.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_replay_manifest_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_replay_manifest_imported.sio`
- Artifact fingerprint: `916482305`
- Audit fingerprint: `274650918`
- Instance fingerprint: `537109264`
- Certificate fingerprint: `820641573`
- Status code: `90`

## Inputs

- Solver-profile bridge: `784215609` / `639104872`
- Solver proof profile: `964210753` / `526184309`
- Child4 boundary-gluing input guard: `508913246` / `672184905`
- Child4 boundary-face preflight bundle: `812640573` / `294736811`
- Solver domain: `4` (Sounio numeric receipt)
- Proof format: `6` (Lorenz i128/i256 numeric receipt)
- Profile status: `88`
- Input guard status: `87`
- Bridge status: `89`
- Replay manifest status: `90`

## Replay Surface

- `focused_gate_mask = 15`
- `offload_review_mask = 3`
- `replay_dependency_mask = 31`
- `manifest_dependency_mask = 31`
- `known_failure_mask = 1`
- `full_portfolio_green_mask = 0`
- `heavy_validation_mask = 0`
- `replay_anchor_mask = 255`
- `replay_status_mask = 15`
- `local_gate_complete_mask = 15`
- `offload_complete_mask = 3`
- `known_imported_native_lowering_failure_mask = 1`
- `no_full_portfolio_evidence_mask = 0`
- `no_heavy_validation_evidence_mask = 0`
- `public_theorem_ready_mask = 0`
- `global_promotion_ready_mask = 0`
- `replay_next_action_mask = 31`

## Claim Boundary

This manifest preserves these nonclaims:

- `formal_theorem_ready = 0`
- `public_claim_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `child4_discharge_mask = 0`
- `finite_cover_certificate_mask = 0`

This is not a public theorem promotion, not a boundary-gluing proof, not child4
discharge, not a finite-cover certificate, not a global cover certificate, not
an invariant or shadowing proof, not an unbounded-time proof, and not a global
Lorenz flowpipe theorem. The imported smoke remains a known runtime failure
for the current imported/native lowering path; the replay manifest records that
failure instead of hiding it.
