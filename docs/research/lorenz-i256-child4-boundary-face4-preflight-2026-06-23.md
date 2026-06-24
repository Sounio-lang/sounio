<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-face4-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-face4-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary Face4 Preflight

Date: 2026-06-23

This note records the fifth face-local boundary-gluing preflight for child4 in
the Lorenz i256 cover lane. It consumes the face3 preflight and the per-face
obligation table, then selects only face `4` for a future proof-checker pass.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_face4_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face4_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face4_preflight_imported.sio`
- Artifact fingerprint: `486720951`
- Audit fingerprint: `730519284`
- Instance fingerprint: `918374062`
- Certificate fingerprint: `604283791`
- Status code: `85`

## Inputs

- Child4 boundary face3 preflight: `371946280` / `809527416`
- Child4 boundary face obligations: `458306927` / `766184391`
- Boundary-gluing obligation mask: `31`
- Face-obligation mask: `31`
- Prior face-preflight mask: `15`

## Face4 Preflight Surface

- `boundary_face_count = 5`
- `selected_face_index = 4`
- `selected_face_mask = 16`
- `face4_obligation_mask = 16`
- `face4_topology_mask = 16`
- `face4_orientation_mask = 16`
- `face4_adjacency_mask = 16`
- `face4_trace_mask = 16`
- `face4_dependency_mask = 31`
- `face4_preflight_mask = 16`
- `face_preflight_progress_mask = 31`
- `face4_pending_proof_mask = 16`
- `solver_face4_action_mask = 16`
- `ready_for_face4_proof_mask = 0`

## Claim Boundary

This receipt completes the face-local preflight chain, not the proof chain. It
preserves these nonclaims:

- `face4_verified_mask = 0`
- `face4_proof_mask = 0`
- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a face4 proof, not a five-face proof, not a boundary-gluing proof,
not child4 discharge, not a finite-cover certificate, not a global cover
certificate, not an invariant or shadowing proof, not an unbounded-time proof,
and not a global Lorenz flowpipe theorem.
