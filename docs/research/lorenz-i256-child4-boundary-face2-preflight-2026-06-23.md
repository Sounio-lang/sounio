<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-face2-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-face2-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary Face2 Preflight

Date: 2026-06-23

This note records the third face-local boundary-gluing preflight for child4 in
the Lorenz i256 cover lane. It consumes the face1 preflight and the per-face
obligation table, then selects only face `2` for a future proof-checker pass.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_face2_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face2_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face2_preflight_imported.sio`
- Artifact fingerprint: `903174618`
- Audit fingerprint: `541786290`
- Instance fingerprint: `267490135`
- Certificate fingerprint: `692308417`
- Status code: `83`

## Inputs

- Child4 boundary face1 preflight: `715930842` / `268401557`
- Child4 boundary face obligations: `458306927` / `766184391`
- Boundary-gluing obligation mask: `31`
- Face-obligation mask: `31`
- Prior face-preflight mask: `3`

## Face2 Preflight Surface

- `boundary_face_count = 5`
- `selected_face_index = 2`
- `selected_face_mask = 4`
- `face2_obligation_mask = 4`
- `face2_topology_mask = 4`
- `face2_orientation_mask = 4`
- `face2_adjacency_mask = 4`
- `face2_trace_mask = 4`
- `face2_dependency_mask = 7`
- `face2_preflight_mask = 4`
- `face_preflight_progress_mask = 7`
- `face2_pending_proof_mask = 4`
- `solver_face2_action_mask = 4`
- `ready_for_face2_proof_mask = 0`

## Claim Boundary

This receipt extends the face-local preflight chain, not the proof chain. It
preserves these nonclaims:

- `face2_verified_mask = 0`
- `face2_proof_mask = 0`
- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a face2 proof, not a three-face proof, not a boundary-gluing proof,
not child4 discharge, not a finite-cover certificate, not a global cover
certificate, not an invariant or shadowing proof, not an unbounded-time proof,
and not a global Lorenz flowpipe theorem.
