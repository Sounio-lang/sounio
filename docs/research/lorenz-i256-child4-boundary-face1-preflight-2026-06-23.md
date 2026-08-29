<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-face1-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-face1-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary Face1 Preflight

Date: 2026-06-23

This note records the second face-local boundary-gluing preflight for child4 in
the Lorenz i256 cover lane. It consumes the face0 preflight and the per-face
obligation table, then selects only face `1` for a future proof-checker pass.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_face1_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face1_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face1_preflight_imported.sio`
- Artifact fingerprint: `715930842`
- Audit fingerprint: `268401557`
- Instance fingerprint: `834572019`
- Certificate fingerprint: `450916328`
- Status code: `82`

## Inputs

- Child4 boundary face0 preflight: `629184705` / `884062113`
- Child4 boundary face obligations: `458306927` / `766184391`
- Boundary-gluing obligation mask: `31`
- Face-obligation mask: `31`
- Prior face-preflight mask: `1`

## Face1 Preflight Surface

- `boundary_face_count = 5`
- `selected_face_index = 1`
- `selected_face_mask = 2`
- `face1_obligation_mask = 2`
- `face1_topology_mask = 2`
- `face1_orientation_mask = 2`
- `face1_adjacency_mask = 2`
- `face1_trace_mask = 2`
- `face1_dependency_mask = 3`
- `face1_preflight_mask = 2`
- `face_preflight_progress_mask = 3`
- `face1_pending_proof_mask = 2`
- `solver_face1_action_mask = 2`
- `ready_for_face1_proof_mask = 0`

## Claim Boundary

This receipt extends the face-local preflight chain, not the proof chain. It
preserves these nonclaims:

- `face1_verified_mask = 0`
- `face1_proof_mask = 0`
- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a face1 proof, not a boundary-gluing proof, not child4 discharge,
not a finite-cover certificate, not a global cover certificate, not an invariant
or shadowing proof, not an unbounded-time proof, and not a global Lorenz
flowpipe theorem.
