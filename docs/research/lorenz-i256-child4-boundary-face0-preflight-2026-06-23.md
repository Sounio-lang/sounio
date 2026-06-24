<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-face0-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-face0-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary Face0 Preflight

Date: 2026-06-23

This note records the first face-local boundary-gluing preflight for child4 in
the Lorenz i256 cover lane. It consumes the child4 per-face obligation table and
the child4 boundary-gluing proof skeleton, then selects only face `0` for a
future proof-checker pass.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_face0_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face0_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face0_preflight_imported.sio`
- Artifact fingerprint: `629184705`
- Audit fingerprint: `884062113`
- Instance fingerprint: `517903246`
- Certificate fingerprint: `703681994`
- Status code: `81`

## Inputs

- Child4 boundary face obligations: `458306927` / `766184391`
- Child4 boundary-gluing proof skeleton: `836470291` / `190735846`
- Boundary-gluing obligation mask: `31`
- Face-obligation mask: `31`

## Face0 Preflight Surface

- `boundary_face_count = 5`
- `selected_face_index = 0`
- `selected_face_mask = 1`
- `face0_obligation_mask = 1`
- `face0_topology_mask = 1`
- `face0_orientation_mask = 1`
- `face0_adjacency_mask = 1`
- `face0_trace_mask = 1`
- `face0_dependency_mask = 1`
- `face0_preflight_mask = 1`
- `face0_pending_proof_mask = 1`
- `solver_face0_action_mask = 1`
- `ready_for_face0_proof_mask = 0`

## Claim Boundary

This receipt selects a face-local preflight obligation, not a proof. It preserves
these nonclaims:

- `face0_verified_mask = 0`
- `face0_proof_mask = 0`
- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a face0 proof, not a boundary-gluing proof, not child4 discharge,
not a finite-cover certificate, not a global cover certificate, not an invariant
or shadowing proof, not an unbounded-time proof, and not a global Lorenz
flowpipe theorem.
