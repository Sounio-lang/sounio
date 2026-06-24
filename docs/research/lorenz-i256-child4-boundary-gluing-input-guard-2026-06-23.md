<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-gluing-input-guard-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-gluing-input-guard-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary-Gluing Input Guard

Date: 2026-06-23

This note records a conservative input guard for the future child4
boundary-gluing proof in the Lorenz i256 cover lane. It consumes the five-face
preflight bundle, the boundary-gluing proof skeleton, and the boundary-gluing
obligation seed, then records that those inputs are mutually anchored.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_gluing_input_guard.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_gluing_input_guard_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_gluing_input_guard_imported.sio`
- Artifact fingerprint: `508913246`
- Audit fingerprint: `672184905`
- Instance fingerprint: `901436527`
- Certificate fingerprint: `245690813`
- Status code: `87`

## Inputs

- Child4 boundary face preflight bundle: `812640573` / `294736811`
- Child4 boundary-gluing proof skeleton: `836470291` / `190735846`
- Child4 boundary-gluing obligation seed: `604182937` / `218570463`
- Boundary-gluing obligation mask: `31`
- Face-preflight bundle mask: `31`
- Face-pending proof mask: `31`
- Skeleton slot mask: `31`

## Guard Surface

- `boundary_face_count = 5`
- `boundary_gluing_input_guard_mask = 31`
- `face_preflight_dependency_mask = 31`
- `skeleton_dependency_mask = 31`
- `gluing_seed_dependency_mask = 31`
- `boundary_gluing_goal_pending_mask = 31`
- `solver_gluing_next_action_mask = 31`
- `ready_for_boundary_gluing_proof_mask = 0`

## Claim Boundary

This receipt completes an input guard, not a boundary-gluing proof. It preserves
these nonclaims:

- `skeleton_verified_mask = 0`
- `face_verified_mask = 0`
- `face_proof_mask = 0`
- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a skeleton verification, not a five-face proof, not a
boundary-gluing proof, not child4 discharge, not a finite-cover certificate,
not a global cover certificate, not an invariant or shadowing proof, not an
unbounded-time proof, and not a global Lorenz flowpipe theorem.
