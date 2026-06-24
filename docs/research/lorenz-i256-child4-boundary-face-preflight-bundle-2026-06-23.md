<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-face-preflight-bundle-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-face-preflight-bundle-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary Face Preflight Bundle

Date: 2026-06-23

This note records a conservative bundle receipt for the five child4
boundary-face preflights in the Lorenz i256 cover lane. It consumes the face4
preflight and the per-face obligation table, then records that all five
face-local preflight surfaces are present.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_face_preflight_bundle.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face_preflight_bundle_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face_preflight_bundle_imported.sio`
- Artifact fingerprint: `812640573`
- Audit fingerprint: `294736811`
- Instance fingerprint: `673205948`
- Certificate fingerprint: `138729456`
- Status code: `86`

## Inputs

- Child4 boundary face4 preflight: `486720951` / `730519284`
- Child4 boundary face obligations: `458306927` / `766184391`
- Boundary-gluing obligation mask: `31`
- Face-obligation mask: `31`
- Face-preflight progress mask: `31`

## Bundle Surface

- `boundary_face_count = 5`
- `face_preflight_bundle_mask = 31`
- `face_preflight_dependency_mask = 31`
- `face_preflight_trace_mask = 31`
- `face_preflight_ready_mask = 31`
- `face_pending_proof_mask = 31`
- `solver_boundary_next_action_mask = 31`
- `ready_for_boundary_gluing_proof_mask = 0`

## Claim Boundary

This receipt completes the face-local preflight bundle, not the proof bundle. It
preserves these nonclaims:

- `face_verified_mask = 0`
- `face_proof_mask = 0`
- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a five-face proof, not a boundary-gluing proof, not child4
discharge, not a finite-cover certificate, not a global cover certificate, not
an invariant or shadowing proof, not an unbounded-time proof, and not a global
Lorenz flowpipe theorem.
