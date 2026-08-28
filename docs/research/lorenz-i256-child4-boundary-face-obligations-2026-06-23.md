<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-face-obligations-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-face-obligations-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary Face Obligations

Date: 2026-06-23

This note records a per-face obligation table for child4 boundary gluing in the
Lorenz i256 cover lane. It consumes the child4 boundary-gluing proof skeleton
and the child4 boundary-gluing obligation seed, then splits the pending gluing
surface into five face-local obligations.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_face_obligations.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face_obligations_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_face_obligations_imported.sio`
- Artifact fingerprint: `458306927`
- Audit fingerprint: `766184391`
- Instance fingerprint: `328691604`
- Certificate fingerprint: `948157216`
- Status code: `80`

## Inputs

- Child4 boundary-gluing proof skeleton: `836470291` / `190735846`
- Child4 boundary-gluing obligation seed: `604182937` / `218570463`
- Boundary-gluing obligation mask: `31`
- Boundary-gluing pending mask: `31`

## Face Obligation Surface

- `boundary_face_count = 5`
- `face0_obligation_mask = 1`
- `face1_obligation_mask = 2`
- `face2_obligation_mask = 4`
- `face3_obligation_mask = 8`
- `face4_obligation_mask = 16`
- `face_obligation_mask = 31`
- `face_topology_mask = 31`
- `face_orientation_mask = 31`
- `face_adjacency_mask = 31`
- `face_trace_mask = 31`
- `face_verified_mask = 0`
- `face_proof_mask = 0`
- `solver_face_action_mask = 31`
- `ready_for_boundary_gluing_proof_mask = 0`

## Claim Boundary

This receipt is a per-face obligation split, not a proof. It makes the boundary
gluing surface more inspectable for future proof-checker work while preserving
these nonclaims:

- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a boundary-gluing proof, not child4 discharge, not a finite-cover
certificate, not a global cover certificate, not an invariant or shadowing
proof, not an unbounded-time proof, and not a global Lorenz flowpipe theorem.
