<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-gluing-proof-skeleton-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-gluing-proof-skeleton-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary-Gluing Proof Skeleton

Date: 2026-06-23

This note records a boundary-gluing proof skeleton for child4 in the Lorenz i256
cover lane. It consumes the child4 boundary-gluing obligation seed and the
child4 discharge-obligation bundle, then records the structure a future proof
must fill.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_gluing_proof_skeleton.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_gluing_proof_skeleton_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_gluing_proof_skeleton_imported.sio`
- Artifact fingerprint: `836470291`
- Audit fingerprint: `190735846`
- Instance fingerprint: `472619308`
- Certificate fingerprint: `619284057`
- Status code: `79`

## Inputs

- Child4 boundary-gluing obligation seed: `604182937` / `218570463`
- Child4 discharge-obligation bundle: `741805623` / `529164708`
- Boundary-gluing obligation mask: `31`
- Boundary-gluing pending mask: `31`

## Skeleton Surface

- `boundary_face_count = 5`
- `skeleton_slot_mask = 31`
- `skeleton_topology_mask = 31`
- `skeleton_dependency_mask = 31`
- `skeleton_trace_mask = 31`
- `skeleton_verified_mask = 0`
- `solver_skeleton_action_mask = 31`
- `ready_for_boundary_gluing_proof_mask = 0`

## Claim Boundary

This receipt is a proof skeleton, not a proof. It records structure for a future
boundary-gluing argument but preserves these nonclaims:

- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a boundary-gluing proof, not child4 discharge, not a finite-cover
certificate, not a global cover certificate, not an invariant or shadowing
proof, not an unbounded-time proof, and not a global Lorenz flowpipe theorem.
