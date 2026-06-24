<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-boundary-gluing-obligation-seed-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-boundary-gluing-obligation-seed-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Boundary-Gluing Obligation Seed

Date: 2026-06-23

This note records a boundary-gluing obligation seed for child4 in the Lorenz
i256 cover lane. It consumes the child4 discharge-obligation bundle and the
finite-cover promotion guard, then opens the boundary-gluing proof surface
without claiming that any gluing proof exists.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_boundary_gluing_obligation_seed.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_boundary_gluing_obligation_seed_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_boundary_gluing_obligation_seed_imported.sio`
- Artifact fingerprint: `604182937`
- Audit fingerprint: `218570463`
- Instance fingerprint: `785604219`
- Certificate fingerprint: `140926774`
- Status code: `78`

## Inputs

- Child4 discharge-obligation bundle: `741805623` / `529164708`
- Child4 finite-cover promotion guard: `657240918` / `312489670`
- Required boundary-gluing mask: `31`
- Missing discharge proof mask: `255`
- Pending child discharge mask: `255`

## Boundary-Gluing Surface

- `boundary_face_count = 5`
- `boundary_gluing_obligation_mask = 31`
- `boundary_gluing_pending_mask = 31`
- `boundary_gluing_verified_mask = 0`
- `boundary_gluing_candidate_mask = 0`
- `boundary_gluing_audit_mask = 31`
- `solver_boundary_action_mask = 31`
- `ready_for_child_discharge_mask = 0`

## Claim Boundary

This receipt is an obligation seed, not a proof. It opens a proof surface for
boundary gluing but preserves these nonclaims:

- `boundary_gluing_proof_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not a boundary-gluing proof, not child4 discharge, not a finite-cover
certificate, not a global cover certificate, not an invariant or shadowing
proof, not an unbounded-time proof, and not a global Lorenz flowpipe theorem.
