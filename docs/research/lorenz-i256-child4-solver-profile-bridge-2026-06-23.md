<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-solver-profile-bridge-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-solver-profile-bridge-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Solver Profile Bridge

Date: 2026-06-23

This note records a conservative bridge between the shared solver
proof-profile gate and the Lorenz i256 child4 boundary-gluing input guard. The
bridge consumes the solver proof-profile receipt and the child4 input guard,
then records that the current Lorenz numeric receipt is accepted only as a
private checked artifact.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_solver_profile_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_solver_profile_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_solver_profile_bridge_imported.sio`
- Artifact fingerprint: `784215609`
- Audit fingerprint: `639104872`
- Instance fingerprint: `284903761`
- Certificate fingerprint: `731582406`
- Status code: `89`

## Inputs

- Solver proof profile: `964210753` / `526184309`
- Child4 boundary-gluing input guard: `508913246` / `672184905`
- Solver domain: `4` (Sounio numeric receipt)
- Proof format: `6` (Lorenz i128/i256 numeric receipt)
- Accepted profile mask: `15`
- Rejected profile mask: `48`
- Lorenz numeric gate: `1`
- Private acceptance level: `2`

## Bridge Surface

- `profile_status = 88`
- `guard_status = 87`
- `solver_profile_bridge_mask = 31`
- `lorenz_receipt_family_mask = 16`
- `bridge_dependency_mask = 31`
- `checked_private_receipt_mask = 1`
- `public_theorem_ready_mask = 0`
- `global_promotion_ready_mask = 0`
- `solver_bridge_next_action_mask = 31`

## Claim Boundary

This bridge is a private acceptance record only. It preserves these nonclaims:

- `formal_theorem_ready = 0`
- `public_claim_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `child4_discharge_mask = 0`
- `finite_cover_certificate_mask = 0`

This is not a public theorem promotion, not a boundary-gluing proof, not child4
discharge, not a finite-cover certificate, not a global cover certificate, not
an invariant or shadowing proof, not an unbounded-time proof, and not a global
Lorenz flowpipe theorem.
