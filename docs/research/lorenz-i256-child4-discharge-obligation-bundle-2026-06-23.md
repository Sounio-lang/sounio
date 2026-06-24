<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-discharge-obligation-bundle-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-discharge-obligation-bundle-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Discharge-Obligation Bundle

Date: 2026-06-23

This note records a child4 discharge-obligation bundle for the Lorenz i256 cover
lane. It consumes the finite-cover promotion guard and the child4
discharge-preflight ledger, then records the proof obligations that remain
before any discharge or certificate promotion can be considered.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_discharge_obligation_bundle.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_discharge_obligation_bundle_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_discharge_obligation_bundle_imported.sio`
- Artifact fingerprint: `741805623`
- Audit fingerprint: `529164708`
- Instance fingerprint: `364918255`
- Certificate fingerprint: `971203684`
- Status code: `77`

## Inputs

- Child4 finite-cover promotion guard: `657240918` / `312489670`
- Child4 discharge preflight: `938615204` / `405118337`
- Child validated mask: `31`
- Pending child validation mask: `0`
- Pending child discharge mask: `255`

## Remaining Obligations

- `discharge_obligation_mask = 255`
- `required_child_discharge_proof_mask = 255`
- `required_boundary_gluing_mask = 31`
- `required_finite_cover_certificate_mask = 15`
- `required_global_claim_review_mask = 15`
- `missing_discharge_proof_mask = 255`
- `missing_certificate_proof_mask = 15`
- `solver_next_action_mask = 31`
- `ready_for_promotion_mask = 0`

## Claim Boundary

This receipt is an obligation bundle, not a proof. It translates the
fail-closed promotion guard into explicit missing proof surfaces.

Preserved nonclaims:

- `promotion_allowed_mask = 0`
- `finite_cover_candidate_mask = 0`
- `global_certificate_candidate_mask = 0`
- `global_flowpipe_claim_mask = 0`

This is not child4 discharge, not a finite-cover certificate, not a global cover
certificate, not an invariant or shadowing proof, not an unbounded-time proof,
and not a global Lorenz flowpipe theorem.
