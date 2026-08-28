<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-finite-cover-promotion-guard-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-finite-cover-promotion-guard-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Finite-Cover Promotion Guard

Date: 2026-06-23

This note records a fail-closed finite-cover promotion guard for the Lorenz i256
cover lane. It consumes the child4 discharge-preflight ledger and the child4
validation core, then records that the lane is not promotable to a finite-cover
or global-flowpipe certificate while discharge bits remain zero.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_finite_cover_promotion_guard.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_finite_cover_promotion_guard_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_finite_cover_promotion_guard_imported.sio`
- Artifact fingerprint: `657240918`
- Audit fingerprint: `312489670`
- Instance fingerprint: `880163417`
- Certificate fingerprint: `246971305`
- Status code: `76`

## Inputs

- Child4 discharge preflight: `938615204` / `405118337`
- Child4 validation core: `392846115` / `644973208`
- Child validated mask: `31`
- Pending child validation mask: `0`
- Pending child discharge mask: `255`

## Claim Boundary

This receipt is a non-promotion guard. It records a proof-checker boundary:
local child-validation bits are closed, but child discharge and certificate
promotion are still blocked.

Preserved blockers and nonclaims:

- `pending_child_discharge_mask = 255`
- `child_discharge_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `promotion_allowed_mask = 0`
- `finite_cover_candidate_mask = 0`
- `global_certificate_candidate_mask = 0`

This is not child4 discharge, not a finite-cover certificate, not a global cover
certificate, not an invariant or shadowing proof, not an unbounded-time proof,
and not a global Lorenz flowpipe theorem.
