<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-discharge-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-discharge-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Discharge Preflight

Date: 2026-06-23

This note records the child4 discharge-preflight ledger in the Lorenz i256 cover
lane. It consumes the child4 validation core and the inherited child3 discharge
preflight. It is a blocker ledger after local validation, not a discharge
receipt.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_discharge_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_discharge_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_discharge_preflight_imported.sio`
- Artifact fingerprint: `938615204`
- Audit fingerprint: `405118337`
- Instance fingerprint: `182647903`
- Certificate fingerprint: `719306552`
- Status code: `75`

## Inputs

- Child4 validation core: `392846115` / `644973208`
- Child3 discharge preflight: `57643306` / `784781218`
- Child index: `4`
- Child validated mask: `31`

## Claim Boundary

This receipt records that the tracked local child-validation segment has no
remaining child-validation bits:

- `pending_child_validation_mask = 0`
- `remaining_child_validation_mask = 0`
- `missing_child_validation_mask = 0`

It preserves the following nonclaims:

- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

The resulting ledger is a preflight blocker surface only. It is not child4
discharge, not a finite-cover certificate, not an invariant or shadowing proof,
not an unbounded-time proof, and not a global Lorenz flowpipe theorem.
