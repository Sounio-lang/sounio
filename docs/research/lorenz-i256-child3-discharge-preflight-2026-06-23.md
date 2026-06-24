<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child3-discharge-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child3-discharge-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child3 Discharge Preflight

Date: 2026-06-23

This note records the child3 discharge-preflight ledger in the Lorenz i256 cover
lane. It consumes the child3 validation-core receipt and the inherited child2
discharge-preflight receipt. The result is a blocker-aware preflight artifact,
not a discharge certificate.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child3_discharge_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child3_discharge_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child3_discharge_preflight_imported.sio`
- Artifact fingerprint: `57643306`
- Audit fingerprint: `784781218`
- Instance fingerprint: `431923454`
- Certificate fingerprint: `729470348`
- Status code: `71`

## Inputs

- Child3 validation core: `686090643` / `159869521`
- Child2 discharge preflight: `865117026` / `22656757`
- Child index: `3`
- Child validated mask: `15`

## Claim Boundary

This receipt records that child3 has a discharge-preflight ledger after local
validation. It does not discharge child3 and it does not certify a finite cover.

Preserved blockers and nonclaims:

- `pending_child_validation_mask = 16`
- `remaining_child_validation_mask = 16`
- `missing_child_validation_mask = 16`
- `pending_child_discharge_mask = 255`
- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

The result is intentionally local and ledger-shaped. It is not an invariant
proof, not a shadowing proof, not an unbounded-time proof, and not a global
Lorenz flowpipe theorem.
