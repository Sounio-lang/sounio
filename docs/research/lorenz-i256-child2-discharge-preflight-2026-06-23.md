<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child2-discharge-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child2-discharge-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child2 Discharge Preflight

Date: 2026-06-23

This note records the child2 discharge-preflight ledger in the Lorenz i256 cover
lane. It consumes the child2 validation-core receipt and the inherited child1
discharge-preflight receipt. The result is a blocker-aware preflight artifact,
not a discharge certificate.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child2_discharge_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child2_discharge_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child2_discharge_preflight_imported.sio`
- Artifact fingerprint: `865117026`
- Audit fingerprint: `22656757`
- Instance fingerprint: `362446006`
- Certificate fingerprint: `449775673`
- Status code: `67`

## Inputs

- Child2 validation core: `173471211` / `173431050`
- Child1 discharge preflight: `719480263` / `602184970`
- Child index: `2`
- Child validated mask: `7`

## Claim Boundary

This receipt records that child2 has a discharge-preflight ledger after local
validation. It does not discharge child2 and it does not certify a finite cover.

Preserved blockers and nonclaims:

- `pending_child_validation_mask = 24`
- `remaining_child_validation_mask = 24`
- `missing_child_validation_mask = 24`
- `pending_child_discharge_mask = 255`
- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `finite_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

The result is intentionally local and ledger-shaped. It is not an invariant
proof, not a shadowing proof, not an unbounded-time proof, and not a global
Lorenz flowpipe theorem.
