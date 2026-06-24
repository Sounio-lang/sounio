<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child3-validation-core-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child3-validation-core-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child3 Validation Core

Date: 2026-06-23

This note records the child3 validation-core receipt in the Lorenz i256 cover
lane. It consumes the child3 local-flowpipe preflight, the child3 obligation
seed, the child2 validation core, and the existing five-step local-flowpipe
chain.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child3_validation_core.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child3_validation_core_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child3_validation_core_imported.sio`
- Artifact fingerprint: `686090643`
- Audit fingerprint: `159869521`
- Instance fingerprint: `516858439`
- Certificate fingerprint: `931250157`
- Status code: `70`

## Inputs

- Child3 local-flowpipe preflight: `730124563` / `713941204`
- Child3 obligation seed: `853502997` / `176202003`
- Child2 validation core: `173471211` / `173431050`
- Five-step local-flowpipe chain: `911209450` / `709377850`
- Child index: `3`
- Child slot: `(1, 1, 0)`

## Claim Boundary

This receipt only moves child3 from preflight state to local validation-core
state. It preserves the following nonclaims:

- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

The resulting `child_validated_mask = 15` records local validation for children
0 through 3 in this lane. The `pending_child_validation_mask = 16` keeps the
next child obligation open. This is not child3 discharge, not a finite-cover
certificate, not an invariant or shadowing proof, not an unbounded-time proof,
and not a global Lorenz flowpipe theorem.
