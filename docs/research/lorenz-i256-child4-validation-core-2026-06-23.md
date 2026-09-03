<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-validation-core-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-validation-core-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Validation Core

Date: 2026-06-23

This note records the child4 validation-core receipt in the Lorenz i256 cover
lane. It consumes the child4 local-flowpipe preflight, the child4 obligation
seed, the child3 validation core, and the existing five-step local-flowpipe
chain.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_validation_core.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_validation_core_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_validation_core_imported.sio`
- Artifact fingerprint: `392846115`
- Audit fingerprint: `644973208`
- Instance fingerprint: `918502367`
- Certificate fingerprint: `276341909`
- Status code: `74`

## Inputs

- Child4 local-flowpipe preflight: `886312874` / `82275288`
- Child4 obligation seed: `17166205` / `78794130`
- Child3 validation core: `686090643` / `159869521`
- Five-step local-flowpipe chain: `911209450` / `709377850`
- Child index: `4`
- Child slot: `(0, 0, 1)`

## Claim Boundary

This receipt only moves child4 from preflight state to local validation-core
state. It preserves the following nonclaims:

- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

The resulting `child_validated_mask = 31` records local validation for the
currently tracked child-validation lane. The `pending_child_validation_mask = 0`
closes this local validation segment, but it does not discharge child4 and does
not promote the segment to a finite-cover certificate.

This is not child4 discharge, not a global finite-cover certificate, not an
invariant or shadowing proof, not an unbounded-time proof, and not a global
Lorenz flowpipe theorem.
