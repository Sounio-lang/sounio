<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child2-validation-core-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child2-validation-core-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child2 Validation Core

Date: 2026-06-23

This note records the next local receipt in the Lorenz i256 cover lane: child2
validation core. It consumes the child2 local-flowpipe preflight, the child2
obligation seed, the child1 validation core, and the existing five-step local
flowpipe-chain receipt.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child2_validation_core.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child2_validation_core_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child2_validation_core_imported.sio`
- Artifact fingerprint: `173471211`
- Audit fingerprint: `173431050`
- Instance fingerprint: `61203519`
- Certificate fingerprint: `907847810`
- Status code: `66`

## Inputs

- Child2 local-flowpipe preflight: `312780944` / `542916038`
- Child2 obligation seed: `844216507` / `781563019`
- Child1 validation core: `648831016` / `63312412`
- Five-step local-flowpipe chain: `911209450` / `709377850`
- Child index: `2`
- Child slot: `(0, 1, 0)`

## Claim Boundary

This receipt only moves child2 from preflight state to local validation-core
state. It preserves the following nonclaims:

- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

The resulting `child_validated_mask = 7` means children 0, 1, and 2 are now
locally marked in this lane, while `pending_child_validation_mask = 24` keeps
the remaining child obligations open. This is not a global Lorenz theorem, not
a finite-cover certificate, and not a shadowing or invariant proof.
