<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-local-flowpipe-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-local-flowpipe-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Local-Flowpipe Preflight

Date: 2026-06-23

This note records the child4 local-flowpipe preflight in the Lorenz i256 cover
lane. It attaches the child4 obligation seed to the existing five-step local
flowpipe-chain evidence, but it does not prove the local flowpipe and it does
not validate or discharge child4.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_local_flowpipe_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_local_flowpipe_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_local_flowpipe_preflight_imported.sio`
- Artifact fingerprint: `886312874`
- Audit fingerprint: `82275288`
- Instance fingerprint: `635431038`
- Certificate fingerprint: `801960582`
- Status code: `73`

## Inputs

- Child4 obligation seed: `17166205` / `78794130`
- Child3 discharge preflight: `57643306` / `784781218`
- Five-step local-flowpipe chain: `911209450` / `709377850`
- Child index: `4`
- Child slot: `(0, 0, 1)`
- Selected child mask: `16`
- Prior child validated mask: `15`
- Pending child validation mask: `16`

## Claim Boundary

This receipt is a preflight ledger only. It records that the child4 obligation
is attached to the existing local-chain evidence and that the proof dependency
surface is present.

Preserved nonclaims:

- `local_flowpipe_proof_mask = 0`
- `child_validated_mask = 0`
- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

It is not a local-flowpipe proof, not child4 validation, not child4 discharge,
not a finite-cover certificate, not an invariant or shadowing proof, and not a
global Lorenz flowpipe theorem.
