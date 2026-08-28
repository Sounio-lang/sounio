<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child3-local-flowpipe-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child3-local-flowpipe-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child3 Local-Flowpipe Preflight

Date: 2026-06-23

This note records the child3 local-flowpipe preflight in the Lorenz i256 cover
lane. It attaches the child3 obligation seed to the existing five-step local
flowpipe-chain evidence, but it does not prove the local flowpipe and it does
not validate or discharge child3.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child3_local_flowpipe_preflight.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child3_local_flowpipe_preflight_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child3_local_flowpipe_preflight_imported.sio`
- Artifact fingerprint: `730124563`
- Audit fingerprint: `713941204`
- Instance fingerprint: `362477255`
- Certificate fingerprint: `911332062`
- Status code: `69`

## Inputs

- Child3 obligation seed: `853502997` / `176202003`
- Child2 discharge preflight: `865117026` / `22656757`
- Five-step local-flowpipe chain: `911209450` / `709377850`
- Child index: `3`
- Child slot: `(1, 1, 0)`
- Selected child mask: `8`
- Prior child validated mask: `7`
- Pending child validation mask: `24`

## Claim Boundary

This receipt is a preflight ledger only. It records that the child3 obligation
is attached to the existing local-chain evidence and that the proof dependency
surface is present.

Preserved nonclaims:

- `local_flowpipe_proof_mask = 0`
- `child_validated_mask = 0`
- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

It is not a local-flowpipe proof, not child3 validation, not child3 discharge,
not a finite-cover certificate, not an invariant or shadowing proof, and not a
global Lorenz flowpipe theorem.
