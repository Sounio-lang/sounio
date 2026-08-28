<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child4-obligation-seed-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child4-obligation-seed-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child4 Obligation Seed

Date: 2026-06-23

This note records the child4 obligation seed in the Lorenz i256 cover lane. It
extends the child-ledger sequence after child3 discharge preflight and child3
validation core, but it does not validate or discharge child4.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child4_obligation_seed.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child4_obligation_seed_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child4_obligation_seed_imported.sio`
- Artifact fingerprint: `17166205`
- Audit fingerprint: `78794130`
- Instance fingerprint: `75459241`
- Certificate fingerprint: `224045073`
- Status code: `72`

## Inputs

- Child3 discharge preflight: `57643306` / `784781218`
- Child3 validation core: `686090643` / `159869521`
- Child index: `4`
- Child slot: `(0, 0, 1)`
- Selected child mask: `16`
- Prior child validated mask: `15`
- Pending child validation mask: `16`

## Claim Boundary

This receipt selects child4 as the next local obligation. It preserves the
following nonclaims:

- `local_flowpipe_proof_mask = 0`
- `child_validated_mask = 0`
- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

It is not child4 validation, not child4 discharge, not a finite-cover
certificate, not an invariant or shadowing proof, and not a global Lorenz
flowpipe theorem.
