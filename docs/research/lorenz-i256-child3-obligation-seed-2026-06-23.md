<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child3-obligation-seed-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child3-obligation-seed-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child3 Obligation Seed

Date: 2026-06-23

This note records the child3 obligation seed in the Lorenz i256 cover lane. It
extends the child-ledger sequence after child2 discharge preflight and child2
validation core, but it does not validate or discharge child3.

## Receipt

- Module: `stdlib/systems/lorenz_i256_child3_obligation_seed.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_cover_child3_obligation_seed_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_cover_child3_obligation_seed_imported.sio`
- Artifact fingerprint: `853502997`
- Audit fingerprint: `176202003`
- Instance fingerprint: `669599681`
- Certificate fingerprint: `307153093`
- Status code: `68`

## Inputs

- Child2 discharge preflight: `865117026` / `22656757`
- Child2 validation core: `173471211` / `173431050`
- Child index: `3`
- Child slot: `(1, 1, 0)`
- Selected child mask: `8`
- Prior child validated mask: `7`
- Pending child validation mask: `24`

## Claim Boundary

This receipt selects child3 as the next local obligation. It preserves the
following nonclaims:

- `local_flowpipe_proof_mask = 0`
- `child_validated_mask = 0`
- `child_discharge_mask = 0`
- `global_cover_certificate_mask = 0`
- `global_flowpipe_claim_mask = 0`

It is not child3 validation, not child3 discharge, not a finite-cover
certificate, not an invariant or shadowing proof, and not a global Lorenz
flowpipe theorem.
