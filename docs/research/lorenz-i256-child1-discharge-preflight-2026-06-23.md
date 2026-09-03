<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child1-discharge-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child1-discharge-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child-1 Discharge Preflight

This note records the next local gate in the Lorenz i256 proof-carrying
dynamics lane. It is deliberately a **preflight** after local child-1 validation,
not a discharge certificate and not a global Lorenz theorem.

## New surface

- `stdlib/systems/lorenz_i256_child1_discharge_preflight.sio`
- `tests/run-pass/lorenz_i256_cover_child1_discharge_preflight_tiny.sio`
- `tests/run-pass/lorenz_i256_cover_child1_discharge_preflight_imported.sio`

The implementation is split out of `stdlib/systems/lorenz_i256_cert.sio` to avoid
reopening the parser/size fragility already observed when appending more code to
the monolithic certificate module.

## Anchors

The preflight consumes:

- child-1 validation-core artifact/audit `648831016`/`63312412`;
- child-0 discharge-preflight artifact/audit `367693213`/`255409323`.

It records:

- instance/certificate `302491877`/`917204631`;
- artifact/audit `719480263`/`602184970`;
- child index `1`;
- `child_validated_mask = 3`, meaning the local ledger has child `0` and child
  `1` validation receipts available;
- `pending_child_validation_mask = 28` and
  `remaining_child_validation_mask = 28`, meaning children `2`, `3`, and `4`
  remain pending in the five-child cover ledger;
- `child_validation_dependency_mask = 63`;
- `child1_validation_receipt_mask = 31`;
- `child1_discharge_preflight_mask = 2`;
- `discharge_blocker_mask = 15`;
- `missing_child_validation_mask = 28`;
- `pending_child_discharge_mask = 255`;
- status `63`;
- `ok_mask = 255`.

## Nonclaims

The preflight requires these masks to remain zero:

- `child_discharge_mask = 0`;
- `global_cover_certificate_mask = 0`;
- `finite_cover_certificate_mask = 0`;
- `global_flowpipe_claim_mask = 0`.

Therefore this gate does **not** discharge child `1`, does not certify the finite
cover, does not prove invariant or shadowing behavior, and does not assert a
global Lorenz flowpipe theorem. It is a blocker ledger that says the chain has
moved from "child-1 local validation exists" to "child-1 is ready for discharge
analysis, but discharge is still blocked by remaining children and global proof
obligations."

## Validation

Current focused gates:

- `./bin/souc check stdlib/systems/lorenz_i256_child1_discharge_preflight.sio`
- `./bin/souc check tests/run-pass/lorenz_i256_cover_child1_discharge_preflight_imported.sio`
- `./bin/souc run tests/run-pass/lorenz_i256_cover_child1_discharge_preflight_tiny.sio`
- `./scripts/run_sio_test_suite.sh cover_child1_discharge_preflight`
- `./scripts/run_sio_test_suite.sh cover_child1_validation_core`
- `./bin/souc check stdlib/systems/lorenz_i256_child1_validation_core.sio`
- `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`

The imported test is marked known-failure for runtime because current Madaros
imported/native lowering exits `139` on this module family. Its API typecheck is
green; the self-contained tiny test executes the semantics.
