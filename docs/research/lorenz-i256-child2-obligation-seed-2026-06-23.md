<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child2-obligation-seed-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child2-obligation-seed-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child-2 Obligation Seed

This note records a split-module child-2 obligation seed in the Lorenz i256
proof-carrying dynamics lane. It is a pending-obligation seed only, not child-2
validation and not discharge.

## New surface

- `stdlib/systems/lorenz_i256_child2_obligation_seed.sio`
- `tests/run-pass/lorenz_i256_cover_child2_obligation_seed_tiny.sio`
- `tests/run-pass/lorenz_i256_cover_child2_obligation_seed_imported.sio`

The module is split out of the monolithic certificate file for the same reason
as the child-1 validation-core and discharge-preflight modules: appending more
large ledger code to `stdlib/systems/lorenz_i256_cert.sio` has already exposed
Madaros parser/size fragility.

## Anchors

The seed consumes:

- child-1 discharge-preflight artifact/audit `719480263`/`602184970`;
- child-1 validation-core artifact/audit `648831016`/`63312412`.

It records:

- instance/certificate `234881913`/`671092444`;
- artifact/audit `844216507`/`781563019`;
- child index `2`;
- child slots `(0, 1, 0)`;
- `selected_child_mask = 4`;
- `prior_child_validated_mask = 3`, meaning children `0` and `1` are available
  as inherited local-validation context;
- `pending_child_validation_mask = 28`, meaning children `2`, `3`, and `4`
  remain pending in the five-child cover ledger;
- status `64`;
- `ok_mask = 255`.

## Nonclaims

The seed keeps these proof/claim masks closed:

- `local_flowpipe_proof_mask = 0`;
- `child_validated_mask = 0`;
- `child_discharge_mask = 0`;
- `global_cover_certificate_mask = 0`;
- `global_flowpipe_claim_mask = 0`.

Therefore this gate does **not** validate child `2`, does not prove a local
flowpipe for child `2`, does not discharge any child, does not certify a finite
cover, and does not assert a global Lorenz theorem. It only opens child `2` as
the next replayable cover obligation after the child-1 discharge preflight.

## Validation

Current focused gates:

- `./bin/souc check stdlib/systems/lorenz_i256_child2_obligation_seed.sio`
- `./bin/souc check tests/run-pass/lorenz_i256_cover_child2_obligation_seed_imported.sio`
- `./bin/souc run tests/run-pass/lorenz_i256_cover_child2_obligation_seed_tiny.sio`
- `./scripts/run_sio_test_suite.sh cover_child2_obligation_seed`
- `./scripts/run_sio_test_suite.sh cover_child1_discharge_preflight`

The imported test is marked known-failure for runtime because current Madaros
imported/native lowering exits `139` on this module family. Its API typecheck is
green; the self-contained tiny test executes the semantics.
