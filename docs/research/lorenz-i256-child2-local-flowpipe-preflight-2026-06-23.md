<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-child2-local-flowpipe-preflight-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-child2-local-flowpipe-preflight-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Child-2 Local-Flowpipe Preflight

This note records a split-module child-2 local-flowpipe preflight in the Lorenz
i256 proof-carrying dynamics lane. It attaches the child-2 obligation seed to
existing five-step local-chain evidence, but it is **not** a local-flowpipe proof.

## New surface

- `stdlib/systems/lorenz_i256_child2_local_flowpipe_preflight.sio`
- `tests/run-pass/lorenz_i256_cover_child2_local_flowpipe_preflight_tiny.sio`
- `tests/run-pass/lorenz_i256_cover_child2_local_flowpipe_preflight_imported.sio`

The module is split out of `stdlib/systems/lorenz_i256_cert.sio` to avoid
reopening the known parser/size fragility in the monolithic certificate module.

## Anchors

The preflight consumes:

- child-2 obligation-seed artifact/audit `844216507`/`781563019`;
- child-1 discharge-preflight artifact/audit `719480263`/`602184970`;
- five-step local-flowpipe-chain artifact/audit `911209450`/`709377850`.

It records:

- instance/certificate `586307202`/`103884775`;
- artifact/audit `312780944`/`542916038`;
- child index `2`;
- child slots `(0, 1, 0)`;
- `selected_child_mask = 4`;
- `prior_child_validated_mask = 3`;
- `pending_child_validation_mask = 28`;
- `local_flowpipe_preflight_mask = 31`;
- `proof_dependency_mask = 31`;
- `available_local_chain_mask = 31`;
- `pending_local_proof_mask = 31`;
- status `65`;
- `ok_mask = 255`.

## Nonclaims

The preflight keeps these proof/claim masks closed:

- `local_flowpipe_proof_mask = 0`;
- `child_validated_mask = 0`;
- `child_discharge_mask = 0`;
- `global_cover_certificate_mask = 0`;
- `global_flowpipe_claim_mask = 0`.

Therefore this gate does **not** prove a local flowpipe for child `2`, does not
validate child `2`, does not discharge any child, does not certify a finite
cover, and does not assert a global Lorenz theorem. It only records that the
child-2 seed has the local-chain evidence needed before a child-2 proof skeleton
or replay executor may be introduced.

## Validation

Current focused gates:

- `./bin/souc check stdlib/systems/lorenz_i256_child2_local_flowpipe_preflight.sio`
- `./bin/souc check tests/run-pass/lorenz_i256_cover_child2_local_flowpipe_preflight_imported.sio`
- `./bin/souc run tests/run-pass/lorenz_i256_cover_child2_local_flowpipe_preflight_tiny.sio`
- `./scripts/run_sio_test_suite.sh cover_child2_local_flowpipe_preflight`
- `./scripts/run_sio_test_suite.sh cover_child2_obligation_seed`

The imported test is marked known-failure for runtime because current Madaros
imported/native lowering exits `139` on this module family. Its API typecheck is
green; the self-contained tiny test executes the semantics.
