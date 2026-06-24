<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-trajectory2-portfolio-local-guard-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-trajectory2-portfolio-local-guard-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Trajectory2 Portfolio-Local Guard

Date: 2026-06-24

This note records the local portfolio-local guard (status `122`) for the bounded
two-step Lorenz manifest lane. It programmatically enforces the
no-global-promotion boundary on the dispatch(121) readiness receipt: the
dispatch envelope fingerprint (`9387383`) must be anchored and every
promotion/claim mask must be zero. Any nonzero promotion or claim mask is
rejected (`-1`). It is a local no-overclaim safety guard, not portfolio wiring
and not a theorem promotion.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_trajectory2_portfolio_local_guard.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_trajectory2_portfolio_local_guard_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_trajectory2_portfolio_local_guard_imported.sio`
- Local scoping mask: `63`
- Guard fingerprint: `9400841`
- Guard artifact fingerprint: `517294038`
- Status code: `122`

## Local Scoping Mask

The local scoping mask records, for the dispatch(121) receipt:

- bit `1`: dispatch envelope fingerprint is `9387383`
- bit `2`: `public_claim_mask = 0`
- bit `4`: `global_promotion_ready = 0`
- bit `8`: `formal_theorem_ready = 0`
- bit `16`: `native_i256_evidence_mask = 0`
- bit `32`: `imported_runtime_evidence_mask = 0`

The complete local scoping mask is therefore `63`. A nonzero public-claim mask
drops bit `2` (mask `61`); a nonzero global-promotion flag drops bit `4`
(mask `59`).

## Guard Rule

```text
guard_fp =
  dispatch_envelope_fp
  + 29*local_scoping_mask
  + 31*checker_family_id
  + 37*checker_kind_id
  + 41*bridge_status
  mod 1000000000
```

For the locally-scoped guard:

```text
dispatch_envelope_fp = 9387383
local_scoping_mask = 63
checker_family_id = 73
checker_kind_id = 118
bridge_status = 122
guard_fp = 9400841
```

The guard fingerprint is derived from (and strictly larger than) the dispatch
envelope fingerprint, recording clean lineage `9371853 -> 9387383 -> 9400841`.
The guard function additionally rejects (returns `-1`) if any of
`public_claim_mask`, `global_promotion_ready`, or `formal_theorem_ready` is
nonzero, so the receipt cannot be emitted when a promotion path is open.

## Anchors

This guard anchors the dispatch(121) envelope fingerprint `9387383`, acceptance
readiness `9371853`, acceptance mask `31`, manifest fingerprint `294601254`,
and certificate bridge `918274650`. The guard artifact fingerprint is
`517294038`.

Status lineage: `118`, `119`, `120`, `121`, and `122` are local decimal-limb
audit status codes, not theorem numbers, not older portfolio version numbers,
and not public mathematical milestones.

## Boundary

This guard records:

- `target_integer_width = 256`
- `limb_base = 1000000000`
- `native_i256_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `global_promotion_ready = 0`
- `formal_theorem_ready = 0`

The imported smoke remains frontend/typecheck evidence only while the current
imported/native runtime ABI blocker remains active
(`BLK-20260623-madaros-mm-seed-segfault`).

## Claim Boundary

This is not portfolio wiring, not a global promotion, not a cryptographic
acceptance proof, not a complete Lorenz integrator, not a stability or accuracy
theorem, not arbitrary signed-state coverage, not a general four-limb i256
product, not adaptive stepping, not interval integration, not a finite-cover
certificate, not a boundary-gluing proof, not a global flowpipe theorem, not
native `i256` execution, and not imported/native runtime evidence. It only
enforces the local no-overclaim boundary on the dispatch(121) readiness receipt.
