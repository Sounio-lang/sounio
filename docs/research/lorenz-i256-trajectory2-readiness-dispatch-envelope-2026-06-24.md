<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-trajectory2-readiness-dispatch-envelope-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-trajectory2-readiness-dispatch-envelope-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Trajectory2 Readiness Dispatch Envelope

Date: 2026-06-24

This note records the local readiness dispatch envelope (status `121`) for the
bounded two-step Lorenz manifest lane. It exposes the acceptance(120)
readiness receipt to a local solver-router query: given the bounded profile
metadata, a valid query returns the acceptance readiness fingerprint
(`9371853`); any metadata or claim-mask mismatch is rejected. It is a local
solver-router readiness receipt, not portfolio wiring and not a theorem
promotion.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_trajectory2_readiness_dispatch_envelope.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_trajectory2_readiness_dispatch_envelope_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_trajectory2_readiness_dispatch_envelope_imported.sio`
- Acceptance readiness fingerprint (returned by a valid query): `9371853`
- Dispatch mask: `63`
- Envelope fingerprint: `9387383`
- Dispatch artifact fingerprint: `682914503`
- Status code: `121`

## Dispatch Mask

The dispatch mask records, for a bounded-profile router query:

- bit `1`: query family is `73`
- bit `2`: query kind is `118`
- bit `4`: status lineage anchors the acceptance(120) receipt
- bit `8`: acceptance mask `31` is verified
- bit `16`: acceptance readiness fingerprint `9371853` is verified
- bit `32`: native/runtime/public/formal theorem masks are all zero

The complete local dispatch mask is therefore `63`. A wrong family drops bit
`1` (mask `62`); a nonzero public-claim mask drops bit `32` (mask `31`).

## Query Rule

A valid local query (family `73`, kind `118`, acceptance mask `31`, acceptance
readiness `9371853`, dispatch mask `63`, all boundary masks zero, width `256`,
limb base `1000000000`) returns the acceptance readiness receipt `9371853`.
Any mismatch returns `-1` (rejected before any receipt is exposed).

## Envelope Rule

```text
envelope_fp =
  acceptance_readiness_fp
  + 29*dispatch_mask
  + 31*acceptance_mask
  + 37*checker_family_id
  + 41*checker_kind_id
  + 43*bridge_status
  mod 1000000000
```

For the accepted bounded dispatch:

```text
acceptance_readiness_fp = 9371853
dispatch_mask = 63
acceptance_mask = 31
checker_family_id = 73
checker_kind_id = 118
bridge_status = 121
envelope_fp = 9387383
```

The envelope fingerprint is derived from (and strictly larger than) the
acceptance readiness receipt, recording clean lineage `9371853 -> 9387383`.

## Anchors

This envelope anchors the acceptance(120) readiness receipt `9371853`,
acceptance mask `31`, manifest fingerprint `294601254`, and certificate bridge
`918274650`. The dispatch artifact fingerprint is `682914503`.

Status lineage: `118`, `119`, `120`, and `121` are local decimal-limb audit
status codes, not theorem numbers, not older portfolio version numbers, and
not public mathematical milestones.

## Boundary

This envelope records:

- `target_integer_width = 256`
- `limb_base = 1000000000`
- `native_i256_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

The imported smoke remains frontend/typecheck evidence only while the current
imported/native runtime ABI blocker remains active
(`BLK-20260623-madaros-mm-seed-segfault`).

## Claim Boundary

This is not portfolio wiring, not a cryptographic acceptance proof, not a
complete Lorenz integrator, not a stability or accuracy theorem, not arbitrary
signed-state coverage, not a general four-limb i256 product, not adaptive
stepping, not interval integration, not a finite-cover certificate, not a
boundary-gluing proof, not a global flowpipe theorem, not native `i256`
execution, and not imported/native runtime evidence. It only exposes the local
acceptance(120) readiness receipt to a local solver-router query under the
explicit restrictions inherited from the acceptance bridge.
