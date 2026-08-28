<!-- docs:meta
topic_id: repo.docs.research.solver-proof-profile-dispatch-envelope-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.solver-proof-profile-dispatch-envelope-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Solver Proof-Profile Dispatch Envelope

Date: 2026-06-24

This note records the local dispatch/query envelope (status `91`) for the shared
solver proof profile. It exposes the profile receipt (artifact `964210753`,
status `88`) to a local solver-router query: a valid query returns the profile
artifact; any anchor, accepted-mask, or claim-mask mismatch returns `-1`. It is
a local router-readiness receipt, not a new solver and not a theorem promotion.

The shared profile is **cross-family** (SAT/LRAT/FRAT, VeriPB-style PB,
Farkas/QF_LRA, Sounio numeric receipt), so this envelope — unlike the Lorenz
i256 bridges — does **not** assert a `target_integer_width` and makes no native
i256 claim.

## Gate Record

- Module: `stdlib/theorem/solver_proof_profile_dispatch_envelope.sio`
- Tiny runtime test: `tests/run-pass/solver_proof_profile_dispatch_envelope_tiny.sio`
- Imported smoke test: `tests/run-pass/solver_proof_profile_dispatch_envelope_imported.sio`
- Profile artifact (returned by a valid query): `964210753`
- Dispatch mask: `63`
- Envelope fingerprint: `964221344`
- Dispatch artifact fingerprint: `382940617`
- Status code: `123` (free: the lane maintains a coordinated status registry — `88`
  profile, `91` verifier-preflight, `92` runtime-ABI-blocker, `93`
  kernel-replay-router, `94`-`101` private-envelope/lift/family/microkernels,
  `102`-`122` Lorenz bridges — so `91`/`93` are taken and `123` is the first free
  code above the registry).

## Dispatch Mask

The dispatch mask records, for a profile router query:

- bit `1`: profile artifact is `964210753`
- bit `2`: `accepted_profile_mask = 15` (all four proof families accepted)
- bit `4`: `rejected_profile_mask = 48`
- bit `8`: profile status is `88`
- bit `16`: query scope is local (`query_scope_ok = 1`)
- bit `32`: `public_claim_mask`, `formal_theorem_ready`, `global_flowpipe_claim_mask`
  all zero

The complete dispatch mask is therefore `63`. A wrong accepted-mask drops bit
`2` (mask `61`); a nonzero public-claim mask drops bit `32` (mask `31`).

## Query Rule

A valid local query (profile artifact `964210753`, accepted mask `15`, rejected
mask `48`, status `88`, dispatch mask `63`, scope local, all claim masks zero,
limb base `1000000000`) returns the profile artifact `964210753`. Any mismatch
returns `-1` (rejected before any receipt is exposed).

## Envelope Rule

```text
envelope_fp =
  profile_artifact_fp
  + 29*dispatch_mask
  + 31*accepted_profile_mask
  + 37*profile_status
  + 41*bridge_status
  mod 1000000000
```

For the accepted dispatch:

```text
profile_artifact_fp = 964210753
dispatch_mask = 63
accepted_profile_mask = 15
profile_status = 88
bridge_status = 123
envelope_fp = 964221344
```

The envelope fingerprint is derived from (and strictly larger than) the profile
artifact, recording clean lineage `964210753 -> 964221344`.

## Anchors

This envelope anchors the shared solver proof-profile artifact `964210753`,
instance `710284936`, certificate `295748103`, audit `526184309`,
`accepted_profile_mask=15`, `rejected_profile_mask=48`, status `88`. The
dispatch artifact fingerprint is `382940617`.

Status `123` is a local decimal-limb audit status code, not a theorem number,
not an older portfolio version number, and not a public mathematical milestone.

## Boundary

This envelope records:

- `limb_base = 1000000000`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`
- `global_flowpipe_claim_mask = 0`
- no `target_integer_width` assertion (cross-family profile)

The imported smoke remains frontend/typecheck evidence only while the current
imported/native runtime ABI blocker remains active
(`BLK-20260623-madaros-mm-seed-segfault`).

## Claim Boundary

This is not a new SAT/SMT/PB solver, not portfolio/global wiring, not a public
theorem promotion, not a cryptographic acceptance proof, not a Lorenz integrator
or stability/accuracy theorem, not finite-cover/boundary-gluing/flowpipe
certification, not native i256, and not imported/native runtime evidence. It
only exposes the local shared proof-profile receipt to a local solver-router
query.
