<!-- docs:meta
topic_id: repo.docs.research.runtime-abi-gate-blocker-query-envelope-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.runtime-abi-gate-blocker-query-envelope-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Runtime ABI Gate-Blocker Query Envelope

Date: 2026-06-24

This note records the local query envelope (status `93`) for the runtime ABI
gate blocker. It lets a local router ask "is the imported/native runtime ABI
blocker active / is imported-runtime evidence gated?" and get the blocker
receipt (artifact `584291376`). A valid query returns the blocker artifact;
mismatches return `-1`. This records the **blocker state** as router-queryable
metadata. It does **not** fix the blocker (`BLK-20260623-madaros-mm-seed-segfault`,
owned by the madaros native-lowering lane) and is **not** runtime evidence.

## Gate Record

- Module: `stdlib/safety/runtime_abi_gate_blocker_query_envelope.sio`
- Tiny runtime test: `tests/run-pass/runtime_abi_gate_blocker_query_envelope_tiny.sio`
- Imported smoke test: `tests/run-pass/runtime_abi_gate_blocker_query_envelope_imported.sio`
- Blocker artifact (returned by a valid query): `584291376`
- Query mask: `63`
- Envelope fingerprint: `584298578`
- Query artifact fingerprint: `426913058`
- Status code: `124` (free: the lane maintains a coordinated status registry — `88`
  profile, `91` verifier-preflight, `92` runtime-ABI-blocker, `93`
  kernel-replay-router, `94`-`101` private-envelope/lift/family/microkernels,
  `102`-`122` Lorenz bridges, `123` solver proof-profile dispatch — so `93` is
  taken and `124` is the first free code above the registry).

## Query Mask

The query mask records, for a runtime-ABI-state router query:

- bit `1`: blocker artifact is `584291376`
- bit `2`: `known_madaros_runtime_blocker_mask = 7` (blocker ACTIVE — three
  independent blockers; this encodes `BLK-20260623` and friends, it is not
  evidence)
- bit `4`: `runtime_evidence_private_level = 2` (ABI gate unsatisfied)
- bit `8`: `imported_runtime_missing_mask = 7` (imported runtime gated)
- bit `16`: `imported_runtime_promotion_mask = 0`
- bit `32`: `public_claim_mask`, `formal_theorem_ready`, `global_flowpipe_claim_mask`,
  `boundary_gluing_proof_mask`, `finite_cover_certificate_mask` all zero

The complete query mask is therefore `63`. An inactive blocker
(`known_madaros_runtime_blocker_mask = 0`) drops bit `2` (mask `61`); a nonzero
imported-runtime-promotion mask drops bit `16` (mask `47`).

## Query Rule

A valid local query (blocker artifact `584291376`, blocker active `7`, level
`2`, imported-missing `7`, query mask `63`, no promotion, all claim masks zero,
limb base `1000000000`) returns the blocker artifact `584291376`. Any mismatch
returns `-1`. The receipt only re-exposes the blocker state; it does not assert
the blocker is resolved.

## Envelope Rule

```text
envelope_fp =
  blocker_artifact_fp
  + 29*query_mask
  + 31*known_madaros_runtime_blocker_mask
  + 37*runtime_evidence_private_level
  + 41*bridge_status
  mod 1000000000
```

For the blocker-state query:

```text
blocker_artifact_fp = 584291376
query_mask = 63
known_madaros_runtime_blocker_mask = 7
runtime_evidence_private_level = 2
bridge_status = 124
envelope_fp = 584298578
```

The envelope fingerprint is derived from (and strictly larger than) the blocker
artifact, recording clean lineage `584291376 -> 584298578`.

## Anchors

This query envelope anchors the runtime ABI gate-blocker artifact `584291376`,
instance `265813904`, certificate `718406529`, audit `936740152`, the shared
solver proof-profile artifact `964210753`, and the Lorenz replay-verifier
preflight `391742608`. The query artifact fingerprint is `426913058`.

Status `124` is a local decimal-limb audit status code, not a theorem number,
not an older portfolio version number, and not a public mathematical milestone.
`known_madaros_runtime_blocker_mask = 7` records three active blockers as state,
**not** as evidence — lifting it requires the multimodule witness and imported
solver runtime masks to become positive under a real compiler artifact.

## Boundary

This query envelope records:

- `limb_base = 1000000000`
- `known_madaros_runtime_blocker_mask = 7` (blocker state, not evidence)
- `runtime_evidence_private_level = 2` (ABI unsatisfied, not a proof tier)
- `imported_runtime_promotion_mask = 0`
- `public_claim_mask = 0`, `formal_theorem_ready = 0`,
  `global_flowpipe_claim_mask = 0`, `boundary_gluing_proof_mask = 0`,
  `finite_cover_certificate_mask = 0`

The imported smoke remains frontend/typecheck evidence only while the current
imported/native runtime ABI blocker remains active
(`BLK-20260623-madaros-mm-seed-segfault`).

## Claim Boundary

This is not a fix for `BLK-20260623-madaros-mm-seed-segfault`, not runtime
evidence, not imported/native runtime promotion, not portfolio/global wiring,
not a public theorem, not a cryptographic proof, not a Lorenz integrator or
theorem, not finite-cover/boundary-gluing/flowpipe certification, not native
i256, and not solver runtime evidence. It only records the runtime ABI blocker
state as local router-queryable metadata.
