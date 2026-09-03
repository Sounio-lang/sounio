<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-trajectory2-acceptance-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-trajectory2-acceptance-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Trajectory2 Acceptance Bounded Bridge

Date: 2026-06-24

This note records a local acceptance envelope for the bounded two-step Lorenz
manifest lane. It is meant as a small solver-router readiness receipt: the
manifest is accepted for the local bounded replay checker profile, while all
native/runtime/public/formal theorem masks remain zero.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_trajectory2_acceptance_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_trajectory2_acceptance_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_trajectory2_acceptance_bounded_bridge_imported.sio`
- Artifact fingerprint: `769418205`
- Audit fingerprint: `487690132`
- Instance fingerprint: `926035714`
- Certificate fingerprint: `158407263`
- Readiness fingerprint: `9371853`
- Acceptance mask: `31`
- Status code: `120`

## Acceptance Mask

The acceptance mask records:

- bit `1`: manifest fingerprint is `294601254`
- bit `2`: certificate/replay gate is present
- bit `4`: checker family is `73`
- bit `8`: checker kind is `118`
- bit `16`: native/runtime/public/formal theorem masks are all zero

The complete local acceptance mask is therefore `31`.

## Readiness Rule

```text
readiness_fp =
  manifest_artifact_fp
  + 7*manifest_fp
  + 11*certificate_bridge_fp
  + 13*acceptance_mask
  + 17*checker_family_id
  + 19*checker_kind_id
  + 23*bridge_status
  mod 1000000000
```

For the accepted bounded lane:

```text
manifest_artifact_fp = 846135279
manifest_fp = 294601254
certificate_bridge_fp = 918274650
acceptance_mask = 31
checker_family_id = 73
checker_kind_id = 118
bridge_status = 120
readiness_fp = 9371853
```

The imported smoke checks both successful readiness and rejection when metadata
or claim masks are not compatible with the scoped local profile.

## Anchors

This bridge anchors the bounded manifest artifact/audit
`846135279` / `579264308`, status `119`, and the bounded certificate artifact
`918274650`, status `118`.

Status lineage: `118`, `119`, and `120` are local decimal-limb audit status
codes, not theorem numbers, not older portfolio version numbers, and not public
mathematical milestones.

## Boundary

This bridge records:

- `target_integer_width = 256`
- `limb_base = 1000000000`
- `native_i256_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

The imported smoke remains frontend/typecheck evidence only while the current
imported/native runtime ABI blocker remains active.

## Claim Boundary

This is not portfolio wiring, not a cryptographic acceptance proof, not a
complete Lorenz integrator, not a stability or accuracy theorem, not arbitrary
signed-state coverage, not a general four-limb i256 product, not adaptive
stepping, not interval integration, not a finite-cover certificate, not a
boundary-gluing proof, not a global flowpipe theorem, not native `i256`
execution, and not imported/native runtime evidence. It only checks local
solver-router readiness metadata for a bounded exact two-step replay manifest
under the explicit restrictions inherited from the manifest bridge.
