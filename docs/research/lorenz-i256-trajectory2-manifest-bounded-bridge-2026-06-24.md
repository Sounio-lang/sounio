<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-trajectory2-manifest-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-trajectory2-manifest-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Trajectory2 Manifest Bounded Bridge

Date: 2026-06-24

This note records a compact manifest receipt for the bounded two-step Lorenz
certificate lane. The manifest binds the replay/certificate anchors to solver
metadata:

- checker family: `73` (`lorenz_i256_bounded_replay`)
- checker kind: `118` (`trajectory2_certificate_bounded`)
- step count: `2`
- integer width: `256`
- limb base: `1000000000`

The manifest can be computed directly from already-reviewed anchors or through
`lorenz_i256_trajectory2_manifest_bounded_from_certificate`, which first
recomputes the status `118` certificate fingerprint from the supplied limbs.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_trajectory2_manifest_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_trajectory2_manifest_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_trajectory2_manifest_bounded_bridge_imported.sio`
- Artifact fingerprint: `846135279`
- Audit fingerprint: `579264308`
- Instance fingerprint: `314708592`
- Certificate fingerprint: `860125473`
- Manifest fingerprint: `294601254`
- Status code: `119`

## Manifest Rule

```text
manifest_fp =
  trajectory_certificate_fp
  + 19*replay_fp
  + 23*trajectory2_fp
  + 29*checker_family_id
  + 31*checker_kind_id
  + 37*step_count
  + 41*target_integer_width
  mod 1000000000
```

For the accepted certificate lane:

```text
trajectory_certificate_fp = 657108232
replay_fp = 734260981
trajectory2_fp = 812457306
checker_family_id = 73
checker_kind_id = 118
step_count = 2
target_integer_width = 256
manifest_fp = 294601254
```

The imported smoke also checks that a replay mismatch is rejected before
manifest emission, and that incorrect family/kind metadata is rejected.

## Anchors

This bridge anchors the bounded certificate artifact/audit
`918274650` / `341902786`, status `118`.

Status lineage: `117`, `118`, and `119` are local decimal-limb audit status
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

This is not portfolio wiring, not a cryptographic manifest, not a complete
Lorenz integrator, not a stability or accuracy theorem, not arbitrary
signed-state coverage, not a general four-limb i256 product, not adaptive
stepping, not interval integration, not a finite-cover certificate, not a
boundary-gluing proof, not a global flowpipe theorem, not native `i256`
execution, and not imported/native runtime evidence. It only checks compact
manifest metadata for a bounded exact two-step replay certificate under the
explicit restrictions inherited from the certificate bridge.
