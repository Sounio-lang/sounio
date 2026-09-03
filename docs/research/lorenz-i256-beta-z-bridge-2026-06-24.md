<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-beta-z-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-beta-z-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Beta Z Bridge

Date: 2026-06-24

This note records an exact decimal-limb bridge for the Lorenz `beta*z` term
with classical `beta = 8/3`.

The bridge computes:

```text
beta*z = (8*z)/3
```

It first multiplies the four-limb decimal value by `8` using the existing
small-scalar limb bridge, then divides the result by `3` using high-to-low
long division in base `1000000000`. If the final remainder is nonzero, the
case is rejected. No rounding is performed.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_beta_z_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_beta_z_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_beta_z_bridge_imported.sio`
- Artifact fingerprint: `639182704`
- Audit fingerprint: `174905863`
- Instance fingerprint: `582607319`
- Certificate fingerprint: `918340256`
- Status code: `112`

## Checked Term Cases

The imported smoke checks:

- `z=1.500000000`: `8*z=12.000000000`, `(8*z)/3=4.000000000`
- `z=0.375000000`: `8*z=3.000000000`, `(8*z)/3=1.000000000`
- `z=2.250000000`: `8*z=18.000000000`, `(8*z)/3=6.000000000`
- inexact rejection:
  `z=0.100000000` gives `8*z=0.800000000`, which is not divisible by `3`
  at the accepted decimal-limb scale
- invalid limb rejection:
  an input limb equal to the base is rejected upstream by the scale path

The imported smoke also anchors the limb-scale bridge
`734815269` / `216903584`.

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

This is not a complete Lorenz z-derivative, not `x*y-beta*z`, not a complete
Lorenz stepper, not rounded rational arithmetic, not signed interval
integration, not replay execution, not replay verification, not a finite-cover
certificate, not a boundary-gluing proof, not a global flowpipe theorem, not
native `i256` execution, and not imported/native runtime evidence. It only
checks exact accepted `8/3*z` decimal-limb cases and explicit rejection for
non-divisible inputs.
