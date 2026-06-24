<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-z-derivative-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-z-derivative-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Z-Derivative Bounded Bridge

Date: 2026-06-24

This note records a bounded composition bridge for the Lorenz z-derivative
fragment:

```text
x*y - beta*z
```

It composes:

- bounded fixed-scale product for `x*y`
- exact `beta*z` bridge with `beta = 8/3`
- signed delta for `x*y - beta*z`

The inherited restrictions are important: the product bridge accepts only
nonnegative two-limb operands with exact 9-decimal-scale products, and the
`beta*z` bridge accepts only exact decimal-limb divisions by `3`.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_z_derivative_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_z_derivative_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_z_derivative_bounded_bridge_imported.sio`
- Artifact fingerprint: `471806392`
- Audit fingerprint: `806135247`
- Instance fingerprint: `294718650`
- Certificate fingerprint: `730462819`
- Status code: `113`

## Checked Term Cases

The imported smoke checks:

- positive term:
  `x=1.500000000`, `y=2.250000000`, `z=0.375000000`
  gives `x*y=3.375000000`, `beta*z=1.000000000`, and term `+2.375000000`
- negative term:
  `x=0.500000000`, `y=0.500000000`, `z=0.375000000`
  gives `x*y=0.250000000`, `beta*z=1.000000000`, and term `-0.750000000`
- zero term:
  `x=0.500000000`, `y=2.000000000`, `z=0.375000000`
  gives `x*y=1.000000000`, `beta*z=1.000000000`, and zero term
- inexact product rejection:
  `x=0.333333333`, `y=0.333333333` is rejected by the product bridge
- inexact beta rejection:
  `z=0.100000000` is rejected by the exact `beta*z` bridge

The imported smoke also anchors the scaled-product bridge
`920174638` / `481036275`, the exact beta-z bridge
`639182704` / `174905863`, and the signed-delta bridge
`492681735` / `837260419`.

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

This is not a complete Lorenz z-step, not a complete Lorenz stepper, not a
general four-limb i256 product, not rounded rational arithmetic, not signed
interval integration, not replay execution, not replay verification, not a
finite-cover certificate, not a boundary-gluing proof, not a global flowpipe
theorem, not native `i256` execution, and not imported/native runtime evidence.
It only checks bounded exact fixed-scale arithmetic for `x*y-beta*z` under the
explicit product and beta-z restrictions.
