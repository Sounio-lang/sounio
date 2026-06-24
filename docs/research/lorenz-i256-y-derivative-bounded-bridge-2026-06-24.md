<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-y-derivative-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-y-derivative-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Y-Derivative Bounded Bridge

Date: 2026-06-24

This note records a bounded composition bridge for the Lorenz y-derivative
fragment:

```text
x*(rho-z)-y
```

The bridge fixes `rho = 28` and accepts only cases where `z <= rho` and the
derived `rho-z` value fits the existing two-limb fixed-scale product bridge.
It composes:

- signed delta for `rho-z`
- fixed-scale product for `x*(rho-z)`
- signed delta for `x*(rho-z)-y`

## Gate Record

- Module: `stdlib/systems/lorenz_i256_y_derivative_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_y_derivative_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_y_derivative_bounded_bridge_imported.sio`
- Artifact fingerprint: `742619083`
- Audit fingerprint: `396508712`
- Instance fingerprint: `815204637`
- Certificate fingerprint: `204973856`
- Status code: `110`

## Checked Term Cases

The imported smoke checks:

- positive term:
  `x=1.500000000`, `z=26.000000000`, `y=1.000000000`
  gives `rho-z=2.000000000`, `x*(rho-z)=3.000000000`, and term `+2.000000000`
- negative term:
  `x=0.500000000`, `z=27.000000000`, `y=1.000000000`
  gives `rho-z=1.000000000`, `x*(rho-z)=0.500000000`, and term `-0.500000000`
- zero term:
  `x=0.500000000`, `z=27.000000000`, `y=0.500000000`
  gives zero term
- `z > rho` rejection:
  the bounded positive-product bridge rejects `z=29.000000000`
- inexact product rejection:
  `x=0.333333333`, `rho-z=0.333333333` is rejected because the product would
  require precision beyond the accepted 9-decimal scale

The imported smoke also anchors the scaled-product bridge
`920174638` / `481036275` and the signed-delta bridge
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

This is not a general Lorenz y-derivative over arbitrary signed states, not a
complete Lorenz stepper, not a general four-limb i256 product, not signed
interval integration, not replay execution, not replay verification, not a
finite-cover certificate, not a boundary-gluing proof, not a global flowpipe
theorem, not native `i256` execution, and not imported/native runtime evidence.
It only checks bounded exact fixed-scale arithmetic for `x*(rho-z)-y` under
the explicit `z <= rho` and two-limb product restrictions.
