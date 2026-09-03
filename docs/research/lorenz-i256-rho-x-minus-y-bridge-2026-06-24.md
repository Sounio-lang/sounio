<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-rho-x-minus-y-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-rho-x-minus-y-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Rho X Minus Y Bridge

Date: 2026-06-24

This note records a portable composition bridge for the Lorenz y-derivative
fragment `rho*x - y`, with `rho = 28`, restricted to the `z = 0` fragment.
It composes two existing high-width limb bridges:

- scalar multiplication: multiplying `x` by `rho = 28`
- signed delta: sign plus absolute `rho*x - y` over four base-`1_000_000_000` limbs

## Gate Record

- Module: `stdlib/systems/lorenz_i256_rho_x_minus_y_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_rho_x_minus_y_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_rho_x_minus_y_bridge_imported.sio`
- Artifact fingerprint: `570239184`
- Audit fingerprint: `908416527`
- Instance fingerprint: `314762905`
- Certificate fingerprint: `726591438`
- Status code: `107`

## Checked Term Cases

The imported smoke checks:

- positive term:
  `x=[100000000,0,0,0]`, `y=[800000000,1,0,0]`
  gives `rho*x=[800000000,2,0,0]`, so `rho*x-y=+[0,1,0,0]`
- negative term:
  `x=[50000000,0,0,0]`, `y=[0,2,0,0]`
  gives `rho*x=[400000000,1,0,0]`, so `rho*x-y=-[600000000,0,0,0]`
- zero term:
  `x=[50000000,0,0,0]`, `y=[400000000,1,0,0]`
  gives `rho*x-y=0`
- invalid limb rejection:
  an input limb equal to the base is rejected by the scaled-limb path

The imported smoke also anchors the scale bridge
`734815269` / `216903584` and the signed-delta bridge
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

This is not a complete Lorenz stepper, not the full y-derivative term
`x*(rho-z)-y`, not a limb-by-limb product kernel, not signed interval
integration, not replay execution, not replay verification, not a finite-cover
certificate, not a boundary-gluing proof, not a global flowpipe theorem, not
native `i256` execution, and not imported/native runtime evidence. It only
checks the bounded `rho*x-y` composition over decimal limbs under the explicit
`z=0` restriction.
