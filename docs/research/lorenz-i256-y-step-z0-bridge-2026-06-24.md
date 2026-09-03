<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-y-step-z0-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-y-step-z0-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Y-Step Z0 Bridge

Date: 2026-06-24

This note records a portable y-axis toy step for the restricted Lorenz fragment
with `z = 0` and `dt = 1`:

```text
y_next = y + (rho*x - y)
```

It composes existing high-width limb bridges:

- `rho*x-y` term bridge with `rho = 28`
- decimal-limb addition for positive term updates
- signed delta for negative term updates

## Gate Record

- Module: `stdlib/systems/lorenz_i256_y_step_z0_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_y_step_z0_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_y_step_z0_bridge_imported.sio`
- Artifact fingerprint: `760481923`
- Audit fingerprint: `135972604`
- Instance fingerprint: `619384257`
- Certificate fingerprint: `847205316`
- Status code: `108`

## Checked Step Cases

The imported smoke checks:

- positive step:
  `x=[100000000,0,0,0]`, `y=[800000000,1,0,0]`
  gives `rho*x-y=+[0,1,0,0]`, so `y_next=[800000000,2,0,0]`
- negative step:
  `x=[50000000,0,0,0]`, `y=[0,2,0,0]`
  gives `rho*x-y=-[600000000,0,0,0]`, so `y_next=[400000000,1,0,0]`
- zero step:
  `x=[50000000,0,0,0]`, `y=[400000000,1,0,0]`
  gives zero term and preserves `y`
- invalid limb rejection:
  an input limb equal to the base is rejected by the upstream term path

The imported smoke also anchors the rho term bridge
`570239184` / `908416527` and the limb-add bridge
`681429570` / `352908746`.

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
checks the bounded `z=0`, `dt=1`, `y + (rho*x-y)` toy update over decimal
limbs.
