<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-xyz-step-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-xyz-step-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 XYZ-Step Bounded Bridge

Date: 2026-06-24

This note records a bounded component-wise `dt = 1` Lorenz step over one shared
input state:

```text
x_next = x + sigma*(y-x)
y_next = y + (x*(rho-z)-y)
z_next = z + (x*y - beta*z)
```

It composes the existing bounded x-step, y-step, and z-step bridges. Each axis
uses the original input state; this is not a sequential update where later axes
consume earlier updated coordinates.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_xyz_step_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_xyz_step_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_xyz_step_bounded_bridge_imported.sio`
- Artifact fingerprint: `701846329`
- Audit fingerprint: `263917508`
- Instance fingerprint: `538104672`
- Certificate fingerprint: `916470253`
- Status code: `115`

## Checked Step Case

The imported smoke checks the shared input:

```text
x = 1.500000000
y = 2.250000000
z = 0.375000000
```

The component-wise updates are:

- `x_next = 1.500000000 + 10*(2.250000000-1.500000000) = 9.000000000`
- `y_next = 2.250000000 + 1.500000000*(28.000000000-0.375000000)-2.250000000 = 41.437500000`
- `z_next = 0.375000000 + 1.500000000*2.250000000 - (8/3)*0.375000000 = 2.750000000`

The imported smoke also checks axis-specific rejection propagation for x, y,
and z underflow sentinels.

## Anchors

This bridge anchors:

- x-step artifact/audit `645218970` / `184760392`, status `106`
- y-step artifact/audit `286140975` / `653928410`, status `111`
- z-step artifact/audit `902461738` / `317604925`, status `114`

Status lineage: `106`, `111`, and `114` are local prior axis-step audit codes;
`115` is this local vector-step receipt. These are not theorem numbers or
public mathematical milestones.

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

This is not a complete Lorenz integrator, not a stability or accuracy theorem,
not arbitrary signed-state coverage, not a general four-limb i256 product, not
adaptive stepping, not interval integration, not replay execution, not replay
verification, not a finite-cover certificate, not a boundary-gluing proof, not a
global flowpipe theorem, not native `i256` execution, and not imported/native
runtime evidence. It only checks bounded exact fixed-scale arithmetic for one
component-wise `dt = 1` vector update under the explicit restrictions inherited
from the three axis-step bridges.
