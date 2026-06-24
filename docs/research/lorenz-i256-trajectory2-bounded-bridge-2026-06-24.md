<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-trajectory2-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-trajectory2-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Trajectory2 Bounded Bridge

Date: 2026-06-24

This note records a bounded exact two-step trajectory receipt over decimal
i256-like limbs. It composes the local bounded xyz-step bridge twice:

```text
state_1 = xyz_step(state_0)
state_2 = xyz_step(state_1)
```

Each `xyz_step` remains the component-wise `dt = 1` update documented by the
status `115` bridge. This is not adaptive stepping, not an accuracy claim, and
not a continuous-time Lorenz proof.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_trajectory2_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_trajectory2_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_trajectory2_bounded_bridge_imported.sio`
- Artifact fingerprint: `812457306`
- Audit fingerprint: `459028173`
- Instance fingerprint: `690314528`
- Certificate fingerprint: `237861904`
- Status code: `116`

## Checked Trajectory

The imported smoke checks:

```text
state_0 = (0.500000000, 1.625000000, 0.375000000)
state_1 = (11.750000000, 13.812500000, 0.187500000)
state_2 = (32.375000000, 326.796875000, 161.984375000)
```

The first step arithmetic is:

- `x1 = 0.500000000 + 10*(1.625000000-0.500000000) = 11.750000000`
- `y1 = 0.500000000*(28.000000000-0.375000000) = 13.812500000`
- `z1 = 0.375000000 + 0.500000000*1.625000000 - (8/3)*0.375000000 = 0.187500000`

The second step arithmetic is:

- `x2 = 11.750000000 + 10*(13.812500000-11.750000000) = 32.375000000`
- `y2 = 11.750000000*(28.000000000-0.187500000) = 326.796875000`
- `z2 = 0.187500000 + 11.750000000*13.812500000 - (8/3)*0.187500000 = 161.984375000`

The imported smoke also checks rejection propagation: if the first xyz-step
rejects, the second-step query rejects instead of manufacturing a state.

## Anchors

This bridge anchors the bounded xyz-step artifact/audit
`701846329` / `263917508`, status `115`.

Status lineage: `115` is the prior local decimal-limb xyz-step receipt and
`116` is this local decimal-limb trajectory receipt. These are local audit
status codes, not theorem numbers, not the older portfolio v116 entry, and not
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
runtime evidence. It only checks bounded exact fixed-scale arithmetic for two
component-wise `dt = 1` vector updates under the explicit restrictions inherited
from the xyz-step bridge.
