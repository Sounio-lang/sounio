<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-z-step-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-z-step-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Z-Step Bounded Bridge

Date: 2026-06-24

This note records a bounded `dt = 1` z-axis step:

```text
z_next = z + (x*y - beta*z)
```

It composes the bounded z-derivative bridge with decimal-limb addition and
guarded subtraction. The derivative bridge already carries the restrictions:
nonnegative two-limb fixed-scale product operands, exact 9-decimal-scale
products only, and exact `beta*z` divisibility for `beta = 8/3`.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_z_step_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_z_step_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_z_step_bounded_bridge_imported.sio`
- Artifact fingerprint: `902461738`
- Audit fingerprint: `317604925`
- Instance fingerprint: `658209431`
- Certificate fingerprint: `149372806`
- Status code: `114`

## Checked Step Cases

The imported smoke checks:

- positive step:
  `x=1.500000000`, `y=2.250000000`, `z=0.375000000`
  gives derivative `+2.375000000`, so `z_next=2.750000000`
- negative step:
  `x=1.500000000`, `y=2.250000000`, `z=1.500000000`
  gives derivative `-0.625000000`, so `z_next=0.875000000`
- zero step:
  `x=0.500000000`, `y=2.000000000`, `z=0.375000000`
  gives zero derivative and preserves `z`
- underflow rejection:
  a negative derivative with magnitude greater than the nonnegative `z` state is
  rejected

The imported smoke also anchors the bounded z-derivative bridge
`471806392` / `806135247` and the limb-add bridge
`681429570` / `352908746`.

Status lineage: `102` is the prior decimal limb-add bridge, `113` is the
bounded z-derivative bridge, and `114` is this z-step receipt. These are local
audit status codes, not theorem numbers or public mathematical milestones.

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

This is not a complete Lorenz stepper, not a general z-step over arbitrary
signed states, not a general four-limb i256 product, not signed interval
integration, not replay execution, not replay verification, not a finite-cover
certificate, not a boundary-gluing proof, not a global flowpipe theorem, not
native `i256` execution, and not imported/native runtime evidence. It only
checks bounded exact fixed-scale arithmetic for `z + (x*y - beta*z)` under the
explicit derivative-bridge restrictions.
