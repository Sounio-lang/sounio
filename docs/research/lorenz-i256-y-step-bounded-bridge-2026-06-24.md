<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-y-step-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-y-step-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Y-Step Bounded Bridge

Date: 2026-06-24

This note records a bounded `dt = 1` y-axis step:

```text
y_next = y + (x*(rho-z)-y)
```

It composes the bounded y-derivative bridge with decimal-limb addition and
guarded subtraction. The derivative bridge already carries the restrictions:
`rho = 28`, `z <= rho`, nonnegative two-limb fixed-scale product operands, and
exact 9-decimal-scale products only.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_y_step_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_y_step_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_y_step_bounded_bridge_imported.sio`
- Artifact fingerprint: `286140975`
- Audit fingerprint: `653928410`
- Instance fingerprint: `731064852`
- Certificate fingerprint: `508216749`
- Status code: `111`

## Checked Step Cases

The imported smoke checks:

- positive step:
  `x=1.500000000`, `z=26.000000000`, `y=1.000000000`
  gives derivative `+2.000000000`, so `y_next=3.000000000`
- negative step:
  `x=0.500000000`, `z=27.000000000`, `y=1.000000000`
  gives derivative `-0.500000000`, so `y_next=0.500000000`
- zero step:
  `x=0.500000000`, `z=27.000000000`, `y=0.500000000`
  gives zero derivative and preserves `y`
- underflow rejection:
  a negative derivative with magnitude greater than the nonnegative `y` state is
  rejected

The imported smoke also anchors the bounded y-derivative bridge
`742619083` / `396508712` and the limb-add bridge
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

This is not a complete Lorenz stepper, not a general y-step over arbitrary
signed states, not a general four-limb i256 product, not signed interval
integration, not replay execution, not replay verification, not a finite-cover
certificate, not a boundary-gluing proof, not a global flowpipe theorem, not
native `i256` execution, and not imported/native runtime evidence. It only
checks bounded exact fixed-scale arithmetic for `y + (x*(rho-z)-y)` under the
explicit derivative-bridge restrictions.
