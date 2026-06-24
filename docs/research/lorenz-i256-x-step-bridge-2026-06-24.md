<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-x-step-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-x-step-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 X-Step Bridge

Date: 2026-06-24

This note records a portable one-axis Lorenz x-step receipt over four
base-`1_000_000_000` limbs. It composes the existing `sigma*(y-x)` bridge with
a bounded `dt = 1` state update:

```text
x_next = x + sigma*(y-x)
```

This is a toy arithmetic receipt, not a scientific integrator.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_x_step_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_x_step_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_x_step_bridge_imported.sio`
- Artifact fingerprint: `645218970`
- Audit fingerprint: `184760392`
- Instance fingerprint: `927405816`
- Certificate fingerprint: `360918274`
- Status code: `106`

## Checked Step Cases

The imported smoke checks:

- positive update:
  `x=0.950000000`, `y=1.150000000`
  gives `sigma*(y-x)=+2.000000000`, so
  `x_next=2.950000000`
- negative update:
  `x=3.150000000`, `y=2.950000000`
  gives `sigma*(y-x)=-2.000000000`, so
  `x_next=1.150000000`
- zero update:
  equal `x` and `y` leaves `x_next=x`
- underflow rejection:
  a negative term whose magnitude exceeds the nonnegative `x` state is rejected

The imported smoke also anchors the sigma x-term bridge
`318704692` / `729516083`.

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

This is not a complete Lorenz stepper, not a stable numerical integrator, not
signed interval integration, not replay execution, not replay verification, not
a finite-cover certificate, not a boundary-gluing proof, not a global flowpipe
theorem, not native `i256` execution, and not imported/native runtime evidence.
It only checks a bounded one-axis `dt=1` arithmetic transition over decimal
limbs.
