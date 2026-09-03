<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-trajectory2-replay-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-trajectory2-replay-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Trajectory2 Replay Bounded Bridge

Date: 2026-06-24

This note records a bounded replay verifier for a supplied two-step Lorenz
trajectory certificate over decimal i256-like limbs. Unlike the trajectory
receipt that computes `state_1` and `state_2`, this bridge receives all three
states and accepts only when both supplied transitions replay exactly:

```text
state_1 == xyz_step(state_0)
state_2 == xyz_step(state_1)
```

The underlying `xyz_step` remains the component-wise `dt = 1` update documented
by the local status `115` bridge.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_trajectory2_replay_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_trajectory2_replay_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_trajectory2_replay_bounded_bridge_imported.sio`
- Artifact fingerprint: `734260981`
- Audit fingerprint: `605913274`
- Instance fingerprint: `482709136`
- Certificate fingerprint: `193840625`
- Status code: `117`

## Checked Replay

The imported smoke accepts the supplied certificate:

```text
state_0 = (0.500000000, 1.625000000, 0.375000000)
state_1 = (11.750000000, 13.812500000, 0.187500000)
state_2 = (32.375000000, 326.796875000, 161.984375000)
```

It also rejects:

- a first-step certificate with `state_1.x0` changed by one limb unit
- a second-step certificate with `state_2.x0` changed by one limb unit
- an invalid first step whose bounded xyz-step update rejects

This turns the two-step arithmetic receipt into a small proof-checker surface:
the checker does not trust supplied intermediate/final states unless replay
recomputes them.

## Anchors

This bridge anchors:

- bounded trajectory2 artifact/audit `812457306` / `459028173`, status `116`
- bounded xyz-step artifact/audit `701846329` / `263917508`, status `115`

Status lineage: `115`, `116`, and `117` are local decimal-limb audit status
codes, not theorem numbers, not older portfolio version numbers, and not public
mathematical milestones.

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
adaptive stepping, not interval integration, not a finite-cover certificate, not
a boundary-gluing proof, not a global flowpipe theorem, not native `i256`
execution, and not imported/native runtime evidence. It only checks bounded
exact fixed-scale replay for a supplied two-step certificate under the explicit
restrictions inherited from the xyz-step bridge.
