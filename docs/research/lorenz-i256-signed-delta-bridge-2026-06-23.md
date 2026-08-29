<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-signed-delta-bridge-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-signed-delta-bridge-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Signed Delta Bridge

Date: 2026-06-23

This note records a portable signed-delta bridge for the Lorenz i256 lane. It
computes the sign and absolute difference between two nonnegative four-limb
decimal values in base `1_000_000_000`. This is the missing bridge between raw
state limbs and Lorenz terms such as `sigma*(y-x)`.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_signed_delta_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_signed_delta_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_signed_delta_bridge_imported.sio`
- Artifact fingerprint: `492681735`
- Audit fingerprint: `837260419`
- Instance fingerprint: `604913728`
- Certificate fingerprint: `195740286`
- Status code: `104`

## Checked Delta Cases

The imported smoke checks:

- positive delta:
  `[150000000,1,0,0] - [950000000,0,0,0] = +[200000000,0,0,0]`
- negative delta:
  `[950000000,0,0,0] - [150000000,1,0,0] = -[200000000,0,0,0]`
- zero delta:
  `[950000000,0,0,0] - [950000000,0,0,0] = 0`
- borrow delta:
  `[0,1,0,0] - [1,0,0,0] = +[999999999,0,0,0]`

The tiny self-contained smoke separately pins comparison signs and invalid-limb
rejection.

## Boundary

This bridge records:

- `target_integer_width = 256`
- `limb_base = 1000000000`
- `native_i256_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

This is a signed-delta arithmetic bridge over nonnegative limb magnitudes. It is
not native `i256` execution and not a complete interval arithmetic engine.

## Claim Boundary

This is not a Lorenz solver, not a full signed interval system, not replay
execution, not replay verification, not a finite-cover certificate, not a
boundary-gluing proof, not a global flowpipe theorem, and not imported/native
runtime evidence. It only checks bounded sign plus absolute-difference
calculation over decimal limbs.
