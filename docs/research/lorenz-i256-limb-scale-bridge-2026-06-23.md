<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-limb-scale-bridge-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-limb-scale-bridge-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Limb Scale Bridge

Date: 2026-06-23

This note records a portable decimal-limb scalar-multiplication bridge for the
Lorenz i256 lane. It extends the limb-add bridge toward terms such as
`sigma*(y-x)`, while native executable `i256` values remain unavailable in the
current self-hosted compiler.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_limb_scale_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_limb_scale_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_limb_scale_bridge_imported.sio`
- Artifact fingerprint: `734815269`
- Audit fingerprint: `216903584`
- Instance fingerprint: `508642719`
- Certificate fingerprint: `971304826`
- Status code: `103`

## Checked Scale Cases

The bridge uses four decimal limbs in base `1_000_000_000`.

The Lorenz-shaped tiny case checks a positive delta term:

```text
sigma = 10
y - x = 0.200000000
sigma*(y-x) = 2.000000000
```

represented as:

```text
[200000000, 0, 0, 0] * 10 = [0, 2, 0, 0]
```

The carry case checks:

```text
[150000000, 1, 0, 0] * 10 = [500000000, 11, 0, 0]
```

The tests also reject a wrong expected limb, an invalid scale above `1_000_000`,
and an invalid limb equal to the base.

## Boundary

This bridge records:

- `target_integer_width = 256`
- `limb_base = 1000000000`
- `native_i256_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

This is a portable limb-arithmetic step toward Lorenz high-width receipt
checking. It is not native `i256` execution and not a complete Lorenz stepper.

## Claim Boundary

This is not a Lorenz solver, not a signed interval arithmetic engine, not replay
execution, not replay verification, not a finite-cover certificate, not a
boundary-gluing proof, not a global flowpipe theorem, and not imported/native
runtime evidence. It only checks bounded positive scalar multiplication over
decimal limbs.
