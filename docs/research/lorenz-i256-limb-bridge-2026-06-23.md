<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-limb-bridge-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-limb-bridge-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Limb Bridge

Date: 2026-06-23

This note records a portable high-width arithmetic bridge for the Lorenz i256
lane. The current self-hosted compiler does not yet provide executable native
`i256` values, so this bridge uses four decimal limbs in base `1_000_000_000`
to check a tiny high-precision state increment.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_limb_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_limb_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_limb_bridge_imported.sio`
- Artifact fingerprint: `681429570`
- Audit fingerprint: `352908746`
- Instance fingerprint: `905714263`
- Certificate fingerprint: `247816039`
- Status code: `102`

## Checked Increment

The self-contained and imported tests check decimal-limb addition with carry:

```text
0.950000000 + 0.200000000 = 1.150000000
```

represented as:

```text
[950000000, 0, 0, 0] + [200000000, 0, 0, 0]
  = [150000000, 1, 0, 0]
```

The tests also reject a wrong expected low limb and reject an invalid limb equal
to the base.

## High-Width Boundary

This bridge records:

- `target_integer_width = 256`
- `limb_base = 1000000000`
- `native_i256_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

That means it is a portable limb-arithmetic seed for high-width Lorenz receipts,
not proof that native `i256` arithmetic is available in executable Sounio code.

## Claim Boundary

This is not a Lorenz solver, not replay execution, not replay verification, not
a finite-cover certificate, not a boundary-gluing proof, not a global flowpipe
theorem, and not imported/native runtime evidence. It is one small high-width
arithmetic bridge that gives the Lorenz lane an executable path toward i256-like
receipt checking while the native type path remains blocked.
