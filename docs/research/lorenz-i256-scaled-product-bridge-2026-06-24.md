<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-scaled-product-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-scaled-product-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Scaled Product Bridge

Date: 2026-06-24

This note records a bounded fixed-scale product bridge for Lorenz i256
receipts. It multiplies two nonnegative two-limb decimal values and rescales by
the decimal limb base:

```text
scaled_product(a,b) = (a*b) / 1000000000
```

This is the first explicit arithmetic bridge toward the Lorenz `x*z` product
term. It is intentionally bounded to two input limbs per operand and rejects
products whose low-limb multiplication is not exactly divisible by the base, so
accepted cases do not silently truncate precision beyond the 9-decimal scale.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_scaled_product_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_scaled_product_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_scaled_product_bridge_imported.sio`
- Artifact fingerprint: `920174638`
- Audit fingerprint: `481036275`
- Instance fingerprint: `703918426`
- Certificate fingerprint: `268540197`
- Status code: `109`

## Checked Product Cases

The imported smoke checks:

- low-limb product:
  `0.250000000 * 0.400000000 = 0.100000000`
- mixed-limb product:
  `1.500000000 * 2.250000000 = 3.375000000`
- carry product:
  `2.500000000 * 3.500000000 = 8.750000000`
- inexact rejection:
  `0.333333333 * 0.333333333` is rejected because the result would require
  precision beyond the accepted 9-decimal scale
- upper-limb rejection:
  nonzero limbs above the two-limb input window are rejected

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

This is not a general i256 multiplier, not signed multiplication, not interval
arithmetic, not a complete `x*z` Lorenz product over arbitrary four-limb
operands, not a complete Lorenz stepper, not replay verification, not a
finite-cover certificate, not a boundary-gluing proof, not a global flowpipe
theorem, not native `i256` execution, and not imported/native runtime evidence.
It only checks bounded exact fixed-scale products for two-limb nonnegative
operands.
