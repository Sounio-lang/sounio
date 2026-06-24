<!-- docs:meta
topic_id: repo.docs.research.pb-highwidth-microkernel-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pb-highwidth-microkernel-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# PB High-Width Microkernel

Date: 2026-06-23

This note records a bounded pseudo-Boolean proof-checker seed for the
PB/VeriPB side of the solver/proof-checker family. The executable kernel checks
one three-literal PB row, detects a conflicting assignment, and detects one
forced literal by asking whether the row can still reach its threshold with the
candidate literal set false.

## Gate Record

- Module: `stdlib/theorem/pb_highwidth_microkernel.sio`
- Tiny runtime test: `tests/run-pass/pb_highwidth_microkernel_tiny.sio`
- Imported smoke test: `tests/run-pass/pb_highwidth_microkernel_imported.sio`
- Artifact fingerprint: `671902843`
- Audit fingerprint: `238570416`
- Instance fingerprint: `914035627`
- Certificate fingerprint: `506781239`
- Status code: `99`

## Checked Row

The tiny row is:

```text
2*x + 2*y + z >= 4
```

The self-contained and imported tests check:

- `x=1, y=1, z=0` satisfies the row.
- With `x=1` and `y,z` unknown, `y` is forced because the maximum sum with
  `y=false` is `3`, below the threshold `4`.
- `x=1, y=0, z=1` is a conflict because the sum is `3`.
- `z` is not forced in the same partial state because the row can still reach
  `4` with `x=1, y=1, z=false`.
- `x=1, y=1, z=0` is not a conflict.

## High-Width Boundary

The receipt records `target_integer_width = 128`, but the current executable
arithmetic is deliberately bounded over `i64`. The local source tree already
documents that the self-hosted compiler does not yet accept `i128` value
declarations. Therefore this microkernel keeps:

- `native_i128_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

This is a target-width bridge, not proof that native `i128` arithmetic is
available in runtime-checked Sounio code.

## Claim Boundary

This is not a full VeriPB parser, not a cutting-planes checker, not a PB solver,
not imported/native runtime evidence, and not a public theorem. It is one small
runnable PB row checker that gives the solver lane a concrete PB/VeriPB-shaped
kernel next to the SAT/RUP seed.
