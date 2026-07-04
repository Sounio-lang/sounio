<!-- docs:meta
topic_id: repo.docs.research.madaros-v2-s2-contract-scaffold-2026-07-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.madaros-v2-s2-contract-scaffold-2026-07-04
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

---
title: "Madaros v2 S2 contract scaffold"
date: 2026-07-04
status: active
owner: Codex
---

# Madaros v2 S2 contract scaffold

Status: deterministic S2 audit/contract scaffold is implemented and locally
gated. It is not a claim that current Madaros emits native typed HIR/THIR.

## Implemented surface

- `scripts/dev/madaros_v2_s2_receipt.py`
- `scripts/dev/madaros_v2_s2_gate.sh`
- `self-hosted/compiler/madaros_v2_s2_receipt.sio`
- `bin/madaros s2-receipt <source.sio|project-dir> [--out-dir OUT]`
- `tests/madaros/v2_s2/manifest.tsv`

The receipt schema is `madaros.v2.s2.receipt/0.1`.

Each case emits:

- `<case>.s2.receipt.json`
- `<case>.s2.public_symbols.tsv`
- `<case>.s2.import_audit.tsv`
- `<case>.s2.effects.tsv`
- `<case>.s2.refinements.tsv`
- `<case>.s2.epistemic_decls.tsv`
- `<case>.s2.diagnostics.json`

The receipt links to a deterministic S1 receipt and records:

- `claim_level = s2_contract_scaffold`
- `s2_complete = false`
- `s2_status = no_current_madaros_typed_hir_serializer`
- `typed_hir_sha256 = null`
- `typed_hir_status = not_emitted_by_current_madaros`
- `typed_hir_roundtrip_status = not_available`

That null typed-HIR field is intentional. It prevents the gate from pretending
that Madaros already exposes a native typed-HIR serializer.

## Gate cases

The S2 gate runs each case twice with deterministic time and byte-compares every
JSON/TSV sidecar:

| Case | Source |
|---|---|
| `hello` | `examples/hello.sio` |
| `smt_basic` | `tests/stdlib/theorem/test_smt_solver_basic.sio` |
| `selfhost_s2_contract` | `self-hosted/compiler/madaros_v2_s2_receipt.sio` |
| `gpu_ptx_combo` | `tests/madaros/v2_s1/gpu_ptx_combo.sio` |

It requires `bin/madaros check` success and rejects panic/segfault or bulk
diagnostic spew for the known GPU/PTX failure shape (`E175`, `E177`, `E046`).

Latest local proof:

```text
env MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros \
  SOUNIO_STDLIB_PATH=$PWD/stdlib \
  bash scripts/dev/madaros_v2_s2_gate.sh

[madaros-v2-s2] PASS: deterministic S2 contract scaffold; native typed HIR not yet emitted
```

## S3 readiness companion

`scripts/dev/madaros_v2_s3_readiness_gate.sh` now performs the first bounded S3
readiness check: `HlirTypeKind` variants in `self-hosted/hlir/ir.sio` must be
unique and must retain the required epistemic variants.

The duplicate `HlirTypeContest` / `HlirTypeRobust` entries were removed from
`self-hosted/hlir/ir.sio`. The current S3 readiness gate passes:

```text
[madaros-v2-s3] HlirTypeKind variants=42 duplicates=0
[madaros-v2-s3] PASS: HLIR type enum unique; native HLIR roundtrip gate still pending
```

S3 is still not complete. The remaining contractual step is a native HLIR
serializer plus roundtrip/hash gate.
