<!-- docs:meta
topic_id: repo.docs.research.madaros-v2-s1-receipt-implementation-2026-07-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.madaros-v2-s1-receipt-implementation-2026-07-04
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Madaros v2 S1 receipt implementation

Status: local S1/L1 implementation witness, not a claim that Stage1 exposes a
stable canonical AST serializer.

Worktree: `/tmp/sounio-madaros-v2-sota-codex`
Branch: `work/madaros-v2-sota-codex`
Date: 2026-07-04

## What landed

S1 now has an executable receipt surface:

- `bin/madaros s1-receipt <source.sio> [--out-dir OUT]`
- `scripts/dev/madaros_v2_s1_receipt.py`
- `scripts/dev/madaros_v2_s1_gate.sh`
- `self-hosted/compiler/madaros_v2_s1_receipt.sio`
- `tests/madaros/v2_s1/gpu_ptx_combo.sio`

The emitter writes:

- `<case>.s1.receipt.json`
- `<case>.s1.module_edges.tsv`

The receipt records source identity, compiler route, parser SHA, module graph,
module graph hash, source graph hash, diagnostics hash, phase caps, and compiler
check witness.

## Honest AST boundary

The current Madaros Stage1 route does not expose a stable machine-readable AST
serialization. Therefore this implementation deliberately keeps:

```json
"canonical_ast_sha256": null,
"canonical_ast_status": "blocked_until_stable_stage1_ast_serializer",
"ast_surface_kind": "opaque"
```

The stable L1 hash is:

```json
"canonical_source_graph_sha256": "<sha256>",
"canonical_source_graph_status": "stable_l1_source_import_public_symbol_surrogate"
```

This prevents the lane from overclaiming AST canonicalization while still giving
S2/S3 a deterministic source/module receipt to consume.

## Gate

Run:

```bash
bash scripts/dev/madaros_v2_s1_gate.sh
```

The gate emits each receipt twice and byte-compares both the JSON receipt and TSV
edge table. Required cases:

| Case | Source | Purpose |
|---|---|---|
| `hello` | `examples/hello.sio` | canonical tiny source |
| `smt_basic` | `tests/stdlib/theorem/test_smt_solver_basic.sio` | imported stdlib theorem module |
| `selfhost_s1_contract` | `self-hosted/compiler/madaros_v2_s1_receipt.sio` | self-hosted S1 contract module |
| `gpu_ptx_combo` | `tests/madaros/v2_s1/gpu_ptx_combo.sio` | `gpu::kernel_ir` + `gpu::lower_to_ptx` + `gpu::ptx` import blind spot |

Latest local result:

```text
[madaros-v2-s1] PASS: 4 receipts deterministic + contract module checks
```

## Wave B subagents

| Agent | Role | Model | Effort | Mode | Contribution |
|---|---|---|---|---|---|
| Rawls | parser/module surface scout | `gpt-5.4-mini` | medium | read-only | recommended future native home in `module_frontend.sio` plus digest reuse from `module_loader.sio` |
| Pauli | S1 gate scout | `gpt-5.4-mini` | medium | read-only | confirmed smallest gate shape: hello + imported SMT + GPU/PTX combo |
| Peirce | S1 completion auditor | `gpt-5.4` | high | read-only | required deterministic emitted receipt, self-hosted case, and honest AST boundary |

All subagents were read-only. Codex integrated the patch locally.

## Native follow-up

This is a working L1 S1 receipt lane. The native S1b follow-up is to move the
receipt implementation closer to the compiler frontend:

- primary home: `self-hosted/compiler/module_frontend.sio`;
- digest helpers: `self-hosted/compiler/module_loader.sio`;
- semantic graph input: `self-hosted/resolve/modules.sio`;
- thin CLI dispatch: `self-hosted/compiler/main.sio`.

That follow-up should not change this receipt schema silently. It should first
emit byte-identical JSON/TSV for the four S1 cases above, then replace the
Python implementation only after parity is proven.
