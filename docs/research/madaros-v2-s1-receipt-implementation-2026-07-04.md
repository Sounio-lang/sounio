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

Status: S1 AST receipt witness implemented and locally gated against a freshly
built Stage1 artifact. The source-built Madaros SMT regression noted below is
now closed; the same artifact passes the full Madaros gate including imported
SMT 6/6.

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
- `<case>.s1.ast.json`
- `<case>.s1.module_edges.tsv`

The receipt records source identity, compiler route, parser SHA, module graph,
module graph hash, source graph hash, compiler-native AST hash, diagnostics
hash, phase caps, and compiler check witness.

## Compiler-native AST boundary

The Stage1 route now exposes a small deterministic top-level AST JSON surface
through `--emit-ast`. The receipt schema was bumped from `0.1` to `0.2`; the
S1 completion witness is no longer the L1 source graph surrogate.

```json
"schema_version": "madaros.v2.s1.receipt/0.2",
"canonical_ast_sha256": "<sha256>",
"canonical_ast_status": "stable_stage1_ast_serializer",
"ast_surface_kind": "compiler_native_top_level_ast_json",
"ast_serializer_version": "madaros.stage1.ast/0.1"
```

The compiler-native AST sidecar currently serializes the Stage1 parser boundary:
source path, top-level item count, and each top-level item's index, kind, name,
and visibility. It is intentionally small; it is not a pretty-printed source
round trip and it does not yet serialize every expression/body node.

The L1 source graph remains as a secondary witness:

```json
"canonical_source_graph_sha256": "<sha256>",
"canonical_source_graph_status": "stable_l1_source_import_public_symbol_surrogate"
```

## Gate

Run:

```bash
bash scripts/dev/madaros_v2_s1_gate.sh
```

The gate emits each receipt twice and byte-compares the JSON receipt, AST JSON
sidecar, and TSV edge table. Required cases:

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

Latest literal S1 proof (fresh local artifact):

```text
make build-madaros
env MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros \
  SOUNIO_STDLIB_PATH=$PWD/stdlib \
  bash scripts/dev/madaros_v2_s1_gate.sh
```

Observed `canonical_ast_sha256` values:

| Case | AST sha256 |
|---|---|
| `hello` | `47fffa1238f8ecad174fdbf7611c743b940681407fce34b1c55a07cff99dc0fd` |
| `smt_basic` | `6c5307e78e6e18d8cf66e7477e93ce3a9b1b94f381ab28d4a048f705aedd41ac` |
| `selfhost_s1_contract` | `70953c9b8064d6d0bd9c2a97d4b88dda8bf968a1004b4eea67bab052d63727e9` |
| `gpu_ptx_combo` | `53890b94b835617119374534695c1f63f398c1425bfe0112aaa8cd9d182c71f2` |

Raw-ELF determinism was also checked with two direct
`artifacts/self-hosted/madaros --emit-ast` runs per case; the AST sidecars were
byte-identical and matched the hashes above. Wrapper parity was checked for
`hello` with `MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros`.

## Operational blocker (closed)

The freshly built `artifacts/self-hosted/madaros` still does **not** earn the
normal full-gate receipt:

```text
env MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros \
  SOUNIO_STDLIB_PATH=$PWD/stdlib \
  bash scripts/ci/madaros_full_gate.sh

[madaros-full] imported-SMT failure: test_smt_adaptive_epistemic.sio rc=1
[madaros-full] FAIL: imported-SMT solver gate failed
```

This blocker is closed as of 2026-07-04. The source-built
`artifacts/self-hosted/madaros` now passes `scripts/ci/madaros_full_gate.sh`,
including imported-SMT solver gate 6/6, and writes a local gate receipt. S1
also remains green on the same artifact.

## Wave B subagents

| Agent | Role | Model | Effort | Mode | Contribution |
|---|---|---|---|---|---|
| Rawls | parser/module surface scout | `gpt-5.4-mini` | medium | read-only | recommended future native home in `module_frontend.sio` plus digest reuse from `module_loader.sio` |
| Pauli | S1 gate scout | `gpt-5.4-mini` | medium | read-only | confirmed smallest gate shape: hello + imported SMT + GPU/PTX combo |
| Peirce | S1 completion auditor | `gpt-5.4` | high | read-only | required deterministic emitted receipt, self-hosted case, and honest AST boundary |

All subagents were read-only. Codex integrated the patch locally.

## Native follow-up

This is a working S1 AST receipt lane with a deliberately small top-level AST
serializer in `self-hosted/compiler/main.sio`. A future S1b can move more of the
receipt implementation closer to the compiler frontend:

- primary home: `self-hosted/compiler/module_frontend.sio`;
- digest helpers: `self-hosted/compiler/module_loader.sio`;
- semantic graph input: `self-hosted/resolve/modules.sio`;
- thin CLI dispatch: `self-hosted/compiler/main.sio`.

That follow-up should not change this receipt schema silently. It should first
emit byte-identical JSON/AST/TSV for the four S1 cases above, then replace the
Python receipt implementation only after parity is proven.
