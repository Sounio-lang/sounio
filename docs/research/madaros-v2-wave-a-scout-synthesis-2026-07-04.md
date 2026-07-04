<!-- docs:meta
topic_id: repo.docs.research.madaros-v2-wave-a-scout-synthesis-2026-07-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.madaros-v2-wave-a-scout-synthesis-2026-07-04
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Madaros v2 Wave A scout synthesis

Status: read-only swarm synthesis for the locked Madaros v2 SOTA+++ lane.

Worktree: `/tmp/sounio-madaros-v2-sota-codex`
Branch: `work/madaros-v2-sota-codex`
Date: 2026-07-04

## Scouts

| Scout | Role | Model | Effort | Result |
|---|---|---|---|---|
| Copernicus | Compiler architecture scout | `gpt-5.4-mini` | medium | complete |
| Hume | Literature/SOTA scout | `gpt-5.4-mini` | medium | complete |
| Ampere | E-KAN semantics scout | `gpt-5.4` | high | complete |
| Ptolemy | Validation and receipts scout | `gpt-5.4` | high | complete |

All scouts were read-only. No scout edited files.

## Consolidated Findings

### Architecture

- The current Stage1 compiler surface is still broad/monolithic: it pulls lexer,
  parser, checker, IR, native, wasm, HLIR, and GPU concerns into the active
  compiler route.
- The repo has two important IR surfaces:
  - flat post-checker IR in `self-hosted/ir/`;
  - SSA-ish HLIR in `self-hosted/hlir/`.
- GPU KernelIR remains a target IR and direct PTX lowering surface; it is not
  the mid-level optimizer.
- `souc_v2` exists as a separate lane, but it is not the current Madaros route.
- S1 should therefore create source/AST/module-graph receipts before touching
  type checking, HLIR, GPU, ABI, or E-KAN.

### Gates and Receipts

- S0 is locally healthy in the isolated worktree:
  - `bash scripts/dev/madaros_two_gate.sh bin/madaros-relocgate` reported
    `A=6/6 B=pass`;
  - `bash scripts/ci/madaros_source_to_elf_gate.sh` passed;
  - `bash scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh` passed 4/4.
- `bash scripts/ci/madaros_operational_contract_gate.sh` failed on a
  `docs/MADAROS_STATUS.md` marker mismatch. Treat this as contract drift, not
  compiler semantics.
- The main route remains receipt-gated through `scripts/lib/resolve_madaros.sh`
  and the full proof gate remains `make madaros-full-gate`.

### Literature and Novelty

- Safe novelty wording:
  - Madaros v2 is a receipt-carrying compiler architecture for epistemic
    scientific programs.
  - E-KAN/equality reasoning are first-class compiler objects only after they
    have declarations and receipts.
- Unsafe wording:
  - "Madaros implements E-KAN optimization";
  - "Madaros is MLIR/TensorIR";
  - "E-KAN rewriting is sound";
  - performance or SOTA claims without benchmark tables.
- Before publication, replace URL anchors with a formal bibliography and add
  benchmark/ablation tables for determinism, translation validation,
  GPU/PTX import coverage, and E-KAN declaration-vs-runtime evidence.

### E-KAN

- This isolated worktree does not contain
  `docs/research/ekan_native_bridge_status_2026-07-04.md`; do not cite it as
  committed evidence here.
- Current E-KAN support is runtime/witness-oriented:
  - examples such as `examples/epistemic_kan.sio`,
    `examples/ekan_knowledge.sio`, `examples/ekan_gum_vs_montecarlo.sio`, and
    `examples/ossm_ekan_pipeline.sio`;
  - tiny witness `tests/run-pass/ekan_knowledge_basic.sio`.
- Existing compiler epistemic infrastructure lives in HLIR/effects, but there
  is no compiler-native E-KAN declaration surface yet.
- The repo has generic uncertainty-aware/e-graph-like optimization surfaces,
  but not E-KAN-specific rewrite receipts or validators.

### Validation

- S1 receipt should emit both:
  - one JSON receipt;
  - one TSV module-edge table.
- Add `receipt_sha256`, `module_graph_sha256`, `phase_caps`, and
  `compiler_route_kind` to the proposed S1 receipt.
- S2 needs separate hashes for symbol, effect, refinement, and epistemic
  declaration tables.
- S3 must not become contractual until the duplicate HLIR enum audit is closed.
- S4 needs one receipt per selected rewrite, not merely one receipt per pass.
- S5 should be an ABI witness matrix, not a type-existence checklist.
- S6 parity must be per-target and narrow at first. WASM needs a dedicated gate
  before CPU/GPU/WASM parity can be claimed.
- S7 should be a receipt chain:
  `tree_sha -> S1 -> S2 -> S3 -> S4 -> S5 -> S6 -> binary_sha`.

## Blockers

```text
Blocker-ID: BLK-20260704-madaros-v2-contract-drift
Status: reproduced
Severity: B2
Class: doc-claim
Owner: Codex
Lane: Madaros v2 SOTA
Worktree: /tmp/sounio-madaros-v2-sota-codex
Branch: work/madaros-v2-sota-codex
Files-Owned: docs/MADAROS_STATUS.md or scripts/ci/madaros_operational_contract_gate.sh (future lane only)
Repro: bash scripts/ci/madaros_operational_contract_gate.sh
Observed: status-doc marker mismatch
Expected: operational contract gate passes
Acceptance-Gate: bash scripts/ci/madaros_operational_contract_gate.sh
Evidence-Level: E1
Next-Action: align the status-doc marker or the gate expectation before using status-doc claims as greenline evidence.
```

```text
Blocker-ID: BLK-20260704-madaros-v2-s3-hlir-dup-enums
Status: reproduced
Severity: B1
Class: compiler-semantics
Owner: unassigned
Lane: Madaros v2 S3
Worktree: /tmp/sounio-madaros-v2-sota-codex
Branch: work/madaros-v2-sota-codex
Files-Owned: self-hosted/hlir/ir.sio (future lane only)
Repro: rg -n "HlirTypeContest|HlirTypeRobust" self-hosted/hlir/ir.sio
Observed: duplicate enum entries for Contest/Robust
Expected: no duplicate enum definitions in contractual HLIR
Acceptance-Gate: HLIR duplicate audit closed plus HLIR roundtrip/hash gate passes
Evidence-Level: E1
Next-Action: keep HLIR out of S1 write-set; open separate S3 audit lane later.
```

```text
Blocker-ID: BLK-20260704-madaros-v2-s6-wasm-evidence-gap
Status: proposed
Severity: B3
Class: evidence-gap
Owner: unassigned
Lane: Madaros v2 S6
Worktree: /tmp/sounio-madaros-v2-sota-codex
Branch: work/madaros-v2-sota-codex
Files-Owned: scripts/ci or scripts/dev wasm gate files (future lane only)
Repro: repo scan of scripts/ci and scripts/dev for dedicated wasm gate
Observed: S6 plan names CPU/GPU/WASM parity, but no dedicated wasm gate was found
Expected: wasm-specific gate before CPU/GPU/WASM parity claim
Acceptance-Gate: wasm gate added and included in shared parity manifest
Evidence-Level: E0
Next-Action: treat WASM parity as unproven until a gate exists.
```

## Next Implementation Lane

Start Wave B with exactly one worker:

```text
Role: S1 receipt worker
Worktree: /tmp/sounio-madaros-v2-sota-codex or a child worktree from work/madaros-v2-sota-codex
Write-Set:
  - self-hosted/compiler/*s1*
  - scripts/dev/madaros_v2_s1_*
  - S1 tests/docs
Do-Not-Touch:
  - self-hosted/hlir/**
  - self-hosted/gpu/**
  - ABI/codegen/native paths
  - E-KAN optimizer paths
First gate:
  - tests/run-pass/hello.sio
  - one imported stdlib module
  - gpu::kernel_ir + gpu::lower_to_ptx + gpu::ptx combination
```

Best first command:

```bash
git status --short --branch
rg -n "s1|receipt" self-hosted/compiler scripts/dev scripts/ci
```
