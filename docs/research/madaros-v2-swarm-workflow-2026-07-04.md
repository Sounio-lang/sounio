<!-- docs:meta
topic_id: repo.docs.research.madaros-v2-swarm-workflow-2026-07-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.madaros-v2-swarm-workflow-2026-07-04
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Madaros v2 swarm workflow

Status: activation plan for the locked Madaros v2 SOTA+++ lane.

This document turns the architecture plan into parallel work. It exists because
the project needs a coordinated swarm, not one agent wandering between compiler
bugs, E-KAN theory, GPU/PTX failures, and paper novelty claims.

## Coordinator

Coordinator: Codex in `/tmp/sounio-madaros-v2-sota-codex`

The coordinator owns:

- the lock file;
- architecture synthesis;
- final plan integration;
- cross-agent conflict checks;
- commit boundaries.

The coordinator does not delegate the immediate critical path if the next local
action is blocked on that result.

## Model and Effort Routing

| Role | Agent type | Model | Effort | Mode | Output |
|---|---|---|---|---|---|
| Compiler architecture scout | explorer | `gpt-5.4-mini` | medium | read-only | IR/stage map and hazards |
| Literature/SOTA scout | explorer | `gpt-5.4-mini` | medium | read-only | current anchors and claim limits |
| E-KAN semantics scout | explorer | `gpt-5.4` | high | read-only | S2/S4 E-KAN compiler mapping |
| Validation scout | explorer | `gpt-5.4` | high | read-only | receipt/translation-validation gates |
| S1 receipt worker | worker | `gpt-5.4` | high | write | S1 receipt skeleton and tiny gate |
| HLIR contract worker | worker | `gpt-5.4` | high | write | HLIR canonicalization audit/tests |
| GPU/PTX gate worker | worker | `gpt-5.3-codex-spark` | high | write | module-combination gate |
| Governance reviewer | explorer | `gpt-5.4-mini` | low | read-only | overclaim/offload/registry audit |

Use stronger models only where semantic correctness or architecture synthesis is
the bottleneck. Use mini/spark for bounded scans and gate plumbing.

## Wave A - Read-Only Scouts

Run these in parallel when subagent capacity exists:

1. Compiler architecture scout:
   - read `self-hosted/compiler/**`, `self-hosted/ir/**`, `self-hosted/hlir/**`,
     `scripts/ci/**`, `scripts/dev/**`, `docs/MADAROS_STATUS.md`;
   - return current stage surfaces, caps, gates, receipts, and hazards.
2. Literature/SOTA scout:
   - refresh MLIR, TensorIR, equality saturation/e-graphs, persistent e-graphs,
     KAN/FastKAN/KAN 2.0, translation validation;
   - return safe novelty wording and missing benchmarks.
3. E-KAN semantics scout:
   - read current E-KAN examples, tests, and status docs;
   - map runtime witnesses into S2 declarations and S4 optimizer receipts.
4. Validation scout:
   - design receipt schema, deterministic hashing, translation validation,
     differential checks, counterexample fuzzing, and offload triggers.

No Wave A scout edits files.

## Wave B - Disjoint Implementation Workers

Only start Wave B after Wave A outputs are integrated.

| Worker | Write-Set | First deliverable | Gate |
|---|---|---|---|
| S1 receipt worker | `self-hosted/compiler/*s1*`, `scripts/dev/madaros_v2_s1_*`, S1 tests/docs | canonical source + module graph receipt skeleton | tiny S1 receipt gate |
| HLIR contract worker | `self-hosted/hlir/**`, HLIR tests/docs | duplicate enum/canonical hash audit | HLIR roundtrip/hash gate |
| GPU/PTX gate worker | GPU/PTX gate scripts and fixtures only | module-combination witness for `lower_to_ptx.sio` + `ptx.sio` | CI gate reproduces/fixes combo |
| E-KAN declaration worker | E-KAN declaration/receipt files and tests only | E-KAN declaration schema, no optimizer rewrite | declaration-only gate |

Workers must say which files they changed. They must not revert edits from other
agents and must adapt to existing dirty state.

## Stop Conditions

Stop and record a blocker if:

- two workers need the same file;
- a worker needs a high-risk shared file from `AGENTS.md`;
- a math/clinical/public claim is about to be committed without required
  LLM-offload review;
- a remote protection operation fails due permissions;
- a gate produces evidence that contradicts the architecture claim.

## Activation Prompts

Compiler scout prompt:

```text
Read-only in /tmp/sounio-madaros-v2-sota-codex. Map current Madaros compiler
architecture surfaces for the v2 plan. Do not edit files. Return concise bullets
for parser/check/lower/IR/HLIR/GPU KernelIR, gates/receipts, and hazards.
```

E-KAN scout prompt:

```text
Read-only in /tmp/sounio-madaros-v2-sota-codex. Map current E-KAN examples,
tests, and docs into compiler-stage concepts. Separate runtime witness,
declaration surface, optimizer proposal, and proof/receipt requirements.
```

Validation scout prompt:

```text
Read-only in /tmp/sounio-madaros-v2-sota-codex. Design stage receipts and gates
for S1-S7. Include deterministic hashing, translation validation,
differential CPU/GPU/WASM, counterexample fuzzing, and offload policy triggers.
```

## Wave A Status

Attempted earlier on 2026-07-04:

```text
multi_agent spawn explorer -> agent thread limit reached
```

After retry, Wave A launched and completed:

| Scout | Role | Result |
|---|---|---|
| Copernicus | Compiler architecture scout | complete |
| Hume | Literature/SOTA scout | complete |
| Ampere | E-KAN semantics scout | complete |
| Ptolemy | Validation and receipts scout | complete |

Integrated synthesis:

- `docs/research/madaros-v2-wave-a-scout-synthesis-2026-07-04.md`

The lane is no longer merely swarm-ready; Wave A has run. Wave B should start
with one S1 receipt worker and keep all other implementation lanes read-only
until the S1 receipt gate exists.

## Wave B S1 Status

S1/L1 receipt implementation landed on 2026-07-04:

- `bin/madaros s1-receipt <source.sio> [--out-dir OUT]`
- `scripts/dev/madaros_v2_s1_receipt.py`
- `scripts/dev/madaros_v2_s1_gate.sh`
- `self-hosted/compiler/madaros_v2_s1_receipt.sio`
- `tests/madaros/v2_s1/gpu_ptx_combo.sio`
- `docs/research/madaros-v2-s1-receipt-implementation-2026-07-04.md`

Wave B S1 subagents:

| Agent | Role | Model | Effort | Mode | Result |
|---|---|---|---|---|---|
| Rawls | parser/module surface scout | `gpt-5.4-mini` | medium | read-only | complete |
| Pauli | S1 gate scout | `gpt-5.4-mini` | medium | read-only | complete |
| Peirce | S1 completion auditor | `gpt-5.4` | high | read-only | complete |

S1b update: `canonical_ast_sha256` is now a real compiler-native AST sidecar
hash (`madaros.v2.s1.receipt/0.2`, `madaros.stage1.ast/0.1`) emitted through a
small `--emit-ast` Stage1 path. The gate now byte-compares receipt JSON, AST
JSON, and module-edge TSV for the four S1 gate cases. The broader source-built
compiler lane is now green: the freshly rebuilt `artifacts/self-hosted/madaros`
passes `scripts/ci/madaros_full_gate.sh`, including imported-SMT 6/6, and
passes `scripts/dev/madaros_v2_s1_gate.sh`.

## Wave C S2/S3 Status

S2 contract scaffold landed on 2026-07-04:

- `scripts/dev/madaros_v2_s2_receipt.py`
- `scripts/dev/madaros_v2_s2_gate.sh`
- `self-hosted/compiler/madaros_v2_s2_receipt.sio`
- `bin/madaros s2-receipt`

The scaffold is intentionally honest: `s2_complete = false` and
`typed_hir_sha256 = null` until Madaros exposes a native typed-HIR/THIR
serializer. The S2 gate emits deterministic JSON/TSV sidecars for hello,
imported SMT, self-hosted S2 contract, and GPU/PTX import-combination cases.

S3 now has a native HLIR JSON/hash/roundtrip gate:

- `scripts/dev/madaros_v2_s3_readiness_gate.sh`
- `scripts/dev/madaros_v2_s3_gate.sh`
- `scripts/dev/madaros_v2_s3_receipt.py`
- `scripts/dev/madaros_v2_s4_preflight_gate.sh`
- `bin/madaros --emit-hlir`
- `bin/madaros s3-receipt`
- `self-hosted/hlir/ir.sio` duplicate `HlirTypeContest` / `HlirTypeRobust`
  variants removed.

The S3 gate validates byte-identical deterministic re-emission, parseable
`madaros.hlir.module/0.2` JSON, canonical JSON roundtrip hash, structural
count consistency, and representative string/call/control/GPU-PTX-import
witnesses. S4 can now consume HLIR JSON hashes; S4 e-graph/E-KAN optimization
receipts remain future work. The S4-ready boundary is executable through
`scripts/dev/madaros_v2_s4_preflight_gate.sh`, which emits
`madaros.v2.s4.preflight/0.1` with `s4_ready = true` and
`s4_implemented = false`.

## Wave D S4/S5 Status

S4 exact-rewrite receipts landed on 2026-07-05:

- `bin/madaros s4-receipt`
- `scripts/dev/madaros_v2_s4_receipt.py`
- `scripts/dev/madaros_v2_s4_gate.sh`
- `tests/madaros/v2_s4/manifest.tsv`
- `tests/madaros/v2_s4/exact_identity.sio`

The S4 gate consumes S3 HLIR receipts, builds persistent e-graph artifacts, and
emits `madaros.v2.ekan.rewrite/0.1` receipts. This is a completed boundary for
one exact subset, not global S4 completion. The accepted subset is exact and
conservative: `constant_fold_i64`, `basis_family = exact_symbolic`,
`validator = translation-validation`, `error_bound = 0`, exact fallback hash,
and original/rewritten e-node hashes.

S5 now has an executable input-contract preflight:

- `scripts/dev/madaros_v2_s5_preflight_gate.sh`

The preflight consumes current S4 boundary receipts and rejects rewrites that
could change MIR or ABI semantics. It emits `madaros.v2.s5.preflight/0.1` with
`s5_input_contract_ready = true`, `s5_ready = false`, and
`s5_implemented = false`.
