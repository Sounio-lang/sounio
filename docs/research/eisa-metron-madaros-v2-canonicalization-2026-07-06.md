<!-- docs:meta
topic_id: repo.docs.research.eisa-metron-madaros-v2-canonicalization-2026-07-06
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.eisa-metron-madaros-v2-canonicalization-2026-07-06
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# EISA/METRON -> Madaros v2 Canonicalization Plan

Status: internal execution plan for canonizing the existing EISA/METRON stack
into the Madaros v2 architecture lane. This is not a public novelty claim and
does not by itself claim that Madaros default-lane can compile EISA today.

Date: 2026-07-06
Work branch: `work/eisa-metron-canon-codex`
Base: `origin/canon/madaros-v2-sota` at `cc33b84f305d6785b85cc3c220094684391359d0`

## 1. Executive Decision

Madaros v2 should treat EISA/METRON as a first-class architectural axis, not as
a later optional backend. The current v2 canon branch already has staged
receipt work for S1-S5, S4 e-graph/E-KAN receipt boundaries, and numeric/ABI
receipts, but it has no EISA/METRON tree in the branch. The EISA/METRON stack
exists on `origin/gpu/epistemic-tensor-core-next` and should be canonized into
Madaros v2 through an explicit integration lane.

Target architecture:

```text
Sounio source
  -> Madaros v2 AST/HIR/SemanticIR/NumericIR
  -> MetronIR
  -> EISA .eisax
  -> Metron VM / x86 bridge / future PTX bridge
  -> execution receipts + translation-validation receipts
```

The first implementation tranche is not a compiler patch. It is a
canonicalization package that fixes source-of-truth refs, gates, blockers, and
merge order so later imports are not evidence theatre.

## 2. Source-of-Truth Refs

Use refs, not dirty worktrees, as import sources.

| Surface | Ref / branch | Current role | Evidence |
|---|---|---|---|
| Madaros v2 canon | `origin/canon/madaros-v2-sota` | Base for this lane; has S1-S5 receipt work but no EISA tree | `git ls-tree origin/canon/madaros-v2-sota docs/research/madaros-v2-*` |
| EISA/METRON | `origin/gpu/epistemic-tensor-core-next` at `059062c2c` | Canonical EISA source after reconciliation/push | `docs/handoff/eisa_w4_v2_bridge_continuation.md` says W4/W5 are complete and on origin |
| Default-lane blocker | `origin/fix/madaros-imported-depclosure-eisa` at `8c208d26f` | Diagnosis of why Madaros default-lane does not yet compile EISA | `docs/handoff/continuity/wp-b1-witness/README.md` |

Do not import from `/workspace/sounio-eisa` by copying the dirty working tree.
That worktree still has modified docs/audit and local artifacts; its branch is
useful for inspection, but the import source is the pushed ref.

Cross-branch/worktree audit on 2026-07-06 found additional non-canonical EISA
refs:

- `eisa/bridge-highreg` contains an older EISA tree ending at `5cf18e8af`
  (`feat(eisa): bridge v1 full high-register support`). Diffing it against
  `origin/gpu/epistemic-tensor-core-next` shows it lacks later W4/W5 material
  and would delete the pushed v2 bridge/receipt docs if used as source.
- `fix/madaros-strlib-assignment-block-parser` and
  `fix/parser-assign-rhs-block-hang-v2` also contain older EISA snapshots mixed
  with parser/compiler repair work. They are not import sources.
- `/workspace/sounio-eisa` is checked out at the same commit as
  `origin/gpu/epistemic-tensor-core-next`, but it has dirty local docs/artifacts.
- `docs/eisa-suite-verification` and
  `fix/madaros-imported-depclosure-eisa` worktrees contain useful read-only
  evidence, but their local worktrees are dirty/untracked and must not be copied
  wholesale.

The audit also found `origin/work/madaros-s-next-codex` with post-canon
Madaros v2 S-next/f128 receipt work. This is relevant future input, but this
canonicalization lane still bases on `origin/canon/madaros-v2-sota` until the
S-next work is explicitly promoted to canon.

## 3. Existing EISA/METRON Reality

`origin/gpu/epistemic-tensor-core-next` contains a self-contained EISA/METRON
stack:

- `.eisax` container and hash/validation: `stdlib/eisa/format.sio`.
- `.eisa` assembler: `stdlib/eisa/asm.sio`.
- Metron VM primary executor: `stdlib/eisa/evm.sio`.
- Metron surface compiler: `stdlib/eisa/backend.sio`.
- Reference semantics: `stdlib/eisa/core.sio` for dd64 v0/v1 and
  `stdlib/eisa/core_v2.sio` for qd128 v2.
- x86-64 AOT conformance bridge: `stdlib/eisa/bridge_x86.sio`.
- VM/bridge tools: `tools/eisa/eisa_evm_run.sio` and
  `tools/eisa/eisa_bridge_emit.sio`.
- Bridge conformance gate: `scripts/ci/eisa_bridge_conformance_gate.sh`.
- EISA suites under `tests/stdlib/eisa/`.

The as-built chronicle states that every register carries `val`, `err`, and
`u`, and that receipts v1/v2/v3 cite the container `prog_hash`. W4 handoff
states that v2 qd128 AOT bridge receipts are byte-identical with the Metron VM
over 33 lanes, including `v2-rump-qd`, `v1-rump-dd`, `v2-mem-poison`, tamper,
and anti-vacuity lanes.

Important honesty note: the W4 handoff header still contains an older line
saying the commits are local/not pushed. Later in the same file, and the current
Git ref state, supersede that: the branch was reconciled and pushed at
`059062c2c`; the file says W4/W5 are complete and on origin.

Stale or bounded claims to harmonize before any public statement:

- `docs/research/eisa-v1-asbuilt-2026-07-06.md` still contains older W4/W5
  wording that does not by itself prove the final pushed state.
- `docs/research/eisa-v2-positioning-2026-07-05.md` still presents itself as a
  W5 draft/pre-review artifact, even though the later handoff says W5 review was
  adopted.
- Do not invent a W0 milestone. The stable foundation labels are E0-E5, V1, and
  W1-W5.
- The current Metron surface is intentionally small: `Str` is capped at 256
  bytes, and general conditions, calls, memory indexing, and driver integration
  remain later work.
- Subnormal-injective receipts and per-site policy gates are not yet part of
  the canonized integration plan.

## 4. Existing Madaros v2 Reality

`origin/canon/madaros-v2-sota` has Madaros v2 receipt infrastructure but no
EISA/METRON tree. A current exact search in this worktree finds no EISA/METRON
matches under `docs/research`, `scripts/dev`, `scripts/ci`, `self-hosted`,
`stdlib`, or `tests`.

There is a separate operational compiler line in the shared checkout
`/workspace/sounio` where the currently visible gates are named
`native_v2_*`, not `madaros_v2_*`. That checkout is dirty and belongs to the
current operational branch, so it is not the import base for this
canonicalization lane. Treat it as current compiler reality to respect, not as
proof that the Madaros-v2 canon branch lacks receipts. This lane deliberately
uses `origin/canon/madaros-v2-sota` as its base and records the divergence as a
state-sync fact.

Implemented or scaffolded Madaros v2 surfaces include:

- S1 source/AST/module receipts:
  `scripts/dev/madaros_v2_s1_receipt.py`,
  `scripts/dev/madaros_v2_s1_gate.sh`, and
  `self-hosted/compiler/madaros_v2_s1_receipt.sio`.
- S2 contract scaffold receipts:
  `scripts/dev/madaros_v2_s2_receipt.py`,
  `scripts/dev/madaros_v2_s2_gate.sh`, and
  `self-hosted/compiler/madaros_v2_s2_receipt.sio`.
- S3 HLIR receipts and readiness gates.
- S4 e-graph/E-KAN receipt boundary with accepted/rejected/blocked rewrite
  receipts and S5 preflight.
- S5 scalar/MIR/ABI/f128/wide-int receipt families.

Those receipts are valuable, but they remain compiler-stage receipts. EISA adds
execution receipts from a semantic machine. The integration target is therefore
not "replace S1-S5 receipts"; it is to connect them to a Metron executable
target whose receipts are source-observable.

## 5. Blocker Record

```text
Blocker-ID: BLK-20260706-EISA-MADAROS-DEFAULTLANE-CHECKER
Status: classified
Severity: B1
Class: compiler-semantics
Owner: unassigned
Lane: EISA/METRON default-lane integration
Worktree: n/a for this plan; diagnosis source is origin/fix/madaros-imported-depclosure-eisa
Branch: origin/fix/madaros-imported-depclosure-eisa
Files-Owned: none in this tranche
Files-Read-Only: self-hosted/check/**, stdlib/str/lib.sio, stdlib/eisa/**, tests/stdlib/eisa/**
Do-Not-Touch: current shared checkout /workspace/sounio, /workspace/sounio-eisa dirty docs
Repro: documented in docs/handoff/continuity/wp-b1-witness/README.md
Observed: EISA lean_single witnesses pass, but Madaros default-lane blocks before codegen
Expected: test_eisa_isa and test_eisa_evm type-check and run on Madaros default-lane
Acceptance-Gate: SOUNIO_MADAROS_BIN=<fresh> ./bin/souc run tests/stdlib/eisa/test_eisa_isa.sio && SOUNIO_MADAROS_BIN=<fresh> ./bin/souc run tests/stdlib/eisa/test_eisa_evm.sio
Evidence-Level: E2
Evidence: origin/fix/madaros-imported-depclosure-eisa:docs/handoff/continuity/wp-b1-witness/README.md
Fallback-Path: lean_single remains the EISA validation lane until this closes
Legacy-Kept: yes
LLM-Offload: required before any checker semantic commit that changes numeric or GUM claims; not required for this read-only classification
Next-Action: create a separate checker lane for int-width widening and merged transitive symbol/field resolution
```

The falsified premise is important: this is not a `ud2`/SIGILL dependency
closure problem. The documented failure is pre-codegen in
`check_modules_verdict_boot4`: `str::lib` i32/i64 mixing plus unresolved
transitive symbols/structs for EISA.

Read-only blocker audit sharpened the failure surface:

- `self-hosted/compiler/module_frontend.sio:4621-4639` imports and merges before
  lower/native execution, so this is a checker failure before codegen.
- `self-hosted/check/mod.sio:544-565` intentionally uses a shared merged
  checker table; do not "fix" this by weakening the merge.
- `self-hosted/check/check.sio:6765-6774` has a function-signature fallback, but
  that is not a full transitive namespace resolver for fields and non-function
  symbols.
- `stdlib/str/lib.sio:23-26` defines `Str.len` as `i32`; exact integer matching
  in `self-hosted/check/compat.sio:59-88` and `825-933` makes loop arithmetic
  explode as `i32` versus `i64`.
- `tests/stdlib/eisa/test_eisa_isa.sio` and `tests/stdlib/eisa/test_eisa_evm.sio`
  import through `math::dd64` and `eisa::core`, exposing unresolved transitive
  symbols as `E137`/`E015`.

## 6. Canonicalization Roadmap

### M0 - Canonicalization preflight

Purpose: prove the refs and files needed for the integration exist, without
running compiler stress.

Gate:

```bash
bash scripts/dev/eisa_metron_canon_preflight.sh
```

Expected result:

- `origin/gpu/epistemic-tensor-core-next` has EISA docs, stdlib modules, tests,
  tools, and conformance gate.
- `origin/canon/madaros-v2-sota` has Madaros v2 receipt docs/scripts.
- `origin/fix/madaros-imported-depclosure-eisa` has blocker evidence.
- Current branch reports whether the EISA source tree is absent or already
  imported.

### M1 - Documentation-only canon PR

Purpose: land this plan and the preflight gate in the Madaros v2 lane.

Write set:

- `docs/research/eisa-metron-madaros-v2-canonicalization-2026-07-06.md`
- `scripts/dev/eisa_metron_canon_preflight.sh`

Required checks:

- `bash scripts/dev/eisa_metron_canon_preflight.sh`
- `bin/llm-offload -t math-review -p xai -i docs/research/eisa-metron-madaros-v2-canonicalization-2026-07-06.md`
  before commit or PR, because this artifact names qd128, GUM lanes, numeric
  receipts, and a compiler novelty claim.
- append the offload outcome to `.claude/llm_offload_log.md`.

### M2 - Import EISA/METRON tree

Purpose: import the EISA source tree from the pushed EISA ref without changing
Madaros checker semantics.

Import source: `origin/gpu/epistemic-tensor-core-next`.

Initial write set:

- `docs/research/eisa-*`
- `docs/handoff/eisa_w4_v2_bridge_continuation.md`
- `stdlib/eisa/**`
- `tests/stdlib/eisa/**`
- `tools/eisa/**`
- `scripts/ci/eisa_bridge_conformance_gate.sh`
- `slurm-jobs/eisa/**` if needed for heavy validation handoff

Also harmonize stale prose inside the imported EISA docs so W4/W5 status,
draft/pre-review labels, and pushed-ref claims agree with `059062c2c`. This is
documentation cleanup only; it must not be used to claim new behavior.

Required gates:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/eisa/test_eisa_core.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/eisa/test_eisa_evm_v2.sio
bash scripts/ci/eisa_bridge_conformance_gate.sh
```

Heavy/full EISA battery stays on Slurm/Foundry. Do not run heavy stress in
`/workspace/sounio`.

### M3 - Default-lane checker unblock

Purpose: make Madaros default-lane able to type-check and run the EISA seed
suites.

Separate branch required. Do not combine with M2 import unless explicitly
opened as a serialized integration window.

Fix surface from the blocker diagnosis:

- `self-hosted/check/compat.sio`: controlled integer-width compatibility for the
  specific arithmetic/comparison family now blocking `Str`, with a negative test
  proving non-integer mismatches still fail.
- `self-hosted/check/check.sio`: transitive resolution for imported non-function
  symbols, fields, and types, while preserving the function fallback and private
  visibility boundaries.
- `self-hosted/check/mod.sio`: keep the shared merged checker wiring unless a
  separate proof shows it is the real bug.

Required gates:

```bash
SOUNIO_MADAROS_BIN=<fresh> ./bin/souc run tests/stdlib/eisa/test_eisa_isa.sio
SOUNIO_MADAROS_BIN=<fresh> ./bin/souc run tests/stdlib/eisa/test_eisa_evm.sio
```

Add negative tests for the old false premise and for the true type-checker
family:

- a 3-module transitive `str::lib` importer that must not emit `ud2`;
- a merged-checker fixture that proves i32/i64 widening is intentional and
  does not swallow non-integer type errors;
- a fixture that proves transitive EISA symbols resolve without disabling
  visibility globally.

### M4 - MetronIR boundary

Purpose: stop treating `stdlib/eisa/backend.sio` as the future compiler
interface. It is the seed implementation, not the final boundary.

Define a compiler-native MetronIR receipt:

```text
madaros.v2.metron_ir/0.1 {
  source_stage_receipt: sha256
  numeric_ir_receipt: sha256
  value_lane_semantics: f64-ieee-strict
  err_lane_semantics: dd64 | qd128 | future
  u_lane_semantics: gum-first-order
  gates: [...]
  fuel_policy: ...
  lowering_obligations: [...]
}
```

M4 is complete only when one source-observable Sounio program lowers through
Madaros v2 into `.eisax`, runs in the Metron VM, and emits a receipt tied back
to the S1/S3/S4/S5 receipt chain.

### M5 - Bridge matrix

Purpose: treat x86 and future PTX as conforming executors of `.eisax`, not as
the semantic definition.

Executors:

- reference chains;
- Metron VM;
- x86-64 bridge;
- future PTX bridge.

Required evidence:

- MVM vs x86 byte-identical receipts on the corpus;
- MVM vs PTX either byte-identical where possible or formally classified
  equivalent where GPU behavior prevents byte identity;
- anti-vacuity gate that proves receipts are not baked into bridge outputs.

## 7. E-KAN Integration Rule

E-KAN belongs in Madaros v2 as an advisor, not as semantic authority.

Allowed:

- propose rewrites;
- rank lowering choices;
- predict frail gates;
- suggest precision depth (`dd64`, `qd128`, future);
- guide PTX/autotuning candidates.

Required:

- every accepted E-KAN proposal must have a receipt with `validator =
  translation-validation`, `error_bound`, fallback hash, accepted/rejected/blocked
  classification, and extraction status.

Rule:

```text
E-KAN may propose. Receipts and translation validation dispose.
```

## 8. Offload and Governance

Before commit or PR:

- This file requires `bin/llm-offload -t math-review -p xai -i
  docs/research/eisa-metron-madaros-v2-canonicalization-2026-07-06.md` before
  commit or PR, even though it is internal, because it discusses qd128, GUM,
  numerical receipts, and novelty positioning.
- If later changes touch GUM, qd128, f128, p-box, interval semantics, or
  mathematical claims, run `bin/llm-offload -t math-review -p xai -i <file_or_diff>`.
- If publishing/paper/dissertation language is produced from this document,
  fan out external-facing review before submission.

No push directly to `canon/*`. This branch should become a PR into the chosen
canon branch after gates and offload are logged.

## 9. Subagent Evidence Used

This plan incorporated read-only subagent work as follows:

- EISA/METRON cartography: accepted that `origin/gpu/epistemic-tensor-core-next`
  at `059062c2c` is the import source, that the branch contains the EISA stdlib,
  tools, tests, bridge gate, and W4/W5 docs, and that `/workspace/sounio-eisa`
  is only an inspection worktree because it is dirty. Rejected the stale header
  wording that still says W4 commits are local/not pushed.
- Madaros v2 cartography: accepted that `origin/canon/madaros-v2-sota` has
  Madaros v2 receipt infrastructure but no EISA/METRON tree. Also accepted that
  the shared checkout currently exposes a `native_v2_*` compiler lane; this is
  recorded as operational divergence, not used as the import base.
- Default-lane blocker audit: accepted that the EISA default-lane failure is
  pre-codegen in the merged checker, involving integer-width compatibility and
  transitive symbol/field resolution. Rejected the older `ud2`/SIGILL closure
  hypothesis.
- Governance audit: accepted the branch/PR discipline, no direct `canon/*`
  push, no heavy validation in `/workspace/sounio`, and mandatory offload for
  this document before commit/PR.
- Preflight review: accepted that exact prose-grep checks were too brittle and
  replaced them with structural ref/path checks plus explicit suite paths.
  Accepted that offload wording was too permissive and made it mandatory.

## 10. Immediate Next Command

For this M1 package, run the preflight:

```bash
bash scripts/dev/eisa_metron_canon_preflight.sh
```

After M1 lands, the next lane is M2: import the EISA/METRON tree from
`origin/gpu/epistemic-tensor-core-next` and harmonize stale W4/W5 prose without
changing Madaros checker semantics.
