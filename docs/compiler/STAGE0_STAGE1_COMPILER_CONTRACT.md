<!-- docs:meta
topic_id: repo.docs.compiler.stage0-stage1-compiler-contract
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.stage0-stage1-compiler-contract
-->

# Stage0/Stage1 Compiler Contract

Status: mandatory for compiler-lane work.

This contract turns the old monolithic compiler shape into an explicit bootstrap artifact instead of letting it remain the default architecture by accident.

## Decision

Sounio has three compiler stages.

Stage0 is `self-hosted/compiler/lean_single.sio`.

Stage0 is the frozen bootstrap and recovery compiler. Its job is fixed-point proof, disaster recovery, release archaeology, and compatibility comparison. It is allowed to receive narrow bug fixes, but it is not the home for new language semantics.

Stage1 is the modular compiler lane:

- `self-hosted/compiler/lean.sio`
- `self-hosted/compiler/lean_frontend.sio`
- `self-hosted/compiler/module_parse.sio`
- `self-hosted/compiler/module_frontend.sio`
- `self-hosted/compiler/module_loader.sio`
- `self-hosted/compiler/module_native_driver.sio`
- `self-hosted/compiler/module_native_streaming.sio`
- `self-hosted/compiler/souc_v2/`
- `self-hosted/check/`
- `self-hosted/ir/`
- `self-hosted/native/`

Stage1 is the serious compiler. New parser, resolver, typechecker, diagnostic, IR, and backend semantics must land here first, or land with a same-commit parity witness proving that Stage1 is not behind Stage0.

Stage2 is the promoted default compiler after parity gates prove that Stage1 can replace Stage0 as the normal developer path. Stage2 may still use Stage0 as an oracle, but not as the default implementation surface.

## Non-Negotiables

1. No new language semantic rule may live only in `lean_single.sio`.
2. A diagnostic test may not be de-ignored unless the compiler binary used by the harness rejects it with a non-zero exit or a `typecheck: failed` diagnostic.
3. A Stage1 lane may not silently fall back to Stage0 without recording `fallback_path=<reason>` in its gate artifact.
4. A Stage0 fix must include a Stage1 follow-up issue, manifest row, or parity test when the fixed behavior is semantic.
5. `self-hosted/compiler/main.sio` is a compatibility shim target, not the architecture target. Shrinking it is a later phase after modular parity is proven.

## Phase Contracts

Parser/import graph:
Stage1 owns parse/import behavior. `module_parse.sio` and `module_frontend.sio` are the first place to fix source graph behavior. Stage0 can be used as an oracle.

Resolver/checker:
The full checker lives under `self-hosted/check/`. The narrow Stage1 API is `check_program_epistemic_into`; heavyweight state remains behind `check_program_with_artifacts` until parity is proven. Compiler agents must classify failures as checker API drift, diagnostic routing, or harness routing instead of editing all layers at once.

Diagnostics:
The test harness contract is observable, not philosophical: compile-fail tests must exit non-zero or emit `typecheck: failed`. Any compiler path that emits an ELF after a compile-fail fixture is a blocker.

IR/HIR:
IR lowering and summaries must be available through modular entrypoints before promotion. `lean_frontend.sio --ir-summary` and `lean.sio --ir-summary` are the parity probes.

Native backend:
`module_native_driver.sio` is the conservative Stage1 native facade. It may try the streaming/native-v2 path first, but must keep a legacy native fallback until streaming parity is proven.

CLI:
Stage1 must match the user-facing command shape that agents and scripts rely on: `--check`, `--ir-summary`, `--self-test`, compile output path, and target selection.

CI:
CI promotion happens in two steps. First, the contract gate records known blockers and prevents unclassified drift. Second, the same gate is tightened from `known_blocker` to required pass as each lane turns green.

Monolith shrink:
Only after Stage1 parity is green should `lean_single.sio` and `main.sio` be reduced. Shrinkage is a reward for parity, not a replacement for it.

## Current Blocker Classes

B1 diagnostic harness contract:
Compile-fail UI tests must be rejected by the binary used by the harness. The sentinel fixture is `tests/ui/type/assign_to_immut.sio`.

B2 Stage1 typecheck contract:
`lean.sio` and `lean_frontend.sio` must pass `souc check` under the checked compiler. If this regresses, the failure belongs to the Stage1 checker/import API lane, not the Stage0 bootstrap lane.

B3 Stage1 runtime frontend contract:
`lean.sio --self-test`, `lean_frontend.sio --self-test`, and `lean_frontend.sio --check examples/hello.sio` are required Stage1 runtime probes.

B4 native driver parity:
`module_native_driver.sio` must keep the streaming-first, legacy-fallback shape until native-v2 supports the required IR surface.

B5 monolith token pressure:
`self-hosted/compiler/main.sio` remains a compatibility shim. Token-cap or size pressure there is not a reason to add more semantics to Stage0.

## Parallel Agent Ownership

Diagnostic hardening lane:
Owns `self-hosted/compiler/lean_single.sio`, `tests/ui/type/`, and harness expectations. Stops if the fix requires broad Stage1 checker surgery.

Stage1 checker lane:
Owns `self-hosted/check/`, `module_frontend.sio`, and the checker API seam. Stops if native backend behavior changes are required.

Stage1 CLI/runtime lane:
Owns `lean.sio`, `lean_frontend.sio`, and their self-test/summary commands. Stops if checker semantics must change.

Native driver lane:
Owns `module_native_driver.sio`, `module_native_streaming.sio`, and native-v2 fallback classification. Stops if parser/checker behavior changes are required.

CI contract lane:
Owns `scripts/ci/compiler_stage_contract_gate.sh` and `tests/compiler/stage_parity/manifest.tsv`. It may tighten expected statuses only after the owning lane provides a green witness.

No two agents edit the same owned file in the same phase. Agents may inspect or review another lane's files, but write ownership is exclusive.
