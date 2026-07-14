<!-- docs:meta
topic_id: repo.docs.handoff.madaros-fixed-array-call-boundary-alias-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.madaros-fixed-array-call-boundary-alias-2026-07-14
-->

# Madaros Fixed-Array Call-Boundary Alias

The local identifier-copy vertical does not change call ABI. This blocker keeps
that boundary executable and separate from local known-extent word-array copy.

```text
Blocker-ID: BLK-20260714-madaros-fixed-array-call-boundary-alias
Status: review-ready
Severity: B1
Class: compiler-semantics
Owner: Codex agent /root/fixed_array_call_abi
Lane: Madaros fixed-array by-value call-boundary semantics
Worktree: /tmp/sounio-fixed-array-call-abi-20260714
Branch: codex/fixed-array-call-abi-20260714
Files-Owned: self-hosted/ir/lower.sio, self-hosted/native/codegen_x86_linux.sio, tests/known_failures/madaros_fixed_array_call_boundary_alias_probe.sio, scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh, docs/handoff/madaros_fixed_array_call_boundary_alias_2026-07-14.md, docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json
Files-Read-Only: self-hosted/ir/ir.sio, scripts/ci/build_modular_madaros.sh
Do-Not-Touch: scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh semantics; aggregate and nested-array legacy paths
Repro: SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_BIN=/tmp/sounio-pr915-madaros-current-source-artifact/madaros SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_DIR=/tmp/sounio-pr915-array-call-boundary-v2 SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_KEEP=1 bash scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh
Observed: PR #915 current-source Madaros a1b63e5d651bbe21c6eb0f19b4f24aa967977f4734d687793322b09e6b573095 executes the probe with rc=61 and exact diagnostic BLOCKED fixed_array_call_boundary_alias caller_changed_after_by_value_param_mutation
Expected: rc=0, exact probe stdout PASS fixed_array_call_boundary_value_semantics caller=unchanged, and exact focused-witness stdout covering i64/i8/bool/f64 ownership for 1<=N<=16, mutable-reference visibility, and prefix/array/suffix parameter stability
Acceptance-Gate: bash scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh
Evidence-Level: E4
Evidence: docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json records the original f841 gate and Stage2 acceptance; PR #915 Actions run 29360223188 produced the a1b63e5d source-fresh control artifact; the candidate Stage2 focused witness passes locally
Fallback-Path: none
Legacy-Kept: yes; N>16, aggregate, nested-array, and unknown-extent paths retain the unproven loop/legacy behavior
LLM-Offload: not-required
Next-Action: commit the reviewed candidate, request a source-fresh Madaros CI build from that exact commit, download its compiler artifact, and require the acceptance gate to return rc=0
```

## Semantic Lane Declaration

```text
Semantic-Lane-ID: fixed-array-call-abi-20260714
Owner: Codex agent /root/fixed_array_call_abi
Concept-IDs: none
Intent-Preserved: by-value direct word-scalar fixed-array parameters with 1<=N<=16 own backing storage distinct from the caller
Transformation: bind an eligible callee parameter name to a fresh element-wise copy while keeping the received ABI parameter register unchanged
Types-Changed: none
Effects-Changed: none
IR-Changed: instruction sequence only; no opcode or field meaning changed
Claims-Introduced: the named gates may prove caller isolation and local-copy independence for their witnessed direct word-scalar arrays with 1<=N<=16
Claims-Forbidden: general known-extent semantics; N>16; f128/f256 arithmetic support; structs; nested arrays; unknown extents; arbitrary whitelist-wide runtime parity; reference isolation
Assumptions: direct fixed arrays are represented by aggregate handles and each whitelisted element occupies one native-v2 word slot
Write-Set: self-hosted/ir/lower.sio; focused call-boundary probe, witness, gate, handoff, and receipt
Read-Set: self-hosted/ir/ir.sio; scripts/ci/build_modular_madaros.sh; self-hosted/native/codegen_x86_linux.sio
Positive-Witness: tests/native-v2/fixed_array_call_boundary_value_witness.sio
Negative-Witness: mutable &![i64; 2] parameter mutation remains caller-visible in the same witness
Acceptance-Gate: bash scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh
Integration-Target: stacked on PR #915 head 5efddb9411c2b5982274f3f9e29778673b0f2383
Authoritative-Only-If: both local-copy and call-boundary gates return rc=0 with a Madaros built from the exact post-base-update candidate commit
```

The Stage2 result is an acceptance oracle, not evidence that the candidate
current-source Madaros already satisfies the call-boundary contract. Struct,
nested-array, unknown-extent, and reference exclusions are currently enforced
by the top-level `TypeArray` and direct word-scalar predicates; only mutable
reference behavior has an executable negative witness in this lane.

## Source-Fresh Local-Copy Blocker

The PR #915 CI artifact builds successfully, but the dedicated local-copy gate
fails when that compiler executes the `u64` parameter-source case.

```text
Blocker-ID: BLK-20260714-madaros-local-array-copy-bootstrap-replay
Status: classified
Severity: B2
Class: bootstrap-runtime
Owner: Codex agent /root/fixed_array_call_abi
Lane: PR #915 local known-extent word-array copy promotion
Worktree: /tmp/sounio-madaros-fixed-array-copy-main-20260714
Branch: codex/madaros-fixed-array-copy-main-20260714
Files-Owned: self-hosted/ir/lower.sio, scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh, tests/native-v2/local_known_extent_word_array_copy_witness.sio, docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json, docs/handoff/madaros_fixed_array_call_boundary_alias_2026-07-14.md
Files-Read-Only: scripts/ci/build_modular_madaros.sh, .github/workflows/ci.yml
Do-Not-Touch: canonical compiler wrappers and CI workflow; aggregate, nested-array, and unknown-extent legacy paths
Repro: SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_BIN=/tmp/sounio-pr915-madaros-artifact-8321978196/madaros SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_DIR=/tmp/sounio-pr915-word-array-source-fresh SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_KEEP=1 bash scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh
Observed: PR #915 merge-candidate artifact a1b63e5d651bbe21c6eb0f19b4f24aa967977f4734d687793322b09e6b573095 exits the witness at rc=51 with FAIL local_known_extent_word_array_copy u64 parameter source copy
Expected: gate rc=0 and exact PASS receipt for every narrow local-copy case
Acceptance-Gate: bash scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh against a compiler rebuilt from the candidate head
Evidence-Level: E4
Evidence: docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json records CI run 29360223188, artifact 8321978196, merge-candidate and head SHAs, compiler SHA, command, exit codes, stdout, and gate-log SHA
Fallback-Path: none
Legacy-Kept: yes; N>16, aggregate, nested-array, and unknown-extent paths remain unchanged and unproven
LLM-Offload: not-required
Next-Action: determine whether a self-host replay internalizes the lowering change or whether the compiler requires an additional bootstrap-safe repair; then rerun both local-copy and call-boundary gates
```
