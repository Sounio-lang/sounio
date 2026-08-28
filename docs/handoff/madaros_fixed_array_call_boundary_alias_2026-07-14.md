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
Status: source-fresh-semantic-pass
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
Expected: rc=0, exact probe stdout PASS fixed_array_call_boundary_value_semantics caller=unchanged, and exact focused-witness stdout covering witnessed N=2 i64/i8/bool/f64 ownership, mutable-reference visibility, and prefix/array/suffix parameter stability
Acceptance-Gate: bash scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh
Evidence-Level: E4
Evidence: PR #916 run 29365161389 artifact 8323918253 compiler 842fad7d passes the exact call-boundary gate and the repaired local-copy gate; docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json records hashes and outcomes
Fallback-Path: none
Legacy-Kept: yes; the existing copy loop and aggregate, nested-array, and unknown-extent paths remain unchanged
LLM-Offload: not-required
Next-Action: commit the harness-only two-line PASS receipt, then require final-head CI and both gates on its exact source-fresh compiler artifact before marking the stacked PR merge-ready
```

## Semantic Lane Declaration

```text
Semantic-Lane-ID: fixed-array-call-abi-20260714
Owner: Codex agent /root/fixed_array_call_abi
Concept-IDs: none
Intent-Preserved: witnessed by-value direct word-scalar fixed-array parameters own backing storage distinct from the caller, and inferred local copies retain their fixed-array shape metadata
Transformation: bind an eligible callee parameter name to a fresh element-wise copy while keeping the received ABI parameter register unchanged; record fixed-array length and word-scalar metadata through the summary-owned mutable LocalStack path
Types-Changed: none
Effects-Changed: none
IR-Changed: instruction sequence only; no opcode or field meaning changed
Claims-Introduced: the named gates may prove caller isolation and local-copy independence for their witnessed direct word-scalar arrays at N=2 and N=4
Claims-Forbidden: general known-extent semantics; f128/f256 arithmetic support; structs; nested arrays; unknown extents; arbitrary whitelist-wide runtime parity; reference isolation
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

The PR #915 artifact first exposed the `u64` parameter-source failure. PR #916
head `d9234e228` proved call-boundary isolation, but the same local-copy case
still failed because the active summary-owned parameter path discarded the
fixed-array length and word-scalar LocalStack metadata after creating the
callee-owned parameter copy.

```text
Blocker-ID: BLK-20260714-madaros-local-array-copy-bootstrap-replay
Status: resolved-on-semantic-head
Severity: B2
Class: compiler-lowering-metadata
Owner: Codex agent /root/fixed_array_call_abi
Lane: PR #915 local known-extent word-array copy promotion
Worktree: /tmp/sounio-madaros-fixed-array-copy-main-20260714
Branch: codex/madaros-fixed-array-copy-main-20260714
Files-Owned: self-hosted/ir/lower.sio, scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh, tests/native-v2/local_known_extent_word_array_copy_witness.sio, docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json, docs/handoff/madaros_fixed_array_call_boundary_alias_2026-07-14.md
Files-Read-Only: scripts/ci/build_modular_madaros.sh, .github/workflows/ci.yml
Do-Not-Touch: canonical compiler wrappers and CI workflow; aggregate, nested-array, and unknown-extent legacy paths
Repro: SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_BIN=/tmp/sounio-pr915-madaros-artifact-8321978196/madaros SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_DIR=/tmp/sounio-pr915-word-array-source-fresh SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_KEEP=1 bash scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh
Observed: PR #916 head 3a6c5ebc artifact 842fad7d13e281cb38aa9c516f3721d0b6b21f93464103e8a7f4f48b3d14251b runs every local-copy assertion to rc=0 and passes the call-boundary gate; the first local-gate replay exposed only the pre-existing 128-byte string-literal limit, repaired by splitting the final receipt into two exact lines
Expected: gate rc=0 and exact PASS receipt for every narrow local-copy case
Acceptance-Gate: bash scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh against a compiler rebuilt from the candidate head
Evidence-Level: E4
Evidence: d9234e reducers distinguish inferred rc=81/one callee allocation from annotated rc=0/two allocations and manual rc=0/two allocations; Stage2 runs all three at rc=0; source-fresh 3a6c5ebc runs the original parameter-source witness to rc=0 and both named gates pass on compiler 842fad7d
Fallback-Path: none
Legacy-Kept: yes; the original copy loop and aggregate, nested-array, and unknown-extent paths remain unchanged
LLM-Offload: not-required
Next-Action: confirm final harness-only head CI and replay both gates against its exact source-fresh compiler artifact
```
