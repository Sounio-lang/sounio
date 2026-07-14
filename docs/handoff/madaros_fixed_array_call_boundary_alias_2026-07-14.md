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
Expected: rc=0, exact probe stdout PASS fixed_array_call_boundary_value_semantics caller=unchanged, and exact focused-witness stdout covering i64/i8/bool/f64 ownership, mutable-reference visibility, and prefix/array/suffix parameter stability
Acceptance-Gate: bash scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh
Evidence-Level: E4
Evidence: docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json records the original f841 gate and Stage2 acceptance; PR #915 Actions run 29360223188 produced the a1b63e5d source-fresh control artifact; the candidate Stage2 focused witness passes locally
Fallback-Path: none
Legacy-Kept: yes; local identifier-copy behavior and aggregate or nested-array paths are unchanged by the future call-boundary lane
LLM-Offload: not-required
Next-Action: commit the reviewed candidate, request a source-fresh Madaros CI build from that exact commit, download its compiler artifact, and require the acceptance gate to return rc=0
```

## Semantic Lane Declaration

```text
Semantic-Lane-ID: fixed-array-call-abi-20260714
Owner: Codex agent /root/fixed_array_call_abi
Concept-IDs: none
Intent-Preserved: by-value direct known-extent word-scalar fixed-array parameters own backing storage distinct from the caller
Transformation: bind an eligible callee parameter name to a fresh element-wise copy while keeping the received ABI parameter register unchanged
Types-Changed: none
Effects-Changed: none
IR-Changed: instruction sequence only; no opcode or field meaning changed
Claims-Introduced: the named gate may prove caller isolation for its witnessed direct word-scalar arrays
Claims-Forbidden: f128/f256 support; structs; nested arrays; unknown extents; arbitrary whitelist-wide runtime parity; reference isolation
Assumptions: direct fixed arrays are represented by aggregate handles and each whitelisted element occupies one native-v2 word slot
Write-Set: self-hosted/ir/lower.sio; focused call-boundary probe, witness, gate, handoff, and receipt
Read-Set: self-hosted/ir/ir.sio; scripts/ci/build_modular_madaros.sh; self-hosted/native/codegen_x86_linux.sio
Positive-Witness: tests/native-v2/fixed_array_call_boundary_value_witness.sio
Negative-Witness: mutable &![i64; 2] parameter mutation remains caller-visible in the same witness
Acceptance-Gate: bash scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh
Integration-Target: stacked on PR #915 head 31fb049731b38600568fb4d525e92913e2c32e23
Authoritative-Only-If: the gate returns rc=0 with a Madaros built from the exact candidate commit
```

The Stage2 result is an acceptance oracle, not evidence that the candidate
current-source Madaros already satisfies the call-boundary contract. Struct,
nested-array, unknown-extent, and reference exclusions are currently enforced
by the top-level `TypeArray` and direct word-scalar predicates; only mutable
reference behavior has an executable negative witness in this lane.
