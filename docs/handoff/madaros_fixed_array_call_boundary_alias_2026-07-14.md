# Madaros Fixed-Array Call-Boundary Alias

The local identifier-copy vertical does not change call ABI. This blocker keeps
that boundary executable and separate from local known-extent word-array copy.

```text
Blocker-ID: BLK-20260714-madaros-fixed-array-call-boundary-alias
Status: classified
Severity: B1
Class: compiler-semantics
Owner: Codex agent /root/fixed_array_call_abi
Lane: Madaros fixed-array by-value call-boundary semantics
Worktree: /tmp/sounio-madaros-fixed-array-copy-main-20260714
Branch: codex/madaros-fixed-array-copy-main-20260714
Files-Owned: self-hosted/ir/lower.sio, self-hosted/native/codegen_x86_linux.sio, tests/known_failures/madaros_fixed_array_call_boundary_alias_probe.sio, scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh, docs/handoff/madaros_fixed_array_call_boundary_alias_2026-07-14.md, docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json
Files-Read-Only: self-hosted/ir/ir.sio, scripts/ci/build_modular_madaros.sh
Do-Not-Touch: scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh semantics; aggregate and nested-array legacy paths
Repro: SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_BIN=/tmp/madaros-current-source-f64-lowering-899/madaros SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_DIR=/tmp/sounio-madaros-array-call-boundary-f841 SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_KEEP=1 bash scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh
Observed: CI-derived Madaros f841534799c53be79801c31d218b6f76bb1e7dfe3958b0c441475f516abfe3f7 executes the probe with rc=61 and exact diagnostic BLOCKED fixed_array_call_boundary_alias caller_changed_after_by_value_param_mutation
Expected: rc=0 and exact stdout PASS fixed_array_call_boundary_value_semantics caller=unchanged
Acceptance-Gate: bash scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh
Evidence-Level: E3
Evidence: docs/audit/receipts/madaros_fixed_array_call_boundary_alias_2026-07-14.json records exact commands, hashes, rc values, and stdout for the f841 gate and Stage2 acceptance runs
Fallback-Path: none
Legacy-Kept: yes; local identifier-copy behavior and aggregate or nested-array paths are unchanged by the future call-boundary lane
LLM-Offload: not-required
Next-Action: implement a separate call-boundary ABI fix, then rebuild current-source Madaros and require the acceptance gate to return rc=0
```

The Stage2 result is an acceptance oracle, not evidence that current-source
Madaros already satisfies the call-boundary contract.
