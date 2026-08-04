<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260804-p0b-qd128-native-v2
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260804-p0b-qd128-native-v2
-->

# Blocker: BLK-20260804-p0b-qd128-native-v2

```text
Blocker-ID: BLK-20260804-p0b-qd128-native-v2
Status: classified
Severity: B1
Class: compiler-native
Owner: unassigned (needs native codegen / IR lower window; do not steal from ACTIVE Codex claims on lower.sio / codegen_x86_linux.sio)
Lane: p0b-native-v2-zero-event-20260804 (residual after sedenion array-ref closeout)
Worktree: /tmp/sounio-p0b-native-v2-20260804
Branch: research/p0b-native-v2-zero-event-20260804
Files-Owned: (open — likely self-hosted/native/codegen_x86_linux.sio and/or stdlib math::qd128 shape)
Do-Not-Touch: self-hosted/ir/lower.sio, self-hosted/native/codegen_x86_linux.sio while Codex epistemic claim is ACTIVE
Repro: SOUNIO_NV2_IR_TRACE=1 ./bin/souc compile tests/known_failures/qd128_import_native_v2_probe.sio -o /tmp/qd.elf
Observed: imported lowering completes (into_acc_done 2); native emit fails closed with
  "NV2_IR unsupported fn=… name=qd_mul" then "Failed to write native binary … rc=12"
Expected: qd_zero() import compiles+runs under default Madaros and prints QD128_IMPORT_NATIVE_V2 PASS
Acceptance-Gate: scripts/ci/zero_event_native_v2_matrix.sh qd128 lane flips to rc=0 + PASS marker;
  neighboring zero_provenance probe (eisa→qd_mul) and/or stdlib receipt constructors may remain separate BLKs
Evidence-Level: E3
Evidence: /tmp/p0b_qd.log; SOUNIO_NV2_IR_TRACE dump naming qd_mul; matrix fail-closed markers
Fallback-Path: dd64 import smoke remains green; use lean_single for qd128/oracle surfaces
Legacy-Kept: yes (known_failures probes retained)
LLM-Offload: not-required (compiler residual; no PK/clinical claim change)
Next-Action: When codegen write window is free, claim native surfaces and classify why
  compile_ir_function_v2_from_ir_into rejects qd_mul (opcode / frame / arity). Prefer stdlib
  reshape only if the failure is a known huge-scalar-param pattern (as closed for sedenion).
Related: zero_event_stdlib_native_v2_probe also rc=12 on ze_*_f64 constructors (aggregate return);
  keep lean_single oracle in scripts/ci/zero_event_gate.sh until that surface closes.
```

## Context

Attention P0=B closed the **sedenion** native-v2 import path by rewriting
`stdlib/algebra/sedenion.sio` helpers to array refs (avoiding 16/32-scalar f64
param lists that native-v2 could not emit). The historical matrix expectation
that sedenion “compiles but exits 1” is obsolete; sedenion is now a run-pass
smoke. `qd128` and the combined provenance probe remain fail-closed on `qd_mul`.
