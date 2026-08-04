<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260804-p0b-qd128-native-v2
authority: repo_only
audience: users
last_validated: 2026-08-04
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260804-p0b-qd128-native-v2
-->

# Blocker: BLK-20260804-p0b-qd128-native-v2

```text
Blocker-ID: BLK-20260804-p0b-qd128-native-v2
Status: classified (narrowed 2026-08-04)
Severity: B1
Class: compiler-native
Owner: unassigned (needs native codegen window; do not steal ACTIVE Codex claims on lower.sio / codegen_x86_linux.sio)
Lane: p0b2-qd128-leaf-20260804 (stdlib thin-leaf closeout; arithmetic residual remains)
Worktree: /tmp/sounio-p0b2-qd128-leaf-20260804
Branch: research/p0b2-qd128-leaf-20260804
Files-Owned: (open — self-hosted/native/codegen_x86_linux.sio and/or math::qd128 arithmetic reshape)
Do-Not-Touch: self-hosted/ir/lower.sio, self-hosted/native/codegen_x86_linux.sio while Codex epistemic claim is ACTIVE
Repro: SOUNIO_NV2_IR_TRACE=1 ./bin/souc compile tests/known_failures/qd128_import_native_v2_probe.sio -o /tmp/qd.elf
Observed: imported lowering completes (into_acc_done 2); native emit fails closed with
  "NV2_IR unsupported fn=… name=qd_mul" then "Failed to write native binary … rc=12"
Expected: full math::qd128::{qd_zero,qd_mul,…} import compiles+runs under default Madaros
Acceptance-Gate: known_failures/qd128_import_native_v2_probe.sio flips to rc=0 + QD128_IMPORT_NATIVE_V2 PASS;
  zero_provenance (eisa→qd_mul) may remain a separate residual
Evidence-Level: E3
Evidence: matrix fail-closed full-qd128 lane; SOUNIO_NV2_IR_TRACE naming qd_mul
Fallback-Path: use math::qd128_core::{qd_zero,qd_from_f64} under Madaros
  (gate scripts/ci/madaros_qd128_core_native_v2_gate.sh → MADAROS_QD128_CORE_NATIVE_V2_GATE_OK);
  lean_single for full arithmetic oracles
Legacy-Kept: yes (full math::qd128 + known_failures probe retained)
LLM-Offload: not-required
Next-Action: When codegen write window is free, classify why
  compile_ir_function_v2_from_ir_into rejects qd_mul (opcode / frame / arity).
```

## Context

Constructor import is **closed** via `stdlib/math/qd128_core.sio` (no `qd_mul`).
Full `math::qd128` remains fail-closed because importing the module compiles
`qd_mul` into the native-v2 unit. Combined zero-provenance stays blocked on the
same `qd_mul` edge through eisa.
