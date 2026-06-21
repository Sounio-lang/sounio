<!-- docs:meta
topic_id: repo.docs.audit.madaros-root2-readonly-probe-2026-06-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-root2-readonly-probe-2026-06-21
-->

# Madaros Root 2 Read-Only Probe — 2026-06-21

## Scope

This note records a read-only probe against the active Claude compiler lane for
the Madaros Root 2/SRET/enum/method-call blocker. It intentionally does not
change compiler source.

Current production baseline:

- `origin/main`: `f9e186318d08dbcc1d25b8bdbe454d2eda43626f`
  (`Merge pull request #352 from Sounio-lang/codex/madaros-production-plan`).
- `main` CI run `27891359138`: success.

Read-only probed lane:

- Worktree: `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b`
- Branch: `worktree-agent-adc1cd8b9d52ba53b`
- Base commit: `dcfad57a7e83598d76566ce452e8016a4cb5a705`
- Dirty compiler files:
  - `self-hosted/compiler/main.sio`
  - `self-hosted/native/codegen.sio`
  - `self-hosted/native/codegen_x86_linux.sio`
  - `self-hosted/native/lower_ir.sio`
  - `self-hosted/native/suite.sio`

Those files remain Claude-owned under the non-overlap rule in
`docs/audit/MADAROS_PRODUCTION_READINESS_PLAN_2026-06-21.md`.

## Diff Shape Observed

The active lane converts lower/codegen helpers from by-value return style to
in-place mutation style. Examples observed read-only:

- `lower_instr(nc: NativeCompiler, instr: IrInstr) -> NativeCompiler`
  becomes `lower_instr(nc: &! NativeCompiler, instr: IrInstr)`.
- Callers change from `c = lower_instr(c, instr)` to
  `lower_instr(&!c, instr)`.
- Affected helper families include `lower_instr`, alloc/field/index lowering,
  and native-v2 binop emission helpers.

This matches the Root 2 investigation area: avoiding lost by-value copies while
also ensuring in-place stores are not silently discarded.

## Commands Run

All commands were run from the Claude lane with the compiler-routing
environment unset:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/ci/madaros_operational_contract_gate.sh
```

Result:

```text
[madaros-contract] PASS: status doc, agent contract, default wrapper, and gate wiring are aligned
```

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/ci/native_v2_enum_match_gate.sh
```

Result:

```text
[native-v2-enum-match] souc=/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b/bin/souc
[native-v2-enum-match] out=/tmp/sounio-native-v2-enum-match.JSe8FE
exit 139
```

Preserved log summary:

```text
driver.check.log: check: OK
enum_match.compile.log:
  /workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b/bin/madaros:
  line 151: Segmentation fault "$RAW_MADAROS" --native-v2-compile "$src" "$out"
```

Focused harness probes:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh native_v2_f32_struct_sret --verbose
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh method_receiver_correct --verbose
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh native_enum_basic --verbose
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh sret_forwarding --verbose
```

Results:

| Filter | Selected | Result |
|---|---:|---|
| `native_v2_f32_struct_sret` | 1 | fail, run exited 1 |
| `method_receiver_correct` | 1 | fail, run exited 1 |
| `native_enum_basic` | 1 | fail, run exited 1 |
| `sret_forwarding` | 3 | `sret_forwarding_minimal` and `sret_forwarding_cross_module_cd_mul` fail with exit 139; `sret_forwarding_tuple_aggregate` skipped for no annotation |

Direct probes:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bin/souc run tests/run-pass/sret_forwarding_minimal.sio
```

Result:

```text
native_v2_compile: emitted path=/tmp/madaros-run.1eUMeZ/main.elf
bin/madaros: line 171: Segmentation fault "$out" "$@"
exit 139
```

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bin/souc run tests/run-pass/sret_forwarding_cross_module_cd_mul.sio
```

Result:

```text
native_v2_compile: emitted path=/tmp/madaros-run.3XhgJI/main.elf
bin/madaros: line 171: Segmentation fault "$out" "$@"
exit 139
```

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bin/madaros --native-v2-compile examples/native/enum_match.sio /tmp/enum_match.direct.elf
```

Result:

```text
native_v2_compile: front-half failed: type_check_failed
```

This direct enum compile did not reproduce the `native_v2_enum_match_gate.sh`
segfault shape. The gate path uses `self-hosted/compiler/native_compile_driver.sio`
and crashed inside the raw Madaros `--native-v2-compile` call.

## Classification

| Case | Current classification |
|---|---|
| Operational contract | green; routing/docs/default wrapper contract is not the immediate blocker |
| `native_v2_enum_match_gate.sh` | compiler/runtime-lowering failure; driver check passes, native compile path segfaults |
| `sret_forwarding_minimal.sio` | generated ELF segfaults at runtime; this is current evidence, not stale archive evidence |
| `sret_forwarding_cross_module_cd_mul.sio` | generated ELF segfaults at runtime; this is current evidence, not stale archive evidence |
| `native_v2_f32_struct_sret.sio` | stale or stricter typecheck interaction: f32 fields reject f64 literals before runtime evidence |
| `native_enum_basic.sio` | stale or stricter type/effect interaction: enum value passed where i64 expected and IO effects missing |
| `method_receiver_correct.sio` | compile path fails with no ELF; needs compiler-owner inspection of detailed compile log |

## Blocker Record

```text
Blocker-ID: MADAROS-ROOT2-SRET-ARCHIVE-TRIAGE-2026-06-21
Status: reproduced
Severity: B1
Class: compiler-semantics
Owner: compiler lane
Lane: Madaros Root 2/SRET/enum/method-call repair
Worktree: /workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b
Branch: worktree-agent-adc1cd8b9d52ba53b
Files-Owned: self-hosted/compiler/main.sio, self-hosted/native/codegen.sio, self-hosted/native/codegen_x86_linux.sio, self-hosted/native/lower_ir.sio, self-hosted/native/suite.sio
Files-Read-Only: docs/audit/MADAROS_PRODUCTION_READINESS_PLAN_2026-06-21.md, scripts/ci/native_v2_enum_match_gate.sh, scripts/ci/madaros_operational_contract_gate.sh, scripts/run_sio_test_suite.sh, tests/run-pass/sret_forwarding_minimal.sio, tests/run-pass/sret_forwarding_cross_module_cd_mul.sio
Do-Not-Touch: Claude-owned compiler files listed above unless ownership transfers
Repro: env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN bash scripts/ci/native_v2_enum_match_gate.sh
Observed: driver check passes, raw Madaros native-v2 compile path segfaults with exit 139
Expected: enum native-v2 driver emits executable that prints 2 and exits 0
Acceptance-Gate: native_v2_enum_match_gate.sh plus sret_forwarding_minimal and sret_forwarding_cross_module_cd_mul run-pass probes pass under environment-clean routing
Evidence-Level: E1
Evidence: /tmp/sounio-native-v2-enum-match.JSe8FE/driver.check.log and /tmp/sounio-native-v2-enum-match.JSe8FE/enum_match.compile.log on the 2026-06-21 workspace
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: compiler owner should decide whether the in-place lower/codegen rewrite is meant to close this; then rerun the same env-clean gate and SRET forwarding probes before any PR
```

## Next Commands For Compiler Owner

Run from the Claude lane:

```bash
cd /workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b
git status --short --branch

env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/ci/native_v2_enum_match_gate.sh

env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh sret_forwarding --verbose
```

If those remain red after the in-place rewrite, narrow with a debugger/core on
the raw `--native-v2-compile` crash and the generated SRET ELF runtime crash.

## Non-Actions

This probe intentionally did not:

- edit compiler files,
- edit `bin/madaros` or `bin/souc`,
- run heavy foundry/Slurm validation,
- claim the Root 2/SRET blocker is fixed.
