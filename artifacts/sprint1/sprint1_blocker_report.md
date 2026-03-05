# Sprint 1 Blocker Report

Generated: 2026-03-05T18:53:51.689370+00:00

## Status
- correctness_closed: `True`
- sprint1_overall_status: `pass`
- perf_gate: `pass` reason=`target_met` mode=`jit` net_seconds=`0.92`
- run_lane (non-gating): `fail` reason=`target_not_met` net_seconds=`68.36`
- jit_debug: `available` jit_runtime_blocked=`False` usable_candidate_count=`1`
- jit_import_stress (non-gating): `pass` reason=`ok` stack_overflow_count=`0` timeout_count=`0` jit_panic_count=`0`
- jit_wasm_stack_depth (non-gating): `pass` reason=`ok` first_failure_stage=`` stack_overflow_count=`0` timeout_count=`0`
- jit_ir_timeout_repro (non-gating): `fail` reason=`ir_module_timeout_detected` lane_split=`ir_specific_timeout` timeout_count=`2` compile_error_count=`0`
- jit_root_cause_bigpush (non-gating): `fail` reason=`stack_overflow_detected` hypotheses=`['wasm_shadow_failure_reproduced', 'wasm_shadow_partial_failure', 'ir_real_only_failure']` wasm_first_failure_probe=`wasm_real` ir_first_failure_probe=`ir_real_import`

## Blocker Classes
- `(none)`

## Diagnostic Classes
- `jit_main_source_probe_timeout_non_gating`
- `jit_ir_timeout_lane_non_gating`
- `jit_bigpush_failure_non_gating`
- `jit_bigpush_ir_real_only_non_gating`
- `jit_bigpush_wasm_shadow_repro_non_gating`

## Evidence
- critical_gate: `/home/demetrios/work/sounio/artifacts/sprint1/critical_bug_fixes_gate.v1.json`
- perf_gate: `/home/demetrios/work/sounio/artifacts/sprint1/int_to_string_perf_gate.v1.json`
- run_lane: `/home/demetrios/work/sounio/artifacts/sprint1/int_to_string_perf_run_lane.v1.json`
- jit_debug: `/home/demetrios/work/sounio/artifacts/sprint1/jit_runtime_debug.v1.json`
- jit_import_stress: `/home/demetrios/work/sounio/artifacts/sprint1/jit_import_stress_debug.v1.json`
- jit_wasm_stack_depth: `/home/demetrios/work/sounio/artifacts/sprint1/jit_wasm_stack_depth_debug.v1.json`
- jit_ir_timeout_repro: `/home/demetrios/work/sounio/artifacts/sprint1/jit_ir_timeout_repro.v1.json`
- jit_root_cause_bigpush: `/home/demetrios/work/sounio/artifacts/sprint1/jit_root_cause_bigpush.v1.json`

## Next Actions
- Investigate timeout in diagnostic main.sio JIT probe (non-gating).
- Investigate ir::ir timeout lane using parser_ast vs ir_ir fixture split from jit_ir_timeout_repro.
- Inspect jit_root_cause_bigpush family-first-failure probes and promote strongest hypothesis into focused fix lane.
- Treat ir timeout as real-module-specific and compare real ir::ir against ir shadow stages in jit_probe modules.
- Treat wasm failures as reproducible in shadow stages and focus on smallest failing shadow stage first.
- Re-run scripts/sprint1_jit_runtime_debug.sh and require usable_candidate_count >= 1.
- Re-run scripts/sprint1_critical_bug_fixes_gate.sh and expect int_to_string_benchmark to pass in jit mode.
