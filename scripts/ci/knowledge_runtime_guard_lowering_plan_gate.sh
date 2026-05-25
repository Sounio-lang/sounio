#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_knowledge_runtime_guard_lowering_plan_probe.sio"
ROW_PROBE="self-hosted/compiler/k2_knowledge_runtime_guard_backend_row_probe.sio"
EMIT_PROBE="self-hosted/compiler/k2_knowledge_runtime_guard_native_emission_plan_probe.sio"
STEP_PROBE="self-hosted/compiler/k2_knowledge_runtime_guard_native_emission_step_probe.sio"
DIRECTIVE_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_directive_positive.sio"
ORDINARY_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_call_positive.sio"
RETURN_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_positive.sio"
ASSIGN_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_assign_positive.sio"
UNIT_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_unit_positive.sio"
INTERNAL_LABEL_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_internal_label_positive.sio"
INTERNAL_LABEL_ZERO_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_internal_label_zero_positive.sio"
MULTI_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_multi_positive.sio"
UPPER_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_upper_positive.sio"
EQ_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_eq_positive.sio"
STATIC_SRC="tests/frontend/knowledge_static_value_positive.sio"
DIRECTIVE_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-directive-dynamic.log"
ORDINARY_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-ordinary-dynamic.log"
RETURN_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-return-dynamic.log"
ASSIGN_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-assign-dynamic.log"
UNIT_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-unit-dynamic.log"
INTERNAL_LABEL_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-internal-label-dynamic.log"
INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-internal-label-zero-row-0.log"
MULTI_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-multi-dynamic.log"
MULTI_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-backend-row-0.log"
MULTI_ROW1_LOG="$TMP_DIR/knowledge-runtime-guard-backend-row-1.log"
MULTI_ROW2_LOG="$TMP_DIR/knowledge-runtime-guard-backend-row-2.log"
MULTI_ROW3_LOG="$TMP_DIR/knowledge-runtime-guard-backend-row-3.log"
MULTI_ROW4_LOG="$TMP_DIR/knowledge-runtime-guard-backend-row-4.log"
UPPER_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-backend-row-upper-0.log"
EQ_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-backend-row-eq-0.log"
MULTI_EMIT_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-row-0.log"
MULTI_EMIT_ROW4_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-row-4.log"
UPPER_EMIT_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-upper-row-0.log"
EQ_EMIT_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-eq-row-0.log"
MULTI_STEP0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-step-0.log"
MULTI_STEP1_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-step-1.log"
MULTI_STEP2_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-step-2.log"
MULTI_STEP3_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-step-3.log"
MULTI_STEP4_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-step-4.log"
MULTI_STEP5_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-step-5.log"
MULTI_STEP6_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-step-6.log"
MULTI_STEP7_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-step-7.log"
UNIT_EMIT_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-row-0.log"
INTERNAL_LABEL_EMIT_ROW0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-internal-label-row-0.log"
INTERNAL_LABEL_STEP0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-internal-label-step-0.log"
UNIT_STEP0_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-0.log"
UNIT_STEP1_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-1.log"
UNIT_STEP2_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-2.log"
UNIT_STEP3_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-3.log"
UNIT_STEP4_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-4.log"
UNIT_STEP5_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-5.log"
UNIT_STEP6_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-6.log"
UNIT_STEP7_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-7.log"
UNIT_STEP8_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-8.log"
UNIT_STEP9_LOG="$TMP_DIR/knowledge-runtime-guard-native-emission-unit-step-9.log"
STATIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-static.log"

if bin/souc run "$PROBE" -- "$DIRECTIVE_DYNAMIC_SRC" >"$DIRECTIVE_DYNAMIC_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_lowering_plan_directive_found=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'directive_count=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'directive_malformed=0' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'parse_errors=0' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'semantic_verdict=0' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -Eq 'runtime_obligation_count=[1-9][0-9]*' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'needs_runtime_guards=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'pre_native_guard_expansion_requested=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'pre_native_guard_plan_ready=0' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'native_lowering_ready=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'backend_guard_count=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_obligation_found=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_obligation_site_kind=2' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_obligation_type=PatientRuntimeGuardDirectivePositive' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_obligation_field=age' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_obligation_constraint_kind=2' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_obligation_value_kind=2' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_obligation_int_value=18' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_row_ready=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_compare_opcode=2' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_threshold_kind=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_trap_exit_code=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'second_obligation_found=0' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'second_backend_guard_row_ready=0' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'staged_backend_guard_row_count=1' "$DIRECTIVE_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan routes directive source to bounded direct native lowering\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not classify directive dynamic source\n' >&2
  cat "$DIRECTIVE_DYNAMIC_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$ORDINARY_DYNAMIC_SRC" >"$ORDINARY_DYNAMIC_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_lowering_plan_directive_found=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'directive_count=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'directive_malformed=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'parse_errors=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'semantic_verdict=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -Eq 'runtime_obligation_count=[1-9][0-9]*' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'needs_runtime_guards=1' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'pre_native_guard_expansion_requested=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'pre_native_guard_plan_ready=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'native_lowering_ready=1' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'backend_guard_count=1' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_obligation_found=1' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_obligation_site_kind=2' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_obligation_type=PatientRuntimeGuardCallPositive' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_obligation_field=age' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_row_ready=1' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_compare_opcode=2' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_trap_exit_code=1' "$ORDINARY_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan routes ordinary dynamic source to bounded direct native lowering\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not classify ordinary dynamic source\n' >&2
  cat "$ORDINARY_DYNAMIC_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$RETURN_DYNAMIC_SRC" >"$RETURN_DYNAMIC_LOG" 2>&1 &&
   grep -q 'runtime_obligation_count=2' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'first_obligation_found=1' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'first_obligation_site_kind=1' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'first_obligation_type=PatientRuntimeGuardPositive' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'first_obligation_field=age' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_row_ready=1' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'second_obligation_found=1' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'second_obligation_site_kind=4' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'second_obligation_type=PatientRuntimeGuardPositive' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'second_obligation_field=age' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'second_backend_guard_row_ready=1' "$RETURN_DYNAMIC_LOG" &&
   grep -q 'staged_backend_guard_row_count=2' "$RETURN_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan maps return-site obligation payload to staged backend guard row\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not preserve return-site obligation payload\n' >&2
  cat "$RETURN_DYNAMIC_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$ASSIGN_DYNAMIC_SRC" >"$ASSIGN_DYNAMIC_LOG" 2>&1 &&
   grep -Eq 'runtime_obligation_count=[1-9][0-9]*' "$ASSIGN_DYNAMIC_LOG" &&
   grep -q 'first_obligation_found=1' "$ASSIGN_DYNAMIC_LOG" &&
   grep -q 'first_obligation_site_kind=3' "$ASSIGN_DYNAMIC_LOG" &&
   grep -q 'first_obligation_type=PatientRuntimeGuardAssignPositive' "$ASSIGN_DYNAMIC_LOG" &&
   grep -q 'first_obligation_field=age' "$ASSIGN_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_row_ready=1' "$ASSIGN_DYNAMIC_LOG" &&
   grep -q 'second_obligation_found=0' "$ASSIGN_DYNAMIC_LOG" &&
   grep -q 'second_backend_guard_row_ready=0' "$ASSIGN_DYNAMIC_LOG" &&
   grep -q 'staged_backend_guard_row_count=1' "$ASSIGN_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan maps assignment-site obligation payload to staged backend guard row\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not preserve assignment-site obligation payload\n' >&2
  cat "$ASSIGN_DYNAMIC_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$UNIT_DYNAMIC_SRC" >"$UNIT_DYNAMIC_LOG" 2>&1 &&
   grep -q 'runtime_obligation_count=2' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_obligation_found=1' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_obligation_site_kind=1' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_obligation_type=DoseRuntimeGuardUnitPositive' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_obligation_field=amount' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_obligation_constraint_kind=2' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_obligation_value_kind=2' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_obligation_int_value=500' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_obligation_unit=mg' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_row_ready=1' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_compare_opcode=2' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_threshold_kind=3' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_trap_exit_code=1' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'second_obligation_found=1' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'second_obligation_site_kind=4' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'second_obligation_type=DoseRuntimeGuardUnitPositive' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'second_obligation_field=amount' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'second_obligation_unit=mg' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'second_backend_guard_row_ready=1' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'staged_backend_guard_row_count=2' "$UNIT_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan maps unit-suffixed obligation payload to staged backend guard row\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not preserve unit-suffixed obligation payload\n' >&2
  cat "$UNIT_DYNAMIC_LOG" >&2
  exit 1
fi

if bin/souc run "$EMIT_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 >"$UNIT_EMIT_ROW0_LOG" 2>&1 &&
   grep -q 'plan_ready=1' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'source_valid=1' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'in_bounds=1' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'threshold_kind=3' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'threshold_int_value=500' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'native_cmp_code=10' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'emit_load_threshold=1' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'zero_cmp_code=5' "$UNIT_EMIT_ROW0_LOG" &&
   grep -q 'native_step_count=9' "$UNIT_EMIT_ROW0_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard unit-ratio row builds 9-step native emission plan\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard unit-ratio row did not build 9-step native emission plan\n' >&2
  cat "$UNIT_EMIT_ROW0_LOG" >&2
  exit 1
fi

if bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 0 >"$UNIT_STEP0_LOG" 2>&1 &&
   grep -q 'step_index=0' "$UNIT_STEP0_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP0_LOG" &&
   grep -q 'total_step_count=9' "$UNIT_STEP0_LOG" &&
   grep -q 'op_kind=1' "$UNIT_STEP0_LOG" &&
   grep -q 'dst_role=7' "$UNIT_STEP0_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 1 >"$UNIT_STEP1_LOG" 2>&1 &&
   grep -q 'step_index=1' "$UNIT_STEP1_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP1_LOG" &&
   grep -q 'op_kind=8' "$UNIT_STEP1_LOG" &&
   grep -q 'dst_role=8' "$UNIT_STEP1_LOG" &&
   grep -q 'src1_role=1' "$UNIT_STEP1_LOG" &&
   grep -q 'src2_role=7' "$UNIT_STEP1_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 2 >"$UNIT_STEP2_LOG" 2>&1 &&
   grep -q 'step_index=2' "$UNIT_STEP2_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP2_LOG" &&
   grep -q 'op_kind=9' "$UNIT_STEP2_LOG" &&
   grep -q 'dst_role=9' "$UNIT_STEP2_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 3 >"$UNIT_STEP3_LOG" 2>&1 &&
   grep -q 'step_index=3' "$UNIT_STEP3_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP3_LOG" &&
   grep -q 'op_kind=2' "$UNIT_STEP3_LOG" &&
   grep -q 'dst_role=3' "$UNIT_STEP3_LOG" &&
   grep -q 'src1_role=8' "$UNIT_STEP3_LOG" &&
   grep -q 'src2_role=9' "$UNIT_STEP3_LOG" &&
   grep -q 'native_cmp_code=10' "$UNIT_STEP3_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 4 >"$UNIT_STEP4_LOG" 2>&1 &&
   grep -q 'step_index=4' "$UNIT_STEP4_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP4_LOG" &&
   grep -q 'op_kind=3' "$UNIT_STEP4_LOG" &&
   grep -q 'dst_role=4' "$UNIT_STEP4_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 5 >"$UNIT_STEP5_LOG" 2>&1 &&
   grep -q 'step_index=5' "$UNIT_STEP5_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP5_LOG" &&
   grep -q 'op_kind=4' "$UNIT_STEP5_LOG" &&
   grep -q 'dst_role=5' "$UNIT_STEP5_LOG" &&
   grep -q 'src1_role=3' "$UNIT_STEP5_LOG" &&
   grep -q 'src2_role=4' "$UNIT_STEP5_LOG" &&
   grep -q 'native_cmp_code=5' "$UNIT_STEP5_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 6 >"$UNIT_STEP6_LOG" 2>&1 &&
   grep -q 'step_index=6' "$UNIT_STEP6_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP6_LOG" &&
   grep -q 'op_kind=5' "$UNIT_STEP6_LOG" &&
   grep -q 'src1_role=5' "$UNIT_STEP6_LOG" &&
   grep -q 'label_role=1' "$UNIT_STEP6_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 7 >"$UNIT_STEP7_LOG" 2>&1 &&
   grep -q 'step_index=7' "$UNIT_STEP7_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP7_LOG" &&
   grep -q 'op_kind=6' "$UNIT_STEP7_LOG" &&
   grep -q 'dst_role=6' "$UNIT_STEP7_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$UNIT_STEP7_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 8 >"$UNIT_STEP8_LOG" 2>&1 &&
   grep -q 'step_index=8' "$UNIT_STEP8_LOG" &&
   grep -q 'step_ready=1' "$UNIT_STEP8_LOG" &&
   grep -q 'op_kind=7' "$UNIT_STEP8_LOG" &&
   grep -q 'label_role=1' "$UNIT_STEP8_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$UNIT_DYNAMIC_SRC" 0 9 >"$UNIT_STEP9_LOG" 2>&1 &&
   grep -q 'step_index=9' "$UNIT_STEP9_LOG" &&
   grep -q 'step_ready=0' "$UNIT_STEP9_LOG" &&
   grep -q 'plan_ready=1' "$UNIT_STEP9_LOG" &&
   grep -q 'total_step_count=9' "$UNIT_STEP9_LOG" &&
   grep -q 'op_kind=0' "$UNIT_STEP9_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard unit-ratio emission plan exposes 9-step indexed backend steps\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard unit-ratio emission plan did not expose 9-step indexed backend steps\n' >&2
  cat "$UNIT_STEP0_LOG" >&2
  cat "$UNIT_STEP1_LOG" >&2
  cat "$UNIT_STEP2_LOG" >&2
  cat "$UNIT_STEP3_LOG" >&2
  cat "$UNIT_STEP4_LOG" >&2
  cat "$UNIT_STEP5_LOG" >&2
  cat "$UNIT_STEP6_LOG" >&2
  cat "$UNIT_STEP7_LOG" >&2
  cat "$UNIT_STEP8_LOG" >&2
  cat "$UNIT_STEP9_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$INTERNAL_LABEL_DYNAMIC_SRC" >"$INTERNAL_LABEL_DYNAMIC_LOG" 2>&1 &&
   grep -q 'native_lowering_ready=1' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'backend_guard_count=2' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'first_obligation_type=PatientRuntimeGuardInternalLabelPositive' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'first_obligation_field=glucose' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'first_obligation_value_kind=3' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'first_obligation_float_value=126.000000' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'first_obligation_scaled_value=1260' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'first_obligation_value_scale=10' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'first_obligation_unit=mg_dL' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_threshold_kind=3' "$INTERNAL_LABEL_DYNAMIC_LOG" &&
   bin/souc run "$EMIT_PROBE" -- "$INTERNAL_LABEL_DYNAMIC_SRC" 0 >"$INTERNAL_LABEL_EMIT_ROW0_LOG" 2>&1 &&
   grep -q 'plan_ready=1' "$INTERNAL_LABEL_EMIT_ROW0_LOG" &&
   grep -q 'threshold_kind=3' "$INTERNAL_LABEL_EMIT_ROW0_LOG" &&
   grep -q 'threshold_int_value=1260' "$INTERNAL_LABEL_EMIT_ROW0_LOG" &&
   grep -q 'threshold_scaled_value=1260' "$INTERNAL_LABEL_EMIT_ROW0_LOG" &&
   grep -q 'threshold_scale=10' "$INTERNAL_LABEL_EMIT_ROW0_LOG" &&
   grep -q 'native_step_count=9' "$INTERNAL_LABEL_EMIT_ROW0_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$INTERNAL_LABEL_DYNAMIC_SRC" 0 0 >"$INTERNAL_LABEL_STEP0_LOG" 2>&1 &&
   grep -q 'step_ready=1' "$INTERNAL_LABEL_STEP0_LOG" &&
   grep -q 'threshold_int_value=1260' "$INTERNAL_LABEL_STEP0_LOG" &&
   grep -q 'threshold_scaled_value=1260' "$INTERNAL_LABEL_STEP0_LOG" &&
   grep -q 'threshold_scale=10' "$INTERNAL_LABEL_STEP0_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan preserves decimal unit threshold payloads\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not preserve decimal unit threshold payloads\n' >&2
  cat "$INTERNAL_LABEL_DYNAMIC_LOG" >&2
  cat "$INTERNAL_LABEL_EMIT_ROW0_LOG" >&2
  cat "$INTERNAL_LABEL_STEP0_LOG" >&2
  exit 1
fi

if bin/souc run "$EMIT_PROBE" -- "$INTERNAL_LABEL_ZERO_DYNAMIC_SRC" 0 >"$INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG" 2>&1 &&
   grep -q 'plan_ready=1' "$INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG" &&
   grep -q 'threshold_kind=3' "$INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG" &&
   grep -q 'threshold_int_value=0' "$INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG" &&
   grep -q 'threshold_scaled_value=0' "$INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG" &&
   grep -q 'threshold_scale=10' "$INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG" &&
   grep -q 'native_step_count=7' "$INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan preserves zero decimal unit threshold payload\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not preserve zero decimal unit threshold payload\n' >&2
  cat "$INTERNAL_LABEL_ZERO_EMIT_ROW0_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$MULTI_DYNAMIC_SRC" >"$MULTI_DYNAMIC_LOG" 2>&1 &&
   grep -q 'runtime_obligation_count=4' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'first_obligation_found=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'first_obligation_site_kind=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'first_obligation_type=PatientRuntimeGuardMultiPositive' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'first_obligation_field=glucose' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_row_ready=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_compare_opcode=2' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_threshold_kind=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_trap_exit_code=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_obligation_found=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_obligation_site_kind=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_obligation_type=PatientRuntimeGuardMultiPositive' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_obligation_field=age' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_obligation_int_value=18' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_backend_guard_row_ready=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_backend_guard_compare_opcode=2' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_backend_guard_threshold_kind=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'second_backend_guard_trap_exit_code=1' "$MULTI_DYNAMIC_LOG" &&
   grep -q 'staged_backend_guard_row_count=4' "$MULTI_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan enumerates conjunctive backend guard row payloads\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not stage all conjunctive obligation rows\n' >&2
  cat "$MULTI_DYNAMIC_LOG" >&2
  exit 1
fi

if bin/souc run "$ROW_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 >"$MULTI_ROW0_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_backend_row_index=0' "$MULTI_ROW0_LOG" &&
   grep -q 'source_valid=1' "$MULTI_ROW0_LOG" &&
   grep -q 'runtime_obligation_count=4' "$MULTI_ROW0_LOG" &&
   grep -q 'in_bounds=1' "$MULTI_ROW0_LOG" &&
   grep -q 'obligation_found=1' "$MULTI_ROW0_LOG" &&
   grep -q 'site_kind=1' "$MULTI_ROW0_LOG" &&
   grep -q 'type=PatientRuntimeGuardMultiPositive' "$MULTI_ROW0_LOG" &&
   grep -q 'field=glucose' "$MULTI_ROW0_LOG" &&
   grep -q 'int_value=126' "$MULTI_ROW0_LOG" &&
   grep -q 'row_ready=1' "$MULTI_ROW0_LOG" &&
   grep -q 'compare_opcode=2' "$MULTI_ROW0_LOG" &&
   grep -q 'native_cmp_code=10' "$MULTI_ROW0_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$MULTI_ROW0_LOG" &&
   grep -q 'native_trap_on_false=1' "$MULTI_ROW0_LOG" &&
   grep -q 'threshold_kind=1' "$MULTI_ROW0_LOG" &&
   grep -q 'trap_exit_code=1' "$MULTI_ROW0_LOG" &&
   bin/souc run "$ROW_PROBE" -- "$MULTI_DYNAMIC_SRC" 1 >"$MULTI_ROW1_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_backend_row_index=1' "$MULTI_ROW1_LOG" &&
   grep -q 'runtime_obligation_count=4' "$MULTI_ROW1_LOG" &&
   grep -q 'in_bounds=1' "$MULTI_ROW1_LOG" &&
   grep -q 'obligation_found=1' "$MULTI_ROW1_LOG" &&
   grep -q 'site_kind=1' "$MULTI_ROW1_LOG" &&
   grep -q 'type=PatientRuntimeGuardMultiPositive' "$MULTI_ROW1_LOG" &&
   grep -q 'field=age' "$MULTI_ROW1_LOG" &&
   grep -q 'int_value=18' "$MULTI_ROW1_LOG" &&
   grep -q 'row_ready=1' "$MULTI_ROW1_LOG" &&
   grep -q 'compare_opcode=2' "$MULTI_ROW1_LOG" &&
   grep -q 'native_cmp_code=10' "$MULTI_ROW1_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$MULTI_ROW1_LOG" &&
   grep -q 'native_trap_on_false=1' "$MULTI_ROW1_LOG" &&
   grep -q 'threshold_kind=1' "$MULTI_ROW1_LOG" &&
   grep -q 'trap_exit_code=1' "$MULTI_ROW1_LOG" &&
   bin/souc run "$ROW_PROBE" -- "$MULTI_DYNAMIC_SRC" 2 >"$MULTI_ROW2_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_backend_row_index=2' "$MULTI_ROW2_LOG" &&
   grep -q 'runtime_obligation_count=4' "$MULTI_ROW2_LOG" &&
   grep -q 'in_bounds=1' "$MULTI_ROW2_LOG" &&
   grep -q 'obligation_found=1' "$MULTI_ROW2_LOG" &&
   grep -q 'site_kind=4' "$MULTI_ROW2_LOG" &&
   grep -q 'type=PatientRuntimeGuardMultiPositive' "$MULTI_ROW2_LOG" &&
   grep -q 'field=glucose' "$MULTI_ROW2_LOG" &&
   grep -q 'int_value=126' "$MULTI_ROW2_LOG" &&
   grep -q 'row_ready=1' "$MULTI_ROW2_LOG" &&
   grep -q 'compare_opcode=2' "$MULTI_ROW2_LOG" &&
   grep -q 'native_cmp_code=10' "$MULTI_ROW2_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$MULTI_ROW2_LOG" &&
   grep -q 'native_trap_on_false=1' "$MULTI_ROW2_LOG" &&
   grep -q 'threshold_kind=1' "$MULTI_ROW2_LOG" &&
   grep -q 'trap_exit_code=1' "$MULTI_ROW2_LOG" &&
   bin/souc run "$ROW_PROBE" -- "$MULTI_DYNAMIC_SRC" 3 >"$MULTI_ROW3_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_backend_row_index=3' "$MULTI_ROW3_LOG" &&
   grep -q 'runtime_obligation_count=4' "$MULTI_ROW3_LOG" &&
   grep -q 'in_bounds=1' "$MULTI_ROW3_LOG" &&
   grep -q 'obligation_found=1' "$MULTI_ROW3_LOG" &&
   grep -q 'site_kind=4' "$MULTI_ROW3_LOG" &&
   grep -q 'type=PatientRuntimeGuardMultiPositive' "$MULTI_ROW3_LOG" &&
   grep -q 'field=age' "$MULTI_ROW3_LOG" &&
   grep -q 'int_value=18' "$MULTI_ROW3_LOG" &&
   grep -q 'row_ready=1' "$MULTI_ROW3_LOG" &&
   grep -q 'compare_opcode=2' "$MULTI_ROW3_LOG" &&
   grep -q 'native_cmp_code=10' "$MULTI_ROW3_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$MULTI_ROW3_LOG" &&
   grep -q 'native_trap_on_false=1' "$MULTI_ROW3_LOG" &&
   grep -q 'threshold_kind=1' "$MULTI_ROW3_LOG" &&
   grep -q 'trap_exit_code=1' "$MULTI_ROW3_LOG" &&
   bin/souc run "$ROW_PROBE" -- "$MULTI_DYNAMIC_SRC" 4 >"$MULTI_ROW4_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_backend_row_index=4' "$MULTI_ROW4_LOG" &&
   grep -q 'source_valid=1' "$MULTI_ROW4_LOG" &&
   grep -q 'runtime_obligation_count=4' "$MULTI_ROW4_LOG" &&
   grep -q 'in_bounds=0' "$MULTI_ROW4_LOG" &&
   grep -q 'obligation_found=0' "$MULTI_ROW4_LOG" &&
   grep -q 'row_ready=0' "$MULTI_ROW4_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan exposes indexed backend guard row_at payloads\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not expose indexed backend row_at payloads\n' >&2
  cat "$MULTI_ROW0_LOG" >&2
  cat "$MULTI_ROW1_LOG" >&2
  cat "$MULTI_ROW2_LOG" >&2
  cat "$MULTI_ROW3_LOG" >&2
  cat "$MULTI_ROW4_LOG" >&2
  exit 1
fi

if bin/souc run "$ROW_PROBE" -- "$UPPER_DYNAMIC_SRC" 0 >"$UPPER_ROW0_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_backend_row_index=0' "$UPPER_ROW0_LOG" &&
   grep -q 'runtime_obligation_count=2' "$UPPER_ROW0_LOG" &&
   grep -q 'in_bounds=1' "$UPPER_ROW0_LOG" &&
   grep -q 'obligation_found=1' "$UPPER_ROW0_LOG" &&
   grep -q 'field=score' "$UPPER_ROW0_LOG" &&
   grep -q 'constraint_kind=4' "$UPPER_ROW0_LOG" &&
   grep -q 'compare_opcode=4' "$UPPER_ROW0_LOG" &&
   grep -q 'native_cmp_code=8' "$UPPER_ROW0_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$UPPER_ROW0_LOG" &&
   grep -q 'native_trap_on_false=1' "$UPPER_ROW0_LOG" &&
   grep -q 'row_ready=1' "$UPPER_ROW0_LOG" &&
   bin/souc run "$ROW_PROBE" -- "$EQ_DYNAMIC_SRC" 0 >"$EQ_ROW0_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_backend_row_index=0' "$EQ_ROW0_LOG" &&
   grep -q 'runtime_obligation_count=2' "$EQ_ROW0_LOG" &&
   grep -q 'in_bounds=1' "$EQ_ROW0_LOG" &&
   grep -q 'obligation_found=1' "$EQ_ROW0_LOG" &&
   grep -q 'field=repeats' "$EQ_ROW0_LOG" &&
   grep -q 'constraint_kind=1' "$EQ_ROW0_LOG" &&
   grep -q 'compare_opcode=1' "$EQ_ROW0_LOG" &&
   grep -q 'native_cmp_code=5' "$EQ_ROW0_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$EQ_ROW0_LOG" &&
   grep -q 'native_trap_on_false=1' "$EQ_ROW0_LOG" &&
   grep -q 'row_ready=1' "$EQ_ROW0_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard row_at maps supported guard operators to native cmp/trap shape\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard row_at did not map supported guard operators to native cmp/trap shape\n' >&2
  cat "$UPPER_ROW0_LOG" >&2
  cat "$EQ_ROW0_LOG" >&2
  exit 1
fi

if bin/souc run "$EMIT_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 >"$MULTI_EMIT_ROW0_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_native_emission_row_index=0' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'plan_ready=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'source_valid=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'in_bounds=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'runtime_obligation_count=4' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'needs_value_register=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'threshold_kind=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'threshold_int_value=126' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'native_cmp_code=10' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'native_trap_on_false=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'emit_load_threshold=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'emit_compare_value_threshold=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'emit_load_zero=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'emit_eq_zero=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'emit_branch_false_to_ok=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'emit_trap_call=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'emit_ok_label=1' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'zero_cmp_code=5' "$MULTI_EMIT_ROW0_LOG" &&
   grep -q 'native_step_count=7' "$MULTI_EMIT_ROW0_LOG" &&
   bin/souc run "$EMIT_PROBE" -- "$MULTI_DYNAMIC_SRC" 4 >"$MULTI_EMIT_ROW4_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_native_emission_row_index=4' "$MULTI_EMIT_ROW4_LOG" &&
   grep -q 'plan_ready=0' "$MULTI_EMIT_ROW4_LOG" &&
   grep -q 'source_valid=1' "$MULTI_EMIT_ROW4_LOG" &&
   grep -q 'in_bounds=0' "$MULTI_EMIT_ROW4_LOG" &&
   grep -q 'runtime_obligation_count=4' "$MULTI_EMIT_ROW4_LOG" &&
   grep -q 'native_step_count=0' "$MULTI_EMIT_ROW4_LOG" &&
   bin/souc run "$EMIT_PROBE" -- "$UPPER_DYNAMIC_SRC" 0 >"$UPPER_EMIT_ROW0_LOG" 2>&1 &&
   grep -q 'plan_ready=1' "$UPPER_EMIT_ROW0_LOG" &&
   grep -q 'threshold_int_value=100' "$UPPER_EMIT_ROW0_LOG" &&
   grep -q 'native_cmp_code=8' "$UPPER_EMIT_ROW0_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$UPPER_EMIT_ROW0_LOG" &&
   grep -q 'zero_cmp_code=5' "$UPPER_EMIT_ROW0_LOG" &&
   grep -q 'native_step_count=7' "$UPPER_EMIT_ROW0_LOG" &&
   bin/souc run "$EMIT_PROBE" -- "$EQ_DYNAMIC_SRC" 0 >"$EQ_EMIT_ROW0_LOG" 2>&1 &&
   grep -q 'plan_ready=1' "$EQ_EMIT_ROW0_LOG" &&
   grep -q 'threshold_int_value=3' "$EQ_EMIT_ROW0_LOG" &&
   grep -q 'native_cmp_code=5' "$EQ_EMIT_ROW0_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$EQ_EMIT_ROW0_LOG" &&
   grep -q 'zero_cmp_code=5' "$EQ_EMIT_ROW0_LOG" &&
   grep -q 'native_step_count=7' "$EQ_EMIT_ROW0_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard row_at builds native emission plan shape\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard row_at did not build native emission plan shape\n' >&2
  cat "$MULTI_EMIT_ROW0_LOG" >&2
  cat "$MULTI_EMIT_ROW4_LOG" >&2
  cat "$UPPER_EMIT_ROW0_LOG" >&2
  cat "$EQ_EMIT_ROW0_LOG" >&2
  exit 1
fi

if bin/souc run "$STEP_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 0 >"$MULTI_STEP0_LOG" 2>&1 &&
   grep -q 'step_index=0' "$MULTI_STEP0_LOG" &&
   grep -q 'step_ready=1' "$MULTI_STEP0_LOG" &&
   grep -q 'plan_ready=1' "$MULTI_STEP0_LOG" &&
   grep -q 'total_step_count=7' "$MULTI_STEP0_LOG" &&
   grep -q 'op_kind=1' "$MULTI_STEP0_LOG" &&
   grep -q 'dst_role=2' "$MULTI_STEP0_LOG" &&
   grep -q 'threshold_int_value=126' "$MULTI_STEP0_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 1 >"$MULTI_STEP1_LOG" 2>&1 &&
   grep -q 'step_index=1' "$MULTI_STEP1_LOG" &&
   grep -q 'step_ready=1' "$MULTI_STEP1_LOG" &&
   grep -q 'op_kind=2' "$MULTI_STEP1_LOG" &&
   grep -q 'dst_role=3' "$MULTI_STEP1_LOG" &&
   grep -q 'src1_role=1' "$MULTI_STEP1_LOG" &&
   grep -q 'src2_role=2' "$MULTI_STEP1_LOG" &&
   grep -q 'native_cmp_code=10' "$MULTI_STEP1_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 2 >"$MULTI_STEP2_LOG" 2>&1 &&
   grep -q 'step_index=2' "$MULTI_STEP2_LOG" &&
   grep -q 'step_ready=1' "$MULTI_STEP2_LOG" &&
   grep -q 'op_kind=3' "$MULTI_STEP2_LOG" &&
   grep -q 'dst_role=4' "$MULTI_STEP2_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 3 >"$MULTI_STEP3_LOG" 2>&1 &&
   grep -q 'step_index=3' "$MULTI_STEP3_LOG" &&
   grep -q 'step_ready=1' "$MULTI_STEP3_LOG" &&
   grep -q 'op_kind=4' "$MULTI_STEP3_LOG" &&
   grep -q 'dst_role=5' "$MULTI_STEP3_LOG" &&
   grep -q 'src1_role=3' "$MULTI_STEP3_LOG" &&
   grep -q 'src2_role=4' "$MULTI_STEP3_LOG" &&
   grep -q 'native_cmp_code=5' "$MULTI_STEP3_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 4 >"$MULTI_STEP4_LOG" 2>&1 &&
   grep -q 'step_index=4' "$MULTI_STEP4_LOG" &&
   grep -q 'step_ready=1' "$MULTI_STEP4_LOG" &&
   grep -q 'op_kind=5' "$MULTI_STEP4_LOG" &&
   grep -q 'src1_role=5' "$MULTI_STEP4_LOG" &&
   grep -q 'label_role=1' "$MULTI_STEP4_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 5 >"$MULTI_STEP5_LOG" 2>&1 &&
   grep -q 'step_index=5' "$MULTI_STEP5_LOG" &&
   grep -q 'step_ready=1' "$MULTI_STEP5_LOG" &&
   grep -q 'op_kind=6' "$MULTI_STEP5_LOG" &&
   grep -q 'dst_role=6' "$MULTI_STEP5_LOG" &&
   grep -q 'native_trap_builtin_id=20' "$MULTI_STEP5_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 6 >"$MULTI_STEP6_LOG" 2>&1 &&
   grep -q 'step_index=6' "$MULTI_STEP6_LOG" &&
   grep -q 'step_ready=1' "$MULTI_STEP6_LOG" &&
   grep -q 'op_kind=7' "$MULTI_STEP6_LOG" &&
   grep -q 'label_role=1' "$MULTI_STEP6_LOG" &&
   bin/souc run "$STEP_PROBE" -- "$MULTI_DYNAMIC_SRC" 0 7 >"$MULTI_STEP7_LOG" 2>&1 &&
   grep -q 'step_index=7' "$MULTI_STEP7_LOG" &&
   grep -q 'step_ready=0' "$MULTI_STEP7_LOG" &&
   grep -q 'plan_ready=1' "$MULTI_STEP7_LOG" &&
   grep -q 'total_step_count=7' "$MULTI_STEP7_LOG" &&
   grep -q 'op_kind=0' "$MULTI_STEP7_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard native emission plan exposes concrete indexed backend steps\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard native emission plan did not expose concrete indexed backend steps\n' >&2
  cat "$MULTI_STEP0_LOG" >&2
  cat "$MULTI_STEP1_LOG" >&2
  cat "$MULTI_STEP2_LOG" >&2
  cat "$MULTI_STEP3_LOG" >&2
  cat "$MULTI_STEP4_LOG" >&2
  cat "$MULTI_STEP5_LOG" >&2
  cat "$MULTI_STEP6_LOG" >&2
  cat "$MULTI_STEP7_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$STATIC_SRC" >"$STATIC_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_lowering_plan_directive_found=0' "$STATIC_LOG" &&
   grep -q 'directive_count=0' "$STATIC_LOG" &&
   grep -q 'directive_malformed=0' "$STATIC_LOG" &&
   grep -q 'parse_errors=0' "$STATIC_LOG" &&
   grep -q 'semantic_verdict=0' "$STATIC_LOG" &&
   grep -q 'runtime_obligation_count=0' "$STATIC_LOG" &&
   grep -q 'needs_runtime_guards=0' "$STATIC_LOG" &&
   grep -q 'pre_native_guard_expansion_requested=0' "$STATIC_LOG" &&
   grep -q 'pre_native_guard_plan_ready=0' "$STATIC_LOG" &&
   grep -q 'native_lowering_ready=0' "$STATIC_LOG" &&
   grep -q 'backend_guard_count=0' "$STATIC_LOG" &&
   grep -q 'first_obligation_found=0' "$STATIC_LOG" &&
   grep -q 'first_backend_guard_row_ready=0' "$STATIC_LOG" &&
   grep -q 'second_obligation_found=0' "$STATIC_LOG" &&
   grep -q 'second_backend_guard_row_ready=0' "$STATIC_LOG" &&
   grep -q 'staged_backend_guard_row_count=0' "$STATIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan leaves statically discharged source empty\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not classify statically discharged source\n' >&2
  cat "$STATIC_LOG" >&2
  exit 1
fi

cat <<'MSG'
Knowledge runtime guard lowering plan gate passed.
This proves the compiler-side Sounio path can join three pieces of evidence for
dynamic Knowledge<T> proof contexts: the raw-source `//@ knowledge-runtime-guards`
intent, ordinary parser/checker semantic acceptance, and checker-visible runtime
obligation counting plus indexed obligation payload extraction. Directive-bearing
dynamic sources now route to the bounded direct native-v2 lowering slice, while
ordinary dynamic sources still expose obligations without opt-in expansion
intent. Return/call/assignment sites preserve their first obligation payload,
unit-suffixed constraints preserve their unit label, and dynamic descriptors
are mapped to staged backend-guard rows with comparison opcode, threshold kind,
and trap exit code. Conjunctive dynamic proof contexts now stage one backend
guard row per checker-visible obligation and expose the second row payload as
evidence that rows can be enumerated, not just counted. A backend-shaped
`row_at(index)` surface now exposes all four bounded conjunctive row payloads
plus an out-of-bounds empty row, so the next native lowering rung can consume
row payloads by index. Those row payloads now also carry the native driver
comparison code and sys_exit_1 trap shape for supported GE, LE, and EQ guard
operators. Unit-typed constraints (e.g. `amount >= 500 <mg>` and
`glucose >= 126.0 <mg_dL>`) produce threshold_kind=3 (unit_ratio) in backend
rows, with scaled_value/value_scale preserving the numeric divisor and
float_value carrying 1.0 as the ratio comparison target. This matches the
ratio-based pre-native bash expansion that divides the field value by the
threshold and checks ratio >= 1.0. A native emission-plan for unit-ratio rows
expands into a 9-step path: load-threshold, divide-value-by-threshold,
load-one, compare-ratio-to-one, load-zero, eq-zero, branch-false-to-ok,
trap-call, ok-label; with new step op codes 8 (divide) and 9 (load-one) and
new register roles 7 (unit_threshold), 8 (ratio), and 9 (one). Plain int/float
constraints continue to use the existing 7-step load-threshold, compare,
zero-check, branch-false, trap-call, ok-label path. Statically discharged
sources produce no guard plan. The bounded direct native-v2 slice reports
backend_guard_count=1 and native_lowering_ready=1 for representative directive
sources and is consumed by native_compile_driver.sio through the row-stream
step-walk checked in knowledge_runtime_guard_direct_native_gate.sh. The native
driver now hydrates row_at(index) payloads and consumes them for integer,
integer-unit, and decimal-unit guard rows.
MSG
