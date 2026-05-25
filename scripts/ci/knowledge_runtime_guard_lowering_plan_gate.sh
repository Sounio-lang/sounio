#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_knowledge_runtime_guard_lowering_plan_probe.sio"
DIRECTIVE_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_directive_positive.sio"
ORDINARY_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_call_positive.sio"
RETURN_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_positive.sio"
ASSIGN_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_assign_positive.sio"
UNIT_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_unit_positive.sio"
MULTI_DYNAMIC_SRC="tests/frontend/knowledge_runtime_guard_multi_positive.sio"
STATIC_SRC="tests/frontend/knowledge_static_value_positive.sio"
DIRECTIVE_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-directive-dynamic.log"
ORDINARY_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-ordinary-dynamic.log"
RETURN_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-return-dynamic.log"
ASSIGN_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-assign-dynamic.log"
UNIT_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-unit-dynamic.log"
MULTI_DYNAMIC_LOG="$TMP_DIR/knowledge-runtime-guard-lowering-plan-multi-dynamic.log"
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
   grep -q 'pre_native_guard_plan_ready=1' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'native_lowering_ready=0' "$DIRECTIVE_DYNAMIC_LOG" &&
   grep -q 'backend_guard_count=0' "$DIRECTIVE_DYNAMIC_LOG" &&
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
   grep -q 'staged_backend_guard_row_count=1' "$DIRECTIVE_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan is ready for directive pre-native expansion\n'
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
   grep -q 'native_lowering_ready=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'backend_guard_count=0' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_obligation_found=1' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_obligation_site_kind=2' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_obligation_type=PatientRuntimeGuardCallPositive' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_obligation_field=age' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_row_ready=1' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_compare_opcode=2' "$ORDINARY_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_trap_exit_code=1' "$ORDINARY_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan sees ordinary dynamic obligations without opt-in intent\n'
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
   grep -q 'first_backend_guard_threshold_kind=1' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'first_backend_guard_trap_exit_code=1' "$UNIT_DYNAMIC_LOG" &&
   grep -q 'staged_backend_guard_row_count=2' "$UNIT_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan maps unit-suffixed obligation payload to staged backend guard row\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not preserve unit-suffixed obligation payload\n' >&2
  cat "$UNIT_DYNAMIC_LOG" >&2
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
   grep -q 'staged_backend_guard_row_count=4' "$MULTI_DYNAMIC_LOG"; then
  printf 'PASS  compiler-side Knowledge runtime guard lowering plan stages all checker-visible conjunctive backend guard rows\n'
else
  printf 'FAIL  compiler-side Knowledge runtime guard lowering plan did not stage all conjunctive obligation rows\n' >&2
  cat "$MULTI_DYNAMIC_LOG" >&2
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
obligation counting plus first-obligation payload extraction. Directive-bearing
dynamic sources become ready for the existing pre-native guard expansion plan,
ordinary dynamic sources still expose obligations without opt-in expansion
intent, return/call/assignment sites preserve their first obligation payload,
unit-suffixed constraints preserve their unit label, and dynamic descriptors
are mapped to staged backend-guard rows with comparison opcode, threshold kind,
and trap exit code. Conjunctive dynamic proof contexts now stage one backend
guard row per checker-visible obligation. Statically discharged sources produce
no guard plan. The emitted backend_guard_count remains 0 and
native_lowering_ready=0 by design: direct backend guard/trap lowering is still
the next rung, not claimed here.
MSG
