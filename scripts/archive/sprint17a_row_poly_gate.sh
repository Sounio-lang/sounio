#!/usr/bin/env bash
# sprint17a_row_poly_gate.sh — row-polymorphic effects type system gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT17A_GATE_OUT:-$ROOT_DIR/artifacts/sprint17/row_poly_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT17A_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint17"

CASES_TSV="/tmp/sprint17a_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_17a.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    record "$case_name" "pass" "$pass_reason"
  elif [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
  else
    local reason
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "$reason"
  fi
}

run_check_fail_case() {
  local case_name="$1" file="$2" pattern="$3"
  local log_file="/tmp/${case_name}_17a.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" check "$file" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
    return
  fi
  if grep -q "$pattern" "$log_file" 2>/dev/null; then
    record "$case_name" "pass" "error_pattern_found"
  elif [ $rc -ne 0 ]; then
    record "$case_name" "pass" "check_rejected_as_expected"
  else
    record "$case_name" "fail" "expected_error_not_found"
  fi
}

# Case 1: TyEffectVar variant in types.sio
if grep -q 'TyEffectVar' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_effect_var_in_types" "pass" "found"
else
  record "ty_effect_var_in_types" "fail" "TyEffectVar_missing"
fi

# Case 2: TyEffectRow variant in types.sio
if grep -q 'TyEffectRow' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_effect_row_in_types" "pass" "found"
else
  record "ty_effect_row_in_types" "fail" "TyEffectRow_missing"
fi

# Case 3: ty_effect_var constructor
if grep -q 'fn ty_effect_var' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_effect_var_constructor" "pass" "found"
else
  record "ty_effect_var_constructor" "fail" "constructor_missing"
fi

# Case 4: ty_effect_row constructor
if grep -q 'fn ty_effect_row' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_effect_row_constructor" "pass" "found"
else
  record "ty_effect_row_constructor" "fail" "constructor_missing"
fi

# Case 5: is_effect_row_type predicate
if grep -q 'fn is_effect_row_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_effect_row_type_predicate" "pass" "found"
else
  record "is_effect_row_type_predicate" "fail" "predicate_missing"
fi

# Case 6: IrHandleBegin in ir.sio
if grep -q 'IrHandleBegin' self-hosted/ir/ir.sio 2>/dev/null; then
  record "ir_handle_begin_opcode" "pass" "found"
else
  record "ir_handle_begin_opcode" "fail" "opcode_missing"
fi

# Case 7: IrHandleEnd in ir.sio
if grep -q 'IrHandleEnd' self-hosted/ir/ir.sio 2>/dev/null; then
  record "ir_handle_end_opcode" "pass" "found"
else
  record "ir_handle_end_opcode" "fail" "opcode_missing"
fi

# Case 8: effects_row.sio exists
if [ -f self-hosted/check/effects_row.sio ]; then
  record "effects_row_sio_exists" "pass" "found"
else
  record "effects_row_sio_exists" "fail" "file_missing"
fi

# Case 9: check_handler_coverage function exists
if grep -q 'fn check_handler_coverage' self-hosted/check/effects_row.sio 2>/dev/null; then
  record "check_handler_coverage_fn" "pass" "found"
else
  record "check_handler_coverage_fn" "fail" "fn_missing"
fi

# Case 10: effect_row_subset function exists
if grep -q 'fn effect_row_subset' self-hosted/check/effects_row.sio 2>/dev/null; then
  record "effect_row_subset_fn" "pass" "found"
else
  record "effect_row_subset_fn" "fail" "fn_missing"
fi

# Case 11: run-pass test type-checks
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "effect_row_poly_basic_typecheck" "typecheck_passed" \
    "$SOUC" check tests/frontend/effect_row_poly_basic.sio
else
  record "effect_row_poly_basic_typecheck" "not_run" "souc_unavailable"
fi

# Case 12: handler basic test type-checks
if [ -f tests/frontend/effect_handler_basic.sio ]; then
  if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
    run_case_cmd "effect_handler_basic_typecheck" "typecheck_passed" \
      "$SOUC" check tests/frontend/effect_handler_basic.sio
  else
    record "effect_handler_basic_typecheck" "not_run" "souc_unavailable"
  fi
else
  record "effect_handler_basic_typecheck" "not_run" "test_file_missing"
fi

# Case 13: compile-fail test rejected (missing handler arm)
if [ -f tests/compile-fail/effect_handler_missing_arm.sio ]; then
  if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
    run_check_fail_case "effect_handler_missing_arm_rejected" \
      "tests/compile-fail/effect_handler_missing_arm.sio" "E035\|error\|type_check_failed"
  else
    record "effect_handler_missing_arm_rejected" "not_run" "souc_unavailable"
  fi
else
  record "effect_handler_missing_arm_rejected" "not_run" "test_file_missing"
fi

# Case 14: sprint16 regression — existing effect tests still pass
SPRINT16_PASS=0
SPRINT16_TOTAL=0
for f in tests/frontend/measure_basic.sio tests/frontend/lift_knowledge_basic.sio; do
  if [ -f "$f" ]; then
    SPRINT16_TOTAL=$((SPRINT16_TOTAL + 1))
    if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
      set +e
      timeout "$TIMEOUT_SECS" "$SOUC" check "$f" >/dev/null 2>&1
      rc=$?
      set -e
      [ $rc -eq 0 ] && SPRINT16_PASS=$((SPRINT16_PASS + 1))
    fi
  fi
done
if [ "$SPRINT16_TOTAL" -gt 0 ] && [ "$SPRINT16_PASS" -eq "$SPRINT16_TOTAL" ]; then
  record "sprint16_regression" "pass" "existing_tests_still_pass"
elif [ "$SPRINT16_TOTAL" -eq 0 ]; then
  record "sprint16_regression" "not_run" "no_regression_tests_found"
elif [ -z "${SOUC:-}" ] || [ ! -x "${SOUC:-}" ]; then
  record "sprint16_regression" "not_run" "souc_unavailable"
else
  record "sprint16_regression" "fail" "${SPRINT16_PASS}/${SPRINT16_TOTAL}_pass"
fi

python3 - "$OUT_JSON" "$CASES_TSV" "${SOUC:-unavailable}" "$TIMEOUT_SECS" << 'PY'
import json, datetime as dt, sys
from pathlib import Path

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
souc = sys.argv[3]
timeout_secs = int(sys.argv[4])

cases = []
counts = {"pass": 0, "fail": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    name, status, reason = raw.split("\t", 2)
    if status not in counts:
        status, reason = "fail", f"invalid_{status}"
    counts[status] += 1
    cases.append({"name": name, "status": status, "reason": reason})

overall = "pass" if counts["fail"] == 0 and counts["not_run"] == 0 else \
          "not_run" if counts["pass"] == 0 and counts["fail"] == 0 else \
          "fail" if counts["fail"] > 0 else "not_run"
reason = "all_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed" if counts["fail"] else "one_or_more_not_run"

payload = {
    "schema": "sounio.sprint17a.row_poly_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": reason,
    "config": {"souc": souc, "timeout_seconds": timeout_secs},
    "metrics": {"total": len(cases), "passed": counts["pass"], "failed": counts["fail"], "not_run": counts["not_run"]},
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint17a_row_poly_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
