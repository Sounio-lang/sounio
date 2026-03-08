#!/usr/bin/env bash
# sprint19_resolution_gate.sh — acquisition and recourse resolution gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  SOUC="$ROOT_DIR/souc"
fi

OUT_JSON="${SOUNIO_SPRINT19_GATE_OUT:-$ROOT_DIR/artifacts/sprint19/resolution_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT19_TIMEOUT_SECS:-240}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint19_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_$$.log"
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
    reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "$reason"
  fi
}

run_compile_fail_case() {
  local case_name="$1" file="$2" pat_a="$3" pat_b="$4"
  local log_file="/tmp/${case_name}_$$.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" check "$file" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
    return
  fi
  if [ $rc -eq 0 ]; then
    record "$case_name" "fail" "expected_compile_failure_but_passed"
    return
  fi
  if grep -qiF "$pat_a" "$log_file" 2>/dev/null; then
    record "$case_name" "pass" "semantic_failure_observed"
  else
    local reason
    reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "missing_expected_error: $reason"
  fi
}

run_selfhost_probe_fail_case() {
  local case_name="$1" file="$2" pat_a="$3"
  local log_file="/tmp/${case_name}_$$.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- --probe-epistemic-check "$file" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
    return
  fi
  if grep -q "probe_epistemic_check: ok" "$log_file" 2>/dev/null; then
    record "$case_name" "fail" "expected_probe_failure_but_passed"
    return
  fi
  if grep -q "probe_epistemic_check: fail" "$log_file" 2>/dev/null && grep -qiF "$pat_a" "$log_file" 2>/dev/null; then
    record "$case_name" "pass" "semantic_failure_observed"
  else
    local reason
    reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "missing_expected_error: $reason"
  fi
}

run_case_cmd "selfhost_compiler_main_self_test" "all_checks_passed" \
  "$SOUC" run self-hosted/compiler/main.sio -- --self-test

if grep -q "ItemAcquisitionPolicy" self-hosted/parser/ast.sio 2>/dev/null \
   && grep -q "ItemRecoursePolicy" self-hosted/parser/ast.sio 2>/dev/null \
   && grep -q "TypeAcquisitionPolicy" self-hosted/parser/types.sio 2>/dev/null \
   && grep -q "TypeRecoursePolicy" self-hosted/parser/types.sio 2>/dev/null; then
  record "resolution_items_and_types_present" "pass" "found"
else
  record "resolution_items_and_types_present" "fail" "not_found"
fi

if grep -q "plan_acquisition" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "plan_recourse" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "acquisition_reason" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "recourse_reason" self-hosted/check/check.sio 2>/dev/null; then
  record "resolution_checker_paths_present" "pass" "found"
else
  record "resolution_checker_paths_present" "fail" "not_found"
fi

run_case_cmd "ordinary_frontend_plan_acquisition_basic_check" "ordinary_check_ok" \
  "$SOUC" check tests/frontend/plan_acquisition_basic.sio
run_case_cmd "selfhost_probe_plan_recourse_counterfactual_basic" "selfhost_probe_ok" \
  "$SOUC" run self-hosted/compiler/main.sio -- --probe-epistemic-check tests/frontend/plan_recourse_counterfactual_basic.sio

run_compile_fail_case \
  "compile_fail_plan_acquisition_requires_policy" \
  "tests/compile-fail/plan_acquisition_requires_policy.sio" \
  "plan_acquisition(...) requires a declared acquisition_policy item" \
  "acquisition remains explicit and auditable"

run_compile_fail_case \
  "compile_fail_plan_recourse_requires_policy" \
  "tests/compile-fail/plan_recourse_requires_policy.sio" \
  "plan_recourse(...) requires a declared recourse_policy item" \
  "recourse remains explicit and auditable"

run_selfhost_probe_fail_case \
  "probe_fail_plan_recourse_requires_deferred" \
  "tests/compile-fail/plan_recourse_requires_deferred.sio" \
  "plan_recourse(...) requires Deferred<T>"

run_selfhost_probe_fail_case \
  "probe_fail_acquisition_reason_requires_plan" \
  "tests/compile-fail/acquisition_reason_requires_plan.sio" \
  "acquisition_reason(...) requires AcquisitionPlan<T>"

python3 - "$OUT_JSON" "$CASES_TSV" "$SOUC" "$TIMEOUT_SECS" << 'PY'
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
          "not_run" if counts["pass"] == 0 and counts["fail"] == 0 else "fail"
reason = "all_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed" if counts["fail"] else "one_or_more_not_run"

payload = {
    "schema": "sounio.sprint19.resolution_gate.v1",
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
echo "sprint19_resolution_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
