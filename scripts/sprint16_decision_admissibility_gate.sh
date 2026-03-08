#!/usr/bin/env bash
# sprint16_decision_admissibility_gate.sh — decision admissibility gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  # shellcheck source=/dev/null
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="$SOUC_BIN"
fi

OUT_JSON="${SOUNIO_SPRINT16_GATE_OUT:-$ROOT_DIR/artifacts/sprint16/decision_admissibility_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT16_TIMEOUT_SECS:-180}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint16_cases_$$.tsv"
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
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "$reason"
  fi
}

run_compile_fail_case() {
  local case_name="$1" file="$2"
  local pat_a="$3"
  local pat_b="$4"
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
  if grep -qiF "$pat_a" "$log_file" 2>/dev/null && grep -qiF "$pat_b" "$log_file" 2>/dev/null; then
    record "$case_name" "pass" "semantic_failure_observed"
  else
    local reason
    reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "missing_expected_error: $reason"
  fi
}

run_manifest_case() {
  local case_name="$1" file="$2"
  local log_file="/tmp/${case_name}_$$.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- compile --emit-manifest "$file" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
    return
  fi
  if [ $rc -ne 0 ]; then
    local reason
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "compile_failed: $reason"
    return
  fi
  if grep -q '"decision_policies": \[' "$log_file" 2>/dev/null \
     && grep -q '"decision_certificates": \[' "$log_file" 2>/dev/null \
     && grep -q '"DefaultDecisionPolicy"' "$log_file" 2>/dev/null; then
    record "$case_name" "pass" "manifest_contains_decision_metadata"
  else
    record "$case_name" "fail" "manifest_missing_decision_metadata"
  fi
}

# Case 1: self-hosted compiler driver self-test
run_case_cmd "selfhost_compiler_main_self_test" "all_checks_passed" \
  "$SOUC" run self-hosted/compiler/main.sio -- --self-test

# Case 2: parser/checker surface is present
if grep -q "DecisionPolicyLower" self-hosted/parser/items.sio 2>/dev/null \
   && grep -q "TypeDecisionPolicy" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "TypeAdmissible" self-hosted/check/check.sio 2>/dev/null; then
  record "decision_types_and_items_present" "pass" "found"
else
  record "decision_types_and_items_present" "fail" "not_found"
fi

# Case 3: checker builtin path is present
if grep -q "call_expr_is_builtin_admit_action" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "check_admit_action_expr" self-hosted/check/check.sio 2>/dev/null; then
  record "admit_action_checker_path_present" "pass" "found"
else
  record "admit_action_checker_path_present" "fail" "not_found"
fi

# Case 4: IR/lowering path is present
if grep -q "IrAdmitAction" self-hosted/ir/ir.sio 2>/dev/null \
   && grep -q "lower_admit_action_call" self-hosted/ir/lower.sio 2>/dev/null; then
  record "admit_action_ir_lowering_present" "pass" "found"
else
  record "admit_action_ir_lowering_present" "fail" "not_found"
fi

# Case 5: backend lowering handles IrAdmitAction
if grep -q "IrAdmitAction" self-hosted/native/lower_ir.sio 2>/dev/null \
   && grep -q "IrAdmitAction" self-hosted/wasm/lower.sio 2>/dev/null; then
  record "admit_action_backends_present" "pass" "found"
else
  record "admit_action_backends_present" "fail" "not_found"
fi

# Case 6: fixture exists with explicit decision admissibility surface
if [ -f "tests/frontend/admit_action_basic.sio" ] \
   && grep -q "decision_policy DefaultDecisionPolicy for i64" tests/frontend/admit_action_basic.sio 2>/dev/null \
   && grep -q "Admissible<i64>" tests/frontend/admit_action_basic.sio 2>/dev/null \
   && grep -q "admit_action(d, r, DefaultDecisionPolicy)" tests/frontend/admit_action_basic.sio 2>/dev/null; then
  record "fixture_admit_action_basic" "pass" "full_surface_fixture_present"
else
  record "fixture_admit_action_basic" "fail" "fixture_missing_or_incomplete"
fi

# Case 7: ordinary wrapper-backed check passes on the positive fixture
run_case_cmd "ordinary_frontend_admit_action_basic_check" "ordinary_frontend_check_ok" \
  "$SOUC" check tests/frontend/admit_action_basic.sio

# Case 8: Robust<T> does not implicitly coerce to Admissible<T>
run_compile_fail_case \
  "compile_fail_robust_not_admissible" \
  "tests/compile-fail/robust_not_admissible.sio" \
  "expected Admissible<i64>" \
  "found Robust<i64, level=2, scope=0>"

# Case 9: manifest export contains decision metadata
run_manifest_case "manifest_admit_action_basic" "tests/frontend/admit_action_basic.sio"

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
    "schema": "sounio.sprint16.decision_admissibility_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": reason,
    "config": {"souc": souc, "timeout_seconds": timeout_secs},
    "metrics": {
        "total": len(cases),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "not_run": counts["not_run"],
    },
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint16_decision_admissibility_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
