#!/usr/bin/env bash
# sprint17_deferral_gate.sh — typed deferral and escalation certificate gate
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

if [ -x "$ROOT_DIR/target/debug/souc" ]; then
  PROBE_SOUC="$ROOT_DIR/target/debug/souc"
elif [ -x "$ROOT_DIR/target/release/souc" ]; then
  PROBE_SOUC="$ROOT_DIR/target/release/souc"
else
  PROBE_SOUC="$SOUC"
fi

OUT_JSON="${SOUNIO_SPRINT17_GATE_OUT:-$ROOT_DIR/artifacts/sprint17/deferral_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT17_TIMEOUT_SECS:-180}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint17_cases_$$.tsv"
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

run_probe_case() {
  local case_name="$1" source_file="$2"
  local log_file="/tmp/${case_name}_$$.log"
  rm -f "$log_file"
  set +e
  "$PROBE_SOUC" run self-hosted/compiler/main.sio -- --probe-frontend "$source_file" >"$log_file" 2>&1 &
  local pid=$!
  local deadline=$((SECONDS + TIMEOUT_SECS))
  while kill -0 "$pid" >/dev/null 2>&1; do
    if grep -q "probe_frontend: ok" "$log_file" 2>/dev/null; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      set -e
      record "$case_name" "pass" "selfhost_frontend_probe_ok"
      return
    fi
    if grep -q "error\\[" "$log_file" 2>/dev/null || grep -q "probe_frontend: fail" "$log_file" 2>/dev/null; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      set -e
      local reason
      reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 160 || echo error)"
      record "$case_name" "fail" "$reason"
      return
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      set -e
      record "$case_name" "not_run" "timeout"
      return
    fi
    sleep 1
  done
  local rc=$?
  set -e
  if grep -q "probe_frontend: ok" "$log_file" 2>/dev/null; then
    record "$case_name" "pass" "selfhost_frontend_probe_ok"
  else
    local reason
    reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "${reason:-probe_frontend_missing_ok}"
  fi
}

run_compile_fail_case() {
  local case_name="$1" file="$2" pat_a="$3" pat_b="$4"
  local log_file="/tmp/${case_name}_$$.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$PROBE_SOUC" run self-hosted/compiler/main.sio -- --probe-frontend "$file" >"$log_file" 2>&1
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

run_case_cmd "selfhost_compiler_main_self_test" "all_checks_passed" \
  "$SOUC" run self-hosted/compiler/main.sio -- --self-test

if grep -q "ItemDeferralPolicy" self-hosted/parser/ast.sio 2>/dev/null \
   && grep -q "TypeDeferralPolicy" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "TypeDeferred" self-hosted/check/check.sio 2>/dev/null; then
  record "deferral_types_and_items_present" "pass" "found"
else
  record "deferral_types_and_items_present" "fail" "not_found"
fi

if grep -q "call_expr_is_builtin_defer_action" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "check_defer_action_expr" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "check_deferral_reason_expr" self-hosted/check/check.sio 2>/dev/null; then
  record "defer_action_checker_path_present" "pass" "found"
else
  record "defer_action_checker_path_present" "fail" "not_found"
fi

if grep -q "IrDeferAction" self-hosted/ir/ir.sio 2>/dev/null \
   && grep -q "lower_defer_action_call" self-hosted/ir/lower.sio 2>/dev/null \
   && grep -q "lower_deferral_reason_call" self-hosted/ir/lower.sio 2>/dev/null; then
  record "deferral_ir_lowering_present" "pass" "found"
else
  record "deferral_ir_lowering_present" "fail" "not_found"
fi

if grep -q "IrDeferAction" self-hosted/native/lower_ir.sio 2>/dev/null \
   && grep -q "IrDeferAction" self-hosted/wasm/lower.sio 2>/dev/null; then
  record "deferral_backends_present" "pass" "found"
else
  record "deferral_backends_present" "fail" "not_found"
fi

run_probe_case "selfhost_frontend_defer_action_contest_basic_probe" "tests/frontend/defer_action_contest_basic.sio"

run_compile_fail_case \
  "compile_fail_robust_not_deferred" \
  "tests/compile-fail/robust_not_deferred.sio" \
  "expected Deferred<i64>" \
  "found Robust<i64, level=2, scope=0>"

run_compile_fail_case \
  "compile_fail_deferral_reason_requires_deferred" \
  "tests/compile-fail/deferral_reason_requires_deferred.sio" \
  "deferral_reason(...) requires Deferred<T>" \
  "call deferral_reason(...) on the result of defer_action(...)"

run_compile_fail_case \
  "compile_fail_defer_action_requires_full_evidence" \
  "tests/compile-fail/defer_action_requires_full_evidence.sio" \
  "error[E099]" \
  "requires Contest<T, Family, Policy> or proof-carrying Robust<T> evidence" \
  "requires Contest<T, Family, Policy> or proof-carrying Robust<T> evidence"

run_compile_fail_case \
  "compile_fail_defer_action_requires_deferral_policy" \
  "tests/compile-fail/defer_action_requires_deferral_policy.sio" \
  "requires a declared deferral_policy item" \
  "pass that named policy as the third argument"

run_compile_fail_case \
  "compile_fail_defer_action_target_mismatch" \
  "tests/compile-fail/defer_action_target_mismatch.sio" \
  "deferral policy target does not match the deferred action type" \
  "expected i64"

run_compile_fail_case \
  "compile_fail_defer_action_requires_proof_carrying_robust" \
  "tests/compile-fail/defer_action_requires_proof_carrying_robust.sio" \
  "error[E099]" \
  "requires Contest<T, Family, Policy> or proof-carrying Robust<T> evidence" \
  "requires Contest<T, Family, Policy> or proof-carrying Robust<T> evidence"

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
    "schema": "sounio.sprint17.deferral_gate.v1",
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
echo "sprint17_deferral_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
