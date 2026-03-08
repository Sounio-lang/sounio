#!/usr/bin/env bash
# sprint14_contest_witness_gate.sh — contest witness checker/frontend gate
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

OUT_JSON="${SOUNIO_SPRINT14_GATE_OUT:-$ROOT_DIR/artifacts/sprint14/contest_witness_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT14_TIMEOUT_SECS:-300}"
BOOTSTRAP_TIMEOUT_SECS="${SOUNIO_SPRINT14_BOOTSTRAP_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint14_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

bootstrap_summary_passed_since() {
  local summary_json="$1" start_epoch="$2"
  python3 - "$summary_json" "$start_epoch" <<'PY'
import json, os, sys
from pathlib import Path

summary = Path(sys.argv[1])
start_epoch = int(sys.argv[2])
if not summary.exists():
    raise SystemExit(1)
if int(summary.stat().st_mtime) < start_epoch:
    raise SystemExit(1)
data = json.loads(summary.read_text(encoding="utf-8"))
if data.get("check", {}).get("status") == "pass" and data.get("run", {}).get("status") == "pass":
    raise SystemExit(0)
raise SystemExit(1)
PY
}

extract_error_patterns() {
  local source_file="$1"
  python3 - "$source_file" <<'PY'
import re, sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
for line in source.splitlines():
    m = re.match(r"\s*//@\s*error-pattern:\s*(.*)", line)
    if m:
        print(m.group(1))
PY
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}.log"
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

run_expect_fail_case() {
  local case_name="$1" source_file="$2" log_file="$3"
  shift 3
  local patterns=("$@")
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" check "$source_file" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
    return
  fi
  if [ $rc -eq 0 ]; then
    record "$case_name" "fail" "expected_compile_failure_but_check_succeeded"
    return
  fi
  local pattern
  for pattern in "${patterns[@]}"; do
    if ! grep -qiF "$pattern" "$log_file" 2>/dev/null; then
      record "$case_name" "fail" "missing_error_pattern: $pattern"
      return
    fi
  done
  record "$case_name" "pass" "ordinary_compile_fail_check_matched"
}

run_bootstrap_case() {
  local case_name="$1"
  local log_file="/tmp/${case_name}.log"
  local summary_json="$ROOT_DIR/artifacts/omega/bootstrap_knowledge_tests.v1.json"
  local start_epoch
  start_epoch="$(date +%s)"
  rm -f "$log_file"
  set +e
  timeout "$BOOTSTRAP_TIMEOUT_SECS" bash scripts/bootstrap/run_knowledge_bootstrap_tests.sh >"$log_file" 2>&1
  local rc=$?
  set -e
  if bootstrap_summary_passed_since "$summary_json" "$start_epoch"; then
    record "$case_name" "pass" "bootstrap_suite_passed"
    return
  fi
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
  elif [ $rc -eq 0 ]; then
    record "$case_name" "fail" "bootstrap_summary_missing_or_stale"
  else
    local reason
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "$reason"
  fi
}

# Case 1: self-hosted compiler driver self-test passes
run_case_cmd "selfhost_compiler_main_self_test" "all_checks_passed" \
  "$SOUC" run self-hosted/compiler/main.sio -- --self-test

# Case 2: witness checker entrypoints present
if grep -q "check_contest_witness_expr" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "report_contest_witness_requires_contest" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "report_posterior_weights_unavailable" self-hosted/check/check.sio 2>/dev/null; then
  record "witness_checker_entrypoints_present" "pass" "found"
else
  record "witness_checker_entrypoints_present" "fail" "missing_checker_entrypoint"
fi

# Case 3: witness lowering entrypoints present
if grep -q "lower_contest_witness_call" self-hosted/ir/lower.sio 2>/dev/null \
   && grep -q "find_contest_witness_id_for_span" self-hosted/ir/lower.sio 2>/dev/null; then
  record "witness_lowering_entrypoints_present" "pass" "found"
else
  record "witness_lowering_entrypoints_present" "fail" "missing_ir_lowering_entrypoint"
fi

# Case 4: witness metadata tables present
if grep -q "struct PosteriorWeightsInfo" self-hosted/check/defs.sio 2>/dev/null \
   && grep -q "struct ContestWitnessInfo" self-hosted/check/defs.sio 2>/dev/null; then
  record "witness_metadata_tables_present" "pass" "found"
else
  record "witness_metadata_tables_present" "fail" "missing_witness_metadata_table"
fi

# Case 5: witness stdlib structs present
if [ -f "stdlib/epistemic/contest_witnesses.sio" ] \
   && grep -q "struct ContestDisagreement" stdlib/epistemic/contest_witnesses.sio 2>/dev/null \
   && grep -q "struct ContestCounterexample" stdlib/epistemic/contest_witnesses.sio 2>/dev/null \
   && grep -q "struct ContestWeights" stdlib/epistemic/contest_witnesses.sio 2>/dev/null; then
  record "witness_structs_present" "pass" "stdlib_structs_present"
else
  record "witness_structs_present" "fail" "missing_witness_structs"
fi

# Case 6: compile-fail witness and audit-attachment fixtures exist
if [ -f "tests/compile-fail/contest_witness_wrong_operand.sio" ] \
   && [ -f "tests/compile-fail/contest_witness_missing_metric.sio" ] \
   && [ -f "tests/compile-fail/contest_witness_missing_mechanistic_report.sio" ] \
   && [ -f "tests/compile-fail/contest_witness_missing_counterexample.sio" ] \
   && [ -f "tests/compile-fail/contest_witness_missing_posterior_weights.sio" ] \
   && [ -f "tests/compile-fail/contest_witness_incomplete_metadata.sio" ] \
   && [ -f "tests/compile-fail/audit_definition_shorthand_target.sio" ] \
   && [ -f "tests/compile-fail/audit_attachment_non_contest.sio" ] \
   && [ -f "tests/compile-fail/audit_target_mismatch_inner.sio" ] \
   && [ -f "tests/compile-fail/audit_target_mismatch_family.sio" ] \
   && [ -f "tests/compile-fail/audit_target_mismatch_policy.sio" ] \
   && [ -f "tests/compile-fail/audit_mechanistic_incomplete.sio" ] \
   && [ -f "tests/compile-fail/audit_counterexample_incomplete.sio" ] \
   && [ -f "tests/compile-fail/audit_posterior_weights_missing_member.sio" ] \
   && [ -f "tests/compile-fail/audit_posterior_weights_extra_member.sio" ] \
   && [ -f "tests/compile-fail/audit_posterior_weights_not_normalized.sio" ] \
   && [ -f "tests/compile-fail/audit_partial_fail_closed_witness.sio" ] \
   && [ -f "tests/compile-fail/audit_partial_fail_closed_prove_robust.sio" ]; then
  record "witness_and_audit_compile_fail_fixtures_present" "pass" "fixtures_present"
else
  record "witness_and_audit_compile_fail_fixtures_present" "fail" "missing_fixture"
fi

# Case 7: bootstrap-safe epistemic knowledge suite passes with witness tests
run_bootstrap_case "bootstrap_knowledge_witness_suite"

# Case 8-13: ordinary wrapper-backed ./souc check enforces the witness compile-fail fixtures with semantic patterns
for f in \
  contest_witness_wrong_operand \
  contest_witness_missing_metric \
  contest_witness_missing_mechanistic_report \
  contest_witness_missing_counterexample \
  contest_witness_missing_posterior_weights \
  contest_witness_incomplete_metadata; do
  tgt="tests/compile-fail/${f}.sio"
  mapfile -t patterns < <(extract_error_patterns "$tgt")
  run_expect_fail_case \
    "compile_fail_fixture_${f}" \
    "$tgt" \
    "/tmp/${f}.ordinary_check.log" \
    "${patterns[@]}"
done

# Case 14-25: ordinary wrapper-backed ./souc check enforces audit attachment compile-fail fixtures
for f in \
  audit_definition_shorthand_target \
  audit_attachment_non_contest \
  audit_target_mismatch_inner \
  audit_target_mismatch_family \
  audit_target_mismatch_policy \
  audit_mechanistic_incomplete \
  audit_counterexample_incomplete \
  audit_posterior_weights_missing_member \
  audit_posterior_weights_extra_member \
  audit_posterior_weights_not_normalized \
  audit_partial_fail_closed_witness \
  audit_partial_fail_closed_prove_robust; do
  tgt="tests/compile-fail/${f}.sio"
  mapfile -t patterns < <(extract_error_patterns "$tgt")
  run_expect_fail_case \
    "compile_fail_fixture_${f}" \
    "$tgt" \
    "/tmp/${f}.ordinary_check.log" \
    "${patterns[@]}"
done

# Case 26-28: positive audit-backed witness source fixtures
run_case_cmd "ordinary_frontend_contest_witness_basic_check" "ordinary_frontend_check_ok" \
  "$SOUC" check tests/frontend/contest_witness_basic.sio

run_case_cmd "ordinary_frontend_contest_counterexample_basic_check" "ordinary_frontend_check_ok" \
  "$SOUC" check tests/frontend/contest_counterexample_basic.sio

run_case_cmd "ordinary_frontend_contest_counterfactual_witness_basic_check" "ordinary_frontend_check_ok" \
  "$SOUC" check tests/frontend/contest_counterfactual_witness_basic.sio

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
    "schema": "sounio.sprint14.contest_witness_gate.v1",
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
echo "sprint14_contest_witness_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
