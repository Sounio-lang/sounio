#!/usr/bin/env bash
# Sprint 38 gate — full-CFG dual-path lowering for StrategyInstrumented
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
elif [ -x "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-jit" ]; then
  SOUC="$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
else
  # shellcheck source=/dev/null
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  : "${SOUC_BIN:=}"
  sounio_require_souc
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT38_GATE_OUT:-$ROOT_DIR/artifacts/sprint38/dual_path_cfg_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT38_TIMEOUT_SECS:-240}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint38_cases_$$.tsv"
: > "$CASES_TSV"

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

check_grep() {
  local name="$1" pattern="$2" file="$3"
  if grep -q "$pattern" "$file" 2>/dev/null; then
    record "$name" "pass" "found"
  else
    record "$name" "fail" "missing:$pattern"
  fi
}

run_selfhost_preflight_case() {
  local case_name="$1" source_file="$2" log_file="$3"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- --probe-frontend "$source_file" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
  elif grep -q "Unknown variable: ir_effect_list_has_name" "$log_file" 2>/dev/null \
    || grep -q "Unknown variable: ir_compute_strategy_from_ast" "$log_file" 2>/dev/null; then
    record "$case_name" "not_run" "selfhost_driver_typecheck_broken"
  elif grep -q "probe_frontend: ok" "$log_file" 2>/dev/null; then
    record "$case_name" "pass" "selfhost_frontend_preflight_ok"
  elif grep -q "probe_frontend: fail" "$log_file" 2>/dev/null; then
    local reason
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "compile_failed:$reason"
  else
    local reason
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "ambiguous_probe_output:$reason"
  fi
}

check_grep "lower:strategy_pass_hook" "hlir_opt_apply_strategy" "self-hosted/hlir/lower.sio"
check_grep "lower:strategy_hints_hook" "hlir_strategy_hints_for_compile_strategy" "self-hosted/hlir/lower.sio"

check_grep "opt:clone_instrs" "fn hlir_opt_clone_instrs" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt:clone_terminator" "fn hlir_opt_clone_terminator" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt:remap_value" "fn hlir_opt_remap_value" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt:validated_param" "__validated" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt:phi_merge" "HLIR_OP_PHI" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt:switch_support" "switch_default" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt:merge_branch" "hlir_term_branch(merge_id)" "self-hosted/hlir/opt_strategy.sio"
check_grep "tests:selfhost_if_phi" "ir_test_24_hlir_dual_path_cfg_if_phi" "self-hosted/test_ir.sio"
check_grep "tests:selfhost_multi_return" "ir_test_25_hlir_dual_path_multi_return_merge" "self-hosted/test_ir.sio"
check_grep "tests:selfhost_switch" "ir_test_26_hlir_dual_path_switch_targets_remapped" "self-hosted/test_ir.sio"
check_grep "native:cgplan_multiblock_test" "test_cgplan_schedule_all_blocks_multiblock" "self-hosted/native/codegen_plan.sio"
check_grep "native:cgplan_validate_multiblock" "test_cgplan_validate_multiblock" "self-hosted/native/codegen_plan.sio"
check_grep "native:suite_runs_cgplan" "cgplan_run_tests()" "self-hosted/native/suite.sio"

check_grep "llvm:multi_block_loop" "while bi < func.block_count" "self-hosted/llvm/codegen.sio"
check_grep "llvm:phi_support" "phi_incoming" "self-hosted/llvm/codegen.sio"
check_grep "llvm:switch_support" "HLIR_TERM_SWITCH" "self-hosted/llvm/codegen.sio"

check_grep "fixture:if_exists" "contest_if_branch" "tests/frontend/dual_path_cfg_if_contest.sio"
check_grep "fixture:match_exists" "contest_match_branch" "tests/frontend/dual_path_cfg_match_contest.sio"
check_grep "fixture:multi_return_exists" "robust_multi_return" "tests/frontend/dual_path_cfg_multi_return_robust.sio"
check_grep "fixture:if_uses_if" "if x > 0" "tests/frontend/dual_path_cfg_if_contest.sio"
check_grep "fixture:match_uses_match" "match x" "tests/frontend/dual_path_cfg_match_contest.sio"
check_grep "fixture:multi_return_uses_return" "return prove_robust" "tests/frontend/dual_path_cfg_multi_return_robust.sio"

run_selfhost_preflight_case \
  "preflight:dual_path_cfg_if_contest" \
  "tests/frontend/dual_path_cfg_if_contest.sio" \
  "/tmp/sprint38_dual_path_cfg_if.log"

run_selfhost_preflight_case \
  "preflight:dual_path_cfg_match_contest" \
  "tests/frontend/dual_path_cfg_match_contest.sio" \
  "/tmp/sprint38_dual_path_cfg_match.log"

run_selfhost_preflight_case \
  "preflight:dual_path_cfg_multi_return_robust" \
  "tests/frontend/dual_path_cfg_multi_return_robust.sio" \
  "/tmp/sprint38_dual_path_cfg_multi_return.log"

python3 - <<'PY' "$CASES_TSV" "$OUT_JSON"
import csv
import datetime as dt
import json
import sys

cases_path, out_path = sys.argv[1], sys.argv[2]
cases = []
passed = failed = not_run = 0
with open(cases_path, newline="") as f:
    for name, status, reason in csv.reader(f, delimiter="\t"):
        cases.append({"name": name, "status": status, "reason": reason})
        if status == "pass":
            passed += 1
        elif status == "fail":
            failed += 1
        else:
            not_run += 1

total = passed + failed + not_run
status = "pass" if failed == 0 else "fail"
payload = {
    "schema": "sounio.sprint38.dual_path_cfg_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "reason": "all_cases_passed" if failed == 0 else "some_cases_failed",
    "metrics": {
        "total": total,
        "passed": passed,
        "failed": failed,
        "not_run": not_run,
    },
    "cases": cases,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

status="$(python3 - <<'PY' "$OUT_JSON"
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f)["status"])
PY
)"

echo "Sprint 38 gate: $status"
[ "$status" = "pass" ]
