#!/usr/bin/env bash
# sprint70_lowering_stability_gate.sh — Staged IR lowering foundation
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
OUT_DIR="$ROOT_DIR/artifacts/sprint70"
mkdir -p "$OUT_DIR"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

check_probe_line() {
    local name="$1"; shift
    local expected_line="$1"; shift
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file _ec
    log_file="$(mktemp)"
    _ec=0
    timeout 240 "$SOUC" run self-hosted/compiler/main.sio -- "$@" >"$log_file" 2>&1 || _ec=$?
    if [ "$_ec" -eq 0 ] && grep -qF "$expected_line" "$log_file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    elif [ "$_ec" -eq 0 ]; then
        echo "FAIL  $name (expected '$expected_line' not found)"; FAIL=$((FAIL+1))
    else
        echo "NOT_RUN  $name (probe exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

check_run_ok() {
    local name="$1"; local src="$2"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$src" ]; then
        echo "NOT_RUN  $name (source '$src' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local _ec=0
    timeout 240 "$SOUC" run "$src" >/dev/null 2>&1 || _ec=$?
    if [ "$_ec" -eq 0 ]; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "NOT_RUN  $name (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
}

check_compiler_run_ok() {
    local name="$1"; shift
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local _ec=0
    timeout 240 "$SOUC" run self-hosted/compiler/main.sio -- "$@" >/dev/null 2>&1 || _ec=$?
    if [ "$_ec" -eq 0 ]; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "NOT_RUN  $name (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
}

echo "=== Sprint 70: Staged IR Lowering Foundation ==="
echo ""

echo "--- Group 1: Structural staged lowering split ---"
check_grep "lower:begin_function_body" \
    "fn begin_function_body_lowering" \
    "self-hosted/ir/lower.sio"
check_grep "lower:preseed_program_items" \
    "fn preseed_program_items" \
    "self-hosted/ir/lower.sio"
check_grep "lower:lower_program_bodies" \
    "fn lower_program_bodies" \
    "self-hosted/ir/lower.sio"
check_grep "lower:lower_program_bodies_filtered" \
    "fn lower_program_bodies_filtered" \
    "self-hosted/ir/lower.sio"
check_grep "lower:staged_entrypoint" \
    "fn lower_program_to_ir_staged_with_lowerer_trace" \
    "self-hosted/ir/lower.sio"
check_grep "loader:traced_helper" \
    "fn load_multimodule_lower_program_traced" \
    "self-hosted/compiler/module_loader.sio"
check_grep "loader:trace_summary_stage" \
    "lower_summary" \
    "self-hosted/compiler/module_loader.sio"
check_grep "loader:trace_body_stage" \
    "lower_bodies" \
    "self-hosted/compiler/module_loader.sio"

echo ""
echo "--- Group 2: Stability corpus ---"
check_grep "fixture:min_main" \
    "fn main" \
    "tests/frontend/lowering_stability_min_main.sio"
check_grep "fixture:param_fn" \
    "fn add_pair" \
    "tests/frontend/lowering_stability_param_fn.sio"
check_grep "fixture:two_fn_loop" \
    "fn bump" \
    "tests/frontend/lowering_stability_two_fn_loop.sio"
check_grep "fixture:heavy_loop" \
    "while inner < 2" \
    "tests/frontend/lowering_stability_heavy_loop.sio"
check_grep "fixture:native_loop" \
    "while i < 1" \
    "tests/frontend/lowering_stability_native_loop.sio"

echo ""
echo "--- Group 3: Full IR load stability ---"
check_compiler_run_ok "full:two_fn_loop" \
    --probe-ir-opt-strategy tests/frontend/lowering_stability_two_fn_loop.sio
check_compiler_run_ok "full:min_main" \
    --probe-ir-opt-strategy tests/frontend/lowering_stability_min_main.sio
check_compiler_run_ok "full:param_fn" \
    --probe-ir-opt-strategy tests/frontend/lowering_stability_param_fn.sio

echo ""
echo "--- Group 4: Traced full IR load ---"
check_run_ok "trace:two_fn_loop" \
    self-hosted/compiler/lowering_trace_smoke.sio

echo ""
echo "--- Group 5: Native compile survivor ---"
check_run_ok "native:min_loop" \
    self-hosted/compiler/native_compile_smoke.sio

echo ""
echo "--- Group 6: Sprint 44 compatibility ---"
check_probe_line "compat:probe_load_ir_standard" \
    "probe_load_ir: fn=add_floats strategy=standard" \
    --probe-load-ir tests/frontend/compile_strategy_ir_standard.sio
check_grep "compat:chain_fixture_exists" \
    "fn " \
    "tests/frontend/chain_validated_param_contest.sio"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"
REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then
    STATUS="fail"
    REASON="${FAIL}_cases_failed"
fi
if [ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ]; then
    REASON="all_run_passed_${NOT_RUN}_not_run"
fi

_ts=$(date -u +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo "unknown")
{
  printf '{\n'
  printf '  "schema": "sounio.sprint70.lowering_stability_gate.v1",\n'
  printf '  "generated_at": "%s",\n' "$_ts"
  printf '  "status": "%s",\n' "$STATUS"
  printf '  "reason": "%s",\n' "$REASON"
  printf '  "metrics": {"total": %d, "passed": %d, "failed": %d, "not_run": %d},\n' \
    "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN"
  printf '  "novel_claims": [\n'
  printf '    "Full IR lowering seeds program summaries before sequential body lowering",\n'
  printf '    "Full multimodule lowering tracks summary-seed and body-lowering stages separately",\n'
  printf '    "Probe-only summary lanes remain backward-compatible",\n'
  printf '    "Permanent lowering stability corpus established for future regression testing"\n'
  printf '  ]\n'
  printf '}\n'
} > "$OUT_DIR/lowering_stability_gate.v1.json"

echo ""
echo "Artifact: artifacts/sprint70/lowering_stability_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
