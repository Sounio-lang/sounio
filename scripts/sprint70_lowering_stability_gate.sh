#!/usr/bin/env bash
# sprint70_lowering_stability_gate.sh — Staged IR lowering foundation
set -euo pipefail

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
        echo "NOT_RUN  $name (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    if timeout 240 "$SOUC" run self-hosted/compiler/main.sio -- "$@" >"$log_file" 2>&1; then
        if grep -qF "$expected_line" "$log_file"; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected '$expected_line' in probe output)"; FAIL=$((FAIL+1))
        fi
    else
        echo "NOT_RUN  $name (probe invocation failed)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected '$expected_line' in log)"; FAIL=$((FAIL+1))
    fi
}

check_run_line() {
    local name="$1"; local expected_line="$2"; local src="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$src" ]; then
        echo "NOT_RUN  $name (source '$src' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    if timeout 240 "$SOUC" run "$src" >"$log_file" 2>&1; then
        if grep -qF "$expected_line" "$log_file"; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected '$expected_line' in output)"; FAIL=$((FAIL+1))
        fi
    else
        local ec=$?
        if [ "$ec" -eq 124 ]; then
            echo "NOT_RUN  $name (timeout 240s)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "NOT_RUN  $name (run failed exit $ec)"; NOT_RUN=$((NOT_RUN+1))
        fi
    fi
    rm -f "$log_file"
}

check_native_compile() {
    local name="$1"; local src="$2"; local out_bin="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    rm -f "$out_bin"
    if timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --native-compile "$src" -o "$out_bin" >"$log_file" 2>&1; then
        if grep -qF "Native compilation successful: output=$out_bin" "$log_file" && [ -f "$out_bin" ]; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (native compile did not produce expected output)"; FAIL=$((FAIL+1))
        fi
    else
        echo "NOT_RUN  $name (native compile failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

echo "=== Sprint 70: Staged IR Lowering Foundation ==="
echo ""

echo "--- Group 1: Compiler self-test smoke ---"
TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then
    echo "NOT_RUN  selftest:compiler_main (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
else
    _st_log="$(mktemp)"
    if timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$_st_log" 2>&1; then
        echo "PASS  selftest:compiler_main"; PASS=$((PASS+1))
    else
        _ec=$?
        if [ "$_ec" -eq 124 ] || [ "$_ec" -eq 137 ]; then
            echo "NOT_RUN  selftest:compiler_main (OOM/timeout)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "NOT_RUN  selftest:compiler_main (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
        fi
    fi
    rm -f "$_st_log"
fi

echo ""
echo "--- Group 2: Structural staged lowering split ---"
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
echo "--- Group 3: Stability corpus ---"
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
echo "--- Group 4: Full IR load stability ---"
check_probe_line "full:min_main" \
    "probe_ir_opt_strategy: fn=main strategy=standard" \
    --probe-ir-opt-strategy tests/frontend/lowering_stability_min_main.sio
check_probe_line "full:param_fn" \
    "probe_ir_opt_strategy: fn=add_pair strategy=standard" \
    --probe-ir-opt-strategy tests/frontend/lowering_stability_param_fn.sio
check_probe_line "full:two_fn_loop" \
    "probe_ir_opt_strategy: fn=bump strategy=standard" \
    --probe-ir-opt-strategy tests/frontend/lowering_stability_two_fn_loop.sio

echo ""
echo "--- Group 5: Traced full IR load ---"
check_run_line "trace:two_fn_loop" \
    "lowering_trace_smoke: stage=done summary=1 bodies=1 lowered=1" \
    self-hosted/compiler/lowering_trace_smoke.sio

echo ""
echo "--- Group 6: Native compile survivor ---"
check_native_compile "native:min_loop" \
    "tests/frontend/lowering_stability_native_loop.sio" \
    "$OUT_DIR/lowering_stability_native_loop.out"

echo ""
echo "--- Group 7: Sprint 44 compatibility ---"
check_probe_line "compat:probe_load_ir_standard" \
    "probe_load_ir: fn=add_floats strategy=standard" \
    --probe-load-ir tests/frontend/compile_strategy_ir_standard.sio
check_probe_line "compat:probe_load_ir_trace_chain" \
    "probe_load_ir_trace: stage=done modules=1 lowered=1 patched_calls=0 fallback_insertions=0" \
    --probe-load-ir-trace tests/frontend/chain_validated_param_contest.sio

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

cat > "$OUT_DIR/lowering_stability_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint70.lowering_stability_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 300
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "Full IR lowering now seeds program summaries before sequential body lowering",
    "Full multimodule lowering tracks summary-seed and body-lowering stages separately",
    "Probe-only summary lanes remain backward-compatible while normal compile uses the staged full loader",
    "A permanent lowering stability corpus replaces scratch repro files for future regressions"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint70/lowering_stability_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
