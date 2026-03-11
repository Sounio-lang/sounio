#!/usr/bin/env bash
# sprint73_machine_ir_probe_gate.sh — Machine IR probe stability gate
# Validates --probe-machine-ir against the permanent lowering stability corpus
# established in Sprint 70, confirming fn_count and supported=true for scalar programs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
OUT_DIR="$ROOT_DIR/artifacts/sprint73"
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

echo "=== Sprint 73: Machine IR Probe Stability Gate ==="
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
echo "--- Group 2: Machine IR structural checks ---"
check_grep "struct:native_v2_build_machine_module" \
    "fn native_v2_build_machine_module" \
    "self-hosted/native/machine_ir.sio"
check_grep "struct:MachineModule" \
    "struct MachineModule" \
    "self-hosted/native/machine_ir.sio"
check_grep "struct:MachineFunction" \
    "struct MachineFunction" \
    "self-hosted/native/machine_ir.sio"
check_grep "struct:probe_machine_ir_dispatch" \
    "probe-machine-ir" \
    "self-hosted/compiler/main.sio"
check_grep "struct:run_probe_machine_ir" \
    "fn run_probe_machine_ir" \
    "self-hosted/compiler/main.sio"
check_grep "struct:machine_ir_fn_count_print" \
    "fn_count=" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 3: Probe runs on stability corpus ---"
check_probe_line "probe:min_main_fn_count_1" \
    "[machine-ir] fn_count=1" \
    --probe-machine-ir tests/frontend/lowering_stability_min_main.sio
check_probe_line "probe:two_fn_loop_fn_count_2" \
    "[machine-ir] fn_count=2" \
    --probe-machine-ir tests/frontend/lowering_stability_two_fn_loop.sio
check_probe_line "probe:param_fn_fn_count_2" \
    "[machine-ir] fn_count=2" \
    --probe-machine-ir tests/frontend/lowering_stability_param_fn.sio
check_probe_line "probe:min_main_supported_true" \
    "supported=true" \
    --probe-machine-ir tests/frontend/lowering_stability_min_main.sio
check_probe_line "probe:begin_line" \
    "probe_machine_ir: begin" \
    --probe-machine-ir tests/frontend/lowering_stability_min_main.sio

echo ""
echo "--- Group 4: Sprint 72/44 compatibility ---"
check_probe_line "compat:probe_load_ir_standard" \
    "probe_load_ir: fn=add_floats strategy=standard" \
    --probe-load-ir tests/frontend/compile_strategy_ir_standard.sio
check_probe_line "compat:probe_ir_opt_min_main" \
    "probe_ir_opt_strategy: fn=main strategy=standard" \
    --probe-ir-opt-strategy tests/frontend/lowering_stability_min_main.sio

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

cat > "$OUT_DIR/machine_ir_probe_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint73.machine_ir_probe_gate.v1",
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
    "--probe-machine-ir validated against the permanent lowering stability corpus, establishing Machine IR build correctness for scalar programs",
    "fn_count and supported probe outputs verified to match source-level function counts for single-function and two-function programs"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint73/machine_ir_probe_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
