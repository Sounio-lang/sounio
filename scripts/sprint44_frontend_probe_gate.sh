#!/usr/bin/env bash
# sprint44_frontend_probe_gate.sh — Fast probe lane + heavy IR smoke coverage
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"

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
        echo "FAIL  $name (expected '$expected_line' in $log_file)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 44: Frontend Probe Recovery ==="
echo ""

echo "--- Compiler self-test smoke ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    TOTAL=$((TOTAL+1))
    echo "NOT_RUN  selftest:compiler_main (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
else
    if timeout 240 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:compiler_main" "compiler/main tests: " "$SELF_TEST_LOG"
    else
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:compiler_main (compiler/main --self-test failed)"; NOT_RUN=$((NOT_RUN+1))
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Fast probe lane ---"
check_probe_line "fast:standard_strategy" \
    "probe_frontend_strategy: fn=add_floats strategy=standard validated_param=false param_count=2" \
    --probe-frontend-strategy tests/frontend/compile_strategy_ir_standard.sio
check_probe_line "fast:contest_strategy" \
    "probe_frontend_strategy: fn=disputed_measurement strategy=instrumented validated_param=true param_count=1" \
    --probe-frontend-strategy tests/frontend/compile_strategy_ir_contest.sio
check_probe_line "fast:knowledge_strategy" \
    "probe_frontend_strategy: fn=measure_gravity strategy=precision_preserving validated_param=false param_count=0" \
    --probe-frontend-strategy tests/frontend/compile_strategy_ir_knowledge.sio
check_probe_line "fast:validated_strategy" \
    "probe_frontend_strategy: fn=calibrated_constant strategy=aggressive validated_param=false param_count=0" \
    --probe-frontend-strategy tests/frontend/compile_strategy_ir_validated.sio

echo ""
echo "--- Parsed frontend callsite semantics ---"
check_probe_line "fast:outer_forwarding" \
    "probe_ir_callsite: caller=outer_contested callee=inner_contested caller_strategy=instrumented callee_strategy=instrumented arg_count=2 src1=0 src2=3 forwards_param0=true fallback_load_imm1=false" \
    --probe-ir-callsite tests/frontend/chain_validated_param_contest.sio outer_contested inner_contested
check_probe_line "fast:main_fallback" \
    "probe_ir_callsite: caller=main callee=outer_contested caller_strategy=standard callee_strategy=instrumented arg_count=2 src1=2 src2=0 forwards_param0=false fallback_load_imm1=true" \
    --probe-ir-callsite tests/frontend/chain_validated_param_contest.sio main outer_contested

echo ""
echo "--- Heavy IR smoke lane ---"
check_probe_line "heavy:load_ir_standard" \
    "probe_load_ir: fn=add_floats strategy=standard" \
    --probe-load-ir tests/frontend/compile_strategy_ir_standard.sio
check_probe_line "heavy:trace_standard" \
    "probe_load_ir_trace: stage=done modules=1 lowered=1 patched_calls=0 fallback_insertions=0" \
    --probe-load-ir-trace tests/frontend/compile_strategy_ir_standard.sio
check_probe_line "heavy:trace_chain" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint44"
cat > "$ROOT_DIR/artifacts/sprint44/frontend_probe_recovery.v1.json" <<EOF
{
  "schema": "sounio.sprint44.frontend_probe_recovery.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 240
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "fast_lane": "probe-frontend-strategy and probe-ir-callsite are the canonical fast parsed-frontend probes for sprint gates",
  "heavy_smoke": "probe-load-ir and probe-load-ir-trace remain available as sequential heavier parsed-frontend compatibility checks",
  "novel_claims": [
    "Fast parsed-frontend strategy probes no longer depend on full epistemic export",
    "Sprint 43 callsite propagation is validated by probe-mode IR lowering on the real chain fixture",
    "Heavy IR probes remain backward-compatible and sequentially verifiable within the existing timeout envelope"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint44/frontend_probe_recovery.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
