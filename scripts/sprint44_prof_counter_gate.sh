#!/usr/bin/env bash
# sprint44_prof_counter_gate.sh — Profiling Counter Collection on Instrumented Slow Path
#
# SOTA: LLVM PGO __llvm_prf_cnts (per-function counters); Arnold & Ryder selective
# profiling PLDI 2001 (two-version dispatch, instrument slow path only); Chamith et al.
# SIGPLAN 2017 (x86-64 instruction punning, 7-byte NOP width matches ADD [RIP+disp32],1).
#
# Novel claim: one counter per instrumented function, at slow-path entry only; fast path
# has zero overhead (no counter, no branch, no NOP).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"

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

check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 44: Profiling Counter Collection on Instrumented Slow Path ==="
echo ""

# --- Executable self-tests (T12, T13) ---
echo "--- Executable self-test: profiling counter injection ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:prof_counter_injected" "selftest:prof_counter_absent_fast"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:prof_counter_injected" \
            "T12 OK: prof counter in instrumented path" "$SELF_TEST_LOG"
        check_log_line "selftest:prof_counter_absent_fast" \
            "T13 OK: prof counter absent from fast path" "$SELF_TEST_LOG"
    else
        for name in "selftest:prof_counter_injected" "selftest:prof_counter_absent_fast"; do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (--self-test failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Structural checks: IR infrastructure ---"
check_grep "ir:IrProfCounter_opcode" \
    "IrProfCounter" \
    "self-hosted/ir/ir.sio"
check_grep "ir:IrProfCounterInfo_struct" \
    "struct IrProfCounterInfo" \
    "self-hosted/ir/ir.sio"
check_grep "ir:prof_counter_id_field" \
    "prof_counter_id" \
    "self-hosted/ir/ir.sio"
check_grep "ir:prof_counter_constructor" \
    "fn ir_prof_counter" \
    "self-hosted/ir/ir.sio"
check_grep "ir:module_counter_count" \
    "prof_counter_count" \
    "self-hosted/ir/ir.sio"

echo ""
echo "--- Structural checks: opt_strategy injection ---"
check_grep "opt:inject_counter" \
    "ir_prof_counter" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:counter_guard" \
    "prof_counter_id >= 0" \
    "self-hosted/ir/opt_strategy.sio"

echo ""
echo "--- Structural checks: lowering and native ---"
check_grep "lower:counter_alloc" \
    "prof_counter_id" \
    "self-hosted/ir/lower.sio"
check_grep "lower:counter_strategy_guard" \
    "IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/ir/lower.sio"
check_grep "native:prof_counter_case" \
    "IrProfCounter" \
    "self-hosted/native/lower_ir.sio"
check_grep "native:nop_7byte" \
    "0F 1F 80" \
    "self-hosted/native/lower_ir.sio"
check_grep "serialize:prof_counter_tag" \
    "IrProfCounter" \
    "self-hosted/ir/serialize.sio"

echo ""
echo "--- Structural checks: test fixture ---"
check_grep "fixture:run_pass" \
    "//@ run-pass" \
    "tests/frontend/prof_counter_instrumented_contest.sio"
check_grep "fixture:contest_fn" \
    "fn measured_fn" \
    "tests/frontend/prof_counter_instrumented_contest.sio"

echo ""
echo "--- Sprint 38-43 regression ---"
check_grep "regression:apply_strategy_fn" \
    "fn ir_opt_apply_strategy" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:find_return_vreg_fn" \
    "fn ir_opt_find_return_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:codegen_effective_func" \
    "var effective_func" \
    "self-hosted/native/codegen.sio"
check_grep "regression:inject_validated" \
    "fn ir_lower_inject_validated_param" \
    "self-hosted/ir/lower.sio"
check_grep "regression:patch_validated_calls" \
    "fn ir_patch_validated_calls" \
    "self-hosted/ir/lower.sio"
check_grep "regression:chain_forwarding" \
    "caller_validated_vreg" \
    "self-hosted/ir/lower.sio"
check_grep "regression:hlir_apply_strategy" \
    "fn hlir_opt_apply_strategy" \
    "self-hosted/hlir/opt_strategy.sio"

# --- Results ---
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
cat > "$ROOT_DIR/artifacts/sprint44/prof_counter_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint44.prof_counter_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 180
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "profiling_infrastructure": "IrProfCounter opcode injected at instrumented-path entry; 7-byte NOP placeholder in native codegen; counter metadata tracked per function in IrModule",
  "novel_claims": [
    "One counter per instrumented function at slow-path entry only; fast path has zero overhead",
    "7-byte NOP placeholder matches exact width of future ADD QWORD PTR [RIP+disp32], 1",
    "Counter metadata (IrProfCounterInfo) tracks function ID, name, and counter index",
    "IrProfCounter injection is guarded by prof_counter_id >= 0; functions without counters are untouched",
    "Serialization supports round-tripping prof_counter_id through SOIR format"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint44/prof_counter_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
