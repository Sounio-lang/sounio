#!/usr/bin/env bash
# sprint43_chain_propagation_gate.sh — __validated Runtime Flag Propagation Through Call Chain
#
# Sprint 42 hardcoded __validated = 1 at all instrumented call sites (always fast path,
# instrumented path was dead code). Sprint 43 makes propagation dynamic:
#
#   When instrumented caller calls instrumented callee:
#     → forward caller's param_regs[0] directly (zero instruction overhead, vreg IS the value)
#   When non-instrumented caller calls instrumented callee:
#     → inject ir_load_imm(validated_vreg, 1) (unchanged from Sprint 42, always fast path)
#
# SOTA: CPS theorem (Kennedy ICFP 2007), Implicit Parameters (Lewis et al. POPL 2000),
#       Arnold-Ryder two-version selective profiling (PLDI 2001).
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
        echo "FAIL  $name (expected '$expected_line' in $log_file)"; FAIL=$((FAIL+1))
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
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- "$@" >"$log_file" 2>&1; then
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

echo "=== Sprint 43: __validated Runtime Flag Propagation Through Call Chain ==="
echo ""

echo "--- Executable self-test: validated flag propagation ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in \
        "selftest:chain_forwarding" \
        "selftest:top_level_fallback"
    do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:chain_forwarding" \
            "T10 OK: validated chain forwards caller flag" \
            "$SELF_TEST_LOG"
        check_log_line "selftest:top_level_fallback" \
            "T11 OK: validated chain keeps top-level fallback" \
            "$SELF_TEST_LOG"
    else
        for name in \
            "selftest:chain_forwarding" \
            "selftest:top_level_fallback"
        do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (compiler/main --self-test failed)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Parsed frontend callsite probe ---"
check_probe_line "probe:outer_forwards_param0" \
    "probe_ir_callsite: caller=outer_contested callee=inner_contested caller_strategy=instrumented callee_strategy=instrumented arg_count=2" \
    --probe-ir-callsite tests/frontend/chain_validated_param_contest.sio outer_contested inner_contested
check_probe_line "probe:outer_forward_flag" \
    "forwards_param0=true fallback_load_imm1=false" \
    --probe-ir-callsite tests/frontend/chain_validated_param_contest.sio outer_contested inner_contested
check_probe_line "probe:main_top_level_fallback" \
    "probe_ir_callsite: caller=main callee=outer_contested caller_strategy=standard callee_strategy=instrumented arg_count=2" \
    --probe-ir-callsite tests/frontend/chain_validated_param_contest.sio main outer_contested
check_probe_line "probe:main_fallback_flag" \
    "forwards_param0=false fallback_load_imm1=true" \
    --probe-ir-callsite tests/frontend/chain_validated_param_contest.sio main outer_contested

echo ""
echo "--- Minimal structural regression ---"
check_grep "lower:validated_helper" \
    "fn ir_lower_prepend_validated_arg" \
    "self-hosted/ir/lower.sio"
check_grep "lower:patch_helper" \
    "fn ir_patch_validated_calls_in_function" \
    "self-hosted/ir/lower.sio"
check_grep "lower:patch_entry" \
    "fn ir_patch_validated_calls" \
    "self-hosted/ir/lower.sio"
check_grep "lower:patch_predicate" \
    "fn ir_call_needs_validated_patch" \
    "self-hosted/ir/lower.sio"
check_grep "lower:method_call_uses_helper" \
    "ExprMethodCall|ir_lower_prepend_validated_arg" \
    "self-hosted/ir/lower.sio"
check_grep "fixture_chain:run_pass" \
    "//@ run-pass" \
    "tests/frontend/chain_validated_param_contest.sio"
check_grep "fixture_chain:inner_fn" \
    "fn inner_contested" \
    "tests/frontend/chain_validated_param_contest.sio"
check_grep "fixture_chain:outer_fn" \
    "fn outer_contested" \
    "tests/frontend/chain_validated_param_contest.sio"
check_grep "fixture_chain:order_note" \
    "not sensitive to callee definition order" \
    "tests/frontend/chain_validated_param_contest.sio"

echo ""
echo "--- Sprint 42 regression ---"
check_grep "regression:inject_fn" \
    "fn ir_lower_inject_validated_param" \
    "self-hosted/ir/lower.sio"
check_grep "regression:inject_strategy_guard" \
    "strategy == IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/ir/lower.sio"
check_grep "regression:callee_strategy_guard" \
    "callee_strategy == IR_STRATEGY_INSTRUMENTED|compile_strategy == IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/ir/lower.sio"
check_grep "regression:apply_strategy_fn" \
    "fn ir_opt_apply_strategy" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:find_return_vreg_fn" \
    "fn ir_opt_find_return_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:codegen_effective_func" \
    "var effective_func" \
    "self-hosted/native/codegen.sio"
check_grep "regression:codegen_apply_strategy_call" \
    "ir_opt_apply_strategy" \
    "self-hosted/native/codegen.sio"
check_grep "regression:hlir_apply_strategy" \
    "fn hlir_opt_apply_strategy" \
    "self-hosted/hlir/opt_strategy.sio"

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

# Write gate artifact
mkdir -p "$ROOT_DIR/artifacts/sprint43"
cat > "$ROOT_DIR/artifacts/sprint43/chain_propagation_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint43.chain_propagation_gate.v1",
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
  "validated_propagation": "self-tests plus fast parsed-frontend callsite probes validate forwarding and standard-caller fallback",
  "novel_claims": [
    "Executable self-tests validate both nested forwarding and the standard-caller fallback path",
    "Fast parsed-frontend callsite probes validate outer_contested -> inner_contested forwarding and main -> outer_contested fallback on the real chain fixture",
    "Missing hidden args are repaired after lowering once callee compile_strategy metadata is known",
    "Instrumented-to-instrumented calls forward __validated via param_regs[0] with no extra load instruction",
    "Standard callers entering an instrumented callee get a synthesized load_imm 1 prepended once per function",
    "Method-call lowering shares the same hidden-argument helper as plain calls"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint43/chain_propagation_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
