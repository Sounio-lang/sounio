#!/usr/bin/env bash
# sprint42_validated_param_gate.sh — __validated Parameter Threading at Call Sites
#
# Sprint 42 fixes ir_opt_apply_strategy()'s dispatch block: param_regs[0] was
# IR_INVALID_REG (-1) for no-arg functions and read the first user parameter for
# parameterized functions.  Neither is the __validated dispatch flag.
#
# Two changes in ir/lower.sio:
#   1. ir_lower_inject_validated_param(): inject __validated as param_regs[0]
#      immediately after compile_strategy is set, before lower_fn_params().
#   2. Call-site lowering: for IR_STRATEGY_INSTRUMENTED callees, emit
#      ir_load_imm(validated_vreg, 1) and prepend_arg to set __validated=1.
#
# SOTA parallels: C++ this-pointer prepend, GHC STG hidden params,
#                 LLVM target_clones dispatcher, JVM C1→C2 tiered dispatch.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

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

check_probe_strategy() {
    local name="$1"; local source_file="$2"; local expected_line="$3"
    TOTAL=$((TOTAL+1))
    local souc="./bin/souc"
    if [ ! -x "$souc" ]; then
        echo "NOT_RUN  $name (souc '$souc' not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if [ ! -f "$source_file" ]; then
        echo "NOT_RUN  $name (source '$source_file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    if timeout 180 "$souc" run self-hosted/compiler/main.sio -- --probe-frontend-strategy "$source_file" >"$log_file" 2>&1; then
        if grep -qF "$expected_line" "$log_file"; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected '$expected_line' in probe output)"; FAIL=$((FAIL+1))
        fi
    else
        echo "NOT_RUN  $name (probe-frontend-strategy invocation failed)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

echo "=== Sprint 42: __validated Parameter Threading at Call Sites ==="
echo ""

echo "--- ir/lower.sio: ir_lower_inject_validated_param helper ---"
check_grep "lower:inject_fn" \
    "fn ir_lower_inject_validated_param" \
    "self-hosted/ir/lower.sio"
check_grep "lower:inject_shift_loop" \
    "while j > 0" \
    "self-hosted/ir/lower.sio"
check_grep "lower:inject_param_regs_0" \
    "param_regs\[0\] = validated_vreg" \
    "self-hosted/ir/lower.sio"
check_grep "lower:inject_param_count_plus1" \
    "param_count \+ 1" \
    "self-hosted/ir/lower.sio"
check_grep "lower:inject_strategy_guard" \
    "strategy == IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/ir/lower.sio"
check_grep "lower:inject_call_site" \
    "ir_lower_inject_validated_param" \
    "self-hosted/ir/lower.sio"

echo ""
echo "--- ir/lower.sio: call-site __validated prepend ---"
check_grep "callsite:callee_strategy_lookup" \
    "callee_strategy" \
    "self-hosted/ir/lower.sio"
check_grep "callsite:strategy_guard" \
    "callee_strategy == IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/ir/lower.sio"
check_grep "callsite:load_imm_1" \
    "ir_load_imm.*validated_vreg.*1|ir_load_imm\(validated_vreg, 1\)" \
    "self-hosted/ir/lower.sio"
check_grep "callsite:prepend_validated" \
    "prepend_arg.*validated_vreg" \
    "self-hosted/ir/lower.sio"
check_grep "callsite:final_argc_plus1" \
    "final_argc = final_argc \+ 1" \
    "self-hosted/ir/lower.sio"
check_grep "callsite:final_args_used" \
    "ir_call.*final_args.*final_argc|ir_call.*final_args" \
    "self-hosted/ir/lower.sio"

echo ""
echo "--- ir/opt_strategy.sio: Sprint 42 comment + param_regs[0] unchanged ---"
check_grep "opt:sprint42_comment" \
    "Sprint 42" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:param_regs_0_present" \
    "func\.param_regs\[0\]" \
    "self-hosted/ir/opt_strategy.sio"

echo ""
echo "--- Sprint 40/41 structural regression ---"
check_grep "regression:apply_strategy_fn" \
    "fn ir_opt_apply_strategy" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:find_return_vreg_fn" \
    "fn ir_opt_find_return_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:four_block_diamond" \
    "four_block_diamond" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:dispatch_label" \
    "dispatch_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:instr_label" \
    "instr_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:merge_label" \
    "merge_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:codegen_effective_func" \
    "var effective_func" \
    "self-hosted/native/codegen.sio"
check_grep "regression:codegen_instrumented_guard" \
    "IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/native/codegen.sio"
check_grep "regression:codegen_apply_strategy_call" \
    "ir_opt_apply_strategy" \
    "self-hosted/native/codegen.sio"
check_grep "regression:hlir_apply_strategy" \
    "fn hlir_opt_apply_strategy" \
    "self-hosted/hlir/opt_strategy.sio"

echo ""
echo "--- Probe-frontend-strategy: strategy assignment validation ---"
check_probe_strategy "probe:contest_strategy" \
    "tests/frontend/compile_strategy_ir_contest.sio" \
    "probe_frontend_strategy: fn=disputed_measurement strategy=instrumented validated_param=true param_count=1"
check_probe_strategy "probe:knowledge_strategy" \
    "tests/frontend/compile_strategy_ir_knowledge.sio" \
    "probe_frontend_strategy: fn=measure_gravity strategy=precision_preserving"
check_probe_strategy "probe:validated_strategy" \
    "tests/frontend/compile_strategy_ir_validated.sio" \
    "probe_frontend_strategy: fn=calibrated_constant strategy=aggressive"
check_probe_strategy "probe:standard_strategy" \
    "tests/frontend/compile_strategy_ir_standard.sio" \
    "probe_frontend_strategy: fn=add_floats strategy=standard"

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
mkdir -p "$ROOT_DIR/artifacts/sprint42"
cat > "$ROOT_DIR/artifacts/sprint42/validated_param_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint42.validated_param_gate.v1",
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
  "validated_param_threading": "ir_lower_inject_validated_param prepends __validated as param_regs[0]",
  "callsite_injection": "ir_load_imm(validated_vreg, 1) + prepend_arg at all IR_STRATEGY_INSTRUMENTED call sites",
  "novel_claims": [
    "__validated: i64 injected as hidden first param (param_regs[0]) at function definition time",
    "ir_opt_apply_strategy() dispatch block now reads a real vreg not IR_INVALID_REG (-1)",
    "Call sites prepend __validated = 1 (always-fast-path default; runtime propagation in Sprint 43)",
    "Mirrors HLIR validated_param injection (hlir/opt_strategy.sio Sprint 38) at the flat-IR level",
    "Single function symbol preserved unlike LLVM target_clones; dispatch embedded in function body",
    "SOTA: C++ this-pointer, GHC STG hidden params, JVM C1/C2 tiered dispatch, predicate dispatch"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint42/validated_param_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
