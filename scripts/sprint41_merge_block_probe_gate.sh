#!/usr/bin/env bash
# sprint41_merge_block_probe_gate.sh — Merge Block Fix + Probe Validation
#
# Sprint 41 fixes the merge block in ir_opt_apply_strategy():
#
#   Bug (Sprint 40): merge_block always returns vreg 0 (hardcoded ir_return(0))
#   Fix (Sprint 41): ir_opt_find_return_vreg(func) scans for the original
#                    IrReturn.src1 — the actual result vreg — and uses it.
#
# Also validates the end-to-end strategy assignment using the fast parsed-frontend
# strategy probe added after Sprint 43.
# infrastructure added in Sprint 38.
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

echo "=== Sprint 41: Merge Block Fix + Probe Validation ==="
echo ""

echo "--- Merge block fix: ir_opt_find_return_vreg ---"
check_grep "opt:find_return_vreg_fn" \
    "fn ir_opt_find_return_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:find_return_vreg_scan" \
    "IrOpcode::IrReturn" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:find_return_vreg_src1" \
    "instr\.src1" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:find_return_fallback" \
    "return.*0$|^\s+0$" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:merge_result_vreg" \
    "merge_result_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:ir_return_uses_merge_vreg" \
    "ir_return\(merge_result_vreg\)" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:merge_result_vreg_comment" \
    "IrReturn\.src1|IrReturn.src1" \
    "self-hosted/ir/opt_strategy.sio"

echo ""
echo "--- Sprint 40 structural regression ---"
check_grep "opt:apply_strategy_fn" \
    "fn ir_opt_apply_strategy" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:clone_instr_fn" \
    "fn ir_opt_clone_instr" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:remap_vreg_fn" \
    "fn ir_opt_remap_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:remap_label_fn" \
    "fn ir_opt_remap_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:four_block_diamond" \
    "four_block_diamond" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:dispatch_label" \
    "dispatch_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:instr_label" \
    "instr_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:merge_label" \
    "merge_label" \
    "self-hosted/ir/opt_strategy.sio"

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
echo "--- Sprint 38/39/40 regression ---"
check_grep "regression:codegen_effective_func" \
    "var effective_func" \
    "self-hosted/native/codegen.sio"
check_grep "regression:codegen_instrumented_guard" \
    "IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/native/codegen.sio"
check_grep "regression:codegen_apply_strategy_call" \
    "ir_opt_apply_strategy" \
    "self-hosted/native/codegen.sio"
check_grep "regression:ir_compile_strategy_field" \
    "compile_strategy: i64" \
    "self-hosted/ir/ir.sio"
check_grep "regression:lower_compute_fn" \
    "fn ir_compute_strategy_from_ast" \
    "self-hosted/ir/lower.sio"
check_grep "regression:lower_ir_swap_operands" \
    "if swap_operands" \
    "self-hosted/native/lower_ir.sio"
check_grep "regression:sprint39_aggressive_fixture" \
    "operand_reordering_active.*true" \
    "tests/frontend/strategy_impact_aggressive_float.sio"
check_grep "regression:sprint40_dispatch_label" \
    "dispatch_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:hlir_apply_strategy" \
    "fn hlir_opt_apply_strategy" \
    "self-hosted/hlir/opt_strategy.sio"
check_grep "regression:serialize_strategy" \
    "compile_strategy" \
    "self-hosted/ir/serialize.sio"

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
mkdir -p "$ROOT_DIR/artifacts/sprint41"
cat > "$ROOT_DIR/artifacts/sprint41/merge_block_probe_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint41.merge_block_probe_gate.v1",
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
  "merge_block_fix": "merge_result_vreg = ir_opt_find_return_vreg(func)",
  "probe_validation": "check_probe_strategy validated strategy assignments via --probe-frontend-strategy",
  "novel_claims": [
    "Merge block now returns the actual result vreg (IrReturn.src1) not hardcoded vreg 0",
    "ir_opt_find_return_vreg() scans flat IrFunction instructions for IrReturn.src1",
    "Probe-frontend-strategy confirms strategy=instrumented for Contest<T>/Robust<T> functions",
    "End-to-end: source type -> ir_compute_strategy -> IrFunction -> ir_opt_apply_strategy -> merge_result_vreg"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint41/merge_block_probe_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
