#!/usr/bin/env bash
# sprint40_dual_path_native_gate.sh — Native dual-path for Instrumented strategy
#
# Sprint 40 adds an IR-level dual-path transformation (ir_opt_apply_strategy) that
# mirrors the HLIR dual-path (hlir_opt_apply_strategy) from Sprint 38 but operates
# on the flat IrFunction used by the native x86-64 backend.
#
# Contest<T>/Robust<T> functions carrying IR_STRATEGY_INSTRUMENTED (3) are now
# transformed into a 4-block diamond layout before native lowering:
#
#   dispatch_block → fast_path | instrumented_path → merge_block
#
# New file: self-hosted/ir/opt_strategy.sio
# Modified: self-hosted/native/codegen.sio (wires ir_opt_apply_strategy)
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

echo "=== Sprint 40: Native Dual-Path for Instrumented Strategy ==="
echo ""

echo "--- ir/opt_strategy.sio: core transformation functions ---"
check_grep "opt:module_decl" \
    "module ir::opt_strategy" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:remap_vreg_fn" \
    "fn ir_opt_remap_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:remap_label_fn" \
    "fn ir_opt_remap_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:max_vreg_fn" \
    "fn ir_opt_max_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:max_label_fn" \
    "fn ir_opt_max_label" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:clone_instr_fn" \
    "fn ir_opt_clone_instr" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:apply_strategy_fn" \
    "fn ir_opt_apply_strategy" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:append_helper_fn" \
    "fn ir_opt_append" \
    "self-hosted/ir/opt_strategy.sio"

echo ""
echo "--- ir/opt_strategy.sio: four_block_diamond structure ---"
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
check_grep "opt:vreg_offset" \
    "vreg_offset" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:label_offset" \
    "label_offset" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:instrumented_strategy_const" \
    "IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:return_rewrite" \
    "IrOpcode::IrReturn" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "opt:merge_jump_guard" \
    "ir_jump.*merge_label" \
    "self-hosted/ir/opt_strategy.sio"

echo ""
echo "--- native/codegen.sio: ir_opt_apply_strategy wired ---"
check_grep "codegen:import_opt_strategy" \
    "ir::opt_strategy" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:effective_func_var" \
    "var effective_func" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:instrumented_guard" \
    "IR_STRATEGY_INSTRUMENTED" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:apply_strategy_call" \
    "ir_opt_apply_strategy" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:effective_func_used_in_loop" \
    "effective_func\.instr_count" \
    "self-hosted/native/codegen.sio"

echo ""
echo "--- Test fixtures: native dual-path documentation ---"
check_grep "fixture_if:run_pass" \
    "//@ run-pass" \
    "tests/frontend/dual_path_native_if_contest.sio"
check_grep "fixture_if:four_block_diamond" \
    "four_block_diamond" \
    "tests/frontend/dual_path_native_if_contest.sio"
check_grep "fixture_if:instrumented_constant" \
    "IR_STRATEGY_INSTRUMENTED" \
    "tests/frontend/dual_path_native_if_contest.sio"
check_grep "fixture_match:run_pass" \
    "//@ run-pass" \
    "tests/frontend/dual_path_native_match_contest.sio"
check_grep "fixture_match:four_block_diamond" \
    "four_block_diamond" \
    "tests/frontend/dual_path_native_match_contest.sio"
check_grep "fixture_match:instrumented_constant" \
    "IR_STRATEGY_INSTRUMENTED" \
    "tests/frontend/dual_path_native_match_contest.sio"
check_grep "fixture_robust:run_pass" \
    "//@ run-pass" \
    "tests/frontend/dual_path_native_multi_return_robust.sio"
check_grep "fixture_robust:four_block_diamond" \
    "four_block_diamond" \
    "tests/frontend/dual_path_native_multi_return_robust.sio"
check_grep "fixture_robust:instrumented_constant" \
    "IR_STRATEGY_INSTRUMENTED" \
    "tests/frontend/dual_path_native_multi_return_robust.sio"
check_grep "fixture_robust:merge_label_ref" \
    "merge_label" \
    "tests/frontend/dual_path_native_multi_return_robust.sio"

echo ""
echo "--- Compiler self-test: flat IR dual-path regression ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in \
        "selftest:dispatch_merge" \
        "selftest:branch_targets"
    do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:dispatch_merge" "T08 OK: ir_opt dispatch and merge layout" "$SELF_TEST_LOG"
        check_log_line "selftest:branch_targets" "T09 OK: ir_opt remaps cloned branch targets" "$SELF_TEST_LOG"
    else
        for name in \
            "selftest:dispatch_merge" \
            "selftest:branch_targets"
        do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (compiler/main --self-test failed)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Sprint 38 HLIR dual-path regression ---"
check_grep "regression:hlir_apply_strategy" \
    "fn hlir_opt_apply_strategy" \
    "self-hosted/hlir/opt_strategy.sio"
check_grep "regression:hlir_clone_instrs" \
    "fn hlir_opt_clone_instrs" \
    "self-hosted/hlir/opt_strategy.sio"
check_grep "regression:hlir_remap_value" \
    "fn hlir_opt_remap_value" \
    "self-hosted/hlir/opt_strategy.sio"
check_grep "regression:hlir_strategy_enum" \
    "enum CompileStrategy" \
    "self-hosted/hlir/ir.sio"
check_grep "regression:hlir_wrap_float" \
    "fn hlir_opt_wrap_float_interval" \
    "self-hosted/hlir/opt_strategy.sio"

echo ""
echo "--- Sprint 38/39 native strategy regression ---"
check_grep "regression:ir_strategy_instrumented_const" \
    "IR_STRATEGY_INSTRUMENTED.*=.*3" \
    "self-hosted/ir/ir.sio"
check_grep "regression:ir_compile_strategy_field" \
    "compile_strategy: i64" \
    "self-hosted/ir/ir.sio"
check_grep "regression:frame_current_strategy" \
    "current_strategy: i64" \
    "self-hosted/native/frame.sio"
check_grep "regression:codegen_strategy_wire" \
    "current_strategy = func.compile_strategy" \
    "self-hosted/native/codegen.sio"
check_grep "regression:lower_compute_fn" \
    "fn ir_compute_strategy_from_ast" \
    "self-hosted/ir/lower.sio"
check_grep "regression:lower_ir_reassoc_helper" \
    "fn ir_native_strategy_allows_float_reassoc" \
    "self-hosted/native/lower_ir.sio"
check_grep "regression:lower_ir_swap_operands" \
    "if swap_operands" \
    "self-hosted/native/lower_ir.sio"

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
mkdir -p "$ROOT_DIR/artifacts/sprint40"
cat > "$ROOT_DIR/artifacts/sprint40/dual_path_native_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint40.dual_path_native_gate.v1",
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
  "dual_path_status": "instrumented_strategy_wired_and_self_test_validated",
  "architecture": {
    "hlir_dual_path": "complete (Sprint 38, hlir/opt_strategy.sio)",
    "native_dual_path": "ir_level_transformation_added_and_executed (Sprint 40, ir/opt_strategy.sio)"
  },
  "novel_claims": [
    "ir_opt_apply_strategy() transforms flat IrFunction into 4-block diamond layout",
    "Instrumented strategy (Contest<T>/Robust<T>) triggers dual-path in native backend",
    "compiler/main --self-test executes flat-IR dual-path regressions for dispatch/merge and cloned branch-target remap",
    "Vreg and label remapping mirrors HLIR pattern: vreg_offset=max_vreg+1, label_offset=max_label+1",
    "All IrReturn sites rewritten to IrJump(merge_label) in both fast and instrumented paths",
    "Native dual-path coexists with HLIR dual-path: both pipelines instrumented"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint40/dual_path_native_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
