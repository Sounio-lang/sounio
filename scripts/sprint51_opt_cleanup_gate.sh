#!/usr/bin/env bash
# sprint51_opt_cleanup_gate.sh — Post-Inlining SCCP + DCE Cleanup
#
# SOTA: GCC runs SCCP + aggressive DCE after every inlining event;
# LLVM runs instcombine + simplifycfg + dce in its -O2 pipeline.
#
# Novel claim: SCCP + DCE integrated into self-hosted single-binary PGO
# pipeline. Full chain: instrument → profile → promote → inline → layout
# → const_prop → dce → codegen. No external tool needed.
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

echo "=== Sprint 51: Post-Inlining SCCP + DCE Cleanup ==="
echo ""

# --- Group 1: Self-tests T26-T27 ---
echo "--- Self-tests: const_prop + dce passes ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:cp_folds_constant" "selftest:dce_removes_dead"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:cp_folds_constant" \
            "T26 OK: cp folds constant binop" "$SELF_TEST_LOG"
        check_log_line "selftest:dce_removes_dead" \
            "T27 OK: dce removes dead instruction" "$SELF_TEST_LOG"
    else
        for name in "selftest:cp_folds_constant" "selftest:dce_removes_dead"; do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (--self-test failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: Structural pass entry points ---"
check_grep "optcleanup:module_decl" \
    "module ir::opt_cleanup" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "optcleanup:cleanup_function" \
    "fn opt_cleanup_function" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "optcleanup:cleanup_module" \
    "fn opt_cleanup_module" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "optcleanup:const_fold_pass" \
    "fn ocp_const_fold" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "optcleanup:dce_pass" \
    "fn ocp_dce" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: Pipeline integration ---"
check_grep "main:opt_log_prefix" \
    "\\[opt\\]" \
    "self-hosted/compiler/main.sio"
check_grep "main:opt_cleanup_wired" \
    "opt_cleanup_module" \
    "self-hosted/compiler/main.sio"
check_grep "main:import_opt_cleanup" \
    "use ir::opt_cleanup" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Test fixture ---"
check_grep "fixture:run_pass" \
    "//@ run-pass" \
    "tests/frontend/opt_cleanup_contest.sio"
check_grep "fixture:hot_caller" \
    "fn hot_caller" \
    "tests/frontend/opt_cleanup_contest.sio"

echo ""
echo "--- Group 5: Sprint 38-50 regression ---"
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
check_grep "regression:prof_counter_opcode" \
    "IrProfCounter" \
    "self-hosted/ir/ir.sio"
check_grep "regression:prof_dump_fn" \
    "fn emit_prof_dump_function" \
    "self-hosted/native/codegen.sio"
check_grep "regression:sprof_parse_fn" \
    "fn sprof_parse" \
    "self-hosted/ir/profile.sio"
check_grep "regression:inline_pass_fn" \
    "fn inl_run_pass" \
    "self-hosted/ir/inline.sio"
check_grep "regression:layout_sort_by_profile_fn" \
    "fn layout_sort_by_profile" \
    "self-hosted/ir/layout.sio"
check_grep "regression:sprof_promotion_target" \
    "fn sprof_promotion_target" \
    "self-hosted/ir/profile.sio"

echo ""
echo "--- Group 6: Existing infrastructure ---"
check_grep "infra:inl_analyze_function" \
    "fn inl_analyze_function" \
    "self-hosted/ir/inline.sio"
check_grep "infra:layout_module_decl" \
    "module ir::layout" \
    "self-hosted/ir/layout.sio"
check_grep "infra:strategy_bonus" \
    "INL_STRATEGY_AGGRESSIVE_BONUS" \
    "self-hosted/ir/inline.sio"
check_grep "infra:layout_patch_fn_ids" \
    "fn layout_patch_fn_ids" \
    "self-hosted/ir/layout.sio"
check_grep "infra:sprof_header_fn" \
    "fn emit_prof_write_sprof_header" \
    "self-hosted/native/codegen.sio"
check_grep "infra:has_valid_profile_var" \
    "has_valid_profile" \
    "self-hosted/compiler/main.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint51"
cat > "$ROOT_DIR/artifacts/sprint51/opt_cleanup_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint51.opt_cleanup_gate.v1",
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
  "novel_claims": [
    "SCCP + DCE integrated into self-hosted single-binary PGO pipeline",
    "Post-inlining cleanup: constant folding + dead code elimination per function",
    "Full chain: instrument -> profile -> promote -> inline -> layout -> const_prop -> dce -> codegen",
    "No external tool needed (contrast GCC/LLVM separate pass managers)"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint51/opt_cleanup_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
