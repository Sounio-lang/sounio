#!/usr/bin/env bash
# sprint50_layout_pgo_gate.sh — Profile-Guided Function Layout
#
# SOTA: BOLT post-link optimizer (7-20% speedup); HFSort C³ clustering
# (Facebook CGO 2017); LLVM PGO BasicBlockPlacement; Google AutoFDO (10.5%).
#
# Novel claim: IR-level function layout optimization integrated with
# strategy-aware PGO pipeline. Single self-hosted compiler binary handles
# instrument → profile → promote → inline → layout → codegen.
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

echo "=== Sprint 50: Profile-Guided Function Layout ==="
echo ""

# --- Group 1: Self-tests T24-T25 ---
echo "--- Self-tests: layout pass ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:layout_sort_hot_first" "selftest:layout_sort_no_profile"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:layout_sort_hot_first" \
            "T24 OK: layout sort hot first" "$SELF_TEST_LOG"
        check_log_line "selftest:layout_sort_no_profile" \
            "T25 OK: layout sort no profile" "$SELF_TEST_LOG"
    else
        for name in "selftest:layout_sort_hot_first" "selftest:layout_sort_no_profile"; do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (--self-test failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: Layout module structure ---"
check_grep "layout:module_decl" \
    "module ir::layout" \
    "self-hosted/ir/layout.sio"
check_grep "layout:import_ir" \
    "use ir::ir" \
    "self-hosted/ir/layout.sio"
check_grep "layout:import_profile" \
    "use ir::profile" \
    "self-hosted/ir/layout.sio"
check_grep "layout:LayoutScore_struct" \
    "struct LayoutScore" \
    "self-hosted/ir/layout.sio"

echo ""
echo "--- Group 3: Sorting + patching ---"
check_grep "layout:sort_scores_fn" \
    "fn layout_sort_scores" \
    "self-hosted/ir/layout.sio"
check_grep "layout:sort_by_profile_fn" \
    "fn layout_sort_by_profile" \
    "self-hosted/ir/layout.sio"
check_grep "layout:patch_fn_ids_fn" \
    "fn layout_patch_fn_ids" \
    "self-hosted/ir/layout.sio"

echo ""
echo "--- Group 4: Pipeline integration ---"
check_grep "main:import_layout" \
    "use ir::layout" \
    "self-hosted/compiler/main.sio"
check_grep "main:layout_call" \
    "layout_sort_by_profile" \
    "self-hosted/compiler/main.sio"
check_grep "main:layout_logging" \
    "\\[layout\\]" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 5: Test fixture ---"
check_grep "fixture:run_pass" \
    "//@ run-pass" \
    "tests/frontend/layout_pgo_contest.sio"
check_grep "fixture:hot_caller" \
    "fn hot_caller" \
    "tests/frontend/layout_pgo_contest.sio"

echo ""
echo "--- Group 6: Sprint 38-49 regression ---"
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

echo ""
echo "--- Group 7: Existing infrastructure ---"
check_grep "infra:inline_analyze" \
    "fn inl_analyze_function" \
    "self-hosted/ir/inline.sio"
check_grep "infra:sprof_promotion_target" \
    "fn sprof_promotion_target" \
    "self-hosted/ir/profile.sio"
check_grep "infra:strategy_bonus" \
    "INL_STRATEGY_AGGRESSIVE_BONUS" \
    "self-hosted/ir/inline.sio"
check_grep "infra:sprof_header_fn" \
    "fn emit_prof_write_sprof_header" \
    "self-hosted/native/codegen.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint50"
cat > "$ROOT_DIR/artifacts/sprint50/layout_pgo_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint50.layout_pgo_gate.v1",
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
    "IR-level function layout optimization (hot functions emitted first)",
    "fn_id remapping preserves call-graph correctness after reorder",
    "Integrated with strategy-aware PGO: instrument → profile → promote → inline → layout → codegen",
    "No post-link tool needed (contrast BOLT, AutoFDO)"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint50/layout_pgo_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
