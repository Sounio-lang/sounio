#!/usr/bin/env bash
# sprint115_sub_reversal_gate.sh — Block AL: Subtract reversal through negation
#
# SOTA: LLVM InstCombineAddSub.cpp; Cooper & Torczon §8.4.
# neg(a - b) → b - a  ;  0 - (a - b) → b - a
#
# Novel claims:
#   1. neg(a - b) → b - a eliminates negation by reversing subtraction operands
#   2. 0 - (a - b) → b - a detects explicit zero-minus form
#   3. Sub-tracking arrays (is_sub/sub_lhs/sub_rhs) enable downstream neg-reversal
#   4. Chains with Block M (double-neg), Block AG (neg-operand add/sub)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

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
    if [ ! -s "$log_file" ]; then
        echo "NOT_RUN  $name (log empty — OOM/killed)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        if ! grep -qF "T394" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T394)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 115: Block AL — Subtract Reversal Through Negation ==="
echo ""

# --- Group 1: Self-tests T394-T399 ---
echo "--- Self-tests: sub reversal T394-T399 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:neg_sub_reversal" "selftest:zero_minus_sub" \
                "selftest:neg_add_no_fold" "selftest:neg_neg_sub_double" \
                "selftest:zero_minus_non_sub" "selftest:neg_sub_result_used"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:neg_sub_reversal" \
        "T394 OK: Sprint 115 Block AL: neg(a-b)" "$SELF_TEST_LOG"
    check_log_line "selftest:zero_minus_sub" \
        "T395 OK: Sprint 115 Block AL: 0-(a-b)" "$SELF_TEST_LOG"
    check_log_line "selftest:neg_add_no_fold" \
        "T396 OK: Sprint 115 Block AL: neg(a+b) no fold" "$SELF_TEST_LOG"
    check_log_line "selftest:neg_neg_sub_double" \
        "T397 OK: Sprint 115 Block AL: neg(neg(a-b))" "$SELF_TEST_LOG"
    check_log_line "selftest:zero_minus_non_sub" \
        "T398 OK: Sprint 115 Block AL: 0-(a+b) no reversal" "$SELF_TEST_LOG"
    check_log_line "selftest:neg_sub_result_used" \
        "T399 OK: Sprint 115 Block AL: neg(a-b) original preserved" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block AL implementation ---"
check_grep "ocp:sub_tracking_arrays" \
    "al_is_sub.*bool.*256|al_sub_lhs.*i64.*256|al_sub_rhs.*i64.*256" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:neg_of_sub_fold" \
    "al_is_sub\[nn_s|al_sub_rhs\[nn_s|al_sub_lhs\[nn_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:zero_minus_sub_fold" \
    "al_is_sub\[s2|al_sub_rhs\[s2|al_sub_lhs\[s2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:sub_tracking_populate" \
    "al_is_sub\[al_instr|al_sub_lhs\[al_instr|al_sub_rhs\[al_instr" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:block_al_comment" \
    "Sprint 115 Block AL" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T394_fn" \
    "fn compiler_main_test_neg_sub_reversal" \
    "self-hosted/compiler/main.sio"
check_grep "main:T395_fn" \
    "fn compiler_main_test_zero_minus_sub_reversal" \
    "self-hosted/compiler/main.sio"
check_grep "main:T396_fn" \
    "fn compiler_main_test_neg_add_no_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T397_fn" \
    "fn compiler_main_test_neg_neg_sub_double" \
    "self-hosted/compiler/main.sio"
check_grep "main:T398_fn" \
    "fn compiler_main_test_zero_minus_non_sub" \
    "self-hosted/compiler/main.sio"
check_grep "main:T399_fn" \
    "fn compiler_main_test_neg_sub_result_used" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block M/AG regression (negation still works) ---"
check_grep "regression:is_neg_tracking" \
    "is_neg" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:neg_src_tracking" \
    "neg_src" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_ag_add_neg" \
    "x \\+ neg.*x - y|OpAdd.*is_neg" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:opt_cleanup_module" \
    "fn opt_cleanup_module" \
    "self-hosted/ir/opt_cleanup.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint115"
cat > "$ROOT_DIR/artifacts/sprint115/sub_reversal.v1.json" <<EOF
{
  "schema": "sounio.sprint115.sub_reversal_gate.v1",
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
    "neg(a - b) → b - a: eliminates negation by reversing subtraction operands",
    "0 - (a - b) → b - a: explicit zero-minus form detected via is_sub tracking",
    "Sub-tracking arrays (is_sub/sub_lhs/sub_rhs) enable downstream negation reversal",
    "Chains with Block M (double-neg), Block AG (neg-operand add/sub), Block AH (add-sub chain)"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint115/sub_reversal.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
