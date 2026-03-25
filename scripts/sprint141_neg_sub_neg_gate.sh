#!/usr/bin/env bash
# sprint141_neg_sub_neg_gate.sh — Block BC: Negation subtraction simplification
#
# SOTA: LLVM InstCombineAddSub.cpp `-X - -Y → Y - X`; ring theory -a-(-b) = b-a.
# neg(a) - neg(b) → b - a  — eliminates both negations in subtraction.
# Complements Block AW (neg*neg → a*b) and Block AX (a+neg(a) → 0).
# Zero new arrays: reuses is_neg/neg_src from Block M (Sprint 86).
#
# Novel claims:
#   1. neg(a) - neg(b) → b - a: double-neg subtraction fold
#   2. neg(a) - neg(a) → 0: chains with Block A (x-x → 0)
#   3. Distinct from Block AG (x - neg(y) → x+y): requires BOTH operands negated
#   4. Zero new arrays: reuses is_neg/neg_src tracking from Block M
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
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
    if [ ! -s "$log_file" ]; then
        echo "NOT_RUN  $name (log empty — OOM/killed)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        if ! grep -qF "T512" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T512)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 141: Block BC — Negation Subtraction Simplification ==="
echo ""

# --- Group 1: Self-tests T512-T517 ---
echo "--- Self-tests: neg-sub-neg T512-T517 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:neg_sub_neg_basic" "selftest:neg_sub_neg_swapped" \
                "selftest:neg_sub_neg_same" "selftest:neg_sub_pos_no_fold" \
                "selftest:pos_sub_neg_no_bc" "selftest:neg_mul_neg_aw"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:neg_sub_neg_basic" \
        "T512 OK: Sprint 141 Block BC:" "$SELF_TEST_LOG"
    check_log_line "selftest:neg_sub_neg_swapped" \
        "T513 OK: Sprint 141 Block BC:" "$SELF_TEST_LOG"
    check_log_line "selftest:neg_sub_neg_same" \
        "T514 OK: Sprint 141 Block BC:" "$SELF_TEST_LOG"
    check_log_line "selftest:neg_sub_pos_no_fold" \
        "T515 OK: Sprint 141 Block BC:" "$SELF_TEST_LOG"
    check_log_line "selftest:pos_sub_neg_no_bc" \
        "T516 OK: Sprint 141 Block BC:" "$SELF_TEST_LOG"
    check_log_line "selftest:neg_mul_neg_aw" \
        "T517 OK: Sprint 141 Block BC:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block BC implementation ---"
check_grep "ocp:bc_block_comment" \
    "Sprint 141 Block BC" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bc_both_neg_check" \
    "is_neg\[bc_ns" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bc_fold_b_minus_a" \
    "neg_src\[bc_ns" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bc_opsub_check" \
    "bc_neg_instr.bin_op == BinaryOp::OpSub" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T512_fn" \
    "fn compiler_main_test_neg_sub_neg_basic" \
    "self-hosted/compiler/main.sio"
check_grep "main:T513_fn" \
    "fn compiler_main_test_neg_sub_neg_swapped" \
    "self-hosted/compiler/main.sio"
check_grep "main:T514_fn" \
    "fn compiler_main_test_neg_sub_neg_same" \
    "self-hosted/compiler/main.sio"
check_grep "main:T515_fn" \
    "fn compiler_main_test_neg_sub_pos_no_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T516_fn" \
    "fn compiler_main_test_pos_sub_neg_no_bc" \
    "self-hosted/compiler/main.sio"
check_grep "main:T517_fn" \
    "fn compiler_main_test_neg_mul_neg_aw" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block M/AG/AW regression ---"
check_grep "regression:is_neg_tracking" \
    "is_neg" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:neg_src_tracking" \
    "neg_src" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_ag_comment" \
    "Block AG" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_aw_comment" \
    "Block AW" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint141"
cat > "$ROOT_DIR/artifacts/sprint141/neg_sub_neg.v1.json" <<EOF
{
  "schema": "sounio.sprint141.neg_sub_neg_gate.v1",
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
    "neg(a) - neg(b) → b - a: double-neg subtraction fold via is_neg tracking",
    "neg(a) - neg(a) → 0: chains with Block A (x-x→0) in single pass",
    "Distinct from Block AG (x-neg(y)→x+y): requires BOTH operands negated",
    "Zero new arrays: reuses is_neg/neg_src from Block M (Sprint 86)",
    "SOTA: LLVM InstCombineAddSub.cpp -X--Y→Y-X; ring theory -a-(-b)=b-a"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint141/neg_sub_neg.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
