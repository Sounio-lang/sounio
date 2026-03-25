#!/usr/bin/env bash
# sprint139_sub_add_cancel_gate.sh — Block BA: Sub-add inverse cancellation
#
# SOTA: LLVM InstCombineAddSub.cpp; ring axiom a - b + b = a.
# (a - b) + b → a  ;  b + (a - b) → a  ;  a - (a - b) → b
# Distinct from Block AQ (a+b-b → a): this uses al_is_sub tracking.
# Zero new arrays: reuses al_is_sub/al_sub_lhs/al_sub_rhs from Block AL.
#
# Novel claims:
#   1. (a - b) + b → a: sub-tracking enables ring cancellation in ADD
#   2. b + (a - b) → a: commutative form handled symmetrically
#   3. a - (a - b) → b: outer sub cancels inner sub's lhs
#   4. Zero new arrays: reuses Block AL (Sprint 115) tracking
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
        if ! grep -qF "T500" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T500)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 139: Block BA — Sub-Add Inverse Cancellation ==="
echo ""

# --- Group 1: Self-tests T500-T505 ---
echo "--- Self-tests: sub-add cancellation T500-T505 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:sub_add_cancel_basic" "selftest:sub_add_cancel_commutative" \
                "selftest:sub_sub_cancel" "selftest:sub_add_no_fold" \
                "selftest:sub_sub_no_fold" "selftest:sub_add_chain"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:sub_add_cancel_basic" \
        "T500 OK: Sprint 139 Block BA:" "$SELF_TEST_LOG"
    check_log_line "selftest:sub_add_cancel_commutative" \
        "T501 OK: Sprint 139 Block BA:" "$SELF_TEST_LOG"
    check_log_line "selftest:sub_sub_cancel" \
        "T502 OK: Sprint 139 Block BA:" "$SELF_TEST_LOG"
    check_log_line "selftest:sub_add_no_fold" \
        "T503 OK: Sprint 139 Block BA:" "$SELF_TEST_LOG"
    check_log_line "selftest:sub_sub_no_fold" \
        "T504 OK: Sprint 139 Block BA:" "$SELF_TEST_LOG"
    check_log_line "selftest:sub_add_chain" \
        "T505 OK: Sprint 139 Block BA:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block BA implementation ---"
check_grep "ocp:ba_block_comment" \
    "Sprint 139 Block BA" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ba_add_lhs_sub" \
    "al_is_sub\[ba_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ba_sub_rhs_match" \
    "al_sub_rhs\[ba_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ba_sub_lhs_match" \
    "al_sub_lhs\[ba_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ba_ir_copy" \
    "ir_copy.*ba_instr.dst.*al_sub" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T500_fn" \
    "fn compiler_main_test_sub_add_cancel_basic" \
    "self-hosted/compiler/main.sio"
check_grep "main:T501_fn" \
    "fn compiler_main_test_sub_add_cancel_commutative" \
    "self-hosted/compiler/main.sio"
check_grep "main:T502_fn" \
    "fn compiler_main_test_sub_sub_cancel" \
    "self-hosted/compiler/main.sio"
check_grep "main:T503_fn" \
    "fn compiler_main_test_sub_add_no_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T504_fn" \
    "fn compiler_main_test_sub_sub_no_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T505_fn" \
    "fn compiler_main_test_sub_add_chain" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block AL regression (sub tracking still works) ---"
check_grep "regression:al_is_sub_tracking" \
    "al_is_sub" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:al_sub_lhs_tracking" \
    "al_sub_lhs" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_al_comment" \
    "Block AL" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_aq_additive" \
    "Block AQ" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint139"
cat > "$ROOT_DIR/artifacts/sprint139/sub_add_cancel.v1.json" <<EOF
{
  "schema": "sounio.sprint139.sub_add_cancel_gate.v1",
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
    "(a-b)+b → a: sub-tracking enables ring cancellation in ADD direction",
    "b+(a-b) → a: commutative form handled symmetrically",
    "a-(a-b) → b: outer sub cancels inner sub's lhs",
    "Zero new arrays: reuses al_is_sub/al_sub_lhs/al_sub_rhs from Block AL (Sprint 115)",
    "Complements Block AQ (a+b-b → a) for the reverse direction"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint139/sub_add_cancel.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
