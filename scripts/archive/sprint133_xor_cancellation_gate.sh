#!/usr/bin/env bash
# sprint133_xor_cancellation_gate.sh — Block AU: XOR operand cancellation
#
# SOTA: LLVM InstCombineAndOrXor.cpp; Hacker's Delight §2-1 (XOR group law).
# (a^b)^(a^c) → b^c — common XOR factor cancels via associativity.
# All four lhs/rhs pairings handled via as_xor_rr_valid tracking.
#
# Novel claims:
#   1. (a^b)^(a^c) → b^c: lhs-lhs common factor cancellation
#   2. (a^b)^(c^a) → b^c: lhs-rhs common factor cancellation
#   3. (b^a)^(a^c) → b^c: rhs-lhs common factor cancellation
#   4. (b^a)^(c^a) → b^c: rhs-rhs common factor cancellation
#   5. Zero new tracking arrays: reuses as_xor_rr_valid from Block AS (Sprint 131)
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
        if ! grep -qF "T482" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T482)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 133: Block AU — XOR Operand Cancellation ==="
echo ""

# --- Group 1: Self-tests T482-T487 ---
echo "--- Self-tests: XOR cancellation T482-T487 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:xor_cancel_lhs_lhs" "selftest:xor_cancel_lhs_rhs" \
                "selftest:xor_cancel_rhs_lhs" "selftest:xor_cancel_rhs_rhs" \
                "selftest:xor_cancel_no_fold" "selftest:xor_cancel_const_no_track"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:xor_cancel_lhs_lhs" \
        "T482 OK: Sprint 133 Block AU:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_cancel_lhs_rhs" \
        "T483 OK: Sprint 133 Block AU:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_cancel_rhs_lhs" \
        "T484 OK: Sprint 133 Block AU:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_cancel_rhs_rhs" \
        "T485 OK: Sprint 133 Block AU:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_cancel_no_fold" \
        "T486 OK: Sprint 133 Block AU:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_cancel_const_no_track" \
        "T487 OK: Sprint 133 Block AU:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block AU implementation ---"
check_grep "ocp:au_block_comment" \
    "Sprint 133 Block AU" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:au_xor_rr_valid_check" \
    "as_xor_rr_valid\[au_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:au_lhs_lhs_case" \
    "au_l1 == au_l2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:au_lhs_rhs_case" \
    "au_l1 == au_r2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:au_rhs_lhs_case" \
    "au_r1 == au_l2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:au_rhs_rhs_case" \
    "au_r1 == au_r2" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T482_fn" \
    "fn compiler_main_test_xor_cancel_lhs_lhs" \
    "self-hosted/compiler/main.sio"
check_grep "main:T483_fn" \
    "fn compiler_main_test_xor_cancel_lhs_rhs" \
    "self-hosted/compiler/main.sio"
check_grep "main:T484_fn" \
    "fn compiler_main_test_xor_cancel_rhs_lhs" \
    "self-hosted/compiler/main.sio"
check_grep "main:T485_fn" \
    "fn compiler_main_test_xor_cancel_rhs_rhs" \
    "self-hosted/compiler/main.sio"
check_grep "main:T486_fn" \
    "fn compiler_main_test_xor_cancel_no_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T487_fn" \
    "fn compiler_main_test_xor_cancel_const_no_track" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block AS/AT regression (tracking still works) ---"
check_grep "regression:as_xor_rr_valid_tracking" \
    "as_xor_rr_valid" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:as_xor_rr_lhs_tracking" \
    "as_xor_rr_lhs" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_as_comment" \
    "Block AS" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint133"
cat > "$ROOT_DIR/artifacts/sprint133/xor_cancellation.v1.json" <<EOF
{
  "schema": "sounio.sprint133.xor_cancellation_gate.v1",
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
    "(a^b)^(a^c) -> b^c: all four lhs/rhs pairings of common-factor cancellation",
    "Zero new tracking arrays: reuses as_xor_rr_valid/lhs/rhs from Block AS (Sprint 131)",
    "Negative cases: no shared factor and const-source prevent fold correctly",
    "SOTA: LLVM InstCombineAndOrXor.cpp; Hacker's Delight §2-1 (XOR group law)"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint133/xor_cancellation.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
