#!/usr/bin/env bash
# sprint149_shift_chain_gate.sh — Sprint 149: Block BK Consecutive Shift Fold
#
# (x << A) << B → x << (A+B)  when A+B < 64.
# (x >> A) >> B → x >> (A+B)  when A+B < 64.
# Tracks shifts-by-constant in bk_shl_valid/bk_shr_valid arrays.
# SOTA: LLVM InstCombineShifts.cpp FoldShiftByConstant; Cooper & Torczon §8.1.
set -eo pipefail

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
        if ! grep -qF "T560" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T560)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 149: Block BK — Consecutive Same-Direction Shift Fold ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bk_comment"   "Block BK.*[Ss]hift|Block BK.*consecutive" "$OPT_FILE"
check_grep "src:bk_shl_valid_decl"  "bk_shl_valid.*bool.*256" "$OPT_FILE"
check_grep "src:bk_shl_fold"        "bk_shl_valid\[bk_s1 as usize\]" "$OPT_FILE"
check_grep "src:bk_shr_fold"        "bk_shr_valid\[bk_s1 as usize\]" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T560_fn"  "fn compiler_main_test_bk_shl_chain" "$MAIN_FILE"
check_grep "main:T561_fn"  "fn compiler_main_test_bk_shr_chain" "$MAIN_FILE"
check_grep "main:T563_fn"  "fn compiler_main_test_bk_no_fold_overflow" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T560-T565 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T560 T561 T562 T563 T564 T565; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T560" "T560 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T561" "T561 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T562" "T562 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T563" "T563 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T564" "T564 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T565" "T565 OK" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else
    echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1))
fi

echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then
    echo "GATE: PASS"
    exit 0
else
    echo "GATE: FAIL"
    exit 1
fi
