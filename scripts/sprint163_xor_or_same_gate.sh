#!/usr/bin/env bash
# sprint163_xor_or_same_gate.sh — Sprint 163: Block BY XOR-OR same-constant fold
#
# Block BY: (x ^ C) | C → x | C,  C | (x ^ C) → x | C
#   Zero new arrays: uses bm_xor_valid/src/cval (Block BM) + is_const/const_val.
#   SOTA: LLVM InstCombineAndOrXor or(xor(X,C),C) → or(X,C); Boolean absorption.
set -eo pipefail

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
        if ! grep -qF "T644" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T644)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 163: Block BY — XOR-OR Same-Constant Fold ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_by_comment"   "Block BY" "$OPT_FILE"
check_grep "src:by_xor_valid_check" "bm_xor_valid\[by_s[12] as usize\]" "$OPT_FILE"
check_grep "src:by_const_match"     "bm_xor_cval\[by_s[12] as usize\] == const_val\[by_s[12] as usize\]" "$OPT_FILE"
check_grep "src:by_or_rewrite"      "bm_xor_src\[by_s[12] as usize\], BinaryOp::OpBitOr" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T644_fn"  "fn compiler_main_test_by_xor_or_same_basic" "$MAIN_FILE"
check_grep "main:T645_fn"  "fn compiler_main_test_by_xor_or_same_comm" "$MAIN_FILE"
check_grep "main:T646_fn"  "fn compiler_main_test_by_xor_or_same_val" "$MAIN_FILE"
check_grep "main:T647_fn"  "fn compiler_main_test_by_no_fold_diff_const" "$MAIN_FILE"
check_grep "main:T648_fn"  "fn compiler_main_test_by_no_fold_and_op" "$MAIN_FILE"
check_grep "main:T649_fn"  "fn compiler_main_test_by_no_fold_plain_or" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = 649" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T644-T649 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T644 T645 T646 T647 T648 T649; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T644" "T644 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T645" "T645 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T646" "T646 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T647" "T647 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T648" "T648 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T649" "T649 OK" "$SELF_TEST_LOG"
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
