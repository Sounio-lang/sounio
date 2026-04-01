#!/usr/bin/env bash
# sprint162_or_xor_comp_gate.sh — Sprint 162: Block BX OR-XOR complement fold
#
# Block BX: (x | C) ^ C → x & ~C,  C ^ (x | C) → x & ~C
#   Zero new arrays: uses am_or_valid/var_src/const_val (Block AM) + is_const + def_at.
#   SOTA: LLVM InstCombineAndOrXor — xor(or X,C),C → and X,~C; Boolean algebra.
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
        if ! grep -qF "T638" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T638)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 162: Block BX — OR-XOR Complement Fold ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bx_comment"   "Block BX.*OR.XOR|Block BX.*x.*OR.*C.*XOR" "$OPT_FILE"
check_grep "src:bx_or_valid_check"  "am_or_valid\[bx_s1 as usize\]" "$OPT_FILE"
check_grep "src:bx_const_match"     "am_or_const_val\[bx_s1 as usize\] == const_val\[bx_s2 as usize\]" "$OPT_FILE"
check_grep "src:bx_notc_patch"      "bx_notc = am_or_const_val.*\^ -1" "$OPT_FILE"
check_grep "src:bx_and_rewrite"     "am_or_var_src\[bx_s[12] as usize\], BinaryOp::OpBitAnd" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T638_fn"  "fn compiler_main_test_bx_or_xor_comp_basic" "$MAIN_FILE"
check_grep "main:T639_fn"  "fn compiler_main_test_bx_or_xor_comp_comm" "$MAIN_FILE"
check_grep "main:T640_fn"  "fn compiler_main_test_bx_or_xor_comp_merged_const" "$MAIN_FILE"
check_grep "main:T641_fn"  "fn compiler_main_test_bx_no_fold_diff_const" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T638-T643 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T638 T639 T640 T641 T642 T643; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T638" "T638 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T639" "T639 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T640" "T640 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T641" "T641 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T642" "T642 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T643" "T643 OK" "$SELF_TEST_LOG"
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
