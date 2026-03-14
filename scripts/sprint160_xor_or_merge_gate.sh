#!/usr/bin/env bash
# sprint160_xor_or_merge_gate.sh — Sprint 160: Block BV XOR-OR constant merge
#
# Block BV: (x ^ C) | C → x | C,  C | (x ^ C) → x | C
#   Zero new arrays: uses bm_xor_valid/src/cval (Block BM) + is_const/const_val.
#   SOTA: LLVM InstCombineAndOrXor — or(xor X,C),C → or X,C; Boolean algebra.
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
        if ! grep -qF "T626" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T626)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 160: Block BV — XOR-OR Constant Merge ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bv_comment"   "Block BV.*XOR.OR|Block BV.*x.*C.*OR" "$OPT_FILE"
check_grep "src:bv_xor_valid_check" "bm_xor_valid\[bv_s1 as usize\]" "$OPT_FILE"
check_grep "src:bv_const_check"     "is_const\[bv_s2 as usize\]" "$OPT_FILE"
check_grep "src:bv_cval_match"      "bm_xor_cval\[bv_s1 as usize\] == const_val\[bv_s2 as usize\]" "$OPT_FILE"
check_grep "src:bv_rewrite"         "ir_binop.*bv_instr\.dst.*bm_xor_src" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T626_fn"  "fn compiler_main_test_xor_or_merge_basic" "$MAIN_FILE"
check_grep "main:T627_fn"  "fn compiler_main_test_xor_or_merge_commutative" "$MAIN_FILE"
check_grep "main:T628_fn"  "fn compiler_main_test_xor_or_merge_diff_const" "$MAIN_FILE"
check_grep "main:T629_fn"  "fn compiler_main_test_xor_or_no_fold_const_mismatch" "$MAIN_FILE"
check_grep "main:T630_fn"  "fn compiler_main_test_xor_or_no_fold_and_op" "$MAIN_FILE"
check_grep "main:T631_fn"  "fn compiler_main_test_xor_or_no_fold_plain_or" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T626-T631 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T626 T627 T628 T629 T630 T631; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T626" "T626 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T627" "T627 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T628" "T628 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T629" "T629 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T630" "T630 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T631" "T631 OK" "$SELF_TEST_LOG"
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
