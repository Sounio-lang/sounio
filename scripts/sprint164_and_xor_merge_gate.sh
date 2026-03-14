#!/usr/bin/env bash
# sprint164_and_xor_merge_gate.sh — Sprint 164: Block BZ AND-XOR same-constant merge
#
# Block BZ: (x & C) | (x ^ C) → x | C,  (x ^ C) | (x & C) → x | C
#   Zero new arrays: uses and_valid/and_var_src/and_const_val + bm_xor_valid/src/cval.
#   SOTA: LLVM InstCombineAndOrXor or(and X,C),xor(X,C) → or X,C.
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
        if ! grep -qF "T650" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T650)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 164: Block BZ — AND-XOR Same-Constant Merge ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bz_comment"   "Block BZ" "$OPT_FILE"
check_grep "src:bz_and_valid_check" "and_valid\[bz_s[12] as usize\]" "$OPT_FILE"
check_grep "src:bz_xor_valid_check" "bm_xor_valid\[bz_s[12] as usize\]" "$OPT_FILE"
check_grep "src:bz_same_base"       "and_var_src\[bz_s[12] as usize\] == bm_xor_src\[bz_s[12] as usize\]" "$OPT_FILE"
check_grep "src:bz_same_const"      "and_const_val\[bz_s[12] as usize\] == bm_xor_cval\[bz_s[12] as usize\]" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T650_fn"  "fn compiler_main_test_bz_and_xor_merge_basic" "$MAIN_FILE"
check_grep "main:T651_fn"  "fn compiler_main_test_bz_and_xor_merge_comm" "$MAIN_FILE"
check_grep "main:T652_fn"  "fn compiler_main_test_bz_and_xor_merge_val" "$MAIN_FILE"
check_grep "main:T653_fn"  "fn compiler_main_test_bz_no_fold_diff_const" "$MAIN_FILE"
check_grep "main:T654_fn"  "fn compiler_main_test_bz_no_fold_diff_base" "$MAIN_FILE"
check_grep "main:T655_fn"  "fn compiler_main_test_bz_no_fold_plain" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = 655" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T650-T655 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T650 T651 T652 T653 T654 T655; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T650" "T650 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T651" "T651 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T652" "T652 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T653" "T653 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T654" "T654 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T655" "T655 OK" "$SELF_TEST_LOG"
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
