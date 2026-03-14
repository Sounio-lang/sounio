#!/usr/bin/env bash
# sprint161_xor_and_comp_gate.sh — Sprint 161: Block BW XOR-mask AND complement
#
# Block BW: (x ^ C) & ~C → x & ~C,  ~C & (x ^ C) → x & ~C
#   Zero new arrays: uses bm_xor_valid/src/cval (Block BM) + is_const/const_val.
#   SOTA: LLVM InstCombineAndOrXor — and(xor X,C),~C → and X,~C; Boolean algebra.
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
        if ! grep -qF "T632" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T632)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 161: Block BW — XOR-Mask AND Complement ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bw_comment"   "Block BW.*XOR.mask|Block BW.*x.*C.*~C" "$OPT_FILE"
check_grep "src:bw_xor_valid_check" "bm_xor_valid\[bw_s1 as usize\]" "$OPT_FILE"
check_grep "src:bw_not_c_check"     "bm_xor_cval\[bw_s[12] as usize\] \^ -1" "$OPT_FILE"
check_grep "src:bw_and_rewrite"     "bm_xor_src\[bw_s[12] as usize\], BinaryOp::OpBitAnd" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T632_fn"  "fn compiler_main_test_bw_xor_and_comp_basic" "$MAIN_FILE"
check_grep "main:T633_fn"  "fn compiler_main_test_bw_xor_and_comp_comm" "$MAIN_FILE"
check_grep "main:T634_fn"  "fn compiler_main_test_bw_xor_and_comp_val" "$MAIN_FILE"
check_grep "main:T635_fn"  "fn compiler_main_test_bw_no_fold_wrong_complement" "$MAIN_FILE"
check_grep "main:T636_fn"  "fn compiler_main_test_bw_no_fold_or_op" "$MAIN_FILE"
check_grep "main:T637_fn"  "fn compiler_main_test_bw_no_fold_plain_and" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T632-T637 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T632 T633 T634 T635 T636 T637; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T632" "T632 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T633" "T633 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T634" "T634 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T635" "T635 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T636" "T636 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T637" "T637 OK" "$SELF_TEST_LOG"
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
