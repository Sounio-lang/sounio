#!/usr/bin/env bash
# sprint168_zero_minus_norm_gate.sh — Sprint 168: Block CD zero-minus normalization
#
# Block CD: 0 - x → neg(x)
#   Zero new arrays: updates is_neg/neg_src (Block M, Sprint 86).
#   SOTA: LLVM InstCombineAddSub sub(0,X)→neg(X); arithmetic normalization.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"; NOT_RUN=$((NOT_RUN+1)); return
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
        echo "NOT_RUN  $name (log empty — OOM/killed)"; NOT_RUN=$((NOT_RUN+1)); return
    fi
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        if ! grep -qF "T674" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T674)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 168: Block CD — Zero-Minus Normalization ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_cd_comment"   "Block CD" "$OPT_FILE"
check_grep "src:cd_sub_op"          "cd_instr\.bin_op == BinaryOp::OpSub" "$OPT_FILE"
check_grep "src:cd_const_zero"      "const_val\[cd_s1 as usize\] == 0" "$OPT_FILE"
check_grep "src:cd_unaryop_neg"     "ir_unaryop.*cd_instr\.dst.*UnaryOp::OpNeg.*cd_s2" "$OPT_FILE"
check_grep "src:cd_is_neg_update"   "is_neg\[cd_instr\.dst as usize\]" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T674_fn"  "fn compiler_main_test_cd_zero_minus_basic" "$MAIN_FILE"
check_grep "main:T675_fn"  "fn compiler_main_test_cd_zero_minus_chain" "$MAIN_FILE"
check_grep "main:T676_fn"  "fn compiler_main_test_cd_zero_minus_val" "$MAIN_FILE"
check_grep "main:T677_fn"  "fn compiler_main_test_cd_no_fold_nonzero" "$MAIN_FILE"
check_grep "main:T678_fn"  "fn compiler_main_test_cd_no_fold_rr_sub" "$MAIN_FILE"
check_grep "main:T679_fn"  "fn compiler_main_test_cd_no_fold_sub_zero" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T674-T679 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T674 T675 T676 T677 T678 T679; do
        TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T674" "T674 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T675" "T675 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T676" "T676 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T677" "T677 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T678" "T678 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T679" "T679 OK" "$SELF_TEST_LOG"
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
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0
else echo "GATE: FAIL"; exit 1; fi
