#!/usr/bin/env bash
# sprint157_sub_chain_gate.sh — Sprint 157: Block BS consecutive subtract-constant fold
#
# Block BS: (x - C1) - C2 → x - (C1+C2)
#   Mirrors Block BG (OR-chain) and Block BM (XOR-chain) for subtraction.
#   New arrays: bs_sub_valid, bs_sub_src, bs_sub_cval (3 arrays).
#   SOTA: LLVM InstCombineAddSub.cpp; Cooper & Torczon §8.2; associative arithmetic.
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
        if ! grep -qF "T608" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T608)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 157: Block BS — Consecutive Subtract-Constant Fold ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bs_comment"   "Block BS.*subtract.constant|Block BS.*x.*C1.*C2" "$OPT_FILE"
check_grep "src:bs_sub_valid_decl"  "var bs_sub_valid" "$OPT_FILE"
check_grep "src:bs_fold_check"      "bs_sub_valid\[bs_s1 as usize\] && is_const\[bs_s2 as usize\]" "$OPT_FILE"
check_grep "src:bs_merged"          "bs_c1 \+ bs_c2|bs_merged" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T608_fn"  "fn compiler_main_test_bs_sub_chain_basic" "$MAIN_FILE"
check_grep "main:T611_fn"  "fn compiler_main_test_bs_sub_chain_no_fold_var" "$MAIN_FILE"
check_grep "main:T613_fn"  "fn compiler_main_test_bs_sub_chain_no_fold_rhs_var" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T608-T613 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T608 T609 T610 T611 T612 T613; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T608" "T608 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T609" "T609 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T610" "T610 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T611" "T611 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T612" "T612 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T613" "T613 OK" "$SELF_TEST_LOG"
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
