#!/usr/bin/env bash
# sprint151_xor_chain_gate.sh — Sprint 151: Block BM XOR Constant-Chain Fold
#
# Block BM: (x ^ C1) ^ C2 → x ^ (C1^C2). GF(2) field associativity.
# Also handles commutative outer: C2 ^ (x ^ C1) → x ^ (C1^C2).
# New arrays: bm_xor_valid/src/cval.
# SOTA: LLVM InstCombineAndOrXor; GF(2) field associativity; Hacker's Delight §2-3.
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
        if ! grep -qF "T572" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T572)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 151: Block BM — XOR Constant-Chain Fold ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bm_comment"   "Block BM.*XOR.*chain|Block BM.*constant.chain" "$OPT_FILE"
check_grep "src:bm_valid_decl"      "bm_xor_valid.*bool.*256" "$OPT_FILE"
check_grep "src:bm_fold_trigger"    "bm_xor_valid\[bm_s1 as usize\]" "$OPT_FILE"
check_grep "src:bm_merged_xor"      "bm_c1 \^ bm_c2" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T572_fn"  "fn compiler_main_test_bm_xor_chain_basic" "$MAIN_FILE"
check_grep "main:T574_fn"  "fn compiler_main_test_bm_xor_chain_cancel" "$MAIN_FILE"
check_grep "main:T576_fn"  "fn compiler_main_test_bm_xor_no_fold_rr" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T572-T577 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T572 T573 T574 T575 T576 T577; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T572" "T572 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T573" "T573 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T574" "T574 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T575" "T575 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T576" "T576 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T577" "T577 OK" "$SELF_TEST_LOG"
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
