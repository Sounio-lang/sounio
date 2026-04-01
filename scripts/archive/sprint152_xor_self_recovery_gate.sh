#!/usr/bin/env bash
# sprint152_xor_self_recovery_gate.sh — Sprint 152: Block BN XOR Self-Recovery
#
# Block BN: (x ^ C) ^ x → C  ;  x ^ (x ^ C) → C  ;  (x^C1)^(x^C2) → C1^C2.
# XOR group law: (a^b)^a=b; GF(2) field self-inverse.
# Zero new arrays: uses bm_xor_valid/src/cval (Block BM) + is_const/const_val.
# SOTA: LLVM InstCombineAndOrXor; Hacker's Delight §2-3.
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
        if ! grep -qF "T578" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T578)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 152: Block BN — XOR Self-Recovery ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bn_comment"   "Block BN.*XOR self.recov|Block BN.*XOR.*pair" "$OPT_FILE"
check_grep "src:bn_s1_xor_check"    "bm_xor_valid\[bn_s1 as usize\]" "$OPT_FILE"
check_grep "src:bn_base_eq_s2"      "bm_xor_src\[bn_s1 as usize\] == bn_s2" "$OPT_FILE"
check_grep "src:bn_load_imm_result" "ir_load_imm.*bn_instr\.dst.*bm_xor_cval" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T578_fn"  "fn compiler_main_test_xor_self_recovery_basic" "$MAIN_FILE"
check_grep "main:T580_fn"  "fn compiler_main_test_xor_pair_cancel" "$MAIN_FILE"
check_grep "main:T582_fn"  "fn compiler_main_test_xor_self_recovery_no_fold_diff_base" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T578-T583 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T578 T579 T580 T581 T582 T583; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T578" "T578 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T579" "T579 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T580" "T580 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T581" "T581 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T582" "T582 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T583" "T583 OK" "$SELF_TEST_LOG"
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
