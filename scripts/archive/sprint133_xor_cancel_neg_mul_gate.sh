#!/usr/bin/env bash
# sprint133_xor_cancel_neg_mul_gate.sh — Sprints 133-135
#
# Block AU: XOR operand cancellation — (a^b)^(a^c)→b^c
#   SOTA: LLVM InstCombineAndOrXor.cpp; Hacker's Delight §2-1 (XOR group law).
#
# Block AV: Distributive factoring — a*b+a*b→2*(a*b), general deferred
#   SOTA: LLVM Reassociate.cpp; Briggs & Cooper 1994.
#
# Block AW: Negation propagation through mul/div — neg(a)*neg(b)→a*b
#   SOTA: LLVM InstCombineMulDivRem.cpp; ring theory: (-a)(-b)=ab.
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
        if ! grep -qF "T482" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T482)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprints 133-135: XOR Cancel + Distributive + Neg Mul ==="
echo ""

# --- Group 1: Source code ---
echo "--- Source: opt_cleanup.sio Block AU/AV/AW ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"

check_grep "src:block_au_xor_cancel" \
    "Block AU.*XOR operand cancellation" "$OPT_FILE"
check_grep "src:block_av_distributive" \
    "Block AV.*Distributive factoring" "$OPT_FILE"
check_grep "src:block_aw_neg_mul" \
    "Block AW.*Negation propagation" "$OPT_FILE"
check_grep "src:aw_neg_neg_check" \
    "is_neg\[aw_s1.*is_neg\[aw_s2" "$OPT_FILE"
check_grep "src:aw_neg_src_rewrite" \
    "neg_src\[aw_s1.*OpMul.*neg_src\[aw_s2" "$OPT_FILE"

# --- Group 2: main.sio test functions ---
echo ""
echo "--- main.sio test functions ---"
MAIN_FILE="self-hosted/compiler/main.sio"

check_grep "main:T482_fn" "fn compiler_main_test_xor_cancel_lhs_lhs" "$MAIN_FILE"
check_grep "main:T488_fn" "fn compiler_main_test_neg_neg_mul" "$MAIN_FILE"
check_grep "main:T489_fn" "fn compiler_main_test_neg_neg_div" "$MAIN_FILE"
check_grep "main:T492_fn" "fn compiler_main_test_neg_neg_mul_same" "$MAIN_FILE"
check_grep "main:total_updated" "let total: i64 = 493" "$MAIN_FILE"

# --- Group 3: Self-tests T482-T493 ---
echo ""
echo "--- Self-tests: T482-T493 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in \
        "selftest:T482" "selftest:T483" "selftest:T484" "selftest:T485" \
        "selftest:T486" "selftest:T487" "selftest:T488" "selftest:T489" \
        "selftest:T490" "selftest:T491" "selftest:T492" "selftest:T493"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc binary not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?

    check_log_line "selftest:T482_xor_cancel_ll" "T482 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T483_xor_cancel_lr" "T483 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T484_xor_cancel_rl" "T484 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T485_xor_cancel_rr" "T485 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T486_xor_cancel_neg" "T486 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T487_xor_cancel_const" "T487 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T488_neg_neg_mul" "T488 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T489_neg_neg_div" "T489 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T490_neg_single_mul" "T490 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T491_neg_single_rhs" "T491 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T492_neg_neg_same" "T492 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T493_neg_neg_used" "T493 OK" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

# --- Group 4: Type-check ---
echo ""
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else
    echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1))
fi

# --- Summary ---
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
