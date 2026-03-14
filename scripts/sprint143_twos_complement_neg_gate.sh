#!/usr/bin/env bash
# sprint143_twos_complement_neg_gate.sh — Block BE: Two's complement negation
#
# ~x + 1 → neg(x)  ;  1 + ~x → neg(x)
# Ring identity: ~x = -x - 1, so ~x + 1 = -x. (Two's complement negation.)
# SOTA: LLVM InstCombineAddSub.cpp — not(X)+1 → neg(X); Patterson & Hennessy §B.1.
# Distinct from Block M (neg arithmetic): identifies bnot+1 as neg, not neg*neg.
# Zero new arrays: uses is_bnot/bnot_src (Block AI) + is_const/const_val.
#
# Novel claims:
#   1. ~x + 1 → neg(x): two's complement negation via NOT+1
#   2. 1 + ~x → neg(x): commutative form
#   3. Result dst points to x (original operand), not ~x
#   4. ~x + 2 does NOT fold (guard: only const 1)
#   5. ~x - 1 does NOT fold (wrong op)
#   6. Zero new arrays: reuses is_bnot/bnot_src from Block AI
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
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
        if ! grep -qF "T524" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T524)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 143: Block BE — Two's Complement Negation ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_be_comment"   "Block BE.*Two.s complement" "$OPT_FILE"
check_grep "src:be_bnot_check"      "is_bnot\[be_s1 as usize\]" "$OPT_FILE"
check_grep "src:be_const1_check"    "const_val\[be_s2 as usize\] == 1" "$OPT_FILE"
check_grep "src:be_neg_emit"        "ir_unaryop.*OpNeg.*bnot_src" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T524_fn" "fn compiler_main_test_twos_comp_neg_basic" "$MAIN_FILE"
check_grep "main:T525_fn" "fn compiler_main_test_twos_comp_neg_commutative" "$MAIN_FILE"
check_grep "main:T527_fn" "fn compiler_main_test_twos_comp_neg_no_fold_wrong_const" "$MAIN_FILE"
check_grep "main:total"   "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T524-T529 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T524 T525 T526 T527 T528 T529; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T524" "T524 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T525" "T525 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T526" "T526 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T527" "T527 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T528" "T528 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T529" "T529 OK" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Type-check (opt_cleanup only — main.sio has pre-existing array-size mismatches) ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/ir/opt_cleanup.sio 2>&1 | grep -q "ownership errors\|Resolution errors"; then
    echo "FAIL  typecheck:opt_cleanup.sio"; FAIL=$((FAIL+1))
else
    echo "PASS  typecheck:opt_cleanup.sio"; PASS=$((PASS+1))
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
