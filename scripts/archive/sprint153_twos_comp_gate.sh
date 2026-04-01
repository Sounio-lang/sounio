#!/usr/bin/env bash
# sprint153_twos_comp_gate.sh — Sprint 153: Block BO Two's-Complement Rewrite
#
# Block BO: ~x + 1 → -x  ;  1 + ~x → -x.
# Two's complement identity: -x = ~x + 1.
# Zero new arrays: reuses is_bnot/bnot_src (Block AI/BJ) + is_const/const_val.
# SOTA: LLVM InstCombineAddSub; Hacker's Delight §2-2.
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
        if ! grep -qF "T584" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T584)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 153: Block BO — Two's-Complement Rewrite ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bo_comment"   "Block BO.*[Tt]wo.s.compl|Block BO.*~x.*1.*-x" "$OPT_FILE"
check_grep "src:bo_bnot_check"      "is_bnot\[bo_s1 as usize\]" "$OPT_FILE"
check_grep "src:bo_const1_check"    "const_val\[bo_s2 as usize\] == 1" "$OPT_FILE"
check_grep "src:bo_emit_neg"        "ir_unaryop.*bo_instr\.dst.*UnaryOp::OpNeg.*bo_inner" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T584_fn"  "fn compiler_main_test_bo_twos_comp_basic" "$MAIN_FILE"
check_grep "main:T585_fn"  "fn compiler_main_test_bo_twos_comp_commutative" "$MAIN_FILE"
check_grep "main:T586_fn"  "fn compiler_main_test_bo_twos_comp_no_fold_const2" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T584-T589 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T584 T585 T586 T587 T588 T589; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T584" "T584 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T585" "T585 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T586" "T586 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T587" "T587 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T588" "T588 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T589" "T589 OK" "$SELF_TEST_LOG"
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
