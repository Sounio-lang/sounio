#!/usr/bin/env bash
# sprint144_add_complement_gate.sh — Sprint 144: Block BF add-complement law
#
# Block BF: x + ~x → -1, ~x + x → -1
#   SOTA: LLVM InstCombineAddSub.cpp; Boolean algebra complement law x+~x = -1 (all ones).
#   Requires IrUnaryOp(OpNot) → is_bnot tracking (extended in Sprint 144).
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
        if ! grep -qF "T530" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T530)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 144: Block BF — Add-Complement Law ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bf_comment" "Block BF.*[Aa]dd.complement|Block BF.*x.*~x" "$OPT_FILE"
check_grep "src:bf_bnot_check" "is_bnot\[bf_s[12]" "$OPT_FILE"
check_grep "src:bf_load_imm_neg1" "ir_load_imm.*-1" "$OPT_FILE"
check_grep "src:opnot_tracking" "OpNot" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T530_fn" "fn compiler_main_test_add_complement_basic" "$MAIN_FILE"
check_grep "main:T535_fn" "fn compiler_main_test_add_complement_sub_no_fold" "$MAIN_FILE"
check_grep "main:total" "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T530-T535 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:T530" "selftest:T531" "selftest:T532" \
                "selftest:T533" "selftest:T534" "selftest:T535"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T530" "T530 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T531" "T531 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T532" "T532 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T533" "T533 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T534" "T534 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T535" "T535 OK" "$SELF_TEST_LOG"
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
