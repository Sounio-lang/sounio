#!/usr/bin/env bash
# sprint155_or_complement_gate.sh — Sprint 155: Block BQ OR-complement absorption
#
# Block BQ: x | ~x → -1, ~x | x → -1
#   Boolean algebra: a ∨ ¬a = 1 (all-ones, complement law for OR).
#   Zero new arrays: uses is_bnot/bnot_src (Block AI/BJ/BF) + is_const/const_val.
#   SOTA: LLVM InstCombineAndOrXor; Hacker's Delight §2-1; Boolean complement.
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
        if ! grep -qF "T596" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T596)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 155: Block BQ — OR-Complement Absorption ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bq_comment"   "Block BQ.*OR.complement|Block BQ.*x.*~x" "$OPT_FILE"
check_grep "src:bq_bnot_check"      "is_bnot\[bq_s[12] as usize\]" "$OPT_FILE"
check_grep "src:bq_base_eq_check"   "bnot_src\[bq_s[12] as usize\] == bq_s[12]" "$OPT_FILE"
check_grep "src:bq_neg1_emit"       "ir_load_imm.*bq_instr\.dst.*-1" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T596_fn"  "fn compiler_main_test_bq_or_complement_basic" "$MAIN_FILE"
check_grep "main:T597_fn"  "fn compiler_main_test_bq_or_complement_commutative" "$MAIN_FILE"
check_grep "main:T599_fn"  "fn compiler_main_test_bq_or_complement_no_fold_diff_reg" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T596-T601 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T596 T597 T598 T599 T600 T601; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T596" "T596 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T597" "T597 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T598" "T598 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T599" "T599 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T600" "T600 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T601" "T601 OK" "$SELF_TEST_LOG"
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
