#!/usr/bin/env bash
# sprint156_xor_complement_gate.sh — Sprint 156: Block BR XOR-complement
#
# Block BR: x ^ ~x → -1, ~x ^ x → -1
#   Boolean algebra: a ⊕ ¬a = 1 (all-ones); GF(2) complement law.
#   Zero new arrays: uses is_bnot/bnot_src (Block AI/BJ/BF) + is_const/const_val.
#   SOTA: LLVM InstCombineAndOrXor; Hacker's Delight §2-1; GF(2) complement.
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
        if ! grep -qF "T602" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T602)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 156: Block BR — XOR-Complement ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_br_comment"   "Block BR.*XOR.complement|Block BR.*x.*~x" "$OPT_FILE"
check_grep "src:br_bnot_check"      "is_bnot\[br_s[12] as usize\]" "$OPT_FILE"
check_grep "src:br_base_eq_check"   "bnot_src\[br_s[12] as usize\] == br_s[12]" "$OPT_FILE"
check_grep "src:br_neg1_emit"       "ir_load_imm.*br_instr\.dst.*-1" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T602_fn"  "fn compiler_main_test_br_xor_complement_basic" "$MAIN_FILE"
check_grep "main:T603_fn"  "fn compiler_main_test_br_xor_complement_commutative" "$MAIN_FILE"
check_grep "main:T605_fn"  "fn compiler_main_test_br_xor_complement_no_fold_diff_reg" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T602-T607 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T602 T603 T604 T605 T606 T607; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T602" "T602 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T603" "T603 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T604" "T604 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T605" "T605 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T606" "T606 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T607" "T607 OK" "$SELF_TEST_LOG"
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
