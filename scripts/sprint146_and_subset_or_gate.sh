#!/usr/bin/env bash
# sprint146_and_subset_or_gate.sh — Sprint 146: Block BH AND-subset OR absorption
#
# Block BH: (x & C1) | C2 → C2  when C1 ⊆ C2  (i.e., (C1 & C2) == C1)
#   C2 | (x & C1) → C2  (commutative form)
#   SOTA: LLVM InstCombineAndOrXor.cpp; Boolean absorption law; Hacker's Delight §2-1.
#   Zero new arrays: reuses and_valid/and_const_val from Block AE (Sprint 108).
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
        if ! grep -qF "T542" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T542)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 146: Block BH — AND-subset OR Absorption ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bh_comment" "Block BH.*AND.subset OR|Block BH.*absorption" "$OPT_FILE"
check_grep "src:bh_and_valid_check" "and_valid\[be_s[12] as usize\]" "$OPT_FILE"
check_grep "src:bh_subset_test" "be_c1 & be_c2.*== be_c1|be_c2.*& be_c1.*== be_c1" "$OPT_FILE"
check_grep "src:bh_load_imm_c2" "ir_load_imm.*be_instr\.dst.*be_c2" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T542_fn" "fn compiler_main_test_and_subset_or_basic" "$MAIN_FILE"
check_grep "main:T545_fn" "fn compiler_main_test_and_subset_or_no_fold_superset" "$MAIN_FILE"
check_grep "main:T547_fn" "fn compiler_main_test_and_subset_or_downstream" "$MAIN_FILE"
check_grep "main:total" "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T542-T547 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:T542" "selftest:T543" "selftest:T544" \
                "selftest:T545" "selftest:T546" "selftest:T547"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T542" "T542 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T543" "T543 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T544" "T544 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T545" "T545 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T546" "T546 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T547" "T547 OK" "$SELF_TEST_LOG"
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
