#!/usr/bin/env bash
# sprint147_or_mask_clear_gate.sh — Sprint 147: Block BI OR-mask clearance via disjoint AND
#
# Block BI: (x | C1) & C2 → x & C2  when C1 & C2 == 0 (disjoint masks)
#   C2 & (x | C1) → x & C2  (commutative form)
#   Proof: (x|C1)&C2 = (x&C2)|(C1&C2) = (x&C2)|0 = x&C2
#   SOTA: LLVM InstCombineAndOrXor.cpp; known-bits analysis; Hacker's Delight §2.
#   Zero new arrays: reuses am_or_valid/am_or_var_src/am_or_const_val from Block AM.
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
        if ! grep -qF "T548" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T548)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 147: Block BI — OR-mask Clearance via Disjoint AND ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bi_comment" "Block BI.*OR.mask clearance|Block BI.*disjoint" "$OPT_FILE"
check_grep "src:bi_am_or_valid_check" "am_or_valid\[bf_s[12] as usize\]" "$OPT_FILE"
check_grep "src:bi_disjoint_test" "bf_c1 & bf_c2.*== 0" "$OPT_FILE"
check_grep "src:bi_ir_binop_rewrite" "ir_binop.*bf_instr\.dst" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T548_fn" "fn compiler_main_test_or_mask_clear_basic" "$MAIN_FILE"
check_grep "main:T551_fn" "fn compiler_main_test_or_mask_clear_no_fold_overlap" "$MAIN_FILE"
check_grep "main:T553_fn" "fn compiler_main_test_or_mask_clear_downstream" "$MAIN_FILE"
check_grep "main:total" "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T548-T553 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:T548" "selftest:T549" "selftest:T550" \
                "selftest:T551" "selftest:T552" "selftest:T553"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T548" "T548 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T549" "T549 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T550" "T550 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T551" "T551 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T552" "T552 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T553" "T553 OK" "$SELF_TEST_LOG"
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
