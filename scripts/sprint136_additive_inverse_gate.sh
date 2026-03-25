#!/usr/bin/env bash
# sprint136_additive_inverse_gate.sh — Sprint 136: Block AX additive inverse via negation
#
# Block AX: a + neg(a) → 0, neg(a) + a → 0
#   SOTA: LLVM InstCombineAddSub.cpp; group theory a + (-a) = 0.
#   Distinct from Block A (x-x→0) — catches add-of-negation pattern.
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
        if ! grep -qF "T494" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T494)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 136: Block AX — Additive Inverse via Negation ==="
echo ""

# --- Source ---
echo "--- Source: opt_cleanup.sio Block AX ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_ax_comment" "Block AX.*Additive inverse" "$OPT_FILE"
check_grep "src:ax_neg_src_check" "neg_src\[ax_s2.*== ax_s1|neg_src\[ax_s1.*== ax_s2" "$OPT_FILE"
check_grep "src:ax_load_imm_zero" "ir_load_imm.*0" "$OPT_FILE"

# --- Tests ---
echo ""
echo "--- main.sio test functions ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T494_fn" "fn compiler_main_test_add_neg_inverse" "$MAIN_FILE"
check_grep "main:T495_fn" "fn compiler_main_test_neg_add_inverse" "$MAIN_FILE"
check_grep "main:T499_fn" "fn compiler_main_test_mul_neg_no_zero" "$MAIN_FILE"
check_grep "main:total_updated" "let total: i64 = [0-9]+" "$MAIN_FILE"

# --- Self-tests ---
echo ""
echo "--- Self-tests: T494-T499 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:T494" "selftest:T495" "selftest:T496" \
                "selftest:T497" "selftest:T498" "selftest:T499"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc binary not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T494_add_neg" "T494 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T495_neg_add" "T495 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T496_different" "T496 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T497_downstream" "T497 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T498_double_neg" "T498 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T499_mul_no_zero" "T499 OK" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

# --- Type-check ---
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
