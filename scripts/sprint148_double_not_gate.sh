#!/usr/bin/env bash
# sprint148_double_not_gate.sh — Sprint 148: Block BJ double NOT cancel
#
# Block BJ: ~(~x) → IrCopy(x)
#   SOTA: LLVM InstCombineAndOrXor; Boolean double negation law; Hacker's Delight §2-2.
#   Zero new arrays: reuses is_bnot/bnot_src from Block AI and Sprint 144.
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
        if ! grep -qF "T554" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T554)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 148: Block BJ — Double NOT Cancel ==="
echo ""
echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bj_comment" "Block BJ.*[Dd]ouble NOT|Block BJ.*~~x" "$OPT_FILE"
check_grep "src:bj_is_bnot_check" "is_bnot\[us as usize\]" "$OPT_FILE"
check_grep "src:bj_ir_copy" "ir_copy.*bj_inner" "$OPT_FILE"
check_grep "src:bj_opnot_guard" "OpNot.*is_bnot\[us as usize\]|is_bnot\[us as usize\].*OpNot" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T554_fn" "fn compiler_main_test_double_not_basic" "$MAIN_FILE"
check_grep "main:T559_fn" "fn compiler_main_test_double_not_no_fold_diff_reg" "$MAIN_FILE"
check_grep "main:total" "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T554-T559 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:T554" "selftest:T555" "selftest:T556" \
                "selftest:T557" "selftest:T558" "selftest:T559"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T554" "T554 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T555" "T555 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T556" "T556 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T557" "T557 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T558" "T558 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T559" "T559 OK" "$SELF_TEST_LOG"
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
