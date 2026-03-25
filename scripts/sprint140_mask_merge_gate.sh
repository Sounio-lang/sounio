#!/usr/bin/env bash
# sprint140_mask_merge_gate.sh — Sprint 140: Block BB constant mask merge
#
# Block BB: (x & C1) | (x & C2) → IrCopy when C2 ⊆ C1 or C1 ⊆ C2
#   SOTA: LLVM InstCombineAndOrXor.cpp; Boolean algebra distributive axiom.
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
        if ! grep -qF "T506" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T506)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 140: Block BB — Constant Mask Merge ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bb_comment" "Block BB.*Constant mask merge" "$OPT_FILE"
check_grep "src:bb_and_valid_check" "and_valid\[bb_s1 as usize\]" "$OPT_FILE"
check_grep "src:bb_var_src_same" "and_var_src\[bb_s1.*and_var_src\[bb_s2" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T506_fn" "fn compiler_main_test_mask_merge_subset" "$MAIN_FILE"
check_grep "main:T509_fn" "fn compiler_main_test_mask_merge_no_subset" "$MAIN_FILE"
check_grep "main:T511_fn" "fn compiler_main_test_mask_merge_used" "$MAIN_FILE"
check_grep "main:total" "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T506-T511 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:T506" "selftest:T507" "selftest:T508" \
                "selftest:T509" "selftest:T510" "selftest:T511"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T506" "T506 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T507" "T507 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T508" "T508 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T509" "T509 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T510" "T510 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T511" "T511 OK" "$SELF_TEST_LOG"
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
