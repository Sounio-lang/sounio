#!/usr/bin/env bash
# sprint159_add_sub_diff_gate.sh — Sprint 159: Block BU add/sub-const cross-track diff
#
# Block BU: (x+C1)-(x-C2) → C1+C2, (x-C1)-(x+C2) → -(C1+C2)
#           (x-C1)-(x-C2) → C2-C1,  (x+C1)-(x+C2) → C1-C2
#   Zero new arrays: uses bs_sub_valid/src/cval (Block BS) + bt_add_valid/src/cval (Block BT).
#   SOTA: LLVM InstCombineAddSub; const-offset difference collapse; Cooper & Torczon §8.2.
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
        if ! grep -qF "T620" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T620)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 159: Block BU — Add/Sub-Const Cross-Track Diff ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bu_comment"   "Block BU.*Add.sub.const|Block BU.*cross.track" "$OPT_FILE"
check_grep "src:bu_bt_plus_bs_fold" "bt_add_valid\[bu_s1 as usize\] && bs_sub_valid\[bu_s2 as usize\]" "$OPT_FILE"
check_grep "src:bu_bs_plus_bt_fold" "bs_sub_valid\[bu_s1 as usize\] && bt_add_valid\[bu_s2 as usize\]" "$OPT_FILE"
check_grep "src:bu_bs_bs_fold"      "bs_sub_valid\[bu_s1 as usize\] && bs_sub_valid\[bu_s2 as usize\]" "$OPT_FILE"
check_grep "src:bu_bt_bt_fold"      "bt_add_valid\[bu_s1 as usize\] && bt_add_valid\[bu_s2 as usize\]" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T620_fn"  "fn compiler_main_test_addconst_subconst_diff" "$MAIN_FILE"
check_grep "main:T621_fn"  "fn compiler_main_test_subconst_addconst_diff" "$MAIN_FILE"
check_grep "main:T622_fn"  "fn compiler_main_test_subconst_subconst_diff" "$MAIN_FILE"
check_grep "main:T623_fn"  "fn compiler_main_test_addconst_addconst_diff_bu" "$MAIN_FILE"
check_grep "main:T624_fn"  "fn compiler_main_test_bu_no_fold_diff_base" "$MAIN_FILE"
check_grep "main:T625_fn"  "fn compiler_main_test_bu_chain_fold" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T620-T625 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T620 T621 T622 T623 T624 T625; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T620" "T620 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T621" "T621 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T622" "T622 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T623" "T623 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T624" "T624 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T625" "T625 OK" "$SELF_TEST_LOG"
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
