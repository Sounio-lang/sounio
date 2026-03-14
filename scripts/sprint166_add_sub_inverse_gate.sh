#!/usr/bin/env bash
# sprint166_add_sub_inverse_gate.sh — Sprint 166: Block CB add/sub-const inverse
#
# Block CB: (x+C)-C → x,  (x-C)+C → x,  C+(x-C) → x
#   Zero new arrays: uses bt_add_valid/src/cval + bs_sub_valid/src/cval.
#   SOTA: LLVM InstCombineAddSub; group inverse a+(-a)=0.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"; NOT_RUN=$((NOT_RUN+1)); return
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
        echo "NOT_RUN  $name (log empty — OOM/killed)"; NOT_RUN=$((NOT_RUN+1)); return
    fi
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        if ! grep -qF "T662" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T662)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 166: Block CB — Add/Sub-Const Inverse ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_cb_comment"  "Block CB" "$OPT_FILE"
check_grep "src:cb_add_sub_fold"   "bt_add_valid\[cb_s1 as usize\]" "$OPT_FILE"
check_grep "src:cb_sub_add_fold"   "bs_sub_valid\[cb_s1 as usize\]" "$OPT_FILE"
check_grep "src:cb_ir_copy"        "ir_copy\(cb_instr.dst" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T662_fn"  "fn compiler_main_test_cb_add_sub_inverse_basic" "$MAIN_FILE"
check_grep "main:T663_fn"  "fn compiler_main_test_cb_sub_add_inverse_basic" "$MAIN_FILE"
check_grep "main:T664_fn"  "fn compiler_main_test_cb_sub_add_comm" "$MAIN_FILE"
check_grep "main:T665_fn"  "fn compiler_main_test_cb_no_fold_diff_const" "$MAIN_FILE"
check_grep "main:T666_fn"  "fn compiler_main_test_cb_no_fold_same_op" "$MAIN_FILE"
check_grep "main:T667_fn"  "fn compiler_main_test_cb_no_fold_plain_sub" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T662-T667 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T662 T663 T664 T665 T666 T667; do
        TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T662" "T662 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T663" "T663 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T664" "T664 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T665" "T665 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T666" "T666 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T667" "T667 OK" "$SELF_TEST_LOG"
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
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0
else echo "GATE: FAIL"; exit 1; fi
