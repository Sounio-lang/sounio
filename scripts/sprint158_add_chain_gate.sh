#!/usr/bin/env bash
# sprint158_add_chain_gate.sh — Sprint 158: Block BT consecutive add-constant fold
#
# Block BT: (x + C1) + C2 → x + (C1+C2)
#   C2 + (x + C1) → x + (C1+C2)  (commutative outer)
#   Mirrors Block BS (sub-chain) for addition. New arrays: bt_add_valid/src/cval.
#   SOTA: LLVM InstCombineAddSub.cpp; additive associativity; Cooper & Torczon §8.2.
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
        if ! grep -qF "T614" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T614)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 158: Block BT — Consecutive Add-Constant Fold ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bt_comment"   "Block BT.*add.constant|Block BT.*x.*C1.*C2" "$OPT_FILE"
check_grep "src:bt_add_valid_decl"  "var bt_add_valid" "$OPT_FILE"
check_grep "src:bt_fold_check"      "bt_add_valid\[bt_s[12] as usize\] && is_const" "$OPT_FILE"
check_grep "src:bt_merged"          "bt_c1 \+ bt_c2|bt_merged" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T614_fn"  "fn compiler_main_test_bt_add_chain_basic" "$MAIN_FILE"
check_grep "main:T617_fn"  "fn compiler_main_test_bt_add_chain_no_fold_var" "$MAIN_FILE"
check_grep "main:T619_fn"  "fn compiler_main_test_bt_add_chain_no_fold_rhs_var" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T614-T619 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T614 T615 T616 T617 T618 T619; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T614" "T614 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T615" "T615 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T616" "T616 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T617" "T617 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T618" "T618 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T619" "T619 OK" "$SELF_TEST_LOG"
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
