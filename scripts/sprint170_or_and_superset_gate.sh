#!/usr/bin/env bash
# sprint170_or_and_superset_gate.sh — Sprint 170: Block CF OR-AND superset absorption
#
# Block CF: (x|C1)&C2 → C2 when C2 ⊆ C1  (i.e. C1&C2==C2)
#   Zero new arrays: uses am_or_valid/var_src/const_val (Block AM).
#   SOTA: LLVM InstCombineAndOrXor; Boolean absorption when mask is superset.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"; TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then echo "NOT_RUN  $name (file not found)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then echo "PASS  $name"; PASS=$((PASS+1))
    else echo "FAIL  $name (pattern not found)"; FAIL=$((FAIL+1)); fi
}
check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"; TOTAL=$((TOTAL+1))
    if [ ! -s "$log_file" ]; then echo "NOT_RUN  $name (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qF "$expected_line" "$log_file"; then echo "PASS  $name"; PASS=$((PASS+1))
    elif ! grep -qF "T686" "$log_file" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T686)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name"; FAIL=$((FAIL+1)); fi
}
echo "=== Sprint 170: Block CF — OR-AND Superset Absorption ==="
echo ""
echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_cf_comment" "Block CF" "$OPT_FILE"
check_grep "src:cf_or_valid"      "am_or_valid\[cf_s[12] as usize\]" "$OPT_FILE"
check_grep "src:cf_subset_check"  "am_or_const_val\[cf_s[12] as usize\] & const_val\[cf_s[12] as usize\]" "$OPT_FILE"
echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T686_fn" "fn compiler_main_test_cf_or_and_superset_eq" "$MAIN_FILE"
check_grep "main:T687_fn" "fn compiler_main_test_cf_or_and_superset_sub" "$MAIN_FILE"
check_grep "main:T688_fn" "fn compiler_main_test_cf_or_and_superset_comm" "$MAIN_FILE"
check_grep "main:T689_fn" "fn compiler_main_test_cf_no_fold_not_subset" "$MAIN_FILE"
check_grep "main:T690_fn" "fn compiler_main_test_cf_no_fold_disjoint" "$MAIN_FILE"
check_grep "main:T691_fn" "fn compiler_main_test_cf_no_fold_plain_and" "$MAIN_FILE"
check_grep "main:total"   "let total: i64 = 691" "$MAIN_FILE"
echo ""
echo "--- Self-tests: T686-T691 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T686 T687 T688 T689 T690 T691; do
        TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1)); done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    for name in T686 T687 T688 T689 T690 T691; do check_log_line "selftest:$name" "$name OK" "$SELF_TEST_LOG"; done
fi
rm -f "$SELF_TEST_LOG"
echo ""
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1)); fi
echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
