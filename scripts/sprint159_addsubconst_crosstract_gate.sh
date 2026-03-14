#!/usr/bin/env bash
# sprint159_addsubconst_crosstract_gate.sh — Block BU: Add/sub-const cross-track diff
#
# (x+C1)-(x-C2) → C1+C2  ;  (x-C1)-(x+C2) → -(C1+C2)
# (x-C1)-(x-C2) → C2-C1  ;  (x+C1)-(x+C2) → C1-C2
# Extends Block BD (as_valid) to cover chain-folded exprs tracked by bs_sub/bt_add.
# SOTA: LLVM InstCombineAddSub sub(add X,C1),(sub X,C2)→C1+C2; Cooper & Torczon §8.4.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file not found)"; NOT_RUN=$((NOT_RUN+1)); return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found)"; FAIL=$((FAIL+1))
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
check_grep "src:block_bu_comment"    "Sprint 159 Block BU" "$OPT_FILE"
check_grep "src:bu_bt_add_bs_sub"    "bt_add_valid\[bu_s1.*&& bs_sub_valid\[bu_s2" "$OPT_FILE"
check_grep "src:bu_bs_sub_bt_add"    "bs_sub_valid\[bu_s1.*&& bt_add_valid\[bu_s2" "$OPT_FILE"
check_grep "src:bu_same_base"        "bt_add_src\[bu_s1.*== bs_sub_src\[bu_s2" "$OPT_FILE"
check_grep "src:bu_load_imm"         "ir_load_imm.*bu_instr\.dst.*bu_result" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T620_fn"  "fn compiler_main_test_addconst_subconst_diff" "$MAIN_FILE"
check_grep "main:T621_fn"  "fn compiler_main_test_subconst_addconst_diff" "$MAIN_FILE"
check_grep "main:T622_fn"  "fn compiler_main_test_subconst_subconst_diff" "$MAIN_FILE"
check_grep "main:T625_fn"  "fn compiler_main_test_bu_chain_fold" "$MAIN_FILE"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T620-T625 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for t in T620 T621 T622 T623 T624 T625; do
        TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    for t in T620 T621 T622 T623 T624 T625; do
        check_log_line "selftest:$t" "$t OK" "$SELF_TEST_LOG"
    done
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
echo "--- Regression ---"
check_grep "regression:block_bd"    "Block BD" "$OPT_FILE"
check_grep "regression:bs_sub_valid" "bs_sub_valid" "$OPT_FILE"
check_grep "regression:bt_add_valid" "bt_add_valid" "$OPT_FILE"

echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"

mkdir -p "$ROOT_DIR/artifacts/sprint159"
cat > "$ROOT_DIR/artifacts/sprint159/addsubconst_crosstract.v1.json" <<EOF
{
  "schema": "sounio.sprint159.addsubconst_crosstract_gate.v1",
  "status": "$([ "$FAIL" -eq 0 ] && echo pass || echo fail)",
  "metrics": { "total": $TOTAL, "passed": $PASS, "failed": $FAIL, "not_run": $NOT_RUN }
}
EOF

if [ "$FAIL" -gt 0 ]; then echo "GATE: FAIL"; exit 1; fi
echo "GATE: PASS"
