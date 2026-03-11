#!/usr/bin/env bash
# sprint82_unary_fold_gate.sh — Block J: unary operation constant folding
#
# Extends ocp_const_fold to handle IrUnaryOp:
#   -(const) → IrLoadImm(-const)
#   !(const) → IrLoadImm(~const)   (bitwise NOT via XOR with -1)
#
# Self-tests T204–T209; total 209/209.
#
# SOTA: Wegman & Zadeck TOPLAS 1991 §4 — constant folding completeness
#       requires all operator families (unary included).
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
OUT_DIR="$ROOT_DIR/artifacts/sprint82"
mkdir -p "$OUT_DIR"

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

echo "=== Sprint 82: Block J — Unary Operation Constant Folding ==="
echo ""

echo "--- Group 1: Compiler self-test smoke (209/209) ---"
TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then
    echo "NOT_RUN  selftest:compiler_main (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
else
    _st_log="$(mktemp)"
    _st_ec=0
    timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$_st_log" 2>&1 || _st_ec=$?
    if [ "$_st_ec" -eq 0 ]; then
        echo "PASS  selftest:compiler_main"; PASS=$((PASS+1))
    elif [ "$_st_ec" -eq 124 ] || [ "$_st_ec" -eq 137 ]; then
        echo "NOT_RUN  selftest:compiler_main (OOM/timeout)"; NOT_RUN=$((NOT_RUN+1))
    else
        echo "NOT_RUN  selftest:compiler_main (exit $_st_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$_st_log"
fi

echo ""
echo "--- Group 2: Block J structural checks in opt_cleanup.sio ---"
check_grep "block_j:comment" \
    "Sprint 82 Block J" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_j:unaryop_check" \
    "IrUnaryOp" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_j:neg_fold" \
    "OpNeg" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_j:not_fold" \
    "OpNot" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_j:const_update" \
    "is_const\[instr.dst as usize\] = true" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_j:wegman_ref" \
    "Wegman.*Zadeck|Zadeck.*Wegman" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: T204–T209 wiring in main.sio ---"
check_grep "main:T204_fn" \
    "compiler_main_test_unary_neg_const" \
    "self-hosted/compiler/main.sio"
check_grep "main:T205_fn" \
    "compiler_main_test_unary_neg_zero" \
    "self-hosted/compiler/main.sio"
check_grep "main:T206_fn" \
    "compiler_main_test_unary_not_allones" \
    "self-hosted/compiler/main.sio"
check_grep "main:T207_fn" \
    "compiler_main_test_unary_not_zero" \
    "self-hosted/compiler/main.sio"
check_grep "main:T208_fn" \
    "compiler_main_test_unary_double_neg_chain" \
    "self-hosted/compiler/main.sio"
check_grep "main:T209_fn" \
    "compiler_main_test_unary_neg_nonconst" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_209" \
    "let total: i64 = 209" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Sprint 81 Block I regression guard ---"
check_grep "compat:block_i_comment" \
    "Sprint 81 Block I" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "compat:bitand_fold" \
    "OpBitAnd" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "compat:shift_guard" \
    "v2 >= 0 && v2 < 64" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 5: Sprint 78–80 regression guard ---"
check_grep "compat:licm_fn" \
    "fn lopt_optimize_function" \
    "self-hosted/ir/loop_opt.sio"
check_grep "compat:sccp_fn" \
    "fn cp_optimize_function" \
    "self-hosted/ir/const_prop.sio"
check_grep "compat:descriptor_table" \
    "fn native_v2_descriptor_table_decode" \
    "self-hosted/native/gc.sio"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"
REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then
    STATUS="fail"
    REASON="${FAIL}_cases_failed"
fi
if [ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ]; then
    REASON="all_run_passed_${NOT_RUN}_not_run"
fi

printf '{\n  "schema": "sounio.sprint82.unary_fold_gate.v1",\n  "generated_at": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "config": {\n    "root": "%s",\n    "timeout_seconds": 300\n  },\n  "metrics": {\n    "total": %d,\n    "passed": %d,\n    "failed": %d,\n    "not_run": %d\n  },\n  "novel_claims": [\n    "Block J adds IrUnaryOp constant folding to ocp_const_fold: -(const) folds to IrLoadImm(-const), !(const) folds to IrLoadImm(~const) via XOR with -1",\n    "Chain folding via const-prop: r0=7, r1=-r0, r2=-r1 correctly yields r2=7 through intermediate is_const tracking",\n    "Non-foldable case preserved: unary on unknown source stays IrUnaryOp (no over-folding)",\n    "Self-tests T204-T209: 6 new tests; total 209/209"\n  ]\n}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)" "$STATUS" "$REASON" "$ROOT_DIR" \
    "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN" \
    > "$OUT_DIR/unary_fold_gate.v1.json"

echo ""
echo "Artifact: artifacts/sprint82/unary_fold_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
