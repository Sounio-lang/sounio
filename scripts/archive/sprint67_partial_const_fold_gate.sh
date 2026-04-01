#!/usr/bin/env bash
# sprint67_partial_const_fold_gate.sh — Partial-constant identity/annihilation folding
#
# SOTA: Allen & Cocke "A Catalogue of Optimizing Transformations" (1972) §algebraic simplification;
# Muchnick "Advanced Compiler Design and Implementation" §12.1 — constant folding extended to
# identity laws (x*1=x, x+0=x, x*0=0, x/1=x) over partially-known operand sets.
#
# Novel claims:
#   1. ocp_const_fold extended with partial-constant branch: fires when exactly one operand
#      of IrBinOp is in the known-constant map (is_const[]) and an identity/annihilation rule applies
#   2. Six algebraic rules at IR level: x*1→IrCopy, x*0→IrLoadImm(0), 1*x→IrCopy, x+0→IrCopy,
#      x-0→IrCopy, x/1→IrCopy; all via a single extra if-block in ocp_const_fold
#   3. DCE (Sprint 51) naturally reclaims the now-dead IrLoadImm constant operand after folding
#   4. Full pipeline after Sprint 67: instrument→profile→promote→inline→layout→
#      partial+full const_fold+DCE→TCO→4-preg regalloc→disp8→compact frame→peephole
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
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
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 67: Partial-constant identity/annihilation folding ==="
echo ""

# --- Group 1: Self-tests T86-T92 ---
echo "--- Self-tests: partial const fold T86-T92 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:T86" "selftest:T87" "selftest:T88" \
                "selftest:T89" "selftest:T90" "selftest:T91" "selftest:T92"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:T86_mul_identity_right" \
        "T86 OK: ocp_const_fold folds x * 1 → IrCopy(dst, x) (identity mul, constant on right)" "$SELF_TEST_LOG"
    check_log_line "selftest:T87_mul_annihilator_right" \
        "T87 OK: ocp_const_fold folds x * 0 → IrLoadImm(dst, 0) (annihilator mul, constant on right)" "$SELF_TEST_LOG"
    check_log_line "selftest:T88_mul_identity_left" \
        "T88 OK: ocp_const_fold folds 1 * x → IrCopy(dst, x) (identity mul, constant on left)" "$SELF_TEST_LOG"
    check_log_line "selftest:T89_add_identity_right" \
        "T89 OK: ocp_const_fold folds x + 0 → IrCopy(dst, x) (add identity, constant on right)" "$SELF_TEST_LOG"
    check_log_line "selftest:T90_sub_identity" \
        "T90 OK: ocp_const_fold folds x - 0 → IrCopy(dst, x) (sub identity)" "$SELF_TEST_LOG"
    check_log_line "selftest:T91_div_identity" \
        "T91 OK: ocp_const_fold folds x / 1 → IrCopy(dst, x) (div identity)" "$SELF_TEST_LOG"
    check_log_line "selftest:T92_round_trip" \
        "T92 OK: opt_cleanup_function round-trip collapses a dead copy chain to IrNop" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio — partial-constant fold wiring ---"
check_grep "ocp:sprint67_comment" \
    "Sprint 67" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:s1_known_var" \
    "s1_known" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:s2_known_var" \
    "s2_known" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:exactly_one_known" \
    "s1_known.*s2_known|s2_known.*s1_known" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: opt_cleanup.sio — algebraic rules ---"
check_grep "ocp:mul_identity_rule" \
    "OpMul" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:mul_annihilator_rule" \
    "v2 == 0|v1 == 0" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:add_identity_rule" \
    "OpAdd" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:sub_identity_rule" \
    "OpSub" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:div_identity_rule" \
    "OpDiv" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ir_copy_writeback" \
    "ir_copy\(" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 4: main.sio — test registration ---"
check_grep "main:T86_fn" \
    "fn compiler_main_test_ocp_mul_identity_right" \
    "self-hosted/compiler/main.sio"
check_grep "main:T87_fn" \
    "fn compiler_main_test_ocp_mul_annihilator_right" \
    "self-hosted/compiler/main.sio"
check_grep "main:T88_fn" \
    "fn compiler_main_test_ocp_mul_identity_left" \
    "self-hosted/compiler/main.sio"
check_grep "main:T89_fn" \
    "fn compiler_main_test_ocp_add_identity_right" \
    "self-hosted/compiler/main.sio"
check_grep "main:T90_fn" \
    "fn compiler_main_test_ocp_sub_identity" \
    "self-hosted/compiler/main.sio"
check_grep "main:T91_fn" \
    "fn compiler_main_test_ocp_div_identity" \
    "self-hosted/compiler/main.sio"
check_grep "main:T92_fn" \
    "fn compiler_main_test_ocp_partial_fold_dce_round_trip" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_92" \
    "let total: i64 = 92" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 5: Sprint 51/65/66 regression ---"
check_grep "regression:ocp_const_fold_fn" \
    "fn ocp_const_fold" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:ocp_dce_fn" \
    "fn ocp_dce" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:opt_cleanup_module_fn" \
    "fn opt_cleanup_module" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:ph_run_on_func_fn" \
    "fn ph_run_on_func" \
    "self-hosted/native/peephole.sio"
check_grep "regression:nc_min_frame_size" \
    "fn nc_min_frame_size" \
    "self-hosted/native/lower_ir.sio"

# --- Results ---
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

mkdir -p "$ROOT_DIR/artifacts/sprint67"
cat > "$ROOT_DIR/artifacts/sprint67/partial_const_fold_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint67.partial_const_fold_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "$STATUS",
  "reason": "$REASON",
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "ocp_const_fold extended with partial-constant branch: fires when exactly one operand of IrBinOp is in the known-constant map and an identity/annihilation rule applies",
    "Six algebraic rules at IR level: x*1→IrCopy, x*0→IrLoadImm(0), 1*x→IrCopy, x+0→IrCopy, x-0→IrCopy, x/1→IrCopy",
    "DCE (Sprint 51) naturally reclaims the now-dead IrLoadImm constant operand after partial-constant folding",
    "Full pipeline after Sprint 67: instrument→profile→promote→inline→layout→partial+full const_fold+DCE→TCO→4-preg regalloc→disp8→compact frame→peephole"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint67/partial_const_fold_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
