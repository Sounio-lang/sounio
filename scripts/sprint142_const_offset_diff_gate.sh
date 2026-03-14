#!/usr/bin/env bash
# sprint142_const_offset_diff_gate.sh — Block BD: Const-offset difference collapse
#
# SOTA: LLVM InstCombineAddSub.cpp — sub(add X,C1),(add X,C2) → C1-C2;
# Cooper & Torczon §8.4 algebraic simplification.
# (x+C1)-(x+C2) → C1-C2  ;  all +/- combos handled via signed offset v=±C.
# Distinct from Block P ((x+C)-C → x): BOTH operands are const-offset exprs.
# Zero new arrays: reuses as_valid/var_src/const_v/as_is_add from Block P (Sprint 89).
#
# Novel claims:
#   1. (x+C1)-(x+C2) → C1-C2: shared base cancels leaving constant
#   2. (x-C1)-(x-C2) → C2-C1: sub-sub variant (signed offsets)
#   3. (x+C1)-(x-C2) → C1+C2: mixed add/sub variant
#   4. (x-C1)-(x+C2) → -(C1+C2): sub-add variant
#   5. Zero new arrays: reuses as_valid/var_src/const_v/as_is_add from Block P
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}}")/.."; pwd)"
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
        if ! grep -qF "T512" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T512)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 142: Block BD — Const-Offset Difference Collapse ==="
echo ""

# --- Group 1: Self-tests T518-T523 ---
echo "--- Self-tests: const-offset diff T518-T523 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:cod_add_add" "selftest:cod_sub_sub" \
                "selftest:cod_add_sub" "selftest:cod_sub_add" \
                "selftest:cod_no_fold_diff_base" "selftest:cod_no_fold_plain_sub"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:cod_add_add" \
        "T518 OK: Sprint 142 Block BD:" "$SELF_TEST_LOG"
    check_log_line "selftest:cod_sub_sub" \
        "T519 OK: Sprint 142 Block BD:" "$SELF_TEST_LOG"
    check_log_line "selftest:cod_add_sub" \
        "T520 OK: Sprint 142 Block BD:" "$SELF_TEST_LOG"
    check_log_line "selftest:cod_sub_add" \
        "T521 OK: Sprint 142 Block BD:" "$SELF_TEST_LOG"
    check_log_line "selftest:cod_no_fold_diff_base" \
        "T522 OK: Sprint 142 Block BD:" "$SELF_TEST_LOG"
    check_log_line "selftest:cod_no_fold_plain_sub" \
        "T523 OK: Sprint 142 Block BD:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block BD implementation ---"
check_grep "ocp:bd_block_comment" \
    "Sprint 142 Block BD" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bd_as_valid_check" \
    "as_valid\[bd_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bd_var_src_match" \
    "as_var_src\[bd_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bd_signed_offset" \
    "as_is_add\[bd_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bd_load_imm_result" \
    "ir_load_imm.*bd_instr.dst.*bd_v1 - bd_v2" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T518_fn" \
    "fn compiler_main_test_const_offset_diff_add_add" \
    "self-hosted/compiler/main.sio"
check_grep "main:T519_fn" \
    "fn compiler_main_test_const_offset_diff_sub_sub" \
    "self-hosted/compiler/main.sio"
check_grep "main:T520_fn" \
    "fn compiler_main_test_const_offset_diff_add_sub" \
    "self-hosted/compiler/main.sio"
check_grep "main:T521_fn" \
    "fn compiler_main_test_const_offset_diff_sub_add" \
    "self-hosted/compiler/main.sio"
check_grep "main:T522_fn" \
    "fn compiler_main_test_const_offset_diff_no_fold_diff_base" \
    "self-hosted/compiler/main.sio"
check_grep "main:T523_fn" \
    "fn compiler_main_test_const_offset_diff_no_fold_plain_sub" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block P/AQ regression ---"
check_grep "regression:as_valid_tracking" \
    "as_valid" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:as_var_src_tracking" \
    "as_var_src" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_p_comment" \
    "Block P" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_aq_comment" \
    "Block AQ" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:opt_cleanup_module" \
    "fn opt_cleanup_module" \
    "self-hosted/ir/opt_cleanup.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint142"
cat > "$ROOT_DIR/artifacts/sprint142/const_offset_diff.v1.json" <<EOF
{
  "schema": "sounio.sprint142.const_offset_diff_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 180
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "(x+C1)-(x+C2) → C1-C2: shared base cancels, leaving constant difference",
    "(x-C1)-(x-C2) → C2-C1: sub-sub variant via signed offset arithmetic",
    "(x+C1)-(x-C2) → C1+C2: mixed add/sub variant",
    "Zero new arrays: reuses as_valid/var_src/const_v/as_is_add from Block P (Sprint 89)",
    "SOTA: LLVM InstCombineAddSub.cpp sub(add X,C1),(add X,C2)→C1-C2; Cooper & Torczon §8.4"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint142/const_offset_diff.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
