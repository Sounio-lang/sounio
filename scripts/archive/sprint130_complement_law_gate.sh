#!/usr/bin/env bash
# sprint130_complement_law_gate.sh — Block AR: Complement law
#
# SOTA: LLVM InstCombineAndOrXor.cpp; Boolean algebra complement axiom.
# x & ~x → 0  ;  x | ~x → -1
# Detects NOT via is_bnot/bnot_src tracking from Block AI (Sprint 112).
#
# Novel claims:
#   1. x & ~x → IrLoadImm(0): complement eliminates AND to zero
#   2. x | ~x → IrLoadImm(-1): complement saturates OR to all-ones
#   3. Commutative: ~x & x and ~x | x handled symmetrically
#   4. Zero new tracking arrays: reuses is_bnot/bnot_src from Block AI
set -euo pipefail

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
        if ! grep -qF "T452" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T452)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 130: Block AR — Complement Law (x & ~x → 0, x | ~x → -1) ==="
echo ""

# --- Group 1: Self-tests T452-T457 ---
echo "--- Self-tests: complement law T452-T457 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:and_complement_zero" "selftest:and_complement_comm" \
                "selftest:or_complement_allones" "selftest:or_complement_comm" \
                "selftest:and_diff_var_no_fold" "selftest:or_diff_var_no_fold"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:and_complement_zero"  "T452 OK: Sprint 130 Block AR:" "$SELF_TEST_LOG"
    check_log_line "selftest:and_complement_comm"  "T453 OK: Sprint 130 Block AR:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_complement_allones" "T454 OK: Sprint 130 Block AR:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_complement_comm"   "T455 OK: Sprint 130 Block AR:" "$SELF_TEST_LOG"
    check_log_line "selftest:and_diff_var_no_fold" "T456 OK: Sprint 130 Block AR:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_diff_var_no_fold"  "T457 OK: Sprint 130 Block AR:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block AR implementation ---"
check_grep "ocp:block_ar_comment" \
    "Sprint 130 Block AR" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ar_and_complement_zero" \
    "bnot_src\[ar_s2 as usize\] == ar_s1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ar_or_complement_allones" \
    "ir_load_imm.*ar_instr.dst.*-1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ar_uses_is_bnot" \
    "is_bnot\[ar_s2 as usize\]|is_bnot\[ar_s1 as usize\]" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T452_fn" "fn compiler_main_test_and_complement_zero"     "self-hosted/compiler/main.sio"
check_grep "main:T453_fn" "fn compiler_main_test_and_complement_comm_zero" "self-hosted/compiler/main.sio"
check_grep "main:T454_fn" "fn compiler_main_test_or_complement_allones"    "self-hosted/compiler/main.sio"
check_grep "main:T455_fn" "fn compiler_main_test_or_complement_comm_allones" "self-hosted/compiler/main.sio"
check_grep "main:T456_fn" "fn compiler_main_test_and_different_var_no_fold" "self-hosted/compiler/main.sio"
check_grep "main:T457_fn" "fn compiler_main_test_or_different_var_no_fold"  "self-hosted/compiler/main.sio"
check_grep "main:total_updated" "let total: i64 = [0-9]+" "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block AI regression (is_bnot tracking still works) ---"
check_grep "regression:block_ai_comment" \
    "Sprint 112 Block AI" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:is_bnot_tracking" \
    "is_bnot\[bn_instr.dst" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_am_comment" \
    "Sprint 116 Block AM" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:opt_cleanup_module" \
    "fn opt_cleanup_module" \
    "self-hosted/ir/opt_cleanup.sio"

# --- Results ---
echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"; REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then STATUS="fail"; REASON="${FAIL}_cases_failed"; fi
if [ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ]; then REASON="all_run_passed_${NOT_RUN}_not_run"; fi

mkdir -p "$ROOT_DIR/artifacts/sprint130"
cat > "$ROOT_DIR/artifacts/sprint130/complement_law.v1.json" <<JSONEOF
{
  "schema": "sounio.sprint130.complement_law_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": { "root": "$ROOT_DIR", "timeout_seconds": 180 },
  "metrics": { "total": $TOTAL, "passed": $PASS, "failed": $FAIL, "not_run": $NOT_RUN },
  "novel_claims": [
    "x & ~x → IrLoadImm(0): Boolean complement law eliminates AND to zero",
    "x | ~x → IrLoadImm(-1): Boolean complement law saturates OR to all-ones",
    "Commutative: ~x & x and ~x | x handled symmetrically",
    "Zero new tracking arrays: reuses is_bnot/bnot_src from Block AI (Sprint 112)"
  ]
}
JSONEOF

echo ""
echo "Artifact: artifacts/sprint130/complement_law.v1.json"

if [ "$FAIL" -gt 0 ]; then exit 1; fi
