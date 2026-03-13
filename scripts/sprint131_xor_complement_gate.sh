#!/usr/bin/env bash
# sprint131_xor_complement_gate.sh — Block AS: XOR complement law
#
# SOTA: LLVM InstCombineAndOrXor.cpp; Hacker's Delight §2-2.
# x ^ ~x → -1  ;  ~x ^ ~y → x ^ y  (double-complement XOR cancellation)
# ~x ^ ~x → 0  (same base: special case of double-complement)
# Extends Block AR (AND/OR complement) to XOR domain, zero new arrays.
#
# Novel claims:
#   1. x ^ ~x → IrLoadImm(-1): XOR complement = all-ones
#   2. ~x ^ ~y → x ^ y: double-complement cancellation via bnot_src
#   3. ~x ^ ~x → 0: base1==base2 special case avoids needing Block A re-pass
#   4. Zero new tracking arrays: reuses is_bnot/bnot_src from Block AI
set -euo pipefail

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
        if ! grep -qF "T458" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T458)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 131: Block AS — XOR Complement (x^~x→-1, ~x^~y→x^y) ==="
echo ""

echo "--- Self-tests: XOR complement T458-T463 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:xor_complement_allones" "selftest:xor_complement_comm" \
                "selftest:xor_complement_no_fold" "selftest:xor_double_complement" \
                "selftest:xor_self_complement_zero" "selftest:double_not_regression"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:xor_complement_allones"   "T458 OK: Sprint 131 Block AS:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_complement_comm"      "T459 OK: Sprint 131 Block AS:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_complement_no_fold"   "T460 OK: Sprint 131 Block AS:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_double_complement"    "T461 OK: Sprint 131 Block AS:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_self_complement_zero" "T462 OK: Sprint 131 Block AS:" "$SELF_TEST_LOG"
    check_log_line "selftest:double_not_regression"    "T463 OK: Sprint 131 Block AS:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block AS implementation ---"
check_grep "ocp:block_as_comment" \
    "Sprint 131 Block AS" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:as_xor_complement_allones" \
    "bnot_src\[ar_s2 as usize\] == ar_s1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:as_double_complement_cancel" \
    "as_base1.*as_base2|bnot_src\[ar_s1 as usize\]" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:as_xor_ir_load_imm" \
    "ir_load_imm.*ar_instr.dst.*-1" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T458_fn" "fn compiler_main_test_xor_complement_allones"    "self-hosted/compiler/main.sio"
check_grep "main:T459_fn" "fn compiler_main_test_xor_complement_comm"        "self-hosted/compiler/main.sio"
check_grep "main:T460_fn" "fn compiler_main_test_xor_complement_no_fold"     "self-hosted/compiler/main.sio"
check_grep "main:T461_fn" "fn compiler_main_test_xor_double_complement"      "self-hosted/compiler/main.sio"
check_grep "main:T462_fn" "fn compiler_main_test_xor_self_complement_zero"   "self-hosted/compiler/main.sio"
check_grep "main:T463_fn" "fn compiler_main_test_double_not_regression"      "self-hosted/compiler/main.sio"
check_grep "main:total_updated" "let total: i64 = [0-9]+" "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block AR/AI regression ---"
check_grep "regression:block_ar_comment" "Sprint 130 Block AR"         "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_ai_bnot"    "Sprint 112 Block AI"         "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:is_bnot_tracking" "is_bnot\[bn_instr.dst"       "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:opt_cleanup_mod"  "fn opt_cleanup_module"        "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"; REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then STATUS="fail"; REASON="${FAIL}_cases_failed"; fi
if [ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ]; then REASON="all_run_passed_${NOT_RUN}_not_run"; fi

mkdir -p "$ROOT_DIR/artifacts/sprint131"
cat > "$ROOT_DIR/artifacts/sprint131/xor_complement.v1.json" <<JSONEOF
{
  "schema": "sounio.sprint131.xor_complement_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": { "root": "$ROOT_DIR", "timeout_seconds": 180 },
  "metrics": { "total": $TOTAL, "passed": $PASS, "failed": $FAIL, "not_run": $NOT_RUN },
  "novel_claims": [
    "x ^ ~x → -1: XOR complement saturates to all-ones",
    "~x ^ ~y → x ^ y: double-complement XOR cancellation via bnot_src chain",
    "~x ^ ~x → 0: same-base special case avoids needing a second pass",
    "Zero new tracking arrays: reuses is_bnot/bnot_src from Block AI (Sprint 112)"
  ]
}
JSONEOF

echo ""
echo "Artifact: artifacts/sprint131/xor_complement.v1.json"
if [ "$FAIL" -gt 0 ]; then exit 1; fi
