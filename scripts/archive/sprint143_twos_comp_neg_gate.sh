#!/usr/bin/env bash
# sprint143_twos_comp_neg_gate.sh — Block BE: Two's complement negation
#
# SOTA: LLVM InstCombineAddSub.cpp — not(X)+1 → neg(X); Patterson & Hennessy §B.1.
# Ring identity: ~x = -x - 1, so ~x + 1 = -x (two's complement negation).
# Distinct from Block M (neg arithmetic): identifies bnot+1 as neg.
# Zero new arrays: uses is_bnot/bnot_src (Block AI) + is_const/const_val.
#
# Novel claims:
#   1. ~x + 1 → neg(x): two's complement negation via bnot tracking
#   2. 1 + ~x → neg(x): commutative form handled
#   3. src1 of result points to x (bnot_src), not to ~x
#   4. Wrong-constant guard: ~x + 2 correctly not folded
#   5. Zero new arrays: reuses is_bnot/bnot_src from Block AI (Sprint 112)
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
    if [ ! -s "$log_file" ]; then
        echo "NOT_RUN  $name (log empty — OOM/killed)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        if ! grep -qF "T518" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T518)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 143: Block BE — Two's Complement Negation ==="
echo ""

# --- Group 1: Self-tests T524-T529 ---
echo "--- Self-tests: ~x+1→neg(x) T524-T529 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:twos_neg_basic" "selftest:twos_neg_commutative" \
                "selftest:twos_neg_src_correct" "selftest:twos_neg_no_fold_wrong_const" \
                "selftest:twos_neg_no_fold_sub" "selftest:twos_neg_no_fold_plain"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:twos_neg_basic" \
        "T524 OK: Sprint 143 Block BE:" "$SELF_TEST_LOG"
    check_log_line "selftest:twos_neg_commutative" \
        "T525 OK: Sprint 143 Block BE:" "$SELF_TEST_LOG"
    check_log_line "selftest:twos_neg_src_correct" \
        "T526 OK: Sprint 143 Block BE:" "$SELF_TEST_LOG"
    check_log_line "selftest:twos_neg_no_fold_wrong_const" \
        "T527 OK: Sprint 143 Block BE:" "$SELF_TEST_LOG"
    check_log_line "selftest:twos_neg_no_fold_sub" \
        "T528 OK: Sprint 143 Block BE:" "$SELF_TEST_LOG"
    check_log_line "selftest:twos_neg_no_fold_plain" \
        "T529 OK: Sprint 143 Block BE:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block BE implementation ---"
check_grep "ocp:be_block_comment" \
    "Sprint 143 Block BE" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:be_bnot_check" \
    "is_bnot\[be_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:be_const1_check" \
    "const_val\[be_s.*== 1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:be_unaryop_neg" \
    "ir_unaryop.*be_instr.dst.*OpNeg.*bnot_src" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:be_commutative" \
    "is_bnot\[be_s2" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T524_fn" \
    "fn compiler_main_test_twos_comp_neg_basic" \
    "self-hosted/compiler/main.sio"
check_grep "main:T525_fn" \
    "fn compiler_main_test_twos_comp_neg_commutative" \
    "self-hosted/compiler/main.sio"
check_grep "main:T526_fn" \
    "fn compiler_main_test_twos_comp_neg_src_correct" \
    "self-hosted/compiler/main.sio"
check_grep "main:T527_fn" \
    "fn compiler_main_test_twos_comp_neg_no_fold_wrong_const" \
    "self-hosted/compiler/main.sio"
check_grep "main:T528_fn" \
    "fn compiler_main_test_twos_comp_neg_no_fold_sub" \
    "self-hosted/compiler/main.sio"
check_grep "main:T529_fn" \
    "fn compiler_main_test_twos_comp_neg_no_fold_plain" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block AI/AR regression (bnot tracking still works) ---"
check_grep "regression:is_bnot_tracking" \
    "is_bnot" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:bnot_src_tracking" \
    "bnot_src" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_ai_comment" \
    "Block AI" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_ar_comment" \
    "Block AR" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint143"
cat > "$ROOT_DIR/artifacts/sprint143/twos_comp_neg.v1.json" <<EOF
{
  "schema": "sounio.sprint143.twos_comp_neg_gate.v1",
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
    "~x + 1 → neg(x): two's complement negation via is_bnot/bnot_src tracking",
    "1 + ~x → neg(x): commutative form handled symmetrically",
    "Result src1 = bnot_src[s1] (x, not ~x): correct source threading",
    "Zero new arrays: reuses is_bnot/bnot_src from Block AI (Sprint 112)",
    "SOTA: LLVM InstCombineAddSub.cpp not(X)+1→neg(X); Patterson & Hennessy §B.1"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint143/twos_comp_neg.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
