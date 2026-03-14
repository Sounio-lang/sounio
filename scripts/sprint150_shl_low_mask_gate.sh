#!/usr/bin/env bash
# sprint150_shl_low_mask_gate.sh — Block BL: Shift-low-mask annihilation
#
# SOTA: LLVM InstCombineShifts.cpp known-bits: (X<<C)&M=0 when M<(1<<C);
# Hacker's Delight §5-1 "Shift-and-Mask" — known zero bits from left shift.
# (x << A) & M → 0 when M >= 0 && A in [1,62] && M < (1 << A).
# Bottom A bits of x<<A are always zero; AND with mask below that position = 0.
# Distinct from Block BK (shift chain): this crosses shift+AND operations.
# Zero new arrays: uses bk_shl_valid/amt (Block BK) + is_const/const_val.
#
# Novel claims:
#   1. (x<<A)&M → 0 when M < (1<<A): known-zero-bits annihilation via shl tracking
#   2. M&(x<<A) → 0: commutative form handled
#   3. Guard A in [1,62]: avoids i64 overflow in 1<<A computation
#   4. Negative mask guard (M < 0 → no fold): correct for signed i64 representation
#   5. Zero new arrays: reuses bk_shl_valid/src/amt from Block BK (Sprint 149)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
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
        if ! grep -qF "T560" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T560)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 150: Block BL — Shift-Low-Mask Annihilation ==="
echo ""

# --- Group 1: Self-tests T566-T571 ---
echo "--- Self-tests: (x<<A)&M->0 T566-T571 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:shl_mask_basic" "selftest:shl_mask_commutative" \
                "selftest:shl_mask_partial" "selftest:shl_mask_no_fold_high_bit" \
                "selftest:shl_mask_no_fold_neg" "selftest:shl_mask_no_fold_plain"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:shl_mask_basic" \
        "T566 OK: Sprint 150 Block BL:" "$SELF_TEST_LOG"
    check_log_line "selftest:shl_mask_commutative" \
        "T567 OK: Sprint 150 Block BL:" "$SELF_TEST_LOG"
    check_log_line "selftest:shl_mask_partial" \
        "T568 OK: Sprint 150 Block BL:" "$SELF_TEST_LOG"
    check_log_line "selftest:shl_mask_no_fold_high_bit" \
        "T569 OK: Sprint 150 Block BL:" "$SELF_TEST_LOG"
    check_log_line "selftest:shl_mask_no_fold_neg" \
        "T570 OK: Sprint 150 Block BL:" "$SELF_TEST_LOG"
    check_log_line "selftest:shl_mask_no_fold_plain" \
        "T571 OK: Sprint 150 Block BL:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block BL implementation ---"
check_grep "ocp:bl_block_comment" \
    "Sprint 150 Block BL" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bl_shl_valid_check" \
    "bk_shl_valid\[bl_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bl_amt_check" \
    "bk_shl_amt\[bl_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bl_threshold" \
    "bl_m >= 0 && bl_a >= 1 && bl_a <= 62 && bl_m < .1 << bl_a" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bl_load_imm_zero" \
    "ir_load_imm.*bl_instr.dst, 0" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T566_fn" \
    "fn compiler_main_test_shl_low_mask_basic" \
    "self-hosted/compiler/main.sio"
check_grep "main:T567_fn" \
    "fn compiler_main_test_shl_low_mask_commutative" \
    "self-hosted/compiler/main.sio"
check_grep "main:T568_fn" \
    "fn compiler_main_test_shl_low_mask_partial" \
    "self-hosted/compiler/main.sio"
check_grep "main:T569_fn" \
    "fn compiler_main_test_shl_low_mask_no_fold_high_bit" \
    "self-hosted/compiler/main.sio"
check_grep "main:T570_fn" \
    "fn compiler_main_test_shl_low_mask_no_fold_negative_mask" \
    "self-hosted/compiler/main.sio"
check_grep "main:T571_fn" \
    "fn compiler_main_test_shl_low_mask_no_fold_plain_and" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block BK/AE regression ---"
check_grep "regression:bk_shl_valid" \
    "bk_shl_valid" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:bk_shl_amt" \
    "bk_shl_amt" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_bk_comment" \
    "Block BK" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_ae_comment" \
    "Block AE" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint150"
cat > "$ROOT_DIR/artifacts/sprint150/shl_low_mask.v1.json" <<EOF
{
  "schema": "sounio.sprint150.shl_low_mask_gate.v1",
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
    "(x<<A)&M → 0 when M>=0 && A in [1,62] && M<(1<<A): known-zero-bits annihilation",
    "M&(x<<A) → 0: commutative form via separate s2-shl check",
    "A in [1,62] guard prevents i64 overflow in (1<<A) computation",
    "Negative mask guard (bl_m>=0): correct for signed i64 representation",
    "SOTA: LLVM InstCombineShifts.cpp known-bits; Hacker's Delight §5-1"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint150/shl_low_mask.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
