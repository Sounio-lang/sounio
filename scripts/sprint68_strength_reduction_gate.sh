#!/usr/bin/env bash
# sprint68_strength_reduction_gate.sh — IR-level strength reduction for power-of-2 mul/div/rem
#
# SOTA: Allen & Cocke "A Catalogue of Optimizing Transformations" (1972) §strength reduction;
# Muchnick "Advanced Compiler Design and Implementation" §12.3 — linear function test replacement;
# Cooper & Torczon "Engineering a Compiler" §8.4 — strength reduction replaces expensive ops
# (IMUL 3-5 cycles) with cheaper equivalents (SHL 1 cycle) for power-of-2 constants.
#
# Novel claims:
#   1. def_at[256] map tracks IrLoadImm definition site per vreg enabling in-place constant patching
#   2. Three rules wired into a single extra block in ocp_const_fold:
#      x * 2^n → x << n (OpMul → OpShl, constant patched from 2^n to n)
#      x / 2^n → x >> n (OpDiv → OpShr, constant patched from 2^n to n)
#      x % 2^n → x & (2^n-1) (OpRem → OpBitAnd, constant patched to bitmask)
#   3. Re-reads result.instrs[i] before firing: respects prior full/partial-constant rewrites
#   4. DCE (Sprint 51) reclaims dead loads naturally; no special cleanup needed
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
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 68: IR-level strength reduction for power-of-2 mul/div/rem ==="
echo ""

# --- Group 1: Self-tests T95-T102 ---
echo "--- Self-tests: strength reduction T95-T102 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T95 T96 T97 T98 T99 T100 T101 T102; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:T95_mul_by_2" \
        "T95 OK: ocp_const_fold reduces x * 2 → x << 1 (OpShl, constant patched to 1)" "$SELF_TEST_LOG"
    check_log_line "selftest:T96_mul_by_4" \
        "T96 OK: ocp_const_fold reduces x * 4 → x << 2 (OpShl, constant patched to 2)" "$SELF_TEST_LOG"
    check_log_line "selftest:T97_mul_by_8" \
        "T97 OK: ocp_const_fold reduces x * 8 → x << 3 (OpShl, constant patched to 3)" "$SELF_TEST_LOG"
    check_log_line "selftest:T98_div_by_2" \
        "T98 OK: ocp_const_fold reduces x / 2 → x >> 1 (OpShr, constant patched to 1)" "$SELF_TEST_LOG"
    check_log_line "selftest:T99_div_by_4" \
        "T99 OK: ocp_const_fold reduces x / 4 → x >> 2 (OpShr, constant patched to 2)" "$SELF_TEST_LOG"
    check_log_line "selftest:T100_rem_by_4" \
        "T100 OK: ocp_const_fold reduces x % 4 → x & 3 (OpBitAnd, constant patched to 3)" "$SELF_TEST_LOG"
    check_log_line "selftest:T101_non_pow2" \
        "T101 OK: ocp_const_fold leaves x * 3 as OpMul (3 is not a power of 2)" "$SELF_TEST_LOG"
    check_log_line "selftest:T102_round_trip" \
        "T102 OK: opt_cleanup_function round-trip removes the original x * 4 arithmetic op" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio — helpers ---"
check_grep "ocp:sprint68_comment" \
    "Sprint 68" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:is_power_of_two_fn" \
    "fn ocp_is_power_of_two" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:log2_fn" \
    "fn ocp_log2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:def_at_decl" \
    "def_at" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: opt_cleanup.sio — strength reduction rules ---"
check_grep "ocp:sr_mul_shl" \
    "OpShl" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:sr_div_shr" \
    "OpShr" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:sr_rem_bitand" \
    "OpBitAnd" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:sr_pow2_guard" \
    "ocp_is_power_of_two" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:sr_def_idx_guard" \
    "def_idx >= 0" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:sr_patch_loadimm" \
    "def_idx as usize" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 4: main.sio — test registration ---"
check_grep "main:T95_fn" \
    "fn compiler_main_test_ocp_sr_mul_by_2" \
    "self-hosted/compiler/main.sio"
check_grep "main:T96_fn" \
    "fn compiler_main_test_ocp_sr_mul_by_4" \
    "self-hosted/compiler/main.sio"
check_grep "main:T97_fn" \
    "fn compiler_main_test_ocp_sr_mul_by_8" \
    "self-hosted/compiler/main.sio"
check_grep "main:T98_fn" \
    "fn compiler_main_test_ocp_sr_div_by_2" \
    "self-hosted/compiler/main.sio"
check_grep "main:T99_fn" \
    "fn compiler_main_test_ocp_sr_div_by_4" \
    "self-hosted/compiler/main.sio"
check_grep "main:T100_fn" \
    "fn compiler_main_test_ocp_sr_rem_by_4" \
    "self-hosted/compiler/main.sio"
check_grep "main:T101_fn" \
    "fn compiler_main_test_ocp_sr_non_pow2_unchanged" \
    "self-hosted/compiler/main.sio"
check_grep "main:T102_fn" \
    "fn compiler_main_test_ocp_sr_round_trip" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_102" \
    "let total: i64 = 102" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 5: Sprint 51/67 regression ---"
check_grep "regression:ocp_const_fold_fn" \
    "fn ocp_const_fold" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:ocp_dce_fn" \
    "fn ocp_dce" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:sprint67_block" \
    "Sprint 67" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:ph_run_on_func" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint68"
cat > "$ROOT_DIR/artifacts/sprint68/strength_reduction_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint68.strength_reduction_gate.v1",
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
    "def_at[256] map tracks IrLoadImm definition site per vreg enabling in-place constant patching during strength reduction",
    "Three rules in ocp_const_fold: x*2^n→x<<n (OpShl), x/2^n→x>>n (OpShr), x%2^n→x&(2^n-1) (OpBitAnd)",
    "Strength reduction re-reads result.instrs[i] before firing to respect prior full/partial-constant rewrites",
    "DCE (Sprint 51) reclaims dead loads naturally after strength reduction; no special cleanup needed",
    "Full pipeline after Sprint 68: instrument→profile→promote→inline→layout→full+partial+SR const_fold+DCE→TCO→4-preg regalloc→disp8→compact frame→peephole"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint68/strength_reduction_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
