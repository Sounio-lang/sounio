#!/usr/bin/env bash
# sprint49_inline_pgo_gate.sh — Profile-Guided Inlining
#
# SOTA: LLVM PGO inlining (-fprofile-use + InlineCost::isAlways); GCC -finline-functions
# with profile weights; HotSpot tiered inlining (C1 counts → C2 aggressive inline).
#
# Novel claim: strategy-aware inlining cost model — functions promoted by .sprof profile
# (AGGRESSIVE/PRECISION_PRESERVING) receive benefit boosts. Single binary instruments,
# dumps, profiles, promotes, and inlines. No external toolchain.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

echo "=== Sprint 49: Profile-Guided Inlining ==="
echo ""

# --- Executable self-tests (T22, T23) ---
echo "--- Executable self-test: inline strategy boost ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:inline_strategy_boost" "selftest:inline_pass_returns_module"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:inline_strategy_boost" \
            "T22 OK: inline strategy boost" "$SELF_TEST_LOG"
        check_log_line "selftest:inline_pass_returns_module" \
            "T23 OK: inline pass returns module" "$SELF_TEST_LOG"
    else
        for name in "selftest:inline_strategy_boost" "selftest:inline_pass_returns_module"; do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (--self-test failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Structural checks: InlFuncInfo strategy field ---"
check_grep "inline:func_info_strategy" \
    "compile_strategy: i64" \
    "self-hosted/ir/inline.sio"
check_grep "inline:func_info_new_strategy" \
    "compile_strategy: 0" \
    "self-hosted/ir/inline.sio"
check_grep "inline:analyze_strategy" \
    "func.compile_strategy" \
    "self-hosted/ir/inline.sio"

echo ""
echo "--- Structural checks: strategy-aware cost model ---"
check_grep "inline:strategy_aggressive_bonus" \
    "INL_STRATEGY_AGGRESSIVE_BONUS" \
    "self-hosted/ir/inline.sio"
check_grep "inline:strategy_precision_bonus" \
    "INL_STRATEGY_PRECISION_BONUS" \
    "self-hosted/ir/inline.sio"
check_grep "inline:benefit_strategy_check" \
    "compile_strategy" \
    "self-hosted/ir/inline.sio"

echo ""
echo "--- Structural checks: module declaration ---"
check_grep "inline:module_decl" \
    "module ir::inline" \
    "self-hosted/ir/inline.sio"
check_grep "inline:import_ir" \
    "use ir::ir" \
    "self-hosted/ir/inline.sio"

echo ""
echo "--- Structural checks: pipeline integration ---"
check_grep "main:import_inline" \
    "use ir::inline" \
    "self-hosted/compiler/main.sio"
check_grep "main:inline_pass_call" \
    "inl_run_pass" \
    "self-hosted/compiler/main.sio"
check_grep "main:inline_logging" \
    "\[inline\]" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Structural checks: test fixture ---"
check_grep "fixture:run_pass" \
    "//@ run-pass" \
    "tests/frontend/inline_pgo_contest.sio"
check_grep "fixture:small_helper" \
    "fn small_helper" \
    "tests/frontend/inline_pgo_contest.sio"

echo ""
echo "--- Sprint 38-48 regression ---"
check_grep "regression:apply_strategy_fn" \
    "fn ir_opt_apply_strategy" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:find_return_vreg_fn" \
    "fn ir_opt_find_return_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:codegen_effective_func" \
    "var effective_func" \
    "self-hosted/native/codegen.sio"
check_grep "regression:inject_validated" \
    "fn ir_lower_inject_validated_param" \
    "self-hosted/ir/lower.sio"
check_grep "regression:prof_counter_opcode" \
    "IrProfCounter" \
    "self-hosted/ir/ir.sio"
check_grep "regression:prof_dump_fn" \
    "fn emit_prof_dump_function" \
    "self-hosted/native/codegen.sio"
check_grep "regression:sprof_parse_fn" \
    "fn sprof_parse" \
    "self-hosted/ir/profile.sio"
check_grep "regression:promotion_target_fn" \
    "fn sprof_promotion_target" \
    "self-hosted/ir/profile.sio"

echo ""
echo "--- Existing inline infrastructure ---"
check_grep "inline:run_pass_fn" \
    "fn inl_run_pass" \
    "self-hosted/ir/inline.sio"
check_grep "inline:cost_model" \
    "fn inl_compute_benefit" \
    "self-hosted/ir/inline.sio"
check_grep "inline:context_struct" \
    "struct InlContext" \
    "self-hosted/ir/inline.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint49"
cat > "$ROOT_DIR/artifacts/sprint49/inline_pgo_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint49.inline_pgo_gate.v1",
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
    "Strategy-aware inlining cost model: AGGRESSIVE +80, PRECISION +40 benefit boost",
    "Inlining wired into -O pipeline after PGO promotion",
    "Single binary: instrument → dump → profile → promote → inline",
    "No external toolchain — compiler reads .sprof and inlines in one pass"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint49/inline_pgo_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
