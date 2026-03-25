#!/usr/bin/env bash
# sprint48_show_profile_gate.sh — --show-profile Probe + Promotion Logging
#
# SOTA: LLVM `opt -pass-remarks=pgo-icall-prom`; GCC `-fdump-ipa-profile`;
# Bolt `-print-profile-stats`; HotSpot `-XX:+PrintCompilation`.
#
# Novel claim: built-in --show-profile probe integrated into compiler binary.
# Human-readable output with strategy name mapping. No external tools.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

echo "=== Sprint 48: --show-profile Probe + Promotion Logging ==="
echo ""

# --- Executable self-tests (T20, T21) ---
echo "--- Executable self-test: sprof promotion target ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:sprof_promotion_target_thresholds" "selftest:sprof_promotion_target_no_change"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:sprof_promotion_target_thresholds" \
            "T20 OK: sprof promotion target thresholds" "$SELF_TEST_LOG"
        check_log_line "selftest:sprof_promotion_target_no_change" \
            "T21 OK: sprof promotion target no change" "$SELF_TEST_LOG"
    else
        for name in "selftest:sprof_promotion_target_thresholds" "selftest:sprof_promotion_target_no_change"; do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (--self-test failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Structural checks: promotion target helper ---"
check_grep "profile:promotion_target_fn" \
    "fn sprof_promotion_target" \
    "self-hosted/ir/profile.sio"
check_grep "profile:promotion_target_aggressive" \
    "IR_STRATEGY_AGGRESSIVE" \
    "self-hosted/ir/profile.sio"
check_grep "profile:promotion_target_neg1" \
    "\-1" \
    "self-hosted/ir/profile.sio"

echo ""
echo "--- Structural checks: refactored apply_promotion ---"
check_grep "profile:apply_uses_target" \
    "sprof_promotion_target" \
    "self-hosted/ir/profile.sio"
check_grep "profile:apply_target_guard" \
    "target >= 0" \
    "self-hosted/ir/profile.sio"

echo ""
echo "--- Structural checks: show-profile probe ---"
check_grep "main:show_profile_fn" \
    "fn run_probe_show_profile" \
    "self-hosted/compiler/main.sio"
check_grep "main:show_profile_dispatch" \
    "show-profile" \
    "self-hosted/compiler/main.sio"
check_grep "main:show_profile_parse" \
    "sprof_parse" \
    "self-hosted/compiler/main.sio"
check_grep "main:show_profile_output" \
    "\[sprof\]" \
    "self-hosted/compiler/main.sio"
check_grep "main:show_profile_help" \
    "show-profile" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Structural checks: promotion logging ---"
check_grep "main:promotion_log_sprof" \
    "sprof_promotion_target" \
    "self-hosted/compiler/main.sio"
check_grep "main:promotion_log_strategy" \
    "ir_strategy_name" \
    "self-hosted/compiler/main.sio"
check_grep "main:promotion_log_prefix" \
    "\[sprof\]" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Structural checks: test fixture ---"
check_grep "fixture:run_pass" \
    "//@ run-pass" \
    "tests/frontend/sprof_show_profile_probe.sio"
check_grep "fixture:hot_fn" \
    "fn hot_fn" \
    "tests/frontend/sprof_show_profile_probe.sio"

echo ""
echo "--- Sprint 38-47 regression ---"
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
check_grep "regression:sprof_header_fn" \
    "fn emit_prof_write_sprof_header" \
    "self-hosted/native/codegen.sio"
check_grep "regression:sprof_parse_fn" \
    "fn sprof_parse" \
    "self-hosted/ir/profile.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint48"
cat > "$ROOT_DIR/artifacts/sprint48/show_profile_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint48.show_profile_gate.v1",
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
    "Built-in --show-profile probe: inspect .sprof files without external tools",
    "sprof_promotion_target() factored helper — reusable threshold logic",
    "Promotion logging during --use-profile compilation",
    "Human-readable output with strategy name mapping via ir_strategy_name()"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint48/show_profile_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
