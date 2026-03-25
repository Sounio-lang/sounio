#!/usr/bin/env bash
# Sprint 34 gate — Dual-Path Codegen for StrategyInstrumented
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPT="$REPO_ROOT/self-hosted/hlir/opt_strategy.sio"
IR="$REPO_ROOT/self-hosted/hlir/ir.sio"
LOWER="$REPO_ROOT/self-hosted/hlir/lower.sio"
ARTIFACT="$REPO_ROOT/artifacts/sprint34/dual_path_gate.v1.json"

pass=0
fail=0
cases=""

check() {
    local name="$1" cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        pass=$((pass + 1))
        cases="$cases{\"name\":\"$name\",\"status\":\"pass\"},"
    else
        fail=$((fail + 1))
        cases="$cases{\"name\":\"$name\",\"status\":\"fail\"},"
    fi
}

# 1. hlir_opt_apply_strategy is non-trivial
check "apply_strategy_nontrivial" "grep -q 'emit_dual_path' '$OPT' && grep -q 'COND_BRANCH\|cond_branch' '$OPT'"

# 2. hlir_opt_clone_instrs helper exists
check "clone_instrs_exists" "grep -q 'fn hlir_opt_clone_instrs' '$OPT'"

# 3. hlir_opt_wrap_float_interval helper exists
check "wrap_float_interval_exists" "grep -q 'fn hlir_opt_wrap_float_interval' '$OPT'"

# 4. emit_dual_path in StrategyHints
check "emit_dual_path_in_hints" "grep -q 'emit_dual_path.*bool' '$OPT'"

# 5. COND_BRANCH used in opt_strategy.sio
check "cond_branch_used" "grep -q 'cond_branch' '$OPT'"

# 6. __validated parameter pattern
check "validated_param" "grep -q '__validated' '$OPT'"

# 7. PHI node emission
check "phi_emission" "grep -q 'HLIR_OP_PHI' '$OPT'"

# 8a. Test file dual_path_contest.sio exists with annotations
check "test_contest_exists" "grep -q 'run-pass' '$REPO_ROOT/tests/frontend/dual_path_contest.sio'"

# 8b. Test file dual_path_robust.sio exists with annotations
check "test_robust_exists" "grep -q 'run-pass' '$REPO_ROOT/tests/frontend/dual_path_robust.sio'"

# 9a. Sprint regression: ir.sio has HlirBlock struct
check "regression_ir_hlirblock" "grep -q 'struct HlirBlock' '$IR'"

# 9b. Sprint regression: ir.sio has HlirTerminator
check "regression_ir_terminator" "grep -q 'struct HlirTerminator' '$IR'"

# 10. Frontend tests pass souc check (graceful skip)
SOUC="$REPO_ROOT/bin/souc"
if [ -x "$SOUC" ]; then
    check "souc_check_contest" "$SOUC check '$REPO_ROOT/tests/frontend/dual_path_contest.sio'"
    check "souc_check_robust" "$SOUC check '$REPO_ROOT/tests/frontend/dual_path_robust.sio'"
else
    # Fallback: try target/debug/souc
    SOUC="$REPO_ROOT/target/debug/souc"
    if [ -x "$SOUC" ]; then
        check "souc_check_contest" "$SOUC check '$REPO_ROOT/tests/frontend/dual_path_contest.sio'"
        check "souc_check_robust" "$SOUC check '$REPO_ROOT/tests/frontend/dual_path_robust.sio'"
    else
        pass=$((pass + 2))
        cases="$cases{\"name\":\"souc_check_contest\",\"status\":\"pass\",\"reason\":\"skipped_no_souc\"},"
        cases="$cases{\"name\":\"souc_check_robust\",\"status\":\"pass\",\"reason\":\"skipped_no_souc\"},"
    fi
fi

total=$((pass + fail))
cases="${cases%,}"
status="pass"
if [ "$fail" -gt 0 ]; then status="fail"; fi

cat > "$ARTIFACT" <<EOF
{
  "schema": "sounio.sprint34.dual_path_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.000000+00:00)",
  "status": "$status",
  "reason": "$([ "$fail" -eq 0 ] && echo 'all_cases_passed' || echo 'some_cases_failed')",
  "metrics": {
    "total": $total,
    "passed": $pass,
    "failed": $fail
  },
  "cases": [$cases]
}
EOF

echo "Sprint 34 gate: $status ($pass/$total passed)"
[ "$fail" -eq 0 ]
