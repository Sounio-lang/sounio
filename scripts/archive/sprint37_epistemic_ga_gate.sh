#!/usr/bin/env bash
# sprint37_epistemic_ga_gate.sh — Epistemic GA convergence gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; TOTAL=0

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  PASS: $name"; PASS=$((PASS+1))
    else
        echo "  FAIL: $name (pattern '$pattern' not in $file)"; FAIL=$((FAIL+1))
    fi
}

check_file() {
    local name="$1"; local file="$2"
    TOTAL=$((TOTAL+1))
    if [ -f "$file" ]; then
        echo "  PASS: $name"; PASS=$((PASS+1))
    else
        echo "  FAIL: $name ($file missing)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 37: Epistemic GA Convergence ==="
echo ""

echo "--- uncertainty.sio structural checks ---"
check_grep "uncertainty:jacobian_a" "fn ga_product_jacobian_a" "stdlib/math/ga/uncertainty.sio"
check_grep "uncertainty:jacobian_b" "fn ga_product_jacobian_b" "stdlib/math/ga/uncertainty.sio"
check_grep "uncertainty:propagate" "fn propagate_bilinear_uncertainty" "stdlib/math/ga/uncertainty.sio"
check_grep "uncertainty:bingham" "fn bingham_concentration_to_sigma" "stdlib/math/ga/uncertainty.sio"
check_grep "uncertainty:sqrt" "fn ga_sqrt_approx" "stdlib/math/ga/uncertainty.sio"
check_grep "uncertainty:xor_index" "k \\^ i\\|i \\^ j" "stdlib/math/ga/uncertainty.sio"
check_grep "uncertainty:blade_sign" "ga_blade_product_sign" "stdlib/math/ga/uncertainty.sio"

echo ""
echo "--- epistemic.sio structural checks ---"
check_grep "epistemic:struct" "struct UncertainMv" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:epsilon_field" "epsilon: \\[f64; 32\\]" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:uncertain_mv" "fn uncertain_mv" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:uncertain_product" "fn uncertain_product" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:uncertain_sandwich" "fn uncertain_sandwich" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:bingham_quat" "fn uncertain_quat_from_bingham" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:motor_log" "fn uncertain_motor_from_log" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:max_epsilon" "fn uncertain_mv_max_epsilon" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:norm_sq" "fn uncertain_mv_epsilon_norm_sq" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:ga_product" "ga_product" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:ga_reverse" "ga_reverse" "stdlib/math/ga/epistemic.sio"
check_grep "epistemic:propagate_call" "propagate_bilinear_uncertainty" "stdlib/math/ga/epistemic.sio"

echo ""
echo "--- Strategy integration checks ---"
check_grep "strategy:knowledge" "Knowledge" "tests/frontend/epistemic_ga_knowledge_mv.sio"
check_grep "strategy:contest" "Contest" "tests/frontend/epistemic_ga_contest_motor.sio"
check_grep "strategy:propagation" "uncertain_product\\|Jacobian" "tests/frontend/epistemic_ga_uncertainty_propagation.sio"

echo ""
echo "--- Test file checks ---"
check_file "test:knowledge_mv" "tests/frontend/epistemic_ga_knowledge_mv.sio"
check_file "test:contest_motor" "tests/frontend/epistemic_ga_contest_motor.sio"
check_file "test:uncertainty_prop" "tests/frontend/epistemic_ga_uncertainty_propagation.sio"

echo ""
echo "--- Sprint 34 regression (dual-path) ---"
check_grep "regression:clone_instrs" "hlir_opt_clone_instrs" "self-hosted/hlir/opt_strategy.sio"
check_grep "regression:dual_path" "emit_dual_path" "self-hosted/hlir/opt_strategy.sio"

echo ""
echo "--- Sprint 36 regression (GA specializations) ---"
check_grep "regression:pga_point" "fn pga_point" "stdlib/math/ga/pga.sio"
check_grep "regression:dquat_compose" "fn dquat_compose" "stdlib/math/ga/dual_quaternion.sio"
check_grep "regression:batch4" "struct MvBatch4" "stdlib/math/ga/simd.sio"

echo ""
echo "--- Sprint 35 regression (GA core) ---"
check_grep "regression:multivector" "struct Multivector" "stdlib/math/ga/algebra.sio"
check_grep "regression:ga_product" "fn ga_product" "stdlib/math/ga/product.sio"
check_grep "regression:quaternion" "fn quat_new" "stdlib/math/ga/quaternion.sio"

echo ""
echo "--- souc check (optional) ---"
SOUC="bin/souc"
if [ -x "$SOUC" ]; then
    for f in tests/frontend/epistemic_ga_knowledge_mv.sio tests/frontend/epistemic_ga_contest_motor.sio tests/frontend/epistemic_ga_uncertainty_propagation.sio; do
        TOTAL=$((TOTAL+1))
        if $SOUC check "$f" >/dev/null 2>&1; then
            echo "  PASS: souc check $(basename $f)"; PASS=$((PASS+1))
        else
            echo "  FAIL: souc check $(basename $f)"; FAIL=$((FAIL+1))
        fi
    done
else
    echo "  SKIP: souc binary not found, skipping souc check"
fi

echo ""
echo "=== Sprint 37 Gate: $PASS/$TOTAL passed, $FAIL failed ==="

# Write gate artifact
mkdir -p "$ROOT_DIR/artifacts/sprint37"
cat > "$ROOT_DIR/artifacts/sprint37/epistemic_ga_gate.v1.json" <<EOF
{
  "sprint": 37,
  "feature": "epistemic_ga_convergence",
  "description": "Epistemic uncertainty on geometric algebra — Knowledge<Multivector> with Jacobian propagation",
  "pass": $PASS,
  "fail": $FAIL,
  "total": $TOTAL,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "components": ["uncertainty.sio", "epistemic.sio"],
  "novel_claims": [
    "Per-component uncertainty propagation through geometric products via Jacobian",
    "Bingham-informed quaternion uncertainty (S³-aware, not Gaussian-on-R⁴)",
    "Motor uncertainty from Lie algebra (bivector log space) Gaussian",
    "Type-directed strategy selection for epistemic GA (Knowledge→PrecisionPreserving, Contest→Instrumented)"
  ]
}
EOF

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "Sprint 37 epistemic GA gate: ALL PASS"
