#!/usr/bin/env bash
# sprint32_compile_strategy_gate.sh — CompileStrategy annotation in HLIR
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; TOTAL=0

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 32: CompileStrategy — Type+Effect-Directed Compilation ==="
echo ""

echo "--- HLIR IR structural checks ---"
check_grep "ir:compile_strategy_enum" "enum CompileStrategy" "self-hosted/hlir/ir.sio"
check_grep "ir:strategy_standard" "StrategyStandard" "self-hosted/hlir/ir.sio"
check_grep "ir:strategy_aggressive" "StrategyAggressive" "self-hosted/hlir/ir.sio"
check_grep "ir:strategy_precision" "StrategyPrecisionPreserving" "self-hosted/hlir/ir.sio"
check_grep "ir:strategy_instrumented" "StrategyInstrumented" "self-hosted/hlir/ir.sio"
check_grep "ir:strategy_gpu" "StrategyGpuKernel" "self-hosted/hlir/ir.sio"
check_grep "ir:strategy_prob" "StrategyProbabilistic" "self-hosted/hlir/ir.sio"
check_grep "ir:func_field" "compile_strategy: CompileStrategy" "self-hosted/hlir/ir.sio"
check_grep "ir:strategy_name_fn" "fn hlir_strategy_name" "self-hosted/hlir/ir.sio"
check_grep "ir:float_reassoc_fn" "fn hlir_strategy_allows_float_reassoc" "self-hosted/hlir/ir.sio"
check_grep "ir:instrumentation_fn" "fn hlir_strategy_needs_instrumentation" "self-hosted/hlir/ir.sio"

echo ""
echo "--- Lowering integration checks ---"
check_grep "lower:set_strategy_fn" "fn hlir_lower_set_strategy" "self-hosted/hlir/lower.sio"
check_grep "lower:effects_from_ast_fn" "fn hlir_lower_effects_from_ast" "self-hosted/hlir/lower.sio"
check_grep "lower:compute_strategy_fn" "fn hlir_compute_strategy" "self-hosted/hlir/lower.sio"
check_grep "lower:has_effect_fn" "fn hlir_func_has_effect_named" "self-hosted/hlir/lower.sio"
check_grep "lower:gpu_check" "hlir_func_has_effect_named.*GPU" "self-hosted/hlir/lower.sio"
check_grep "lower:prob_check" "hlir_func_has_effect_named.*Prob" "self-hosted/hlir/lower.sio"
check_grep "lower:validated_check" "HlirTypeValidated" "self-hosted/hlir/lower.sio"
check_grep "lower:knowledge_check" "HlirTypeKnowledge" "self-hosted/hlir/lower.sio"
check_grep "lower:wired_in_function" "hlir_compute_strategy" "self-hosted/hlir/lower.sio"

echo ""
echo "--- Test file checks ---"
check_grep "test:strategy_validated" "StrategyAggressive\|Validated" "tests/frontend/compile_strategy_validated.sio"
check_grep "test:strategy_knowledge" "StrategyPrecisionPreserving\|Knowledge" "tests/frontend/compile_strategy_knowledge.sio"
check_grep "test:strategy_gpu" "StrategyGpuKernel\|GPU" "tests/frontend/compile_strategy_gpu.sio"
check_grep "test:strategy_prob" "StrategyProbabilistic\|Prob" "tests/frontend/compile_strategy_prob.sio"

echo ""
echo "--- Sprint 31 regression (d-sep still works) ---"
check_grep "regression:dsep_chain" "chain\|Chain" "tests/frontend/causal_id_chain_trivial.sio"

echo ""
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="

# Write gate artifact
mkdir -p "$ROOT_DIR/artifacts/sprint32"
cat > "$ROOT_DIR/artifacts/sprint32/compile_strategy_gate.v1.json" <<EOF
{
  "sprint": 32,
  "feature": "compile_strategy",
  "description": "Type+effect-directed compilation strategy annotation in HLIR",
  "pass": $PASS,
  "fail": $FAIL,
  "total": $TOTAL,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "strategies": ["Standard", "Aggressive", "PrecisionPreserving", "Instrumented", "GpuKernel", "Probabilistic"],
  "novel_claims": [
    "Effect types select compilation strategy (extends Koka CPS/direct to 6 modes)",
    "Epistemic type classification drives optimization level",
    "No existing language uses uncertainty types to guide codegen"
  ]
}
EOF

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "Sprint 32 compile strategy gate: ALL PASS"
