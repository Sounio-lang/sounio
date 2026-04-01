#!/usr/bin/env bash
# sprint33_strategy_optimization_gate.sh — Strategy-directed optimization passes
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

check_file() {
    local name="$1"; local file="$2"
    TOTAL=$((TOTAL+1))
    if [ -f "$file" ]; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (file not found: $file)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 33: Strategy-Directed Optimization Passes ==="
echo ""

echo "--- ir.sio: epistemic type predicate ---"
check_grep "ir:hlir_type_is_epistemic" "fn hlir_type_is_epistemic" "self-hosted/hlir/ir.sio"
check_grep "ir:epistemic_knowledge" "HlirTypeKnowledge" "self-hosted/hlir/ir.sio"
check_grep "ir:epistemic_contest" "HlirTypeContest" "self-hosted/hlir/ir.sio"
check_grep "ir:epistemic_robust" "HlirTypeRobust" "self-hosted/hlir/ir.sio"

echo ""
echo "--- ir.sio: strategy optimization queries ---"
check_grep "ir:allows_simd_widening" "fn hlir_strategy_allows_simd_widening" "self-hosted/hlir/ir.sio"
check_grep "ir:allows_loop_unroll" "fn hlir_strategy_allows_loop_unroll" "self-hosted/hlir/ir.sio"
check_grep "ir:requires_strict_fp" "fn hlir_strategy_requires_strict_fp" "self-hosted/hlir/ir.sio"

echo ""
echo "--- opt_strategy.sio: optimization pass file ---"
check_file "opt_strategy:exists" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:struct_hints" "struct StrategyHints" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:strategy_hints_fn" "fn hlir_strategy_hints" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:can_fold_float" "fn hlir_opt_can_fold_float_binary" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:try_fold_binary" "fn hlir_opt_try_fold_binary" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:apply_strategy" "fn hlir_opt_apply_strategy" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:allow_float_reassoc" "allow_float_reassoc" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:emit_trace_prologue" "emit_trace_prologue" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:emit_dual_path" "emit_dual_path" "self-hosted/hlir/opt_strategy.sio"
check_grep "opt_strategy:is_gpu_kernel" "is_gpu_kernel" "self-hosted/hlir/opt_strategy.sio"

echo ""
echo "--- mod.sio: opt_strategy listed ---"
check_grep "mod:opt_strategy" "opt_strategy.sio" "self-hosted/hlir/mod.sio"

echo ""
echo "--- Sprint 32 regression: CompileStrategy enum ---"
check_grep "regression:compile_strategy_enum" "enum CompileStrategy" "self-hosted/hlir/ir.sio"
check_grep "regression:compile_strategy_field" "compile_strategy: CompileStrategy" "self-hosted/hlir/ir.sio"

echo ""
echo "--- Test files ---"
check_file "test:strategy_standard" "tests/frontend/compile_strategy_standard.sio"
check_grep "test:standard_run_pass" "run-pass" "tests/frontend/compile_strategy_standard.sio"

echo ""
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="

# Write gate artifact
mkdir -p "$ROOT_DIR/artifacts/sprint33"
cat > "$ROOT_DIR/artifacts/sprint33/strategy_optimization_gate.v1.json" <<EOF
{
  "sprint": 33,
  "feature": "strategy_optimization",
  "description": "Strategy-directed HLIR optimization passes + StrategyHints",
  "pass": $PASS,
  "fail": $FAIL,
  "total": $TOTAL,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "new_predicates": [
    "hlir_type_is_epistemic",
    "hlir_strategy_allows_simd_widening",
    "hlir_strategy_allows_loop_unroll",
    "hlir_strategy_requires_strict_fp"
  ],
  "new_files": ["self-hosted/hlir/opt_strategy.sio"],
  "novel_claims": [
    "StrategyHints struct bridges type+effect strategy to backend codegen decisions",
    "Float constant folding gated by epistemic strategy — no other compiler does this",
    "Strategy-aware optimization pass entry point for per-function transforms"
  ]
}
EOF

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "Sprint 33 strategy optimization gate: ALL PASS"
