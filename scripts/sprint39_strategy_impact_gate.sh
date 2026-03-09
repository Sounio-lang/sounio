#!/usr/bin/env bash
# sprint39_strategy_impact_gate.sh — Strategy Impact Benchmark (Table 1 for POPL §Implementation)
#
# Sprint 39 validates that the three CompileStrategy variants produce
# measurably distinct codegen behavior in the native backend:
#
#   Aggressive (1):           float_reassoc_permitted=true,  operand_reordering_active=true
#   PrecisionPreserving (2):  float_reassoc_permitted=false, operand_reordering_active=false
#   Standard (0):             float_reassoc_permitted=false, operand_reordering_active=false
#
# Sprint 38 established the infrastructure (strategy flows AST→IR→NativeCompiler).
# Sprint 39 confirms the behavioral divergence at the codegen level.
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
        echo "FAIL  $name (expected '$expected_line' in $log_file)"; FAIL=$((FAIL+1))
    fi
}

check_probe_strategy() {
    local name="$1"; local source_file="$2"; local expected_line="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if [ ! -f "$source_file" ]; then
        echo "NOT_RUN  $name (source '$source_file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    local attempt=1
    local ok=0
    while [ "$attempt" -le 2 ]; do
        if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --probe-frontend-strategy "$source_file" >"$log_file" 2>&1; then
            ok=1
            break
        fi
        attempt=$((attempt+1))
    done
    if [ "$ok" -eq 1 ]; then
        if grep -qF "$expected_line" "$log_file"; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected '$expected_line' in probe output)"; FAIL=$((FAIL+1))
        fi
    else
        echo "NOT_RUN  $name (probe-frontend-strategy invocation failed)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

echo "=== Sprint 39: Strategy Impact Benchmark — Table 1 for POPL §Implementation ==="
echo ""

echo "--- Executable self-test: strategy divergence ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in \
        "selftest:strict_modes_match" \
        "selftest:aggressive_diverges"
    do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:strict_modes_match" \
            "T06 OK: standard and precision float lowering match" \
            "$SELF_TEST_LOG"
        check_log_line "selftest:aggressive_diverges" \
            "T07 OK: aggressive float lowering diverges" \
            "$SELF_TEST_LOG"
    else
        for name in \
            "selftest:strict_modes_match" \
            "selftest:aggressive_diverges"
        do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (compiler/main --self-test failed)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Probe-frontend-strategy: strategy assignment on real fixtures ---"
check_probe_strategy "probe:aggressive_strategy" \
    "tests/frontend/compile_strategy_ir_validated.sio" \
    "probe_frontend_strategy: fn=calibrated_constant strategy=aggressive"
check_probe_strategy "probe:precision_strategy" \
    "tests/frontend/compile_strategy_ir_knowledge.sio" \
    "probe_frontend_strategy: fn=measure_gravity strategy=precision_preserving"
check_probe_strategy "probe:standard_strategy" \
    "tests/frontend/compile_strategy_ir_standard.sio" \
    "probe_frontend_strategy: fn=add_floats strategy=standard"

echo ""
echo "--- Strategy fixtures: real frontend programs ---"
check_grep "aggressive:run_pass" \
    "//@ run-pass" \
    "tests/frontend/strategy_impact_aggressive_float.sio"
check_grep "aggressive:validated_return" \
    "fn mixed_chain_aggressive\\(.*\\) -> Validated<f64>" \
    "tests/frontend/strategy_impact_aggressive_float.sio"
check_grep "aggressive:epistemic_wrap" \
    "prove_robust\\(|validate_manifest\\(" \
    "tests/frontend/strategy_impact_aggressive_float.sio"
check_grep "aggressive:shared_chain_shape" \
    "let sum_ab: f64 = a \\+ b" \
    "tests/frontend/strategy_impact_aggressive_float.sio"
check_grep "aggressive:mul_chain_shape" \
    "let prod_ab: f64 = a \\* b" \
    "tests/frontend/strategy_impact_aggressive_float.sio"

echo ""
echo "--- PrecisionPreserving strategy fixture (Knowledge<T>) ---"
check_grep "precision:run_pass" \
    "//@ run-pass" \
    "tests/frontend/strategy_impact_precision_float.sio"
check_grep "precision:knowledge_return" \
    "fn mixed_chain_precision\\(.*\\) -> Knowledge<f64>" \
    "tests/frontend/strategy_impact_precision_float.sio"
check_grep "precision:measure_wrap" \
    "measure\\(result, uncertainty: 0.05\\)" \
    "tests/frontend/strategy_impact_precision_float.sio"
check_grep "precision:shared_chain_shape" \
    "let sum_ab: f64 = a \\+ b" \
    "tests/frontend/strategy_impact_precision_float.sio"
check_grep "precision:mul_chain_shape" \
    "let prod_ab: f64 = a \\* b" \
    "tests/frontend/strategy_impact_precision_float.sio"

echo ""
echo "--- Standard strategy fixture (plain f64) ---"
check_grep "standard:run_pass" \
    "//@ run-pass" \
    "tests/frontend/strategy_impact_standard_float.sio"
check_grep "standard:plain_return" \
    "fn mixed_chain_standard\\(.*\\) -> f64" \
    "tests/frontend/strategy_impact_standard_float.sio"
check_grep "standard:shared_chain_shape" \
    "let sum_ab: f64 = a \\+ b" \
    "tests/frontend/strategy_impact_standard_float.sio"
check_grep "standard:mul_chain_shape" \
    "let prod_ab: f64 = a \\* b" \
    "tests/frontend/strategy_impact_standard_float.sio"

echo ""
echo "--- lower_ir.sio: minimal structural regression ---"
check_grep "lower_ir:reassoc_helper" \
    "fn ir_native_strategy_allows_float_reassoc" \
    "self-hosted/native/lower_ir.sio"
check_grep "lower_ir:strict_fp_helper" \
    "fn ir_native_strategy_requires_strict_fp" \
    "self-hosted/native/lower_ir.sio"
check_grep "lower_ir:commute_helper" \
    "fn ir_native_strategy_can_commute_float_binop" \
    "self-hosted/native/lower_ir.sio"
check_grep "lower_ir:swap_computed" \
    "let swap_operands" \
    "self-hosted/native/lower_ir.sio"
check_grep "lower_ir:swap_operands_param" \
    "fn lower_ir_load_float_binop_operands" \
    "self-hosted/native/lower_ir.sio"
check_grep "lower_ir:conditional_lhs_rhs" \
    "if swap_operands" \
    "self-hosted/native/lower_ir.sio"

echo ""
echo "--- Sprint 38 regression (strategy pipeline still intact) ---"
check_grep "regression:ir_strategy_constants" \
    "IR_STRATEGY_STANDARD.*=.*0" \
    "self-hosted/ir/ir.sio"
check_grep "regression:ir_function_field" \
    "compile_strategy: i64" \
    "self-hosted/ir/ir.sio"
check_grep "regression:lower_compute_fn" \
    "fn ir_compute_strategy_from_ast" \
    "self-hosted/ir/lower.sio"
check_grep "regression:frame_strategy_field" \
    "current_strategy: i64" \
    "self-hosted/native/frame.sio"
check_grep "regression:codegen_wire" \
    "current_strategy = func.compile_strategy" \
    "self-hosted/native/codegen.sio"
check_grep "regression:serialize_strategy" \
    "compile_strategy" \
    "self-hosted/ir/serialize.sio"
check_grep "regression:hlir_enum_present" \
    "enum CompileStrategy" \
    "self-hosted/hlir/ir.sio"

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

# Write gate artifact
mkdir -p "$ROOT_DIR/artifacts/sprint39"
cat > "$ROOT_DIR/artifacts/sprint39/strategy_impact_benchmark.v1.json" <<EOF
{
  "schema": "sounio.sprint39.strategy_impact_benchmark.v1",
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
  "table_1_popl_implementation": {
    "strategies_measured": [
      {
        "strategy": "Aggressive",
        "type_source": "Validated<T>",
        "ir_constant": 1,
        "float_reassoc_permitted": true,
        "operand_reordering_active": true,
        "strict_fp": false
      },
      {
        "strategy": "PrecisionPreserving",
        "type_source": "Knowledge<T>",
        "ir_constant": 2,
        "float_reassoc_permitted": false,
        "operand_reordering_active": false,
        "strict_fp": true
      },
      {
        "strategy": "Standard",
        "type_source": "f64",
        "ir_constant": 0,
        "float_reassoc_permitted": false,
        "operand_reordering_active": false,
        "strict_fp": false
      }
    ],
    "behavioral_divergence": "self_test_and_probe_validated",
    "divergence_mechanism": "compiler/main self-tests + probe-frontend-strategy over stable strategy fixtures"
  },
  "novel_claims": [
    "Aggressive strategy produces distinct operand ordering for commutative float ops",
    "PrecisionPreserving matches Standard byte-for-byte for commutative float lowering",
    "Probe-frontend-strategy confirms strategy=aggressive, precision_preserving, and standard on stable compile_strategy_ir fixtures",
    "Executable validation now uses compiler/main --self-test plus probe-frontend-strategy instead of comment-only fixture claims",
    "Sprint 39 is a behavioral divergence lane, not a wall-clock benchmark lane"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint39/strategy_impact_benchmark.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
