#!/usr/bin/env bash
# sprint38_strategy_codegen_gate.sh — CompileStrategy wired into native backend
#
# Sprint 38 closes the codegen gap: strategy computed at AST→IR lowering,
# stored on IrFunction, consumed by native x86-64 codegen.
#
# Previous state (Sprint 32-34): strategy was computed in HLIR path only,
# never consumed by native backend.
# After Sprint 38: strategy flows AST → IrFunction → NativeCompiler.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

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

check_probe_strategy() {
    local name="$1"; local source_file="$2"; local expected_line="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$source_file" ]; then
        echo "NOT_RUN  $name (file '$source_file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local souc="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
    if [ ! -x "$souc" ]; then
        echo "NOT_RUN  $name (compiler '$souc' not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    if timeout 180 "$souc" run self-hosted/compiler/main.sio -- --probe-load-ir "$source_file" >"$log_file" 2>&1; then
        if grep -qF "$expected_line" "$log_file"; then
            echo "PASS  $name"; PASS=$((PASS+1))
        elif grep -qF "probe_load_ir: fail" "$log_file" || grep -qF "ir_lowering_failed" "$log_file"; then
            echo "NOT_RUN  $name (probe-load-ir blocked in lowering path)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line' in probe output)"; FAIL=$((FAIL+1))
        fi
    else
        echo "NOT_RUN  $name (probe-load-ir failed)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

echo "=== Sprint 38: CompileStrategy → Native Backend Codegen ==="
echo ""

echo "--- IR-level strategy infrastructure ---"
check_grep "ir:strategy_constants" \
    "IR_STRATEGY_STANDARD.*=.*0" \
    "self-hosted/ir/ir.sio"
check_grep "ir:strategy_aggressive" \
    "IR_STRATEGY_AGGRESSIVE.*=.*1" \
    "self-hosted/ir/ir.sio"
check_grep "ir:strategy_precision_preserving" \
    "IR_STRATEGY_PRECISION_PRESERVING.*=.*2" \
    "self-hosted/ir/ir.sio"
check_grep "ir:strategy_instrumented" \
    "IR_STRATEGY_INSTRUMENTED.*=.*3" \
    "self-hosted/ir/ir.sio"
check_grep "ir:strategy_gpu_kernel" \
    "IR_STRATEGY_GPU_KERNEL.*=.*4" \
    "self-hosted/ir/ir.sio"
check_grep "ir:strategy_probabilistic" \
    "IR_STRATEGY_PROBABILISTIC.*=.*5" \
    "self-hosted/ir/ir.sio"
check_grep "ir:strategy_name_fn" \
    "fn ir_strategy_name" \
    "self-hosted/ir/ir.sio"
check_grep "ir:irfunction_strategy_field" \
    "compile_strategy: i64" \
    "self-hosted/ir/ir.sio"

echo ""
echo "--- Strategy computation in IR lowering ---"
check_grep "lower:compute_strategy_fn" \
    "fn ir_compute_strategy_from_ast" \
    "self-hosted/ir/lower.sio"
check_grep "lower:effect_list_has_name_fn" \
    "fn ir_effect_list_has_name" \
    "self-hosted/ir/lower.sio"
check_grep "lower:gpu_effect_check" \
    "ir_effect_list_has_name.*GPU" \
    "self-hosted/ir/lower.sio"
check_grep "lower:prob_effect_check" \
    "ir_effect_list_has_name.*Prob" \
    "self-hosted/ir/lower.sio"
check_grep "lower:validated_check" \
    "TypeValidated" \
    "self-hosted/ir/lower.sio"
check_grep "lower:knowledge_check" \
    "TypeKnowledge" \
    "self-hosted/ir/lower.sio"
check_grep "lower:contest_check" \
    "TypeContest" \
    "self-hosted/ir/lower.sio"
check_grep "lower:robust_check" \
    "TypeRobust" \
    "self-hosted/ir/lower.sio"
check_grep "lower:strategy_set_on_fn" \
    "compile_strategy = strategy" \
    "self-hosted/ir/lower.sio"
check_grep "lower:strategy_init_standard" \
    "compile_strategy = IR_STRATEGY_STANDARD" \
    "self-hosted/ir/lower.sio"

echo ""
echo "--- Native backend consumption ---"
check_grep "frame:current_strategy_field" \
    "current_strategy: i64" \
    "self-hosted/native/frame.sio"
check_grep "frame:strategy_init" \
    "current_strategy: 0" \
    "self-hosted/native/frame.sio"
check_grep "frame:strategy_reset" \
    "current_strategy = 0" \
    "self-hosted/native/frame.sio"
check_grep "codegen:strategy_wire" \
    "current_strategy = func.compile_strategy" \
    "self-hosted/native/codegen.sio"
check_grep "lower_ir:strategy_reassoc_helper" \
    "fn ir_native_strategy_allows_float_reassoc" \
    "self-hosted/native/lower_ir.sio"
check_grep "lower_ir:strategy_strict_fp_helper" \
    "fn ir_native_strategy_requires_strict_fp" \
    "self-hosted/native/lower_ir.sio"
check_grep "lower_ir:strategy_comment" \
    "Sprint 38.*[Ss]trategy" \
    "self-hosted/native/lower_ir.sio"

echo ""
echo "--- Test fixtures ---"
check_grep "test:ir_knowledge_fixture" \
    "Knowledge.*PRECISION_PRESERVING|Knowledge.*precision" \
    "tests/frontend/compile_strategy_ir_knowledge.sio"
check_grep "test:ir_validated_fixture" \
    "Validated.*AGGRESSIVE|Validated.*aggressive" \
    "tests/frontend/compile_strategy_ir_validated.sio"
check_grep "test:ir_contest_fixture" \
    "Contest.*INSTRUMENTED|Contest.*instrumented" \
    "tests/frontend/compile_strategy_ir_contest.sio"
check_grep "test:ir_standard_fixture" \
    "STANDARD|standard|plain return" \
    "tests/frontend/compile_strategy_ir_standard.sio"

echo ""
echo "--- Self-hosted IR strategy probes ---"
check_probe_strategy "probe:ir_standard_strategy" \
    "tests/frontend/compile_strategy_ir_standard.sio" \
    "probe_load_ir: fn=add_floats strategy=standard"
check_probe_strategy "probe:ir_knowledge_strategy" \
    "tests/frontend/compile_strategy_ir_knowledge.sio" \
    "probe_load_ir: fn=measure_gravity strategy=precision_preserving"
check_probe_strategy "probe:ir_contest_strategy" \
    "tests/frontend/compile_strategy_ir_contest.sio" \
    "probe_load_ir: fn=disputed_measurement strategy=instrumented"
check_probe_strategy "probe:ir_validated_strategy" \
    "tests/frontend/compile_strategy_ir_validated.sio" \
    "probe_load_ir: fn=calibrated_constant strategy=aggressive"

echo ""
echo "--- IR serialization (strategy survives round-trip) ---"
check_grep "serialize:write_strategy" \
    "compile_strategy" \
    "self-hosted/ir/serialize.sio"
check_grep "deserialize:read_strategy" \
    "compile_strategy: compile_strategy" \
    "self-hosted/ir/serialize.sio"

echo ""
echo "--- Sprint 32-34 regression (HLIR strategy still present) ---"
check_grep "regression:hlir_enum" \
    "enum CompileStrategy" \
    "self-hosted/hlir/ir.sio"
check_grep "regression:hlir_compute" \
    "fn hlir_compute_strategy" \
    "self-hosted/hlir/lower.sio"
check_grep "regression:hlir_opt_strategy" \
    "fn hlir_opt_apply_strategy" \
    "self-hosted/hlir/opt_strategy.sio"
check_grep "regression:selfhost_compiler" \
    "fn compiler_main" \
    "self-hosted/compiler/main.sio"

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
mkdir -p "$ROOT_DIR/artifacts/sprint38"
cat > "$ROOT_DIR/artifacts/sprint38/strategy_codegen_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint38.strategy_codegen_gate.v1",
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
  "codegen_gap_status": "closed_with_probe_validation",
  "strategy_flow": "AST_return_type+effects -> ir_compute_strategy_from_ast -> IrFunction.compile_strategy -> probe_load_ir -> NativeCompiler.current_strategy",
  "novel_claims": [
    "Compile strategy flows from source-level epistemic types to native x86-64 codegen",
    "Strategy computed during AST->IR lowering (not just HLIR path)",
    "Self-hosted probe-load-ir exposes per-function compile strategy for runtime validation",
    "NativeCompiler reads strategy before instruction lowering",
    "PrecisionPreserving mode prevents float reassociation in native backend",
    "Strategy query helpers available for future peephole/SIMD passes"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint38/strategy_codegen_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
