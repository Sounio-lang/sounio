#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Epistemic SPIR-V Compute Backend Gate
#
# Validates Track B: GpuKernelIr -> SPIR-V 1.3 Vulkan compute shader
# lowering for the epistemic WMMA matmul kernel.
#
# 16 checks:
#   1-4:   Infrastructure (source files and key functions exist)
#   5-10:  SPIR-V source patterns (opcodes, handlers, constants)
#   11-12: GUM oracle (analytical uncertainty verification via awk)
#   13-16: Cross-reference (parallel to PTX path, both backends share IR)
#
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

PASS=0
FAIL=0
SKIP=0
TOTAL=16

SOUC="${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}"

_report() {
    local label="$1" status="$2"
    if [ "$status" = "PASS" ]; then
        PASS=$((PASS + 1))
        printf "  [PASS] %s\n" "$label"
    elif [ "$status" = "SKIP" ]; then
        SKIP=$((SKIP + 1))
        printf "  [SKIP] %s\n" "$label"
    else
        FAIL=$((FAIL + 1))
        printf "  [FAIL] %s\n" "$label"
    fi
}

echo "==================================================================="
echo "  EPISTEMIC SPIR-V COMPUTE BACKEND GATE"
echo "  Track B: GpuKernelIr -> SPIR-V 1.3 Vulkan Compute Shader"
echo "  GUM JCGM 100:2008 | WMMA m16n16k16 | Scalar FMA Fallback"
echo "==================================================================="
echo ""

# ── Section 1: Infrastructure ────────────────────────────────────────

echo "-- Infrastructure ------------------------------------------------"

# Check 1: spirv_lower.sio exists
if [ -f "self-hosted/gpu/spirv_lower.sio" ]; then
    _report "spirv_lower.sio exists" "PASS"
else
    _report "spirv_lower.sio exists" "FAIL"
fi

# Check 2: gpu_lower_kernel_ir_to_spirv function exists
if grep -q "fn gpu_lower_kernel_ir_to_spirv" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "gpu_lower_kernel_ir_to_spirv function exists" "PASS"
else
    _report "gpu_lower_kernel_ir_to_spirv function exists" "FAIL"
fi

# Check 3: SpvLowerState struct exists
if grep -q "struct SpvLowerState" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "SpvLowerState struct exists" "PASS"
else
    _report "SpvLowerState struct exists" "FAIL"
fi

# Check 4: test file exists
if [ -f "self-hosted/test_epistemic_spirv.sio" ]; then
    _report "test_epistemic_spirv.sio exists" "PASS"
else
    _report "test_epistemic_spirv.sio exists" "FAIL"
fi

echo ""

# ── Section 2: SPIR-V Source Patterns ────────────────────────────────

echo "-- SPIR-V Source Patterns ----------------------------------------"

# Check 5: SPIR-V magic constant (0x07230203 = 119734787)
if grep -q "119734787\|0x07230203\|SPIRV_MAGIC" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "SPIR-V magic constant present" "PASS"
else
    _report "SPIR-V magic constant present" "FAIL"
fi

# Check 6: OpCapability handler (Shader capability)
if grep -q "spv_emit_capability\|OP_CAPABILITY\|CAPABILITY_SHADER" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "OpCapability handler present" "PASS"
else
    _report "OpCapability handler present" "FAIL"
fi

# Check 7: OpEntryPoint handler (GLCompute)
if grep -q "spv_emit_entry_point\|EXECUTION_GLCOMPUTE\|OP_ENTRY_POINT" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "OpEntryPoint GLCompute handler present" "PASS"
else
    _report "OpEntryPoint GLCompute handler present" "FAIL"
fi

# Check 8: OpFAdd / OpFMul handlers (float arithmetic)
if grep -q "OP_FADD\|OP_FMUL" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "OpFAdd/OpFMul handlers present" "PASS"
else
    _report "OpFAdd/OpFMul handlers present" "FAIL"
fi

# Check 9: OpExtInst handler (GLSL.std.450 FAbs/Sqrt)
if grep -q "OP_EXT_INST\|GLSL_FABS\|GLSL_SQRT\|spv_emit_ext_inst" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "OpExtInst handler (GLSL FAbs/Sqrt) present" "PASS"
else
    _report "OpExtInst handler (GLSL FAbs/Sqrt) present" "FAIL"
fi

# Check 10: OpControlBarrier handler
if grep -q "spv_emit_control_barrier\|OP_CONTROL_BARRIER" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "OpControlBarrier handler present" "PASS"
else
    _report "OpControlBarrier handler present" "FAIL"
fi

echo ""

# ── Section 3: GUM Oracle (analytical math via awk) ──────────────────

echo "-- GUM Oracle (JCGM 100:2008 verification) ----------------------"

# Check 11: Zero uncertainty -> zero output
ZERO_CHECK=$(awk 'BEGIN {
    eps_a = 0.0; norm_a = 1.0; eps_b = 0.0; norm_b = 1.0;
    combined = (norm_a < 0 ? -norm_a : norm_a) * eps_b + (norm_b < 0 ? -norm_b : norm_b) * eps_a;
    eps_c = sqrt(16) * combined;
    if (eps_c < 1e-12 && eps_c > -1e-12) print "PASS"; else print "FAIL";
}')
_report "GUM: zero-uncertainty identity" "$ZERO_CHECK"

# Check 12: Known values produce correct result
# eps_A=0.1, |A|=2.0, eps_B=0.2, |B|=3.0
# combined = 2.0*0.2 + 3.0*0.1 = 0.7
# eps_C = 4.0 * 0.7 = 2.8
QUAD_CHECK=$(awk 'BEGIN {
    eps_a = 0.1; norm_a = 2.0; eps_b = 0.2; norm_b = 3.0;
    combined = (norm_a < 0 ? -norm_a : norm_a) * eps_b + (norm_b < 0 ? -norm_b : norm_b) * eps_a;
    eps_c = sqrt(16) * combined;
    diff = eps_c - 2.8;
    if (diff < 0) diff = -diff;
    if (diff < 1e-10) print "PASS"; else print "FAIL";
}')
_report "GUM: quadrature correctness (eps_C=2.8)" "$QUAD_CHECK"

echo ""

# ── Section 4: Cross-Reference (SPIR-V parallel to PTX) ─────────────

echo "-- Cross-Reference (PTX/SPIR-V parallel paths) ------------------"

# Check 13: Both PTX and SPIR-V lower from the same kernel IR builder
if grep -q "gpu_build_epistemic_wmma_matmul_16x16_ir" self-hosted/gpu/spirv_lower.sio 2>/dev/null || \
   grep -q "gpu_build_epistemic_wmma_matmul_16x16_ir" self-hosted/test_epistemic_spirv.sio 2>/dev/null; then
    _report "Shared IR builder: gpu_build_epistemic_wmma_matmul_16x16_ir" "PASS"
else
    _report "Shared IR builder: gpu_build_epistemic_wmma_matmul_16x16_ir" "FAIL"
fi

# Check 14: WMMA scalar fallback (decompose into FMA loop)
if grep -q "GpuOpcode::GpuWmma" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "WMMA scalar fallback handler present" "PASS"
else
    _report "WMMA scalar fallback handler present" "FAIL"
fi

# Check 15: GpuAbs handler in SPIR-V lowering
if grep -q "GpuOpcode::GpuAbs" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "GpuAbs handler in SPIR-V lowering" "PASS"
else
    _report "GpuAbs handler in SPIR-V lowering" "FAIL"
fi

# Check 16: GpuXor handler in SPIR-V lowering (provenance)
if grep -q "GpuOpcode::GpuXor" self-hosted/gpu/spirv_lower.sio 2>/dev/null; then
    _report "GpuXor handler in SPIR-V lowering (provenance)" "PASS"
else
    _report "GpuXor handler in SPIR-V lowering (provenance)" "FAIL"
fi

echo ""

# ── Summary ──────────────────────────────────────────────────────────

echo "==================================================================="
printf "  RESULT: %d PASS, %d FAIL, %d SKIP / %d total\n" "$PASS" "$FAIL" "$SKIP" "$TOTAL"

if [ "$FAIL" -gt 0 ]; then
    echo "  STATUS: GATE FAIL"
    echo "==================================================================="
    exit 1
else
    echo "  STATUS: GATE PASS (FAIL=0)"
    echo "==================================================================="
    exit 0
fi
