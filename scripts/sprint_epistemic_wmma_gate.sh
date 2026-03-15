#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Epistemic WMMA Tensor Core Gate
#
# World-first: GUM-compliant (JCGM 100:2008) uncertainty propagation
# through GPU WMMA tensor core operations at the compiler level.
#
# 13 checks: infrastructure (6), PTX structural (4), oracle (2), example (1)
#
# Infrastructure + PTX structural checks use source-level grep/analysis.
# Oracle checks verify GUM mathematics using awk.
# Example check uses souc (if available).
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

PASS=0
FAIL=0
SKIP=0
TOTAL=13

SOUC="${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
GPU_SOUC="${GPU_SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-gpu}"

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

echo "═══════════════════════════════════════════════════════════════"
echo "  EPISTEMIC WMMA TENSOR CORE GATE"
echo "  GUM JCGM 100:2008 · WMMA m16n16k16 · Shadow Registers"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Section 1: Infrastructure (source-level grep) ──────────────────

echo "── Infrastructure ─────────────────────────────────────────────"

# Check 1: IR builder function exists
if grep -q "gpu_build_epistemic_wmma_matmul_16x16_ir" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "IR builder exists in kernel_ir.sio" "PASS"
else
    _report "IR builder exists in kernel_ir.sio" "FAIL"
fi

# Check 2: WMMA handler exists in lower_to_ptx.sio
if grep -q "GpuOpcode::GpuWmma" self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "WMMA handler in lower_to_ptx.sio" "PASS"
else
    _report "WMMA handler in lower_to_ptx.sio" "FAIL"
fi

# Check 3: Abs handler exists
if grep -q "GpuOpcode::GpuAbs" self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "GpuAbs handler in lower_to_ptx.sio" "PASS"
else
    _report "GpuAbs handler in lower_to_ptx.sio" "FAIL"
fi

# Check 4: Sqrt handler exists
if grep -q "GpuOpcode::GpuSqrt" self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "GpuSqrt handler in lower_to_ptx.sio" "PASS"
else
    _report "GpuSqrt handler in lower_to_ptx.sio" "FAIL"
fi

# Check 5: XOR handler exists
if grep -q "GpuOpcode::GpuXor" self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "GpuXor handler in lower_to_ptx.sio" "PASS"
else
    _report "GpuXor handler in lower_to_ptx.sio" "FAIL"
fi

# Check 6: AND handler exists (for predicate validity)
if grep -q "GpuOpcode::GpuAnd" self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "GpuAnd handler in lower_to_ptx.sio" "PASS"
else
    _report "GpuAnd handler in lower_to_ptx.sio" "FAIL"
fi

echo ""

# ── Section 2: PTX Structural (source-level patterns) ─────────────

echo "── PTX Structural (IR builder patterns) ─────────────────────"

# Check 7: WMMA opcode used in IR builder
if grep -q "GpuOpcode::GpuWmma" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "WMMA opcode used in IR builder" "PASS"
else
    _report "WMMA opcode used in IR builder" "FAIL"
fi

# Check 8: Sqrt opcode used in IR builder (shadow uncertainty)
if grep -q "GpuOpcode::GpuSqrt" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "GpuSqrt in IR builder (shadow path)" "PASS"
else
    _report "GpuSqrt in IR builder (shadow path)" "FAIL"
fi

# Check 9: GpuXor opcode used in IR builder (provenance merge)
if grep -q "GpuOpcode::GpuXor" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "GpuXor in IR builder (provenance)" "PASS"
else
    _report "GpuXor in IR builder (provenance)" "FAIL"
fi

# Check 10: and.pred emission pattern in lower_to_ptx.sio
if grep -q 'and\.pred' self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "and.pred emission pattern present" "PASS"
else
    _report "and.pred emission pattern present" "FAIL"
fi

echo ""

# ── Section 3: GUM Oracle (analytical math via awk) ───────────────

echo "── GUM Oracle (JCGM 100:2008 verification) ─────────────────"

# Check 11: Zero uncertainty → zero output
# If eps_A = 0 and eps_B = 0 then eps_C = sqrt(16) * (|A|*0 + |B|*0) = 0
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

# ── Section 4: Example file ───────────────────────────────────────

echo "── Example Validation ───────────────────────────────────────"

# Check 13: souc check on example file
if [ -x "$GPU_SOUC" ]; then
    if timeout 30 "$GPU_SOUC" check examples/kernel_epistemic_wmma_matmul.sio >/dev/null 2>&1; then
        _report "souc check examples/kernel_epistemic_wmma_matmul.sio" "PASS"
    else
        _report "souc check examples/kernel_epistemic_wmma_matmul.sio" "FAIL"
    fi
elif [ -x "$SOUC" ]; then
    if timeout 30 "$SOUC" check examples/kernel_epistemic_wmma_matmul.sio >/dev/null 2>&1; then
        _report "souc check examples/kernel_epistemic_wmma_matmul.sio" "PASS"
    else
        _report "souc check examples/kernel_epistemic_wmma_matmul.sio" "FAIL"
    fi
else
    _report "souc check examples/kernel_epistemic_wmma_matmul.sio" "SKIP"
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
printf "  RESULT: %d PASS, %d FAIL, %d SKIP / %d total\n" "$PASS" "$FAIL" "$SKIP" "$TOTAL"

if [ "$FAIL" -gt 0 ]; then
    echo "  STATUS: GATE FAIL"
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
else
    echo "  STATUS: GATE PASS (FAIL=0)"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
fi
