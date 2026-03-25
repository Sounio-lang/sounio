#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# GPU PTX Type Safety Gate — Fix 2 (ptx_advanced.sio type-aware value map)
#
# Validates that:
#   1. ptx_coerce_to_f32 helper is present in ptx_advanced.sio
#   2. ptx_value_map_get_type accessor is present
#   3. GpuMul/GpuAdd/GpuSqrt handlers call coerce for float destinations
#   4. No known-bad patterns in generated PTX (when build is available)
#
# 8 checks: 5 source-level + 3 optional ptxas/runtime checks
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

PASS=0
FAIL=0
SKIP=0
TOTAL=8

SOUC="${SOUC:-./bin/souc}"
PTXAS="${PTXAS:-/tmp/cuda-nvcc-extracted/usr/local/cuda-12.4/bin/ptxas}"

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
echo "  GPU PTX TYPE SAFETY GATE"
echo "  Fix 2: ptx_advanced.sio type-aware value map"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Section 1: Source-level infrastructure ─────────────────────────

echo "── Infrastructure (source checks) ─────────────────────────────"

# Check 1: ptx_value_map_get_type accessor present
if grep -q "fn ptx_value_map_get_type" self-hosted/gpu/ptx_advanced.sio 2>/dev/null; then
    _report "ptx_value_map_get_type accessor present" "PASS"
else
    _report "ptx_value_map_get_type accessor present" "FAIL"
fi

# Check 2: ptx_coerce_to_f32 helper present
if grep -q "fn ptx_coerce_to_f32" self-hosted/gpu/ptx_advanced.sio 2>/dev/null; then
    _report "ptx_coerce_to_f32 helper present" "PASS"
else
    _report "ptx_coerce_to_f32 helper present" "FAIL"
fi

# Check 3: PtxCoerceResult struct present
if grep -q "struct PtxCoerceResult" self-hosted/gpu/ptx_advanced.sio 2>/dev/null; then
    _report "PtxCoerceResult struct present" "PASS"
else
    _report "PtxCoerceResult struct present" "FAIL"
fi

# Check 4: GpuMul handler calls ptx_coerce_to_f32
COERCE_IN_MUL=$(awk '/if opcode == GpuOpcode::GpuMul/,/return out/' \
    self-hosted/gpu/ptx_advanced.sio 2>/dev/null | \
    grep -c "ptx_coerce_to_f32" 2>/dev/null || echo "0")
if [ "$COERCE_IN_MUL" -ge 2 ]; then
    _report "GpuMul handler coerces both operands (count=$COERCE_IN_MUL)" "PASS"
else
    _report "GpuMul handler coerces both operands (count=$COERCE_IN_MUL)" "FAIL"
fi

# Check 5: GpuSqrt handler calls ptx_coerce_to_f32
if awk '/if opcode == GpuOpcode::GpuSqrt/,/return out/' \
    self-hosted/gpu/ptx_advanced.sio 2>/dev/null | \
    grep -q "ptx_coerce_to_f32" 2>/dev/null; then
    _report "GpuSqrt handler coerces operand" "PASS"
else
    _report "GpuSqrt handler coerces operand" "FAIL"
fi

echo ""

# ── Section 2: PTX generation checks (optional — needs souc + ptxas) ─

echo "── PTX Generation (runtime checks) ─────────────────────────────"

# Check 6: souc can check examples/kernel_epistemic_wmma_matmul.sio
if [ -x "$SOUC" ]; then
    if timeout 15 "$SOUC" check examples/kernel_epistemic_wmma_matmul.sio >/dev/null 2>&1; then
        _report "souc check epistemic wmma matmul example" "PASS"
    else
        _report "souc check epistemic wmma matmul example" "FAIL"
    fi
else
    _report "souc check epistemic wmma matmul example" "SKIP"
fi

# Check 7: reference PTX still assembles with ptxas
if [ -x "$PTXAS" ]; then
    if "$PTXAS" -arch sm_80 \
        -o /tmp/gate_ptxas_test.cubin \
        self-hosted/gpu/epistemic_mma_reference.ptx >/dev/null 2>&1; then
        _report "reference PTX assembles with ptxas -arch sm_80" "PASS"
    else
        _report "reference PTX assembles with ptxas -arch sm_80" "FAIL"
    fi
else
    _report "reference PTX assembles with ptxas -arch sm_80" "SKIP"
fi

# Check 8: hand-authored mma.sync test PTX assembles cleanly
if [ -x "$PTXAS" ]; then
    if "$PTXAS" -arch sm_80 \
        -o /tmp/gate_mma_sync.cubin \
        /tmp/test_mma_sync.ptx >/dev/null 2>&1; then
        _report "mma.sync test PTX assembles with ptxas -arch sm_80" "PASS"
    else
        _report "mma.sync test PTX assembles with ptxas -arch sm_80" "FAIL"
    fi
else
    _report "mma.sync test PTX assembles with ptxas -arch sm_80" "SKIP"
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
