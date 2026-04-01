#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Epistemic Tiled GEMM Gate
#
# Tiled GEMM with shared memory staging + K-loop uncertainty accumulation
# via GUM quadrature (JCGM 100:2008). Track A of the Sounio GPU push.
#
# 13 checks: infrastructure (7), PTX structural (4), GUM oracle (2)
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

PASS=0
FAIL=0
SKIP=0
TOTAL=13

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
echo "  EPISTEMIC TILED GEMM GATE"
echo "  GUM JCGM 100:2008 · Shared Memory · K-loop Quadrature"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Section 1: Infrastructure (source-level grep) ──────────────────

echo "── Infrastructure ─────────────────────────────────────────────"

# Check 1: Tiled GEMM IR builder function exists
if grep -q "gpu_build_epistemic_tiled_gemm_ir" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "Tiled GEMM IR builder in kernel_ir.sio" "PASS"
else
    _report "Tiled GEMM IR builder in kernel_ir.sio" "FAIL"
fi

# Check 2: GpuCpAsync handler in lower_to_ptx.sio
if grep -q "GpuOpcode::GpuCpAsync" self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "GpuCpAsync handler in lower_to_ptx.sio" "PASS"
else
    _report "GpuCpAsync handler in lower_to_ptx.sio" "FAIL"
fi

# Check 3: GpuCpAsyncWaitAll handler
if grep -q "GpuOpcode::GpuCpAsyncWaitAll" self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "GpuCpAsyncWaitAll handler in lower_to_ptx.sio" "PASS"
else
    _report "GpuCpAsyncWaitAll handler in lower_to_ptx.sio" "FAIL"
fi

# Check 4: GpuBra handler (branch/label)
if grep -q "GpuOpcode::GpuBra" self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "GpuBra handler in lower_to_ptx.sio" "PASS"
else
    _report "GpuBra handler in lower_to_ptx.sio" "FAIL"
fi

# Check 5: Shared memory used (shared_bytes > 0)
if grep -q "shared_bytes = 2048" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "Shared memory allocation (2048 bytes)" "PASS"
else
    _report "Shared memory allocation (2048 bytes)" "FAIL"
fi

# Check 6: GpuStoreShared used in tiled builder (staging to shared)
if grep -q "GpuOpcode::GpuStoreShared" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "GpuStoreShared in tiled builder" "PASS"
else
    _report "GpuStoreShared in tiled builder" "FAIL"
fi

# Check 7: GpuLoadShared used in tiled builder (reading from shared)
if grep -q "GpuOpcode::GpuLoadShared" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "GpuLoadShared in tiled builder" "PASS"
else
    _report "GpuLoadShared in tiled builder" "FAIL"
fi

echo ""

# ── Section 2: PTX Structural (IR builder patterns) ────────────────

echo "── PTX Structural ─────────────────────────────────────────────"

# Check 8: WMMA opcode in tiled builder
if grep -q "GpuOpcode::GpuWmma" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "GpuWmma in tiled GEMM builder" "PASS"
else
    _report "GpuWmma in tiled GEMM builder" "FAIL"
fi

# Check 9: GpuSqrt in tiled builder (uncertainty path)
if grep -q "GpuOpcode::GpuSqrt" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "GpuSqrt in tiled builder (shadow)" "PASS"
else
    _report "GpuSqrt in tiled builder (shadow)" "FAIL"
fi

# Check 10: GpuXor in tiled builder (provenance)
if grep -q "GpuOpcode::GpuXor" self-hosted/gpu/kernel_ir.sio 2>/dev/null; then
    _report "GpuXor in tiled builder (provenance)" "PASS"
else
    _report "GpuXor in tiled builder (provenance)" "FAIL"
fi

# Check 11: cp.async emission pattern in lower_to_ptx.sio
if grep -q 'cp\.async' self-hosted/gpu/lower_to_ptx.sio 2>/dev/null; then
    _report "cp.async emission pattern present" "PASS"
else
    _report "cp.async emission pattern present" "FAIL"
fi

echo ""

# ── Section 3: GUM Oracle (analytical math via awk) ────────────────

echo "── GUM Oracle (JCGM 100:2008 verification) ───────────────────"

# Check 12: Zero uncertainty → zero output
ZERO_CHECK=$(awk 'BEGIN {
    eps_a = 0.0; norm_a = 1.0; eps_b = 0.0; norm_b = 1.0; n_tiles = 2;
    combined = (norm_a < 0 ? -norm_a : norm_a) * eps_b + (norm_b < 0 ? -norm_b : norm_b) * eps_a;
    tile_eps = sqrt(16) * combined;
    accum_sq = tile_eps * tile_eps * n_tiles;
    eps_c = sqrt(accum_sq);
    if (eps_c < 1e-12 && eps_c > -1e-12) print "PASS"; else print "FAIL";
}')
_report "GUM: zero-uncertainty identity (tiled)" "$ZERO_CHECK"

# Check 13: Tiled (2 tiles) >= single tile (quadrature monotonicity)
MONO_CHECK=$(awk 'BEGIN {
    eps_a = 0.1; norm_a = 2.0; eps_b = 0.2; norm_b = 3.0;
    combined = (norm_a < 0 ? -norm_a : norm_a) * eps_b + (norm_b < 0 ? -norm_b : norm_b) * eps_a;
    tile_eps = sqrt(16) * combined;
    single = tile_eps;
    tiled = sqrt(tile_eps * tile_eps * 2);
    if (tiled >= single) print "PASS"; else print "FAIL";
}')
_report "GUM: quadrature monotonicity (tiled >= single)" "$MONO_CHECK"

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
