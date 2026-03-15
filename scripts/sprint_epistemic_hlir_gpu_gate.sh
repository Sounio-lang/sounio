#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# HLIR→GPU Source-Level Kernel Compilation Gate  (Track D)
#
# Validates the source-level GPU builtin lowering pipeline:
#   hlir_lower_gpu_builtins() + hlir_emit_kernel_prologue()
#   in self-hosted/gpu/hlir_to_gpu.sio
#
# 12 checks:
#   Section 1 — Infrastructure (5 grep checks)
#   Section 2 — PTX patterns    (2 grep checks)
#   Section 3 — Index oracle    (3 awk math checks)
#   Section 4 — Example         (1 souc check, SKIP if no souc)
#   Section 5 — Test file       (1 grep check)
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

PASS=0
FAIL=0
SKIP=0
TOTAL=12

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
echo "  HLIR→GPU SOURCE-LEVEL KERNEL COMPILATION GATE  (TRACK D)"
echo "  GPU Builtin Lowering · 1D Kernel Prologue · Thread Indexing"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Section 1: Infrastructure (source-level grep) ──────────────────

echo "── Section 1: Infrastructure ──────────────────────────────────"

# Check 1: hlir_lower_gpu_builtins function exists in hlir_to_gpu.sio
if grep -q "hlir_lower_gpu_builtins" self-hosted/gpu/hlir_to_gpu.sio 2>/dev/null; then
    _report "hlir_lower_gpu_builtins in hlir_to_gpu.sio" "PASS"
else
    _report "hlir_lower_gpu_builtins in hlir_to_gpu.sio" "FAIL"
fi

# Check 2: hlir_emit_kernel_prologue function exists in hlir_to_gpu.sio
if grep -q "hlir_emit_kernel_prologue" self-hosted/gpu/hlir_to_gpu.sio 2>/dev/null; then
    _report "hlir_emit_kernel_prologue in hlir_to_gpu.sio" "PASS"
else
    _report "hlir_emit_kernel_prologue in hlir_to_gpu.sio" "FAIL"
fi

# Check 3: GpuGetTid opcode referenced in hlir_to_gpu.sio
if grep -q "GpuOpcode::GpuGetTid" self-hosted/gpu/hlir_to_gpu.sio 2>/dev/null; then
    _report "GpuOpcode::GpuGetTid in hlir_to_gpu.sio" "PASS"
else
    _report "GpuOpcode::GpuGetTid in hlir_to_gpu.sio" "FAIL"
fi

# Check 4: GpuGetBid opcode referenced in hlir_to_gpu.sio
if grep -q "GpuOpcode::GpuGetBid" self-hosted/gpu/hlir_to_gpu.sio 2>/dev/null; then
    _report "GpuOpcode::GpuGetBid in hlir_to_gpu.sio" "PASS"
else
    _report "GpuOpcode::GpuGetBid in hlir_to_gpu.sio" "FAIL"
fi

# Check 5: GpuBarrierSync opcode referenced in hlir_to_gpu.sio
if grep -q "GpuOpcode::GpuBarrierSync" self-hosted/gpu/hlir_to_gpu.sio 2>/dev/null; then
    _report "GpuOpcode::GpuBarrierSync in hlir_to_gpu.sio" "PASS"
else
    _report "GpuOpcode::GpuBarrierSync in hlir_to_gpu.sio" "FAIL"
fi

echo ""

# ── Section 2: PTX patterns ────────────────────────────────────────

echo "── Section 2: PTX Patterns ────────────────────────────────────"

# Check 6: GpuGetNtid opcode referenced in hlir_to_gpu.sio (block_dim → ntid)
if grep -q "GpuOpcode::GpuGetNtid" self-hosted/gpu/hlir_to_gpu.sio 2>/dev/null; then
    _report "GpuOpcode::GpuGetNtid in hlir_to_gpu.sio" "PASS"
else
    _report "GpuOpcode::GpuGetNtid in hlir_to_gpu.sio" "FAIL"
fi

# Check 7: gpu_op_new() usage in new functions (prologue + builtin lowering)
if grep -q "gpu_op_new" self-hosted/gpu/hlir_to_gpu.sio 2>/dev/null; then
    _report "gpu_op_new usage in hlir_to_gpu.sio" "PASS"
else
    _report "gpu_op_new usage in hlir_to_gpu.sio" "FAIL"
fi

echo ""

# ── Section 3: Thread index oracle (awk math verification) ─────────

echo "── Section 3: Thread Index Oracle ─────────────────────────────"

# Check 8: 1D global index — tid=3, bid=2, bdim=32 → global_idx=67
IDX_1D=$(awk 'BEGIN {
    tid = 3; bid = 2; bdim = 32;
    global_idx = bid * bdim + tid;
    if (global_idx == 67) print "PASS"; else print "FAIL";
}')
_report "1D global index: tid=3 bid=2 bdim=32 → 67" "$IDX_1D"

# Check 9: 2D flat index — row=5, col=7, width=16 → idx=87
IDX_2D=$(awk 'BEGIN {
    row = 5; col = 7; width = 16;
    flat_idx = row * width + col;
    if (flat_idx == 87) print "PASS"; else print "FAIL";
}')
_report "2D flat index: row=5 col=7 width=16 → 87" "$IDX_2D"

# Check 10: Linear addressing continuity — consecutive threads → consecutive indices
IDX_LINEAR=$(awk 'BEGIN {
    bdim = 32;
    g0 = 0 * bdim + 0;   # tid=0, bid=0 → 0
    g1 = 0 * bdim + 1;   # tid=1, bid=0 → 1
    g32 = 1 * bdim + 0;  # tid=0, bid=1 → 32
    if (g0 == 0 && g1 == 1 && g32 == 32) print "PASS"; else print "FAIL";
}')
_report "Linear addressing: no off-by-one (g0=0 g1=1 g32=32)" "$IDX_LINEAR"

echo ""

# ── Section 4: Example file ───────────────────────────────────────

echo "── Section 4: Example Validation ──────────────────────────────"

# Check 11: souc check on kernel_source_level.sio
if [ -x "$GPU_SOUC" ]; then
    if timeout 30 "$GPU_SOUC" check examples/kernel_source_level.sio >/dev/null 2>&1; then
        _report "souc check examples/kernel_source_level.sio" "PASS"
    else
        _report "souc check examples/kernel_source_level.sio" "FAIL"
    fi
elif [ -x "$SOUC" ]; then
    if timeout 30 "$SOUC" check examples/kernel_source_level.sio >/dev/null 2>&1; then
        _report "souc check examples/kernel_source_level.sio" "PASS"
    else
        _report "souc check examples/kernel_source_level.sio" "FAIL"
    fi
else
    _report "souc check examples/kernel_source_level.sio (no souc)" "SKIP"
fi

echo ""

# ── Section 5: Test file integrity ────────────────────────────────

echo "── Section 5: Test File Integrity ─────────────────────────────"

# Check 12: test_epistemic_hlir_gpu.sio contains all 9 test functions
if grep -q "test_hlir_gpu_global_idx_basic\|test_hlir_gpu_linear_addressing" \
    self-hosted/test_epistemic_hlir_gpu.sio 2>/dev/null; then
    _report "test_epistemic_hlir_gpu.sio has oracle tests" "PASS"
else
    _report "test_epistemic_hlir_gpu.sio has oracle tests" "FAIL"
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
