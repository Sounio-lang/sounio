#!/usr/bin/env bash
# Sprint 28 Gate: GPU Pipeline Hardening — End-to-End PTX + Fusion + Epistemic
#
# Validates:
# 1. Sprint 27 gate still passes (no regressions)
# 2. New run-pass tests type-check
# 3. New example files type-check
# 4. Self-hosted gpu/hlir_to_gpu.sio infrastructure (orchestrators exist)
# 5. Self-hosted main.sio gpu-emit command dispatch exists
# 6. Epistemic shadow register emission infrastructure exists
# 7. Multi-backend emission (Metal) infrastructure exists

set -euo pipefail

SOUC="${SOUC:-target/debug/souc}"
PASS=0
FAIL=0
TOTAL=0

check_pass() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        PASS=$((PASS + 1))
        echo "  [PASS] $desc"
    else
        FAIL=$((FAIL + 1))
        echo "  [FAIL] $desc"
    fi
}

check_file_contains() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"
    local pattern="$2"
    local file="$3"
    if [ -f "$file" ] && grep -q "$pattern" "$file"; then
        PASS=$((PASS + 1))
        echo "  [PASS] $desc"
    else
        FAIL=$((FAIL + 1))
        echo "  [FAIL] $desc"
    fi
}

echo "=== Sprint 28: GPU Pipeline Hardening Gate ==="
echo ""

# --- Section 1: Sprint 27 non-regression ---
echo "--- Section 1: Sprint 27 non-regression ---"
check_pass "kernel_fn_basic.sio type-checks" \
    "$SOUC" check tests/run-pass/kernel_fn_basic.sio

check_pass "kernel_fn_gpu_effect.sio type-checks" \
    "$SOUC" check tests/run-pass/kernel_fn_gpu_effect.sio

check_pass "kernel_vec_add.sio example type-checks" \
    "$SOUC" check examples/kernel_vec_add.sio

# --- Section 2: New run-pass tests ---
echo ""
echo "--- Section 2: Sprint 28 run-pass tests ---"
check_pass "kernel_ptx_emit.sio type-checks" \
    "$SOUC" check tests/run-pass/kernel_ptx_emit.sio

check_pass "kernel_multi_backend.sio type-checks" \
    "$SOUC" check tests/run-pass/kernel_multi_backend.sio

# --- Section 3: New examples ---
echo ""
echo "--- Section 3: Sprint 28 examples ---"
check_pass "kernel_matmul.sio type-checks" \
    "$SOUC" check examples/kernel_matmul.sio

check_pass "kernel_epistemic_vec_add.sio type-checks" \
    "$SOUC" check examples/kernel_epistemic_vec_add.sio

# --- Section 4: Self-hosted GPU pipeline infrastructure ---
echo ""
echo "--- Section 4: Self-hosted GPU pipeline ---"
check_file_contains \
    "hlir_kernels_to_ptx orchestrator exists" \
    "fn hlir_kernels_to_ptx" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "hlir_kernels_to_tuned_ptx (auto-tuned) exists" \
    "fn hlir_kernels_to_tuned_ptx" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "hlir_kernels_to_fused_ptx (fusion) exists" \
    "fn hlir_kernels_to_fused_ptx" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "hlir_kernels_to_epistemic_ptx (epistemic) exists" \
    "fn hlir_kernels_to_epistemic_ptx" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "hlir_kernels_to_metal (multi-backend) exists" \
    "fn hlir_kernels_to_metal" \
    self-hosted/gpu/hlir_to_gpu.sio

# --- Section 5: Self-hosted CLI dispatch ---
echo ""
echo "--- Section 5: Self-hosted CLI gpu-emit ---"
check_file_contains \
    "gpu-emit command dispatch exists" \
    'str_eq(cmd, "gpu-emit")' \
    self-hosted/main.sio

check_file_contains \
    "run_gpu_emit_pipeline function exists" \
    "fn run_gpu_emit_pipeline" \
    self-hosted/main.sio

check_file_contains \
    "--autotune flag parsed" \
    '"--autotune"' \
    self-hosted/main.sio

check_file_contains \
    "--fuse flag parsed" \
    '"--fuse"' \
    self-hosted/main.sio

check_file_contains \
    "--epistemic flag parsed" \
    '"--epistemic"' \
    self-hosted/main.sio

# --- Section 6: Epistemic GPU infrastructure ---
echo ""
echo "--- Section 6: Epistemic GPU kernels ---"
check_file_contains \
    "Epistemic shadow binary emission exists" \
    "fn gpu_emit_epistemic_shadow_binary" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "Epistemic GPU kernel lowering exists" \
    "fn hlir_function_to_epistemic_gpu_kernel" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "E073 error message registered" \
    'code == 73' \
    self-hosted/check/check.sio

check_file_contains \
    "is_epistemic_kernel field on FnSig" \
    "is_epistemic_kernel: bool" \
    self-hosted/check/defs.sio

# --- Section 7: Expanded instruction coverage ---
echo ""
echo "--- Section 7: HLIR→GPU IR coverage ---"
check_file_contains \
    "CONST instruction lowering" \
    "HLIR_OP_CONST" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "CAST instruction lowering (GpuCvt)" \
    "HLIR_OP_CAST" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "COPY instruction lowering" \
    "HLIR_OP_COPY" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "UNARY instruction lowering" \
    "HLIR_OP_UNARY" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "Block terminator BRANCH lowering" \
    "HLIR_TERM_BRANCH" \
    self-hosted/gpu/hlir_to_gpu.sio

check_file_contains \
    "Block terminator COND_BRANCH lowering" \
    "HLIR_TERM_COND_BRANCH" \
    self-hosted/gpu/hlir_to_gpu.sio

# --- Section 8: PTX buffer helpers ---
echo ""
echo "--- Section 8: PTX/Metal buffer helpers ---"
check_file_contains \
    "ptx_buf_append exists" \
    "fn ptx_buf_append" \
    self-hosted/gpu/ptx.sio

check_file_contains \
    "ptx_buf_print exists" \
    "fn ptx_buf_print" \
    self-hosted/gpu/ptx.sio

check_file_contains \
    "metal_buf_append exists" \
    "fn metal_buf_append" \
    self-hosted/gpu/metal.sio

check_file_contains \
    "metal_buf_print exists" \
    "fn metal_buf_print" \
    self-hosted/gpu/metal.sio

# --- Summary ---
echo ""
echo "=== Sprint 28 Gate: $PASS/$TOTAL passed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "GATE: FAIL ($FAIL failures)"
    exit 1
else
    echo "GATE: PASS"
fi
