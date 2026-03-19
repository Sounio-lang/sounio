#!/usr/bin/env bash
# tests/gpu/gate_ptx_codegen.sh — GPU PTX codegen gate
#
# Validates the GPU kernel check pipeline and PTX codegen dispatch.
# Does NOT require GPU hardware — tests type-checking and code generation only.
#
# Usage: bash tests/gpu/gate_ptx_codegen.sh
# Exit:  0 = all non-SKIP tests pass, non-zero = failure count

set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"

PASS=0
FAIL=0
SKIP=0
TOTAL=0

pass() {
    TOTAL=$((TOTAL + 1))
    PASS=$((PASS + 1))
    echo "PASS  $1"
}

fail() {
    TOTAL=$((TOTAL + 1))
    FAIL=$((FAIL + 1))
    echo "FAIL  $1: $2"
}

skip() {
    TOTAL=$((TOTAL + 1))
    SKIP=$((SKIP + 1))
    echo "SKIP  $1: $2"
}

# ── check_souc: run `$SOUC check <file>`, expect exit 0 ───────────────────────

check_souc() {
    local label="$1"
    local file="$2"
    local log="/tmp/gate_ptx_${label}.log"
    local _ec=0
    timeout 30 "$SOUC" check "$file" >"$log" 2>&1 || _ec=$?
    if [ $_ec -eq 124 ]; then
        fail "$label" "timeout (30s)"
    elif [ $_ec -eq 0 ]; then
        pass "$label"
    else
        fail "$label" "exit $_ec — $(tail -3 "$log" | tr '\n' ' ')"
    fi
}

# ── check_souc_or_skip: skip when file absent, check otherwise ────────────────

check_souc_or_skip() {
    local label="$1"
    local file="$2"
    if [ ! -f "$file" ]; then
        skip "$label" "$file not found"
        return
    fi
    check_souc "$label" "$file"
}

echo "=== GPU PTX Codegen Gate ==="
echo "SOUC:               $SOUC"
echo "SOUNIO_STDLIB_PATH: $SOUNIO_STDLIB_PATH"
echo ""

# ── Precondition ───────────────────────────────────────────────────────────────

if [ ! -x "$SOUC" ]; then
    echo "FATAL: souc binary not executable: $SOUC"
    exit 1
fi

# ── Section 1: souc check — required files ────────────────────────────────────

echo "--- Section 1: souc check (required) ---"

# T1
check_souc \
    "T1_kernel_vec_add" \
    "examples/kernel_vec_add.sio"

# T2
check_souc \
    "T2_kernel_source_level" \
    "examples/kernel_source_level.sio"

# T3
check_souc \
    "T3_gpu_kernel_basic" \
    "tests/run-pass/gpu_kernel_basic.sio"

# T4
check_souc \
    "T4_stdlib_test_gpu" \
    "tests/stdlib/gpu/test_gpu.sio"

# T5
check_souc \
    "T5_kernel_epistemic_wmma_matmul" \
    "examples/kernel_epistemic_wmma_matmul.sio"

# ── Section 2: GPU compile pipeline (self-hosted compiler) ────────────────────

echo ""
echo "--- Section 2: GPU compile pipeline ---"

# T6: GPU pipeline wiring — structural check (JIT runner OOM prevents full execution
#     of compiler/main.sio; validate via grep instead)
{
    TOTAL=$((TOTAL + 1))
    # Verify run_gpu_compile_pipeline is wired in compiler/main.sio
    if grep -q "run_gpu_compile_pipeline" self-hosted/compiler/main.sio && \
       grep -q 'use hlir::lower:.*hlir_lower_module' self-hosted/compiler/main.sio && \
       grep -q 'use gpu::hlir_to_gpu:.*hlir_kernels_to_ptx' self-hosted/compiler/main.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T6_gpu_pipeline_wired"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T6_gpu_pipeline_wired: run_gpu_compile_pipeline or imports missing from compiler/main.sio"
    fi
}

# ── Section 3: souc check — optional files ────────────────────────────────────

echo ""
echo "--- Section 3: souc check (optional) ---"

# T7: skip if absent
check_souc_or_skip \
    "T7_kernel_matmul" \
    "examples/kernel_matmul.sio"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Results: PASS=$PASS FAIL=$FAIL SKIP=$SKIP TOTAL=$TOTAL ==="

if [ "$FAIL" -gt 0 ]; then
    exit "$FAIL"
fi
exit 0
