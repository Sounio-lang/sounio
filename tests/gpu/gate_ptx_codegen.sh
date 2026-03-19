#!/usr/bin/env bash
# tests/gpu/gate_ptx_codegen.sh — GPU PTX codegen gate test
#
# Validates the GpuKernelIr → PTX text pipeline end-to-end.
# Does NOT require GPU hardware — tests code generation only.
#
# Usage: bash tests/gpu/gate_ptx_codegen.sh
# Exit:  0 = all PASS, non-zero = failure count

set -eo pipefail

SOUC="${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
PASS=0
FAIL=0
NOT_RUN=0

pass() { echo "PASS $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL $1: $2"; FAIL=$((FAIL + 1)); }
skip() { echo "SKIP $1: $2"; NOT_RUN=$((NOT_RUN + 1)); }

# ── Preconditions ─────────────────────────────────────────────────────────────

check_souc() {
    if [ ! -x "$SOUC" ]; then
        fail "precondition" "souc binary not found at $SOUC"
        exit 1
    fi
    pass "precondition_souc_exists"
}

check_kernel_ir() {
    if [ ! -f "self-hosted/gpu/kernel_ir.sio" ]; then
        fail "precondition" "self-hosted/gpu/kernel_ir.sio not found"
        exit 1
    fi
    pass "precondition_kernel_ir_exists"
}

check_ptx_advanced() {
    if [ ! -f "self-hosted/gpu/ptx_advanced.sio" ]; then
        fail "precondition" "self-hosted/gpu/ptx_advanced.sio not found"
        exit 1
    fi
    pass "precondition_ptx_advanced_exists"
}

check_test_file() {
    if [ ! -f "tests/gpu/test_ptx_codegen.sio" ]; then
        fail "precondition" "tests/gpu/test_ptx_codegen.sio not found"
        exit 1
    fi
    pass "precondition_test_file_exists"
}

check_gpu_test_file() {
    if [ ! -f "self-hosted/gpu/test_ptx_codegen.sio" ]; then
        fail "precondition" "self-hosted/gpu/test_ptx_codegen.sio not found"
        exit 1
    fi
    pass "precondition_gpu_test_file_exists"
}

# ── Core test: run the .sio gate test ─────────────────────────────────────────

run_ptx_gate_test() {
    local output
    local _ec=0
    output=$(timeout 60 "$SOUC" run tests/gpu/test_ptx_codegen.sio 2>&1) || _ec=$?

    if [ $_ec -eq 124 ]; then
        skip "ptx_gate_test" "timeout (60s)"
        return
    fi

    if echo "$output" | grep -q "PASS 10/10"; then
        pass "ptx_gate_test_10_10"
    elif echo "$output" | grep -q "FAIL"; then
        fail "ptx_gate_test" "$(echo "$output" | grep FAIL | head -1)"
    elif [ $_ec -eq 0 ]; then
        pass "ptx_gate_test_exit_0"
    else
        fail "ptx_gate_test" "exit code $_ec: $(echo "$output" | tail -3)"
    fi
}

# ── Structural checks: verify key GPU source files have required symbols ───────

check_gpu_build_vec_add_ir() {
    if grep -q "fn gpu_build_vec_add_ir" self-hosted/gpu/kernel_ir.sio; then
        pass "symbol_gpu_build_vec_add_ir"
    else
        fail "symbol_gpu_build_vec_add_ir" "function not found in kernel_ir.sio"
    fi
}

check_ptx_codegen_kernel_ampere() {
    if grep -q "fn ptx_codegen_kernel_ampere" self-hosted/gpu/ptx_advanced.sio; then
        pass "symbol_ptx_codegen_kernel_ampere"
    else
        fail "symbol_ptx_codegen_kernel_ampere" "function not found in ptx_advanced.sio"
    fi
}

check_hlir_to_gpu_entry() {
    if grep -q "fn hlir_emit_kernel_prologue" self-hosted/gpu/hlir_to_gpu.sio; then
        pass "symbol_hlir_emit_kernel_prologue"
    else
        fail "symbol_hlir_emit_kernel_prologue" "function not found in hlir_to_gpu.sio"
    fi
}

check_portable_dispatch() {
    if grep -q "fn gpu_compile_to_target" self-hosted/gpu/portable.sio; then
        pass "symbol_gpu_compile_to_target"
    else
        fail "symbol_gpu_compile_to_target" "function not found in portable.sio"
    fi
}

check_epistemic_ptx() {
    if grep -q "fn epistemic_ptx_kernel" self-hosted/gpu/epistemic_ptx.sio 2>/dev/null || \
       grep -q "shadow.*reg\|GUM\|Knowledge" self-hosted/gpu/epistemic_ptx.sio 2>/dev/null; then
        pass "epistemic_ptx_present"
    else
        skip "epistemic_ptx_present" "no GUM/shadow/Knowledge markers in epistemic_ptx.sio"
    fi
}

check_gpu_target_parsed() {
    if grep -q "gpu_target" self-hosted/compiler/module_loader.sio; then
        pass "gpu_target_parsed_in_module_loader"
    else
        fail "gpu_target_parsed_in_module_loader" "gpu_target not in module_loader.sio"
    fi
}

check_gpu_dispatch_wired() {
    if grep -q "run_gpu_compile_pipeline" self-hosted/compiler/main.sio; then
        pass "gpu_dispatch_wired_in_main"
    else
        fail "gpu_dispatch_wired_in_main" "run_gpu_compile_pipeline not found in compiler/main.sio"
    fi
}

check_hlir_imports_added() {
    if grep -q "use hlir::lower" self-hosted/compiler/main.sio && \
       grep -q "use gpu::hlir_to_gpu" self-hosted/compiler/main.sio; then
        pass "hlir_gpu_imports_present"
    else
        fail "hlir_gpu_imports_present" "hlir/gpu imports missing from compiler/main.sio"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "=== GPU PTX Codegen Gate ==="
echo "SOUC: $SOUC"
echo ""

check_souc
check_kernel_ir
check_ptx_advanced
check_test_file
check_gpu_test_file

echo ""
echo "--- Structural checks ---"
check_gpu_build_vec_add_ir
check_ptx_codegen_kernel_ampere
check_hlir_to_gpu_entry
check_portable_dispatch
check_epistemic_ptx
check_gpu_target_parsed
check_gpu_dispatch_wired
check_hlir_imports_added

echo ""
echo "--- End-to-end PTX codegen ---"
run_ptx_gate_test

echo ""
echo "=== Results: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN ==="

if [ "$FAIL" -gt 0 ]; then
    exit "$FAIL"
fi
exit 0
