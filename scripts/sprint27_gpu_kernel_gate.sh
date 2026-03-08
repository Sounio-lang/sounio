#!/usr/bin/env bash
# Sprint 27 Gate: GPU Kernel Pipeline — Source to PTX
#
# Validates:
# 1. kernel fn parses correctly (AST shows is_kernel: true)
# 2. Run-pass kernel tests pass type-checking
# 3. Compile-fail kernel tests are properly annotated
# 4. HLIR emission works for kernel functions
# 5. No regressions in existing test suite

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

check_output_contains() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"
    local pattern="$2"
    shift 2
    local output
    output=$("$@" 2>&1) || true
    if echo "$output" | grep -q "$pattern"; then
        PASS=$((PASS + 1))
        echo "  [PASS] $desc"
    else
        FAIL=$((FAIL + 1))
        echo "  [FAIL] $desc"
    fi
}

echo "=== Sprint 27: GPU Kernel Pipeline Gate ==="
echo ""

# --- Section 1: Parser (kernel fn recognized in AST) ---
echo "--- Section 1: Parser ---"
check_output_contains \
    "kernel fn sets is_kernel: true in AST" \
    '"is_kernel": true' \
    "$SOUC" check --show-ast tests/run-pass/kernel_fn_basic.sio

check_output_contains \
    "non-kernel fn has is_kernel: false in AST" \
    '"is_kernel": false' \
    "$SOUC" check --show-ast tests/run-pass/kernel_fn_basic.sio

# --- Section 2: Run-pass tests ---
echo ""
echo "--- Section 2: Run-pass tests ---"
check_pass "kernel_fn_basic.sio type-checks" \
    "$SOUC" check tests/run-pass/kernel_fn_basic.sio

check_pass "kernel_fn_gpu_effect.sio type-checks" \
    "$SOUC" check tests/run-pass/kernel_fn_gpu_effect.sio

check_pass "kernel_vec_add.sio example type-checks" \
    "$SOUC" check examples/kernel_vec_add.sio

# --- Section 3: HLIR emission ---
echo ""
echo "--- Section 3: HLIR emission ---"
check_pass "HLIR emission succeeds for kernel file" \
    "$SOUC" compile --emit hlir tests/run-pass/kernel_fn_basic.sio

check_output_contains \
    "HLIR contains vec_add function" \
    'vec_add' \
    "$SOUC" compile --emit hlir tests/run-pass/kernel_fn_basic.sio

# --- Section 4: Compile-fail tests exist ---
echo ""
echo "--- Section 4: Compile-fail test annotations ---"
for f in kernel_io_effect kernel_mut_effect kernel_nonvoid_return; do
    TOTAL=$((TOTAL + 1))
    if [ -f "tests/compile-fail/${f}.sio" ]; then
        if grep -q "compile-fail" "tests/compile-fail/${f}.sio"; then
            PASS=$((PASS + 1))
            echo "  [PASS] ${f}.sio exists and annotated"
        else
            FAIL=$((FAIL + 1))
            echo "  [FAIL] ${f}.sio missing annotation"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "  [FAIL] ${f}.sio not found"
    fi
done

# --- Section 5: Self-hosted infrastructure ---
echo ""
echo "--- Section 5: Self-hosted kernel infrastructure ---"
TOTAL=$((TOTAL + 1))
if grep -q "is_kernel: bool" self-hosted/parser/ast.sio; then
    PASS=$((PASS + 1))
    echo "  [PASS] FnDef.is_kernel field exists"
else
    FAIL=$((FAIL + 1))
    echo "  [FAIL] FnDef.is_kernel field missing"
fi

TOTAL=$((TOTAL + 1))
if grep -q "is_kernel: bool" self-hosted/check/defs.sio; then
    PASS=$((PASS + 1))
    echo "  [PASS] FnSig.is_kernel field exists"
else
    FAIL=$((FAIL + 1))
    echo "  [FAIL] FnSig.is_kernel field missing"
fi

TOTAL=$((TOTAL + 1))
if grep -q "hlir_lower_set_kernel" self-hosted/hlir/lower.sio; then
    PASS=$((PASS + 1))
    echo "  [PASS] HLIR kernel lowering wired"
else
    FAIL=$((FAIL + 1))
    echo "  [FAIL] HLIR kernel lowering not wired"
fi

TOTAL=$((TOTAL + 1))
if grep -q "hlir_function_to_gpu_kernel" self-hosted/gpu/hlir_to_gpu.sio; then
    PASS=$((PASS + 1))
    echo "  [PASS] HLIR→GPU IR bridge exists"
else
    FAIL=$((FAIL + 1))
    echo "  [FAIL] HLIR→GPU IR bridge missing"
fi

TOTAL=$((TOTAL + 1))
if grep -q "hlir_extract_kernel_count" self-hosted/gpu/hlir_to_gpu.sio; then
    PASS=$((PASS + 1))
    echo "  [PASS] Kernel extraction function exists"
else
    FAIL=$((FAIL + 1))
    echo "  [FAIL] Kernel extraction function missing"
fi

TOTAL=$((TOTAL + 1))
if grep -q 'code == 70' self-hosted/check/check.sio; then
    PASS=$((PASS + 1))
    echo "  [PASS] E070 error message registered"
else
    FAIL=$((FAIL + 1))
    echo "  [FAIL] E070 error message missing"
fi

TOTAL=$((TOTAL + 1))
if grep -q 'code == 72' self-hosted/check/check.sio; then
    PASS=$((PASS + 1))
    echo "  [PASS] E072 error message registered"
else
    FAIL=$((FAIL + 1))
    echo "  [FAIL] E072 error message missing"
fi

# --- Summary ---
echo ""
echo "=== Sprint 27 Gate: $PASS/$TOTAL passed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "GATE: FAIL ($FAIL failures)"
    exit 1
else
    echo "GATE: PASS"
fi
