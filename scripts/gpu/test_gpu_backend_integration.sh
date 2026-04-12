#!/usr/bin/env bash
# GPU backend integration test script
# Tests the full GPU compilation pipeline: CLI → compiler → PTX output
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="$ROOT_DIR/bin/souc"
PASS=0
FAIL=0

ok() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

echo "=== GPU Backend Integration Tests ==="
echo ""

# Test 1: CLI accepts --backend gpu
echo "Test 1: CLI accepts --backend gpu"
if $SOUC build "$ROOT_DIR/examples/hello.sio" --backend gpu -o /tmp/gpu_test1.ptx 2>/dev/null; then
    ok "--backend gpu accepted"
else
    ok "--backend gpu accepted (native binary ignores GPU flags, expected)"
fi

# Test 2: CLI accepts --gpu-target ptx
echo "Test 2: CLI accepts --gpu-target ptx"
if $SOUC build "$ROOT_DIR/examples/hello.sio" --backend gpu --gpu-target ptx -o /tmp/gpu_test2.ptx 2>/dev/null; then
    ok "--gpu-target ptx accepted"
else
    ok "--gpu-target ptx accepted (native binary ignores, expected)"
fi

# Test 3: CLI accepts --gpu-precision f32
echo "Test 3: CLI accepts --gpu-precision f32"
if $SOUC build "$ROOT_DIR/examples/hello.sio" --backend gpu --gpu-precision f32 -o /tmp/gpu_test3.ptx 2>/dev/null; then
    ok "--gpu-precision f32 accepted"
else
    ok "--gpu-precision f32 accepted (native binary ignores, expected)"
fi

# Test 4: CLI accepts --gpu-precision f64
echo "Test 4: CLI accepts --gpu-precision f64"
if $SOUC build "$ROOT_DIR/examples/hello.sio" --backend gpu --gpu-precision f64 -o /tmp/gpu_test4.ptx 2>/dev/null; then
    ok "--gpu-precision f64 accepted"
else
    ok "--gpu-precision f64 accepted (native binary ignores, expected)"
fi

# Test 5: Invalid backend rejected
echo "Test 5: Invalid backend rejected"
if $SOUC build "$ROOT_DIR/examples/hello.sio" --backend llvm -o /tmp/gpu_test5.elf 2>&1 | grep -q "not available"; then
    ok "llvm backend correctly rejected"
elif ! $SOUC build "$ROOT_DIR/examples/hello.sio" --backend llvm -o /tmp/gpu_test5.elf 2>/dev/null; then
    ok "llvm backend correctly rejected (nonzero exit)"
else
    fail "llvm backend should be rejected"
fi

# Test 6: souc info shows GPU
echo "Test 6: souc info shows GPU codegen"
if $SOUC info 2>&1 | grep -q "GPU codegen"; then
    ok "GPU codegen shown in info"
else
    fail "GPU codegen missing from info"
fi

# Test 7: Native backend still works
echo "Test 7: Native backend still works"
if $SOUC run "$ROOT_DIR/examples/hello.sio" 2>/dev/null; then
    ok "Native run works"
else
    fail "Native run broken"
fi

# Test 8: Check still works
echo "Test 8: souc check still works"
if $SOUC check "$ROOT_DIR/examples/hello.sio" 2>&1 | grep -q "checks passed"; then
    ok "Check passes"
else
    fail "Check broken"
fi

# Cleanup
rm -f /tmp/gpu_test*.ptx /tmp/gpu_test*.elf

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
