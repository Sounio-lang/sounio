#!/usr/bin/env bash
# Epistemic Multi-Drug PBPK Gate -- 3 drugs, ASCII curves, 10 tests
set -eo pipefail

SOUC=./bin/souc
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
SRC=tests/run-pass/epistemic_pbpk_multidrug.sio
ELF=/tmp/epistemic_multidrug_gate.elf

pass=0; fail=0

echo "=== Multi-Drug Epistemic PBPK Gate ==="
echo ""

# Phase 1: Compile
echo "Phase 1: Compile to native ELF..."
t0=$(date +%s%N)
timeout 180 $SOUC run $LEAN -- $SRC $ELF 2>/dev/null
t1=$(date +%s%N)
ms=$(( (t1 - t0) / 1000000 ))

if [ -f "$ELF" ]; then
    chmod +x "$ELF"
    sz=$(stat -c%s "$ELF")
    echo "  PASS: compiled ($sz bytes, ${ms}ms)"
    pass=$((pass+1))
else
    echo "  FAIL: compilation failed"; fail=$((fail+1))
fi

# Phase 2: Run
echo ""
echo "Phase 2: Run native ELF..."
t0=$(date +%s%N)
output=$("$ELF" 2>/dev/null || true)
ec=$?
t1=$(date +%s%N)
rms=$(( (t1 - t0) / 1000000 ))
echo "  Exit: $ec (${rms}ms)"

# Phase 3: Validate
echo ""
echo "Phase 3: Validate..."

if echo "$output" | grep -qF "10/10 PASS, 0 FAIL"; then
    echo "  PASS: 10/10 tests passed"; pass=$((pass+1))
else
    echo "  FAIL: not 10/10"; fail=$((fail+1))
fi

if echo "$output" | grep -qF "Rapamycin" && echo "$output" | grep -qF "Caffeine" && echo "$output" | grep -qF "Metformin"; then
    echo "  PASS: all 3 drugs present"; pass=$((pass+1))
else
    echo "  FAIL: missing drug"; fail=$((fail+1))
fi

if echo "$output" | grep -qF "R=Rapamycin"; then
    echo "  PASS: ASCII plot rendered"; pass=$((pass+1))
else
    echo "  FAIL: no plot"; fail=$((fail+1))
fi

if echo "$output" | grep -qF "+/-"; then
    echo "  PASS: uncertainty bounds present"; pass=$((pass+1))
else
    echo "  FAIL: no uncertainty"; fail=$((fail+1))
fi

if echo "$output" | grep -qF "PASS T7: CL dominates"; then
    echo "  PASS: sensitivity analysis working"; pass=$((pass+1))
else
    echo "  FAIL: sensitivity missing"; fail=$((fail+1))
fi

# Phase 4: Cross-validate with JIT
echo ""
echo "Phase 4: JIT cross-validation..."
jit_out=$(timeout 60 $SOUC run $SRC 2>/dev/null || true)
if echo "$jit_out" | grep -qF "10/10 PASS, 0 FAIL"; then
    echo "  PASS: JIT also 10/10"; pass=$((pass+1))
else
    echo "  FAIL: JIT diverges"; fail=$((fail+1))
fi

# Summary
echo ""
echo "=== Multi-Drug Gate: $pass PASS, $fail FAIL ==="
echo "  Compile: ${ms}ms  Run: ${rms}ms  ELF: ${sz:-0} bytes"
echo ""
if [ $fail -eq 0 ]; then
    echo "  3 drugs, 12 simulations, ASCII curves, 10 tests, 1 native binary."
fi

exit $fail
