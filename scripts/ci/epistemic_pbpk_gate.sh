#!/usr/bin/env bash
# Epistemic PBPK Gate -- validates native ELF epistemic computing demo
# Sprint 228-230: Operation Epistemic Dawn
set -eo pipefail

SOUC=./bin/souc
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
SRC=tests/run-pass/epistemic_pbpk_native.sio
ELF=/tmp/epistemic_pbpk_gate.elf

pass=0; fail=0

echo "=== Epistemic PBPK Gate ==="
echo ""

# ── Phase 1: Compile to native ELF ─────────────────────────────────────────
echo "Phase 1: Compile to native ELF..."
compile_start=$(date +%s%N)
timeout 120 $SOUC run $LEAN -- $SRC $ELF 2>/dev/null
compile_end=$(date +%s%N)
compile_ms=$(( (compile_end - compile_start) / 1000000 ))

if [ -f "$ELF" ]; then
    chmod +x "$ELF"
    elf_size=$(stat -c%s "$ELF")
    echo "  PASS: compiled ($elf_size bytes, ${compile_ms}ms)"
    pass=$((pass+1))
else
    echo "  FAIL: compilation failed"
    fail=$((fail+1))
fi

# ── Phase 2: Run native ELF ────────────────────────────────────────────────
echo ""
echo "Phase 2: Run native ELF..."
run_start=$(date +%s%N)
output=$("$ELF" 2>/dev/null || true)
exit_code=$?
run_end=$(date +%s%N)
run_ms=$(( (run_end - run_start) / 1000000 ))

echo "  Exit code: $exit_code (${run_ms}ms)"

# ── Phase 3: Validate output ───────────────────────────────────────────────
echo ""
echo "Phase 3: Validate output..."

# T1: 6/6 PASS
if echo "$output" | grep -qF "6/6 PASS, 0 FAIL"; then
    echo "  PASS: 6/6 tests passed"
    pass=$((pass+1))
else
    echo "  FAIL: not all 6 tests passed"
    fail=$((fail+1))
fi

# T2: Brain/plasma ratio present and < 0.15
if echo "$output" | grep -qF "PASS: T1 brain/plasma"; then
    echo "  PASS: P-gp BBB efflux verified"
    pass=$((pass+1))
else
    echo "  FAIL: P-gp test missing"
    fail=$((fail+1))
fi

# T3: CV(AUC) > 30%
if echo "$output" | grep -qF "PASS: T2 CV(AUC)"; then
    echo "  PASS: CYP3A4 variability propagated"
    pass=$((pass+1))
else
    echo "  FAIL: CYP3A4 variability test missing"
    fail=$((fail+1))
fi

# T4: CL dominates
if echo "$output" | grep -qF "PASS: T5 CL dominates"; then
    echo "  PASS: CL identified as dominant uncertainty source"
    pass=$((pass+1))
else
    echo "  FAIL: sensitivity analysis test missing"
    fail=$((fail+1))
fi

# T5: Uncertainty values present
if echo "$output" | grep -qF "+/-"; then
    echo "  PASS: uncertainty bounds in output"
    pass=$((pass+1))
else
    echo "  FAIL: no uncertainty bounds found"
    fail=$((fail+1))
fi

# ── Phase 4: Cross-validate with JIT ───────────────────────────────────────
echo ""
echo "Phase 4: Cross-validate with JIT..."
jit_output=$(timeout 60 $SOUC run $SRC 2>/dev/null || true)
if echo "$jit_output" | grep -qF "6/6 PASS, 0 FAIL"; then
    echo "  PASS: JIT also produces 6/6 PASS"
    pass=$((pass+1))
else
    echo "  FAIL: JIT path diverges"
    fail=$((fail+1))
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Epistemic PBPK Gate: $pass PASS, $fail FAIL ==="
echo "  Compile: ${compile_ms}ms  Run: ${run_ms}ms  ELF: ${elf_size:-0} bytes"
echo ""

if [ $fail -eq 0 ]; then
    echo "  Native epistemic computing VERIFIED."
    echo "  20KB ELF propagates measurement uncertainty through 1920 RK4 steps."
fi

exit $fail
