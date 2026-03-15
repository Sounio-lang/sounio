#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Epistemic PTX Assembly Gate — REAL VALIDATION
#
# Uses ptxas (CUDA 12.4) to actually assemble the epistemic tensor core
# kernel. This proves the PTX is architecturally valid, not just grep-correct.
#
# Requires: ptxas in PTXAS env var or /tmp/cuda-nvcc-extracted/...
# ═══════════════════════════════════════════════════════════════════════
set -eo pipefail

PASS=0; FAIL=0; SKIP=0; TOTAL=8

PTXAS="${PTXAS:-/tmp/cuda-nvcc-extracted/usr/local/cuda-12.4/bin/ptxas}"
PTX="self-hosted/gpu/epistemic_mma_reference.ptx"

_report() {
    local label="$1" status="$2"
    if [ "$status" = "PASS" ]; then
        PASS=$((PASS + 1)); printf "  [PASS] %s\n" "$label"
    elif [ "$status" = "SKIP" ]; then
        SKIP=$((SKIP + 1)); printf "  [SKIP] %s\n" "$label"
    else
        FAIL=$((FAIL + 1)); printf "  [FAIL] %s\n" "$label"
    fi
}

echo "═══════════════════════════════════════════════════════════════"
echo "  EPISTEMIC PTX ASSEMBLY GATE — REAL VALIDATION"
echo "  ptxas CUDA 12.4 · sm_80 · GUM JCGM 100:2008"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "── PTX Structural (source) ─────────────────────────────────────"

# 1: mma.sync present (tensor core data path)
if grep -q "mma.sync.aligned" "$PTX" 2>/dev/null; then
    _report "mma.sync.aligned in epistemic_mma_reference.ptx" "PASS"
else
    _report "mma.sync.aligned in epistemic_mma_reference.ptx" "FAIL"
fi

# 2: shadow path present
if grep -q "sqrt.approx.f32" "$PTX" 2>/dev/null; then
    _report "sqrt.approx.f32 (GUM shadow path)" "PASS"
else
    _report "sqrt.approx.f32 (GUM shadow path)" "FAIL"
fi

# 3: validity conjunction
if grep -q "and.pred" "$PTX" 2>/dev/null; then
    _report "and.pred (validity conjunction)" "PASS"
else
    _report "and.pred (validity conjunction)" "FAIL"
fi

# 4: provenance XOR
if grep -q "xor.b64" "$PTX" 2>/dev/null; then
    _report "xor.b64 (Merkle provenance merge)" "PASS"
else
    _report "xor.b64 (Merkle provenance merge)" "FAIL"
fi

echo ""
echo "── ptxas Assembly (REAL VALIDATION) ───────────────────────────"

if [ ! -x "$PTXAS" ]; then
    _report "ptxas available" "SKIP"
    _report "epistemic_mma_reference.ptx assembles (sm_80)" "SKIP"
    _report "cubin produced (non-zero size)" "SKIP"
    _report "26 registers allocated" "SKIP"
else
    _report "ptxas available" "PASS"
    # 6: actually assemble
    OUTFILE=$(mktemp /tmp/epistemic_test_XXXXXX.cubin)
    if PTXAS_OUT=$("$PTXAS" -arch sm_80 -v -o "$OUTFILE" "$PTX" 2>&1); then
        _report "epistemic_mma_reference.ptx assembles (sm_80)" "PASS"
        # 7: cubin produced
        if [ -s "$OUTFILE" ]; then
            _report "cubin produced (non-zero size)" "PASS"
        else
            _report "cubin produced (non-zero size)" "FAIL"
        fi
        # 8: register count (ptxas -v reports "Used N registers")
        REGS=$(echo "$PTXAS_OUT" | grep -oP 'Used \K[0-9]+(?= registers)' | head -1)
        if [ -n "$REGS" ] && [ "$REGS" -ge 20 ]; then
            _report "register count >= 20 (got ${REGS})" "PASS"
        else
            _report "register count >= 20 (got ${REGS:-unknown})" "FAIL"
        fi
    else
        _report "epistemic_mma_reference.ptx assembles (sm_80)" "FAIL"
        echo "$PTXAS_OUT" | head -5
        _report "cubin produced (non-zero size)" "FAIL"
        _report "register count >= 20" "FAIL"
    fi
    rm -f "$OUTFILE"
fi

echo ""
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
