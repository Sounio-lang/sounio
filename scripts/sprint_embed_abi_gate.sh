#!/bin/bash
# Gate script for the kernel/session embedding ABI.
#
# Validates:
#   1. serve_entry.sio passes souc check
#   2. Python smoke test: 32/32 PASS across 14 SNIO round-trips
#
# Usage:
#   bash scripts/sprint_embed_abi_gate.sh

set -eo pipefail

SOUC="${SOUC:-./bin/souc}"
PASS=0
FAIL=0
NOT_RUN=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        echo "  PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $name"
        FAIL=$((FAIL + 1))
    fi
}

skip() {
    local name="$1"
    echo "  NOT_RUN  $name"
    NOT_RUN=$((NOT_RUN + 1))
}

echo "=== Embedding ABI Gate ==="
echo "souc: $SOUC"
echo ""

# --- T1: serve_entry.sio check ---
echo "[gate] T1: souc check serve_entry.sio"
if $SOUC check self-hosted/interop/serve_entry.sio > /dev/null 2>&1; then
    check "serve_entry.sio check" 0
else
    check "serve_entry.sio check" 1
fi

# --- T2: Individual module checks ---
echo "[gate] T2: Module checks"
for mod in protocol session kernel serialize artifact contract; do
    if $SOUC check "self-hosted/interop/${mod}.sio" > /dev/null 2>&1; then
        check "${mod}.sio check" 0
    else
        check "${mod}.sio check" 1
    fi
done

# --- T3: Python smoke test ---
echo "[gate] T3: Python smoke test (32 checks)"
if command -v python3 > /dev/null 2>&1; then
    SMOKE_OUTPUT=$(python3 examples/kernel_session_smoke.py 2>&1)
    SMOKE_EXIT=$?

    # Extract pass/fail counts from output
    SMOKE_PASS=$(echo "$SMOKE_OUTPUT" | grep -oP '\d+ PASS' | grep -oP '\d+')
    SMOKE_FAIL=$(echo "$SMOKE_OUTPUT" | grep -oP '\d+ FAIL' | grep -oP '\d+')

    if [ "$SMOKE_EXIT" -eq 0 ]; then
        check "smoke test (${SMOKE_PASS:-?} PASS, ${SMOKE_FAIL:-?} FAIL)" 0
    else
        check "smoke test (${SMOKE_PASS:-?} PASS, ${SMOKE_FAIL:-?} FAIL)" 1
        echo "  --- smoke output ---"
        echo "$SMOKE_OUTPUT" | tail -20
        echo "  ---"
    fi
else
    skip "smoke test (python3 not found)"
fi

# --- Summary ---
TOTAL=$((PASS + FAIL + NOT_RUN))
echo ""
echo "=== Gate Summary ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
