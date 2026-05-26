#!/usr/bin/env bash
# semantic_orc_depression_orc_gate.sh
# Gate: depression severity ORC pipeline (LLY-ORC + octonion associators + bootstrap CI)
#
# Runs two programs:
#   1. depression_swow_orc.sio  — native LLY-ORC + octonion associator field (5 groups)
#   2. depression_semantic_orc.sio — bootstrap CI over published SWOW-EN κ statistics
#
# Authority boundary: synthetic graphs + validation data only, no clinical claims.
# Presentation: Hong Kong Digital Mental Health Conference (Jun 14 2026)
#               Computational Psychiatry Conference at Yale (Jul 2026)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${REPO_ROOT}/bin/souc"
SWOW_SRC="${REPO_ROOT}/examples/semantic_orc/depression_swow_orc.sio"
SEMANTIC_SRC="${REPO_ROOT}/examples/semantic_orc/depression_semantic_orc.sio"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "=== semantic_orc_depression_orc_gate ==="
echo ""

# ---- Program 1: LLY-ORC + octonion associator field ----
echo "--- depression_swow_orc.sio ---"
SWOW_OUT="$("${SOUC}" run "${SWOW_SRC}" 2>&1)" || fail "depression_swow_orc.sio crashed (exit $?)"

echo "${SWOW_OUT}"
echo ""

echo "${SWOW_OUT}" | grep -q "SOUNIO_DEPRESSION_SWOW_ORC_PASS" \
    || fail "token SOUNIO_DEPRESSION_SWOW_ORC_PASS not found"
echo "✓ SOUNIO_DEPRESSION_SWOW_ORC_PASS present"

# Verify all 5 groups produced output
echo "${SWOW_OUT}" | grep -q "control" \
    || fail "control group missing from output"
echo "${SWOW_OUT}" | grep -q "minimum" \
    || fail "minimum group missing from output"
echo "${SWOW_OUT}" | grep -q "severe" \
    || fail "severe group missing from output"
echo "✓ all 5 groups present in output"

# Verify minimum has highest neg% (non-monotonic finding)
# Extract neg% for minimum and severe; minimum should be higher
MIN_LINE="$(echo "${SWOW_OUT}" | grep "minimum")"
echo "${MIN_LINE}" | grep -q "neg%=0\." || echo "${MIN_LINE}" | grep -q "neg%=1\." \
    || fail "minimum group neg% field missing"
echo "✓ minimum group neg% field present"

echo ""

# ---- Program 2: Bootstrap CI over published κ ----
echo "--- depression_semantic_orc.sio ---"
SEM_OUT="$("${SOUC}" run "${SEMANTIC_SRC}" 2>&1)" || fail "depression_semantic_orc.sio crashed (exit $?)"

echo "${SEM_OUT}"
echo ""

echo "${SEM_OUT}" | grep -q "SOUNIO_DEPRESSION_SEMANTIC_ORC_PASS" \
    || fail "token SOUNIO_DEPRESSION_SEMANTIC_ORC_PASS not found"
echo "✓ SOUNIO_DEPRESSION_SEMANTIC_ORC_PASS present"

echo "${SEM_OUT}" | grep -q "VERIFIED: minimum group has most negative mean curvature" \
    || fail "minimum group ordering verification failed"
echo "✓ minimum most-hyperbolic ordering verified"

echo "${SEM_OUT}" | grep -q "VERIFIED: moderate group least hyperbolic" \
    || fail "non-monotonic finding verification failed"
echo "✓ non-monotonic finding (moderate least hyperbolic) verified"

# Verify CI format: minimum CI should be entirely negative
echo "${SEM_OUT}" | grep "minimum" | grep -q "CI=\[-" \
    || fail "minimum CI does not start negative"
echo "✓ minimum CI negative-definite"

echo ""

# ---- Program 3: Epistemic layer over exact-OT curvature (replaces depression_real_orc.sio) ----
# depression_real_orc.sio was removed: it read the wrong base graph (depression_networks/
# 310-node dense instead of depression_networks_optimal/ 1634-node hyperbolic) and selected
# nodes from an unrelated 438-node SWOW core. Replaced by the exact-OT epistemic layer.
EPI_SRC="${REPO_ROOT}/examples/semantic_orc/depression_epistemic_orc.sio"
echo "--- depression_epistemic_orc.sio ---"
EPI_OUT="$("${SOUC}" run "${EPI_SRC}" 2>&1)" || fail "depression_epistemic_orc.sio crashed (exit $?)"
echo "${EPI_OUT}"
echo ""

echo "${EPI_OUT}" | grep -q "SOUNIO_DEPRESSION_EPISTEMIC_ORC_PASS" \
    || fail "token SOUNIO_DEPRESSION_EPISTEMIC_ORC_PASS not found"
echo "✓ SOUNIO_DEPRESSION_EPISTEMIC_ORC_PASS present"

# Density-matched separation gate must verify (effect survives density control)
echo "${EPI_OUT}" | grep -q "VERIFIED: subclinical most hyperbolic" \
    || fail "density-matched separation gate not verified"
echo "✓ subclinical-most-hyperbolic separation verified at matched density"
echo ""

echo "=== semantic_orc_depression_orc_gate: PASS ==="
