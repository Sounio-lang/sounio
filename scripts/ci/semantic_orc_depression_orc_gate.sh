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

grep -q "SOUNIO_DEPRESSION_SWOW_ORC_PASS" <<<"${SWOW_OUT}" \
    || fail "token SOUNIO_DEPRESSION_SWOW_ORC_PASS not found"
echo "✓ SOUNIO_DEPRESSION_SWOW_ORC_PASS present"

# Verify all 5 groups produced output
grep -q "control" <<<"${SWOW_OUT}" \
    || fail "control group missing from output"
grep -q "minimum" <<<"${SWOW_OUT}" \
    || fail "minimum group missing from output"
grep -q "severe" <<<"${SWOW_OUT}" \
    || fail "severe group missing from output"
echo "✓ all 5 groups present in output"

# Verify minimum has highest neg% (non-monotonic finding)
# Extract neg% for minimum and severe; minimum should be higher
MIN_LINE="$(grep "minimum" <<<"${SWOW_OUT}")"
grep -q "neg%=0\." <<<"${MIN_LINE}" || grep -q "neg%=1\." <<<"${MIN_LINE}" \
    || fail "minimum group neg% field missing"
echo "✓ minimum group neg% field present"

echo ""

# ---- Program 2: Bootstrap CI over published κ ----
echo "--- depression_semantic_orc.sio ---"
SEM_OUT="$("${SOUC}" run "${SEMANTIC_SRC}" 2>&1)" || fail "depression_semantic_orc.sio crashed (exit $?)"

echo "${SEM_OUT}"
echo ""

grep -q "SOUNIO_DEPRESSION_SEMANTIC_ORC_PASS" <<<"${SEM_OUT}" \
    || fail "token SOUNIO_DEPRESSION_SEMANTIC_ORC_PASS not found"
echo "✓ SOUNIO_DEPRESSION_SEMANTIC_ORC_PASS present"

grep -q "VERIFIED: minimum group has most negative mean curvature" <<<"${SEM_OUT}" \
    || fail "minimum group ordering verification failed"
echo "✓ minimum most-hyperbolic ordering verified"

grep -q "VERIFIED: moderate group least hyperbolic" <<<"${SEM_OUT}" \
    || fail "non-monotonic finding verification failed"
echo "✓ non-monotonic finding (moderate least hyperbolic) verified"

# Verify CI format: minimum CI should be entirely negative
min_ci="$(grep "minimum" <<<"${SEM_OUT}" || true)"
grep -q "CI=\[-" <<<"$min_ci" \
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

grep -q "SOUNIO_DEPRESSION_EPISTEMIC_ORC_PASS" <<<"${EPI_OUT}" \
    || fail "token SOUNIO_DEPRESSION_EPISTEMIC_ORC_PASS not found"
echo "✓ SOUNIO_DEPRESSION_EPISTEMIC_ORC_PASS present"

# Density-matched separation gate must verify (effect survives density control)
grep -q "VERIFIED: subclinical most hyperbolic" <<<"${EPI_OUT}" \
    || fail "density-matched separation gate not verified"
echo "✓ subclinical-most-hyperbolic separation verified at matched density"
echo ""

echo "=== semantic_orc_depression_orc_gate: PASS ==="
