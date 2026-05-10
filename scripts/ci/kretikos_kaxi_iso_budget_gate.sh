#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_iso_budget_gate.sh
#
# Dissertation contribution #3: ISO 17025-style GUM uncertainty budget for
# rapamycin 2-comp PBPK.
#
# This gate is CPU-only (no GPU required). It runs the GUM sampler to produce
# a metrological uncertainty budget and validates four truth claims:
#
#   TC-1  combined uncertainty u_c > 0
#   TC-2  expanded uncertainty U95 (k=2) > 0 and U95 == 2*u_c (exact by construction)
#   TC-3  Type A + Type B percentage contributions sum to 100% (+/- 1%)
#   TC-4  ISO 17025 budget CSV written with all required columns and well-formed rows
#
# The budget format follows ISO 17025:2017 / GUM (JCGM 100:2008) metrological
# conventions for uncertainty reporting:
#   component   - uncertainty source identifier
#   source      - physical/statistical source description
#   type        - A (statistical) or B (non-statistical / prior)
#   u           - standard uncertainty (same units as measurand)
#   u_squared   - variance contribution
#   contribution_pct - percentage of combined variance
#
# Knobs (env):
#   ISO_COHORT       total patients (default 1000000 -- 1M, ~1s)
#   ISO_SEED         sampler seed (default 42)
#   ISO_STAGE_DIR    working directory (default mktemp)
#   SOUNIO_ISO_BUDGET_GATE_SKIP=1  skip entirely

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_ISO_BUDGET_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_iso_budget_gate: SKIPPED (SOUNIO_ISO_BUDGET_GATE_SKIP=1)"
  exit 0
fi

if ! command -v cc >/dev/null 2>&1; then
  echo "kretikos_kaxi_iso_budget_gate: SKIPPED (cc missing)"
  exit 0
fi

ISO_COHORT="${ISO_COHORT:-1000000}"
ISO_SEED="${ISO_SEED:-42}"
STAGE_DIR="${ISO_STAGE_DIR:-$(mktemp -d /tmp/iso_budget_gate_XXXXXX)}"
mkdir -p "${STAGE_DIR}"

echo "=== ISO 17025 GUM Uncertainty Budget Gate ==="
echo "  cohort=${ISO_COHORT}  seed=${ISO_SEED}"

# ---------- [1/4] build GUM sampler ----------

echo "[1/4] building GUM sampler (CPU-only)"
cc -O2 -Wall \
   "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_2comp_gum_sampler.c" \
   -lm -o "${STAGE_DIR}/gum_sampler"
echo "  sampler built"

# ---------- [2/4] run CPU analytic ----------

echo "[2/4] CPU analytic -- ${ISO_COHORT} patients"
"${STAGE_DIR}/gum_sampler" \
  --out-dir "${STAGE_DIR}" \
  --cohort  "${ISO_COHORT}" \
  --seed    "${ISO_SEED}" \
  --mean-dose 1.0 \
  --dose-sigma 0.50 \
  --frac-c2 0.1
echo "  expected.summary written"

extract_field() { grep "^${1}=" "${2}" | cut -d= -f2-; }
SUMMARY="${STAGE_DIR}/expected.summary"

U_C="$(extract_field u_c        "${SUMMARY}")"
U95="$(extract_field u95_k2     "${SUMMARY}")"
PCT_B="$(extract_field pct_type_b "${SUMMARY}")"
PCT_A="$(extract_field pct_type_a "${SUMMARY}")"
MEAN_U_B="$(extract_field mean_u_b "${SUMMARY}")"
STDEV_C1="$(extract_field stdev_c1_pop "${SUMMARY}")"

echo "  u_c        = ${U_C}"
echo "  U95 (k=2)  = ${U95}"
echo "  pct_type_b = ${PCT_B}%  pct_type_a = ${PCT_A}%"

# ---------- [3/4] write ISO 17025 budget CSV ----------

echo "[3/4] writing ISO 17025 budget CSV"
BUDGET_CSV="${STAGE_DIR}/gum_budget.iso17025.csv"

# Compute u_squared values (awk for portability)
U_B_SQ="$(awk -v u="${MEAN_U_B}" 'BEGIN { printf "%.17g", u*u }')"
U_A_SQ="$(awk -v u="${STDEV_C1}" 'BEGIN { printf "%.17g", u*u }')"
U_C_SQ="$(awk -v u="${U_C}"      'BEGIN { printf "%.17g", u*u }')"

cat > "${BUDGET_CSV}" << CSV
# ISO 17025 GUM Uncertainty Budget -- Rapamycin 2-Comp PBPK (Dissertation #3)
# Drug: rapamycin (sirolimus), ODE: 2-comp 4-step Euler, J*Sigma*J^T propagation
# Standard: JCGM 100:2008 (GUM), reported per ISO 17025:2017 S 7.6
# Cohort: ${ISO_COHORT} patients, seed: ${ISO_SEED}
# Coverage factor: k=2 (approx 95% coverage, Gaussian assumption)
component,source,type,u,u_squared,contribution_pct
C1_init,dosing_variability_lognormal,B,${MEAN_U_B},${U_B_SQ},${PCT_B}
C1_final_pop,population_pharmacokinetic_variation,A,${STDEV_C1},${U_A_SQ},${PCT_A}
combined,GUM_quadrature,-,${U_C},${U_C_SQ},100.0
expanded_k2,k=2_coverage_factor,-,${U95},-,-
CSV

echo "  ISO budget written: ${BUDGET_CSV}"
echo ""
echo "  +------------------------------------------+"
echo "  |  GUM Uncertainty Budget (ISO 17025:2017) |"
echo "  |  Rapamycin 2-comp PBPK, 4-step Euler ODE |"
echo "  +------------------------------------------+"
printf "  |  %-22s %10s  |\n" "Source" "u (std)"
echo "  +------------------------------------------+"
printf "  |  %-22s %10s  |\n" "Type B (dosing)"       "${MEAN_U_B:0:10}"
printf "  |  %-22s %10s  |\n" "Type A (population)"   "${STDEV_C1:0:10}"
echo "  +------------------------------------------+"
printf "  |  %-22s %10s  |\n" "Combined u_c"           "${U_C:0:10}"
printf "  |  %-22s %10s  |\n" "Expanded U (k=2)"       "${U95:0:10}"
echo "  +------------------------------------------+"

# ---------- [4/4] truth claims ----------

echo "[4/4] truth claims"
fail=0

# TC-1: u_c > 0
if awk -v u="${U_C}" 'BEGIN { exit (u > 0) ? 0 : 1 }'; then
  echo "  TC-1 PASS: u_c > 0 (u_c=${U_C})"
else
  echo "  TC-1 FAIL: u_c is zero or negative (u_c=${U_C})"
  fail=1
fi

# TC-2: U95 > 0 and U95 == 2*u_c exactly (k=2 by construction)
if awk -v u95="${U95}" -v uc="${U_C}" 'BEGIN {
  exit (u95 > 0 && (u95 >= 1.999*uc) && (u95 <= 2.001*uc)) ? 0 : 1
}'; then
  echo "  TC-2 PASS: U95 > 0 and U95 == 2*u_c (k=2)"
else
  echo "  TC-2 FAIL: U95 invalid (U95=${U95}, expected 2*u_c=2*${U_C})"
  fail=1
fi

# TC-3: pct_type_b + pct_type_a in [99.0, 101.0]
if awk -v a="${PCT_A}" -v b="${PCT_B}" 'BEGIN {
  sum = a + b
  exit (sum >= 99.0 && sum <= 101.0) ? 0 : 1
}'; then
  echo "  TC-3 PASS: Type A + Type B = $(awk -v a="${PCT_A}" -v b="${PCT_B}" 'BEGIN { printf "%.4f", a+b }')% (100 +/- 1%)"
else
  echo "  TC-3 FAIL: Type A + B do not sum to 100% (A=${PCT_A} B=${PCT_B})"
  fail=1
fi

# TC-4: ISO CSV has all required columns and at least 4 data rows
CSV_COLS="component,source,type,u,u_squared,contribution_pct"
HEADER_LINE="$(grep "^component," "${BUDGET_CSV}" || true)"
DATA_ROWS="$(grep -c "^[A-Za-z]" "${BUDGET_CSV}" || echo 0)"

if [[ "${HEADER_LINE}" == "${CSV_COLS}" ]] && [[ "${DATA_ROWS}" -ge 4 ]]; then
  echo "  TC-4 PASS: ISO CSV well-formed (header OK, ${DATA_ROWS} data rows)"
else
  echo "  TC-4 FAIL: CSV header='${HEADER_LINE}' (expected '${CSV_COLS}'), rows=${DATA_ROWS}"
  fail=1
fi

# Also copy to repo benchmarks dir for dissertation reference
REPO_CSV="${ROOT_DIR}/benchmarks/pbpk/gum_budget.csv"
mkdir -p "$(dirname "${REPO_CSV}")"
cp "${BUDGET_CSV}" "${REPO_CSV}"
echo "  budget copied to benchmarks/pbpk/gum_budget.csv"

if [[ "${fail}" -ne 0 ]]; then
  echo "kretikos_kaxi_iso_budget_gate: FAIL (${fail} truth claim(s) failed)"
  exit 1
fi
echo "kretikos_kaxi_iso_budget_gate: PASS (4/4 truth claims)"
