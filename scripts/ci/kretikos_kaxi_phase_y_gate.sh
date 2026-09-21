#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_phase_y_gate.sh
#
# Phase Y gate: true GUM variance propagation through the 2-comp PBPK ODE.
#
# Dissertation contribution #1: GUM-through-ODE (JCGM 100:2008).
# The GPU kernel tracks the full 2×2 covariance matrix (V11, V12, V22)
# at each Euler step via J·Σ·Jᵀ. V12 is a per-thread local register
# that builds from 0; omitting it would underestimate V11_final by ~46%
# at these ODE constants.
#
# Truth claims:
#   TC-1  GPU C1_final FNV digest == CPU analytic C1_final digest
#   TC-2  GPU V11_final FNV digest == CPU analytic V11_final digest
#   TC-3  ISO 17025-style uncertainty budget CSV emitted; u_c > 0; u95_k2 > 0
#
# Scale: PHY_COHORT patients (default 10M) — correctness focus, not throughput.
#
# Knobs (env):
#   PHY_COHORT        total patients (default 10000000)
#   PHY_THREADS       threads/block (default 128)
#   PHY_SEED          sampler seed (default 42)
#   PHY_PTX           PTX to load (default tests/golden/kaxi_ptx/f32_gum/pbpk_2comp_gum_4step.ptx)
#   PHY_STAGE_DIR     working directory (default mktemp)
#
# Skip with: SOUNIO_KAXI_PHASE_Y_GATE_SKIP=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_KAXI_PHASE_Y_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_phase_y_gate: SKIPPED (SOUNIO_KAXI_PHASE_Y_GATE_SKIP=1)"
  exit 0
fi

PHY_COHORT="${PHY_COHORT:-10000000}"
PHY_THREADS="${PHY_THREADS:-128}"
PHY_SEED="${PHY_SEED:-42}"
PHY_PTX="${PHY_PTX:-${ROOT_DIR}/tests/golden/kaxi_ptx/f32_gum/pbpk_2comp_gum_4step.ptx}"

# ---------- prerequisite checks ----------

if [[ ! -f "${PHY_PTX}" ]]; then
  echo "kretikos_kaxi_phase_y_gate: FAIL — PTX missing: ${PHY_PTX}"
  exit 1
fi
if ! command -v cc >/dev/null 2>&1; then
  echo "kretikos_kaxi_phase_y_gate: SKIPPED (cc missing)"
  exit 0
fi
# `find | grep -q` EPIPEs when CUDA exists (many matches, find is still
# writing when grep -q exits) and pipefail turns that into "not found" --
# the arm selector concluded no-CUDA exactly when CUDA was present. Find
# stops itself now (-print -quit), and ldconfig's output is captured whole.
if ! grep -q libcuda <<<"$(ldconfig -p 2>/dev/null || true)" &&
   [[ -z "$(find /usr/lib /usr/local/lib -name "libcuda.so*" -maxdepth 3 -print -quit 2>/dev/null || true)" ]]; then
  echo "kretikos_kaxi_phase_y_gate: SKIPPED (libcuda not found)"
  exit 0
fi

STAGE_DIR="${PHY_STAGE_DIR:-$(mktemp -d /tmp/phy_gate_XXXXXX)}"
mkdir -p "${STAGE_DIR}"

# ---------- [1/4] build sampler + runner ----------

echo "[1/4] building GUM sampler and runner"
cc -O2 -Wall \
   "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_2comp_gum_sampler.c" \
   -lm -o "${STAGE_DIR}/gum_sampler"
cc -O2 -Wall \
   "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c" -ldl -lm \
   -o "${STAGE_DIR}/runner"
echo "  sampler and runner built"

# ---------- [2/4] CPU analytic (sampler generates truth + binary inputs) ----------

echo "[2/4] CPU analytic — generating ${PHY_COHORT} patients"
"${STAGE_DIR}/gum_sampler" \
  --out-dir "${STAGE_DIR}" \
  --cohort "${PHY_COHORT}" \
  --seed "${PHY_SEED}" \
  --mean-dose 1.0 \
  --dose-sigma 0.50 \
  --frac-c2 0.1

echo "  expected.summary written"

# Read expected digests
extract_field() { grep "^${1}=" "${2}" | cut -d= -f2-; }
SUMMARY="${STAGE_DIR}/expected.summary"
C1F_EXPECTED="$(extract_field c1_final_digest "${SUMMARY}")"
V11F_EXPECTED="$(extract_field v11_final_digest "${SUMMARY}")"
MEAN_U_B="$(extract_field mean_u_b "${SUMMARY}")"
U95_K2="$(extract_field u95_k2 "${SUMMARY}")"
PCT_TYPE_B="$(extract_field pct_type_b "${SUMMARY}")"
PCT_TYPE_A="$(extract_field pct_type_a "${SUMMARY}")"
STDEV_C1_POP="$(extract_field stdev_c1_pop "${SUMMARY}")"
U_C="$(extract_field u_c "${SUMMARY}")"

echo "  c1_final_digest  = ${C1F_EXPECTED}"
echo "  v11_final_digest = ${V11F_EXPECTED}"
echo "  mean_u_b (GUM)   = ${MEAN_U_B}"
echo "  u95 (k=2)        = ${U95_K2}"

# ---------- [3/4] GPU run ----------

echo "[3/4] GPU run — ${PHY_COHORT} patients, ${PHY_THREADS} threads/block"
PHY_BLOCKS=$(( (PHY_COHORT + PHY_THREADS - 1) / PHY_THREADS ))

GPU_OUT="${STAGE_DIR}/gpu_output.txt"
"${STAGE_DIR}/runner" "${PHY_PTX}" \
  --kernel kaxi_kernel \
  --gum --type f32 \
  --threads "${PHY_THREADS}" \
  --blocks "${PHY_BLOCKS}" \
  --cohort-size "${PHY_COHORT}" \
  --print-count 0 \
  --init-file    "${STAGE_DIR}/init_c1.bin" \
  --init-var-file "${STAGE_DIR}/init_c2.bin" \
  --init-v11-file "${STAGE_DIR}/init_v11.bin" \
  --init-v22-file "${STAGE_DIR}/init_v22.bin" \
  2>&1 | tee "${GPU_OUT}"

echo "  GPU run complete"

# Extract GPU digests from PHY line
GPU_C1_DIG="$(grep '^PHY ' "${GPU_OUT}" | grep -o 'c1_digest=[0-9a-f]*' | cut -d= -f2 || true)"
GPU_V11_DIG="$(grep '^PHY ' "${GPU_OUT}" | grep -o 'v11_digest=[0-9a-f]*' | cut -d= -f2 || true)"
GPU_MEAN_U_B="$(grep '^PHY ' "${GPU_OUT}" | grep -o 'mean_u_b=[^ ]*' | cut -d= -f2 || true)"
GPU_U95="$(grep '^PHY ' "${GPU_OUT}" | grep -o 'u95_k2=[^ ]*' | cut -d= -f2 || true)"

echo "  GPU c1_digest    = ${GPU_C1_DIG}"
echo "  GPU v11_digest   = ${GPU_V11_DIG}"

# ---------- [4/4] truth claims ----------

echo "[4/4] truth claims"
fail=0

# TC-1: C1_final digest match
if [[ "${GPU_C1_DIG}" == "${C1F_EXPECTED}" ]]; then
  echo "  TC-1 PASS: GPU C1_final == CPU analytic C1_final (exact f32, FNV64)"
else
  echo "  TC-1 FAIL: C1_final mismatch"
  echo "    expected: ${C1F_EXPECTED}"
  echo "    got:      ${GPU_C1_DIG}"
  fail=1
fi

# TC-2: V11_final digest match (the GUM variance truth claim)
if [[ "${GPU_V11_DIG}" == "${V11F_EXPECTED}" ]]; then
  echo "  TC-2 PASS: GPU V11_final == CPU analytic GUM V11_final (exact f32, FNV64)"
else
  echo "  TC-2 FAIL: V11_final mismatch — GUM propagation diverges"
  echo "    expected: ${V11F_EXPECTED}"
  echo "    got:      ${GPU_V11_DIG}"
  fail=1
fi

# TC-3: ISO budget is well-formed (u_c > 0 and u95 > 0)
TC3_FAIL=0
if [[ -z "${U95_K2}" ]] || [[ "${U95_K2}" == "0" ]] || [[ "${U95_K2}" == "0.000000" ]]; then
  echo "  TC-3 FAIL: u95_k2 is zero or missing"
  TC3_FAIL=1
fi
if [[ -z "${U_C}" ]] || [[ "${U_C}" == "0" ]]; then
  echo "  TC-3 FAIL: u_c is zero"
  TC3_FAIL=1
fi
if [[ "${TC3_FAIL}" -eq 0 ]]; then
  echo "  TC-3 PASS: ISO budget well-formed"
fi
fail=$(( fail + TC3_FAIL ))

# Emit ISO 17025-style uncertainty budget CSV
BUDGET_CSV="${ROOT_DIR}/benchmarks/pbpk/gum_budget.csv"
mkdir -p "$(dirname "${BUDGET_CSV}")"
cat > "${BUDGET_CSV}" << CSV
# ISO 17025 GUM Uncertainty Budget — Rapamycin 2-Comp PBPK (Phase Y)
# Drug: rapamycin (Cypher stent), ODE: 4-step Euler, Jacobian propagation
# Generated by kretikos_kaxi_phase_y_gate.sh
# Cohort: ${PHY_COHORT} patients, seed: ${PHY_SEED}
component,source,type,u,u_squared,contribution_pct
C1_init,dosing_variability,B,${MEAN_U_B},<computed>,${PCT_TYPE_B}
C1_final_pop,population_variation,A,${STDEV_C1_POP},<computed>,${PCT_TYPE_A}
combined,-,-,${U_C},<computed>,100.0
expanded_k2,-,-,${U95_K2},-,-
CSV
echo "  ISO budget written to benchmarks/pbpk/gum_budget.csv"
echo "    Type B (dosing)     u_B = ${MEAN_U_B}  (${PCT_TYPE_B}%)"
echo "    Type A (population) u_A = ${STDEV_C1_POP}  (${PCT_TYPE_A}%)"
echo "    Combined            u_c = ${U_C}"
echo "    Expanded (k=2)      U95 = ${U95_K2}"

if [[ "${fail}" -ne 0 ]]; then
  echo "kretikos_kaxi_phase_y_gate: FAIL (${fail} truth claim(s) failed)"
  exit 1
fi
echo "kretikos_kaxi_phase_y_gate: PASS (3/3 truth claims)"
