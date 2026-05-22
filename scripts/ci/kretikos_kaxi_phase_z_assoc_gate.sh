#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_phase_z_assoc_gate.sh
#
# Phase Z gate (L1 SCAFFOLD): associator-augmented 8-compartment PBPK.
#
# This is the engineering scaffold for the AssociatorField butterfly thread.
# It extends Phase Y (2-comp GUM covariance) to an 8-compartment octonion-indexed
# model carrying ONE structural-uncertainty scalar per patient:
#
#   aug = KAPPA * ||[A,B,C]||²   (octonion associator of cyclic-shift sub-doses)
#
# mirroring stdlib/algebra/associator_field.sio::af_augmented_variance.
#
# CPU truth claims (runnable today, no GPU required):
#   TC-1  sampler is deterministic: identical seed -> identical FNV digests + aug_sum
#   TC-2  ASSOCIATIVE invariant: with --associative (all mass in real slot c0),
#         aug == 0 for every patient (aug_sum == 0, aug_zero == cohort).
#         This is the L0 theorem (associative inputs carry no order-uncertainty).
#   TC-3  NON-associative cohort has strictly positive aug_sum.
#
# GPU truth claim (DEFERRED, multi-session):
#   TC-4  GPU PTX kernel aug digest == CPU aug digest.
#         Requires tests/golden/kaxi_ptx/f32_assoc_gum/pbpk_8comp_assoc.ptx,
#         which does not exist yet (K-AXI emitter extension is the heavy lift).
#         The gate SKIPS this claim cleanly when the PTX is absent.
#
# Knobs (env):
#   PZ_COHORT     patients (default 100000 -- correctness focus, not throughput)
#   PZ_SEED       sampler seed (default 42)
#   PZ_PTX        GPU kernel PTX (default tests/golden/kaxi_ptx/f32_assoc_gum/pbpk_8comp_assoc.ptx)
#   PZ_STAGE_DIR  working directory (default mktemp)
#
# Skip with: SOUNIO_KAXI_PHASE_Z_GATE_SKIP=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_KAXI_PHASE_Z_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_phase_z_assoc_gate: SKIPPED (SOUNIO_KAXI_PHASE_Z_GATE_SKIP=1)"
  exit 0
fi

PZ_COHORT="${PZ_COHORT:-100000}"
PZ_SEED="${PZ_SEED:-42}"
PZ_PTX="${PZ_PTX:-${ROOT_DIR}/tests/golden/kaxi_ptx/f32_assoc_gum/pbpk_8comp_assoc.ptx}"
SAMPLER_SRC="${ROOT_DIR}/scripts/gpu/kaxi_pbpk_8comp_assoc_sampler.c"

if ! command -v cc >/dev/null 2>&1; then
  echo "kretikos_kaxi_phase_z_assoc_gate: SKIPPED (cc missing)"
  exit 0
fi
if [[ ! -f "${SAMPLER_SRC}" ]]; then
  echo "kretikos_kaxi_phase_z_assoc_gate: FAIL -- sampler missing: ${SAMPLER_SRC}"
  exit 1
fi

STAGE_DIR="${PZ_STAGE_DIR:-$(mktemp -d)}"
SAMPLER_BIN="${STAGE_DIR}/pz_sampler"
OUT_A="${STAGE_DIR}/run_a"
OUT_B="${STAGE_DIR}/run_b"
OUT_ASSOC="${STAGE_DIR}/run_assoc"
mkdir -p "${OUT_A}" "${OUT_B}" "${OUT_ASSOC}"

echo "kretikos_kaxi_phase_z_assoc_gate: cohort=${PZ_COHORT} seed=${PZ_SEED}"

cc -O2 "${SAMPLER_SRC}" -lm -o "${SAMPLER_BIN}"

# --- TC-1: determinism (same seed twice) ---
"${SAMPLER_BIN}" --out-dir "${OUT_A}" --cohort "${PZ_COHORT}" --seed "${PZ_SEED}" >/dev/null
"${SAMPLER_BIN}" --out-dir "${OUT_B}" --cohort "${PZ_COHORT}" --seed "${PZ_SEED}" >/dev/null
if diff -q "${OUT_A}/expected.summary" "${OUT_B}/expected.summary" >/dev/null; then
  echo "  TC-1 determinism: PASS"
else
  echo "  TC-1 determinism: FAIL"; exit 1
fi

# --- TC-2: associative invariant (aug == 0 everywhere) ---
ASSOC_LINE="$("${SAMPLER_BIN}" --out-dir "${OUT_ASSOC}" --cohort "${PZ_COHORT}" --seed "${PZ_SEED}" --associative)"
AUG_SUM_ASSOC="$(echo "${ASSOC_LINE}" | sed -n 's/.*aug_sum=\([^ ]*\).*/\1/p')"
AUG_ZERO_ASSOC="$(echo "${ASSOC_LINE}" | sed -n 's/.*aug_zero=\([^ ]*\).*/\1/p')"
if [[ "${AUG_SUM_ASSOC}" == "0" && "${AUG_ZERO_ASSOC}" == "${PZ_COHORT}" ]]; then
  echo "  TC-2 associative invariant (aug=0): PASS"
else
  echo "  TC-2 associative invariant: FAIL (aug_sum=${AUG_SUM_ASSOC} aug_zero=${AUG_ZERO_ASSOC})"; exit 1
fi

# --- TC-3: non-associative cohort has positive augmentation ---
NONA_AUG="$(grep -o 'aug_sum [0-9.eE+-]*' "${OUT_A}/expected.summary" | awk '{print $2}')"
if awk "BEGIN{exit !(${NONA_AUG} > 0)}"; then
  echo "  TC-3 nonassoc aug_sum>0: PASS (${NONA_AUG})"
else
  echo "  TC-3 nonassoc aug_sum>0: FAIL (${NONA_AUG})"; exit 1
fi

# --- TC-4: GPU PTX kernel parity (DEFERRED) ---
if [[ -f "${PZ_PTX}" ]]; then
  echo "  TC-4 GPU PTX parity: PTX present (${PZ_PTX}) -- runner wiring is the next L1 step"
  # TODO(L1): build /tmp/kaxi_ptx_runner, load PZ_PTX with 8 mem buffers + aug,
  #           compare device aug digest to ${OUT_A} CPU digest. Currently the
  #           K-AXI -> PTX emitter for AssociatorField arithmetic is unimplemented.
  echo "  TC-4 GPU PTX parity: SKIPPED (runner wiring deferred)"
else
  echo "  TC-4 GPU PTX parity: SKIPPED (kernel not built yet: ${PZ_PTX})"
fi

echo "kretikos_kaxi_phase_z_assoc_gate: PASS (CPU scaffold; GPU kernel deferred)"
