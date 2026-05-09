#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_phase_w_gate.sh
#
# Phase W gate: streamed multi-launch over a 1M-patient cohort. Validates
# mem_digest invariance across (streams × chunks) combinations against the
# reference (streams=1, chunks=64) — proves CUDA streams give bit-exact
# correctness equivalent to a single default-stream launch at clinical scale.
#
# SCOPE NOTE: This is a CORRECTNESS gate, not a throughput gate. The current
# Phase W runner does H2D synchronously up front, so streams overlap only
# launch+D2H. Empirically wall-clock is flat across stream count (kernel is
# 0.04 ms per chunk; D2H of u64 buffers leaves nothing meaningful to overlap).
# Genuine concurrent throughput requires H2D async + cuMemHostAlloc pinned
# staging — deferred to Phase W.1. The gate enforces a wall-clock CEILING
# (PHW_WALL_CEILING_US) only as a regression detector against future launch
# overhead growth, not as a positive throughput claim.
#
# Run modes (auto-detected):
#   1. Direct A5000: if `nvidia-smi` is available and a libcuda.so is loadable,
#      build runner+sampler locally, run all variants directly.
#   2. Slurm fallback: submit via slurm-jobs/kretikos/submit-kaxi-ptx-launch.sh
#      against the gpu-orangefs partition with --expected-digest.
#
# Skip with: SOUNIO_KAXI_PHASE_W_GATE_SKIP=1
#
# Knobs (env):
#   PHW_COHORT      cohort size (default 1000000)
#   PHW_THREADS     threads per block (default 32)
#   PHW_SEED        sampler seed (default 42)
#   PHW_PTX         PTX path (default tests/golden/kaxi_ptx/epistemic/vec_sqrt_gate_var_mb.ptx)
#   PHW_STAGE_DIR   working directory (default mktemp)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_KAXI_PHASE_W_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_phase_w_gate: SKIPPED (SOUNIO_KAXI_PHASE_W_GATE_SKIP=1)"
  exit 0
fi

PHW_COHORT="${PHW_COHORT:-1000000}"
PHW_THREADS="${PHW_THREADS:-32}"
PHW_SEED="${PHW_SEED:-42}"
PHW_PTX="${PHW_PTX:-${ROOT_DIR}/tests/golden/kaxi_ptx/epistemic/vec_sqrt_gate_var_mb.ptx}"
PHW_STAGE_DIR="${PHW_STAGE_DIR:-$(mktemp -d /tmp/phase_w_gate.XXXXXX)}"
# Wall-clock ceiling for the worst-observed config (streams×chunks=1024). On
# RTX A5000 the 1M-patient run lands at ~28 ms; ceiling is set generously to
# catch ≥2× regressions in launch overhead without flapping on cluster jitter.
PHW_WALL_CEILING_US="${PHW_WALL_CEILING_US:-60000}"

if [[ ! -f "${PHW_PTX}" ]]; then
  echo "kretikos_kaxi_phase_w_gate: FAIL — PTX missing: ${PHW_PTX}"
  exit 1
fi
if ! command -v cc >/dev/null 2>&1; then
  echo "kretikos_kaxi_phase_w_gate: SKIPPED (cc missing)"
  exit 0
fi

mkdir -p "${PHW_STAGE_DIR}"

echo "[1/5] building sampler + runner"
cc -O2 -Wall "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_sampler.c" -lm -o "${PHW_STAGE_DIR}/sampler"
cc -O2 -Wall "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c"   -ldl -o "${PHW_STAGE_DIR}/runner"

echo "[2/5] sampling cohort=${PHW_COHORT} seed=${PHW_SEED}"
"${PHW_STAGE_DIR}/sampler" --out-dir "${PHW_STAGE_DIR}" --cohort "${PHW_COHORT}" --seed "${PHW_SEED}" >/dev/null

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "kretikos_kaxi_phase_w_gate: SKIPPED (nvidia-smi missing — slurm path not auto-invoked here)"
  exit 0
fi
if ! "${PHW_STAGE_DIR}/runner" >/dev/null 2>&1; then : ; fi  # warm dlopen

run_one() {
  local streams="$1" chunks="$2"
  "${PHW_STAGE_DIR}/runner" "${PHW_PTX}" \
    --kernel kaxi_kernel --epistemic --type f64 \
    --cohort-size "${PHW_COHORT}" --threads "${PHW_THREADS}" \
    --streams "${streams}" --chunks "${chunks}" \
    --init-file "${PHW_STAGE_DIR}/init.mem.bin" \
    --init-var-file "${PHW_STAGE_DIR}/init.var.bin" 2>&1
}

extract_digest() {
  echo "$1" | grep '^PHW' | tail -1 | sed -n 's/.*mem_digest=\([0-9a-f]*\).*/\1/p'
}
extract_wall() {
  echo "$1" | grep '^PHW' | tail -1 | sed -n 's/.*wall_us=\([0-9]*\).*/\1/p'
}

echo "[3/5] reference run (streams=1 chunks=64)"
ref_out="$(run_one 1 64)"
ref_dig="$(extract_digest "${ref_out}")"
ref_wall="$(extract_wall "${ref_out}")"
if [[ -z "${ref_dig}" ]]; then
  echo "kretikos_kaxi_phase_w_gate: FAIL — reference run produced no PHW line"
  echo "${ref_out}"
  exit 1
fi
echo "  reference: mem_digest=${ref_dig} wall_us=${ref_wall}"

echo "[4/5] streamed equivalence matrix"
fail=0
declare -a results=()
for streams in 1 2 4 8; do
  for chunks in 64 256 1024; do
    out="$(run_one "${streams}" "${chunks}")"
    dig="$(extract_digest "${out}")"
    wall="$(extract_wall "${out}")"
    if [[ "${dig}" == "${ref_dig}" ]]; then
      results+=("ok streams=${streams} chunks=${chunks} digest=${dig} wall_us=${wall}")
    else
      results+=("MISMATCH streams=${streams} chunks=${chunks} got=${dig} expected=${ref_dig} wall_us=${wall}")
      fail=$((fail + 1))
    fi
  done
done
for r in "${results[@]}"; do echo "  ${r}"; done

echo "[5/5] summary"
total=${#results[@]}
passed=$((total - fail))
echo "  cohort=${PHW_COHORT} reference_mem_digest=${ref_dig}"
echo "  passed=${passed}/${total} failed=${fail}"

# Wall-clock regression ceiling on the worst-case config in the matrix.
max_wall=0
for r in "${results[@]}"; do
  w="$(echo "${r}" | sed -n 's/.*wall_us=\([0-9]*\).*/\1/p')"
  [[ -n "${w}" && "${w}" -gt "${max_wall}" ]] && max_wall="${w}"
done
echo "  max_wall_us=${max_wall} ceiling_us=${PHW_WALL_CEILING_US}"
wall_fail=0
if [[ "${max_wall}" -gt "${PHW_WALL_CEILING_US}" ]]; then
  echo "  REGRESSION: max_wall_us=${max_wall} > ceiling=${PHW_WALL_CEILING_US}"
  wall_fail=1
fi

if [[ "${fail}" -ne 0 || "${wall_fail}" -ne 0 ]]; then
  echo "kretikos_kaxi_phase_w_gate: FAIL"
  exit 1
fi
echo "kretikos_kaxi_phase_w_gate: PASS"
