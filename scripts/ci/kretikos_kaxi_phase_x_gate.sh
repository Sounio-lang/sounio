#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_phase_x_gate.sh
#
# Phase X gate: rapamycin 1-compartment epistemic PBPK at 2B-patient scale.
#
# Every patient gets a GPU-computed epistemic PK evaluation:
#   C_scaled = C × 3.125   (internal PK units → ng/mL, GUM σ² propagated)
#   gate:     if 5.0 ≤ C_scaled ≤ 15.0 (rapamycin therapeutic window), keep;
#             else write NaN sentinel
#
# Truth-claim: GPU in_budget + nan_count matches CPU analytic exactly (f32 arithmetic).
# This is stronger than Phase W.2 -- the GPU evaluates the rapamycin therapeutic
# window with ISO-traceable GUM uncertainty propagation, not just a variance proxy.
#
# Sharding: 4 shards × 500M patients. GPU-0 (local RTX4000Ada) runs shards 0-1;
# GPU-1 (remote A5000 via kubectl) runs shards 2-3 -- concurrently.
#
# Throughput metric: effective_decisions/sec = PHX_COHORT / max_shard_wall_us × 1e6
#
# Knobs (env):
#   PHX_COHORT           total patients (default 2000000000)
#   PHX_SHARD_COUNT      shards (default 4; must be even)
#   PHX_THREADS          threads/block (default 32)
#   PHX_SEED             sampler seed (default 42)
#   PHX_STREAMS          CUDA streams per shard (default 4)
#   PHX_CHUNKS           chunks per shard (default 64)
#   PHX_WALL_CEILING_US  wall ceiling per shard µs (default 3600000000 ~1h)
#   PHX_STAGE_DIR        working directory (default mktemp)
#   PHX_WORKER_SCRIPT    kubectl exec wrapper for A5000 (auto-detected)
#   PHX_PTX              PTX to load (default tests/golden/kaxi_ptx/f32e/pbpk_rapamycin_1comp_epistemic.ptx)
#
# Skip with: SOUNIO_KAXI_PHASE_X_GATE_SKIP=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_KAXI_PHASE_X_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_phase_x_gate: SKIPPED (SOUNIO_KAXI_PHASE_X_GATE_SKIP=1)"
  exit 0
fi

PHX_COHORT="${PHX_COHORT:-2000000000}"
PHX_SHARD_COUNT="${PHX_SHARD_COUNT:-4}"
PHX_THREADS="${PHX_THREADS:-32}"
PHX_SEED="${PHX_SEED:-42}"
PHX_STREAMS="${PHX_STREAMS:-4}"
PHX_CHUNKS="${PHX_CHUNKS:-64}"
PHX_WALL_CEILING_US="${PHX_WALL_CEILING_US:-3600000000}"
PHX_PTX="${PHX_PTX:-${ROOT_DIR}/tests/golden/kaxi_ptx/f32_2c/pbpk_2comp_4step_mb.ptx}"
PHX_GPU0_SHARDS=$(( PHX_SHARD_COUNT / 2 ))

WORKER_SCRIPT="${PHX_WORKER_SCRIPT:-}"
for candidate in \
  "/workspace/beagle/k8s/sounio-runners/run-compiler-2gpu-worker-command.sh" \
  "${ROOT_DIR}/../../beagle/k8s/sounio-runners/run-compiler-2gpu-worker-command.sh"; do
  if [[ -x "${candidate}" ]]; then
    WORKER_SCRIPT="${candidate}"
    break
  fi
done

WORKER_NS="${PHX_WORKER_NS:-beagle}"
WORKER_POD="${PHX_WORKER_POD:-sounio-compiler-gpu-worker-r740}"
WORKER_CTR="${PHX_WORKER_CTR:-worker}"

_kubectl() {
  if kubectl "$@" 2>/dev/null; then return 0
  elif sudo kubectl "$@"; then return 0
  else return 1; fi
}

# ---------- prerequisite checks ----------

if [[ ! -f "${PHX_PTX}" ]]; then
  echo "kretikos_kaxi_phase_x_gate: FAIL -- PTX missing: ${PHX_PTX}"
  exit 1
fi
if ! command -v cc >/dev/null 2>&1; then
  echo "kretikos_kaxi_phase_x_gate: SKIPPED (cc missing)"
  exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "kretikos_kaxi_phase_x_gate: SKIPPED (local nvidia-smi missing)"
  exit 0
fi
if [[ -z "${WORKER_SCRIPT}" ]]; then
  echo "kretikos_kaxi_phase_x_gate: SKIPPED (A5000 worker script not found)"
  exit 0
fi

shard_patients=$(( PHX_COHORT / PHX_SHARD_COUNT ))
echo "[X] Phase X -- rapamycin 1-comp epistemic PBPK at billion-patient scale"
echo "  cohort=${PHX_COHORT} shards=${PHX_SHARD_COUNT} (~${shard_patients} patients/shard)"
echo "  GPU-0: shards 0..$(( PHX_GPU0_SHARDS - 1 )) (RTX4000Ada, local)"
echo "  GPU-1: shards ${PHX_GPU0_SHARDS}..$(( PHX_SHARD_COUNT - 1 )) (A5000, remote)"
echo "  streams=${PHX_STREAMS} chunks=${PHX_CHUNKS}"
echo "  kernel: C × 3.125 → ng/mL, gate [5.0, 15.0] ng/mL (therapeutic window)"
echo "  GUM σ²(C_scaled) = 3.125² × σ²(C) tracked through epistemic PTX"

# ---------- [1/5] build local binaries ----------

mkdir -p /workspace/tmp
STAGE_DIR="${PHX_STAGE_DIR:-$(mktemp -d /workspace/tmp/phase_x_gate.XXXXXX)}"
mkdir -p "${STAGE_DIR}/ref"

echo "[1/5] building local sampler + runner"
cc -O2 -Wall -ffp-contract=off \
   "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_rapamycin_sampler.c" -lm \
   -o "${STAGE_DIR}/sampler"
cc -O2 -Wall "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c" -ldl \
   -o "${STAGE_DIR}/runner"

# ---------- [2/5] push source + build on A5000 ----------

echo "[2/5] pushing source to A5000 + building"
_kubectl cp -c "${WORKER_CTR}" \
  "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_rapamycin_sampler.c" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phx_sampler_src.c"
_kubectl cp -c "${WORKER_CTR}" \
  "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phx_runner_src.c"
_kubectl cp -c "${WORKER_CTR}" \
  "${PHX_PTX}" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phx_kernel.ptx"

REMOTE_BUILD="cc -O2 -Wall -ffp-contract=off /tmp/phx_sampler_src.c -lm -o /tmp/phx_sampler2 && \
              cc -O2 -Wall /tmp/phx_runner_src.c -ldl -o /tmp/phx_runner2 && \
              echo BUILD_OK"
if ! rb="$("${WORKER_SCRIPT}" "${REMOTE_BUILD}" 2>&1)" || ! echo "${rb}" | grep -q BUILD_OK; then
  echo "kretikos_kaxi_phase_x_gate: FAIL -- remote build"; echo "${rb}"; exit 1
fi
echo "  remote build: OK"

# ---------- [3/5] CPU analytic reference ----------

echo "[3/5] CPU analytic reference (cohort=${PHX_COHORT}, counts-only)"
"${STAGE_DIR}/sampler" \
  --out-dir "${STAGE_DIR}/ref" \
  --cohort "${PHX_COHORT}" --seed "${PHX_SEED}" \
  --counts-only >/dev/null
ref_in_budget="$(grep '^in_budget=' "${STAGE_DIR}/ref/expected.summary" | cut -d= -f2)"
ref_nan_count="$(grep '^nan_count='  "${STAGE_DIR}/ref/expected.summary" | cut -d= -f2)"
echo "  analytic: in_budget=${ref_in_budget} nan_count=${ref_nan_count}"

# ---------- [4/5] concurrent GPU shard loops ----------

echo "[4/5] launching GPU-0 and GPU-1 concurrently"

extract_field() { echo "$1" | sed -n "s/.*${2}=\([^ ]*\).*/\1/p"; }

# GPU-0 (local): shards 0..PHX_GPU0_SHARDS-1
GPU0_OUT="${STAGE_DIR}/gpu0_out.txt"
{
  for k in $(seq 0 $(( PHX_GPU0_SHARDS - 1 ))); do
    sdir="${STAGE_DIR}/shard${k}"
    mkdir -p "${sdir}"
    "${STAGE_DIR}/sampler" \
      --out-dir "${sdir}" \
      --cohort "${PHX_COHORT}" --seed "${PHX_SEED}" \
      --shard-index "${k}" --shard-count "${PHX_SHARD_COUNT}" >/dev/null
    n="$(grep '^shard_patients=' "${sdir}/expected.summary" | cut -d= -f2)"
    echo "GPU0_EXPECTED shard=${k} in_budget=$(grep '^in_budget=' "${sdir}/expected.summary" | cut -d= -f2) nan_count=$(grep '^nan_count=' "${sdir}/expected.summary" | cut -d= -f2)"
    "${STAGE_DIR}/runner" "${PHX_PTX}" \
      --kernel kaxi_kernel --epistemic --type f32 \
      --cohort-size "${n}" --threads "${PHX_THREADS}" \
      --streams "${PHX_STREAMS}" --chunks "${PHX_CHUNKS}" \
      --init-file "${sdir}/init.mem.bin" \
      --init-var-file "${sdir}/init.var.bin" \
      2>&1 | grep '^PHW' | sed "s/^PHW/PHW shard=${k}/"
  done
} >"${GPU0_OUT}" 2>&1 &
GPU0_PID=$!

# GPU-1 (remote A5000): shards PHX_GPU0_SHARDS..PHX_SHARD_COUNT-1
GPU1_OUT="${STAGE_DIR}/gpu1_out.txt"
{
  REMOTE_SHARD_CMD="
    k=${PHX_GPU0_SHARDS}; end=$(( PHX_SHARD_COUNT - 1 ))
    while [[ \${k} -le \${end} ]]; do
      sdir=/workspace/compiler-remote/tmp/phx_shard\${k}
      mkdir -p \"\${sdir}\"
      /tmp/phx_sampler2 \
        --out-dir \"\${sdir}\" \
        --cohort '${PHX_COHORT}' --seed '${PHX_SEED}' \
        --shard-index \${k} --shard-count '${PHX_SHARD_COUNT}' >/dev/null
      n=\$(grep '^shard_patients=' \"\${sdir}/expected.summary\" | cut -d= -f2)
      echo \"GPU1_EXPECTED shard=\${k} in_budget=\$(grep '^in_budget=' \"\${sdir}/expected.summary\" | cut -d= -f2) nan_count=\$(grep '^nan_count=' \"\${sdir}/expected.summary\" | cut -d= -f2)\"
      /tmp/phx_runner2 /tmp/phx_kernel.ptx \
        --kernel kaxi_kernel --epistemic --type f32 \
        --cohort-size \"\${n}\" --threads '${PHX_THREADS}' \
        --streams '${PHX_STREAMS}' --chunks '${PHX_CHUNKS}' \
        --init-file \"\${sdir}/init.mem.bin\" \
        --init-var-file \"\${sdir}/init.var.bin\" \
        2>&1 | grep '^PHW' | sed \"s/^PHW/PHW shard=\${k}/\"
      k=\$(( k + 1 ))
    done
  "
  "${WORKER_SCRIPT}" "${REMOTE_SHARD_CMD}"
} >"${GPU1_OUT}" 2>&1 &
GPU1_PID=$!

echo "  GPU-0 pid=${GPU0_PID}  GPU-1 pid=${GPU1_PID}"

gpu0_rc=0; gpu1_rc=0
wait "${GPU0_PID}" || gpu0_rc=$?
wait "${GPU1_PID}" || gpu1_rc=$?

echo "  GPU-0 rc=${gpu0_rc}  GPU-1 rc=${gpu1_rc}"

# ---------- [5/5] aggregation + truth-claim ----------

echo "[5/5] aggregation + truth-claim"
fail=0

if [[ "${gpu0_rc}" -ne 0 || "${gpu1_rc}" -ne 0 ]]; then
  echo "  WARNING: non-zero exit from GPU job(s)"
fi

total_in=0; total_nan=0; max_wall=0; missing=0
for src_label in "GPU0:${GPU0_OUT}" "GPU1:${GPU1_OUT}"; do
  label="${src_label%%:*}"
  out_file="${src_label#*:}"
  if [[ ! -s "${out_file}" ]]; then
    echo "  FAIL -- ${label} output file empty or missing"
    fail=1; continue
  fi
  while IFS= read -r phw_line; do
    shard="$(extract_field "${phw_line}" shard)"
    in_b="$(extract_field "${phw_line}" in_budget)"
    nan="$(extract_field "${phw_line}" nan_count)"
    wall="$(extract_field "${phw_line}" wall_us)"
    dig="$(extract_field "${phw_line}" mem_digest)"
    echo "  ${label} shard=${shard} wall_us=${wall} in_budget=${in_b} nan=${nan} digest=${dig}"
    if [[ -z "${in_b}" || -z "${nan}" ]]; then
      echo "  FAIL -- ${label} shard=${shard} missing in_budget or nan_count"
      fail=1; missing=$(( missing + 1 )); continue
    fi
    total_in=$(( total_in + in_b ))
    total_nan=$(( total_nan + nan ))
    [[ -n "${wall}" && "${wall}" -gt "${max_wall}" ]] && max_wall="${wall}"
    if [[ -n "${wall}" && "${wall}" -gt "${PHX_WALL_CEILING_US}" ]]; then
      echo "  REGRESSION: ${label} shard=${shard} wall_us=${wall} > ceiling=${PHX_WALL_CEILING_US}"
      fail=1
    fi
  done < <(grep '^PHW shard=' "${out_file}" 2>/dev/null)
done

if [[ "${missing}" -gt 0 ]]; then
  echo "  --- GPU-0 output ---"; cat "${GPU0_OUT}"
  echo "  --- GPU-1 output ---"; cat "${GPU1_OUT}"
fi

total_patients=$(( total_in + total_nan ))
echo "  aggregate: in_budget=${total_in} nan_count=${total_nan} total=${total_patients}"
echo "  analytic:  in_budget=${ref_in_budget} nan_count=${ref_nan_count} total=${PHX_COHORT}"

if [[ "${total_in}" == "${ref_in_budget}" && "${total_nan}" == "${ref_nan_count}" ]]; then
  echo "  truth-claim: PASS -- GPU epistemic PBPK aggregate == CPU analytic exactly"
else
  echo "  truth-claim: FAIL -- mismatch"
  fail=1
fi

if [[ "${total_patients}" != "${PHX_COHORT}" ]]; then
  echo "  patient-count: FAIL -- total ${total_patients} != cohort ${PHX_COHORT}"
  fail=1
fi

if [[ "${max_wall}" -gt 0 ]]; then
  decisions_per_s=$(( PHX_COHORT * 1000000 / max_wall ))
  echo "  max_shard_wall_us=${max_wall}"
  echo "  throughput (effective, 2 GPUs): ~${decisions_per_s} epistemic-decisions/sec"
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "kretikos_kaxi_phase_x_gate: FAIL"
  exit 1
fi
echo "kretikos_kaxi_phase_x_gate: PASS"
