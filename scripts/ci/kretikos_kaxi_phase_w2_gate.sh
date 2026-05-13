#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_phase_w2_gate.sh
#
# Phase W.2 gate: sharded multi-GPU billion-patient sweep across two CUDA
# devices on separate nodes:
#   GPU-0  RTX 4000 Ada (4 GB CUDA ctx) -- workspace pod, direct access
#   GPU-1  RTX A5000  (4 GB CUDA ctx)   -- r740 worker, via kubectl exec wrapper
#
# Sharding: PHW2_SHARD_COUNT shards total. Each GPU runs half the shards
# sequentially; both GPUs run concurrently (background jobs). With the
# default 4 shards of 500M each = 2B patients total at ~2GB per device
# buffer (well inside the 4GB CUDA context limit on both cards).
#
# The kaxi_pbpk_sampler uses O(1) splitmix64 skip-ahead: shard K generates
# patients [K*floor(N/S), ...) byte-identical to slicing the full-cohort
# run -- each node samples its own shard, no file transfer needed.
#
# Truth-claim: GPU aggregate (in_budget + nan_count across all shards) must
# equal the CPU analytic reference from --counts-only (no file I/O).
#
# Knobs (env):
#   PHW2_COHORT            total patients (default 2000000000)
#   PHW2_SHARD_COUNT       number of shards (default 4; must be even)
#   PHW2_THREADS           threads/block (default 32)
#   PHW2_SEED              sampler seed (default 42)
#   PHW2_DIALECT           PTX dialect (default f32e)
#   PHW2_TYPE              sampler type (default f32)
#   PHW2_STREAMS           CUDA streams per shard run (default 4)
#   PHW2_CHUNKS            chunks per shard run (default 64)
#   PHW2_WALL_CEILING_US   wall ceiling per shard (default 3600000000, ~1h)
#   PHW2_WORKER_SCRIPT     path to kubectl exec wrapper (auto-detected)
#
# Skip with: SOUNIO_KAXI_PHASE_W2_GATE_SKIP=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_KAXI_PHASE_W2_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_phase_w2_gate: SKIPPED (SOUNIO_KAXI_PHASE_W2_GATE_SKIP=1)"
  exit 0
fi

PHW2_COHORT="${PHW2_COHORT:-2000000000}"
PHW2_SHARD_COUNT="${PHW2_SHARD_COUNT:-4}"
PHW2_THREADS="${PHW2_THREADS:-32}"
PHW2_SEED="${PHW2_SEED:-42}"
PHW2_DIALECT="${PHW2_DIALECT:-f32e}"
PHW2_TYPE="${PHW2_TYPE:-f32}"
PHW2_STREAMS="${PHW2_STREAMS:-4}"
PHW2_CHUNKS="${PHW2_CHUNKS:-64}"
PHW2_WALL_CEILING_US="${PHW2_WALL_CEILING_US:-3600000000}"
PHW2_PTX="${PHW2_PTX:-${ROOT_DIR}/tests/golden/kaxi_ptx/${PHW2_DIALECT}/vec_sqrt_gate_var_mb.ptx}"

# Shards 0..half-1 run on GPU-0 (local); half..S-1 run on GPU-1 (remote).
PHW2_GPU0_SHARDS="$(( PHW2_SHARD_COUNT / 2 ))"

# Locate the kubectl exec wrapper for the A5000 r740 worker.
WORKER_SCRIPT="${PHW2_WORKER_SCRIPT:-}"
for candidate in \
  "/workspace/beagle/k8s/sounio-runners/run-compiler-2gpu-worker-command.sh" \
  "${ROOT_DIR}/../../beagle/k8s/sounio-runners/run-compiler-2gpu-worker-command.sh"; do
  if [[ -x "${candidate}" ]]; then
    WORKER_SCRIPT="${candidate}"
    break
  fi
done

WORKER_NS="${PHW2_WORKER_NS:-beagle}"
WORKER_POD="${PHW2_WORKER_POD:-sounio-compiler-gpu-worker-r740}"
WORKER_CTR="${PHW2_WORKER_CTR:-worker}"

_kubectl() {
  if kubectl "$@" 2>/dev/null; then return 0
  elif sudo kubectl "$@"; then return 0
  else return 1; fi
}

# ---------- prerequisite checks ----------

if [[ ! -f "${PHW2_PTX}" ]]; then
  echo "kretikos_kaxi_phase_w2_gate: FAIL -- PTX missing: ${PHW2_PTX}"
  exit 1
fi
if ! command -v cc >/dev/null 2>&1; then
  echo "kretikos_kaxi_phase_w2_gate: SKIPPED (cc missing)"
  exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "kretikos_kaxi_phase_w2_gate: SKIPPED (local nvidia-smi missing)"
  exit 0
fi
if [[ -z "${WORKER_SCRIPT}" ]]; then
  echo "kretikos_kaxi_phase_w2_gate: SKIPPED (A5000 worker script not found)"
  exit 0
fi

shard_patients=$(( PHW2_COHORT / PHW2_SHARD_COUNT ))
echo "[W.2] Phase W.2 -- sharded multi-GPU billion-patient sweep"
echo "  cohort=${PHW2_COHORT} shards=${PHW2_SHARD_COUNT} (~${shard_patients} patients/shard)"
echo "  GPU-0: shards 0..$(( PHW2_GPU0_SHARDS - 1 )) (RTX4000Ada, local)"
echo "  GPU-1: shards ${PHW2_GPU0_SHARDS}..$(( PHW2_SHARD_COUNT - 1 )) (A5000, remote)"
echo "  streams=${PHW2_STREAMS} chunks=${PHW2_CHUNKS} type=${PHW2_TYPE}"

# ---------- [1/5] build local binaries ----------

mkdir -p /workspace/tmp
STAGE_DIR="${PHW2_STAGE_DIR:-$(mktemp -d /workspace/tmp/phase_w2_gate.XXXXXX)}"
mkdir -p "${STAGE_DIR}/ref"

echo "[1/5] building local sampler + runner"
cc -O2 -Wall "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_sampler.c" -lm -o "${STAGE_DIR}/sampler"
cc -O2 -Wall "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c"   -ldl -o "${STAGE_DIR}/runner"

# ---------- [2/5] push fresh source + build on A5000 ----------

echo "[2/5] pushing source to A5000 + building"
_kubectl cp -c "${WORKER_CTR}" \
  "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_sampler.c" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phw2_sampler_src.c"
_kubectl cp -c "${WORKER_CTR}" \
  "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phw2_runner_src.c"
_kubectl cp -c "${WORKER_CTR}" \
  "${PHW2_PTX}" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phw2_kernel.ptx"

REMOTE_BUILD="cc -O2 -Wall /tmp/phw2_sampler_src.c -lm -o /tmp/phw2_sampler2 && \
              cc -O2 -Wall /tmp/phw2_runner_src.c   -ldl -o /tmp/phw2_runner2 && \
              echo BUILD_OK"
if ! rb="$("${WORKER_SCRIPT}" "${REMOTE_BUILD}" 2>&1)" || ! echo "${rb}" | grep -q BUILD_OK; then
  echo "kretikos_kaxi_phase_w2_gate: FAIL -- remote build"; echo "${rb}"; exit 1
fi
echo "  remote build: OK"

# ---------- [3/5] CPU analytic reference (counts-only, no file I/O) ----------

echo "[3/5] CPU analytic reference (cohort=${PHW2_COHORT}, counts-only)"
"${STAGE_DIR}/sampler" \
  --out-dir "${STAGE_DIR}/ref" \
  --cohort "${PHW2_COHORT}" --seed "${PHW2_SEED}" --type "${PHW2_TYPE}" \
  --counts-only >/dev/null
ref_in_budget="$(grep '^in_budget=' "${STAGE_DIR}/ref/expected.summary" | cut -d= -f2)"
ref_nan_count="$(grep '^nan_count=' "${STAGE_DIR}/ref/expected.summary" | cut -d= -f2)"
echo "  analytic: in_budget=${ref_in_budget} nan_count=${ref_nan_count}"

# ---------- [4/5] concurrent GPU shard loops ----------

echo "[4/5] launching GPU-0 and GPU-1 concurrently"

extract_field() { echo "$1" | sed -n "s/.*$2=\\([^ ]*\\).*/\\1/p"; }

# GPU-0 (local): shards 0..PHW2_GPU0_SHARDS-1, run sequentially
GPU0_OUT="${STAGE_DIR}/gpu0_out.txt"
{
  for k in $(seq 0 $(( PHW2_GPU0_SHARDS - 1 ))); do
    sdir="${STAGE_DIR}/shard${k}"
    mkdir -p "${sdir}"
    "${STAGE_DIR}/sampler" \
      --out-dir "${sdir}" \
      --cohort "${PHW2_COHORT}" --seed "${PHW2_SEED}" --type "${PHW2_TYPE}" \
      --shard-index "${k}" --shard-count "${PHW2_SHARD_COUNT}" >/dev/null
    n="$(grep '^shard_patients=' "${sdir}/expected.summary" | cut -d= -f2)"
    echo "GPU0_EXPECTED shard=${k} in_budget=$(grep '^in_budget=' "${sdir}/expected.summary" | cut -d= -f2) nan_count=$(grep '^nan_count=' "${sdir}/expected.summary" | cut -d= -f2)"
    "${STAGE_DIR}/runner" "${PHW2_PTX}" \
      --kernel kaxi_kernel --epistemic --type "${PHW2_TYPE}" \
      --cohort-size "${n}" --threads "${PHW2_THREADS}" \
      --streams "${PHW2_STREAMS}" --chunks "${PHW2_CHUNKS}" \
      --init-file "${sdir}/init.mem.bin" \
      --init-var-file "${sdir}/init.var.bin" 2>&1 | grep '^PHW' | sed "s/^PHW/PHW shard=${k}/"
  done
} >"${GPU0_OUT}" 2>&1 &
GPU0_PID=$!

# GPU-1 (remote A5000): shards PHW2_GPU0_SHARDS..PHW2_SHARD_COUNT-1
GPU1_OUT="${STAGE_DIR}/gpu1_out.txt"
{
  REMOTE_SHARD_CMD="
    k=${PHW2_GPU0_SHARDS}; end=$(( PHW2_SHARD_COUNT - 1 ))
    while [[ \${k} -le \${end} ]]; do
      sdir=/workspace/compiler-remote/tmp/phw2_shard\${k}
      mkdir -p \"\${sdir}\"
      /tmp/phw2_sampler2 \
        --out-dir \"\${sdir}\" \
        --cohort '${PHW2_COHORT}' --seed '${PHW2_SEED}' --type '${PHW2_TYPE}' \
        --shard-index \${k} --shard-count '${PHW2_SHARD_COUNT}' >/dev/null
      n=\$(grep '^shard_patients=' \"\${sdir}/expected.summary\" | cut -d= -f2)
      echo \"GPU1_EXPECTED shard=\${k} in_budget=\$(grep '^in_budget=' \"\${sdir}/expected.summary\" | cut -d= -f2) nan_count=\$(grep '^nan_count=' \"\${sdir}/expected.summary\" | cut -d= -f2)\"
      /tmp/phw2_runner2 /tmp/phw2_kernel.ptx \
        --kernel kaxi_kernel --epistemic --type '${PHW2_TYPE}' \
        --cohort-size \"\${n}\" --threads '${PHW2_THREADS}' \
        --streams '${PHW2_STREAMS}' --chunks '${PHW2_CHUNKS}' \
        --init-file \"\${sdir}/init.mem.bin\" \
        --init-var-file \"\${sdir}/init.var.bin\" 2>&1 | grep '^PHW' | sed \"s/^PHW/PHW shard=\${k}/\"
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

# Parse all PHW lines from both GPUs
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
    if [[ -n "${wall}" && "${wall}" -gt "${PHW2_WALL_CEILING_US}" ]]; then
      echo "  REGRESSION: ${label} shard=${shard} wall_us=${wall} > ceiling=${PHW2_WALL_CEILING_US}"
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
echo "  analytic:  in_budget=${ref_in_budget} nan_count=${ref_nan_count} total=${PHW2_COHORT}"

if [[ "${total_in}" == "${ref_in_budget}" && "${total_nan}" == "${ref_nan_count}" ]]; then
  echo "  truth-claim: PASS -- GPU aggregate == CPU analytic exactly"
else
  echo "  truth-claim: FAIL -- mismatch"
  fail=1
fi

if [[ "${total_patients}" != "${PHW2_COHORT}" ]]; then
  echo "  patient-count: FAIL -- total ${total_patients} != cohort ${PHW2_COHORT}"
  fail=1
fi

if [[ "${max_wall}" -gt 0 ]]; then
  gate_wall_us="${max_wall}"
  decisions_per_s=$(( PHW2_COHORT * 1000000 / gate_wall_us ))
  echo "  max_shard_wall_us=${gate_wall_us}"
  echo "  throughput (effective, 2 GPUs): ~${decisions_per_s} decisions/sec"
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "kretikos_kaxi_phase_w2_gate: FAIL"
  exit 1
fi
echo "kretikos_kaxi_phase_w2_gate: PASS"
