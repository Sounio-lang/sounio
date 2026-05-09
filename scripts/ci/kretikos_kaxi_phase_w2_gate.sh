#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_phase_w2_gate.sh
#
# Phase W.2 gate: sharded multi-GPU billion-patient sweep across two CUDA
# devices on separate nodes:
#   GPU-0  RTX 4000 Ada (19 GB) — workspace pod, direct access
#   GPU-1  RTX A5000  (24 GB) — r740 worker, via run-compiler-2gpu-worker-command.sh
#
# Each GPU owns half the cohort (shard 0 / shard 1). The kaxi_pbpk_sampler
# uses O(1) splitmix64 skip-ahead so shard K generates the exact same patients
# as [K*floor(N/2), (K+1)*floor(N/2)) of the full-cohort run — no file
# transfer required; each node samples its own shard independently.
#
# Truth-claim: aggregate GPU (in_budget_0 + in_budget_1) must equal the
# CPU analytic reference from a single full-cohort sampler pass.
#
# Both shards run concurrently (background jobs); wall time = max(T0, T1).
#
# Knobs (env):
#   PHW2_COHORT            total patients (default 200000000)
#   PHW2_THREADS           threads/block (default 32)
#   PHW2_SEED              sampler seed (default 42)
#   PHW2_DIALECT           PTX dialect (default f32e)
#   PHW2_TYPE              sampler type (default f32)
#   PHW2_STREAMS           CUDA streams per shard (default 4)
#   PHW2_CHUNKS            chunks per shard (default 64)
#   PHW2_WALL_CEILING_US   per-shard wall ceiling (default 3600000000, ~1h)
#   PHW2_REMOTE_SRC        source root on A5000 worker (default /workspace/compiler-remote/sounio)
#   PHW2_WORKER_SCRIPT     path to kubectl exec wrapper (auto-detected)
#
# Skip with: SOUNIO_KAXI_PHASE_W2_GATE_SKIP=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_KAXI_PHASE_W2_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_phase_w2_gate: SKIPPED (SOUNIO_KAXI_PHASE_W2_GATE_SKIP=1)"
  exit 0
fi

PHW2_COHORT="${PHW2_COHORT:-200000000}"
PHW2_THREADS="${PHW2_THREADS:-32}"
PHW2_SEED="${PHW2_SEED:-42}"
PHW2_DIALECT="${PHW2_DIALECT:-f32e}"
PHW2_TYPE="${PHW2_TYPE:-f32}"
PHW2_STREAMS="${PHW2_STREAMS:-4}"
PHW2_CHUNKS="${PHW2_CHUNKS:-64}"
PHW2_WALL_CEILING_US="${PHW2_WALL_CEILING_US:-3600000000}"
PHW2_REMOTE_SRC="${PHW2_REMOTE_SRC:-/workspace/compiler-remote/sounio}"
PHW2_PTX="${PHW2_PTX:-${ROOT_DIR}/tests/golden/kaxi_ptx/${PHW2_DIALECT}/vec_sqrt_gate_var_mb.ptx}"
PHW2_REMOTE_PTX="${PHW2_REMOTE_PTX:-${PHW2_REMOTE_SRC}/tests/golden/kaxi_ptx/${PHW2_DIALECT}/vec_sqrt_gate_var_mb.ptx}"

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

# ---------- prerequisite checks ----------

if [[ ! -f "${PHW2_PTX}" ]]; then
  echo "kretikos_kaxi_phase_w2_gate: FAIL — PTX missing locally: ${PHW2_PTX}"
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

echo "[W.2] Phase W.2 — sharded multi-GPU billion-patient sweep"
echo "  cohort=${PHW2_COHORT}  shards=2 (RTX4000Ada + A5000)"
echo "  streams=${PHW2_STREAMS} chunks=${PHW2_CHUNKS} type=${PHW2_TYPE} dialect=${PHW2_DIALECT}"

# Resolve a working kubectl for file copy operations.
_kubectl() {
  if kubectl "$@" 2>/dev/null; then return 0
  elif sudo kubectl "$@"; then return 0
  else return 1; fi
}
WORKER_NS="${PHW2_WORKER_NS:-beagle}"
WORKER_POD="${PHW2_WORKER_POD:-sounio-compiler-gpu-worker-r740}"
WORKER_CTR="${PHW2_WORKER_CTR:-worker}"

# ---------- [1/5] build local binaries ----------

STAGE_DIR="${PHW2_STAGE_DIR:-$(mktemp -d /tmp/phase_w2_gate.XXXXXX)}"
mkdir -p "${STAGE_DIR}/shard0"

echo "[1/5] building local sampler + runner"
cc -O2 -Wall "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_sampler.c" -lm -o "${STAGE_DIR}/sampler"
cc -O2 -Wall "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c"   -ldl -o "${STAGE_DIR}/runner"

# ---------- [2/5] push fresh source + build on A5000 ----------

echo "[2/5] pushing source to A5000 worker + building"
# Always push local source so the gate is independent of remote git sync state.
_kubectl cp -c "${WORKER_CTR}" \
  "${ROOT_DIR}/scripts/gpu/kaxi_pbpk_sampler.c" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phw2_sampler_src.c"
_kubectl cp -c "${WORKER_CTR}" \
  "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phw2_runner_src.c"
_kubectl cp -c "${WORKER_CTR}" \
  "${PHW2_PTX}" \
  "${WORKER_NS}/${WORKER_POD}:/tmp/phw2_kernel.ptx"

REMOTE_BUILD_CMD="\
  cc -O2 -Wall /tmp/phw2_sampler_src.c -lm -o /tmp/phw2_remote_sampler && \
  cc -O2 -Wall /tmp/phw2_runner_src.c   -ldl -o /tmp/phw2_remote_runner && \
  echo BUILD_OK"
if ! remote_build_out="$("${WORKER_SCRIPT}" "${REMOTE_BUILD_CMD}" 2>&1)"; then
  echo "kretikos_kaxi_phase_w2_gate: FAIL — remote build failed"
  echo "${remote_build_out}"
  exit 1
fi
if ! echo "${remote_build_out}" | grep -q 'BUILD_OK'; then
  echo "kretikos_kaxi_phase_w2_gate: FAIL — remote build did not emit BUILD_OK"
  echo "${remote_build_out}"
  exit 1
fi
echo "  remote build: OK"

# ---------- [3/5] CPU analytic reference (full cohort) ----------

echo "[3/5] CPU analytic reference (full cohort=${PHW2_COHORT})"
"${STAGE_DIR}/sampler" \
  --out-dir "${STAGE_DIR}" \
  --cohort "${PHW2_COHORT}" --seed "${PHW2_SEED}" --type "${PHW2_TYPE}" >/dev/null
ref_in_budget="$(grep '^in_budget=' "${STAGE_DIR}/expected.summary" | cut -d= -f2)"
ref_nan_count="$(grep '^nan_count=' "${STAGE_DIR}/expected.summary" | cut -d= -f2)"
echo "  analytic: in_budget=${ref_in_budget} nan_count=${ref_nan_count}"

# ---------- [4/5] concurrent GPU shards ----------

echo "[4/5] launching shards concurrently"

# Helper: extract fields from PHW output line
extract_field() { echo "$1" | grep '^PHW' | tail -1 | sed -n "s/.*$2=\\([^ ]*\\).*/\\1/p"; }

# Shard 0: RTX 4000 Ada (local), patients [0, cohort/2)
SHARD0_OUT="${STAGE_DIR}/shard0_out.txt"
{
  "${STAGE_DIR}/sampler" \
    --out-dir "${STAGE_DIR}/shard0" \
    --cohort "${PHW2_COHORT}" --seed "${PHW2_SEED}" --type "${PHW2_TYPE}" \
    --shard-index 0 --shard-count 2 >/dev/null
  shard0_patients="$(grep '^shard_patients=' "${STAGE_DIR}/shard0/expected.summary" | cut -d= -f2)"
  "${STAGE_DIR}/runner" "${PHW2_PTX}" \
    --kernel kaxi_kernel --epistemic --type "${PHW2_TYPE}" \
    --cohort-size "${shard0_patients}" --threads "${PHW2_THREADS}" \
    --streams "${PHW2_STREAMS}" --chunks "${PHW2_CHUNKS}" \
    --init-file "${STAGE_DIR}/shard0/init.mem.bin" \
    --init-var-file "${STAGE_DIR}/shard0/init.var.bin"
} > "${SHARD0_OUT}" 2>&1 &
SHARD0_PID=$!

# Shard 1: RTX A5000 (remote r740), patients [cohort/2, cohort)
SHARD1_OUT="${STAGE_DIR}/shard1_out.txt"
{
  REMOTE_SHARD1_CMD="
    mkdir -p /tmp/phw2_shard1 && \
    /tmp/phw2_remote_sampler \
      --out-dir /tmp/phw2_shard1 \
      --cohort '${PHW2_COHORT}' --seed '${PHW2_SEED}' --type '${PHW2_TYPE}' \
      --shard-index 1 --shard-count 2 >/dev/null && \
    shard1_patients=\$(grep '^shard_patients=' /tmp/phw2_shard1/expected.summary | cut -d= -f2) && \
    /tmp/phw2_remote_runner /tmp/phw2_kernel.ptx \
      --kernel kaxi_kernel --epistemic --type '${PHW2_TYPE}' \
      --cohort-size \"\${shard1_patients}\" --threads '${PHW2_THREADS}' \
      --streams '${PHW2_STREAMS}' --chunks '${PHW2_CHUNKS}' \
      --init-file /tmp/phw2_shard1/init.mem.bin \
      --init-var-file /tmp/phw2_shard1/init.var.bin
  "
  "${WORKER_SCRIPT}" "${REMOTE_SHARD1_CMD}"
} > "${SHARD1_OUT}" 2>&1 &
SHARD1_PID=$!

echo "  shard0 pid=${SHARD0_PID} (local RTX4000Ada)"
echo "  shard1 pid=${SHARD1_PID} (remote A5000)"

# Wait for both — capture exit codes without set -e killing us
shard0_rc=0; shard1_rc=0
wait "${SHARD0_PID}" || shard0_rc=$?
wait "${SHARD1_PID}" || shard1_rc=$?

# Parse shard outputs
shard0_phw="$(grep '^PHW' "${SHARD0_OUT}" | tail -1)"
shard1_phw="$(grep '^PHW' "${SHARD1_OUT}" | tail -1)"

shard0_wall="$(extract_field "${shard0_phw}" wall_us)"
shard1_wall="$(extract_field "${shard1_phw}" wall_us)"
shard0_nan="$(extract_field "${shard0_phw}" nan_count)"
shard1_nan="$(extract_field "${shard1_phw}" nan_count)"
shard0_in="$(extract_field "${shard0_phw}" in_budget)"
shard1_in="$(extract_field "${shard1_phw}" in_budget)"
shard0_dig="$(extract_field "${shard0_phw}" mem_digest)"
shard1_dig="$(extract_field "${shard1_phw}" mem_digest)"

echo "  shard0 (RTX4000Ada): wall_us=${shard0_wall} in_budget=${shard0_in} nan=${shard0_nan} digest=${shard0_dig} rc=${shard0_rc}"
echo "  shard1 (A5000):      wall_us=${shard1_wall} in_budget=${shard1_in} nan=${shard1_nan} digest=${shard1_dig} rc=${shard1_rc}"

# ---------- [5/5] summary + assertions ----------

echo "[5/5] aggregation + truth-claim"
fail=0

# Both shards must have produced PHW output
if [[ -z "${shard0_phw}" ]]; then
  echo "  FAIL — shard 0 produced no PHW line"
  echo "--- shard0 output ---"; cat "${SHARD0_OUT}"; echo "---"
  fail=1
fi
if [[ -z "${shard1_phw}" ]]; then
  echo "  FAIL — shard 1 produced no PHW line"
  echo "--- shard1 output ---"; cat "${SHARD1_OUT}"; echo "---"
  fail=1
fi

if [[ "${fail}" -eq 0 ]]; then
  total_in=$(( shard0_in + shard1_in ))
  total_nan=$(( shard0_nan + shard1_nan ))
  total_patients=$(( total_in + total_nan ))

  echo "  aggregate: in_budget=${total_in} nan_count=${total_nan} total=${total_patients}"
  echo "  analytic:  in_budget=${ref_in_budget} nan_count=${ref_nan_count} total=${PHW2_COHORT}"

  # Truth-claim: aggregate GPU == CPU analytic
  if [[ "${total_in}" == "${ref_in_budget}" && "${total_nan}" == "${ref_nan_count}" ]]; then
    echo "  truth-claim: PASS — GPU aggregate == CPU analytic exactly"
  else
    echo "  truth-claim: FAIL — GPU aggregate (in=${total_in} nan=${total_nan}) != analytic (in=${ref_in_budget} nan=${ref_nan_count})"
    fail=1
  fi

  # Total patient count must equal full cohort
  if [[ "${total_patients}" != "${PHW2_COHORT}" ]]; then
    echo "  patient-count: FAIL — total ${total_patients} != cohort ${PHW2_COHORT}"
    fail=1
  fi

  # Wall-clock ceiling per shard (regression guard)
  for w in "${shard0_wall}" "${shard1_wall}"; do
    if [[ -n "${w}" && "${w}" -gt "${PHW2_WALL_CEILING_US}" ]]; then
      echo "  REGRESSION: shard wall_us=${w} > ceiling=${PHW2_WALL_CEILING_US}"
      fail=1
    fi
  done

  # Throughput report
  if [[ -n "${shard0_wall}" && -n "${shard1_wall}" ]]; then
    gate_wall_us=$(( shard0_wall > shard1_wall ? shard0_wall : shard1_wall ))
    # decisions/sec = cohort * 1e6 / gate_wall_us (integer approx)
    decisions_per_s=$(( PHW2_COHORT * 1000000 / gate_wall_us ))
    echo "  gate_wall_us=${gate_wall_us} (max of two shards)"
    echo "  throughput: ~${decisions_per_s} decisions/sec across 2 GPUs"
  fi
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "kretikos_kaxi_phase_w2_gate: FAIL"
  exit 1
fi
echo "kretikos_kaxi_phase_w2_gate: PASS"
