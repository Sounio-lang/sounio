#!/usr/bin/env bash
set -euo pipefail

# Submit the K-AXI→PTX driver-JIT acceptance gate to Slurm (gpu-orangefs).
# gpu-orangefs nodes are driver-only (no ptxas), so PTX is validated through
# the CUDA driver's internal JIT via the dlopen-based scripts/gpu/kaxi_ptx_runner.c.
#
# Prereq (staged under ${STAGE_LOCAL}):
#   ${STAGE_LOCAL}/ptx/*.ptx           (emitted locally by bin/kretikos)
#   ${STAGE_LOCAL}/kaxi_ptx_runner     (gcc -O2 ... kaxi_ptx_runner.c -ldl -lm)
#
# Usage:
#   cd /workspace/sounio
#   bash slurm-jobs/kaxi-ptxas-accept/submit_jit.sh
#
# Fetch:
#   kubectl -n slurm-pilot exec <login-pod> -- cat \
#     /orangefs/training/sounio/kaxi-jit-accept/<RUN_ID>/results/summary.txt

# --wait : after submitting, poll the queue and fetch/assert the result
#          (exit 0 only on KAXI_JIT_ACCEPT_OK). Also honoured via WAIT=1.
WAIT="${WAIT:-0}"
for a in "$@"; do [[ "$a" == "--wait" ]] && WAIT=1; done
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"   # max seconds to wait for completion
WAIT_POLL="${WAIT_POLL:-20}"           # seconds between polls

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
RUN_ID="${RUN_ID:-kaxi-jit-accept-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/kaxi-jit-accept/${RUN_ID}"
STAGE_LOCAL="${STAGE_LOCAL:-/tmp/kaxi_ptx_stage}"
# Worker-side runner script (override to run a diagnostic instead of the gate).
RUNNER_SH="${RUNNER_SH:-${SOUNIO_DIR}/slurm-jobs/kaxi-ptxas-accept/run_jit.sh}"
RUNNER_BASE="$(basename "${RUNNER_SH}")"
LOCAL_TARBALL="/tmp/${RUN_ID}.tgz"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
JOB_MEM="${JOB_MEM:-4G}"
JOB_TIME="${JOB_TIME:-00:25:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"

[[ -d "${STAGE_LOCAL}/ptx" ]] || { echo "missing ${STAGE_LOCAL}/ptx" >&2; exit 1; }
[[ -x "${STAGE_LOCAL}/kaxi_ptx_runner" ]] || { echo "missing runner binary ${STAGE_LOCAL}/kaxi_ptx_runner" >&2; exit 1; }
[[ -f "${RUNNER_SH}" ]] || { echo "missing ${RUNNER_SH}" >&2; exit 1; }
NPTX="$(ls "${STAGE_LOCAL}/ptx"/*.ptx 2>/dev/null | wc -l)"
echo "staging ${NPTX} PTX kernels + runner under run ${RUN_ID}"

LOGIN_POD=""
if [[ -n "${LOGIN_POD_NAME}" ]]; then LOGIN_POD="${LOGIN_POD_NAME}"
elif "${KUBECTL_BIN}" -n "${NS}" get deploy "${LOGIN_DEPLOY_NAME}" >/dev/null 2>&1; then
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods \
    -l "app.kubernetes.io/instance=${LOGIN_DEPLOY_NAME},app.kubernetes.io/name=login" \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
fi
[[ -n "${LOGIN_POD}" ]] || LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods -l "${LOGIN_SELECTOR}" \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[[ -n "${LOGIN_POD}" ]] || { echo "no live login pod in ${NS}" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}"

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc '
  for a in $(seq 1 20); do test -S /run/slurm/sack.socket && { scontrol ping >/dev/null; exit 0; }; sleep 1; done
  echo "sack.socket not ready" >&2; exit 1'

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "mkdir -p '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs'"

cp -f "${RUNNER_SH}" "${STAGE_LOCAL}/${RUNNER_BASE}"; chmod +x "${STAGE_LOCAL}/${RUNNER_BASE}"
rm -f "${LOCAL_TARBALL}"
tar -C "${STAGE_LOCAL}" -czf "${LOCAL_TARBALL}" ptx "${RUNNER_BASE}" kaxi_ptx_runner
tar -tzf "${LOCAL_TARBALL}" >/dev/null

if ! "${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_TARBALL}" "${LOGIN_POD}:${STAGE_ROOT}/payload.tgz" >/dev/null 2>&1; then
  cat "${LOCAL_TARBALL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${STAGE_ROOT}/payload.tgz'"
fi
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "tar -tzf '${STAGE_ROOT}/payload.tgz' >/dev/null"

SBATCH_LOCAL="/tmp/${RUN_ID}.sbatch.local"; rm -f "${SBATCH_LOCAL}"
cat > "${SBATCH_LOCAL}" <<EOF
#!/usr/bin/env bash
#SBATCH -J kaxi-jit-accept
#SBATCH -p ${SBATCH_PARTITION}
#SBATCH -A ${SBATCH_ACCOUNT}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
EOF
[[ -n "${SBATCH_NODELIST}" ]] && echo "#SBATCH -w ${SBATCH_NODELIST}" >> "${SBATCH_LOCAL}"

cat >> "${SBATCH_LOCAL}" <<'SBATCH_BODY'
set -uo pipefail
RUN_ROOT='RUN_ROOT_PLACEHOLDER'
LOCAL_ROOT="${TMPDIR:-/tmp}/kaxi-jit-${SLURM_JOB_ID}"
RESULTS_DIR="${RUN_ROOT}/results"; LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${LOCAL_ROOT}"
exec > >(tee "${LOG_DIR}/job-${SLURM_JOB_ID}.log") 2>&1
echo "=== kaxi driver-JIT acceptance — $(date) host=$(hostname) cuda_vis=${CUDA_VISIBLE_DEVICES:-unset} ==="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
tar -xzf "${RUN_ROOT}/payload.tgz" -C "${LOCAL_ROOT}"
chmod +x "${LOCAL_ROOT}/RUNNER_BASE_PLACEHOLDER" "${LOCAL_ROOT}/kaxi_ptx_runner"
cd "${LOCAL_ROOT}"
bash "${LOCAL_ROOT}/RUNNER_BASE_PLACEHOLDER" "${LOCAL_ROOT}/ptx" "${RESULTS_DIR}" "${LOCAL_ROOT}/kaxi_ptx_runner"
rc=$?; echo "runner rc=$rc"; exit $rc
SBATCH_BODY
sed -i "s|RUN_ROOT_PLACEHOLDER|${STAGE_ROOT}|g; s|RUNNER_BASE_PLACEHOLDER|${RUNNER_BASE}|g" "${SBATCH_LOCAL}"

"${KUBECTL_BIN}" -n "${NS}" cp "${SBATCH_LOCAL}" "${LOGIN_POD}:${SBATCH_FILE}" >/dev/null 2>&1 || \
  cat "${SBATCH_LOCAL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${SBATCH_FILE}'"

SBATCH_OUTPUT="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  sbatch '${SBATCH_FILE}'; rm -f '${SBATCH_FILE}'; echo ---
  squeue -o '%.8i %.18j %.10P %.8T %.10M %R' 2>/dev/null || squeue")"
echo "${SBATCH_OUTPUT}"
JOB_ID="$(printf '%s\n' "${SBATCH_OUTPUT}" | awk '/Submitted batch job/ {print $4; exit}')"
[[ -n "${JOB_ID}" ]] || { echo "failed to parse job id" >&2; exit 1; }
echo; echo "RUN_ID=${RUN_ID}"; echo "JOB_ID=${JOB_ID}"; echo "STAGE_ROOT=${STAGE_ROOT}"; echo "LOGIN_POD=${LOGIN_POD}"
echo; echo "# fetch:"; echo "kubectl -n ${NS} exec ${LOGIN_POD} -- cat ${STAGE_ROOT}/results/summary.txt"

if [[ "${WAIT}" != "1" ]]; then
  exit 0
fi

echo; echo "--- waiting for job ${JOB_ID} (timeout ${WAIT_TIMEOUT}s, poll ${WAIT_POLL}s) ---"
elapsed=0
while :; do
  ST="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
        "squeue -j '${JOB_ID}' -h -o '%T' 2>/dev/null" | tr -d '[:space:]')"
  [[ -z "${ST}" ]] && { echo "job left the queue after ${elapsed}s"; break; }
  echo "  t=${elapsed}s state=${ST}"
  if [[ "${elapsed}" -ge "${WAIT_TIMEOUT}" ]]; then
    echo "TIMEOUT waiting for job ${JOB_ID} after ${WAIT_TIMEOUT}s" >&2
    "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "scancel '${JOB_ID}'" >/dev/null 2>&1 || true
    exit 124
  fi
  sleep "${WAIT_POLL}"; elapsed=$((elapsed + WAIT_POLL))
done

echo; echo "=== summary ==="
SUMMARY="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "cat '${STAGE_ROOT}/results/summary.txt' 2>/dev/null || echo '(no summary produced)'")"
echo "${SUMMARY}"
echo "=== job log tail ==="
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "tail -n 20 '${STAGE_ROOT}/logs/'job-*.log 2>/dev/null || true"

if printf '%s\n' "${SUMMARY}" | grep -q "KAXI_JIT_ACCEPT_OK"; then
  echo; echo "ACCEPTANCE GATE: PASS"; exit 0
else
  echo; echo "ACCEPTANCE GATE: FAIL" >&2; exit 1
fi
