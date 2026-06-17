#!/usr/bin/env bash
set -euo pipefail

# Submit the K-AXI→PTX ptxas-acceptance gate to Slurm (gpu-orangefs).
# Assembles every locally-emitted PTX kernel with ptxas on a GPU node
# (no GPU compute needed — ptxas is the CUDA assembler). Decoupled design:
# PTX is emitted locally by bin/kretikos and staged; the worker only runs ptxas.
#
# Prereq: PTX already emitted to ${STAGE_LOCAL}/ptx/*.ptx (see emit step).
#
# Usage:
#   cd /workspace/sounio
#   bash slurm-jobs/kaxi-ptxas-accept/submit.sh
#
# Fetch results after completion:
#   kubectl -n slurm-pilot exec <login-pod> -- cat \
#     /orangefs/training/sounio/kaxi-ptxas-accept/<RUN_ID>/results/summary.txt

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
RUN_ID="${RUN_ID:-kaxi-ptxas-accept-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/kaxi-ptxas-accept/${RUN_ID}"
STAGE_LOCAL="${STAGE_LOCAL:-/tmp/kaxi_ptx_stage}"
RUNNER_SRC="${SOUNIO_DIR}/slurm-jobs/kaxi-ptxas-accept/run_ptxas.sh"
LOCAL_TARBALL="/tmp/${RUN_ID}.tgz"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
JOB_MEM="${JOB_MEM:-4G}"
JOB_TIME="${JOB_TIME:-00:20:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"

# --- preflight --------------------------------------------------------------
[[ -d "${STAGE_LOCAL}/ptx" ]] || { echo "missing ${STAGE_LOCAL}/ptx — emit PTX first" >&2; exit 1; }
NPTX="$(ls "${STAGE_LOCAL}/ptx"/*.ptx 2>/dev/null | wc -l)"
[[ "${NPTX}" -gt 0 ]] || { echo "no .ptx files staged in ${STAGE_LOCAL}/ptx" >&2; exit 1; }
[[ -f "${RUNNER_SRC}" ]] || { echo "missing runner ${RUNNER_SRC}" >&2; exit 1; }
command -v "${KUBECTL_BIN}" >/dev/null 2>&1 || { echo "kubectl not found" >&2; exit 1; }
echo "staging ${NPTX} PTX kernels under run ${RUN_ID}"

# --- resolve login pod ------------------------------------------------------
LOGIN_POD=""
if [[ -n "${LOGIN_POD_NAME}" ]]; then
  LOGIN_POD="${LOGIN_POD_NAME}"
elif "${KUBECTL_BIN}" -n "${NS}" get deploy "${LOGIN_DEPLOY_NAME}" >/dev/null 2>&1; then
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods \
    -l "app.kubernetes.io/instance=${LOGIN_DEPLOY_NAME},app.kubernetes.io/name=login" \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
fi
if [[ -z "${LOGIN_POD}" ]]; then
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi
[[ -n "${LOGIN_POD}" ]] || { echo "could not resolve a live login pod in ${NS}" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}"

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc '
  set -euo pipefail
  for attempt in $(seq 1 20); do
    test -S /run/slurm/sack.socket && { scontrol ping >/dev/null; exit 0; }
    sleep 1
  done
  echo "slurm sack.socket never became ready" >&2; exit 1
'

# --- stage payload ----------------------------------------------------------
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "mkdir -p '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs'"

cp -f "${RUNNER_SRC}" "${STAGE_LOCAL}/run_ptxas.sh"
chmod +x "${STAGE_LOCAL}/run_ptxas.sh"
rm -f "${LOCAL_TARBALL}"
tar -C "${STAGE_LOCAL}" -czf "${LOCAL_TARBALL}" ptx run_ptxas.sh
tar -tzf "${LOCAL_TARBALL}" >/dev/null

if ! "${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_TARBALL}" "${LOGIN_POD}:${STAGE_ROOT}/payload.tgz" >/dev/null 2>&1; then
  cat "${LOCAL_TARBALL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${STAGE_ROOT}/payload.tgz'"
fi
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "tar -tzf '${STAGE_ROOT}/payload.tgz' >/dev/null"

# --- build sbatch -----------------------------------------------------------
SBATCH_LOCAL="/tmp/${RUN_ID}.sbatch.local"; rm -f "${SBATCH_LOCAL}"
cat > "${SBATCH_LOCAL}" <<EOF
#!/usr/bin/env bash
#SBATCH -J kaxi-ptxas-accept
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
LOCAL_ROOT="${TMPDIR:-/tmp}/kaxi-ptxas-${SLURM_JOB_ID}"
RESULTS_DIR="${RUN_ROOT}/results"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${LOCAL_ROOT}"
exec > >(tee "${LOG_DIR}/job-${SLURM_JOB_ID}.log") 2>&1

echo "=== kaxi ptxas-acceptance — $(date) host=$(hostname) ==="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

tar -xzf "${RUN_ROOT}/payload.tgz" -C "${LOCAL_ROOT}"
chmod +x "${LOCAL_ROOT}/run_ptxas.sh"
cd "${LOCAL_ROOT}"
bash "${LOCAL_ROOT}/run_ptxas.sh" "${LOCAL_ROOT}/ptx" "${RESULTS_DIR}"
rc=$?
echo "runner rc=$rc"
exit $rc
SBATCH_BODY

sed -i "s|RUN_ROOT_PLACEHOLDER|${STAGE_ROOT}|g" "${SBATCH_LOCAL}"

"${KUBECTL_BIN}" -n "${NS}" cp "${SBATCH_LOCAL}" "${LOGIN_POD}:${SBATCH_FILE}" >/dev/null 2>&1 || \
  cat "${SBATCH_LOCAL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${SBATCH_FILE}'"

SBATCH_OUTPUT="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  sbatch '${SBATCH_FILE}'
  rm -f '${SBATCH_FILE}'
  echo ---
  squeue -o '%.8i %.20j %.10P %.8T %.10M %R' 2>/dev/null || squeue
")"
echo "${SBATCH_OUTPUT}"

JOB_ID="$(printf '%s\n' "${SBATCH_OUTPUT}" | awk '/Submitted batch job/ {print $4; exit}')"
[[ -n "${JOB_ID}" ]] || { echo "failed to parse job id" >&2; exit 1; }

echo
echo "RUN_ID=${RUN_ID}"
echo "JOB_ID=${JOB_ID}"
echo "STAGE_ROOT=${STAGE_ROOT}"
echo "LOGIN_POD=${LOGIN_POD}"
echo
echo "# fetch result when done:"
echo "kubectl -n ${NS} exec ${LOGIN_POD} -- cat ${STAGE_ROOT}/results/summary.txt"
