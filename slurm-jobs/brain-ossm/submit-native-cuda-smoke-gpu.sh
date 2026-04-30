#!/usr/bin/env bash
set -euo pipefail

# Submit from the control plane:
#   cd /home/devsounio/beagle/k8s/hpc-sota
#   source ops/lab-ops.sh
#   lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/home/devsounio/sounio}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
RUN_ID="${RUN_ID:-native-cuda-smoke-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/native-cuda-smoke/${RUN_ID}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_TARBALL_TMP="${LOCAL_TARBALL}.tmp"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
FIXTURE_REL="${FIXTURE_REL:-tests/run-pass/gpu_vec_add.sio}"
JOB_MEM="${JOB_MEM:-4G}"
JOB_TIME="${JOB_TIME:-00:10:00}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"
FORCE_RESTAGE="${FORCE_RESTAGE:-0}"
PAYLOAD_COPY_MODE="${PAYLOAD_COPY_MODE:-auto}"

REQUIRED_FILES=(
  "bin/souc"
  "artifacts/self-hosted/souc-self-hosted-x86_64"
  "scripts/gpu/run_native_cuda_smoke.sh"
  "stdlib"
  "${FIXTURE_REL}"
)

for rel in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
    echo "missing required file at ${SOUNIO_DIR}/${rel}" >&2
    exit 1
  fi
done

if ! command -v "${KUBECTL_BIN}" >/dev/null 2>&1; then
  echo "required cluster client not found in PATH: ${KUBECTL_BIN}" >&2
  exit 1
fi

LOGIN_POD=""

if [[ -n "${LOGIN_POD_NAME}" ]]; then
  if ! "${KUBECTL_BIN}" -n "${NS}" get pod "${LOGIN_POD_NAME}" >/dev/null 2>&1; then
    echo "could not find pod/${LOGIN_POD_NAME} in namespace ${NS}" >&2
    exit 1
  fi
  LOGIN_POD="${LOGIN_POD_NAME}"
elif "${KUBECTL_BIN}" -n "${NS}" get deploy "${LOGIN_DEPLOY_NAME}" >/dev/null 2>&1; then
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods \
    -l "app.kubernetes.io/instance=${LOGIN_DEPLOY_NAME},app.kubernetes.io/name=login" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}')"
fi

if [[ -z "${LOGIN_POD}" ]]; then
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods \
    -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi

if [[ -z "${LOGIN_POD}" ]]; then
  echo "could not resolve a live login pod in namespace ${NS}" >&2
  echo "tried deploy/${LOGIN_DEPLOY_NAME} and selector ${LOGIN_SELECTOR}" >&2
  echo "set LOGIN_POD_NAME=<pod> or LOGIN_SELECTOR=<label-selector> if the control plane naming changed" >&2
  exit 1
fi

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc '
  set -euo pipefail
  for attempt in $(seq 1 20); do
    if test -S /run/slurm/sack.socket; then
      exec scontrol ping >/dev/null
    fi
    sleep 1
  done
  echo "slurm login pod is running but sack.socket never became ready" >&2
  exit 1
'

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  mkdir -p '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs'
"

REMOTE_STAGE_READY="$(
  "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -euo pipefail
    if [[ '${FORCE_RESTAGE}' != '0' ]]; then
      echo NO
    elif [[ -f '${STAGE_ROOT}/payload.tgz' ]]; then
      echo YES
    else
      echo NO
    fi
  " | tr -d '[:space:]'
)"

if [[ "${REMOTE_STAGE_READY}" != "YES" ]]; then
  rm -f "${LOCAL_TARBALL}" "${LOCAL_TARBALL_TMP}"
  tar -C "${SOUNIO_DIR}" -czf "${LOCAL_TARBALL_TMP}" \
    bin/souc \
    artifacts/self-hosted/souc-self-hosted-x86_64 \
    scripts/gpu/run_native_cuda_smoke.sh \
    stdlib \
    "${FIXTURE_REL}"
  tar -tzf "${LOCAL_TARBALL_TMP}" >/dev/null
  mv -f "${LOCAL_TARBALL_TMP}" "${LOCAL_TARBALL}"

  if [[ "${PAYLOAD_COPY_MODE}" == "kubectl-cp" || "${PAYLOAD_COPY_MODE}" == "auto" ]]; then
    if "${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_TARBALL}" "${LOGIN_POD}:${STAGE_ROOT}/payload.tgz" >/dev/null 2>&1; then
      :
    elif [[ "${PAYLOAD_COPY_MODE}" == "kubectl-cp" ]]; then
      echo "failed to copy payload via kubectl cp" >&2
      exit 1
    else
      cat "${LOCAL_TARBALL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${STAGE_ROOT}/payload.tgz'"
    fi
  else
    cat "${LOCAL_TARBALL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${STAGE_ROOT}/payload.tgz'"
  fi
  "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -euo pipefail
    tar -tzf '${STAGE_ROOT}/payload.tgz' >/dev/null
  "
fi

SBATCH_NODE_DIRECTIVES=""
if [[ -n "${SBATCH_NODELIST}" ]]; then
  SBATCH_NODE_DIRECTIVES+="#SBATCH -w ${SBATCH_NODELIST}"$'\n'
fi
if [[ -n "${SBATCH_EXCLUDE}" ]]; then
  SBATCH_NODE_DIRECTIVES+="#SBATCH --exclude=${SBATCH_EXCLUDE}"$'\n'
fi

SBATCH_OUTPUT="$(
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  cat >'${SBATCH_FILE}' <<'EOF'
#!/usr/bin/env bash
#SBATCH -J native-cuda-smoke
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=gpuorangefs
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
${SBATCH_NODE_DIRECTIVES}set -euo pipefail

RUN_ROOT='${STAGE_ROOT}'
LOCAL_ROOT=\"\${TMPDIR:-/tmp}/native-cuda-smoke-\${SLURM_JOB_ID}\"
SOUNIO_DIR=\"\${LOCAL_ROOT}/repo\"
RESULTS_DIR=\"\${RUN_ROOT}/results\"
LOG_DIR=\"\${RUN_ROOT}/logs\"
FIXTURE=\"\${SOUNIO_DIR}/${FIXTURE_REL}\"
PAYLOAD_TGZ=\"\${RUN_ROOT}/payload.tgz\"

mkdir -p \"\${RESULTS_DIR}\" \"\${LOG_DIR}\"
exec > >(tee \"\${LOG_DIR}/job-\${SLURM_JOB_ID}.log\") 2>&1

echo '═══════════════════════════════════════════════════════════════'
echo \"  Native CUDA Smoke — \$(date)\"
echo \"  Host: \$(hostname)\"
echo \"  CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}\"
echo \"  Fixture: \${FIXTURE}\"
echo '═══════════════════════════════════════════════════════════════'

echo
echo '[Phase 0] Extracting staged payload to local worker storage'
rm -rf \"\${LOCAL_ROOT}\"
mkdir -p \"\${SOUNIO_DIR}\"
tar -xzf \"\${PAYLOAD_TGZ}\" -C \"\${SOUNIO_DIR}\"
chmod +x \"\${SOUNIO_DIR}/bin/souc\" \"\${SOUNIO_DIR}/artifacts/self-hosted/souc-self-hosted-x86_64\"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo '[nvidia-smi]'
  nvidia-smi -L || true
fi

echo
echo '[Phase 1] Running native CUDA smoke'
SOUNIO_CUDA_SMOKE_STRICT=1 bash \"\${SOUNIO_DIR}/scripts/gpu/run_native_cuda_smoke.sh\" \"\${FIXTURE}\" \
  | tee \"\${RESULTS_DIR}/native_cuda_smoke.txt\"

if ! grep -q 'CUDA_SMOKE_OK' \"\${RESULTS_DIR}/native_cuda_smoke.txt\"; then
  echo 'native CUDA smoke did not report CUDA_SMOKE_OK' >&2
  exit 1
fi

echo
echo '═══════════════════════════════════════════════════════════════'
echo \"  Smoke complete — \$(date)\"
echo '═══════════════════════════════════════════════════════════════'
EOF
  sbatch '${SBATCH_FILE}'
  rm -f '${SBATCH_FILE}'
  echo ---
  squeue
"
)"

echo "${SBATCH_OUTPUT}"

JOB_ID="$(
  printf '%s\n' "${SBATCH_OUTPUT}" \
    | awk '/Submitted batch job/ {print $4; exit}'
)"

if [[ -z "${JOB_ID}" ]]; then
  echo "failed to parse job id from sbatch output" >&2
  exit 1
fi

echo
echo "Submitted native CUDA smoke job:"
echo "  RUN_ID: ${RUN_ID}"
echo "  JobID: ${JOB_ID}"
echo "  Stage root: ${STAGE_ROOT}"
echo "  Inspect: ${KUBECTL_BIN} -n ${NS} exec ${LOGIN_POD} -- sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Start,End"
