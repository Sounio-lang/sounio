#!/usr/bin/env bash
set -euo pipefail

# Submit the Sounio-owned NVIDIA bare CUBIN gate to the Slurm GPU worker.
# This follows the maintained login-pod -> OrangeFS -> sbatch path used by the
# native CUDA smoke, because the workspace pod itself is not the GPU runtime.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-$ROOT_DIR}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
RUN_ID="${RUN_ID:-nvidia-bare-metal-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_PARENT="${STAGE_PARENT:-/orangefs/training/sounio/slurm-smokes/nvidia-bare-metal}"
STAGE_ROOT="${STAGE_PARENT}/${RUN_ID}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_TARBALL_TMP="${LOCAL_TARBALL}.tmp"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
JOB_MEM="${JOB_MEM:-4G}"
JOB_TIME="${JOB_TIME:-00:10:00}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"
FORCE_RESTAGE="${FORCE_RESTAGE:-0}"
PAYLOAD_COPY_MODE="${PAYLOAD_COPY_MODE:-base64-chunks}"
RUNTIME_RUNG="${SOUNIO_NVIDIA_BARE_RUNTIME_RUNG:-admission}"
case "${RUNTIME_RUNG}" in
  vec_add_f32|epistemic_elementwise_f32)
    RESULT_STEM="sounio_bare_vec_add_f32_sm80"
    ;;
  *)
    RESULT_STEM="sounio_bare_store_u32_const_sm80"
    ;;
esac
VEC_N="${SOUNIO_NVIDIA_BARE_VEC_N:-64}"
WAIT_FOR_RESULT="${WAIT_FOR_RESULT:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-300}"

REQUIRED_FILES=(
  "bin/souc"
  "bin/souc-linux-x86_64"
  "scripts/lib/resolve_souc.sh"
  "scripts/ci/native_v2_nvidia_bare_metal_gate.sh"
  "scripts/gpu/nvidia_bare_cubin_cartographer.py"
  "scripts/gpu/nvidia_bare_driver_loader.c"
  "self-hosted/gpu/nvidia_bare.sio"
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
  echo "[1/4] staging NVIDIA bare payload locally"
  rm -f "${LOCAL_TARBALL}" "${LOCAL_TARBALL_TMP}"
  tar -C "${SOUNIO_DIR}" -czf "${LOCAL_TARBALL_TMP}" \
    bin/souc \
    bin/souc-linux-x86_64 \
    scripts/lib/resolve_souc.sh \
    scripts/ci/native_v2_nvidia_bare_metal_gate.sh \
    scripts/gpu/nvidia_bare_cubin_cartographer.py \
    scripts/gpu/nvidia_bare_driver_loader.c \
    self-hosted/gpu/nvidia_bare.sio \
    artifacts/sass/cubin \
    artifacts/sass/omega
  tar -tzf "${LOCAL_TARBALL_TMP}" >/dev/null
  mv -f "${LOCAL_TARBALL_TMP}" "${LOCAL_TARBALL}"

  echo "[2/4] copying payload to login pod (${PAYLOAD_COPY_MODE})"
  if [[ "${PAYLOAD_COPY_MODE}" == "kubectl-cp" || "${PAYLOAD_COPY_MODE}" == "auto" ]]; then
    if "${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_TARBALL}" "${LOGIN_POD}:${STAGE_ROOT}/payload.tgz" >/dev/null 2>&1; then
      :
    elif [[ "${PAYLOAD_COPY_MODE}" == "kubectl-cp" ]]; then
      echo "failed to copy payload via kubectl cp" >&2
      exit 1
    else
      cat "${LOCAL_TARBALL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${STAGE_ROOT}/payload.tgz'"
    fi
  elif [[ "${PAYLOAD_COPY_MODE}" == "base64-chunks" ]]; then
    B64_PAYLOAD="${LOCAL_TARBALL}.b64"
    B64_CHUNK_DIR="${LOCAL_TARBALL}.chunks"
    rm -rf "${B64_CHUNK_DIR}"
    mkdir -p "${B64_CHUNK_DIR}"
    base64 "${LOCAL_TARBALL}" > "${B64_PAYLOAD}"
    split -b 24000 "${B64_PAYLOAD}" "${B64_CHUNK_DIR}/chunk."
    "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sh -lc "rm -f '${STAGE_ROOT}/payload.tgz.b64' '${STAGE_ROOT}/payload.tgz'"
    for chunk in "${B64_CHUNK_DIR}"/chunk.*; do
      chunk_text="$(tr -d '\n' < "${chunk}")"
      "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sh -lc "printf '%s' '${chunk_text}' >> '${STAGE_ROOT}/payload.tgz.b64'"
    done
    "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sh -lc "base64 -d '${STAGE_ROOT}/payload.tgz.b64' > '${STAGE_ROOT}/payload.tgz'; rm -f '${STAGE_ROOT}/payload.tgz.b64'"
    rm -rf "${B64_CHUNK_DIR}" "${B64_PAYLOAD}"
  else
    cat "${LOCAL_TARBALL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${STAGE_ROOT}/payload.tgz'"
  fi

  echo "[3/4] verifying remote payload"
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
#SBATCH -J nvidia-bare
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
LOCAL_ROOT=\"\${TMPDIR:-/tmp}/nvidia-bare-\${SLURM_JOB_ID}\"
SOUNIO_DIR=\"\${LOCAL_ROOT}/repo\"
RESULTS_DIR=\"\${RUN_ROOT}/results\"
LOG_DIR=\"\${RUN_ROOT}/logs\"
PAYLOAD_TGZ=\"\${RUN_ROOT}/payload.tgz\"

mkdir -p \"\${RESULTS_DIR}\" \"\${LOG_DIR}\"
exec > >(tee \"\${LOG_DIR}/job-\${SLURM_JOB_ID}.log\") 2>&1

echo '═══════════════════════════════════════════════════════════════'
echo \"  Sounio NVIDIA Bare Runtime Admission — \$(date)\"
echo \"  Host: \$(hostname)\"
echo \"  CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}\"
echo \"  Runtime rung: ${RUNTIME_RUNG}\"
echo '═══════════════════════════════════════════════════════════════'

echo
echo '[Phase 0] Extracting staged payload to local worker storage'
rm -rf \"\${LOCAL_ROOT}\"
mkdir -p \"\${SOUNIO_DIR}\"
tar -xzf \"\${PAYLOAD_TGZ}\" -C \"\${SOUNIO_DIR}\"
chmod +x \"\${SOUNIO_DIR}/bin/souc\" \"\${SOUNIO_DIR}/scripts/ci/native_v2_nvidia_bare_metal_gate.sh\"

echo
echo '[Phase 1] Worker CUDA visibility'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo 'nvidia-smi missing'
fi
ldconfig -p 2>/dev/null | grep -E 'libcuda\\.so|libnvidia-ml\\.so' || true

echo
echo '[Phase 2] Running NVIDIA bare gate with runtime enabled'
SOUNIO_NVIDIA_BARE_RUNTIME=1 \
SOUNIO_NVIDIA_BARE_RUNTIME_RUNG='${RUNTIME_RUNG}' \
SOUNIO_NVIDIA_BARE_VEC_N='${VEC_N}' \
SOUNIO_NVIDIA_BARE_GATE_JSON=\"\${RESULTS_DIR}/native_v2_nvidia_bare_metal_gate.v1.json\" \
SOUNIO_NVIDIA_BARE_CUBIN=\"\${RESULTS_DIR}/${RESULT_STEM}.cubin\" \
SOUNIO_NVIDIA_BARE_HEX=\"\${RESULTS_DIR}/${RESULT_STEM}.hex\" \
SOUNIO_NVIDIA_BARE_CARTOGRAPHY_JSON=\"\${RESULTS_DIR}/nvidia_bare_cubin_cartography.v1.json\" \
SOUNIO_NVIDIA_BARE_GATE_LOG=\"\${RESULTS_DIR}/native_v2_nvidia_bare_metal_gate.log\" \
  bash \"\${SOUNIO_DIR}/scripts/ci/native_v2_nvidia_bare_metal_gate.sh\" \
  | tee \"\${RESULTS_DIR}/nvidia_bare_gate_stdout.txt\"

python3 - <<'PY'
import json
from pathlib import Path
p = Path('${STAGE_ROOT}/results/native_v2_nvidia_bare_metal_gate.v1.json')
obj = json.loads(p.read_text())
print('NVIDIA_BARE_STATUS', obj.get('status'), obj.get('reason'))
print('NVIDIA_BARE_RUNTIME', obj.get('runtime', {}).get('status'), obj.get('runtime', {}).get('reason'))
if obj.get('runtime', {}).get('reason') == 'runtime_admission_pass':
    print('NVIDIA_BARE_RUNTIME_ADMISSION_OK')
elif obj.get('runtime', {}).get('reason') == 'runtime_store_u32_const_pass':
    print('NVIDIA_BARE_RUNTIME_STORE_U32_CONST_OK')
elif obj.get('runtime', {}).get('reason') == 'runtime_vec_add_f32_pass':
    print('NVIDIA_BARE_RUNTIME_VEC_ADD_F32_OK')
elif obj.get('runtime', {}).get('reason') == 'runtime_epistemic_elementwise_f32_pass':
    print('NVIDIA_BARE_RUNTIME_EPISTEMIC_ELEMENTWISE_F32_OK')
PY

echo
echo '═══════════════════════════════════════════════════════════════'
echo \"  NVIDIA bare runtime job complete — \$(date)\"
echo '═══════════════════════════════════════════════════════════════'
EOF
  sbatch '${SBATCH_FILE}'
  rm -f '${SBATCH_FILE}'
  echo ---
  squeue
"
)"

echo "[4/4] submitted sbatch"
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
echo "Submitted NVIDIA bare runtime job:"
echo "  RUN_ID: ${RUN_ID}"
echo "  JobID: ${JOB_ID}"
echo "  Stage root: ${STAGE_ROOT}"
echo "  Inspect: ${KUBECTL_BIN} -n ${NS} exec ${LOGIN_POD} -- sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Start,End"

if [[ "${WAIT_FOR_RESULT}" != "1" ]]; then
  exit 0
fi

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- test -f "${STAGE_ROOT}/results/native_v2_nvidia_bare_metal_gate.v1.json" >/dev/null 2>&1; then
    echo
    echo "Result JSON is available:"
    "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- \
      cat "${STAGE_ROOT}/results/native_v2_nvidia_bare_metal_gate.v1.json"
    exit 0
  fi
  sleep 5
done

echo "timed out waiting for result JSON after ${WAIT_TIMEOUT_SECONDS}s" >&2
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- \
  sacct -j "${JOB_ID}" --format=JobID,State,ExitCode,Start,End || true
exit 124
