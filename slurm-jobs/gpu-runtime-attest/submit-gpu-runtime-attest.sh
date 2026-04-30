#!/usr/bin/env bash
set -euo pipefail

# Submit GPU Runtime Attest Gate via Slurm
# =========================================
# Runs the full omega_gpu_runtime_remote_runner.sh on a real GPU node
# through Slurm, producing a signed attestation artifact.
#
# Usage:
#   cd /workspace/sounio
#   bash slurm-jobs/gpu-runtime-attest/submit-gpu-runtime-attest.sh
#
# After completion, pull the artifact:
#   kubectl -n slurm-pilot exec <login-pod> -- cat \
#     /orangefs/training/sounio/gpu-runtime-attest/<RUN_ID>/results/gpu_runtime_attestation.json

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
RUN_ID="${RUN_ID:-gpu-runtime-attest-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/gpu-runtime-attest/${RUN_ID}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_TARBALL_TMP="${LOCAL_TARBALL}.tmp"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
JOB_MEM="${JOB_MEM:-8G}"
JOB_TIME="${JOB_TIME:-00:15:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"
FORCE_RESTAGE="${FORCE_RESTAGE:-0}"
PAYLOAD_COPY_MODE="${PAYLOAD_COPY_MODE:-auto}"

REQUIRED_FILES=(
  "artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
  "scripts/omega/omega_gpu_runtime_remote_runner.sh"
  "scripts/omega/omega_resolve_souc_bin.sh"
  "scripts/fixtures/gpu_minimal.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_literal.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_file_text.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_dynamic_slice.sio"
  "tests/run-pass/gpu_kernel_basic.sio"
  "tests/fixtures/fmri/tiny_real_motion.tsv"
  "tests/fixtures/fmri/tiny_real_slice_nifti1.nii"
  "stdlib"
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
    artifacts/omega/souc-bin/souc-linux-x86_64-gpu \
    scripts/omega/omega_gpu_runtime_remote_runner.sh \
    scripts/omega/omega_resolve_souc_bin.sh \
    scripts/fixtures/gpu_minimal.sio \
    tests/selfhost/gpu_runtime/test_gpu_runtime_literal.sio \
    tests/selfhost/gpu_runtime/test_gpu_runtime_file_text.sio \
    tests/selfhost/gpu_runtime/test_gpu_runtime_dynamic_slice.sio \
    tests/run-pass/gpu_kernel_basic.sio \
    tests/fixtures/fmri/tiny_real_motion.tsv \
    tests/fixtures/fmri/tiny_real_slice_nifti1.nii \
    stdlib

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

# ---------------------------------------------------------------------------
# Build sbatch script locally, then push it to the login pod
# ---------------------------------------------------------------------------
SBATCH_LOCAL="/tmp/${RUN_ID}.sbatch.local"
rm -f "${SBATCH_LOCAL}"

cat > "${SBATCH_LOCAL}" <<EOF
#!/usr/bin/env bash
#SBATCH -J gpu-runtime-attest
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

if [[ -n "${SBATCH_NODELIST}" ]]; then
  echo "#SBATCH -w ${SBATCH_NODELIST}" >> "${SBATCH_LOCAL}"
fi
if [[ -n "${SBATCH_EXCLUDE}" ]]; then
  echo "#SBATCH --exclude=${SBATCH_EXCLUDE}" >> "${SBATCH_LOCAL}"
fi

cat >> "${SBATCH_LOCAL}" <<'SBATCH_BODY'
set -euo pipefail

RUN_ROOT='RUN_ROOT_PLACEHOLDER'
LOCAL_ROOT="${TMPDIR:-/tmp}/gpu-runtime-attest-${SLURM_JOB_ID}"
SOUNIO_DIR="${LOCAL_ROOT}/repo"
RESULTS_DIR="${RUN_ROOT}/results"
LOG_DIR="${RUN_ROOT}/logs"
PAYLOAD_TGZ="${RUN_ROOT}/payload.tgz"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/job-${SLURM_JOB_ID}.log") 2>&1

echo '═══════════════════════════════════════════════════════════════'
echo "  GPU Runtime Attest Gate — $(date)"
echo "  Host: $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo '═══════════════════════════════════════════════════════════════'

echo
echo '[Phase 0] Extracting staged payload to local worker storage'
rm -rf "${LOCAL_ROOT}"
mkdir -p "${SOUNIO_DIR}"
tar -xzf "${PAYLOAD_TGZ}" -C "${SOUNIO_DIR}"
chmod +x "${SOUNIO_DIR}/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo '[nvidia-smi]'
  nvidia-smi -L || true
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || true
fi

echo
echo '[Phase 1] Generating attestation manifest'
cd "${SOUNIO_DIR}"

NONCE="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
EXPECTED_VERSION="1.0.0-beta.4"

MANIFEST_B64="$(python3 - "${SOUNIO_DIR}" "${NONCE}" "${EXPECTED_VERSION}" <<'PY'
import base64
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
nonce = sys.argv[2]
expected_version = sys.argv[3]

probe_files = [
    'tests/selfhost/gpu_runtime/test_gpu_runtime_literal.sio',
    'tests/selfhost/gpu_runtime/test_gpu_runtime_file_text.sio',
    'tests/selfhost/gpu_runtime/test_gpu_runtime_dynamic_slice.sio',
]

probe_hashes = {}
for rel in probe_files:
    p = (root / rel).resolve()
    if p.exists():
        probe_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()

obj = {
    'schema': 'sounio.omega.gpu_runtime_attest_manifest.v1',
    'nonce': nonce,
    'souc_version_expected': expected_version,
    'probe_hashes': probe_hashes,
    'required_tests': [
        'gpu_runtime_literal',
        'gpu_runtime_file_text',
        'gpu_runtime_dynamic_slice',
        'gpu_backend_compile_smoke',
        'gpu_kernel_basic_runtime_smoke',
    ],
}
payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
print(base64.b64encode(payload).decode('ascii'))
PY
)"

echo
echo '[Phase 2] Running GPU runtime remote runner'
export OMEGA_GPU_ATTEST_NONCE="${NONCE}"
export OMEGA_GPU_ATTEST_MANIFEST_B64="${MANIFEST_B64}"
export OMEGA_GPU_ATTEST_BOOTSTRAPPED_SOUC="artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
export SOUNIO_SOUC_VERSION="${EXPECTED_VERSION}"

bash "${SOUNIO_DIR}/scripts/omega/omega_gpu_runtime_remote_runner.sh" \
  > "${RESULTS_DIR}/gpu_runtime_attestation.json" 2>"${LOG_DIR}/remote_runner.log"

echo
echo '[Phase 3] Verifying attestation output'
if [[ -s "${RESULTS_DIR}/gpu_runtime_attestation.json" ]]; then
  STATUS="$(python3 -c "
import json, sys
with open('${RESULTS_DIR}/gpu_runtime_attestation.json') as f:
    w = json.load(f)
p = w.get('payload', {})
print(p.get('status_summary', 'unknown'))
" )"
  echo "Attestation status: ${STATUS}"
  if [[ "${STATUS}" == 'pass' ]]; then
    echo 'GPU_RUNTIME_ATTEST_OK'
  else
    echo 'GPU_RUNTIME_ATTEST_FAIL'
    exit 1
  fi
else
  echo 'GPU_RUNTIME_ATTEST_MISSING'
  exit 1
fi

echo
echo '═══════════════════════════════════════════════════════════════'
echo "  Attest complete — $(date)"
echo '═══════════════════════════════════════════════════════════════'
SBATCH_BODY

# Replace the placeholder with the actual stage root
sed -i "s|RUN_ROOT_PLACEHOLDER|${STAGE_ROOT}|g" "${SBATCH_LOCAL}"

# Copy sbatch script to login pod and submit
"${KUBECTL_BIN}" -n "${NS}" cp "${SBATCH_LOCAL}" "${LOGIN_POD}:${SBATCH_FILE}" >/dev/null 2>&1 || \
  cat "${SBATCH_LOCAL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${SBATCH_FILE}'"

SBATCH_OUTPUT="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  sbatch '${SBATCH_FILE}'
  rm -f '${SBATCH_FILE}'
  echo ---
  squeue
")"

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
echo "Submitted GPU runtime attest job:"
echo "  RUN_ID: ${RUN_ID}"
echo "  JobID: ${JOB_ID}"
echo "  Stage root: ${STAGE_ROOT}"
echo "  Inspect: ${KUBECTL_BIN} -n ${NS} exec ${LOGIN_POD} -- sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Start,End"
echo
echo "Pull attestation when done:"
echo "  ${KUBECTL_BIN} -n ${NS} exec ${LOGIN_POD} -- cat ${STAGE_ROOT}/results/gpu_runtime_attestation.json"
