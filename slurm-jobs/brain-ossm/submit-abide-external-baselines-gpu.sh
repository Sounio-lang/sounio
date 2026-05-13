#!/usr/bin/env bash
set -euo pipefail

# Submit from the control plane:
#   cd /home/devsounio/beagle/k8s/hpc-sota
#   source ops/lab-ops.sh
#   lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-external-baselines-gpu.sh

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/home/devsounio/sounio}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
RUN_ID="${RUN_ID:-brain-ossm-abide-baselines-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/abide-baseline-runs/${RUN_ID}"
ORANGEFS_RESULTS_DIR="${ORANGEFS_RESULTS_DIR:-/orangefs/training/sounio/abide-baselines}"
ABIDE_MANIFEST_PATH="${ABIDE_MANIFEST_PATH:-/orangefs/training/sounio/abide-data/abide_roi_manifest.tsv}"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
LOCAL_SNAPSHOT_DIR="${LOCAL_SNAPSHOT_DIR:-/tmp/${RUN_ID}-snapshot}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_TARBALL_TMP="${LOCAL_TARBALL}.tmp"
FORCE_RESTAGE="${FORCE_RESTAGE:-0}"
BASELINE_MODELS="${BASELINE_MODELS:-lstm,gru,transformer,tcn}"
SEED_COUNT="${SEED_COUNT:-20}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
DROP_CHANNEL_FRAC="${DROP_CHANNEL_FRAC:-0.0}"
NOISE_STD="${NOISE_STD:-0.0}"
MAX_SITES="${MAX_SITES:-}"
LIMIT_SUBJECTS="${LIMIT_SUBJECTS:-}"
JOB_MEM="${JOB_MEM:-8G}"
PYTORCH_VENV_DIR="${PYTORCH_VENV_DIR:-/tmp/${RUN_ID}-venvs/brain-ossm-external-baselines}"
PYTORCH_USERBASE_DIR="${PYTORCH_USERBASE_DIR:-/tmp/${RUN_ID}-python-userbase/brain-ossm-external-baselines}"

REQUIRED_FILES=(
  "scripts/research/abide_campaign_lib.py"
  "scripts/research/build_abide_temporal_manifest.py"
  "scripts/research/normalize_abide_manifest.py"
  "scripts/research/abide_external_baselines.py"
  "scripts/gpu/prepare_abide_campaign_snapshot.sh"
)

for rel in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
    echo "missing required file at ${SOUNIO_DIR}/${rel}" >&2
    exit 1
  fi
done

if ! kubectl -n "${NS}" get deploy "${LOGIN_DEPLOY_NAME}" >/dev/null 2>&1; then
  echo "could not find deploy/${LOGIN_DEPLOY_NAME} in namespace ${NS}" >&2
  exit 1
fi

LOGIN_POD="$(kubectl -n "${NS}" get pods \
  -l "app.kubernetes.io/instance=${LOGIN_DEPLOY_NAME},app.kubernetes.io/name=login" \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')"

if [[ -z "${LOGIN_POD}" ]]; then
  echo "could not resolve a live login pod for ${LOGIN_DEPLOY_NAME}" >&2
  exit 1
fi

kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc '
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

kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  test -f '${ABIDE_MANIFEST_PATH}'
  mkdir -p '${STAGE_ROOT}/repo' '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs' '${ORANGEFS_RESULTS_DIR}'
"

REMOTE_STAGE_READY="$(
  kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -euo pipefail
    if [[ '${FORCE_RESTAGE}' != '0' ]]; then
      echo NO
    elif [[ -f '${STAGE_ROOT}/repo/scripts/research/abide_campaign_lib.py' && -f '${STAGE_ROOT}/repo/scripts/research/build_abide_temporal_manifest.py' && -f '${STAGE_ROOT}/repo/scripts/research/abide_external_baselines.py' ]]; then
      echo YES
    else
      echo NO
    fi
  " | tr -d '[:space:]'
)"

if [[ "${REMOTE_STAGE_READY}" != "YES" && ! -s "${LOCAL_TARBALL}" ]]; then
  rm -rf "${LOCAL_SNAPSHOT_DIR}"
  rm -f "${LOCAL_TARBALL}" "${LOCAL_TARBALL_TMP}"
  OUT_ROOT="${LOCAL_SNAPSHOT_DIR}" SOUNIO_DIR="${SOUNIO_DIR}" \
    bash "${SOUNIO_DIR}/scripts/gpu/prepare_abide_campaign_snapshot.sh" >/dev/null

  TAR_READY=0
  for attempt in $(seq 1 3); do
    rm -f "${LOCAL_TARBALL_TMP}"
    if tar -C "${LOCAL_SNAPSHOT_DIR}" -czf "${LOCAL_TARBALL_TMP}" . && tar -tzf "${LOCAL_TARBALL_TMP}" >/dev/null; then
      TAR_READY=1
      break
    fi
    sleep 1
  done
  if [[ "${TAR_READY}" != "1" ]]; then
    echo "failed to build validated local tarball for ${RUN_ID}" >&2
    exit 1
  fi
  mv -f "${LOCAL_TARBALL_TMP}" "${LOCAL_TARBALL}"
fi

if [[ "${REMOTE_STAGE_READY}" != "YES" ]]; then
  cat "${LOCAL_TARBALL}" | kubectl -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${STAGE_ROOT}/payload.tgz'"
  kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -euo pipefail
    rm -rf '${STAGE_ROOT}/repo'
    mkdir -p '${STAGE_ROOT}/repo'
    tar -xzf '${STAGE_ROOT}/payload.tgz' -C '${STAGE_ROOT}/repo'
    rm -f '${STAGE_ROOT}/payload.tgz'
  "
fi

SBATCH_OUTPUT="$(
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  cat >'${SBATCH_FILE}' <<'EOF'
#!/usr/bin/env bash
#SBATCH -J brain-ossm-abide-ext
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=gpuorangefs
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=01:00:00
set -euo pipefail

RUN_ROOT='${STAGE_ROOT}'
SOUNIO_DIR=\"\${RUN_ROOT}/repo\"
RESULTS_DIR=\"\${RUN_ROOT}/results\"
LOG_DIR=\"\${RUN_ROOT}/logs\"
ORANGEFS_DIR='${ORANGEFS_RESULTS_DIR}'
MANIFEST_PATH='${ABIDE_MANIFEST_PATH}'
RUN_MANIFEST_PATH=\"\${SOUNIO_DIR}/abide_roi_manifest.tsv\"
VENV_DIR='${PYTORCH_VENV_DIR}'
USERBASE_DIR='${PYTORCH_USERBASE_DIR}'

copy_with_retry() {
  local src=\"\$1\"
  local dst=\"\$2\"
  local attempt
  local tmp_dst=\"\${dst}.tmp\"
  for attempt in \$(seq 1 20); do
    mkdir -p \"\$(dirname \"\$dst\")\" || true
    rm -f \"\$dst\" \"\$tmp_dst\" || true
    if cp -f \"\$src\" \"\$tmp_dst\" && [ -s \"\$tmp_dst\" ] && mv -f \"\$tmp_dst\" \"\$dst\" && [ -s \"\$dst\" ]; then
      return 0
    fi
    sleep 1
  done
  echo \"failed to copy \$src -> \$dst\" >&2
  return 1
}

mkdir -p \"\${RESULTS_DIR}\" \"\${ORANGEFS_DIR}\" \"\${LOG_DIR}\" \"\${VENV_DIR}\" \"\${USERBASE_DIR}\"
exec > >(tee \"\${LOG_DIR}/job-\${SLURM_JOB_ID}.log\") 2>&1

echo '═══════════════════════════════════════════════════════════════'
echo \"  Brain O-SSM ABIDE External Baselines — \$(date)\"
echo \"  Host: \$(hostname), GPU: \${CUDA_VISIBLE_DEVICES:-none}\"
echo \"  Run root: \${RUN_ROOT}\"
echo '═══════════════════════════════════════════════════════════════'

echo
echo '[Phase 0] Materializing run-local manifest...'
NORM_ARGS=()
if [[ -n \"${MAX_SITES}\" ]]; then
  NORM_ARGS+=(--max-sites \"${MAX_SITES}\")
fi
if [[ -n \"${LIMIT_SUBJECTS}\" ]]; then
  NORM_ARGS+=(--limit-subjects \"${LIMIT_SUBJECTS}\")
fi
python3 \"\${SOUNIO_DIR}/scripts/research/normalize_abide_manifest.py\" \
  --input \"\${MANIFEST_PATH}\" \
  --output \"\${RUN_MANIFEST_PATH}\" \
  --layout flat \
  \"\${NORM_ARGS[@]}\"
echo \"  Source manifest: \${MANIFEST_PATH}\"
echo \"  Run manifest: \${RUN_MANIFEST_PATH}\"

bootstrap_userbase_torch() {
  local pybin=\"\$1\"
  local get_pip=\"\${LOG_DIR}/get-pip.py\"
  export PYTHONUSERBASE=\"\${USERBASE_DIR}\"
  if ! \"\${pybin}\" -m pip --version >/dev/null 2>&1; then
    LOG_DIR=\"\${LOG_DIR}\" \"\${pybin}\" - <<'PY'
import os
import pathlib
import ssl
import urllib.request

path = pathlib.Path(os.environ['LOG_DIR']) / 'get-pip.py'
ctx = ssl._create_unverified_context()
with urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py', context=ctx) as resp:
    path.write_bytes(resp.read())
print(path)
PY
    \"\${pybin}\" \"\${get_pip}\" --user --break-system-packages --trusted-host bootstrap.pypa.io >/dev/null
  fi
  export PATH=\"\${PYTHONUSERBASE}/bin:\${PATH}\"
  \"\${pybin}\" -m pip install --user --no-cache-dir --break-system-packages \
    --trusted-host download.pytorch.org \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.6.0 \
    numpy==2.2.6
  export PYTHONPATH=\"\$(\"\${pybin}\" - <<'PY'
import site
print(site.getusersitepackages())
PY
):\${PYTHONPATH:-}\"
}

PYTHON_RUNNER=\"\${BASELINE_PYTHON:-}\"
if [[ -n \"\${PYTHON_RUNNER}\" && ! -x \"\${PYTHON_RUNNER}\" ]]; then
  echo \"BASELINE_PYTHON is set but not executable: \${PYTHON_RUNNER}\" >&2
  exit 1
fi

if [[ -n \"\${PYTHON_RUNNER}\" ]] && \"\${PYTHON_RUNNER}\" -c 'import torch' >/dev/null 2>&1; then
  :
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import torch' >/dev/null 2>&1; then
  PYTHON_RUNNER=\"\$(command -v python3)\"
elif [[ -x \"\${VENV_DIR}/bin/python\" ]] && \"\${VENV_DIR}/bin/python\" -c 'import torch' >/dev/null 2>&1; then
  PYTHON_RUNNER=\"\${VENV_DIR}/bin/python\"
else
  PYTHON_RUNNER=\"\"
  if command -v python3 >/dev/null 2>&1; then
    if python3 -m venv \"\${VENV_DIR}\" >/dev/null 2>&1; then
      PYTHON_RUNNER=\"\${VENV_DIR}/bin/python\"
      if ! \"\${PYTHON_RUNNER}\" -c 'import torch' >/dev/null 2>&1; then
        if ! \"\${PYTHON_RUNNER}\" -m pip --version >/dev/null 2>&1; then
          PYTHON_RUNNER=\"\"
        else
          \"\${PYTHON_RUNNER}\" -m pip install --upgrade pip >/dev/null
          \"\${PYTHON_RUNNER}\" -m pip install \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==2.6.0
        fi
      fi
    fi
    if [[ -z \"\${PYTHON_RUNNER}\" ]] || ! \"\${PYTHON_RUNNER}\" -c 'import torch' >/dev/null 2>&1; then
      PYTHON_RUNNER=\"\$(command -v python3)\"
      bootstrap_userbase_torch \"\${PYTHON_RUNNER}\"
    fi
  fi
fi

if [[ -z \"\${PYTHON_RUNNER}\" ]] || ! \"\${PYTHON_RUNNER}\" -c 'import torch' >/dev/null 2>&1; then
  echo 'failed to provision a Python runtime with torch support for external baselines' >&2
  exit 1
fi

echo
echo '[Phase 1] Running external baseline suite...'
\"\${PYTHON_RUNNER}\" \"\${SOUNIO_DIR}/scripts/research/abide_external_baselines.py\" \
  --manifest \"\${RUN_MANIFEST_PATH}\" \
  --output-dir \"\${RESULTS_DIR}/external_baselines\" \
  --models '${BASELINE_MODELS}' \
  --seeds ${SEED_COUNT} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LEARNING_RATE} \
  --train-fraction ${TRAIN_FRACTION} \
  --drop-channel-frac ${DROP_CHANNEL_FRAC} \
  --noise-std ${NOISE_STD} \
  --device cpu

echo
echo '[Phase 2] Persisting results...'
copy_with_retry \"\${RESULTS_DIR}/external_baselines/manifest_meta.json\" \"\${ORANGEFS_DIR}/manifest_meta.json\"
copy_with_retry \"\${RESULTS_DIR}/external_baselines/overall_metrics.json\" \"\${ORANGEFS_DIR}/overall_metrics.json\"
copy_with_retry \"\${RESULTS_DIR}/external_baselines/overall_metrics.tsv\" \"\${ORANGEFS_DIR}/overall_metrics.tsv\"
copy_with_retry \"\${RESULTS_DIR}/external_baselines/per_seed_metrics.tsv\" \"\${ORANGEFS_DIR}/per_seed_metrics.tsv\"
copy_with_retry \"\${RESULTS_DIR}/external_baselines/per_site_metrics.tsv\" \"\${ORANGEFS_DIR}/per_site_metrics.tsv\"
copy_with_retry \"\${RESULTS_DIR}/external_baselines/prediction_rows.tsv\" \"\${ORANGEFS_DIR}/prediction_rows.tsv\"
copy_with_retry \"\${LOG_DIR}/job-\${SLURM_JOB_ID}.log\" \"\${ORANGEFS_DIR}/job-\${SLURM_JOB_ID}.log\"
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
echo "Submitted ABIDE external baseline job:"
echo "  RUN_ID: ${RUN_ID}"
echo "  JobID: ${JOB_ID}"
echo "  Stage root: ${STAGE_ROOT}"
echo "  Results dir: ${ORANGEFS_RESULTS_DIR}"
echo "  Inspect: kubectl -n ${NS} exec ${LOGIN_POD} -- sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Start,End"
