#!/usr/bin/env bash
set -euo pipefail

# Submit from the control plane:
#   cd /home/devsounio/beagle/k8s/hpc-sota
#   source ops/lab-ops.sh
#   lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-gpu.sh

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/home/devsounio/sounio}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
RUN_ID="${RUN_ID:-brain-ossm-abide-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/abide-runs/${RUN_ID}"
ORANGEFS_RESULTS_DIR="${ORANGEFS_RESULTS_DIR:-/orangefs/training/sounio/abide-results}"
ABIDE_MANIFEST_PATH="${ABIDE_MANIFEST_PATH:-/orangefs/training/sounio/abide-data/abide_roi_manifest.tsv}"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
LOCAL_SNAPSHOT_DIR="${LOCAL_SNAPSHOT_DIR:-/tmp/${RUN_ID}-snapshot}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_TARBALL_TMP="${LOCAL_TARBALL}.tmp"
FORCE_RESTAGE="${FORCE_RESTAGE:-0}"

REQUIRED_FILES=(
  "bin/souc"
  "artifacts/self-hosted/souc-self-hosted-x86_64"
  "examples/brain_ossm_abide.sio"
  "scripts/research/abide_campaign_lib.py"
  "scripts/research/build_abide_temporal_manifest.py"
  "scripts/research/parse_brain_ossm_abide_output.py"
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
    elif [[ -x '${STAGE_ROOT}/repo/bin/souc' && -x '${STAGE_ROOT}/repo/artifacts/self-hosted/souc-self-hosted-x86_64' && -f '${STAGE_ROOT}/repo/examples/brain_ossm_abide.sio' && -f '${STAGE_ROOT}/repo/scripts/research/abide_campaign_lib.py' && -f '${STAGE_ROOT}/repo/scripts/research/build_abide_temporal_manifest.py' && -f '${STAGE_ROOT}/repo/scripts/research/parse_brain_ossm_abide_output.py' ]]; then
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
    chmod +x '${STAGE_ROOT}/repo/bin/souc' '${STAGE_ROOT}/repo/artifacts/self-hosted/souc-self-hosted-x86_64'
    '${STAGE_ROOT}/repo/bin/souc' info >/dev/null
  "
fi

SBATCH_OUTPUT="$(
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  cat >'${SBATCH_FILE}' <<'EOF'
#!/usr/bin/env bash
#SBATCH -J brain-ossm-abide
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=gpuorangefs
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail

RUN_ROOT='${STAGE_ROOT}'
SOUNIO_DIR=\"\${RUN_ROOT}/repo\"
SOUC=\"\${SOUNIO_DIR}/bin/souc\"
BUILD_DIR=\"\$(mktemp -d /tmp/abide-build.XXXXXX)\"
RESULTS_DIR=\"\${RUN_ROOT}/results\"
ORANGEFS_DIR='${ORANGEFS_RESULTS_DIR}'
LOG_DIR=\"\${RUN_ROOT}/logs\"

cleanup() {
  rm -rf \"\${BUILD_DIR}\"
}
trap cleanup EXIT

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

mkdir -p \"\${RESULTS_DIR}\" \"\${ORANGEFS_DIR}\" \"\${LOG_DIR}\"
exec > >(tee \"\${LOG_DIR}/job-\${SLURM_JOB_ID}.log\") 2>&1

echo '═══════════════════════════════════════════════════════════════'
echo \"  Brain O-SSM ABIDE GPU Job — \$(date)\"
echo \"  Host: \$(hostname), GPU: \${CUDA_VISIBLE_DEVICES:-none}\"
echo \"  Run root: \${RUN_ROOT}\"
echo '═══════════════════════════════════════════════════════════════'

echo
echo '[Phase 1] Compiling ABIDE benchmark...'
\"\${SOUC}\" compile \"\${SOUNIO_DIR}/examples/brain_ossm_abide.sio\" -o \"\${BUILD_DIR}/brain_ossm_abide.elf\"

echo
echo '[Phase 2] Running ABIDE benchmark...'
time \"\${BUILD_DIR}/brain_ossm_abide.elf\" > \"\${RESULTS_DIR}/brain_ossm_abide_results.txt\" 2>&1

echo
echo '[Phase 3] Parsing structured metrics...'
python3 \"\${SOUNIO_DIR}/scripts/research/parse_brain_ossm_abide_output.py\" \
  --input \"\${RESULTS_DIR}/brain_ossm_abide_results.txt\" \
  --output-dir \"\${RESULTS_DIR}/sounio_structured\"

echo
echo '[Phase 4] Persisting results...'
copy_with_retry \"\${RESULTS_DIR}/brain_ossm_abide_results.txt\" \"\${ORANGEFS_DIR}/brain_ossm_abide_results.txt\"
copy_with_retry \"\${RESULTS_DIR}/sounio_structured/overall_metrics.json\" \"\${ORANGEFS_DIR}/brain_ossm_abide_overall_metrics.json\"
copy_with_retry \"\${RESULTS_DIR}/sounio_structured/overall_metrics.tsv\" \"\${ORANGEFS_DIR}/brain_ossm_abide_overall_metrics.tsv\"
copy_with_retry \"\${RESULTS_DIR}/sounio_structured/per_seed_metrics.tsv\" \"\${ORANGEFS_DIR}/brain_ossm_abide_per_seed_metrics.tsv\"
copy_with_retry \"\${RESULTS_DIR}/sounio_structured/per_site_metrics.tsv\" \"\${ORANGEFS_DIR}/brain_ossm_abide_per_site_metrics.tsv\"
copy_with_retry \"\${RESULTS_DIR}/sounio_structured/prediction_rows.tsv\" \"\${ORANGEFS_DIR}/brain_ossm_abide_prediction_rows.tsv\"
copy_with_retry \"\${LOG_DIR}/job-\${SLURM_JOB_ID}.log\" \"\${ORANGEFS_DIR}/job-\${SLURM_JOB_ID}.log\"

echo
echo '═══════════════════════════════════════════════════════════════'
echo \"  Job complete — \$(date)\"
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
echo "Submitted ABIDE job:"
echo "  RUN_ID: ${RUN_ID}"
echo "  JobID: ${JOB_ID}"
echo "  Stage root: ${STAGE_ROOT}"
echo "  Results dir: ${ORANGEFS_RESULTS_DIR}"
echo "  Inspect: kubectl -n ${NS} exec ${LOGIN_POD} -- sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Start,End"
