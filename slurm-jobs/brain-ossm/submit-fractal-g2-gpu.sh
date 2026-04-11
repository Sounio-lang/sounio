#!/usr/bin/env bash
set -euo pipefail

# Submit from the control plane:
#   cd /home/devsounio/beagle/k8s/hpc-sota
#   source ops/lab-ops.sh
#   lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-fractal-g2-gpu.sh
#
# This wrapper stages the minimum self-hosted compiler payload into OrangeFS so
# the Slurm worker never depends on /home/devsounio/sounio being mounted.

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/home/devsounio/sounio}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
LOGIN_TARGET="deploy/${LOGIN_DEPLOY_NAME}"
RUN_ID="${RUN_ID:-brain-ossm-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_ROOT="/orangefs/training/sounio/brain-ossm-runs/${RUN_ID}"
ORANGEFS_RESULTS_DIR="${ORANGEFS_RESULTS_DIR:-/orangefs/training/sounio/ossm-results}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-}"
LOCAL_SNAPSHOT_DIR="${LOCAL_SNAPSHOT_DIR:-/tmp/${RUN_ID}-snapshot}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_TARBALL_TMP="${LOCAL_TARBALL}.tmp"
FORCE_RESTAGE="${FORCE_RESTAGE:-0}"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"

if [[ -n "${KUBECONFIG_PATH}" && -f "${KUBECONFIG_PATH}" ]]; then
  export KUBECONFIG="${KUBECONFIG_PATH}"
fi

REQUIRED_FILES=(
  "bin/souc"
  "artifacts/self-hosted/souc-self-hosted-x86_64"
  "examples/fractal_g2_ossm_v3.sio"
  "examples/brain_ossm_classifier.sio"
  "examples/hssm_native_algebra.sio"
  "examples/multihead_unit_oct_benchmark.sio"
  "examples/associativity_probe_benchmark.sio"
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

echo "Staging benchmark payload to ${STAGE_ROOT}"
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  mkdir -p '${STAGE_ROOT}/repo' '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs' '${ORANGEFS_RESULTS_DIR}'
"

REMOTE_STAGE_READY="$(
  kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -euo pipefail
    if [[ '${FORCE_RESTAGE}' != '0' ]]; then
      echo NO
    elif [[ -x '${STAGE_ROOT}/repo/bin/souc' && -x '${STAGE_ROOT}/repo/artifacts/self-hosted/souc-self-hosted-x86_64' && -f '${STAGE_ROOT}/repo/examples/fractal_g2_ossm_v3.sio' ]]; then
      echo YES
    else
      echo NO
    fi
  " | tr -d '[:space:]'
)"

if [[ "${REMOTE_STAGE_READY}" == "YES" ]]; then
  echo "Reusing remote staged repo at ${STAGE_ROOT}/repo"
else
  echo "Preparing fresh remote stage at ${STAGE_ROOT}/repo"
fi

if [[ "${REMOTE_STAGE_READY}" != "YES" && ! -s "${LOCAL_TARBALL}" ]]; then
  rm -rf "${LOCAL_SNAPSHOT_DIR}"
  rm -f "${LOCAL_TARBALL}" "${LOCAL_TARBALL_TMP}"
  mkdir -p \
    "${LOCAL_SNAPSHOT_DIR}/bin" \
    "${LOCAL_SNAPSHOT_DIR}/artifacts/self-hosted" \
    "${LOCAL_SNAPSHOT_DIR}/examples"
  rsync -a "${SOUNIO_DIR}/bin/souc" "${LOCAL_SNAPSHOT_DIR}/bin/"
  rsync -a "${SOUNIO_DIR}/artifacts/self-hosted/souc-self-hosted-x86_64" "${LOCAL_SNAPSHOT_DIR}/artifacts/self-hosted/"
  rsync -a "${SOUNIO_DIR}/stdlib/" "${LOCAL_SNAPSHOT_DIR}/stdlib/"
  rsync -a "${SOUNIO_DIR}/examples/fractal_g2_ossm_v3.sio" "${LOCAL_SNAPSHOT_DIR}/examples/"
  rsync -a "${SOUNIO_DIR}/examples/brain_ossm_classifier.sio" "${LOCAL_SNAPSHOT_DIR}/examples/"
  rsync -a "${SOUNIO_DIR}/examples/hssm_native_algebra.sio" "${LOCAL_SNAPSHOT_DIR}/examples/"
  rsync -a "${SOUNIO_DIR}/examples/multihead_unit_oct_benchmark.sio" "${LOCAL_SNAPSHOT_DIR}/examples/"
  rsync -a "${SOUNIO_DIR}/examples/associativity_probe_benchmark.sio" "${LOCAL_SNAPSHOT_DIR}/examples/"

  tar -C "${LOCAL_SNAPSHOT_DIR}" -czf "${LOCAL_TARBALL_TMP}" \
    bin/souc \
    artifacts/self-hosted/souc-self-hosted-x86_64 \
    stdlib \
    examples/fractal_g2_ossm_v3.sio \
    examples/brain_ossm_classifier.sio \
    examples/hssm_native_algebra.sio \
    examples/multihead_unit_oct_benchmark.sio \
    examples/associativity_probe_benchmark.sio
  tar -tzf "${LOCAL_TARBALL_TMP}" >/dev/null
  mv -f "${LOCAL_TARBALL_TMP}" "${LOCAL_TARBALL}"
else
  echo "Reusing prebuilt payload tarball at ${LOCAL_TARBALL}"
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
#SBATCH -J fractal-g2-ossm
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
BUILD_DIR=\"\$(mktemp -d /tmp/ossm-build.XXXXXX)\"
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
echo \"  Fractal-G2 O-SSM GPU Job — \$(date)\"
echo \"  Host: \$(hostname), GPU: \${CUDA_VISIBLE_DEVICES:-none}\"
echo \"  Run root: \${RUN_ROOT}\"
echo '═══════════════════════════════════════════════════════════════'

echo
echo '[Phase 1] Compiling benchmarks...'
TARGETS=(
  'fractal_g2_ossm_v3.sio:fg2v3'
  'brain_ossm_classifier.sio:brain_clf'
  'hssm_native_algebra.sio:native_alg'
  'multihead_unit_oct_benchmark.sio:mh_unit'
  'associativity_probe_benchmark.sio:assoc_probe'
)

for target in \"\${TARGETS[@]}\"; do
  src=\"\${target%%:*}\"
  name=\"\${target##*:}\"
  echo \"  Compiling \${src}...\"
  \"\${SOUC}\" compile \"\${SOUNIO_DIR}/examples/\${src}\" -o \"\${BUILD_DIR}/\${name}.elf\" 2>&1 | tail -1
done
echo '  All targets compiled.'

echo
echo '[Phase 2] Running Fractal-G2 v3 (10-seed probe + ListOps)...'
time \"\${BUILD_DIR}/fg2v3.elf\" > \"\${RESULTS_DIR}/fractal_g2_v3_results.txt\" 2>&1

echo
echo '[Phase 3] Running brain connectome classifier...'
time \"\${BUILD_DIR}/brain_clf.elf\" > \"\${RESULTS_DIR}/brain_classifier_results.txt\" 2>&1

echo
echo '[Phase 4] Running supporting benchmarks...'
time \"\${BUILD_DIR}/native_alg.elf\" > \"\${RESULTS_DIR}/native_algebra_results.txt\" 2>&1
time \"\${BUILD_DIR}/mh_unit.elf\" > \"\${RESULTS_DIR}/multihead_unit_results.txt\" 2>&1
time \"\${BUILD_DIR}/assoc_probe.elf\" > \"\${RESULTS_DIR}/assoc_probe_results.txt\" 2>&1

echo
echo '[Phase 5] Persisting results...'
copy_with_retry \"\${RESULTS_DIR}/fractal_g2_v3_results.txt\" \"\${ORANGEFS_DIR}/fractal_g2_v3_results.txt\"
copy_with_retry \"\${RESULTS_DIR}/brain_classifier_results.txt\" \"\${ORANGEFS_DIR}/brain_classifier_results.txt\"
copy_with_retry \"\${RESULTS_DIR}/native_algebra_results.txt\" \"\${ORANGEFS_DIR}/native_algebra_results.txt\"
copy_with_retry \"\${RESULTS_DIR}/multihead_unit_results.txt\" \"\${ORANGEFS_DIR}/multihead_unit_results.txt\"
copy_with_retry \"\${RESULTS_DIR}/assoc_probe_results.txt\" \"\${ORANGEFS_DIR}/assoc_probe_results.txt\"
copy_with_retry \"\${LOG_DIR}/job-\${SLURM_JOB_ID}.log\" \"\${ORANGEFS_DIR}/job-\${SLURM_JOB_ID}.log\"

echo
echo '═══════════════════════════════════════════════════════════════'
echo \"  Job complete — \$(date)\"
echo '═══════════════════════════════════════════════════════════════'
echo
echo 'Key results:'
grep -E 'PROBE SUMMARY|Gap|Overall|NonAssoc|AssocNorm' \"\${RESULTS_DIR}/fractal_g2_v3_results.txt\" 2>/dev/null || true
echo
echo 'Brain classifier:'
grep -E 'O-SSM|H-SSM|Accuracy' \"\${RESULTS_DIR}/brain_classifier_results.txt\" 2>/dev/null || true
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
echo "Submission summary:"
echo "  RUN_ID: ${RUN_ID}"
echo "  JobID: ${JOB_ID}"
echo "  Login pod: ${LOGIN_POD}"
echo "  Stage root: ${STAGE_ROOT}"
echo "  Stable results: ${ORANGEFS_RESULTS_DIR}"
echo "  To inspect: kubectl -n ${NS} exec ${LOGIN_TARGET} -- sacct -j ${JOB_ID} --format=JobID,JobName,Partition,Account,QOS,State,ExitCode,Start,End,NodeList"
