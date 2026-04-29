#!/usr/bin/env bash
set -euo pipefail

# Submit from the control plane:
#   cd /home/devsounio/beagle/k8s/hpc-sota
#   source ops/lab-ops.sh
#   lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-fractal-g2-gpu.sh
#
# This wrapper intentionally avoids using OrangeFS as a payload transport.
# It compiles benchmark ELFs locally, embeds a compressed ELF bundle into the
# Slurm script, decodes it on worker-local /tmp, and only publishes final text
# artifacts to OrangeFS.

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/home/devsounio/sounio}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
LOGIN_TARGET="deploy/${LOGIN_DEPLOY_NAME}"
RUN_ID="${RUN_ID:-brain-ossm-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_ROOT="/orangefs/training/sounio/brain-ossm-runs/${RUN_ID}"
ORANGEFS_RESULTS_DIR="${ORANGEFS_RESULTS_DIR:-/orangefs/training/sounio/ossm-results}"
PUBLISH_ORANGEFS="${PUBLISH_ORANGEFS:-0}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-}"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
WORK_DIR="${WORK_DIR:-/tmp/${RUN_ID}-embedded}"
PAYLOAD_TGZ="${WORK_DIR}/fractal-g2-elf-payload.tgz"
PAYLOAD_B64="${WORK_DIR}/fractal-g2-elf-payload.b64"
LOCAL_SBATCH="${WORK_DIR}/${RUN_ID}.sbatch"

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

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/bin"

echo "Compiling benchmark ELFs locally for embedded worker payload"
TARGETS=(
  "fractal_g2_ossm_v3.sio:fg2v3"
  "brain_ossm_classifier.sio:brain_clf"
  "hssm_native_algebra.sio:native_alg"
  "multihead_unit_oct_benchmark.sio:mh_unit"
  "associativity_probe_benchmark.sio:assoc_probe"
)

for target in "${TARGETS[@]}"; do
  src="${target%%:*}"
  name="${target##*:}"
  echo "  compiling ${src}"
  "${SOUNIO_DIR}/bin/souc" compile "${SOUNIO_DIR}/examples/${src}" -o "${WORK_DIR}/bin/${name}.elf" >/dev/null
done

tar -C "${WORK_DIR}" -czf "${PAYLOAD_TGZ}" bin
tar -tzf "${PAYLOAD_TGZ}" >/dev/null
base64 -w 76 "${PAYLOAD_TGZ}" > "${PAYLOAD_B64}"

cat > "${LOCAL_SBATCH}" <<EOF
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

RUN_ROOT="/tmp/sounio-brain-ossm-runs/${RUN_ID}"
ORANGEFS_RUN_ROOT='${STAGE_ROOT}'
RESULTS_DIR="\${RUN_ROOT}/results"
ORANGEFS_DIR='${ORANGEFS_RESULTS_DIR}'
PUBLISH_ORANGEFS='${PUBLISH_ORANGEFS}'
LOG_DIR="\${RUN_ROOT}/logs"
BUILD_DIR="\$(mktemp -d /tmp/ossm-embedded.XXXXXX)"

cleanup() {
  rm -rf "\${BUILD_DIR}"
}
trap cleanup EXIT

publish_with_timeout() {
  local src="\$1"
  local dst="\$2"
  local attempt
  local tmp_dst="\${dst}.tmp"
  for attempt in \$(seq 1 3); do
    timeout 10s mkdir -p "\$(dirname "\$dst")" || true
    timeout 5s rm -f "\$dst" "\$tmp_dst" || true
    if timeout 20s cp -f "\$src" "\$tmp_dst" && [ -s "\$tmp_dst" ] && timeout 10s mv -f "\$tmp_dst" "\$dst"; then
      return 0
    fi
    sleep 1
  done
  echo "WARN: publication skipped after timeout: \$src -> \$dst" >&2
  return 0
}

mkdir -p "\${RESULTS_DIR}" "\${LOG_DIR}"
exec > >(tee "\${LOG_DIR}/job-\${SLURM_JOB_ID}.log") 2>&1

echo "Fractal-G2 O-SSM GPU Job"
echo "date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=\$(hostname)"
echo "cuda_visible_devices=\${CUDA_VISIBLE_DEVICES:-none}"
echo "run_root=\${RUN_ROOT}"
echo "orangefs_run_root=\${ORANGEFS_RUN_ROOT}"

cat > "\${BUILD_DIR}/payload.b64" <<'PAYLOAD_B64_EOF'
EOF

cat "${PAYLOAD_B64}" >> "${LOCAL_SBATCH}"

cat >> "${LOCAL_SBATCH}" <<'EOF'
PAYLOAD_B64_EOF

base64 -d "${BUILD_DIR}/payload.b64" > "${BUILD_DIR}/payload.tgz"
tar -xzf "${BUILD_DIR}/payload.tgz" -C "${BUILD_DIR}"
chmod +x "${BUILD_DIR}/bin/"*.elf

echo
echo "[Phase 1] Running Fractal-G2 v3"
time "${BUILD_DIR}/bin/fg2v3.elf" > "${RESULTS_DIR}/fractal_g2_v3_results.txt" 2>&1

echo
echo "[Phase 2] Running brain connectome classifier"
time "${BUILD_DIR}/bin/brain_clf.elf" > "${RESULTS_DIR}/brain_classifier_results.txt" 2>&1

echo
echo "[Phase 3] Running supporting benchmarks"
time "${BUILD_DIR}/bin/native_alg.elf" > "${RESULTS_DIR}/native_algebra_results.txt" 2>&1
time "${BUILD_DIR}/bin/mh_unit.elf" > "${RESULTS_DIR}/multihead_unit_results.txt" 2>&1
time "${BUILD_DIR}/bin/assoc_probe.elf" > "${RESULTS_DIR}/assoc_probe_results.txt" 2>&1

echo
echo "[Phase 4] Persisting results"
tar -C "${RUN_ROOT}" -czf "${RUN_ROOT}/result_bundle.tgz" results logs
if [[ "${PUBLISH_ORANGEFS}" == "1" ]]; then
  publish_with_timeout "${RUN_ROOT}/result_bundle.tgz" "${ORANGEFS_RUN_ROOT}/result_bundle.tgz"
  publish_with_timeout "${RESULTS_DIR}/fractal_g2_v3_results.txt" "${ORANGEFS_DIR}/fractal_g2_v3_results.txt"
  publish_with_timeout "${RESULTS_DIR}/brain_classifier_results.txt" "${ORANGEFS_DIR}/brain_classifier_results.txt"
  publish_with_timeout "${RESULTS_DIR}/native_algebra_results.txt" "${ORANGEFS_DIR}/native_algebra_results.txt"
  publish_with_timeout "${RESULTS_DIR}/multihead_unit_results.txt" "${ORANGEFS_DIR}/multihead_unit_results.txt"
  publish_with_timeout "${RESULTS_DIR}/assoc_probe_results.txt" "${ORANGEFS_DIR}/assoc_probe_results.txt"
  publish_with_timeout "${LOG_DIR}/job-${SLURM_JOB_ID}.log" "${ORANGEFS_DIR}/job-${SLURM_JOB_ID}.log"
else
  echo "OrangeFS publication skipped; worker-local bundle is authoritative for this gate"
fi

echo
echo "Job complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "worker_result_root=${RUN_ROOT}"
echo
echo "Key results:"
grep -E 'PROBE SUMMARY|Gap|Overall|NonAssoc|AssocNorm' "${RESULTS_DIR}/fractal_g2_v3_results.txt" 2>/dev/null || true
echo
echo "Brain classifier:"
grep -E 'O-SSM|H-SSM|Accuracy' "${RESULTS_DIR}/brain_classifier_results.txt" 2>/dev/null || true
EOF

echo "Submitting embedded ELF payload through ${LOGIN_POD}"
cat "${LOCAL_SBATCH}" | kubectl -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${SBATCH_FILE}'"

SBATCH_OUTPUT="$(
  kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -euo pipefail
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
echo "  Payload mode: embedded worker-local ELF bundle"
echo "  OrangeFS publish: ${PUBLISH_ORANGEFS}"
echo "  To inspect: kubectl -n ${NS} exec ${LOGIN_TARGET} -- sacct -j ${JOB_ID} --format=JobID,JobName,Partition,Account,QOS,State,ExitCode,Start,End,NodeList"
