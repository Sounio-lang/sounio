#!/usr/bin/env bash
# Submit a K-AXI → PTX kernel for runtime admission on a real GPU.
#
# Pipeline:
#   1. Locally: kretikos kaxi-emit-ptx <pattern> [--epistemic] -o /tmp/<pattern>.ptx
#   2. Package PTX + nvidia_bare_driver_loader.c into a self-extracting sbatch
#   3. Worker: cc the loader, run `loader <pattern>.ptx kaxi_kernel admission`
#      → cuInit, cuModuleLoadData(PTX), cuModuleGetFunction(kaxi_kernel)
#   4. Result reported via Slurm job comment
#
# This proves: K-AXI assembly text → ptxas-clean PTX → real CUDA driver
# accepts the module and resolves the entry function on a live GPU.
#
# Usage:
#   slurm-jobs/kretikos/submit-kaxi-ptx-admission.sh <pattern> [--epistemic]
#
# Env (mirrors submit-kretikos-source.sh defaults):
#   NS, KUBECTL_BIN, LOGIN_POD_NAME, LOGIN_SELECTOR
#   SBATCH_PARTITION, SBATCH_ACCOUNT, SBATCH_QOS, SBATCH_NODELIST
#   JOB_MEM (4G), JOB_TIME (00:05:00)
#   WAIT_FOR_RESULT (1), WAIT_TIMEOUT_SECONDS (300)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS="${NS:-slurm-pilot}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
JOB_MEM="${JOB_MEM:-4G}"
JOB_TIME="${JOB_TIME:-00:05:00}"
WAIT_FOR_RESULT="${WAIT_FOR_RESULT:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-300}"

usage() {
  cat >&2 <<'EOF'
usage: slurm-jobs/kretikos/submit-kaxi-ptx-admission.sh <pattern> [--epistemic]

<pattern> is any kaxi-emit-ptx pattern: vec_add, vec_sub, vec_mul, vec_div,
  fma, source_*_f32, epistemic_elementwise_f32, epistemic_dual_output_f32,
  source_epistemic_dual_output_f32.
EOF
}

PATTERN="${1:-}"
EPISTEMIC=0
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epistemic) EPISTEMIC=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$PATTERN" ]]; then
  usage; exit 1
fi

if [[ ! -x "${ROOT_DIR}/bin/kretikos" ]]; then
  echo "missing ${ROOT_DIR}/bin/kretikos" >&2; exit 1
fi

RUN_ID="${RUN_ID:-kaxi-ptx-${PATTERN}-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
LOCAL_PTX="${LOCAL_PTX:-/tmp/${RUN_ID}.ptx}"
LOCAL_LOADER_C="${ROOT_DIR}/scripts/gpu/nvidia_bare_driver_loader.c"
LOCAL_TARBALL="/tmp/${RUN_ID}.tgz"
LOCAL_SBATCH="/tmp/${RUN_ID}.sbatch"

cleanup() { rm -f "${LOCAL_PTX}" "${LOCAL_TARBALL}" "${LOCAL_SBATCH}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[0/3] generating K-AXI → PTX locally (epistemic=${EPISTEMIC})"
EMIT_FLAGS=()
[[ "$EPISTEMIC" -eq 1 ]] && EMIT_FLAGS+=(--epistemic)
"${ROOT_DIR}/bin/kretikos" kaxi-emit-ptx "$PATTERN" "${EMIT_FLAGS[@]}" -o "${LOCAL_PTX}"

if [[ ! -s "${LOCAL_PTX}" ]]; then
  echo "error: PTX not generated" >&2; exit 1
fi

if [[ ! -f "${LOCAL_LOADER_C}" ]]; then
  echo "error: loader source missing: ${LOCAL_LOADER_C}" >&2; exit 1
fi

echo "[1/3] resolving login pod"
if [[ -n "${LOGIN_POD_NAME}" ]]; then
  LOGIN_POD="${LOGIN_POD_NAME}"
else
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods \
    -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi
if [[ -z "${LOGIN_POD}" ]]; then
  echo "could not resolve a live login pod in namespace ${NS}" >&2; exit 1
fi

if ! command -v cc >/dev/null 2>&1; then
  echo "local C compiler (cc) is required to prebuild the CUDA Driver API loader" >&2
  exit 1
fi

echo "[2/3] building loader locally + packaging payload"
PTX_NAME="kernel.ptx"
LOADER_NAME="loader"
STAGE_DIR="/tmp/${RUN_ID}.stage"
mkdir -p "${STAGE_DIR}"
cp "${LOCAL_PTX}" "${STAGE_DIR}/${PTX_NAME}"
cc -O2 "${LOCAL_LOADER_C}" -ldl -o "${STAGE_DIR}/${LOADER_NAME}"
tar -C "${STAGE_DIR}" -czf "${LOCAL_TARBALL}" "${PTX_NAME}" "${LOADER_NAME}"
rm -rf "${STAGE_DIR}"
PAYLOAD_B64="$(base64 -w 0 "${LOCAL_TARBALL}" 2>/dev/null || base64 "${LOCAL_TARBALL}" | tr -d '\n')"

cat > "${LOCAL_SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH -J kaxi-ptx-${PATTERN}
#SBATCH -p ${SBATCH_PARTITION}
#SBATCH -A ${SBATCH_ACCOUNT}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH -w ${SBATCH_NODELIST}
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail

LOCAL_ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
LOG_FILE="\${LOCAL_ROOT}/run.log"
PTX_NAME="${PTX_NAME}"
LOADER_NAME="${LOADER_NAME}"

mark() {
  local msg="\$1"
  if [[ -n "\${SLURM_JOB_ID:-}" ]]; then
    scontrol update "JobId=\${SLURM_JOB_ID}" "Comment=\${msg}" >/dev/null 2>&1 || true
  fi
}
fail() {
  local rc="\$1" line="\$2"
  set +e
  local tail_summary=""
  [[ -f "\${LOG_FILE}" ]] && tail_summary="\$(tail -8 "\${LOG_FILE}" | tr '\n' ';' | tr -cd '[:alnum:]_./:=,; -' | cut -c1-700)"
  mark "kaxi_ptx_admission=fail rc=\${rc} line=\${line} log=\${tail_summary}"
  exit "\${rc}"
}
trap 'fail "\$?" "\$LINENO"' ERR

mkdir -p "\${LOCAL_ROOT}"
mark "kaxi_ptx_admission=running phase=decode"
cat > "\${LOCAL_ROOT}/payload.tgz.b64" <<'PAYLOAD_EOF'
${PAYLOAD_B64}
PAYLOAD_EOF
base64 -d "\${LOCAL_ROOT}/payload.tgz.b64" > "\${LOCAL_ROOT}/payload.tgz"
tar -xzf "\${LOCAL_ROOT}/payload.tgz" -C "\${LOCAL_ROOT}"

cd "\${LOCAL_ROOT}"
chmod +x "\${LOADER_NAME}"
export PATH="/usr/local/cuda/bin:/usr/bin:/bin:/usr/sbin:/sbin:\${PATH:-}"

mark "kaxi_ptx_admission=running phase=launch"
{
  echo "host=\$(hostname)"
  echo "job_id=\${SLURM_JOB_ID:-unknown}"
  echo "cuda_visible_devices=\${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi -L || true
  ./\${LOADER_NAME} "\${PTX_NAME}" kaxi_kernel admission
} >>"\${LOG_FILE}" 2>&1

# Loader emits: sounio_nvidia_bare_runtime status=... reason=... stage=... cuda_result=...
status_line="\$(grep -E 'sounio_nvidia_bare_runtime status=' "\${LOG_FILE}" | tail -1 || true)"
status_line="\$(echo "\${status_line}" | tr -cd '[:alnum:]_./:=,; -' | cut -c1-700)"
mark "kaxi_ptx_admission \${status_line:-status=fail-no-status}"
EOF

echo "[3/3] submitting via login pod ${LOGIN_POD}"
"${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch" >/dev/null
JOB_ID="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch" | tr -d '\r\n' | awk '{print $NF}')"

if [[ -z "${JOB_ID}" || ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "submission failed: JOB_ID=${JOB_ID}" >&2
  exit 1
fi
echo "submitted job: ${JOB_ID}"

if [[ "${WAIT_FOR_RESULT}" != "1" ]]; then
  echo "(not waiting; query: scontrol show job ${JOB_ID})"
  exit 0
fi

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while [[ $(date +%s) -lt $deadline ]]; do
  state="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
    "scontrol show job ${JOB_ID} --oneliner 2>/dev/null | tr ' ' '\n' | grep -E '^JobState=' | head -1" \
    | tr -d '\r\n' || true)"
  echo "  job ${JOB_ID}: ${state}"
  case "${state}" in
    JobState=COMPLETED) break ;;
    JobState=FAILED|JobState=CANCELLED|JobState=TIMEOUT|JobState=NODE_FAIL|JobState=BOOT_FAIL)
      break ;;
  esac
  sleep 5
done

# Pull final comment
COMMENT="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "scontrol show job ${JOB_ID} --oneliner 2>/dev/null | tr ' ' '\n' | grep -E '^Comment=' | head -1" \
  | tr -d '\r\n' || true)"
echo "result: ${COMMENT}"

if [[ "${COMMENT}" == *"=pass "* || "${COMMENT}" == *"=pass" ]]; then
  exit 0
fi
exit 1
