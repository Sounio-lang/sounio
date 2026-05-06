#!/usr/bin/env bash
set -euo pipefail

# Submit a real Sounio source file through the Kretikos GPU runtime lane.
#
# This script embeds a tiny payload into the sbatch file, runs entirely from
# the GPU worker's local scratch, and records the acceptance result in the Slurm
# job comment. It intentionally avoids using OrangeFS for binary intermediates.

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
JOB_TIME="${JOB_TIME:-00:10:00}"
WAIT_FOR_RESULT="${WAIT_FOR_RESULT:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-300}"
KRETIKOS_VEC_N="${KRETIKOS_VEC_N:-64}"

usage() {
  cat >&2 <<'EOF'
usage: slurm-jobs/kretikos/submit-kretikos-source.sh <source.sio>

Environment:
  WAIT_FOR_RESULT=0|1          wait for Slurm completion (default: 1)
  KRETIKOS_VEC_N=<n>           vector runtime length, clamped by loader (default: 64)
  LOGIN_POD_NAME=<pod>         explicit login pod override
  SBATCH_NODELIST=<node>       GPU node override
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SOURCE_INPUT="$1"
if [[ ! -f "${SOURCE_INPUT}" ]]; then
  echo "source not found: ${SOURCE_INPUT}" >&2
  exit 1
fi
SOURCE_ABS="$(cd "$(dirname "${SOURCE_INPUT}")" && pwd)/$(basename "${SOURCE_INPUT}")"

if [[ ! -x "${ROOT_DIR}/bin/kretikos" ]]; then
  echo "missing ${ROOT_DIR}/bin/kretikos" >&2
  exit 1
fi

if ! command -v cc >/dev/null 2>&1; then
  echo "local C compiler is required to prebuild the CUDA Driver API loader" >&2
  exit 1
fi

echo "[0/5] profiling source"
"${ROOT_DIR}/bin/kretikos" profile-source "${SOURCE_ABS}"

if [[ -n "${LOGIN_POD_NAME}" ]]; then
  LOGIN_POD="${LOGIN_POD_NAME}"
else
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
  test -S /run/slurm/sack.socket
  scontrol ping >/dev/null
'

RUN_ID="${RUN_ID:-kretikos-source-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_SBATCH="${LOCAL_SBATCH:-/tmp/${RUN_ID}.sbatch}"
REMOTE_SBATCH="${REMOTE_SBATCH:-/tmp/${RUN_ID}.sbatch}"
LOCAL_LOADER="${LOCAL_LOADER:-/tmp/${RUN_ID}.kretikos_nvidia_bare_loader}"
SOURCE_PAYLOAD_NAME="${RUN_ID}.source.sio"
LOADER_PAYLOAD_NAME="$(basename "${LOCAL_LOADER}")"
SOURCE_PAYLOAD="/tmp/${SOURCE_PAYLOAD_NAME}"

cleanup() {
  rm -f "${SOURCE_PAYLOAD}" "${LOCAL_TARBALL}" "${LOCAL_SBATCH}" "${LOCAL_LOADER}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[1/5] building prebuilt CUDA Driver API loader"
cc -O2 "${ROOT_DIR}/scripts/gpu/nvidia_bare_driver_loader.c" -ldl -o "${LOCAL_LOADER}"
cp "${SOURCE_ABS}" "${SOURCE_PAYLOAD}"

echo "[2/5] building embedded Kretikos payload"
tar -C "${ROOT_DIR}" -czf "${LOCAL_TARBALL}" \
  bin/kretikos \
  bin/souc \
  bin/souc-linux-x86_64 \
  self-hosted/gpu/kretikos_emit_ptx.sio \
  self-hosted/gpu/kretikos_emit_cubin.sio \
  self-hosted/gpu/ptx.sio \
  self-hosted/gpu/nvidia_bare.sio \
  scripts/gpu/nvidia_bare_driver_loader.c \
  stdlib \
  -C /tmp "${LOADER_PAYLOAD_NAME}" "${SOURCE_PAYLOAD_NAME}"
tar -tzf "${LOCAL_TARBALL}" >/dev/null

PAYLOAD_B64="$(base64 -w 0 "${LOCAL_TARBALL}" 2>/dev/null || base64 "${LOCAL_TARBALL}" | tr -d '\n')"

cat > "${LOCAL_SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH -J kretikos-src
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

LOCAL_ROOT="\${TMPDIR:-/tmp}/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
SOUNIO_DIR="\${LOCAL_ROOT}/repo"
BUNDLE_DIR="\${LOCAL_ROOT}/bundle"
LOG_FILE="\${LOCAL_ROOT}/kretikos-source.log"
SOURCE_PAYLOAD_NAME="${SOURCE_PAYLOAD_NAME}"
LOADER_PAYLOAD_NAME="${LOADER_PAYLOAD_NAME}"
KRETIKOS_VEC_N="${KRETIKOS_VEC_N}"

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
  if [[ -f "\${LOG_FILE}" ]]; then
    tail_summary="\$(tail -10 "\${LOG_FILE}" | tr '\n' ';' | tr -cd '[:alnum:]_./:=,; -' | cut -c1-800)"
  fi
  mark "kretikos_source=fail phase=script rc=\${rc} line=\${line} log=\${tail_summary}"
  exit "\${rc}"
}
trap 'fail "\$?" "\$LINENO"' ERR

rm -rf "\${LOCAL_ROOT}"
mkdir -p "\${SOUNIO_DIR}" "\${BUNDLE_DIR}"
mark "kretikos_source=running phase=decode_payload"
cat > "\${LOCAL_ROOT}/payload.tgz.b64" <<'PAYLOAD_EOF'
${PAYLOAD_B64}
PAYLOAD_EOF
base64 -d "\${LOCAL_ROOT}/payload.tgz.b64" > "\${LOCAL_ROOT}/payload.tgz"
tar -xzf "\${LOCAL_ROOT}/payload.tgz" -C "\${SOUNIO_DIR}"
mv "\${SOUNIO_DIR}/\${LOADER_PAYLOAD_NAME}" "\${SOUNIO_DIR}/kretikos_nvidia_bare_loader"
chmod +x "\${SOUNIO_DIR}/bin/kretikos" "\${SOUNIO_DIR}/bin/souc" "\${SOUNIO_DIR}/bin/souc-linux-x86_64" "\${SOUNIO_DIR}/kretikos_nvidia_bare_loader"

cd "\${SOUNIO_DIR}"
export SOUNIO_STDLIB_PATH="\${SOUNIO_DIR}/stdlib"
export SOUNIO_KRETIKOS_RUNTIME_LOADER="\${SOUNIO_DIR}/kretikos_nvidia_bare_loader"
export SOUNIO_NVIDIA_BARE_VEC_N="\${KRETIKOS_VEC_N}"
export PATH="/usr/local/cuda/bin:/usr/bin:/bin:/usr/sbin:/sbin:\${PATH:-}"

mark "kretikos_source=running phase=run_source"
{
  echo "host=\$(hostname)"
  echo "job_id=\${SLURM_JOB_ID:-unknown}"
  echo "cuda_visible_devices=\${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi -L || true
  command -v ptxas || true
  command -v nvdisasm || true
  ./bin/kretikos run-source "\${SOUNIO_DIR}/\${SOURCE_PAYLOAD_NAME}" -o "\${BUNDLE_DIR}" --force --validate-toolchain --require-runtime
} > "\${LOG_FILE}" 2>&1

comment="\$(python3 - "\${BUNDLE_DIR}/kretikos_bundle.v1.json" "\${BUNDLE_DIR}/kretikos_source_profile.v1.json" <<'PY'
import json
import sys

bundle = json.load(open(sys.argv[1], encoding="utf-8"))
source = json.load(open(sys.argv[2], encoding="utf-8"))
runtime = bundle.get("runtime_validation") or {}
tool = bundle.get("toolchain_validation") or {}
ptxas = (tool.get("ptxas") or {})
nvdisasm = (tool.get("nvdisasm") or {})
print(
    "kretikos_source={status} profile={profile} reason={reason} stage={stage} "
    "cuda={cuda} driver={driver} devices={devices} cc={cc_major}.{cc_minor} "
    "device={device} rung={rung} kernel={kernel} ptxas={ptxas}/{ptxas_reason} "
    "nvdisasm={nvdisasm}/{nvdisasm_reason}".format(
        status=runtime.get("status"),
        profile=source.get("profile"),
        reason=runtime.get("reason"),
        stage=runtime.get("stage"),
        cuda=runtime.get("cuda_result"),
        driver=runtime.get("driver_version"),
        devices=runtime.get("device_count"),
        cc_major=runtime.get("cc_major"),
        cc_minor=runtime.get("cc_minor"),
        device=runtime.get("device_name"),
        rung=runtime.get("rung"),
        kernel=runtime.get("kernel"),
        ptxas=ptxas.get("status"),
        ptxas_reason=ptxas.get("reason"),
        nvdisasm=nvdisasm.get("status"),
        nvdisasm_reason=nvdisasm.get("reason"),
    )
)
PY
)"
mark "\${comment}"
EOF

echo "[3/5] uploading sbatch to login pod"
"${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_SBATCH}" "${LOGIN_POD}:${REMOTE_SBATCH}" >/dev/null

echo "[4/5] submitting to Slurm"
SBATCH_TEXT="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'; rm -f '${REMOTE_SBATCH}'")"
echo "${SBATCH_TEXT}"
JOB_ID="$(printf '%s\n' "${SBATCH_TEXT}" | awk '/Submitted batch job/ {print $4; exit}')"
if [[ -z "${JOB_ID}" ]]; then
  echo "failed to parse job id from sbatch output" >&2
  exit 1
fi

echo "[5/5] submitted job ${JOB_ID}"
echo "  source: ${SOURCE_ABS}"
echo "  node:   ${SBATCH_NODELIST}"

if [[ "${WAIT_FOR_RESULT}" != "1" ]]; then
  exit 0
fi

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  STATE="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sacct -j "${JOB_ID}" --format=State --noheader | head -n1 | xargs)"
  case "${STATE}" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|"CANCELLED by "*)
      break
      ;;
  esac
  sleep 5
done

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sacct -j "${JOB_ID}" --format=JobID,State,ExitCode,Elapsed,NodeList%30 --noheader
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "scontrol show job '${JOB_ID}' | tr '\n' ' ' | sed 's/  */ /g' | sed -n 's/.*Comment=\\([^ ]*.*\\) StdErr=.*/Comment=\\1/p'"

STATE="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sacct -j "${JOB_ID}" --format=State --noheader | head -n1 | xargs)"
[[ "${STATE}" == "COMPLETED" ]]
