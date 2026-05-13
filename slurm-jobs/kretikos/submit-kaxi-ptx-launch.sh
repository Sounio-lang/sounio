#!/usr/bin/env bash
# Submit a K-AXI → PTX kernel to a real GPU and verify output vs simulator.
#
# Pipeline:
#   1. Locally: kretikos kaxi-emit-ptx <pattern> [--epistemic] -o /tmp/kernel.ptx
#   2. Locally: build kaxi_ptx_runner (CUDA Driver API host launcher)
#   3. Locally: compute expected MEM via the K-AXI simulator (test_kaxi_ptx
#               harness output is structurally identical to what we'll run)
#   4. Package: PTX + runner binary + expected MEM/VAR
#   5. sbatch:  worker runs `kaxi_ptx_runner kernel.ptx --threads <N>`
#               compares MEM (and VAR if epistemic) line-by-line vs expected,
#               reports pass/fail via Slurm Comment
#
# Usage:
#   slurm-jobs/kretikos/submit-kaxi-ptx-launch.sh <pattern> [--epistemic]
#                                                 [--threads N]
#                                                 [--mem-words W]
#                                                 [--expected MEM-csv]
#                                                 [--expected-var VAR-csv]

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

PATTERN="${1:-}"
EPISTEMIC=0
THREADS=1
MEM_WORDS=16
EXPECTED=""
EXPECTED_VAR=""
INIT_MEM=""
INIT_VAR=""
INIT_FILE=""
INIT_VAR_FILE=""
BLOCKS=""
TYPE_FLAG=""
COHORT_SIZE=""   # Phase W: streamed clinical-scale cohort
N_STREAMS=""     # Phase W: # concurrent CUDA streams (>=1 enables streamed path)
N_CHUNKS=""      # Phase W: # chunks across the cohort
EXPECTED_DIGEST=""  # Phase W: expected mem_digest (16 hex chars) for cohort run
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epistemic) EPISTEMIC=1; shift ;;
    --threads) THREADS="$2"; shift 2 ;;
    --blocks) BLOCKS="$2"; shift 2 ;;
    --mem-words) MEM_WORDS="$2"; shift 2 ;;
    --expected) EXPECTED="$2"; shift 2 ;;
    --expected-var) EXPECTED_VAR="$2"; shift 2 ;;
    --init-mem) INIT_MEM="$2"; shift 2 ;;
    --init-var) INIT_VAR="$2"; shift 2 ;;
    --init-file) INIT_FILE="$2"; shift 2 ;;
    --init-var-file) INIT_VAR_FILE="$2"; shift 2 ;;
    --type) TYPE_FLAG="$2"; shift 2 ;;
    --cohort-size) COHORT_SIZE="$2"; shift 2 ;;
    --streams) N_STREAMS="$2"; shift 2 ;;
    --chunks) N_CHUNKS="$2"; shift 2 ;;
    --expected-digest) EXPECTED_DIGEST="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 <pattern> [--epistemic] [--threads N] [--blocks B] [--mem-words W]" >&2
      echo "       [--init-mem CSV] [--init-var CSV] [--init-file PATH] [--init-var-file PATH]" >&2
      echo "       [--expected MEM] [--expected-var VAR] [--type i64|f32|f64]" >&2
      echo "       [--cohort-size N] [--streams N] [--chunks K] [--expected-digest HEX]" >&2
      echo "  --init-file: raw binary init image (mem_words * elem bytes), used when CSV would exceed ARG_MAX" >&2
      echo "  --cohort-size: Phase W streamed multi-launch over an N-patient cohort" >&2
      exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PATTERN" ]]; then
  echo "usage: $0 <pattern> [--epistemic] ..." >&2; exit 1
fi
if [[ ! -x "${ROOT_DIR}/bin/kretikos" ]]; then
  echo "missing ${ROOT_DIR}/bin/kretikos" >&2; exit 1
fi
if ! command -v cc >/dev/null 2>&1; then
  echo "local cc required to prebuild runner" >&2; exit 1
fi

RUN_ID="${RUN_ID:-kaxi-ptx-launch-${PATTERN}-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
LOCAL_PTX="/tmp/${RUN_ID}.ptx"
LOCAL_TARBALL="/tmp/${RUN_ID}.tgz"
LOCAL_SBATCH="/tmp/${RUN_ID}.sbatch"
STAGE_DIR="/tmp/${RUN_ID}.stage"

cleanup() { rm -f "${LOCAL_PTX}" "${LOCAL_TARBALL}" "${LOCAL_SBATCH}"; rm -rf "${STAGE_DIR}"; }
trap cleanup EXIT

echo "[0/3] generating K-AXI → PTX (epistemic=${EPISTEMIC})"
EMIT_FLAGS=()
[[ "$EPISTEMIC" -eq 1 ]] && EMIT_FLAGS+=(--epistemic)
"${ROOT_DIR}/bin/kretikos" kaxi-emit-ptx "$PATTERN" "${EMIT_FLAGS[@]}" -o "${LOCAL_PTX}" >/dev/null

echo "[1/3] resolving login pod"
if [[ -n "${LOGIN_POD_NAME}" ]]; then
  LOGIN_POD="${LOGIN_POD_NAME}"
else
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi
[[ -z "${LOGIN_POD}" ]] && { echo "no live login pod" >&2; exit 1; }

echo "[2/3] building runner + packaging"
mkdir -p "${STAGE_DIR}"
cp "${LOCAL_PTX}" "${STAGE_DIR}/kernel.ptx"
cc -O2 "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c" -ldl -o "${STAGE_DIR}/runner"
[[ -n "${EXPECTED}" ]] && echo "${EXPECTED}" > "${STAGE_DIR}/expected.mem"
[[ -n "${EXPECTED_VAR}" ]] && echo "${EXPECTED_VAR}" > "${STAGE_DIR}/expected.var"
[[ -n "${EXPECTED_DIGEST}" ]] && echo "${EXPECTED_DIGEST}" > "${STAGE_DIR}/expected.digest"
if [[ -n "${INIT_FILE}" ]]; then
  [[ -f "${INIT_FILE}" ]] || { echo "error: --init-file ${INIT_FILE} not found" >&2; exit 1; }
  cp "${INIT_FILE}" "${STAGE_DIR}/init.mem.bin"
fi
if [[ -n "${INIT_VAR_FILE}" ]]; then
  [[ -f "${INIT_VAR_FILE}" ]] || { echo "error: --init-var-file ${INIT_VAR_FILE} not found" >&2; exit 1; }
  cp "${INIT_VAR_FILE}" "${STAGE_DIR}/init.var.bin"
fi
tar -C "${STAGE_DIR}" -czf "${LOCAL_TARBALL}" .
PAYLOAD_B64="$(base64 -w 0 "${LOCAL_TARBALL}" 2>/dev/null || base64 "${LOCAL_TARBALL}" | tr -d '\n')"

EPISTEMIC_FLAG=""
[[ "$EPISTEMIC" -eq 1 ]] && EPISTEMIC_FLAG="--epistemic"
INIT_MEM_FLAG=""
[[ -n "$INIT_MEM" ]] && INIT_MEM_FLAG="--init-mem ${INIT_MEM// /,}"
INIT_VAR_FLAG=""
[[ -n "$INIT_VAR" ]] && INIT_VAR_FLAG="--init-var ${INIT_VAR// /,}"
INIT_FILE_FLAG=""
[[ -n "$INIT_FILE" ]] && INIT_FILE_FLAG="--init-file init.mem.bin"
INIT_VAR_FILE_FLAG=""
[[ -n "$INIT_VAR_FILE" ]] && INIT_VAR_FILE_FLAG="--init-var-file init.var.bin"
BLOCKS_FLAG=""
[[ -n "$BLOCKS" ]] && BLOCKS_FLAG="--blocks ${BLOCKS}"
TYPE_FLAGSTR=""
[[ -n "$TYPE_FLAG" ]] && TYPE_FLAGSTR="--type ${TYPE_FLAG}"
COHORT_FLAG=""
[[ -n "$COHORT_SIZE" ]] && COHORT_FLAG="--cohort-size ${COHORT_SIZE}"
STREAMS_FLAG=""
[[ -n "$N_STREAMS" ]] && STREAMS_FLAG="--streams ${N_STREAMS}"
CHUNKS_FLAG=""
[[ -n "$N_CHUNKS" ]] && CHUNKS_FLAG="--chunks ${N_CHUNKS}"

cat > "${LOCAL_SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH -J ${RUN_ID}
#SBATCH -p ${SBATCH_PARTITION}
#SBATCH -A ${SBATCH_ACCOUNT}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 2
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH -w ${SBATCH_NODELIST}
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail
LOCAL_ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
LOG="\${LOCAL_ROOT}/run.log"
mark() { [[ -n "\${SLURM_JOB_ID:-}" ]] && scontrol update "JobId=\${SLURM_JOB_ID}" "Comment=\$1" >/dev/null 2>&1 || true; }
fail() {
  set +e; local rc=\$1 line=\$2
  local tail=""
  [[ -f "\${LOG}" ]] && tail="\$(tail -8 "\${LOG}" | tr '\n' ';' | tr -cd '[:alnum:]_./:=,; -' | cut -c1-700)"
  mark "kaxi_launch=fail rc=\${rc} line=\${line} log=\${tail}"
  exit "\${rc}"
}
trap 'fail "\$?" "\$LINENO"' ERR
mkdir -p "\${LOCAL_ROOT}"
mark "kaxi_launch=running phase=decode"
cat > "\${LOCAL_ROOT}/payload.tgz.b64" <<'PAYLOAD_EOF'
${PAYLOAD_B64}
PAYLOAD_EOF
base64 -d "\${LOCAL_ROOT}/payload.tgz.b64" > "\${LOCAL_ROOT}/payload.tgz"
tar -xzf "\${LOCAL_ROOT}/payload.tgz" -C "\${LOCAL_ROOT}"
chmod +x "\${LOCAL_ROOT}/runner"
cd "\${LOCAL_ROOT}"
export PATH="/usr/local/cuda/bin:/usr/bin:/bin:/usr/sbin:/sbin:\${PATH:-}"
mark "kaxi_launch=running phase=run"
{
  echo "host=\$(hostname)"; echo "job=\${SLURM_JOB_ID:-?}"
  ./runner kernel.ptx --threads ${THREADS} ${BLOCKS_FLAG} --mem-words ${MEM_WORDS} ${EPISTEMIC_FLAG} ${INIT_MEM_FLAG} ${INIT_VAR_FLAG} ${INIT_FILE_FLAG} ${INIT_VAR_FILE_FLAG} ${TYPE_FLAGSTR} ${COHORT_FLAG} ${STREAMS_FLAG} ${CHUNKS_FLAG}
} >"\${LOG}" 2>&1
status_line="\$(grep '^sounio_kaxi_runtime' "\${LOG}" | tail -1 || true)"
mem_line="\$(grep '^MEM:' "\${LOG}" | tail -1 || true)"
var_line="\$(grep '^VAR:' "\${LOG}" | tail -1 || true)"
phw_line="\$(grep '^PHW' "\${LOG}" | tail -1 || true)"
diff_mem="not_compared"
diff_var="not_compared"
diff_digest="not_compared"
if [[ -f expected.mem ]]; then
  exp="\$(cat expected.mem | tr -d '\r\n')"
  got="\$(echo "\${mem_line#MEM:}" | xargs | tr ' ' ',')"
  exp_norm="\$(echo "\${exp}" | xargs | tr ' ' ',')"
  if [[ "\${got}" == "\${exp_norm}" ]]; then diff_mem="mem_match"
  else diff_mem="mem_mismatch:got=\${got}|exp=\${exp_norm}"; fi
fi
if [[ -f expected.var ]]; then
  exp="\$(cat expected.var | tr -d '\r\n')"
  got="\$(echo "\${var_line#VAR:}" | xargs | tr ' ' ',')"
  exp_norm="\$(echo "\${exp}" | xargs | tr ' ' ',')"
  if [[ "\${got}" == "\${exp_norm}" ]]; then diff_var="var_match"
  else diff_var="var_mismatch:got=\${got}|exp=\${exp_norm}"; fi
fi
# Phase W: digest equivalence is the correctness anchor for streamed runs.
if [[ -f expected.digest ]]; then
  exp_dig="\$(tr -d '\r\n' < expected.digest)"
  got_dig="\$(echo "\${phw_line}" | sed -n 's/.*mem_digest=\([0-9a-f]*\).*/\1/p')"
  if [[ -n "\${got_dig}" && "\${got_dig}" == "\${exp_dig}" ]]; then diff_digest="digest_match=\${got_dig}"
  else diff_digest="digest_mismatch:got=\${got_dig}|exp=\${exp_dig}"; fi
fi
short="\$(echo "\${status_line} \${phw_line} diff_mem=\${diff_mem} diff_var=\${diff_var} diff_digest=\${diff_digest}" | tr -cd '[:alnum:]_./:=,; -' | cut -c1-700)"
mark "kaxi_launch \${short}"
EOF

echo "[3/3] submitting via login pod ${LOGIN_POD}"
"${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch" >/dev/null
JOB_ID="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch" | tr -d '\r\n' | awk '{print $NF}')"
[[ -z "${JOB_ID}" || ! "${JOB_ID}" =~ ^[0-9]+$ ]] && { echo "submit failed: ${JOB_ID}" >&2; exit 1; }
echo "submitted job: ${JOB_ID}"

[[ "${WAIT_FOR_RESULT}" != "1" ]] && exit 0

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while [[ $(date +%s) -lt $deadline ]]; do
  state="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
    "scontrol show job ${JOB_ID} --oneliner 2>/dev/null | tr ' ' '\n' | grep -E '^JobState=' | head -1" \
    | tr -d '\r\n' || true)"
  echo "  job ${JOB_ID}: ${state}"
  case "${state}" in
    JobState=COMPLETED|JobState=FAILED|JobState=CANCELLED|JobState=TIMEOUT|JobState=NODE_FAIL|JobState=BOOT_FAIL) break ;;
  esac
  sleep 5
done

# Full info
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "scontrol show job ${JOB_ID}" \
  | grep -E "JobState|ExitCode|Comment"
