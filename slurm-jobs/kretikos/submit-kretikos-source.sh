#!/usr/bin/env bash
set -euo pipefail

# Submit a real Sounio source file through the Kretikos GPU runtime lane.
#
# This script embeds a tiny payload into the sbatch file, runs entirely from
# the GPU worker's local scratch, and records the acceptance result in the Slurm
# job comment. It intentionally avoids using OrangeFS as a payload transport or
# binary-intermediate dependency. Optional result publication preserves a
# worker-local artifact directory and fetches it back through the Slurm worker
# pod, which is the most reliable path on clusters where compute nodes cannot
# write directly to shared storage.

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
KRETIKOS_HPC_PUBLISH_RESULTS="${KRETIKOS_HPC_PUBLISH_RESULTS:-1}"
KRETIKOS_HPC_FETCH_RESULTS="${KRETIKOS_HPC_FETCH_RESULTS:-1}"
KRETIKOS_HPC_CERTIFY_KAXI="${KRETIKOS_HPC_CERTIFY_KAXI:-1}"
KRETIKOS_HPC_LOCAL_ARTIFACT_DIR="${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR:-}"
KRETIKOS_HPC_WORKER_NODE="${KRETIKOS_HPC_WORKER_NODE:-${SBATCH_NODELIST#gpuorangefs-}}"
KRETIKOS_HPC_WORKER_POD_LABEL="${KRETIKOS_HPC_WORKER_POD_LABEL:-app.kubernetes.io/instance=slurm-pilot-worker-gpuorangefs}"
KRETIKOS_HPC_WORKER_TMP="${KRETIKOS_HPC_WORKER_TMP:-/tmp}"

usage() {
  cat >&2 <<'EOF'
usage: slurm-jobs/kretikos/submit-kretikos-source.sh <source.sio>

Environment:
  WAIT_FOR_RESULT=0|1          wait for Slurm completion (default: 1)
  KRETIKOS_VEC_N=<n>           vector runtime length, clamped by loader (default: 64)
  KRETIKOS_HPC_PUBLISH_RESULTS=0|1 preserve worker-local artifact directory (default: 1)
  KRETIKOS_HPC_FETCH_RESULTS=0|1   fetch published artifacts back locally (default: 1)
  KRETIKOS_HPC_CERTIFY_KAXI=0|1    emit runtime-backed K-AXI certificates when supported (default: 1)
  KRETIKOS_HPC_LOCAL_ARTIFACT_DIR=<dir> local result directory override
  KRETIKOS_HPC_WORKER_NODE=<node>      Kubernetes worker node (default: SBATCH_NODELIST without gpuorangefs-)
  KRETIKOS_HPC_WORKER_POD_LABEL=<label-selector> worker pod selector
  KRETIKOS_HPC_WORKER_TMP=<dir>        worker-local tmp root (default: /tmp)
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
  self-hosted/gpu/kretikos_emit_kaxi.sio \
  self-hosted/gpu/kretikos_kaxi_validate_evidence.sio \
  self-hosted/gpu/kretikos_json_emit.sio \
  self-hosted/gpu/kaxi_backend.sio \
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

LOCAL_ROOT="${KRETIKOS_HPC_WORKER_TMP}/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
SOUNIO_DIR="\${LOCAL_ROOT}/repo"
BUNDLE_DIR="\${LOCAL_ROOT}/bundle"
CERT_DIR="\${LOCAL_ROOT}/kaxi-certificate"
LOG_FILE="\${LOCAL_ROOT}/kretikos-source.log"
CERT_LOG="\${LOCAL_ROOT}/kretikos-kaxi-certificate.log"
PUBLISH_RESULTS="${KRETIKOS_HPC_PUBLISH_RESULTS}"
CERTIFY_KAXI="${KRETIKOS_HPC_CERTIFY_KAXI}"
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
mkdir -p "\${SOUNIO_DIR}" "\${BUNDLE_DIR}" "\${CERT_DIR}"
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

mapfile -t __cm_b < <(./bin/kretikos kaxi-validate-evidence "\${BUNDLE_DIR}/kretikos_bundle.v1.json" \\
    --print-or-empty runtime_validation.status \\
    --print-or-empty runtime_validation.reason \\
    --print-or-empty runtime_validation.stage \\
    --print-or-empty runtime_validation.cuda_result \\
    --print-or-empty runtime_validation.driver_version \\
    --print-or-empty runtime_validation.device_count \\
    --print-or-empty runtime_validation.cc_major \\
    --print-or-empty runtime_validation.cc_minor \\
    --print-or-empty runtime_validation.device_name \\
    --print-or-empty runtime_validation.rung \\
    --print-or-empty runtime_validation.kernel \\
    --print-or-empty toolchain_validation.ptxas.status \\
    --print-or-empty toolchain_validation.ptxas.reason \\
    --print-or-empty toolchain_validation.nvdisasm.status \\
    --print-or-empty toolchain_validation.nvdisasm.reason)
__cm_prof="\$(./bin/kretikos kaxi-validate-evidence "\${BUNDLE_DIR}/kretikos_source_profile.v1.json" --print-or-empty profile)"
comment="\$(printf 'kretikos_source=%s profile=%s reason=%s stage=%s cuda=%s driver=%s devices=%s cc=%s.%s device=%s rung=%s kernel=%s ptxas=%s/%s nvdisasm=%s/%s' \\
    "\${__cm_b[0]}" "\${__cm_prof}" "\${__cm_b[1]}" "\${__cm_b[2]}" "\${__cm_b[3]}" "\${__cm_b[4]}" "\${__cm_b[5]}" "\${__cm_b[6]}" "\${__cm_b[7]}" "\${__cm_b[8]}" "\${__cm_b[9]}" "\${__cm_b[10]}" "\${__cm_b[11]}" "\${__cm_b[12]}" "\${__cm_b[13]}" "\${__cm_b[14]}")"
mark "\${comment}"

KAXI_CERT_STATUS="not_run"
KAXI_CERT_REASON="profile_not_supported"
KAXI_CERT_WORKER_PATH=""
KAXI_CERT_PUBLISHED_PATH=""
SOURCE_PROFILE="\$(./bin/kretikos kaxi-validate-evidence "\${BUNDLE_DIR}/kretikos_source_profile.v1.json" --print-or-empty profile)"

case "\${SOURCE_PROFILE}" in
  vec_add_f32|vec_sub_f32|vec_mul_f32|vec_div_f32|fma_f32|epistemic_elementwise_f32|epistemic_dual_output_f32)
    if [[ "\${CERTIFY_KAXI}" == "1" ]]; then
      mark "kretikos_source=running phase=kaxi_certificate profile=\${SOURCE_PROFILE}"
      KAXI_CERT_JSON="\${CERT_DIR}/kaxi_certificate.v1.json"
      KAXI_CERT_KIND="profile"
      if ! grep -Eq '^[[:space:]]*//[[:space:]]*kretikos:[[:space:]]*profile[[:space:]]*=' "\${SOUNIO_DIR}/\${SOURCE_PAYLOAD_NAME}" \
        && ./bin/kretikos kaxi-lower-source "\${SOUNIO_DIR}/\${SOURCE_PAYLOAD_NAME}" \
          -o "\${CERT_DIR}/kaxi_lowering_probe.json" \
          --asm-output "\${CERT_DIR}/kaxi_lowering_probe.kaxi" \
          --kaxi-witness-output "\${CERT_DIR}/kaxi_lowering_probe.witness.json" >/dev/null 2>&1; then
        KAXI_CERT_KIND="source_lowering"
        ./bin/kretikos kaxi-lowering-certificate "\${SOUNIO_DIR}/\${SOURCE_PAYLOAD_NAME}" \
          -o "\${KAXI_CERT_JSON}" \
          --work-dir "\${CERT_DIR}/work" \
          --force \
          --require-runtime > "\${CERT_LOG}" 2>&1
      else
        ./bin/kretikos kaxi-certificate "\${SOUNIO_DIR}/\${SOURCE_PAYLOAD_NAME}" \
          -o "\${KAXI_CERT_JSON}" \
          --work-dir "\${CERT_DIR}/work" \
          --force \
          --require-runtime > "\${CERT_LOG}" 2>&1
      fi
      __kc_status="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_JSON}" --print-or-empty status)"
      __kc_profile="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_JSON}" --print-or-empty profile.name)"
      __kc_failures="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_JSON}" --print-subtree-or failures '[]')"
      __kc_rt_status="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_JSON}" --print-or-empty runtime.runtime_validation.status)"
      __kc_rt_reason="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_JSON}" --print-or-empty runtime.runtime_validation.reason)"
      __kc_rt_rung="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_JSON}" --print-or-empty runtime.runtime_validation.rung)"
      if [[ "\${__kc_status}" != "pass" ]]; then
        echo "kaxi_certificate_not_pass:\${__kc_status}" >&2; exit 1
      fi
      if [[ "\${__kc_profile}" != "\${SOURCE_PROFILE}" ]]; then
        echo "kaxi_certificate_profile_mismatch:\${__kc_profile}!=\${SOURCE_PROFILE}" >&2; exit 1
      fi
      if [[ "\${__kc_failures}" != "[]" ]]; then
        echo "kaxi_certificate_failures:\${__kc_failures}" >&2; exit 1
      fi
      if [[ "\${__kc_rt_status}" != "pass" ]]; then
        echo "kaxi_certificate_runtime_not_pass:\${__kc_rt_status}/\${__kc_rt_reason}" >&2; exit 1
      fi
      if [[ "\${__kc_rt_rung}" != "\${SOURCE_PROFILE}" ]]; then
        echo "kaxi_certificate_runtime_rung_mismatch:\${__kc_rt_rung}!=\${SOURCE_PROFILE}" >&2; exit 1
      fi
      KAXI_CERT_STATUS="pass"
      KAXI_CERT_REASON="runtime_backed_\${KAXI_CERT_KIND}_certificate_pass"
      KAXI_CERT_WORKER_PATH="\${KAXI_CERT_JSON}"
      KAXI_CERT_PUBLISHED_PATH="certificate/kaxi_certificate.v1.json"
      mark "\${comment} cert=pass cert_kind=\${KAXI_CERT_KIND}"
    else
      KAXI_CERT_STATUS="not_run"
      KAXI_CERT_REASON="certification_disabled"
      printf 'status=not_run reason=certification_disabled profile=%s\n' "\${SOURCE_PROFILE}" > "\${CERT_LOG}"
    fi
    ;;
  *)
    printf 'status=not_run reason=profile_not_supported profile=%s\n' "\${SOURCE_PROFILE}" > "\${CERT_LOG}"
    ;;
esac

if [[ "\${PUBLISH_RESULTS}" == "1" ]]; then
  PUBLISH_DIR="\${LOCAL_ROOT}/publish"
  mkdir -p "\${PUBLISH_DIR}/bundle" "\${PUBLISH_DIR}/certificate"
  tar -C "\${BUNDLE_DIR}" -cf - . | tar -C "\${PUBLISH_DIR}/bundle" -xf -
  if [[ -f "\${CERT_DIR}/kaxi_certificate.v1.json" ]]; then
    tar -C "\${CERT_DIR}" -cf - . | tar -C "\${PUBLISH_DIR}/certificate" -xf -
  fi
  cp "\${LOG_FILE}" "\${PUBLISH_DIR}/kretikos-source.log"
  cp "\${CERT_LOG}" "\${PUBLISH_DIR}/kretikos-kaxi-certificate.log"
  printf '%s\n' "\${comment}" > "\${PUBLISH_DIR}/comment.txt"
  printf '%s\n' "\${LOCAL_ROOT}" > "\${PUBLISH_DIR}/worker_run_dir.txt"
  printf '%s\n' "\$(hostname)" > "\${PUBLISH_DIR}/worker_host.txt"
  __agg_runtime_val="\$(./bin/kretikos kaxi-validate-evidence "\${BUNDLE_DIR}/kretikos_bundle.v1.json" --print-subtree-or runtime_validation '{}')"
  __agg_toolchain_val="\$(./bin/kretikos kaxi-validate-evidence "\${BUNDLE_DIR}/kretikos_bundle.v1.json" --print-subtree-or toolchain_validation '{}')"
  __agg_source_raw="\$(cat "\${BUNDLE_DIR}/kretikos_source_profile.v1.json")"
  __agg_bundle_raw="\$(cat "\${BUNDLE_DIR}/kretikos_bundle.v1.json")"
  if [[ -n "\${KAXI_CERT_WORKER_PATH}" && -f "\${KAXI_CERT_WORKER_PATH}" ]]; then
    __agg_cert_raw="\$(cat "\${KAXI_CERT_WORKER_PATH}")"
    __agg_cert_rt_val="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_WORKER_PATH}" --print-subtree-or runtime.runtime_validation null)"
    __agg_cert_profile="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_WORKER_PATH}" --print-or-empty profile.name)"
    __agg_cert_kaxi_pattern="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_WORKER_PATH}" --print-subtree-or kaxi.pattern null)"
    __agg_cert_kaxi_sempro="\$(./bin/kretikos kaxi-validate-evidence "\${KAXI_CERT_WORKER_PATH}" --print-subtree-or kaxi.semantic_profile null)"
  else
    __agg_cert_raw="null"
    __agg_cert_rt_val="null"
    __agg_cert_profile="\$(./bin/kretikos kaxi-validate-evidence "\${BUNDLE_DIR}/kretikos_source_profile.v1.json" --print-or-empty profile)"
    __agg_cert_kaxi_pattern="null"
    __agg_cert_kaxi_sempro="null"
  fi
  __agg_cert_args=(
    --raw-json "certificate=\${__agg_cert_raw}"
    --raw-json "kaxi_pattern=\${__agg_cert_kaxi_pattern}"
    --string  "log_path=\${CERT_LOG}"
    --string  "profile=\${__agg_cert_profile}"
  )
  if [[ -n "\${KAXI_CERT_PUBLISHED_PATH}" ]]; then
    __agg_cert_args+=(--string "published_path=\${KAXI_CERT_PUBLISHED_PATH}")
  else
    __agg_cert_args+=(--null published_path)
  fi
  __agg_cert_args+=(--string "reason=\${KAXI_CERT_REASON}")
  __agg_cert_args+=(--raw-json "runtime_validation=\${__agg_cert_rt_val}")
  __agg_cert_args+=(--raw-json "semantic_profile=\${__agg_cert_kaxi_sempro}")
  __agg_cert_args+=(--string "status=\${KAXI_CERT_STATUS}")
  if [[ -n "\${KAXI_CERT_WORKER_PATH}" ]]; then
    __agg_cert_args+=(--string "worker_path=\${KAXI_CERT_WORKER_PATH}")
  else
    __agg_cert_args+=(--null worker_path)
  fi
  __agg_kaxi_certificate="\$(./bin/kretikos json-emit "\${__agg_cert_args[@]}")"
  __agg_now="\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ./bin/kretikos json-emit \\
    --array-strings 'boundaries=payload_was_embedded_in_sbatch_not_read_from_orangefs|runtime_execution_happened_on_slurm_gpu_worker|published_results_are_post_run_evidence|kaxi_certificate_requires_worker_runtime_pass_for_supported_profiles|worker_side_ptxas_or_nvdisasm_missing_is_not_a_runtime_failure' \\
    --raw-json "bundle=\${__agg_bundle_raw}" \\
    --string "comment=\${comment}" \\
    --string "generated_at_utc=\${__agg_now}" \\
    --string "host=\$(hostname)" \\
    --string "job_id=\${SLURM_JOB_ID:-unknown}" \\
    --raw-json "kaxi_certificate=\${__agg_kaxi_certificate}" \\
    --raw-json "runtime_validation=\${__agg_runtime_val}" \\
    --string 'schema=sounio.kretikos.hpc-source-result.v1' \\
    --raw-json "source_profile=\${__agg_source_raw}" \\
    --raw-json "toolchain_validation=\${__agg_toolchain_val}" \\
    > "\${PUBLISH_DIR}/kretikos_hpc_source_result.v1.json"
fi
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
if [[ "${KRETIKOS_HPC_PUBLISH_RESULTS}" == "1" ]]; then
  echo "  worker-run-dir: ${KRETIKOS_HPC_WORKER_TMP}/${RUN_ID}-${JOB_ID}"
fi

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
echo

STATE="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sacct -j "${JOB_ID}" --format=State --noheader | head -n1 | xargs)"
if [[ "${STATE}" == "COMPLETED" && "${KRETIKOS_HPC_PUBLISH_RESULTS}" == "1" ]]; then
  WORKER_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods -l "${KRETIKOS_HPC_WORKER_POD_LABEL}" -o wide \
    | awk -v node="${KRETIKOS_HPC_WORKER_NODE}" '$7 == node { print $1; exit }')"
  if [[ -z "${WORKER_POD}" ]]; then
    echo "could not resolve worker pod for node ${KRETIKOS_HPC_WORKER_NODE}" >&2
    exit 1
  fi
  WORKER_RUN_DIR="${KRETIKOS_HPC_WORKER_TMP}/${RUN_ID}-${JOB_ID}"
  WORKER_PUBLISH_DIR="${WORKER_RUN_DIR}/publish"
  "${KUBECTL_BIN}" -n "${NS}" exec "${WORKER_POD}" -- test -d "${WORKER_PUBLISH_DIR}"
  if [[ "${KRETIKOS_HPC_FETCH_RESULTS}" == "1" ]]; then
    if [[ -z "${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR}" ]]; then
      KRETIKOS_HPC_LOCAL_ARTIFACT_DIR="/tmp/${RUN_ID}-artifacts"
    fi
    rm -rf "${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR}"
    mkdir -p "${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR}"
    "${KUBECTL_BIN}" -n "${NS}" exec "${WORKER_POD}" -- tar -C "${WORKER_PUBLISH_DIR}" -czf - . \
      | tar -xzf - -C "${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR}"
    echo "WorkerPod=${WORKER_POD}"
    echo "WorkerRunDir=${WORKER_RUN_DIR}"
    echo "Artifacts=${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR}"
    echo "ArtifactJSON=${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR}/kretikos_hpc_source_result.v1.json"
    if [[ -f "${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR}/kretikos_hpc_source_result.v1.json" ]]; then
      mapfile -t __ar_parts < <("${ROOT_DIR}/bin/kretikos" kaxi-validate-evidence \
        "${KRETIKOS_HPC_LOCAL_ARTIFACT_DIR}/kretikos_hpc_source_result.v1.json" \
        --print-or-empty runtime_validation.status \
        --print-or-empty runtime_validation.reason \
        --print-or-empty runtime_validation.rung \
        --print-or-empty runtime_validation.kernel \
        --print-or-empty runtime_validation.device_name)
      printf 'ArtifactRuntime=%s/%s rung=%s kernel=%s device=%s\n' \
        "${__ar_parts[0]}" "${__ar_parts[1]}" "${__ar_parts[2]}" "${__ar_parts[3]}" "${__ar_parts[4]}"
    fi
  else
    echo "ArtifactsRemote=${WORKER_POD}:${WORKER_PUBLISH_DIR}"
  fi
fi
[[ "${STATE}" == "COMPLETED" ]]
