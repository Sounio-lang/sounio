#!/usr/bin/env bash
set -euo pipefail

# Submit from the control plane:
#   cd /home/devsounio/beagle/k8s/hpc-sota
#   source ops/lab-ops.sh
#   lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/home/devsounio/sounio}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
RUN_ID="${RUN_ID:-brain-ossm-abide-campaign-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/abide-campaign-runs/${RUN_ID}"
ORANGEFS_RESULTS_DIR="${ORANGEFS_RESULTS_DIR:-/orangefs/training/sounio/abide-campaign}"
ABIDE_MANIFEST_PATH="${ABIDE_MANIFEST_PATH:-/orangefs/training/sounio/abide-data/abide_roi_manifest.tsv}"
LOCAL_SNAPSHOT_DIR="${LOCAL_SNAPSHOT_DIR:-/tmp/${RUN_ID}-snapshot}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_TARBALL_TMP="${LOCAL_TARBALL}.tmp"
LOCAL_SBATCH_FILE="${LOCAL_SBATCH_FILE:-/tmp/${RUN_ID}.sbatch.local}"
FORCE_RESTAGE="${FORCE_RESTAGE:-0}"
PERSIST_MODE="${PERSIST_MODE:-worker_local}"
PAYLOAD_TRANSFER_MODE="${PAYLOAD_TRANSFER_MODE:-embedded}"
BASELINE_MODELS="${BASELINE_MODELS:-lstm,gru,transformer,tcn}"
SEED_COUNT="${SEED_COUNT:-20}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
DROP_CHANNEL_FRAC="${DROP_CHANNEL_FRAC:-0.0}"
NOISE_STD="${NOISE_STD:-0.0}"
GLOBAL_TRAIN_EPOCHS="${GLOBAL_TRAIN_EPOCHS:-}"
GLOBAL_TRAIN_LR="${GLOBAL_TRAIN_LR:-}"
GLOBAL_CORE_LR_SCALE="${GLOBAL_CORE_LR_SCALE:-}"
OCT_TRAIN_EPOCHS="${OCT_TRAIN_EPOCHS:-}"
OCT_TRAIN_LR="${OCT_TRAIN_LR:-}"
OCT_CORE_LR_SCALE="${OCT_CORE_LR_SCALE:-}"
OCT_WARMUP_EPOCHS="${OCT_WARMUP_EPOCHS:-}"
OCT_TRAIN_PROFILE="${OCT_TRAIN_PROFILE:-o_default}"
OCT_TRAIN_MODE="${OCT_TRAIN_MODE:-}"
OCT_INIT_PRESET="${OCT_INIT_PRESET:-}"
OCT_ASSOC_REG="${OCT_ASSOC_REG:-}"
OCT_ASSOC_TARGET="${OCT_ASSOC_TARGET:-}"
OCT_ASSOC_SCHEDULE_MODE="${OCT_ASSOC_SCHEDULE_MODE:-}"
OCT_ASSOC_SCHEDULE_START="${OCT_ASSOC_SCHEDULE_START:-}"
OCT_ASSOC_SCHEDULE_MID="${OCT_ASSOC_SCHEDULE_MID:-}"
OCT_ASSOC_SCHEDULE_END="${OCT_ASSOC_SCHEDULE_END:-}"
OCT_ASSOC_DRIFT_REG="${OCT_ASSOC_DRIFT_REG:-}"
OCT_HEAD_TRUST_REGION="${OCT_HEAD_TRUST_REGION:-}"
OCT_HEAD_RENORM="${OCT_HEAD_RENORM:-}"
OCT_INPUT_PROJ_MODE="${OCT_INPUT_PROJ_MODE:-identity}"
OCT_PROJ_LR_SCALE="${OCT_PROJ_LR_SCALE:-0.6}"
OCT_PROJ_STRUCTURED_SCALE="${OCT_PROJ_STRUCTURED_SCALE:-0.35}"
OCT_PROJ_DELTA_SCALE="${OCT_PROJ_DELTA_SCALE:-0.18}"
OCT_PROJ_HYBRID_SCALE="${OCT_PROJ_HYBRID_SCALE:-0.12}"
H_TRAIN_EPOCHS="${H_TRAIN_EPOCHS:-}"
H_TRAIN_LR="${H_TRAIN_LR:-}"
H_CORE_LR_SCALE="${H_CORE_LR_SCALE:-}"
H_WARMUP_EPOCHS="${H_WARMUP_EPOCHS:-}"
H_INIT_PRESET="${H_INIT_PRESET:-}"
H_INPUT_PROJ_MODE="${H_INPUT_PROJ_MODE:-identity}"
H_PROJ_LR_SCALE="${H_PROJ_LR_SCALE:-0.6}"
MAX_SITES="${MAX_SITES:-}"
LIMIT_SUBJECTS="${LIMIT_SUBJECTS:-}"
JOB_MEM="${JOB_MEM:-8G}"
JOB_TIME="${JOB_TIME:-01:30:00}"
PYTORCH_VENV_DIR="${PYTORCH_VENV_DIR:-/tmp/${RUN_ID}-venvs/brain-ossm-external-baselines}"
PYTORCH_USERBASE_DIR="${PYTORCH_USERBASE_DIR:-/tmp/${RUN_ID}-python-userbase/brain-ossm-external-baselines}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"
STATE_DIR="${STATE_DIR:-/home/devsounio/beagle/k8s/hpc-sota/state}"
STATE_FILE="${STATE_FILE:-${STATE_DIR}/abide_campaign_last_submit.env}"

RUN_CONFIG_APPEND_TEXT=""
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
SBATCH_NODE_DIRECTIVES=""
if [[ -n "${SBATCH_NODELIST}" ]]; then
  SBATCH_NODE_DIRECTIVES+="#SBATCH -w ${SBATCH_NODELIST}"$'\n'
fi
if [[ -n "${SBATCH_EXCLUDE}" ]]; then
  SBATCH_NODE_DIRECTIVES+="#SBATCH --exclude=${SBATCH_EXCLUDE}"$'\n'
fi

resolve_proj_mode() {
  case "$1" in
    ""|identity) echo 0 ;;
    learned_linear) echo 1 ;;
    residual_linear) echo 2 ;;
    mandelbrot_d2_residual) echo 3 ;;
    mandelbrot_d2_hybrid) echo 4 ;;
    *)
      echo "unsupported projection mode: $1" >&2
      exit 1
      ;;
  esac
}

OCT_PROFILE_ID=0
case "${OCT_TRAIN_PROFILE}" in
  ""|o_default)
    OCT_PROFILE_ID=0
    ;;
  o_g2_transfer)
    OCT_PROFILE_ID=1
    : "${OCT_INIT_PRESET:=1}"
    ;;
  o_assoc_stable)
    OCT_PROFILE_ID=2
    : "${OCT_TRAIN_MODE:=2}"
    : "${OCT_INIT_PRESET:=1}"
    : "${OCT_ASSOC_REG:=0.05}"
    : "${OCT_ASSOC_TARGET:=0.85}"
    ;;
  o_alg_v1)
    OCT_PROFILE_ID=3
    : "${OCT_TRAIN_MODE:=3}"
    : "${OCT_INIT_PRESET:=1}"
    : "${OCT_ASSOC_REG:=0.04}"
    : "${OCT_ASSOC_TARGET:=0.86}"
    : "${OCT_ASSOC_SCHEDULE_MODE:=1}"
    : "${OCT_ASSOC_SCHEDULE_START:=1.10}"
    : "${OCT_ASSOC_SCHEDULE_MID:=0.86}"
    : "${OCT_ASSOC_SCHEDULE_END:=0.74}"
    : "${OCT_ASSOC_DRIFT_REG:=0.03}"
    : "${OCT_HEAD_TRUST_REGION:=0.05}"
    : "${OCT_HEAD_RENORM:=1}"
    ;;
  o_alg_v1_mandel)
    OCT_PROFILE_ID=4
    : "${OCT_TRAIN_MODE:=4}"
    : "${OCT_INIT_PRESET:=1}"
    : "${OCT_ASSOC_REG:=0.04}"
    : "${OCT_ASSOC_TARGET:=0.86}"
    : "${OCT_ASSOC_SCHEDULE_MODE:=1}"
    : "${OCT_ASSOC_SCHEDULE_START:=1.10}"
    : "${OCT_ASSOC_SCHEDULE_MID:=0.86}"
    : "${OCT_ASSOC_SCHEDULE_END:=0.74}"
    : "${OCT_ASSOC_DRIFT_REG:=0.03}"
    : "${OCT_HEAD_TRUST_REGION:=0.05}"
    : "${OCT_HEAD_RENORM:=1}"
    : "${OCT_INPUT_PROJ_MODE:=mandelbrot_d2_residual}"
    : "${OCT_PROJ_STRUCTURED_SCALE:=0.35}"
    : "${OCT_PROJ_DELTA_SCALE:=0.18}"
    : "${OCT_PROJ_HYBRID_SCALE:=0.12}"
    ;;
  *)
    echo "unsupported OCT_TRAIN_PROFILE=${OCT_TRAIN_PROFILE}; expected o_default|o_g2_transfer|o_assoc_stable|o_alg_v1|o_alg_v1_mandel" >&2
    exit 1
    ;;
esac

: "${OCT_TRAIN_MODE:=0}"
: "${OCT_ASSOC_SCHEDULE_MODE:=0}"
: "${OCT_ASSOC_SCHEDULE_START:=0.0}"
: "${OCT_ASSOC_SCHEDULE_MID:=0.0}"
: "${OCT_ASSOC_SCHEDULE_END:=0.0}"
: "${OCT_ASSOC_DRIFT_REG:=0.0}"
: "${OCT_HEAD_TRUST_REGION:=0.0}"
: "${OCT_HEAD_RENORM:=1}"

OCT_INPUT_PROJ_MODE_ID="$(resolve_proj_mode "${OCT_INPUT_PROJ_MODE}")"
H_INPUT_PROJ_MODE_ID="$(resolve_proj_mode "${H_INPUT_PROJ_MODE}")"

append_run_config_line() {
  local key="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    RUN_CONFIG_APPEND_TEXT+="${key}=${value}"$'\n'
  fi
}
append_run_config_line global_train_epochs "${GLOBAL_TRAIN_EPOCHS}"
append_run_config_line global_train_lr "${GLOBAL_TRAIN_LR}"
append_run_config_line global_core_lr_scale "${GLOBAL_CORE_LR_SCALE}"
append_run_config_line oct_profile_id "${OCT_PROFILE_ID}"
append_run_config_line oct_train_mode "${OCT_TRAIN_MODE}"
append_run_config_line oct_train_epochs "${OCT_TRAIN_EPOCHS}"
append_run_config_line oct_train_lr "${OCT_TRAIN_LR}"
append_run_config_line oct_core_lr_scale "${OCT_CORE_LR_SCALE}"
append_run_config_line oct_warmup_epochs "${OCT_WARMUP_EPOCHS}"
append_run_config_line oct_init_preset "${OCT_INIT_PRESET}"
append_run_config_line oct_assoc_reg "${OCT_ASSOC_REG}"
append_run_config_line oct_assoc_target "${OCT_ASSOC_TARGET}"
append_run_config_line oct_assoc_schedule_mode "${OCT_ASSOC_SCHEDULE_MODE}"
append_run_config_line oct_assoc_schedule_start "${OCT_ASSOC_SCHEDULE_START}"
append_run_config_line oct_assoc_schedule_mid "${OCT_ASSOC_SCHEDULE_MID}"
append_run_config_line oct_assoc_schedule_end "${OCT_ASSOC_SCHEDULE_END}"
append_run_config_line oct_assoc_drift_reg "${OCT_ASSOC_DRIFT_REG}"
append_run_config_line oct_head_trust_region "${OCT_HEAD_TRUST_REGION}"
append_run_config_line oct_head_renorm "${OCT_HEAD_RENORM}"
append_run_config_line oct_input_proj_mode "${OCT_INPUT_PROJ_MODE_ID}"
append_run_config_line oct_proj_lr_scale "${OCT_PROJ_LR_SCALE}"
append_run_config_line oct_proj_structured_scale "${OCT_PROJ_STRUCTURED_SCALE}"
append_run_config_line oct_proj_delta_scale "${OCT_PROJ_DELTA_SCALE}"
append_run_config_line oct_proj_hybrid_scale "${OCT_PROJ_HYBRID_SCALE}"
append_run_config_line h_train_epochs "${H_TRAIN_EPOCHS}"
append_run_config_line h_train_lr "${H_TRAIN_LR}"
append_run_config_line h_core_lr_scale "${H_CORE_LR_SCALE}"
append_run_config_line h_warmup_epochs "${H_WARMUP_EPOCHS}"
append_run_config_line h_init_preset "${H_INIT_PRESET}"
append_run_config_line h_input_proj_mode "${H_INPUT_PROJ_MODE_ID}"
append_run_config_line h_proj_lr_scale "${H_PROJ_LR_SCALE}"

REQUIRED_FILES=(
  "bin/souc"
  "artifacts/self-hosted/souc-self-hosted-x86_64"
  "examples/brain_ossm_abide.sio"
  "scripts/research/abide_campaign_lib.py"
  "scripts/research/build_abide_temporal_manifest.py"
  "scripts/research/normalize_abide_manifest.py"
  "scripts/research/parse_brain_ossm_abide_output.py"
  "scripts/research/abide_external_baselines.py"
  "scripts/research/aggregate_brain_ossm_campaign.py"
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

if [[ ! -s "${LOCAL_TARBALL}" || "${FORCE_RESTAGE}" != "0" ]]; then
  rm -rf "${LOCAL_SNAPSHOT_DIR}"
  rm -f "${LOCAL_TARBALL}" "${LOCAL_TARBALL_TMP}"
  OUT_ROOT="${LOCAL_SNAPSHOT_DIR}" SOUNIO_DIR="${SOUNIO_DIR}" ABIDE_MANIFEST_PATH="${ABIDE_MANIFEST_PATH}" \
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

case "${PAYLOAD_TRANSFER_MODE}" in
  embedded|sbcast)
    ;;
  *)
    echo "unsupported PAYLOAD_TRANSFER_MODE=${PAYLOAD_TRANSFER_MODE}; expected embedded|sbcast" >&2
    exit 1
    ;;
esac

LOCAL_PAYLOAD_MODE="${PAYLOAD_TRANSFER_MODE}"

rm -f "${LOCAL_SBATCH_FILE}"
cat > "${LOCAL_SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH -J brain-ossm-abide-campaign
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=gpuorangefs
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
${SBATCH_NODE_DIRECTIVES}
set -euo pipefail

ORANGEFS_RUN_ROOT='${STAGE_ROOT}'
PERSIST_MODE='${PERSIST_MODE}'
LOCAL_REPO_DIR="/tmp/${RUN_ID}-repo"
LOCAL_PAYLOAD_TGZ="/tmp/${RUN_ID}-payload.tgz"
LOCAL_SBCAST_PAYLOAD_TGZ="/tmp/${RUN_ID}-payload.sbcast.tgz"
USE_EMBEDDED_PAYLOAD='${LOCAL_PAYLOAD_MODE}'
SOUNIO_DIR=\"\"
SOUC=\"\"
BUILD_DIR="/tmp/${RUN_ID}-build"
LOCAL_RUN_ROOT="/tmp/${RUN_ID}-run"
RESULTS_DIR=\"\${LOCAL_RUN_ROOT}/results\"
ORANGEFS_DIR='${ORANGEFS_RESULTS_DIR}'
LOG_DIR=\"\${LOCAL_RUN_ROOT}/logs\"
MANIFEST_PATH='${ABIDE_MANIFEST_PATH}'
RUN_MANIFEST_PATH=""
LOCAL_VENV_DIR="/tmp/${RUN_ID}-venv"
LOCAL_PY_SITE_DIR="/tmp/${RUN_ID}-pysite"

cleanup() {
  rm -f \"\${LOCAL_PAYLOAD_TGZ}\"
  rm -f \"\${LOCAL_SBCAST_PAYLOAD_TGZ}\"
  rm -rf \"\${LOCAL_VENV_DIR}\"
  rm -rf \"\${LOCAL_PY_SITE_DIR}\"
  rm -rf \"\${LOCAL_REPO_DIR}\"
  rm -rf \"\${BUILD_DIR}\"
  if [[ "\${PERSIST_MODE:-orangefs}" = "orangefs" && "\${KEEP_LOCAL_DEBUG:-0}" != "1" ]]; then
    rm -rf \"\${LOCAL_RUN_ROOT}\"
  fi
}
finalize() {
  local status=\$?
  if [[ "\${status}" -ne 0 || "\${KEEP_LOCAL_DEBUG:-0}" = "1" ]]; then
    if [[ -n "\${JOB_LOG_PATH:-}" && -f "\${JOB_LOG_PATH:-}" ]]; then
      cp -f "\${JOB_LOG_PATH}" "/tmp/abide-campaign-debug-job-\${SLURM_JOB_ID:-manual}.log" || true
    fi
    echo "preserving local run root for debug: \${LOCAL_RUN_ROOT}" >&2 || true
  else
    cleanup
  fi
  exit "\${status}"
}
trap finalize EXIT

copy_with_retry() {
  local src=\"\$1\"
  local dst=\"\$2\"
  local verify_mode=\"\${3:-checksum}\"
  local attempt
  local tmp_dst=\"\${dst}.tmp\"
  local src_size
  local src_sha
  src_size=\$(wc -c < \"\$src\" 2>/dev/null || echo 0)
  src_sha=\$(sha256sum \"\$src\" 2>/dev/null | awk '{print \$1}')
  for attempt in \$(seq 1 20); do
    timeout 10s mkdir -p \"\$(dirname \"\$dst\")\" >/dev/null 2>&1 || true
    timeout 10s rm -f \"\$dst\" \"\$tmp_dst\" >/dev/null 2>&1 || true
    if timeout 30s bash -lc 'cat < \"\$1\" > \"\$2\"' _ \"\$src\" \"\$tmp_dst\" >/dev/null 2>&1 \
      && [ -s \"\$tmp_dst\" ]; then
      local tmp_size
      local tmp_sha
      tmp_size=\$(wc -c < \"\$tmp_dst\" 2>/dev/null || echo 0)
      tmp_sha=\$(sha256sum \"\$tmp_dst\" 2>/dev/null | awk '{print \$1}')
      if [[ \"\$tmp_size\" = \"\$src_size\" && \"\$tmp_sha\" = \"\$src_sha\" ]] \
        && timeout 10s mv -f \"\$tmp_dst\" \"\$dst\" >/dev/null 2>&1; then
        timeout 5s sync >/dev/null 2>&1 || true
        local dst_size
        local dst_sha
        dst_size=\$(wc -c < \"\$dst\" 2>/dev/null || echo 0)
        dst_sha=\$(sha256sum \"\$dst\" 2>/dev/null | awk '{print \$1}')
        if [[ \"\$dst_size\" = \"\$src_size\" && \"\$dst_sha\" = \"\$src_sha\" ]]; then
          if [[ \"\$verify_mode\" = \"tgz\" ]]; then
            if tar -tzf \"\$dst\" >/dev/null 2>&1; then
              printf '%s  %s\n' \"\$src_sha\" \"\$(basename \"\$dst\")\" > \"\${dst}.sha256\"
              return 0
            fi
          else
            printf '%s  %s\n' \"\$src_sha\" \"\$(basename \"\$dst\")\" > \"\${dst}.sha256\"
            return 0
          fi
        fi
      fi
    fi
    sleep 1
  done
  echo \"failed to copy \$src -> \$dst\" >&2
  return 1
}

ensure_dir_with_retry() {
  local dir=\"\$1\"
  local attempt
  for attempt in \$(seq 1 20); do
    if timeout 10s mkdir -p \"\$dir\" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo \"failed to create directory \$dir\" >&2
  return 1
}

rm -rf \"\${LOCAL_REPO_DIR}\" \"\${BUILD_DIR}\" \"\${LOCAL_RUN_ROOT}\" \"\${LOCAL_VENV_DIR}\" \"\${LOCAL_PY_SITE_DIR}\"
mkdir -p \"\${LOCAL_REPO_DIR}\" \"\${BUILD_DIR}\" \"\${LOCAL_RUN_ROOT}\" \"\${RESULTS_DIR}\" \"\${LOG_DIR}\" \"\${LOCAL_VENV_DIR}\" \"\${LOCAL_PY_SITE_DIR}\"
JOB_LOG_PATH=\"\${LOG_DIR}/job-\${SLURM_JOB_ID:-manual}.log\"
: > \"\${JOB_LOG_PATH}\"
exec >> \"\${JOB_LOG_PATH}\" 2>&1

if [[ \"\${USE_EMBEDDED_PAYLOAD}\" != \"embedded\" && \"\${USE_EMBEDDED_PAYLOAD}\" != \"sbcast\" ]]; then
  echo 'unsupported payload transfer mode' >&2
  exit 1
fi
EOF

cat >> "${LOCAL_SBATCH_FILE}" <<'EOF'
base64 -d > "${LOCAL_PAYLOAD_TGZ}" <<'PAYLOAD_EOF'
EOF
base64 -w0 "${LOCAL_TARBALL}" >> "${LOCAL_SBATCH_FILE}"
printf '\nPAYLOAD_EOF\n' >> "${LOCAL_SBATCH_FILE}"

cat >> "${LOCAL_SBATCH_FILE}" <<EOF
PAYLOAD_SOURCE_TGZ=\"\${LOCAL_PAYLOAD_TGZ}\"
if [[ \"\${USE_EMBEDDED_PAYLOAD}\" = \"sbcast\" ]]; then
  command -v sbcast >/dev/null
  rm -f \"\${LOCAL_SBCAST_PAYLOAD_TGZ}\"
  sbcast -f \"\${LOCAL_PAYLOAD_TGZ}\" \"\${LOCAL_SBCAST_PAYLOAD_TGZ}\"
  PAYLOAD_SOURCE_TGZ=\"\${LOCAL_SBCAST_PAYLOAD_TGZ}\"
fi
tar -xzf \"\${PAYLOAD_SOURCE_TGZ}\" -C \"\${LOCAL_REPO_DIR}\"
chmod +x \"\${LOCAL_REPO_DIR}/bin/souc\" \"\${LOCAL_REPO_DIR}/artifacts/self-hosted/souc-self-hosted-x86_64\"
SOUNIO_DIR=\"\${LOCAL_REPO_DIR}\"
if [[ -f \"\${SOUNIO_DIR}/abide_source_manifest.tsv\" ]]; then
  MANIFEST_PATH=\"\${SOUNIO_DIR}/abide_source_manifest.tsv\"
fi
RUN_MANIFEST_PATH=\"\${SOUNIO_DIR}/abide_roi_manifest.tsv\"
SOUC=\"\${SOUNIO_DIR}/bin/souc\"
\"\${SOUC}\" info >/dev/null

echo '═══════════════════════════════════════════════════════════════'
echo \"  Brain O-SSM ABIDE Campaign — \$(date)\"
echo \"  Host: \$(hostname), GPU: \${CUDA_VISIBLE_DEVICES:-none}\"
echo \"  Local run root: \${LOCAL_RUN_ROOT}\"
echo \"  OrangeFS run root: \${ORANGEFS_RUN_ROOT}\"
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
cat > \"\${SOUNIO_DIR}/abide_run_config.tsv\" <<CFG
train_fraction=${TRAIN_FRACTION}
drop_channel_frac=${DROP_CHANNEL_FRAC}
noise_std=${NOISE_STD}
CFG
cat >> \"\${SOUNIO_DIR}/abide_run_config.tsv\" <<CFG_EXTRA
${RUN_CONFIG_APPEND_TEXT}
CFG_EXTRA
echo \"  Source manifest: \${MANIFEST_PATH}\"
echo \"  Run manifest: \${RUN_MANIFEST_PATH}\"
echo \"  Run config: train_fraction=${TRAIN_FRACTION} drop_channel_frac=${DROP_CHANNEL_FRAC} noise_std=${NOISE_STD}\"
echo \"              global_train_epochs=${GLOBAL_TRAIN_EPOCHS:-default} global_train_lr=${GLOBAL_TRAIN_LR:-default} global_core_lr_scale=${GLOBAL_CORE_LR_SCALE:-default}\"
echo \"              oct_profile=${OCT_TRAIN_PROFILE} oct_train_mode=${OCT_TRAIN_MODE} oct_input_proj=${OCT_INPUT_PROJ_MODE}\"
echo \"              oct_proj_lr_scale=${OCT_PROJ_LR_SCALE} oct_proj_structured_scale=${OCT_PROJ_STRUCTURED_SCALE} oct_proj_delta_scale=${OCT_PROJ_DELTA_SCALE} oct_proj_hybrid_scale=${OCT_PROJ_HYBRID_SCALE}\"
echo \"              oct_train_epochs=${OCT_TRAIN_EPOCHS:-default} oct_train_lr=${OCT_TRAIN_LR:-default} oct_core_lr_scale=${OCT_CORE_LR_SCALE:-default}\"
echo \"              oct_warmup_epochs=${OCT_WARMUP_EPOCHS:-default}\"
echo \"              oct_init_preset=${OCT_INIT_PRESET:-default}\"
echo \"              oct_assoc_reg=${OCT_ASSOC_REG:-default} oct_assoc_target=${OCT_ASSOC_TARGET:-default}\"
echo \"              oct_assoc_schedule_mode=${OCT_ASSOC_SCHEDULE_MODE:-default} oct_assoc_schedule_start=${OCT_ASSOC_SCHEDULE_START:-default} oct_assoc_schedule_mid=${OCT_ASSOC_SCHEDULE_MID:-default} oct_assoc_schedule_end=${OCT_ASSOC_SCHEDULE_END:-default}\"
echo \"              oct_assoc_drift_reg=${OCT_ASSOC_DRIFT_REG:-default} oct_head_trust_region=${OCT_HEAD_TRUST_REGION:-default} oct_head_renorm=${OCT_HEAD_RENORM:-default}\"
echo \"              h_input_proj=${H_INPUT_PROJ_MODE} h_proj_lr_scale=${H_PROJ_LR_SCALE}\"
echo \"              h_train_epochs=${H_TRAIN_EPOCHS:-default} h_train_lr=${H_TRAIN_LR:-default} h_core_lr_scale=${H_CORE_LR_SCALE:-default}\"
echo \"              h_warmup_epochs=${H_WARMUP_EPOCHS:-default}\"
echo \"              h_init_preset=${H_INIT_PRESET:-default}\"

echo
echo '[Phase 1] Compiling Sounio ABIDE benchmark...'
\"\${SOUC}\" compile \"\${SOUNIO_DIR}/examples/brain_ossm_abide.sio\" -o \"\${BUILD_DIR}/brain_ossm_abide.elf\"

echo
echo '[Phase 2] Running Sounio benchmark...'
pushd \"\${SOUNIO_DIR}\" >/dev/null
time \"\${BUILD_DIR}/brain_ossm_abide.elf\" > \"\${RESULTS_DIR}/brain_ossm_abide_results.txt\" 2>&1
popd >/dev/null

echo
echo '[Phase 3] Parsing Sounio structured metrics...'
python3 \"\${SOUNIO_DIR}/scripts/research/parse_brain_ossm_abide_output.py\" \
  --input \"\${RESULTS_DIR}/brain_ossm_abide_results.txt\" \
  --output-dir \"\${RESULTS_DIR}/sounio_structured\"

bootstrap_local_target_torch() {
  local pybin=\"\$1\"
  export PIP_BREAK_SYSTEM_PACKAGES=1
  if ! \"\${pybin}\" -m pip --version >/dev/null 2>&1; then
    \"\${pybin}\" - <<'PY'
import ssl, urllib.request
ctx = ssl._create_unverified_context()
with urllib.request.urlopen(\"https://bootstrap.pypa.io/get-pip.py\", context=ctx) as r, open(\"/tmp/get-pip.py\", \"wb\") as f:
    f.write(r.read())
PY
    \"\${pybin}\" /tmp/get-pip.py --user >/dev/null
  fi
  \"\${pybin}\" -m pip install --no-cache-dir \
    --trusted-host download.pytorch.org \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --target \"\${LOCAL_PY_SITE_DIR}\" \
    torch==2.6.0 \
    numpy==2.2.6 >/dev/null
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
else
  PYTHON_RUNNER=\"\"
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_RUNNER=\"\$(command -v python3)\"
    if ! PYTHONPATH=\"\${LOCAL_PY_SITE_DIR}\" \"\${PYTHON_RUNNER}\" -c 'import torch' >/dev/null 2>&1; then
      bootstrap_local_target_torch \"\${PYTHON_RUNNER}\"
    fi
  fi
fi

if [[ -z \"\${PYTHON_RUNNER}\" ]] || ! PYTHONPATH=\"\${LOCAL_PY_SITE_DIR}\" \"\${PYTHON_RUNNER}\" -c 'import torch' >/dev/null 2>&1; then
  echo 'failed to provision a Python runtime with torch support for external baselines' >&2
  exit 1
fi

echo
echo '[Phase 4] Running external baseline suite...'
PYTHONPATH=\"\${LOCAL_PY_SITE_DIR}\" \"\${PYTHON_RUNNER}\" \"\${SOUNIO_DIR}/scripts/research/abide_external_baselines.py\" \
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
echo '[Phase 5] Aggregating campaign leaderboard...'
python3 \"\${SOUNIO_DIR}/scripts/research/aggregate_brain_ossm_campaign.py\" \
  --sounio-dir \"\${RESULTS_DIR}/sounio_structured\" \
  --external-dir \"\${RESULTS_DIR}/external_baselines\" \
  --output-dir \"\${RESULTS_DIR}/campaign\"

echo
echo '[Phase 6] Persisting results...'
RESULTS_ARCHIVE=\"\${LOCAL_RUN_ROOT}/abide_campaign_bundle.tgz\"
mkdir -p \"\${LOCAL_RUN_ROOT}/repo\"
cp -f \"\${SOUNIO_DIR}/abide_run_config.tsv\" \"\${LOCAL_RUN_ROOT}/repo/abide_run_config.tsv\"
tar -C \"\${LOCAL_RUN_ROOT}\" -czf \"\${RESULTS_ARCHIVE}\" results logs repo
if ! tar -tzf \"\${RESULTS_ARCHIVE}\" >/dev/null 2>&1; then
  echo \"local campaign bundle failed validation: \${RESULTS_ARCHIVE}\" >&2
  exit 1
fi
if [[ \"\${PERSIST_MODE}\" = \"orangefs\" ]]; then
  ensure_dir_with_retry \"\${ORANGEFS_RUN_ROOT}\"
  ensure_dir_with_retry \"\${ORANGEFS_DIR}\"
  copy_with_retry \"\${RESULTS_ARCHIVE}\" \"\${ORANGEFS_RUN_ROOT}/abide_campaign_bundle.tgz\" tgz
  copy_with_retry \"\${JOB_LOG_PATH}\" \"\${ORANGEFS_RUN_ROOT}/job-\${SLURM_JOB_ID}.log\"
  copy_with_retry \"\${SOUNIO_DIR}/abide_run_config.tsv\" \"\${ORANGEFS_RUN_ROOT}/abide_run_config.tsv\"
  copy_with_retry \"\${RESULTS_DIR}/brain_ossm_abide_results.txt\" \"\${ORANGEFS_DIR}/brain_ossm_abide_results.txt\"
  copy_with_retry \"\${SOUNIO_DIR}/abide_run_config.tsv\" \"\${ORANGEFS_DIR}/brain_ossm_abide_run_config.tsv\"
  copy_with_retry \"\${RESULTS_DIR}/sounio_structured/overall_metrics.tsv\" \"\${ORANGEFS_DIR}/brain_ossm_abide_overall_metrics.tsv\"
  copy_with_retry \"\${RESULTS_DIR}/sounio_structured/per_site_metrics.tsv\" \"\${ORANGEFS_DIR}/brain_ossm_abide_per_site_metrics.tsv\"
  copy_with_retry \"\${RESULTS_DIR}/external_baselines/overall_metrics.tsv\" \"\${ORANGEFS_DIR}/external_overall_metrics.tsv\"
  copy_with_retry \"\${RESULTS_DIR}/campaign/leaderboard.tsv\" \"\${ORANGEFS_DIR}/campaign_leaderboard.tsv\"
  copy_with_retry \"\${RESULTS_DIR}/campaign/per_site_leaderboard.tsv\" \"\${ORANGEFS_DIR}/campaign_per_site_leaderboard.tsv\"
  copy_with_retry \"\${RESULTS_DIR}/campaign/campaign_summary.json\" \"\${ORANGEFS_DIR}/campaign_summary.json\"
  echo \"Campaign completed. Results available in \${ORANGEFS_DIR}\"
else
  echo \"Campaign completed. Local bundle preserved on worker at \${LOCAL_RUN_ROOT}\"
  echo \"Bundle: \${RESULTS_ARCHIVE}\"
fi
EOF

sed -i 's/\\"/"/g' "${LOCAL_SBATCH_FILE}"

cat "${LOCAL_SBATCH_FILE}" | kubectl -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${SBATCH_FILE}'"

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

mkdir -p "${STATE_DIR}"
cat > "${STATE_FILE}.tmp" <<EOF
RUN_ID='${RUN_ID}'
JOB_ID='${JOB_ID}'
STATE_WRITTEN_UNIX='$(date +%s)'
PAYLOAD_TRANSFER_MODE='${PAYLOAD_TRANSFER_MODE}'
PERSIST_MODE='${PERSIST_MODE}'
SBATCH_NODELIST='${SBATCH_NODELIST}'
SBATCH_EXCLUDE='${SBATCH_EXCLUDE}'
STAGE_ROOT='${STAGE_ROOT}'
ORANGEFS_RESULTS_DIR='${ORANGEFS_RESULTS_DIR}'
ABIDE_MANIFEST_PATH='${ABIDE_MANIFEST_PATH}'
NS='${NS}'
LOGIN_POD='${LOGIN_POD}'
EOF
mv "${STATE_FILE}.tmp" "${STATE_FILE}"
chmod 0644 "${STATE_FILE}"

echo
echo "Submitted ABIDE campaign job:"
echo "  RUN_ID: ${RUN_ID}"
echo "  JobID: ${JOB_ID}"
echo "  Stage root: ${STAGE_ROOT}"
echo "  Persist mode: ${PERSIST_MODE}"
echo "  Payload transfer: ${PAYLOAD_TRANSFER_MODE}"
if [[ "${PERSIST_MODE}" = "orangefs" ]]; then
  echo "  Results dir: ${ORANGEFS_RESULTS_DIR}"
else
  echo "  Results dir: worker-local bundle on the admitted Slurm worker"
fi
echo "  Fetch: /home/devsounio/sounio/scripts/gpu/fetch_abide_campaign_auto.sh --run-id ${RUN_ID} --dest-dir /home/devsounio/sounio/artifacts/research/abide/${RUN_ID}"
echo "  Inspect: kubectl -n ${NS} exec ${LOGIN_POD} -- sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Start,End"
