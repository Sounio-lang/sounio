#!/usr/bin/env bash
set -euo pipefail

SOUNIO_DIR="${SOUNIO_DIR:-/home/devsounio/sounio}"
ROUND_STAGE="${ROUND_STAGE:-smoke}"
TEMPORAL_PROFILE="${TEMPORAL_PROFILE:-v2}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

submit_branch() {
  local label="$1"
  shift
  echo
  echo "=== ${label} ==="
  echo "stage=${ROUND_STAGE} temporal=${TEMPORAL_PROFILE}"
  "$@"
}

case "${ROUND_STAGE}" in
  smoke)
    COMMON=(PROFILE=lowdata TEMPORAL_PROFILE="${TEMPORAL_PROFILE}" MAX_SITES=2 LIMIT_SUBJECTS=64 SEED_COUNT=20)
    ;;
  lowdata_full)
    COMMON=(PROFILE=lowdata TEMPORAL_PROFILE="${TEMPORAL_PROFILE}" SEED_COUNT=20)
    ;;
  full_train)
    COMMON=(TEMPORAL_PROFILE="${TEMPORAL_PROFILE}" SEED_COUNT=20 TRAIN_FRACTION=1.0)
    ;;
  *)
    echo "unsupported ROUND_STAGE=${ROUND_STAGE}; expected smoke|lowdata_full|full_train" >&2
    exit 1
    ;;
esac

if [[ "${ROUND_STAGE}" == "full_train" ]]; then
  submit_branch "winner-o_alg_v1_mandel" \
    env RUN_ID="brain-ossm-abide-${TEMPORAL_PROFILE}-fulltrain-oalgv1mandel-${TIMESTAMP}" \
    OCT_TRAIN_PROFILE=o_alg_v1_mandel OCT_INPUT_PROJ_MODE=mandelbrot_d2_hybrid \
    "${COMMON[@]}" \
    bash "${SOUNIO_DIR}/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh"
  exit 0
fi

submit_branch "champion-o_assoc_stable" \
  env RUN_ID="brain-ossm-abide-${TEMPORAL_PROFILE}-${ROUND_STAGE}-oassocstable-${TIMESTAMP}" \
  OCT_TRAIN_PROFILE=o_assoc_stable OCT_INPUT_PROJ_MODE=identity \
  "${COMMON[@]}" \
  bash "${SOUNIO_DIR}/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh"

submit_branch "o_alg_v1_identity" \
  env RUN_ID="brain-ossm-abide-${TEMPORAL_PROFILE}-${ROUND_STAGE}-oalgv1-id-${TIMESTAMP}" \
  OCT_TRAIN_PROFILE=o_alg_v1 OCT_INPUT_PROJ_MODE=identity \
  "${COMMON[@]}" \
  bash "${SOUNIO_DIR}/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh"

submit_branch "o_alg_v1_mandel_residual" \
  env RUN_ID="brain-ossm-abide-${TEMPORAL_PROFILE}-${ROUND_STAGE}-oalgv1m-md2res-${TIMESTAMP}" \
  OCT_TRAIN_PROFILE=o_alg_v1_mandel OCT_INPUT_PROJ_MODE=mandelbrot_d2_residual \
  "${COMMON[@]}" \
  bash "${SOUNIO_DIR}/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh"

submit_branch "o_alg_v1_mandel_hybrid" \
  env RUN_ID="brain-ossm-abide-${TEMPORAL_PROFILE}-${ROUND_STAGE}-oalgv1m-md2hyb-${TIMESTAMP}" \
  OCT_TRAIN_PROFILE=o_alg_v1_mandel OCT_INPUT_PROJ_MODE=mandelbrot_d2_hybrid \
  "${COMMON[@]}" \
  bash "${SOUNIO_DIR}/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh"

if [[ "${TEMPORAL_PROFILE}" != "v5" ]]; then
  if [[ "${ROUND_STAGE}" == "smoke" ]]; then
    submit_branch "o_alg_v1_identity_on_v5" \
      env RUN_ID="brain-ossm-abide-v5-${ROUND_STAGE}-oalgv1-id-${TIMESTAMP}" \
      OCT_TRAIN_PROFILE=o_alg_v1 OCT_INPUT_PROJ_MODE=identity \
      PROFILE=lowdata TEMPORAL_PROFILE=v5 MAX_SITES=2 LIMIT_SUBJECTS=64 SEED_COUNT=20 \
      bash "${SOUNIO_DIR}/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh"
  else
    submit_branch "o_alg_v1_identity_on_v5" \
      env RUN_ID="brain-ossm-abide-v5-${ROUND_STAGE}-oalgv1-id-${TIMESTAMP}" \
      OCT_TRAIN_PROFILE=o_alg_v1 OCT_INPUT_PROJ_MODE=identity \
      PROFILE=lowdata TEMPORAL_PROFILE=v5 SEED_COUNT=20 \
      bash "${SOUNIO_DIR}/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh"
  fi
fi
