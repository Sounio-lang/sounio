#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-slurm-pilot}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
ABIDE_DATA_ROOT="${ABIDE_DATA_ROOT:-/orangefs/training/sounio/abide-data}"
ABIDE_MANIFEST_PATH="${ABIDE_MANIFEST_PATH:-/orangefs/training/sounio/abide-data/abide_roi_manifest.tsv}"
ATLAS_NAME="${ATLAS_NAME:-cc200}"
ABIDE_BENCHMARK_PATH="${ABIDE_BENCHMARK_PATH:-/home/devsounio/sounio/examples/brain_ossm_abide.sio}"
RUN_ID="${RUN_ID:-brain-ossm-abide-preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-/orangefs/training/sounio/abide-preflight/${RUN_ID}}"

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

BENCHMARK_EXISTS="no"
if [[ -f "${ABIDE_BENCHMARK_PATH}" ]]; then
  BENCHMARK_EXISTS="yes"
fi

kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
  set -euo pipefail
  mkdir -p '${OUT_DIR}'
  {
    echo 'ABIDE preflight'
    echo 'timestamp='\"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    echo 'data_root=${ABIDE_DATA_ROOT}'
    echo 'manifest_path=${ABIDE_MANIFEST_PATH}'
    echo 'atlas=${ATLAS_NAME}'
    echo 'benchmark_path=${ABIDE_BENCHMARK_PATH}'
    echo 'benchmark_exists=${BENCHMARK_EXISTS}'
    if [[ -d '${ABIDE_DATA_ROOT}' ]]; then
      echo 'data_root_exists=yes'
      find '${ABIDE_DATA_ROOT}' -maxdepth 2 -type d | sed -n '1,80p'
    else
      echo 'data_root_exists=no'
    fi
    if [[ -f '${ABIDE_MANIFEST_PATH}' ]]; then
      echo 'manifest_exists=yes'
      echo 'manifest_preview_begin'
      sed -n '1,5p' '${ABIDE_MANIFEST_PATH}'
      echo 'manifest_preview_end'
    else
      echo 'manifest_exists=no'
    fi
  } > '${OUT_DIR}/preflight.txt'
"

echo "ABIDE preflight written to ${OUT_DIR}/preflight.txt"
echo "This wrapper validates dataset visibility and benchmark presence."
if [[ "${BENCHMARK_EXISTS}" != "yes" ]]; then
  echo "missing benchmark source: ${ABIDE_BENCHMARK_PATH}" >&2
  echo "next step: implement brain_ossm_abide.sio before real ABIDE submission" >&2
  exit 2
fi

echo "It does not launch ABIDE training until brain_ossm_abide.sio exists."
