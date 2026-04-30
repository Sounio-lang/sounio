#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-slurm-pilot}"
NODE="${NODE:-r770-proxmox}"
RUN_ID=""
DEST_DIR=""
POD_LABEL="app.kubernetes.io/instance=slurm-pilot-worker-gpuorangefs"

usage() {
  cat <<'EOF'
usage: fetch_abide_campaign_by_run_id.sh --run-id <run-id> --dest-dir <local-dest-dir> [--node <node-name>]

Fetches a preserved worker-local ABIDE campaign bundle by RUN_ID.
Defaults to the stable worker node r770-proxmox.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --dest-dir)
      DEST_DIR="${2:?missing value for --dest-dir}"
      shift 2
      ;;
    --node)
      NODE="${2:?missing value for --node}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_ID}" || -z "${DEST_DIR}" ]]; then
  usage >&2
  exit 1
fi

POD="$(
  kubectl -n "${NS}" get pods -l "${POD_LABEL}" -o wide \
    | awk -v node="${NODE}" '$7 == node { print $1; exit }'
)"

if [[ -z "${POD}" ]]; then
  echo "could not resolve worker pod for node ${NODE}" >&2
  exit 1
fi

RUN_DIR="/tmp/${RUN_ID}-run"

"$(dirname "$0")/fetch_abide_campaign_from_worker.sh" \
  --pod "${POD}" \
  --run-dir "${RUN_DIR}" \
  --dest-dir "${DEST_DIR}" \
  --run-config "${RUN_DIR}/repo/abide_run_config.tsv"
