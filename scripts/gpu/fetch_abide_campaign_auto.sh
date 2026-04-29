#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID=""
DEST_DIR=""
NODE="${NODE:-r770-proxmox}"

usage() {
  cat <<'EOF'
usage: fetch_abide_campaign_auto.sh --run-id <run-id> --dest-dir <local-dest-dir> [--node <node-name>]

Fetches an ABIDE campaign bundle by trying the safer worker-local path first,
then falling back to the OrangeFS bundle if needed.
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

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/abide-fetch-auto.XXXXXX")"
cleanup() {
  rm -rf "${tmp_root}"
}
trap cleanup EXIT

worker_tmp="${tmp_root}/worker"
orangefs_tmp="${tmp_root}/orangefs"

if "${ROOT_DIR}/scripts/gpu/fetch_abide_campaign_by_run_id.sh" \
    --run-id "${RUN_ID}" \
    --node "${NODE}" \
    --dest-dir "${worker_tmp}" >/dev/null 2>&1; then
  rm -rf "${DEST_DIR}"
  mkdir -p "$(dirname "${DEST_DIR}")"
  mv "${worker_tmp}" "${DEST_DIR}"
  echo "Fetched ABIDE campaign bundle via worker-local path into ${DEST_DIR}"
  exit 0
fi

if "${ROOT_DIR}/scripts/gpu/fetch_abide_campaign_from_orangefs.sh" \
    --run-id "${RUN_ID}" \
    --dest-dir "${orangefs_tmp}" >/dev/null 2>&1; then
  rm -rf "${DEST_DIR}"
  mkdir -p "$(dirname "${DEST_DIR}")"
  mv "${orangefs_tmp}" "${DEST_DIR}"
  echo "Fetched ABIDE campaign bundle via OrangeFS fallback into ${DEST_DIR}"
  exit 0
fi

echo "failed to fetch ABIDE campaign ${RUN_ID} from both worker-local and OrangeFS paths" >&2
exit 1
