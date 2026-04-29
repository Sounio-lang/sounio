#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-slurm-pilot}"
POD=""
RUN_DIR=""
DEST_DIR=""
RUN_CONFIG_PATH=""

usage() {
  cat <<'EOF'
usage: fetch_abide_campaign_from_worker.sh --pod <pod> --run-dir <remote-run-dir> --dest-dir <local-dest-dir> [--run-config <remote-config-path>]

Fetches an ABIDE campaign run directory from a worker pod into the local host.
The worker only needs local /tmp state; OrangeFS is not touched during fetch.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pod)
      POD="${2:?missing value for --pod}"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="${2:?missing value for --run-dir}"
      shift 2
      ;;
    --dest-dir)
      DEST_DIR="${2:?missing value for --dest-dir}"
      shift 2
      ;;
    --run-config)
      RUN_CONFIG_PATH="${2:?missing value for --run-config}"
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

if [[ -z "${POD}" || -z "${RUN_DIR}" || -z "${DEST_DIR}" ]]; then
  usage >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

kubectl -n "${NS}" exec "${POD}" -- sh -lc "test -d '${RUN_DIR}'"
kubectl -n "${NS}" exec "${POD}" -- sh -lc "tar -C '${RUN_DIR}' -czf - results logs" \
  | tar -xzf - -C "${DEST_DIR}"

if [[ -n "${RUN_CONFIG_PATH}" ]]; then
  mkdir -p "${DEST_DIR}/repo"
  kubectl -n "${NS}" exec "${POD}" -- sh -lc "cat '${RUN_CONFIG_PATH}'" \
    > "${DEST_DIR}/repo/abide_run_config.tsv"
fi

echo "Fetched ABIDE campaign bundle:"
echo "  Pod: ${POD}"
echo "  Run dir: ${RUN_DIR}"
echo "  Dest dir: ${DEST_DIR}"
