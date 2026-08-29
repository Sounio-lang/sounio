#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-slurm-pilot}"
LOGIN_POD="${LOGIN_POD:-}"
RUN_ID=""
DEST_DIR=""
BUNDLE_FILE="abide_campaign_bundle.tgz"

usage() {
  cat <<'EOF'
usage: fetch_abide_campaign_from_orangefs.sh --run-id <run-id> --dest-dir <local-dest-dir>

Fetches an ABIDE campaign bundle persisted in OrangeFS and extracts it locally.
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

RUN_ROOT="/orangefs/training/sounio/abide-campaign-runs/${RUN_ID}"
mkdir -p "${DEST_DIR}"

if [[ -z "${LOGIN_POD}" ]]; then
  LOGIN_POD="$(kubectl -n "${NS}" get pods \
    -l 'app.kubernetes.io/instance=slurm-pilot-login-slinky,app.kubernetes.io/name=login' \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}')"
fi

if [[ -z "${LOGIN_POD}" ]]; then
  echo "could not resolve a running login pod in namespace ${NS}" >&2
  exit 1
fi

kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "base64 -w0 '${RUN_ROOT}/${BUNDLE_FILE}'" \
  | base64 -d > "${DEST_DIR}/${BUNDLE_FILE}"

bundle_sha=""
if kubectl -n "${NS}" exec "${LOGIN_POD}" -- test -f "${RUN_ROOT}/${BUNDLE_FILE}.sha256"; then
  bundle_sha="$(kubectl -n "${NS}" exec "${LOGIN_POD}" -- cat "${RUN_ROOT}/${BUNDLE_FILE}.sha256" | awk '{print $1}')"
fi

if [[ -n "${bundle_sha}" ]]; then
  local_sha="$(sha256sum "${DEST_DIR}/${BUNDLE_FILE}" | awk '{print $1}')"
  if [[ "${local_sha}" != "${bundle_sha}" ]]; then
    echo "OrangeFS bundle checksum mismatch for ${RUN_ID}: expected ${bundle_sha}, got ${local_sha}" >&2
    exit 2
  fi
fi

if ! tar -xzf "${DEST_DIR}/${BUNDLE_FILE}" -C "${DEST_DIR}" 2>/dev/null; then
  if ! tar -xf "${DEST_DIR}/${BUNDLE_FILE}" -C "${DEST_DIR}" 2>/dev/null; then
    echo "fetched OrangeFS bundle is not a valid tar archive: ${DEST_DIR}/${BUNDLE_FILE}" >&2
    echo "the run root may still contain a corrupt or stale bundle; inspect job.log and consider worker_local fetch if that lane was used" >&2
    exit 2
  fi
fi
kubectl -n "${NS}" exec "${LOGIN_POD}" -- cat "${RUN_ROOT}/abide_run_config.tsv" > "${DEST_DIR}/abide_run_config.tsv"
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "for f in '${RUN_ROOT}'/job-*.log; do [ -f \"\$f\" ] && cat \"\$f\"; break; done" > "${DEST_DIR}/job.log"

echo "Fetched OrangeFS bundle for ${RUN_ID} into ${DEST_DIR}"
