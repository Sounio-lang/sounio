#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVE_FILE="${ACTIVE_FILE:-${ROOT_DIR}/artifacts/research/abide/ACTIVE_CAMPAIGNS.tsv}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/research/abide}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/HEADLESS_SUPERVISOR.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/HEADLESS_SUPERVISOR.pid}"
LOCK_FILE="${LOCK_FILE:-${LOG_DIR}/HEADLESS_SUPERVISOR.lock}"
POLL_SEC="${POLL_SEC:-120}"
DURATION_SEC="${DURATION_SEC:-18000}"
RECONCILE="${ROOT_DIR}/scripts/gpu/reconcile_active_abide_campaigns.sh"
SUMMARY="${ROOT_DIR}/scripts/gpu/summarize_active_abide_campaigns.sh"
SLURM_NS="${SLURM_NS:-slurm-pilot}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/instance=slurm-pilot-login-slinky,app.kubernetes.io/name=login}"

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "headless supervisor already running" >&2
  exit 1
fi
echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

ts() {
  date -Is
}

resolve_login_pod() {
  kubectl -n "${SLURM_NS}" get pods \
    -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
}

active_job_ids() {
  awk -F'\t' '
    NR > 1 && ($8 == "running" || $8 == "queued" || $8 == "pending" || $8 == "completed_unfetched") {
      print $1
    }
  ' "${ACTIVE_FILE}" | paste -sd, -
}

{
  echo "[headless] start $(ts) duration_sec=${DURATION_SEC} poll_sec=${POLL_SEC}"
  start_epoch="$(date +%s)"
  while true; do
    now_epoch="$(date +%s)"
    elapsed="$((now_epoch - start_epoch))"
    if (( elapsed > DURATION_SEC )); then
      echo "[headless] stop $(ts) reason=duration elapsed=${elapsed}s"
      break
    fi

    if ! "${RECONCILE}" >/dev/null 2>&1; then
      echo "[headless] warn $(ts) reconcile_failed"
    fi

    job_ids="$(active_job_ids)"
    if [[ -z "${job_ids}" ]]; then
      echo "[headless] done $(ts) no_active_jobs"
      "${SUMMARY}" || true
      break
    fi

    login_pod="$(resolve_login_pod 2>/dev/null || true)"
    echo "[headless] tick $(ts) elapsed=${elapsed}s jobs=${job_ids}"
    if [[ -n "${login_pod}" ]]; then
      kubectl -n "${SLURM_NS}" exec "${login_pod}" -- \
        sacct -j "${job_ids}" --format=JobID,State,ExitCode,Elapsed,Start,End,NodeList%30 --noheader \
        || echo "[headless] warn $(ts) sacct_failed"
    else
      echo "[headless] warn $(ts) login_pod_unresolved"
    fi

    echo "[headless] summary $(ts)"
    "${SUMMARY}" || true
    sleep "${POLL_SEC}"
  done
} >> "${LOG_FILE}" 2>&1
