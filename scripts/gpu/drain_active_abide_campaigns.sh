#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVE_FILE="${ACTIVE_FILE:-${ROOT_DIR}/artifacts/research/abide/ACTIVE_CAMPAIGNS.tsv}"
RECONCILE="${ROOT_DIR}/scripts/gpu/reconcile_active_abide_campaigns.sh"
SUMMARY="${ROOT_DIR}/scripts/gpu/summarize_active_abide_campaigns.sh"
POLL_SEC="${POLL_SEC:-60}"
MAX_POLLS="${MAX_POLLS:-0}"

poll=0
while true; do
  "${RECONCILE}" >/dev/null
  running_count="$(
    awk -F'\t' 'NR>1 && ($8=="running" || $8=="queued" || $8=="pending" || $8=="completed_unfetched") { c++ } END { print c+0 }' "${ACTIVE_FILE}"
  )"
  echo "[drain] poll=${poll} active=${running_count} at $(date -Is)"
  if [[ "${running_count}" -eq 0 ]]; then
    break
  fi
  if [[ "${MAX_POLLS}" -gt 0 && "${poll}" -ge "${MAX_POLLS}" ]]; then
    echo "[drain] reached MAX_POLLS=${MAX_POLLS}" >&2
    exit 2
  fi
  poll=$((poll + 1))
  sleep "${POLL_SEC}"
done

"${SUMMARY}"
