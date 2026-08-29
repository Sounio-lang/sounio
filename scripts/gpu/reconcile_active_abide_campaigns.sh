#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVE_FILE="${ACTIVE_FILE:-${ROOT_DIR}/artifacts/research/abide/ACTIVE_CAMPAIGNS.tsv}"
DEST_ROOT="${DEST_ROOT:-${ROOT_DIR}/artifacts/research/abide}"
FETCH_HELPER="${ROOT_DIR}/scripts/gpu/fetch_abide_campaign_auto.sh"
SLURM_NS="${SLURM_NS:-slurm-pilot}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/instance=slurm-pilot-login-slinky,app.kubernetes.io/name=login}"

if [[ ! -f "${ACTIVE_FILE}" ]]; then
  echo "ACTIVE_CAMPAIGNS file not found: ${ACTIVE_FILE}" >&2
  exit 1
fi

login_pod="$(
  kubectl -n "${SLURM_NS}" get pods -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
)"
if [[ -z "${login_pod}" ]]; then
  echo "Could not resolve a running Slurm login pod in namespace ${SLURM_NS}" >&2
  exit 1
fi

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

job_snapshot() {
  local job_id="$1"
  kubectl -n "${SLURM_NS}" exec "${login_pod}" -- \
    sacct -P -n -j "${job_id}" --format=JobIDRaw,State,ExitCode,Start,End,NodeList \
    | awk -F'|' -v want="${job_id}" '$1 == want { print; exit }'
}

normalize_status() {
  local state="$1"
  state="${state%% *}"
  state="${state%%+*}"
  case "${state}" in
    COMPLETED) echo "completed" ;;
    RUNNING) echo "running" ;;
    PENDING|CONFIGURING|COMPLETING|STAGE_OUT) echo "running" ;;
    CANCELLED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE)
      echo "failed"
      ;;
    *)
      echo "${state,,}"
      ;;
  esac
}

append_note_once() {
  local note="$1"
  local suffix="$2"
  if [[ -z "${suffix}" ]]; then
    printf '%s' "${note}"
    return 0
  fi
  if [[ "${note}" == *"${suffix}"* ]]; then
    printf '%s' "${note}"
    return 0
  fi
  if [[ -z "${note}" ]]; then
    printf '%s' "${suffix}"
  else
    printf '%s %s' "${note}" "${suffix}"
  fi
}

fetch_if_needed() {
  local run_id="$1"
  local node_name="${2:-r770-proxmox}"
  local dest_dir="${DEST_ROOT}/${run_id}"
  if [[ -d "${dest_dir}/results" || -f "${dest_dir}/results/campaign/leaderboard.tsv" ]]; then
    return 0
  fi
  "${FETCH_HELPER}" --run-id "${run_id}" --node "${node_name}" --dest-dir "${dest_dir}" >/dev/null
}

{
  IFS= read -r header
  printf '%s\n' "${header}"
  while IFS=$'\t' read -r job_id run_id node train_fraction oct_profile oct_input_proj h_input_proj status note; do
    if [[ -z "${job_id:-}" ]]; then
      continue
    fi
    snapshot="$(job_snapshot "${job_id}" || true)"
    if [[ -n "${snapshot}" ]]; then
      IFS='|' read -r _ raw_state exit_code start_ts end_ts node_list <<<"${snapshot}"
      status="$(normalize_status "${raw_state}")"
      if [[ "${status}" == "completed" ]]; then
        fetch_if_needed "${run_id}" "${node}" || status="completed_unfetched"
      elif [[ "${status}" == "failed" ]]; then
        note="$(append_note_once "${note}" "(sacct=${raw_state}, exit=${exit_code})")"
      fi
    else
      status="unknown"
      note="$(append_note_once "${note}" "(no sacct row)")"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${job_id}" "${run_id}" "${node}" "${train_fraction}" "${oct_profile}" \
      "${oct_input_proj}" "${h_input_proj}" "${status}" "${note}"
  done
} < "${ACTIVE_FILE}" > "${tmp}"

mv "${tmp}" "${ACTIVE_FILE}"
