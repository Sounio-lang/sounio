#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVE_FILE="${ACTIVE_FILE:-${ROOT_DIR}/artifacts/research/abide/ACTIVE_CAMPAIGNS.tsv}"
CHAMPION_SUMMARY="${CHAMPION_SUMMARY:-${ROOT_DIR}/artifacts/research/abide/g2_transfer_summary.tsv}"
RECONCILE="${ROOT_DIR}/scripts/gpu/reconcile_active_abide_campaigns.sh"
SUMMARIZER="${ROOT_DIR}/scripts/research/summarize_abide_campaigns.py"
OUT_FILE="${OUT_FILE:-${ROOT_DIR}/artifacts/research/abide/ACTIVE_SUMMARY.tsv}"

"${RECONCILE}" >/dev/null

args=()
if [[ -f "${CHAMPION_SUMMARY}" ]]; then
  args+=(--champion-summary "${CHAMPION_SUMMARY}")
fi

while IFS=$'\t' read -r _job_id run_id _node _train_fraction _oct_profile _oct_input_proj _h_input_proj status _note; do
  [[ "${status}" == "completed" ]] || continue
  run_dir="${ROOT_DIR}/artifacts/research/abide/${run_id}"
  [[ -f "${run_dir}/results/campaign/leaderboard.tsv" ]] || continue
  args+=(--run-dir "${run_dir}")
done < <(tail -n +2 "${ACTIVE_FILE}")

python3 "${SUMMARIZER}" "${args[@]}" | tee "${OUT_FILE}"
