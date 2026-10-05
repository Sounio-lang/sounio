#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
RUNS_JSONL="${PLAN_BIG_OVERNIGHT_JSONL:-$ART_DIR/runs.v1.jsonl}"
HOURLY_JSONL="${PLAN_BIG_OVERNIGHT_HOURLY_REPORT_JSONL:-$ART_DIR/hourly_report.v1.jsonl}"
OUT_JSON="${PLAN_BIG_OVERNIGHT_RETENTION_JSON:-$ROOT_DIR/artifacts/omega/overnight_plan_big_retention.v1.json}"
KEEP_RUN_LOGS="${PLAN_BIG_OVERNIGHT_KEEP_RUN_LOGS:-120}"
KEEP_RUNS_JSONL="${PLAN_BIG_OVERNIGHT_KEEP_RUNS_JSONL:-1000}"
KEEP_HOURLY_JSONL="${PLAN_BIG_OVERNIGHT_KEEP_HOURLY_JSONL:-1000}"
PRUNE_BACKUPS="${PLAN_BIG_OVERNIGHT_PRUNE_BACKUPS:-1}"

usage() {
  cat <<USAGE
Usage: scripts/ops/rotate_overnight_plan_big_artifacts.sh [options]

Options:
  --keep-run-logs N      Keep newest N run_*.log files (default: 120)
  --keep-runs-jsonl N    Keep newest N lines in runs.v1.jsonl (default: 1000)
  --keep-hourly-jsonl N  Keep newest N lines in hourly_report.v1.jsonl (default: 1000)
  --prune-backups 0|1    Remove *.bak.codex from overnight artifact dir (default: 1)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-run-logs)
      KEEP_RUN_LOGS="$2"
      shift 2
      ;;
    --keep-runs-jsonl)
      KEEP_RUNS_JSONL="$2"
      shift 2
      ;;
    --keep-hourly-jsonl)
      KEEP_HOURLY_JSONL="$2"
      shift 2
      ;;
    --prune-backups)
      PRUNE_BACKUPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for v in KEEP_RUN_LOGS KEEP_RUNS_JSONL KEEP_HOURLY_JSONL; do
  if ! [[ "${!v}" =~ ^[0-9]+$ ]] || [[ "${!v}" -eq 0 ]]; then
    echo "error: $v must be positive integer" >&2
    exit 2
  fi
done
if [[ "$PRUNE_BACKUPS" != "0" && "$PRUNE_BACKUPS" != "1" ]]; then
  echo "error: --prune-backups must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$ART_DIR" "$(dirname "$OUT_JSON")"

trim_jsonl() {
  local path="$1"
  local keep_lines="$2"
  local before_lines after_lines trimmed_lines
  before_lines=0
  after_lines=0
  trimmed_lines=0
  if [[ -f "$path" ]]; then
    before_lines="$(wc -l < "$path" | tr -d '[:space:]')"
    if [[ "$before_lines" -gt "$keep_lines" ]]; then
      tmp="$(mktemp)"
      tail -n "$keep_lines" "$path" >"$tmp"
      mv "$tmp" "$path"
    fi
    after_lines="$(wc -l < "$path" | tr -d '[:space:]')"
    trimmed_lines=$((before_lines - after_lines))
    if [[ "$trimmed_lines" -lt 0 ]]; then
      trimmed_lines=0
    fi
  fi
  printf '%s,%s,%s' "$before_lines" "$after_lines" "$trimmed_lines"
}

result_runs="$(trim_jsonl "$RUNS_JSONL" "$KEEP_RUNS_JSONL")"
runs_before="${result_runs%%,*}"
rest="${result_runs#*,}"
runs_after="${rest%%,*}"
runs_trimmed="${rest##*,}"

result_hourly="$(trim_jsonl "$HOURLY_JSONL" "$KEEP_HOURLY_JSONL")"
hourly_before="${result_hourly%%,*}"
rest="${result_hourly#*,}"
hourly_after="${rest%%,*}"
hourly_trimmed="${rest##*,}"

mapfile -t run_logs < <(find "$ART_DIR" -maxdepth 1 -type f -name 'run_*.log' -print | sort)
run_logs_before="${#run_logs[@]}"
run_logs_deleted=0
if [[ "$run_logs_before" -gt "$KEEP_RUN_LOGS" ]]; then
  delete_count=$((run_logs_before - KEEP_RUN_LOGS))
  for ((i=0; i<delete_count; i++)); do
    rm -f "${run_logs[$i]}"
    run_logs_deleted=$((run_logs_deleted + 1))
  done
fi
mapfile -t run_logs_after_arr < <(find "$ART_DIR" -maxdepth 1 -type f -name 'run_*.log' -print | sort)
run_logs_after="${#run_logs_after_arr[@]}"

backup_deleted=0
if [[ "$PRUNE_BACKUPS" -eq 1 ]]; then
  while IFS= read -r bak_file; do
    [[ -z "$bak_file" ]] && continue
    rm -f "$bak_file"
    backup_deleted=$((backup_deleted + 1))
  done < <(find "$ART_DIR" -maxdepth 1 -type f -name '*.bak.codex' -print)
fi

jq -cn \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg art_dir "${ART_DIR#$ROOT_DIR/}" \
  --arg runs_jsonl "${RUNS_JSONL#$ROOT_DIR/}" \
  --arg hourly_jsonl "${HOURLY_JSONL#$ROOT_DIR/}" \
  --argjson keep_run_logs "$KEEP_RUN_LOGS" \
  --argjson keep_runs_jsonl "$KEEP_RUNS_JSONL" \
  --argjson keep_hourly_jsonl "$KEEP_HOURLY_JSONL" \
  --argjson prune_backups "$PRUNE_BACKUPS" \
  --argjson run_logs_before "$run_logs_before" \
  --argjson run_logs_after "$run_logs_after" \
  --argjson run_logs_deleted "$run_logs_deleted" \
  --argjson runs_before "$runs_before" \
  --argjson runs_after "$runs_after" \
  --argjson runs_trimmed "$runs_trimmed" \
  --argjson hourly_before "$hourly_before" \
  --argjson hourly_after "$hourly_after" \
  --argjson hourly_trimmed "$hourly_trimmed" \
  --argjson backup_deleted "$backup_deleted" \
  '{
    schema: "sounio.plan.big.overnight-retention.v1",
    generated_at_utc: $generated_at_utc,
    paths: {
      art_dir: $art_dir,
      runs_jsonl: $runs_jsonl,
      hourly_jsonl: $hourly_jsonl
    },
    policy: {
      keep_run_logs: $keep_run_logs,
      keep_runs_jsonl: $keep_runs_jsonl,
      keep_hourly_jsonl: $keep_hourly_jsonl,
      prune_backups: ($prune_backups == 1)
    },
    results: {
      run_logs_before: $run_logs_before,
      run_logs_after: $run_logs_after,
      run_logs_deleted: $run_logs_deleted,
      runs_jsonl_before: $runs_before,
      runs_jsonl_after: $runs_after,
      runs_jsonl_trimmed: $runs_trimmed,
      hourly_jsonl_before: $hourly_before,
      hourly_jsonl_after: $hourly_after,
      hourly_jsonl_trimmed: $hourly_trimmed,
      backup_deleted: $backup_deleted
    },
    pass_marker: true
  }' > "$OUT_JSON"

echo "OVERNIGHT_PLAN_BIG_RETENTION_PASS"
echo "JSON: $OUT_JSON"
