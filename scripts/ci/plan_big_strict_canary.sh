#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ART_DIR="${PLAN_BIG_STRICT_CANARY_ART_DIR:-$ROOT_DIR/artifacts/omega/plan_big_strict_canary}"
RUNS_JSONL="${PLAN_BIG_STRICT_CANARY_JSONL:-$ART_DIR/runs.v1.jsonl}"
LATEST_JSON="${PLAN_BIG_STRICT_CANARY_LATEST_JSON:-$ART_DIR/latest.v1.json}"
GATE_STATUS_JSON="${PLAN_BIG_GATE_STATUS_JSON:-$ROOT_DIR/artifacts/omega/plan_big_gate_status.v1.json}"
GATE_SCRIPT="${PLAN_BIG_STRICT_CANARY_GATE_SCRIPT:-$ROOT_DIR/scripts/ci/plan_big_gate.sh}"
GATE_ARGS="${PLAN_BIG_STRICT_CANARY_GATE_ARGS:---with-overnight-health --overnight-no-auto-heal --require-overnight-burnin}"

usage() {
  cat <<USAGE
Usage: scripts/ci/plan_big_strict_canary.sh [--gate-script PATH_OR_CMD] [--gate-args "ARGS"]

Environment:
  PLAN_BIG_STRICT_CANARY_GATE_SCRIPT  Gate script/command (default: scripts/ci/plan_big_gate.sh)
  PLAN_BIG_STRICT_CANARY_GATE_ARGS    Gate args string (default: --with-overnight-health --overnight-no-auto-heal --require-overnight-burnin)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate-script)
      GATE_SCRIPT="$2"
      shift 2
      ;;
    --gate-args)
      GATE_ARGS="$2"
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

mkdir -p "$ART_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 2
fi

if [[ "$GATE_SCRIPT" == */* ]]; then
  if [[ "$GATE_SCRIPT" != /* ]]; then
    GATE_SCRIPT="$ROOT_DIR/$GATE_SCRIPT"
  fi
  if [[ ! -e "$GATE_SCRIPT" ]]; then
    echo "error: strict canary gate script does not exist: $GATE_SCRIPT" >&2
    exit 2
  fi
else
  if command -v "$GATE_SCRIPT" >/dev/null 2>&1; then
    GATE_SCRIPT="$(command -v "$GATE_SCRIPT")"
  else
    echo "error: strict canary gate command not found in PATH: $GATE_SCRIPT" >&2
    exit 2
  fi
fi

declare -a GATE_ARGS_ARRAY=()
if [[ -n "$GATE_ARGS" ]]; then
  # shellcheck disable=SC2206
  GATE_ARGS_ARRAY=($GATE_ARGS)
fi

started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
start_epoch="$(date -u +%s)"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
run_log="$ART_DIR/run_${run_id}.log"
run_number=1
if [[ -f "$RUNS_JSONL" ]]; then
  existing_runs="$(wc -l < "$RUNS_JSONL" | tr -d '[:space:]')"
  if [[ -n "$existing_runs" && "$existing_runs" =~ ^[0-9]+$ ]]; then
    run_number=$((existing_runs + 1))
  fi
fi
log_path_rel="${run_log#$ROOT_DIR/}"
gate_status_rel="${GATE_STATUS_JSON#$ROOT_DIR/}"

echo "[strict-canary] run=$run_number started_at_utc=$started_at_utc start" | tee -a "$run_log"

set +e
if [[ -x "$GATE_SCRIPT" ]]; then
  "$GATE_SCRIPT" "${GATE_ARGS_ARRAY[@]}" >>"$run_log" 2>&1
  gate_rc=$?
else
  bash "$GATE_SCRIPT" "${GATE_ARGS_ARRAY[@]}" >>"$run_log" 2>&1
  gate_rc=$?
fi
set -e

status="fail"
pass_marker=false
if [[ "$gate_rc" -eq 0 ]]; then
  status="pass"
  pass_marker=true
fi

finished_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
finish_epoch="$(date -u +%s)"
duration_sec=$((finish_epoch - start_epoch))
if [[ "$duration_sec" -lt 0 ]]; then
  duration_sec=0
fi

result_json="$(jq -cn \
  --arg generated_at_utc "$finished_at_utc" \
  --arg started_at_utc "$started_at_utc" \
  --arg finished_at_utc "$finished_at_utc" \
  --argjson duration_sec "$duration_sec" \
  --argjson run_number "$run_number" \
  --arg status "$status" \
  --argjson rc "$gate_rc" \
  --argjson pass_marker "$pass_marker" \
  --arg gate_script "${GATE_SCRIPT#$ROOT_DIR/}" \
  --arg gate_args "$GATE_ARGS" \
  --arg gate_status_path "$gate_status_rel" \
  --arg log_path "$log_path_rel" \
  '{
    schema: "sounio.plan.big.strict-canary.run.v1",
    generated_at_utc: $generated_at_utc,
    started_at_utc: $started_at_utc,
    finished_at_utc: $finished_at_utc,
    duration_sec: $duration_sec,
    run_number: $run_number,
    status: $status,
    rc: $rc,
    pass_marker: $pass_marker,
    gate_script: $gate_script,
    gate_args: $gate_args,
    gate_status_path: $gate_status_path,
    log_path: $log_path
  }')"

echo "$result_json" | tee -a "$RUNS_JSONL" > "$LATEST_JSON"
echo "[strict-canary] run=$run_number status=$status rc=$gate_rc finished_at_utc=$finished_at_utc duration_sec=$duration_sec log=$run_log" | tee -a "$run_log"

if [[ "$gate_rc" -eq 0 ]]; then
  echo "PLAN_BIG_STRICT_CANARY_PASS"
  echo "JSON: $LATEST_JSON"
  exit 0
fi

echo "error: PLAN_BIG_STRICT_CANARY_FAIL rc=$gate_rc" >&2
echo "JSON: $LATEST_JSON" >&2
exit "$gate_rc"
