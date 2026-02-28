#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INTERVAL_SEC="${PLAN_BIG_OVERNIGHT_INTERVAL_SEC:-900}"
MAX_RUNS="${PLAN_BIG_OVERNIGHT_MAX_RUNS:-0}"
STOP_ON_PASS="${PLAN_BIG_OVERNIGHT_STOP_ON_PASS:-0}"

ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
JSONL_PATH="${PLAN_BIG_OVERNIGHT_JSONL:-$ART_DIR/runs.v1.jsonl}"
LATEST_JSON="${PLAN_BIG_OVERNIGHT_LATEST_JSON:-$ART_DIR/latest.v1.json}"
GATE_STATUS_JSON="${PLAN_BIG_GATE_STATUS_JSON:-$ROOT_DIR/artifacts/omega/plan_big_gate_status.v1.json}"
GATE_SCRIPT="${PLAN_BIG_OVERNIGHT_GATE_SCRIPT:-$ROOT_DIR/scripts/plan_big_gate.sh}"
GATE_ARGS="${PLAN_BIG_OVERNIGHT_GATE_ARGS:-}"
LOCK_DIR="${PLAN_BIG_OVERNIGHT_LOCK_DIR:-$ART_DIR/.lock}"
LOCK_PID_FILE="$LOCK_DIR/pid"
HEARTBEAT_JSON="${PLAN_BIG_OVERNIGHT_HEARTBEAT_JSON:-$ART_DIR/heartbeat.v1.json}"
LOCK_HELD=0

usage() {
  cat <<USAGE
Usage: scripts/overnight_plan_big_runner.sh [--interval-sec N] [--max-runs N] [--stop-on-pass 0|1]

Options:
  --interval-sec N   Seconds between runs (default: 900)
  --max-runs N       Number of runs before exit (0 = infinite, default: 0)
  --stop-on-pass B   Exit early when a run passes (default: 0)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval-sec)
      INTERVAL_SEC="$2"
      shift 2
      ;;
    --max-runs)
      MAX_RUNS="$2"
      shift 2
      ;;
    --stop-on-pass)
      STOP_ON_PASS="$2"
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

if ! [[ "$INTERVAL_SEC" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_SEC" -eq 0 ]]; then
  echo "error: --interval-sec must be positive integer" >&2
  exit 2
fi
if ! [[ "$MAX_RUNS" =~ ^[0-9]+$ ]]; then
  echo "error: --max-runs must be integer >= 0" >&2
  exit 2
fi
if [[ "$STOP_ON_PASS" != "0" && "$STOP_ON_PASS" != "1" ]]; then
  echo "error: --stop-on-pass must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$ART_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required for overnight plan-big runner" >&2
  exit 2
fi
if [[ "$GATE_SCRIPT" == */* ]]; then
  if [[ "$GATE_SCRIPT" != /* ]]; then
    GATE_SCRIPT="$ROOT_DIR/$GATE_SCRIPT"
  fi
  if [[ ! -e "$GATE_SCRIPT" ]]; then
    echo "error: overnight gate script does not exist: $GATE_SCRIPT" >&2
    exit 2
  fi
else
  if command -v "$GATE_SCRIPT" >/dev/null 2>&1; then
    GATE_SCRIPT="$(command -v "$GATE_SCRIPT")"
  else
    echo "error: overnight gate script command not found in PATH: $GATE_SCRIPT" >&2
    exit 2
  fi
fi

declare -a GATE_ARGS_ARRAY=()
if [[ -n "$GATE_ARGS" ]]; then
  # shellcheck disable=SC2206
  GATE_ARGS_ARRAY=($GATE_ARGS)
fi

is_pid_live() {
  local pid="$1"
  [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

recover_stale_lock() {
  if [[ ! -d "$LOCK_DIR" ]]; then
    return 0
  fi
  if [[ -f "$LOCK_PID_FILE" ]]; then
    local lock_pid
    lock_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null | tr -d '[:space:]' || true)"
    if is_pid_live "$lock_pid"; then
      echo "error: overnight runner lock already held by live pid=$lock_pid ($LOCK_DIR)" >&2
      exit 1
    fi
  fi
  rm -rf "$LOCK_DIR"
}

recover_stale_lock
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "error: overnight runner lock already held: $LOCK_DIR" >&2
  exit 1
fi
LOCK_HELD=1
printf '%s\n' "$$" > "$LOCK_PID_FILE"

cleanup_lock() {
  [[ "$LOCK_HELD" -eq 1 ]] || return 0
  rm -f "$LOCK_PID_FILE" 2>/dev/null || true
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

RUN_COUNT=0

run_once() {
  local started_at_utc finished_at_utc run_id run_log gate_rc gate_status duration_sec
  local start_epoch finish_epoch pass_marker run_result_json gate_status_rel log_path_rel
  started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_epoch="$(date -u +%s)"
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  run_log="$ART_DIR/run_${run_id}.log"
  log_path_rel="${run_log#$ROOT_DIR/}"
  gate_status_rel="${GATE_STATUS_JSON#$ROOT_DIR/}"

  echo "[overnight-plan-big] run=$((RUN_COUNT + 1)) started_at_utc=$started_at_utc start" | tee -a "$run_log"
  jq -cn \
    --arg generated_at_utc "$started_at_utc" \
    --arg phase "running" \
    --argjson run_number "$((RUN_COUNT + 1))" \
    --arg pid "$$" \
    '{schema:"sounio.plan.big.overnight-heartbeat.v1",generated_at_utc:$generated_at_utc,phase:$phase,run_number:$run_number,pid:($pid|tonumber)}' \
    > "$HEARTBEAT_JSON"

  set +e
  if [[ -x "$GATE_SCRIPT" ]]; then
    "$GATE_SCRIPT" "${GATE_ARGS_ARRAY[@]}" >>"$run_log" 2>&1
    gate_rc=$?
  else
    bash "$GATE_SCRIPT" "${GATE_ARGS_ARRAY[@]}" >>"$run_log" 2>&1
    gate_rc=$?
  fi
  set -e

  gate_status="fail"
  pass_marker=false
  if [[ "$gate_rc" -eq 0 ]]; then
    gate_status="pass"
    pass_marker=true
  fi

  finished_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  finish_epoch="$(date -u +%s)"
  duration_sec=$((finish_epoch - start_epoch))
  if [[ "$duration_sec" -lt 0 ]]; then
    duration_sec=0
  fi

  run_result_json="$(jq -cn \
    --arg generated_at_utc "$finished_at_utc" \
    --arg started_at_utc "$started_at_utc" \
    --arg finished_at_utc "$finished_at_utc" \
    --arg status "$gate_status" \
    --argjson rc "$gate_rc" \
    --argjson duration_sec "$duration_sec" \
    --arg log_path "$log_path_rel" \
    --arg gate_status_path "$gate_status_rel" \
    --arg gate_script "${GATE_SCRIPT#$ROOT_DIR/}" \
    --arg gate_args "$GATE_ARGS" \
    --argjson run_number "$((RUN_COUNT + 1))" \
    --arg pid "$$" \
    --argjson pass_marker "$pass_marker" \
    '{
      schema: "sounio.plan.big.overnight-run.v1",
      generated_at_utc: $generated_at_utc,
      started_at_utc: $started_at_utc,
      finished_at_utc: $finished_at_utc,
      duration_sec: $duration_sec,
      run_number: $run_number,
      runner_pid: ($pid|tonumber),
      status: $status,
      rc: $rc,
      pass_marker: $pass_marker,
      gate_script: $gate_script,
      gate_args: $gate_args,
      gate_status_path: $gate_status_path,
      log_path: $log_path
    }')"
  echo "$run_result_json" | tee -a "$JSONL_PATH" > "$LATEST_JSON"

  jq -cn \
    --arg generated_at_utc "$finished_at_utc" \
    --arg phase "idle" \
    --arg status "$gate_status" \
    --argjson rc "$gate_rc" \
    --argjson run_number "$((RUN_COUNT + 1))" \
    --arg pid "$$" \
    '{
      schema:"sounio.plan.big.overnight-heartbeat.v1",
      generated_at_utc:$generated_at_utc,
      phase:$phase,
      last_status:$status,
      last_rc:$rc,
      run_number:$run_number,
      pid:($pid|tonumber)
    }' > "$HEARTBEAT_JSON"

  echo "[overnight-plan-big] run=$((RUN_COUNT + 1)) status=$gate_status rc=$gate_rc started_at_utc=$started_at_utc finished_at_utc=$finished_at_utc duration_sec=$duration_sec log=$run_log gate_status_path=$gate_status_rel" | tee -a "$run_log"

  if [[ "$STOP_ON_PASS" -eq 1 && "$gate_status" == "pass" ]]; then
    echo "[overnight-plan-big] stop-on-pass reached; exiting" | tee -a "$run_log"
    return 10
  fi
  return 0
}

while true; do
  rc=0
  if ! run_once; then
    rc=$?
  fi
  RUN_COUNT=$((RUN_COUNT + 1))

  if [[ "$rc" -eq 10 ]]; then
    break
  fi

  if [[ "$MAX_RUNS" -gt 0 && "$RUN_COUNT" -ge "$MAX_RUNS" ]]; then
    break
  fi

  sleep "$INTERVAL_SEC"
done

echo "OVERNIGHT_PLAN_BIG_RUNNER_DONE runs=$RUN_COUNT latest=$LATEST_JSON"
