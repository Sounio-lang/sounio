#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
PID_FILE="${PLAN_BIG_OVERNIGHT_PID_FILE:-$ART_DIR/runner.pid}"
LOCK_DIR="${PLAN_BIG_OVERNIGHT_LOCK_DIR:-$ART_DIR/.lock}"
LOCK_PID_FILE="$LOCK_DIR/pid"
HEARTBEAT_JSON="${PLAN_BIG_OVERNIGHT_HEARTBEAT_JSON:-$ART_DIR/heartbeat.v1.json}"
LATEST_JSON="${PLAN_BIG_OVERNIGHT_LATEST_JSON:-$ART_DIR/latest.v1.json}"
RUNNER_LOG="${PLAN_BIG_OVERNIGHT_BG_LOG:-$ART_DIR/runner.stdout.log}"
HEARTBEAT_STALE_SEC="${PLAN_BIG_OVERNIGHT_HEARTBEAT_STALE_SEC:-1800}"

TAIL_LINES=40
JSON_MODE=0
JSON_ONLY=0

usage() {
  cat <<USAGE
Usage: scripts/overnight_plan_big_status.sh [--json] [--json-only] [--tail-lines N]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_MODE=1
      shift
      ;;
    --json-only)
      JSON_MODE=1
      JSON_ONLY=1
      shift
      ;;
    --tail-lines)
      TAIL_LINES="$2"
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

if ! [[ "$TAIL_LINES" =~ ^[0-9]+$ ]] || [[ "$TAIL_LINES" -eq 0 ]]; then
  echo "error: --tail-lines must be positive integer" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 2
fi

is_pid_live() {
  local pid="$1"
  [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
  local file="$1"
  if [[ -f "$file" ]]; then
    cat "$file" 2>/dev/null | tr -d '[:space:]' || true
  fi
}

runner_pid="$(read_pid "$PID_FILE")"
lock_pid="$(read_pid "$LOCK_PID_FILE")"
runner_live=false
lock_live=false
lock_dir_exists=false

if is_pid_live "$runner_pid"; then
  runner_live=true
fi
if is_pid_live "$lock_pid"; then
  lock_live=true
fi
if [[ -d "$LOCK_DIR" ]]; then
  lock_dir_exists=true
fi

state="unknown"
state_reason=""
if [[ "$runner_live" == "true" && "$lock_live" == "true" && "$runner_pid" == "$lock_pid" ]]; then
  state="running"
  state_reason="runner and lock pid are live and match"
elif [[ "$lock_dir_exists" == "true" && "$lock_live" == "false" ]]; then
  state="stale_lock"
  state_reason="lock directory exists but lock pid is not live"
elif [[ "$runner_live" == "false" && "$lock_live" == "true" ]]; then
  state="stale_runner_pid"
  state_reason="runner pid file is stale but lock pid is live"
elif [[ "$runner_live" == "true" && "$lock_live" == "false" ]]; then
  state="stale_pair"
  state_reason="runner is live but lock pid is stale/missing"
elif [[ "$runner_live" == "false" && "$lock_live" == "false" && "$lock_dir_exists" == "false" ]]; then
  state="stopped"
  state_reason="no live runner and no lock directory"
else
  state="unknown"
  state_reason="state combination not recognized"
fi

heartbeat_present=false
heartbeat_valid=false
heartbeat_schema=""
heartbeat_phase=""
heartbeat_ts=""
heartbeat_run_number=0
heartbeat_pid=""
heartbeat_last_status=""
heartbeat_age_seconds=-1
heartbeat_fresh=false
heartbeat_stale=true
heartbeat_pid_live=false

if [[ -f "$HEARTBEAT_JSON" ]]; then
  heartbeat_present=true
  heartbeat_schema="$(jq -r '.schema // ""' "$HEARTBEAT_JSON" 2>/dev/null || true)"
  if [[ "$heartbeat_schema" == "sounio.plan.big.overnight-heartbeat.v1" ]]; then
    heartbeat_valid=true
    heartbeat_phase="$(jq -r '.phase // ""' "$HEARTBEAT_JSON")"
    heartbeat_ts="$(jq -r '.generated_at_utc // ""' "$HEARTBEAT_JSON")"
    heartbeat_run_number="$(jq -r '.run_number // 0' "$HEARTBEAT_JSON")"
    heartbeat_pid="$(jq -r '.pid // ""' "$HEARTBEAT_JSON")"
    heartbeat_last_status="$(jq -r '.last_status // ""' "$HEARTBEAT_JSON")"
    if is_pid_live "$heartbeat_pid"; then
      heartbeat_pid_live=true
    fi

    if [[ "$heartbeat_pid_live" == "true" && -n "$heartbeat_ts" ]]; then
      hb_epoch="$(date -u -d "$heartbeat_ts" +%s 2>/dev/null || echo "")"
      now_epoch="$(date -u +%s)"
      if [[ -n "$hb_epoch" ]]; then
        heartbeat_age_seconds=$((now_epoch - hb_epoch))
        if [[ "$heartbeat_age_seconds" -le "$HEARTBEAT_STALE_SEC" ]]; then
          heartbeat_fresh=true
          heartbeat_stale=false
        fi
      fi
    fi
  fi
fi

latest_present=false
latest_valid=false
latest_schema=""
latest_run_number=0
latest_ts=""
latest_status=""
latest_rc=0
latest_started_at_utc=""
latest_finished_at_utc=""
latest_duration_sec=0
latest_pass_marker=false
latest_gate_status_path=""
latest_runner_pid=""
latest_log_path=""
latest_log_abs=""
latest_log_exists=false

if [[ -f "$LATEST_JSON" ]]; then
  latest_present=true
  latest_schema="$(jq -r '.schema // ""' "$LATEST_JSON" 2>/dev/null || true)"
  if [[ "$latest_schema" == "sounio.plan.big.overnight-run.v1" ]]; then
    latest_valid=true
    latest_run_number="$(jq -r '.run_number // 0' "$LATEST_JSON")"
    latest_ts="$(jq -r '.generated_at_utc // ""' "$LATEST_JSON")"
    latest_started_at_utc="$(jq -r '.started_at_utc // ""' "$LATEST_JSON")"
    latest_finished_at_utc="$(jq -r '.finished_at_utc // ""' "$LATEST_JSON")"
    latest_duration_sec="$(jq -r '.duration_sec // 0' "$LATEST_JSON")"
    latest_status="$(jq -r '.status // ""' "$LATEST_JSON")"
    latest_rc="$(jq -r '.rc // 0' "$LATEST_JSON")"
    latest_pass_marker="$(jq -r '.pass_marker // false' "$LATEST_JSON")"
    latest_gate_status_path="$(jq -r '.gate_status_path // ""' "$LATEST_JSON")"
    latest_runner_pid="$(jq -r '.runner_pid // ""' "$LATEST_JSON")"
    latest_log_path="$(jq -r '.log_path // ""' "$LATEST_JSON")"
    if [[ -n "$latest_log_path" ]]; then
      latest_log_abs="$ROOT_DIR/$latest_log_path"
      if [[ -f "$latest_log_abs" ]]; then
        latest_log_exists=true
      fi
    fi
  fi
fi

log_path="$RUNNER_LOG"
if [[ "$latest_log_exists" == "true" ]]; then
  log_path="$latest_log_abs"
fi

log_tail_json='[]'
if [[ -f "$log_path" ]]; then
  log_tail_json="$(tail -n "$TAIL_LINES" "$log_path" | jq -R -s 'split("\n") | map(select(length > 0))')"
fi

as_of_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
status_json="$(jq -cn \
  --arg as_of_utc "$as_of_utc" \
  --arg state "$state" \
  --arg state_reason "$state_reason" \
  --arg runner_pid_file "${PID_FILE#$ROOT_DIR/}" \
  --arg runner_pid "$runner_pid" \
  --argjson runner_live "$runner_live" \
  --arg lock_dir "${LOCK_DIR#$ROOT_DIR/}" \
  --arg lock_pid "$lock_pid" \
  --argjson lock_live "$lock_live" \
  --argjson lock_dir_exists "$lock_dir_exists" \
  --arg heartbeat_path "${HEARTBEAT_JSON#$ROOT_DIR/}" \
  --argjson heartbeat_present "$heartbeat_present" \
  --argjson heartbeat_valid "$heartbeat_valid" \
  --arg heartbeat_schema "$heartbeat_schema" \
  --arg heartbeat_phase "$heartbeat_phase" \
  --arg heartbeat_ts "$heartbeat_ts" \
  --argjson heartbeat_run_number "$heartbeat_run_number" \
  --arg heartbeat_pid "$heartbeat_pid" \
  --arg heartbeat_last_status "$heartbeat_last_status" \
  --argjson heartbeat_age_seconds "$heartbeat_age_seconds" \
  --argjson heartbeat_fresh "$heartbeat_fresh" \
  --argjson heartbeat_stale "$heartbeat_stale" \
  --argjson heartbeat_pid_live "$heartbeat_pid_live" \
  --arg latest_path "${LATEST_JSON#$ROOT_DIR/}" \
  --argjson latest_present "$latest_present" \
  --argjson latest_valid "$latest_valid" \
  --arg latest_schema "$latest_schema" \
  --argjson latest_run_number "$latest_run_number" \
  --arg latest_ts "$latest_ts" \
  --arg latest_started_at_utc "$latest_started_at_utc" \
  --arg latest_finished_at_utc "$latest_finished_at_utc" \
  --argjson latest_duration_sec "$latest_duration_sec" \
  --arg latest_status "$latest_status" \
  --argjson latest_rc "$latest_rc" \
  --argjson latest_pass_marker "$latest_pass_marker" \
  --arg latest_gate_status_path "$latest_gate_status_path" \
  --arg latest_runner_pid "$latest_runner_pid" \
  --arg latest_log_path "$latest_log_path" \
  --argjson latest_log_exists "$latest_log_exists" \
  --arg log_path "${log_path#$ROOT_DIR/}" \
  --argjson tail_lines "$TAIL_LINES" \
  --argjson log_tail "$log_tail_json" \
  '{
    schema:"sounio.plan.big.overnight.status.v1",
    as_of_utc:$as_of_utc,
    state:$state,
    state_reason:$state_reason,
    runner:{pid_file:$runner_pid_file,pid:$runner_pid,live:$runner_live,stale:($runner_pid != "" and ($runner_live | not))},
    lock:{dir:$lock_dir,pid:$lock_pid,live:$lock_live,dir_exists:$lock_dir_exists,stale:($lock_dir_exists and ($lock_live | not))},
    heartbeat:{path:$heartbeat_path,present:$heartbeat_present,valid:$heartbeat_valid,schema:$heartbeat_schema,phase:$heartbeat_phase,generated_at_utc:$heartbeat_ts,run_number:$heartbeat_run_number,pid:$heartbeat_pid,pid_live:$heartbeat_pid_live,last_status:$heartbeat_last_status,age_seconds:$heartbeat_age_seconds,fresh:$heartbeat_fresh,stale:$heartbeat_stale},
    latest:{path:$latest_path,present:$latest_present,valid:$latest_valid,schema:$latest_schema,run_number:$latest_run_number,generated_at_utc:$latest_ts,started_at_utc:$latest_started_at_utc,finished_at_utc:$latest_finished_at_utc,duration_sec:$latest_duration_sec,runner_pid:$latest_runner_pid,status:$latest_status,rc:$latest_rc,pass_marker:$latest_pass_marker,gate_status_path:$latest_gate_status_path,log_path:$latest_log_path,log_exists:$latest_log_exists},
    log:{path:$log_path,tail_lines_requested:$tail_lines,tail:$log_tail},
    errors:[]
  }')"

if [[ "$JSON_MODE" -eq 1 ]]; then
  echo "$status_json"
  exit 0
fi

if [[ "$JSON_ONLY" -eq 0 ]]; then
  echo "overnight-plan-big status: $state ($state_reason)"
  echo "runner pid: ${runner_pid:-none} live=$runner_live"
  echo "lock pid: ${lock_pid:-none} live=$lock_live dir_exists=$lock_dir_exists"
  echo "heartbeat: present=$heartbeat_present valid=$heartbeat_valid pid_live=$heartbeat_pid_live phase=${heartbeat_phase:-none} age_sec=$heartbeat_age_seconds fresh=$heartbeat_fresh"
  echo "latest: present=$latest_present valid=$latest_valid run=${latest_run_number:-0} status=${latest_status:-unknown} rc=${latest_rc:-0} pass_marker=${latest_pass_marker:-false} duration_sec=${latest_duration_sec:-0}"
  echo "log tail ($TAIL_LINES lines): ${log_path#$ROOT_DIR/}"
  tail -n "$TAIL_LINES" "$log_path" 2>/dev/null || true
fi
