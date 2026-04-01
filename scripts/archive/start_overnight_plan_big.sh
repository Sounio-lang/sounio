#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
PID_FILE="${PLAN_BIG_OVERNIGHT_PID_FILE:-$ART_DIR/runner.pid}"
LOG_FILE="${PLAN_BIG_OVERNIGHT_BG_LOG:-$ART_DIR/runner.stdout.log}"
LOCK_DIR="${PLAN_BIG_OVERNIGHT_LOCK_DIR:-$ART_DIR/.lock}"
LOCK_PID_FILE="$LOCK_DIR/pid"
HEARTBEAT_JSON="${PLAN_BIG_OVERNIGHT_HEARTBEAT_JSON:-$ART_DIR/heartbeat.v1.json}"
LATEST_JSON="${PLAN_BIG_OVERNIGHT_LATEST_JSON:-$ART_DIR/latest.v1.json}"
STARTUP_TIMEOUT_SEC="${PLAN_BIG_OVERNIGHT_STARTUP_TIMEOUT_SEC:-15}"
STARTUP_REQUIRE_FIRST_RESULT="${PLAN_BIG_OVERNIGHT_STARTUP_REQUIRE_FIRST_RESULT:-1}"
FIRST_RESULT_TIMEOUT_SEC="${PLAN_BIG_OVERNIGHT_FIRST_RESULT_TIMEOUT_SEC:-900}"

mkdir -p "$ART_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: required dependency missing: jq" >&2
  exit 1
fi

if ! [[ "$STARTUP_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$STARTUP_TIMEOUT_SEC" -eq 0 ]]; then
  echo "error: PLAN_BIG_OVERNIGHT_STARTUP_TIMEOUT_SEC must be positive integer" >&2
  exit 1
fi
if [[ "$STARTUP_REQUIRE_FIRST_RESULT" != "0" && "$STARTUP_REQUIRE_FIRST_RESULT" != "1" ]]; then
  echo "error: PLAN_BIG_OVERNIGHT_STARTUP_REQUIRE_FIRST_RESULT must be 0 or 1" >&2
  exit 1
fi
if ! [[ "$FIRST_RESULT_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$FIRST_RESULT_TIMEOUT_SEC" -eq 0 ]]; then
  echo "error: PLAN_BIG_OVERNIGHT_FIRST_RESULT_TIMEOUT_SEC must be positive integer" >&2
  exit 1
fi

is_pid_live() {
  local pid="$1"
  [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

check_stale_startup_state() {
  if [[ -f "$PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]' || true)"
    if is_pid_live "$old_pid"; then
      echo "error: overnight runner already active pid=$old_pid" >&2
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  if [[ -d "$LOCK_DIR" ]]; then
    local lock_pid
    lock_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null | tr -d '[:space:]' || true)"
    if is_pid_live "$lock_pid"; then
      echo "error: overnight runner lock already held by live pid=$lock_pid ($LOCK_DIR)" >&2
      exit 1
    fi
    rm -rf "$LOCK_DIR"
  fi

  if [[ ! -x "$ROOT_DIR/scripts/overnight_plan_big_runner.sh" ]]; then
    echo "error: overnight runner script not executable: $ROOT_DIR/scripts/overnight_plan_big_runner.sh" >&2
    exit 1
  fi
}

check_stale_startup_state

cleanup_stale_startup_state() {
  local expected_pid="$1"
  if [[ -f "$PID_FILE" ]]; then
    local pid_file_pid
    pid_file_pid="$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -z "$pid_file_pid" || "$pid_file_pid" == "$expected_pid" ]]; then
      rm -f "$PID_FILE"
    fi
  fi
  if [[ -f "$LOCK_PID_FILE" ]]; then
    local lock_pid
    lock_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -z "$lock_pid" || "$lock_pid" == "$expected_pid" ]]; then
      rm -f "$LOCK_PID_FILE"
      rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
  fi
}

startup_ready() {
  local expected_pid="$1"
  local runner_live=false
  local lock_match=false
  local heartbeat_match=false
  local lock_pid=""
  local heartbeat_pid=""

  if is_pid_live "$expected_pid"; then
    runner_live=true
  fi
  if [[ -f "$LOCK_PID_FILE" ]]; then
    lock_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "$lock_pid" == "$expected_pid" && "$runner_live" == "true" ]]; then
      lock_match=true
    fi
  fi
  if [[ -f "$HEARTBEAT_JSON" ]]; then
    heartbeat_pid="$(jq -r '.pid // ""' "$HEARTBEAT_JSON" 2>/dev/null || true)"
    if [[ "$heartbeat_pid" == "$expected_pid" ]]; then
      heartbeat_match=true
    fi
  fi

  if [[ "$runner_live" == "true" && "$lock_match" == "true" && "$heartbeat_match" == "true" ]]; then
    return 0
  fi
  return 1
}

latest_result_ready() {
  local expected_pid="$1"
  local launch_epoch="$2"
  local latest_schema latest_pid latest_ts latest_epoch
  if [[ ! -f "$LATEST_JSON" ]]; then
    return 1
  fi
  latest_schema="$(jq -r '.schema // ""' "$LATEST_JSON" 2>/dev/null || true)"
  if [[ "$latest_schema" != "sounio.plan.big.overnight-run.v1" ]]; then
    return 1
  fi
  latest_pid="$(jq -r '.runner_pid // ""' "$LATEST_JSON" 2>/dev/null || true)"
  if [[ "$latest_pid" != "$expected_pid" ]]; then
    return 1
  fi
  latest_ts="$(jq -r '.finished_at_utc // .generated_at_utc // ""' "$LATEST_JSON" 2>/dev/null || true)"
  if [[ -z "$latest_ts" ]]; then
    return 1
  fi
  latest_epoch="$(date -u -d "$latest_ts" +%s 2>/dev/null || true)"
  if [[ -z "$latest_epoch" ]]; then
    return 1
  fi
  if [[ "$latest_epoch" -lt "$launch_epoch" ]]; then
    return 1
  fi
  return 0
}

launch_epoch="$(date -u +%s)"
if command -v setsid >/dev/null 2>&1; then
  setsid bash "$ROOT_DIR/scripts/overnight_plan_big_runner.sh" "$@" </dev/null >"$LOG_FILE" 2>&1 &
else
  nohup bash "$ROOT_DIR/scripts/overnight_plan_big_runner.sh" "$@" >"$LOG_FILE" 2>&1 &
fi
runner_pid=$!
printf '%s\n' "$runner_pid" > "$PID_FILE"
if [[ ! -s "$PID_FILE" ]]; then
  echo "error: failed to write runner pid file: $PID_FILE" >&2
  exit 1
fi

started_ok=false
for ((i=0; i<STARTUP_TIMEOUT_SEC; i++)); do
  if startup_ready "$runner_pid"; then
    started_ok=true
    break
  fi
  if ! is_pid_live "$runner_pid"; then
    break
  fi
  sleep 1
done

if [[ "$started_ok" != "true" ]]; then
  if is_pid_live "$runner_pid"; then
    kill "$runner_pid" 2>/dev/null || true
    sleep 1
    if is_pid_live "$runner_pid"; then
      kill -9 "$runner_pid" 2>/dev/null || true
    fi
  fi
  cleanup_stale_startup_state "$runner_pid"
  echo "error: overnight runner failed startup health within ${STARTUP_TIMEOUT_SEC}s (pid=$runner_pid)" >&2
  echo "error: check log: $LOG_FILE" >&2
  exit 1
fi

first_result_ok=true
if [[ "$STARTUP_REQUIRE_FIRST_RESULT" -eq 1 ]]; then
  first_result_ok=false
  for ((i=0; i<FIRST_RESULT_TIMEOUT_SEC; i++)); do
    if latest_result_ready "$runner_pid" "$launch_epoch"; then
      first_result_ok=true
      break
    fi
    if ! is_pid_live "$runner_pid"; then
      break
    fi
    sleep 1
  done
fi

if [[ "$first_result_ok" != "true" ]]; then
  if is_pid_live "$runner_pid"; then
    kill "$runner_pid" 2>/dev/null || true
    kill "-$runner_pid" 2>/dev/null || true
    sleep 1
    if is_pid_live "$runner_pid"; then
      kill -9 "$runner_pid" 2>/dev/null || true
      kill -9 "-$runner_pid" 2>/dev/null || true
    fi
  fi
  cleanup_stale_startup_state "$runner_pid"
  echo "error: overnight runner failed first-result startup check within ${FIRST_RESULT_TIMEOUT_SEC}s (pid=$runner_pid)" >&2
  echo "error: check log: $LOG_FILE" >&2
  exit 1
fi

echo "OVERNIGHT_PLAN_BIG_STARTED pid=$runner_pid"
echo "PID: $PID_FILE"
echo "LOG: $LOG_FILE"
