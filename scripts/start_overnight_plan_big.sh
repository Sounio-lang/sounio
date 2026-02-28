#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
PID_FILE="${PLAN_BIG_OVERNIGHT_PID_FILE:-$ART_DIR/runner.pid}"
LOG_FILE="${PLAN_BIG_OVERNIGHT_BG_LOG:-$ART_DIR/runner.stdout.log}"
LOCK_DIR="${PLAN_BIG_OVERNIGHT_LOCK_DIR:-$ART_DIR/.lock}"
LOCK_PID_FILE="$LOCK_DIR/pid"

mkdir -p "$ART_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: required dependency missing: jq" >&2
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

nohup bash "$ROOT_DIR/scripts/overnight_plan_big_runner.sh" "$@" >"$LOG_FILE" 2>&1 &
runner_pid=$!
printf '%s\n' "$runner_pid" > "$PID_FILE"
if [[ ! -s "$PID_FILE" ]]; then
  echo "error: failed to write runner pid file: $PID_FILE" >&2
  exit 1
fi

echo "OVERNIGHT_PLAN_BIG_STARTED pid=$runner_pid"
echo "PID: $PID_FILE"
echo "LOG: $LOG_FILE"
