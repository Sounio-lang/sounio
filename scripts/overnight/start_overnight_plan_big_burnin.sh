#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
PID_FILE="${PLAN_BIG_OVERNIGHT_BURNIN_PID_FILE:-$ART_DIR/burnin.pid}"
LOG_FILE="${PLAN_BIG_OVERNIGHT_BURNIN_LOG_FILE:-$ART_DIR/burnin.stdout.log}"

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

mkdir -p "$ART_DIR"

old_pid="$(read_pid "$PID_FILE")"
if is_pid_live "$old_pid"; then
  echo "OVERNIGHT_PLAN_BIG_BURNIN_ALREADY_RUNNING pid=$old_pid"
  echo "PID: $PID_FILE"
  echo "LOG: $LOG_FILE"
  exit 0
fi

if command -v setsid >/dev/null 2>&1; then
  setsid bash "$ROOT_DIR/scripts/overnight_plan_big_burnin.sh" "$@" </dev/null >"$LOG_FILE" 2>&1 &
else
  nohup bash "$ROOT_DIR/scripts/overnight_plan_big_burnin.sh" "$@" >"$LOG_FILE" 2>&1 &
fi

pid=$!
printf '%s\n' "$pid" >"$PID_FILE"

echo "OVERNIGHT_PLAN_BIG_BURNIN_STARTED pid=$pid"
echo "PID: $PID_FILE"
echo "LOG: $LOG_FILE"
