#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
PID_FILE="${PLAN_BIG_OVERNIGHT_PID_FILE:-$ART_DIR/runner.pid}"
LOG_FILE="${PLAN_BIG_OVERNIGHT_BG_LOG:-$ART_DIR/runner.stdout.log}"

mkdir -p "$ART_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "error: overnight runner already active pid=$old_pid" >&2
    exit 1
  fi
fi

nohup bash "$ROOT_DIR/scripts/overnight_plan_big_runner.sh" "$@" >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "OVERNIGHT_PLAN_BIG_STARTED pid=$(cat "$PID_FILE")"
echo "PID: $PID_FILE"
echo "LOG: $LOG_FILE"
