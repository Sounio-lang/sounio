#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
PID_FILE="${PLAN_BIG_OVERNIGHT_PID_FILE:-$ART_DIR/runner.pid}"
LOCK_DIR="${PLAN_BIG_OVERNIGHT_LOCK_DIR:-$ART_DIR/.lock}"
LOCK_PID_FILE="$LOCK_DIR/pid"
WAIT_SEC="${PLAN_BIG_OVERNIGHT_STOP_WAIT_SEC:-15}"

usage() {
  cat <<USAGE
Usage: scripts/ops/stop_overnight_plan_big.sh [--force] [--wait-sec N]

Options:
  --force       Send SIGKILL if process does not stop after wait window
  --wait-sec N  Graceful wait window in seconds (default: 15)
USAGE
}

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --wait-sec)
      WAIT_SEC="$2"
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

if ! [[ "$WAIT_SEC" =~ ^[0-9]+$ ]] || [[ "$WAIT_SEC" -eq 0 ]]; then
  echo "error: --wait-sec must be positive integer" >&2
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

cleanup_stale_files() {
  rm -f "$PID_FILE" 2>/dev/null || true
  rm -f "$LOCK_PID_FILE" 2>/dev/null || true
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

runner_pid="$(read_pid "$PID_FILE")"
lock_pid="$(read_pid "$LOCK_PID_FILE")"

if [[ -z "$runner_pid" && -z "$lock_pid" ]]; then
  cleanup_stale_files
  echo "OVERNIGHT_PLAN_BIG_STOP_SUCCESS mode=already_stopped"
  echo "OVERNIGHT_PLAN_BIG_ALREADY_STOPPED"
  exit 0
fi

target_pid=""
if is_pid_live "$runner_pid"; then
  target_pid="$runner_pid"
elif is_pid_live "$lock_pid"; then
  target_pid="$lock_pid"
fi

if [[ -z "$target_pid" ]]; then
  cleanup_stale_files
  echo "OVERNIGHT_PLAN_BIG_STOP_SUCCESS mode=stale_cleanup"
  echo "OVERNIGHT_PLAN_BIG_STOPPED_STALE_STATE_CLEANED"
  exit 0
fi

kill "$target_pid" 2>/dev/null || true
# If runner was launched with setsid, pid==pgid; signal full process group too.
kill "-$target_pid" 2>/dev/null || true
for _ in $(seq 1 "$WAIT_SEC"); do
  if ! is_pid_live "$target_pid"; then
    cleanup_stale_files
    echo "OVERNIGHT_PLAN_BIG_STOP_SUCCESS mode=graceful pid=$target_pid"
    echo "OVERNIGHT_PLAN_BIG_STOPPED pid=$target_pid"
    exit 0
  fi
  sleep 1
done

# Escalate to SIGKILL by default after graceful timeout.
kill -9 "$target_pid" 2>/dev/null || true
kill -9 "-$target_pid" 2>/dev/null || true
sleep 1
if ! is_pid_live "$target_pid"; then
  cleanup_stale_files
  if [[ "$FORCE" -eq 1 ]]; then
    echo "OVERNIGHT_PLAN_BIG_STOP_SUCCESS mode=force_escalated pid=$target_pid"
    echo "OVERNIGHT_PLAN_BIG_STOPPED_FORCE pid=$target_pid"
  else
    echo "OVERNIGHT_PLAN_BIG_STOP_SUCCESS mode=escalated pid=$target_pid"
    echo "OVERNIGHT_PLAN_BIG_STOPPED pid=$target_pid mode=escalated"
  fi
  exit 0
fi

echo "error: failed to stop overnight runner pid=$target_pid" >&2
exit 1
