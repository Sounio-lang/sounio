#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="${PLAN_BIG_TMUX_SESSION_NAME:-sounio_big_ops_24h}"
ART_DIR="${PLAN_BIG_OVERNIGHT_ART_DIR:-$ROOT_DIR/artifacts/omega/overnight_plan_big}"
BURNIN_LOG="${PLAN_BIG_OVERNIGHT_BURNIN_LOG_FILE:-$ART_DIR/burnin.stdout.log}"

RUNNER_INTERVAL_SEC="${PLAN_BIG_OVERNIGHT_INTERVAL_SEC:-900}"
RUNNER_MAX_RUNS="${PLAN_BIG_OVERNIGHT_MAX_RUNS:-0}"
RUNNER_STOP_ON_PASS="${PLAN_BIG_OVERNIGHT_STOP_ON_PASS:-0}"
RUNNER_GATE_SCRIPT="${PLAN_BIG_OVERNIGHT_GATE_SCRIPT:-$ROOT_DIR/scripts/ci/plan_big_gate.sh}"
RUNNER_GATE_ARGS="${PLAN_BIG_OVERNIGHT_GATE_ARGS:-}"

BURNIN_DURATION_SEC="${PLAN_BIG_OVERNIGHT_BURNIN_DURATION_SEC:-86400}"
BURNIN_CHECK_INTERVAL_SEC="${PLAN_BIG_OVERNIGHT_BURNIN_CHECK_INTERVAL_SEC:-60}"
BURNIN_TAIL_LINES="${PLAN_BIG_OVERNIGHT_BURNIN_TAIL_LINES:-20}"
BURNIN_AUTO_HEAL="${PLAN_BIG_OVERNIGHT_BURNIN_AUTO_HEAL:-1}"
BURNIN_FAIL_FAST="${PLAN_BIG_OVERNIGHT_BURNIN_FAIL_FAST:-0}"

STATUS_LOOP_SEC="${PLAN_BIG_TMUX_STATUS_LOOP_SEC:-60}"
HEALTH_LOOP_SEC="${PLAN_BIG_TMUX_HEALTH_LOOP_SEC:-300}"
GATE_LOOP_SEC="${PLAN_BIG_TMUX_GATE_LOOP_SEC:-900}"
REPORT_LOOP_SEC="${PLAN_BIG_TMUX_REPORT_LOOP_SEC:-3600}"
CANARY_LOOP_SEC="${PLAN_BIG_TMUX_CANARY_LOOP_SEC:-600}"

usage() {
  cat <<USAGE
Usage: scripts/overnight/tmux_big_ops_default.sh <command> [options]

Commands:
  up [--reset]            Apply tmux defaults, ensure runner+burn-in, start/refresh session
  attach                  Attach to default session
  status                  Show tmux + overnight quick status
  down [--stop-processes] Kill session (optional: also stop runner and burn-in)
  apply-defaults          Apply tmux global defaults for ops workflow
USAGE
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "error: tmux not found" >&2
    exit 1
  fi
}

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

session_exists() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

stop_burnin() {
  local burnin_pid_file burnin_pid
  burnin_pid_file="${PLAN_BIG_OVERNIGHT_BURNIN_PID_FILE:-$ART_DIR/burnin.pid}"
  burnin_pid="$(read_pid "$burnin_pid_file")"
  if is_pid_live "$burnin_pid"; then
    kill "$burnin_pid" 2>/dev/null || true
    kill "-$burnin_pid" 2>/dev/null || true
    sleep 1
    if is_pid_live "$burnin_pid"; then
      kill -9 "$burnin_pid" 2>/dev/null || true
      kill -9 "-$burnin_pid" 2>/dev/null || true
    fi
    echo "OVERNIGHT_PLAN_BIG_BURNIN_STOPPED pid=$burnin_pid"
  else
    echo "OVERNIGHT_PLAN_BIG_BURNIN_ALREADY_STOPPED"
  fi
  rm -f "$burnin_pid_file" 2>/dev/null || true
}

apply_defaults() {
  require_tmux
  tmux set-option -g mouse on
  tmux set-option -g history-limit 200000
  tmux set-option -g status-interval 5
  tmux set-option -g renumber-windows on
  tmux set-option -g base-index 1
  tmux set-option -g pane-base-index 1
  tmux set-option -g allow-rename off
  tmux set-option -g set-titles on
  tmux set-window-option -g remain-on-exit on
  tmux set-window-option -g monitor-activity on
  tmux set-window-option -g automatic-rename off
  echo "TMUX_BIG_OPS_DEFAULTS_APPLIED session=$SESSION_NAME"
}

ensure_runner() {
  local state
  state="$(bash scripts/overnight_plan_big_status.sh --json | jq -r '.state // ""' 2>/dev/null || true)"
  if [[ "$state" == "running" ]]; then
    echo "OVERNIGHT_PLAN_BIG_ALREADY_RUNNING"
    return 0
  fi
  PLAN_BIG_OVERNIGHT_GATE_SCRIPT="$RUNNER_GATE_SCRIPT" \
  PLAN_BIG_OVERNIGHT_GATE_ARGS="$RUNNER_GATE_ARGS" \
  bash scripts/start_overnight_plan_big.sh \
    --interval-sec "$RUNNER_INTERVAL_SEC" \
    --max-runs "$RUNNER_MAX_RUNS" \
    --stop-on-pass "$RUNNER_STOP_ON_PASS"
}

ensure_burnin() {
  bash scripts/start_overnight_plan_big_burnin.sh \
    --duration-sec "$BURNIN_DURATION_SEC" \
    --check-interval-sec "$BURNIN_CHECK_INTERVAL_SEC" \
    --tail-lines "$BURNIN_TAIL_LINES" \
    --auto-heal "$BURNIN_AUTO_HEAL" \
    --fail-fast "$BURNIN_FAIL_FAST"
}

mk_session() {
  mkdir -p "$ART_DIR"
  tmux new-session -d -s "$SESSION_NAME" -n status \
    "cd '$ROOT_DIR' && while true; do printf '[%s] ' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; bash scripts/overnight_plan_big_status.sh --json | jq -r '.state + \" pid=\" + (.runner.pid // \"\") + \" hb_fresh=\" + (.heartbeat.fresh|tostring) + \" latest=\" + .latest.status + \"/\" + (.latest.rc|tostring)'; sleep '$STATUS_LOOP_SEC'; done"
  tmux new-window -t "$SESSION_NAME" -n health \
    "cd '$ROOT_DIR' && while true; do printf '[%s] ' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; if bash scripts/check_overnight_plan_big_health.sh --tail-lines 20 >/dev/null 2>&1; then echo 'HEALTH_PASS'; else checks=\$(jq -r '.failed_checks // [] | join(\",\")' artifacts/omega/overnight_plan_big_health.v1.json 2>/dev/null || echo 'unknown'); state=\$(jq -r '.state // \"unknown\"' artifacts/omega/overnight_plan_big_health.v1.json 2>/dev/null || echo 'unknown'); echo \"HEALTH_FAIL state=\$state checks=\$checks\"; fi; sleep '$HEALTH_LOOP_SEC'; done"
  tmux new-window -t "$SESSION_NAME" -n gate \
    "cd '$ROOT_DIR' && while true; do printf '[%s] ' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; if [ -f artifacts/omega/plan_big_gate_status.v1.json ]; then jq -r '.status + \" burnin_required=\" + (.checks.overnight_burnin_required|tostring) + \" burnin_pass=\" + (.checks.overnight_burnin_pass|tostring)' artifacts/omega/plan_big_gate_status.v1.json; else echo 'GATE_STATUS_MISSING'; fi; sleep '$GATE_LOOP_SEC'; done"
  tmux new-window -t "$SESSION_NAME" -n report \
    "cd '$ROOT_DIR' && while true; do printf '[%s] ' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; if bash scripts/overnight_plan_big_hourly_report.sh >/dev/null 2>&1; then echo 'REPORT_UPDATED'; else echo 'REPORT_FAIL'; fi; sleep '$REPORT_LOOP_SEC'; done"
  tmux new-window -t "$SESSION_NAME" -n canary \
    "cd '$ROOT_DIR' && while true; do printf '[%s] ' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; if [ -f artifacts/omega/plan_big_strict_canary/latest.v1.json ]; then jq -r '.status + \"/\" + (.rc|tostring) + \" run=\" + (.run_number|tostring) + \" at=\" + .finished_at_utc' artifacts/omega/plan_big_strict_canary/latest.v1.json; else echo 'STRICT_CANARY_MISSING'; fi; sleep '$CANARY_LOOP_SEC'; done"
  tmux new-window -t "$SESSION_NAME" -n burnin-log \
    "cd '$ROOT_DIR' && touch '$BURNIN_LOG' && tail -n 200 -F '$BURNIN_LOG'"
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage >&2
  exit 2
fi
shift || true

case "$cmd" in
  apply-defaults)
    apply_defaults
    ;;
  up)
    require_tmux
    reset=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --reset)
          reset=1
          shift
          ;;
        *)
          echo "error: unknown option for up: $1" >&2
          exit 2
          ;;
      esac
    done
    apply_defaults
    if session_exists && [[ "$reset" -eq 1 ]]; then
      tmux kill-session -t "$SESSION_NAME"
      echo "TMUX_BIG_OPS_RESET_SESSION name=$SESSION_NAME"
    fi
    if [[ "$reset" -eq 1 ]]; then
      bash scripts/stop_overnight_plan_big.sh --force || true
      stop_burnin
      echo "TMUX_BIG_OPS_RESET_PROCESSES runner=stopped burnin=stopped"
    fi
    ensure_runner
    ensure_burnin
    if ! session_exists; then
      mk_session
    fi
    tmux select-window -t "$SESSION_NAME:status" 2>/dev/null || true
    echo "TMUX_BIG_OPS_UP session=$SESSION_NAME"
    echo "ATTACH: tmux attach -t $SESSION_NAME"
    ;;
  attach)
    require_tmux
    if ! session_exists; then
      echo "error: session not found: $SESSION_NAME (run: scripts/overnight/tmux_big_ops_default.sh up)" >&2
      exit 1
    fi
    exec tmux attach -t "$SESSION_NAME"
    ;;
  status)
    require_tmux
    if session_exists; then
      echo "TMUX_SESSION=running name=$SESSION_NAME"
      tmux list-windows -t "$SESSION_NAME"
    else
      echo "TMUX_SESSION=stopped name=$SESSION_NAME"
    fi
    echo "---"
    bash scripts/overnight_plan_big_status.sh --json | jq -r '.state + " pid=" + (.runner.pid // "") + " hb_fresh=" + (.heartbeat.fresh|tostring) + " latest=" + .latest.status + "/" + (.latest.rc|tostring)'
    if bash scripts/check_overnight_plan_big_health.sh --tail-lines 10 >/dev/null 2>&1; then
      echo "HEALTH=pass"
    else
      echo "HEALTH=fail"
    fi
    ;;
  down)
    require_tmux
    stop_processes=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --stop-processes)
          stop_processes=1
          shift
          ;;
        *)
          echo "error: unknown option for down: $1" >&2
          exit 2
          ;;
      esac
    done
    if session_exists; then
      tmux kill-session -t "$SESSION_NAME"
      echo "TMUX_BIG_OPS_SESSION_STOPPED name=$SESSION_NAME"
    else
      echo "TMUX_BIG_OPS_SESSION_ALREADY_STOPPED name=$SESSION_NAME"
    fi
    if [[ "$stop_processes" -eq 1 ]]; then
      bash scripts/stop_overnight_plan_big.sh --force || true
      stop_burnin
    fi
    ;;
  *)
    echo "error: unknown command '$cmd'" >&2
    usage >&2
    exit 2
    ;;
esac
